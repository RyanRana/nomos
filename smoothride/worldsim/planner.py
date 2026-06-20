"""Route-following planner: turn each car's position into a (target, speed) the
control bridge can chase. Reuses the SAME route pool the kinematic env uses, so
the cars drive real San Francisco streets — just now with 3D physics.

Transport-agnostic: it works on plain xy arrays, so the local MuJoCo self-test and
the Modal/Newton render share one planner. Routes are loaded in the scene's
recentred coordinate frame (origin-shifted UTM minus the map center stored in the
scene metadata), so they line up with the geometry build_sf_scene.py emitted.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from ..data.map_loader import load_road_network
from ..env.routing import build_route_pool


class RoutePlanner:
    def __init__(self, n_cars: int, scene_dir: str, target_speed: float = 8.0,
                 advance_radius: float = 6.0, lookahead_wp: int = 2, seed: int = 0):
        self.n = n_cars
        self.target_speed = target_speed
        self.advance_radius = advance_radius
        self.lookahead_wp = lookahead_wp
        self.rng = np.random.default_rng(seed)

        # recenter + rebuild the SAME route pool build_sf_scene used, and reuse its
        # per-car route assignment, so planner and scene agree exactly.
        meta = json.load(open(os.path.join(scene_dir, "metadata.json")))
        self.cx, self.cy = meta["map"]["center_utm_minus_origin"]
        rt = meta.get("routing", {})

        net = load_road_network()
        self.pool = build_route_pool(
            net, n_routes=rt.get("n_routes", max(256, n_cars * 4)),
            max_length_m=rt.get("max_length_m", 1500.0),
            seed=rt.get("route_seed", seed))
        # (P, W, 2) recentred; n[p] valid waypoints per route
        self.routes = self.pool.xy - np.array([self.cx, self.cy], np.float32)
        self.route_len = self.pool.n

        car_routes = rt.get("car_routes")
        self.car_route = (np.asarray(car_routes[:n_cars], np.int32) if car_routes
                          else self.rng.integers(0, self.pool.n_routes, size=n_cars))
        self.wp_ptr = np.ones(n_cars, np.int32)  # next waypoint index per car

    def _resample(self, i: int):
        self.car_route[i] = self.rng.integers(0, self.pool.n_routes)
        self.wp_ptr[i] = 1

    def spawn_poses(self):
        """Initial (x, y, yaw) per car at the start of its route — feed these to
        build_sf_scene if you want cars placed exactly on their routes."""
        out = []
        for i in range(self.n):
            r = self.car_route[i]
            p0, p1 = self.routes[r, 0], self.routes[r, 1]
            yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
            out.append((float(p0[0]), float(p0[1]), yaw))
        return out

    def update(self, poses):
        """poses: (N, 2) current xy -> (targets (N,2), target_speeds (N,)).

        Progress = the nearest route waypoint at/after the current pointer (robust
        to pure-pursuit corner-cutting). The chase target is a few waypoints beyond
        that; reaching the last waypoint resamples a fresh route."""
        targets = np.zeros((self.n, 2), np.float32)
        speeds = np.full(self.n, self.target_speed, np.float32)
        for i in range(self.n):
            r = self.car_route[i]
            n_wp = int(self.route_len[r])
            pos = np.asarray(poses[i], np.float32)
            # nearest waypoint in a forward window [ptr-1 .. end], monotonic ptr
            lo = max(0, self.wp_ptr[i] - 1)
            seg = self.routes[r, lo:n_wp]
            nearest = lo + int(np.argmin(np.linalg.norm(seg - pos, axis=1)))
            self.wp_ptr[i] = max(self.wp_ptr[i], nearest + 1)
            # reached the end -> new route, ease off for one tick
            if (self.wp_ptr[i] >= n_wp - 1 and
                    np.linalg.norm(self.routes[r, n_wp - 1] - pos) < self.advance_radius):
                self._resample(i)
                speeds[i] = 0.0
            tgt = min(self.wp_ptr[i] + self.lookahead_wp, n_wp - 1)
            targets[i] = self.routes[self.car_route[i], tgt]
        return targets, speeds
