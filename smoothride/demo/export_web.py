"""Reproject rollouts into a compact lon/lat JSON.

Shared geo helpers (`_lonlat_transformer`, `_to_lonlat`, `_roads_geojson`,
`_pack_world`) used by the Cesium exporter and the smoke scripts: metric-frame
trajectories are reprojected back to lon/lat (WGS84) so they land on the real San
Francisco street grid, then rounded + packed into per-car arrays to keep the file
small.

Usage:
  python -m smoothride.demo.export_web \
      --trained runs/trained.msgpack --untrained runs/untrained.msgpack \
      --agents 24 --peds 12 --steps 300 --out smoothride/demo/trajectories.json
"""
from __future__ import annotations

import argparse
import json
import os

import jax
import numpy as np
from pyproj import Transformer

from ..data.map_loader import RoadNetwork, load_road_network
from ..env import kinematic as K
from ..env.routing import build_route_pool
from .render import load_params, rollout

DEMO_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_OUT = os.path.join(DEMO_DIR, "trajectories.json")


def _lonlat_transformer(net: RoadNetwork) -> Transformer:
    """Inverse of map_loader's project+origin-shift: shifted-UTM -> lon/lat."""
    crs = net.G.graph["crs"]  # the UTM CRS osmnx picked in project_graph
    return Transformer.from_crs(crs, "EPSG:4326", always_xy=True)


lonlat_transformer = _lonlat_transformer  # public alias


def _to_lonlat(net: RoadNetwork, tf: Transformer, xy: np.ndarray):
    """xy (..., 2) in shifted-UTM meters -> (lon, lat) arrays of the same shape."""
    east = xy[..., 0] + net.origin[0]
    north = xy[..., 1] + net.origin[1]
    lon, lat = tf.transform(east, north)
    return np.asarray(lon), np.asarray(lat)


to_lonlat = _to_lonlat  # public alias


def _roads_geojson(net: RoadNetwork, tf: Transformer) -> list:
    """Every directed edge as a [[lon,lat],[lon,lat]] segment for the map overlay."""
    segs = net.node_xy[net.edges]  # (E, 2, 2) in meters
    lon, lat = _to_lonlat(net, tf, segs)
    out = []
    for e in range(segs.shape[0]):
        out.append([[round(float(lon[e, 0]), 6), round(float(lat[e, 0]), 6)],
                    [round(float(lon[e, 1]), 6), round(float(lat[e, 1]), 6)]])
    return out


def _pack_world(net, tf, tr, stride: int) -> dict:
    """Reproject + round one rollout into per-car / per-ped arrays."""
    pos, head, spd = tr["pos"], tr["heading"], tr["speed"]
    ped = tr["ped"]
    # just_crashed is a per-step event; turn it into a persistent crashed flag.
    crashed = np.cumsum(tr["crashed"].astype(np.int32), axis=0) > 0
    goals = tr["goals"]

    T, N, _ = pos.shape
    frames = range(0, T, stride)

    car_lon, car_lat = _to_lonlat(net, tf, pos)          # (T, N)
    cars = []
    for i in range(N):
        cars.append({
            "lng": [round(float(car_lon[t, i]), 6) for t in frames],
            "lat": [round(float(car_lat[t, i]), 6) for t in frames],
            "hdg": [round(float(head[t, i]), 4) for t in frames],   # rad, CCW from east
            "spd": [round(float(spd[t, i]), 2) for t in frames],
            "crash": [int(crashed[t, i]) for t in frames],
        })

    peds = []
    if ped.shape[1] > 0:
        ped_lon, ped_lat = _to_lonlat(net, tf, ped)      # (T, M)
        for j in range(ped.shape[1]):
            peds.append({
                "lng": [round(float(ped_lon[t, j]), 6) for t in frames],
                "lat": [round(float(ped_lat[t, j]), 6) for t in frames],
            })

    moving_end = int(((spd[-1] > 1.0) & ~crashed[-1]).sum())
    summary = {
        "cars": int(N),
        "peds": int(ped.shape[1]),
        "trips_end": int(goals[-1].sum()),
        "crashed_end": int(crashed[-1].sum()),
        "moving_end": moving_end,
    }
    # cumulative completed trips at each kept frame (drives the live HUD counter)
    trips_series = [int(goals[t].sum()) for t in frames]
    return {"summary": summary, "trips_series": trips_series,
            "cars": cars, "peds": peds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trained", default="runs/trained.msgpack")
    ap.add_argument("--untrained", default="runs/untrained.msgpack")
    ap.add_argument("--agents", type=int, default=24)
    ap.add_argument("--peds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth frame to shrink the file")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    net = load_road_network()
    x0, y0, x1, y1 = net.bounds()
    pool = build_route_pool(net, n_routes=1024, seed=0)
    env = K.make_env(pool, (x0, y0), (x1, y1), n_agents=args.agents,
                     n_peds=args.peds, max_steps=args.steps)
    tf = _lonlat_transformer(net)

    worlds = {}
    for name, ckpt in [("trained", args.trained), ("untrained", args.untrained)]:
        params = load_params(env, ckpt)
        tr = rollout(env, params, jax.random.PRNGKey(args.seed), sample=True)
        worlds[name] = _pack_world(net, tf, tr, args.stride)
        s = worlds[name]["summary"]
        print(f"{name:10s} cars={s['cars']} trips_end={s['trips_end']} "
              f"crashed_end={s['crashed_end']} moving_end={s['moving_end']}")

    # map framing: reproject the metric bounds box corners
    corners = np.array([[x0, y0], [x1, y1]], np.float32)
    clon, clat = _to_lonlat(net, tf, corners)
    center = [round(float(clon.mean()), 6), round(float(clat.mean()), 6)]
    bounds = [[round(float(clon[0]), 6), round(float(clat[0]), 6)],
              [round(float(clon[1]), 6), round(float(clat[1]), 6)]]

    data = {
        "meta": {
            "dt": float(env.dt) * args.stride,
            "n_steps": len(range(0, args.steps, args.stride)),
            "vmax": float(env.v_max),
            "center": center,
            "bounds": bounds,
            "zoom": 15.5,
        },
        "roads": _roads_geojson(net, tf),
        "worlds": worlds,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    kb = os.path.getsize(args.out) / 1024
    print(f"saved: {args.out}  ({kb:.0f} KB, {len(data['roads'])} road segs)")


if __name__ == "__main__":
    main()
