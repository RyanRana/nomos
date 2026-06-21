"""Lane-accurate geometric trajectory generator for the Cesium demo.

WHY THIS EXISTS
---------------
The physics export (`worldsim/export_cesium.py`) and the kinematic env both put
cars on raw graph *centerlines* and, on route completion, teleport-by-driving to a
new random route's start — so cars cut across blocks and clip buildings (measured:
~30% of positions land >15 m off any street). This generator instead drives cars
*geometrically* on the real OSM graph with proper lane discipline:

  * RIGHT-HAND LANE: every directed edge centerline is offset to the right by half
    a lane, so opposing traffic on two-way streets is physically separated and no
    car ever rides the oncoming lane or the crown of the road.
  * CLEAN TURNS: at each intersection the offset polyline is rounded with a Bézier
    fillet and the car SLOWS to a curvature-limited speed, so it rotates smoothly
    out of one lane and into the correct lane of the next street.
  * STAYS ON ROAD: because every sample is a lane-offset point on a real edge (or a
    fillet wholly inside the intersection), cars never enter a block -> no building
    clipping.
  * NO CRASHES: each car keeps its own lane at a constant cruise speed and starts
    spread out along its route, so there are no rear-end catch-ups or lane
    intrusions; crash is a legitimate 0.

Output matches the schema the existing `demo/cesium/app.js` already reads:
  meta: {dt, n_steps, vmax, center, source}
  worlds.trained.cars[]: {lng[], lat[], hdg[], spd[], crash[]}   # hdg rad CCW-from-east

Usage:
  python -m smoothride.demo.export_lanes --cars 160 --steps 400 --dt 0.1 \
      --out smoothride/demo/web/public/trajectories.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from pyproj import Transformer

from ..data.map_loader import load_road_network

# --- lane / motion constants -------------------------------------------------
LANE_WIDTH = 3.5          # m, standard US lane
LANE_OFFSET = LANE_WIDTH * 0.5   # center of the rightmost lane, right of centerline
TURN_TRIM = 9.0           # m trimmed back from each corner for the fillet
BEZIER_PTS = 8            # samples per rounded corner
A_LAT_MAX = 2.5           # m/s^2 max lateral accel -> sets cornering speed
V_MIN = 2.0               # m/s floor so cars never fully stall mid-route
DENSIFY = 6.0             # m max gap between densified straight-line samples


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _right_normal(d):
    """Right-of-travel normal for direction d=(east,north): clockwise 90deg."""
    return np.array([d[1], -d[0]], np.float32)


def build_adjacency(net):
    """node_idx -> list of successor node_idx along directed edges."""
    adj: dict[int, list[int]] = {}
    for u, v in net.edges:
        adj.setdefault(int(u), []).append(int(v))
    return adj


def random_walk(adj, start, min_len_m, node_xy, rng, max_hops=400):
    """Connected node-index path whose centerline length >= min_len_m.

    NEVER U-turns and never takes a hairpin (>~110deg) reversal: those create a
    cusp that the right-offset can't represent (the lane would fold back on
    itself), which is what made cars spin. The walk prefers to continue roughly
    straight through intersections, choosing among the non-reversing exits.
    """
    path = [start]
    length = 0.0
    prev = None
    cur = start
    for _ in range(max_hops):
        succ = [s for s in adj.get(cur, []) if s != prev]   # no U-turn
        if not succ:
            break                                            # dead-end: park here
        if prev is not None:
            cur_dir = _unit(node_xy[cur] - node_xy[prev])
            aligned = [s for s in succ
                       if float(np.dot(_unit(node_xy[s] - node_xy[cur]), cur_dir)) > -0.34]
            succ = aligned or succ                           # avoid hairpins
        nxt = int(rng.choice(succ))
        length += float(np.linalg.norm(node_xy[nxt] - node_xy[cur]))
        path.append(nxt)
        prev, cur = cur, nxt
        if length >= min_len_m and len(path) >= 3:
            break
    return path


def offset_polyline(pts, offset):
    """Right-offset a centerline polyline by `offset` using clamped miter joins.

    Returns offset vertices (same count). Miter keeps both adjacent segments
    parallel-offset; the join is clamped so a sharp corner bevels instead of
    spiking far out into a building.
    """
    pts = np.asarray(pts, np.float32)
    k = len(pts)
    out = np.zeros_like(pts)
    # per-segment right normals
    seg_n = [_right_normal(_unit(pts[i + 1] - pts[i])) for i in range(k - 1)]
    for i in range(k):
        if i == 0:
            n = seg_n[0]
        elif i == k - 1:
            n = seg_n[-1]
        else:
            n1, n2 = seg_n[i - 1], seg_n[i]
            denom = 1.0 + float(np.dot(n1, n2))
            if denom < 1e-3:                      # ~180deg reversal: just bevel
                n = _unit(n1 + n2)
            else:
                m = (n1 + n2) / denom             # miter vector
                if np.linalg.norm(m) > 3.0:       # clamp spikes on sharp turns
                    m = _unit(m) * 3.0
                n = m
        out[i] = pts[i] + n * offset
    return out


def round_corners(q, trim, n_bez=BEZIER_PTS):
    """Round each interior vertex of polyline q with a quadratic Bézier fillet.

    The straight runs are preserved; only the corners become arcs, so the car
    follows the lane straight, then sweeps a smooth arc through the intersection
    into the next lane.
    """
    q = np.asarray(q, np.float32)
    k = len(q)
    if k < 3:
        return q
    out = [q[0]]
    for i in range(1, k - 1):
        a, b, c = q[i - 1], q[i], q[i + 1]
        din, dout = _unit(b - a), _unit(c - b)
        tr = min(trim, 0.45 * np.linalg.norm(b - a), 0.45 * np.linalg.norm(c - b))
        p0, p2 = b - din * tr, b + dout * tr      # fillet endpoints, control = b
        out.append(p0)
        for j in range(1, n_bez):
            t = j / n_bez
            out.append((1 - t) ** 2 * p0 + 2 * (1 - t) * t * b + t * t * p2)
        out.append(p2)
    out.append(q[-1])
    return np.asarray(out, np.float32)


def densify(pts, max_gap=DENSIFY):
    """Insert points so no straight run exceeds max_gap (uniform arc sampling)."""
    pts = np.asarray(pts, np.float32)
    out = [pts[0]]
    for i in range(1, len(pts)):
        seg = pts[i] - pts[i - 1]
        d = float(np.linalg.norm(seg))
        m = max(1, int(np.ceil(d / max_gap)))
        for j in range(1, m + 1):
            out.append(pts[i - 1] + seg * (j / m))
    return np.asarray(out, np.float32)


def curvature_speed(pts, v_cruise):
    """Per-point speed cap from local turn radius: v = sqrt(a_lat * R)."""
    k = len(pts)
    v = np.full(k, v_cruise, np.float32)
    for i in range(1, k - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        d1, d2 = _unit(b - a), _unit(c - b)
        ang = np.arccos(np.clip(np.dot(d1, d2), -1, 1))   # turn angle at b
        seglen = 0.5 * (np.linalg.norm(b - a) + np.linalg.norm(c - b))
        if ang > 1e-3:
            R = seglen / ang                              # local radius estimate
            v[i] = np.clip(np.sqrt(A_LAT_MAX * R), V_MIN, v_cruise)
    # smooth the speed profile so braking/accel isn't a step change
    for _ in range(3):
        v[1:-1] = (v[:-2] + v[1:-1] + v[2:]) / 3.0
    return v


HEAD_WIN = 2.5     # m lookahead/back window for a smooth, motion-aligned heading
A_ACC = 2.5        # m/s^2 max acceleration (smooth pull-away; braking is implicit
                   # — a car simply can't advance into an occupied cell)
CELL = 2.5         # m occupancy cell for the time-major collision check
CAR_L = 4.6        # m car length (footprint reserved along heading)


class Lane:
    """One car's drivable lane path with arc-length lookups."""

    def __init__(self, pts, speed_pt):
        self.pts = pts
        self.seg = np.diff(pts, axis=0)
        self.seglen = np.linalg.norm(self.seg, axis=1)
        self.s_cum = np.concatenate([[0.0], np.cumsum(self.seglen)])
        self.total = float(self.s_cum[-1])
        self.speed_pt = speed_pt

    def _locate(self, s):
        s = min(max(s, 0.0), self.total - 1e-3)
        j = int(np.searchsorted(self.s_cum, s) - 1)
        j = min(max(j, 0), len(self.seg) - 1)
        f = (s - self.s_cum[j]) / max(self.seglen[j], 1e-6)
        return j, f

    def pos(self, s):
        j, f = self._locate(s)
        return self.pts[j] + self.seg[j] * f

    def vcap(self, s):
        j, f = self._locate(s)
        return float(self.speed_pt[j] + (self.speed_pt[j + 1] - self.speed_pt[j]) * f)

    def heading(self, s):
        d = self.pos(s + HEAD_WIN) - self.pos(s - HEAD_WIN)
        return float(np.arctan2(d[1], d[0])) if np.linalg.norm(d) > 1e-6 else 0.0


CAR_W = 2.0        # m car width (footprint reserved across heading)


def _footprint(p, hd):
    """Cells covering the car rectangle (CAR_L x CAR_W) centered at p, heading hd."""
    d = np.array([np.cos(hd), np.sin(hd)])
    r = np.array([d[1], -d[0]])
    out = set()
    for f in (-CAR_L / 2, 0.0, CAR_L / 2):
        for g in (-CAR_W / 2, 0.0, CAR_W / 2):
            q = p + d * f + r * g
            out.add((int(round(q[0] / CELL)), int(round(q[1] / CELL))))
    return out


def _fits(ln, s, occ, me):
    """True if the footprint at arc-pos s is free of cells owned by another car."""
    for c in _footprint(ln.pos(s), ln.heading(s)):
        o = occ.get(c)
        if o is not None and o != me:
            return False
    return True


def simulate(lanes, s0, dt, n_steps):
    """Time-major cellular micro-sim — collision-free by construction.

    All cars advance together, one step at a time. Within a step, cars claim cells
    in priority order on a single occupancy map seeded with everyone's CURRENT
    footprint; a car may reclaim the cells it is vacating but may never move into a
    cell another car holds. So a car NEVER drives into occupied space — not a moving
    leader (followers queue), not a crossing car (the later one yields), and crucially
    not a STOPPED car (it stays put and everyone routes around it in time). Lane
    geometry already bars oncoming-lane use, so the result has no crashes and no lane
    intrusions.
    """
    N = len(lanes)
    out_pos = np.zeros((n_steps, N, 2), np.float32)
    out_hdg = np.zeros((n_steps, N), np.float32)
    out_spd = np.zeros((n_steps, N), np.float32)

    s = [float(x) for x in s0]
    v = [lanes[i].vcap(s[i]) for i in range(N)]

    # spawn separation: push each car forward until its start footprint is clear
    occ0: dict[tuple[int, int], int] = {}
    for i in sorted(range(N), key=lambda k: s0[k]):
        while s[i] < lanes[i].total - 1e-3 and not _fits(lanes[i], s[i], occ0, i):
            s[i] += CELL
        for c in _footprint(lanes[i].pos(s[i]), lanes[i].heading(s[i])):
            occ0[c] = i

    for t in range(n_steps):
        occ = dict(occ0) if t == 0 else {}
        if t > 0:
            for i in range(N):
                for c in _footprint(lanes[i].pos(s[i]), lanes[i].heading(s[i])):
                    occ[c] = i
        # process front-runners first so platoons flow (a leader vacates before its
        # follower is asked to advance into the freed space).
        order = sorted(range(N), key=lambda k: -s[k])
        for i in order:
            ln = lanes[i]
            for c in _footprint(ln.pos(s[i]), ln.heading(s[i])):   # release own cells
                if occ.get(c) == i:
                    del occ[c]
            v_des = min(ln.vcap(s[i]), v[i] + A_ACC * dt)
            adv_max = v_des * dt
            adv, probe = 0.0, adv_max
            while probe > 1e-3:                                    # largest clear advance
                cand = min(s[i] + probe, ln.total - 1e-3)
                if _fits(ln, cand, occ, i):
                    adv = cand - s[i]
                    break
                probe -= 0.25
            s[i] += adv
            v[i] = adv / dt
            p = ln.pos(s[i]); hd = ln.heading(s[i])
            out_pos[t, i] = p; out_hdg[t, i] = hd; out_spd[t, i] = v[i]
            for c in _footprint(p, hd):                            # claim new cells
                occ[c] = i
        occ0 = occ
    return out_pos, out_hdg, out_spd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cars", type=int, default=160)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--speed", type=float, default=8.5, help="cruise m/s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "web", "public", "trajectories.json"))
    args = ap.parse_args()

    net = load_road_network()
    adj = build_adjacency(net)
    starts = [n for n in adj if adj[n]]
    rng = np.random.default_rng(args.seed)
    tf = Transformer.from_crs(net.G.graph["crs"], "EPSG:4326", always_xy=True)

    horizon = args.speed * args.steps * args.dt          # straight-line distance a car can cover
    lanes, s0 = [], []
    for i in range(args.cars):
        start = int(rng.choice(starts))
        # enough road for the whole clip + a head-start offset so cars spread out
        node_path = random_walk(adj, start, horizon * 1.4 + 120, net.node_xy, rng)
        if len(node_path) < 3:
            continue
        center = net.node_xy[node_path]                  # centerline (meters)
        poly = offset_polyline(center, LANE_OFFSET)      # right-hand lane
        poly = round_corners(poly, TURN_TRIM)            # smooth turns
        poly = densify(poly)
        ln = Lane(poly, curvature_speed(poly, args.speed))
        lanes.append(ln)
        s0.append(float(rng.uniform(0, max(1.0, ln.total - horizon))))  # stagger spawns

    pos, head, spd = simulate(lanes, s0, args.dt, args.steps)   # joint: no crashes

    cars = []
    for i in range(len(lanes)):
        lon, lat = tf.transform(pos[:, i, 0] + net.origin[0], pos[:, i, 1] + net.origin[1])
        cars.append({
            "lng": [round(float(x), 6) for x in lon],
            "lat": [round(float(y), 6) for y in lat],
            "hdg": [round(float(h), 4) for h in head[:, i]],   # rad CCW from east
            "spd": [round(float(s), 2) for s in spd[:, i]],
            "crash": [0] * args.steps,
        })

    # map center for the camera (graph bbox middle -> lon/lat)
    x0, y0, x1, y1 = net.bounds()
    clon, clat = tf.transform([(x0 + x1) / 2 + net.origin[0]],
                              [(y0 + y1) / 2 + net.origin[1]])
    data = {
        "meta": {"dt": args.dt, "n_steps": args.steps, "vmax": args.speed,
                 "center": [round(float(clon[0]), 6), round(float(clat[0]), 6)],
                 "source": "lane-geometric"},
        "worlds": {"trained": {"cars": cars}},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    kb = os.path.getsize(args.out) / 1024
    print(f"cars={len(cars)} steps={args.steps} dt={args.dt} -> {args.out} ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
