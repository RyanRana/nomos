"""Run the SF fleet under **real car physics** (local MuJoCo) and export a
trajectories.json the Cesium 3D view consumes — same schema as the kinematic
`demo/export_web`, so the viewer is unchanged.

Why this exists: the kinematic export replays a bicycle model that snaps to
waypoints and *respawns* (teleports) cars on goal/crash. Driving the actual
Ackermann physics car instead gives momentum, a real turn radius, slip/under-
steer, and — because a physics car just keeps rolling when it picks a new route —
**no teleports**. More cars, too: local MuJoCo handles dozens.

Geo-reference comes straight from the scene metadata build_sf_scene wrote
(center_utm_minus_origin + origin_utm + crs), reprojected to WGS84 exactly like
export_web does, so the cars land on the real streets.

    python -m smoothride.worldsim.export_cesium --cars 50 --seconds 30 \
        --out smoothride/demo/web/public/trajectories.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from pyproj import Transformer

from .control_bridge import MultiCarController, yaw_from_quat
from .planner import RoutePlanner

DEFAULT_OUT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "demo", "web", "public", "trajectories.json"))


def _geo(scene_dir):
    """-> (transform fn scene_xy(...,2)->（lon,lat), center [lon,lat]）."""
    meta = json.load(open(os.path.join(scene_dir, "metadata.json")))["map"]
    cx, cy = meta["center_utm_minus_origin"]
    ox_, oy_ = meta["origin_utm"]
    tf = Transformer.from_crs(meta["crs"], "EPSG:4326", always_xy=True)

    def to_lonlat(xy):
        e = xy[..., 0] + cx + ox_
        n = xy[..., 1] + cy + oy_
        lon, lat = tf.transform(e, n)
        return np.asarray(lon), np.asarray(lat)

    clon, clat = to_lonlat(np.zeros(2))   # scene origin = map center
    return to_lonlat, [round(float(clon), 6), round(float(clat), 6)]


def main():
    import mujoco
    from .build_sf_scene import build

    ap = argparse.ArgumentParser()
    ap.add_argument("--cars", type=int, default=50)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--dt", type=float, default=0.1, help="control/sample period")
    ap.add_argument("--speed", type=float, default=9.0, help="target cruise m/s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--buildings", action="store_true", default=True)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    scene_dir = os.path.join(os.path.dirname(__file__), "scenes", "sf-city-v1")
    build(cars=args.cars, out=scene_dir, seed=args.seed, buildings=args.buildings)

    m = mujoco.MjModel.from_xml_path(os.path.join(scene_dir, "scene.xml"))
    d = mujoco.MjData(m)
    bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"c{i}_chassis")
           for i in range(args.cars)]
    for _ in range(200):
        mujoco.mj_step(m, d)                       # settle on the ground

    mc = MultiCarController(args.cars)
    pl = RoutePlanner(args.cars, scene_dir, target_speed=args.speed, seed=args.seed)
    sub = max(1, int(round(args.dt / m.opt.timestep)))
    n_frames = int(args.seconds / args.dt)

    pos_T, yaw_T, spd_T = [], [], []
    for _ in range(n_frames):
        poses = np.array([d.xpos[bid[i]][:2] for i in range(args.cars)])
        yaws = np.array([yaw_from_quat(d.xquat[bid[i]]) for i in range(args.cars)])
        speeds = np.array([float(d.cvel[bid[i]][3:5] @
                                 [np.cos(yaws[i]), np.sin(yaws[i])])
                           for i in range(args.cars)])
        targets, tsp = pl.update(poses)
        d.ctrl[:] = mc.action(poses, yaws, speeds, targets, tsp)
        for _ in range(sub):
            mujoco.mj_step(m, d)
        pos_T.append(poses.copy()); yaw_T.append(yaws.copy()); spd_T.append(speeds.copy())

    pos = np.stack(pos_T)            # (T, N, 2) scene meters
    yaw = np.stack(yaw_T)            # (T, N)
    spd = np.stack(spd_T)            # (T, N)

    to_lonlat, center = _geo(scene_dir)
    lon, lat = to_lonlat(pos)        # (T, N)

    cars = []
    for i in range(args.cars):
        cars.append({
            "lng": [round(float(lon[t, i]), 6) for t in range(n_frames)],
            "lat": [round(float(lat[t, i]), 6) for t in range(n_frames)],
            "hdg": [round(float(yaw[t, i]), 4) for t in range(n_frames)],
            "spd": [round(float(spd[t, i]), 2) for t in range(n_frames)],
            "crash": [0 for _ in range(n_frames)],
        })

    data = {
        "meta": {"dt": args.dt, "n_steps": n_frames, "vmax": args.speed,
                 "center": center, "source": "worldsim-physics"},
        "worlds": {"trained": {"cars": cars}},   # physics fleet (no shadow world)
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    travel = float(np.linalg.norm(pos[-1] - pos[0], axis=1).mean())
    kb = os.path.getsize(args.out) / 1024
    print(f"cars={args.cars} frames={n_frames} mean_travel={travel:.1f} m  "
          f"-> {args.out} ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
