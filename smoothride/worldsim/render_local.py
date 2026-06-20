"""Render the worldsim SF physics demo LOCALLY with stock MuJoCo (CPU) — no
Newton, no Modal, no auth. Same scene + planner + control_bridge the Modal/Newton
path uses; this is the always-works artifact.

    python -m smoothride.worldsim.render_local --cars 16 --seconds 15 --camera oblique
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .control_bridge import MultiCarController, yaw_from_quat
from .planner import RoutePlanner

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "runs", "artifacts"))


def main():
    import mujoco

    ap = argparse.ArgumentParser()
    ap.add_argument("--cars", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--camera", default="oblique")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="worldsim_sf")
    args = ap.parse_args()

    scene_dir = os.path.join(os.path.dirname(__file__), "scenes", "sf-city-v1")
    # (re)generate so --cars matches the scene
    from .build_sf_scene import build
    build(cars=args.cars, out=scene_dir, seed=args.seed, buildings=True)

    m = mujoco.MjModel.from_xml_path(os.path.join(scene_dir, "scene.xml"))
    d = mujoco.MjData(m)
    bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"c{i}_chassis")
           for i in range(args.cars)]
    for _ in range(200):
        mujoco.mj_step(m, d)  # settle

    mc = MultiCarController(args.cars)
    pl = RoutePlanner(args.cars, scene_dir, seed=args.seed)

    renderer = mujoco.Renderer(m, height=args.height, width=args.width)
    dt = float(m.opt.timestep)
    n_steps = int(args.seconds / dt)
    render_every = max(1, int(1.0 / (args.fps * dt)))
    frames, moved_total = [], 0.0
    start = np.array([d.xpos[bid[i]][:2].copy() for i in range(args.cars)])

    for t in range(n_steps):
        poses = np.array([d.xpos[bid[i]][:2] for i in range(args.cars)])
        yaws = [yaw_from_quat(d.xquat[bid[i]]) for i in range(args.cars)]
        speeds = [float(np.linalg.norm(d.cvel[bid[i]][3:5])) for i in range(args.cars)]
        targets, tsp = pl.update(poses)
        d.ctrl[:] = mc.action(poses, yaws, speeds, targets, tsp)
        mujoco.mj_step(m, d)
        if t % render_every == 0:
            renderer.update_scene(d, camera=args.camera)
            frames.append(renderer.render().copy())

    end = np.array([d.xpos[bid[i]][:2] for i in range(args.cars)])
    moved_total = float(np.linalg.norm(end - start, axis=1).mean())

    os.makedirs(OUT, exist_ok=True)
    import imageio
    mp4 = os.path.join(OUT, f"{args.name}.mp4")
    imageio.mimsave(mp4, frames, fps=args.fps)
    for tag, fr in [("start", 0), ("mid", len(frames) // 2), ("end", -1)]:
        imageio.imwrite(os.path.join(OUT, f"{args.name}_{tag}.png"), frames[fr])

    print(f"cars={args.cars}  frames={len(frames)}  mean_travel={moved_total:.1f} m")
    print(f"saved: {mp4} (+ _start/_mid/_end.png)")


if __name__ == "__main__":
    main()
