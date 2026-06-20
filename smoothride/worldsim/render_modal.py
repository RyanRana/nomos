"""Render the 3D San Francisco physics demo on a Modal GPU (Newton needs CUDA).

Builds an image with the Antim/HUD worldsim-template + the bundled Newton wheel,
generates our SF scene, drives every car along a real route with the
control_bridge (pure-pursuit + P throttle), renders frames through the worldsim
sim, and writes an mp4 to a Volume.

    modal run smoothride/worldsim/render_modal.py --cars 24 --seconds 20

Honesty: the driving loop (planner + control_bridge) is MuJoCo-validated locally;
the Newton runtime + worldsim image build are NOT verifiable from this Mac. The
image recipe is faithful; expect to tweak versions/paths on the first run. The
sim is driven over its supported MCP tool API (same path environment/env.py uses).
"""
from __future__ import annotations

import modal

APP = "smoothride-worldsim"
GPU = "L4"                       # Newton/Warp render; L4/A10G plenty for this
WORLDSIM_REPO = "https://github.com/hud-evals/worldsim-template.git"
REPO_DIR = "/worldsim"
OUT_DIR = "/out"

app = modal.App(APP)
volume = modal.Volume.from_name("smoothride-worldsim-out", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "libegl1")
    .run_commands(
        f"git clone --depth 1 {WORLDSIM_REPO} {REPO_DIR}",
        # the template's deps + the bundled Newton wheel (NOT on PyPI)
        "pip install warp-lang mujoco mujoco-warp numpy Pillow scipy trimesh "
        "hud-python fastmcp imageio imageio-ffmpeg",
        f"pip install {REPO_DIR}/wheels/newton-*.whl",
    )
    # our package (planner/control_bridge/build_sf_scene) + the cached SF graph
    .add_local_python_source("smoothride")
    .add_local_dir("data_cache", "/root/data_cache")
    .env({"MUJOCO_GL": "egl"})   # headless offscreen render
)


def _png_bytes(render_result) -> bytes:
    """Extract PNG bytes from a FastMCP render() tool result (ImageContent)."""
    import base64
    content = getattr(render_result, "content", None) or []
    for c in content:
        data = getattr(c, "data", None)
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        if isinstance(data, str):           # base64 in some transports
            return base64.b64decode(data)
    raise RuntimeError("render() returned no image content")


@app.function(image=image, gpu=GPU, timeout=60 * 60,
              volumes={OUT_DIR: volume})
async def render_demo(cars: int = 24, seconds: float = 20.0,
                      camera: str = "oblique", fps: int = 30,
                      width: int = 1280, height: int = 720, seed: int = 0):
    import os
    import sys

    import imageio
    import numpy as np

    sys.path.insert(0, REPO_DIR)            # so `from sim.host import SimHost` resolves
    os.chdir(REPO_DIR)                      # sim/server.py finds ./scenes
    # make osmnx read the baked graph cache, no network on Modal
    os.environ.setdefault("SMOOTHRIDE_DATA_CACHE", "/root/data_cache")

    from fastmcp import Client

    from smoothride.worldsim.build_sf_scene import build
    from smoothride.worldsim.control_bridge import MultiCarController
    from smoothride.worldsim.planner import RoutePlanner

    scene_dir = os.path.join(REPO_DIR, "scenes", "sf-city-v1")
    build(cars=cars, out=scene_dir, seed=seed, buildings=True)

    from sim.host import SimHost
    host = SimHost("mcp")
    await host.start()
    client = Client(host.mcp_url)
    await client.__aenter__()

    async def call(name, **kw):
        res = await client.call_tool(name, kw)
        return getattr(res, "data", None) if name != "render" else res

    await call("reset", scene_id="sf-city-v1")

    mc = MultiCarController(cars)
    pl = RoutePlanner(cars, scene_dir, seed=seed)
    spawn = pl.spawn_poses()
    prev = np.array([[s[0], s[1]] for s in spawn], np.float32)
    yaws = np.array([s[2] for s in spawn], np.float32)

    dt = 0.004                              # scene timestep
    n_steps = int(seconds / dt)
    render_every = max(1, int(1.0 / (fps * dt)))
    frames = []

    for t in range(n_steps):
        poses = np.zeros((cars, 2), np.float32)
        for i in range(cars):
            st = await call("get_object_state", object_name=f"c{i}_chassis")
            p = st["position"]
            poses[i] = (p["x"], p["y"])
        # heading + speed from motion (avoids needing quats over the wire)
        dpos = poses - prev
        moved = np.linalg.norm(dpos, axis=1)
        for i in range(cars):
            if moved[i] > 1e-3:
                yaws[i] = np.arctan2(dpos[i, 1], dpos[i, 0])
        speeds = moved / dt
        prev = poses

        targets, tsp = pl.update(poses)
        await call("step", action=mc.action(poses, yaws, speeds, targets, tsp))

        if t % render_every == 0:
            frames.append(np.asarray(imageio.imread(
                _png_bytes(await call("render", camera=camera, width=width, height=height)))))

    await client.__aexit__(None, None, None)
    await host.stop()

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"sf_physics_{cars}cars.mp4")
    imageio.mimsave(out, frames, fps=fps)
    volume.commit()
    print(f"wrote {out}  ({len(frames)} frames @ {fps}fps)")
    return out


@app.local_entrypoint()
def main(cars: int = 24, seconds: float = 20.0, camera: str = "oblique", seed: int = 0):
    out = render_demo.remote(cars=cars, seconds=seconds, camera=camera, seed=seed)
    print(f"done -> Volume '{volume.name}':{out}")
    print(f"download: modal volume get smoothride-worldsim-out {out.split('/')[-1]} .")
