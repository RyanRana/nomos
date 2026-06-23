"""Render the rigid-body WheeledLab hero shot in Isaac Sim — GATED on a GPU/RTX box.

Pipeline per car, per physics step:
    setpoint (target waypoint/velocity/heading)
        -> FROZEN low-level WheeledLab controller  (command-tracking policy)
        -> steering + throttle
        -> rigid-body car (Isaac physics)
        -> RTX-rendered frame

This file is a STRUCTURED STUB: the orchestration and the exact Isaac/WheeledLab
calls are laid out, but the heavy steps are marked `TODO(isaac)` because they need
hardware that isn't present here. A real preflight check runs first, so off-GPU
this exits with a clear explanation instead of a confusing import traceback.

Requirements (the gated box):
    NVIDIA RTX / L40S GPU with RT Cores · Isaac Sim v4.5.0 · IsaacLab v2.0.2 ·
    WheeledLab (BSD-3) car asset + a trained-and-frozen low-level controller ckpt.

Run there with:
    python -m smoothride.demo.isaac.run_isaac_demo \
        --setpoints runs/isaac/trained_setpoints.npz \
        --controller runs/lowlevel/controller.pt \
        --config smoothride/demo/isaac/config.yaml --out runs/isaac/frames
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys


def preflight() -> list[str]:
    """Return a list of missing requirements (empty == good to go)."""
    missing = []
    for mod in ("isaaclab", "isaacsim", "omni"):
        if importlib.util.find_spec(mod) is None:
            missing.append(f"python module '{mod}' (Isaac Sim / IsaacLab)")
    # GPU check (best-effort; torch may not be installed off-box either)
    try:
        import torch
        if not torch.cuda.is_available():
            missing.append("a CUDA GPU (torch.cuda.is_available() == False)")
    except ImportError:
        missing.append("torch with CUDA (for the RTX render box)")
    return missing


def explain_and_exit(missing, args):
    print("=" * 72)
    print(" Isaac / WheeledLab physics demo — NOT runnable on this machine")
    print("=" * 72)
    print(" Missing:")
    for m in missing:
        print(f"   • {m}")
    print()
    print(" This is expected on a Mac / non-GPU box. The Cesium viewer")
    print(" (smoothride/demo/cesium) is the always-works demo; this Isaac path is")
    print(" the cinematic upgrade and needs an RTX/L40S box with Isaac Sim 4.5.0")
    print(" + IsaacLab 2.0.2 + WheeledLab.")
    print()
    print(" What it WOULD do with the setpoints you already exported")
    print(f"   ({args.setpoints}):")
    print("   1. boot Isaac Sim, build the SF road USD (road-mesh-builder)")
    print("   2. spawn N WheeledLab cars (the asset named in the .npz)")
    print("   3. load the FROZEN low-level controller (--controller)")
    print("   4. each step: setpoint -> controller -> throttle/steer -> physics")
    print("   5. RTX-render frames -> stitch to the hero-shot video")
    print()
    print(" Export the setpoints anywhere (no GPU needed):")
    print("   python -m smoothride.demo.isaac.export_setpoints --ckpt runs/trained.msgpack")
    sys.exit(2)


# --------------------------------------------------------------------------
# The real run (only reached on a properly provisioned box).
# --------------------------------------------------------------------------
def run(args):
    import numpy as np
    data = np.load(args.setpoints, allow_pickle=True)
    pos, heading, speed = data["pos"], data["heading"], data["speed"]
    T, N, _ = pos.shape
    car_asset = str(data["car_asset"])
    print(f"loaded setpoints: {N} cars x {T} steps, asset={car_asset}")

    # TODO(isaac): from isaaclab.app import AppLauncher; launch headless+RTX.
    # TODO(isaac): build/import the SF drivable USD (see components/road-mesh-builder).
    # TODO(isaac): spawn N instances of `car_asset` (WheeledLab MuSHR/HOUND).
    # TODO(isaac): load frozen low-level controller from args.controller (RSL-RL/SB3).
    # TODO(isaac): map metric setpoints -> Isaac world frame using data["origin"]/["crs"].
    # TODO(isaac): physics loop:
    #     for t in range(T):
    #         obs = car_states()                       # ego state from Isaac
    #         cmd = controller(obs, setpoint=(pos[t], heading[t], speed[t]))
    #         apply(cmd); world.step()                 # rigid-body integration
    #         if t % args.decimation == 0: capture_rtx_frame(args.out)
    # TODO(isaac): ffmpeg frames -> runs/isaac/hero.mp4
    raise NotImplementedError(
        "Isaac steps are stubbed (TODO(isaac) markers). Implement against your "
        "installed Isaac Sim 4.5.0 / IsaacLab 2.0.2 + WheeledLab on the GPU box.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setpoints", default="runs/isaac/setpoints.npz")
    ap.add_argument("--controller", default="runs/lowlevel/controller.pt",
                    help="frozen low-level WheeledLab command-tracking policy")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__),
                                                     "config.yaml"))
    ap.add_argument("--out", default="runs/isaac/frames")
    ap.add_argument("--decimation", type=int, default=4)
    args = ap.parse_args()

    missing = preflight()
    if missing:
        explain_and_exit(missing, args)
    run(args)


if __name__ == "__main__":
    main()
