"""Drive the worldsim cars from setpoints — the handoff between SmoothRide's
coordination layer and the Newton physics scene.

Our high-level policy (or any planner) emits, per car, a target waypoint + target
speed. This module turns that into the per-car [drive_rl, drive_rr, steer_l,
steer_r] the scene's step(action) expects, using a classic, weight-free low-level
controller (pure-pursuit steering + a P throttle). It is transport-agnostic: feed
it plain pose arrays, whether they came from the worldsim sensor dict or a local
MuJoCo MjData.

This is the "physics for all cars, no GPU, no learned controller required" path —
the same role the WheeledLab controller would play, but classical. Swap in the
learned controller.pt later by replacing CarController.command().
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

WHEEL_RADIUS = 0.33
WHEELBASE = 2.7
DRIVE_RANGE = (-40.0, 80.0)   # must match the scene's drive ctrlrange
STEER_MAX = 0.7               # must match the scene's steer ctrlrange


def yaw_from_quat(q) -> float:
    """Heading (rad) about +z from a (w, x, y, z) quaternion."""
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class CarController:
    wheel_radius: float = WHEEL_RADIUS
    wheelbase: float = WHEELBASE
    kp_speed: float = 3.0
    min_lookahead: float = 4.0

    def command(self, pos, yaw: float, speed: float, target_xy, target_speed: float):
        """-> (drive_ctrl rad/s, steer_ctrl rad). Both rear wheels share drive;
        both front wheels share steer (Ackermann simplification)."""
        dx = target_xy[0] - pos[0]
        dy = target_xy[1] - pos[1]
        # pure-pursuit steering toward the lookahead point
        alpha = _wrap(math.atan2(dy, dx) - yaw)
        ld = max(self.min_lookahead, math.hypot(dx, dy))
        steer = math.atan2(2.0 * self.wheelbase * math.sin(alpha), ld)
        steer = float(np.clip(steer, -STEER_MAX, STEER_MAX))
        # P throttle (feedforward target + correction), as wheel angular velocity
        v_cmd = target_speed + self.kp_speed * (target_speed - speed)
        drive = float(np.clip(v_cmd / self.wheel_radius, *DRIVE_RANGE))
        return drive, steer


class MultiCarController:
    """One CarController shared across N homogeneous cars -> the flat action vector
    the generated SF scene consumes (4 entries per car, in spawn order)."""

    def __init__(self, n_cars: int, **kw):
        self.n = n_cars
        self.ctrl = CarController(**kw)

    def action(self, poses, yaws, speeds, targets, target_speeds) -> list[float]:
        out: list[float] = []
        for i in range(self.n):
            drive, steer = self.ctrl.command(
                poses[i], yaws[i], speeds[i], targets[i], target_speeds[i])
            out += [drive, drive, steer, steer]
        return out


def states_from_worldsim(get_state_result: dict, n_cars: int):
    """Adapt the worldsim get_state() sensor_data dict -> (poses, yaws, speeds).

    The generated scene exposes per-car sensors cI_pos (3), cI_quat (4),
    cI_linvel (3). get_state() flattens 1-dim sensors only, so for full vectors
    read get_object_state('cI_chassis') instead; this helper shows the mapping."""
    raise NotImplementedError(
        "Read per-car pose via get_object_state('cN_chassis') over MCP, or the "
        "cN_pos/cN_quat/cN_linvel sensors — see HANDOFF.md §control loop.")


# --------------------------------------------------------------------------
# Local self-test: drive a car in the generated SF scene with plain MuJoCo
# (no Newton needed) to prove the controller closes the loop.
#   python -m smoothride.worldsim.control_bridge
# --------------------------------------------------------------------------
def _selftest():
    import os
    import mujoco

    scene = os.path.join(os.path.dirname(__file__), "scenes", "sf-city-v1", "scene.xml")
    if not os.path.exists(scene):
        print("generate the scene first: python -m smoothride.worldsim.build_sf_scene --cars 8")
        return
    m = mujoco.MjModel.from_xml_path(scene)
    d = mujoco.MjData(m)
    for _ in range(200):
        mujoco.mj_step(m, d)  # settle

    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "c0_chassis")
    start = d.xpos[bid][:2].copy()
    # a waypoint 25 m ahead of car0's current heading
    yaw0 = yaw_from_quat(d.xquat[bid])
    target = start + 25.0 * np.array([math.cos(yaw0), math.sin(yaw0)])

    ctrl = CarController()
    d0 = float(np.linalg.norm(target - start))
    for _ in range(900):  # ~3.6 s
        pos = d.xpos[bid][:2]
        yaw = yaw_from_quat(d.xquat[bid])
        speed = float(np.linalg.norm(d.cvel[bid][3:5]))  # linear vel xy
        drive, steer = ctrl.command(pos, yaw, speed, target, target_speed=8.0)
        d.ctrl[0:4] = [drive, drive, steer, steer]
        mujoco.mj_step(m, d)

    end = d.xpos[bid][:2].copy()
    dist = float(np.linalg.norm(target - end))
    print(f"car0 start={np.round(start,1)} target={np.round(target,1)} end={np.round(end,1)}")
    print(f"distance to target: {d0:.1f} m -> {dist:.1f} m "
          f"({'CLOSED THE LOOP' if dist < d0 - 5 else 'check gains'})")


if __name__ == "__main__":
    _selftest()
