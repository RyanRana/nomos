"""Newtonian physics -> the SAME state SmoothRide uses in 2D.

The 2D env (`smoothride.env.kinematic`) integrates a kinematic bicycle to produce
`State(pos, heading, speed, ...)` and packs rollouts as
`tr = {pos (T,N,2), heading (T,N), speed (T,N), crashed, goals, ped}`. The web/Cesium
exporters (`smoothride.demo.export_web`) consume exactly that.

This module makes a **Newton/MuJoCo** rollout emit the *same* objects, so the 3D
physics car is a drop-in for the kinematic car everywhere downstream — including
the Cesium 3D view, which just reprojects `pos` to lon/lat and orients the glTF
model by `heading`. "Bake the physics into the motion; keep the state identical."

Transport-agnostic: it reads pose from anything exposing per-car (pos, quat,
linvel) — a local `mujoco.MjData` (validation here) or the worldsim sim's MCP
sensor dict (`cN_pos/_quat/_linvel`) on a Newton GPU box. The physics is real;
only the *interface* matches the kinematic env.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .control_bridge import MultiCarController, yaw_from_quat


@dataclass
class PhysicsState:
    """Per-car snapshot, field-for-field the dynamic core of the 2D `State`.

    pos (N,2), heading (N,), speed (N,) are the shared physical state. route_idx /
    wp_ptr / lane mirror the 2D bookkeeping so a physics rollout slots into the same
    observation/reward code; they're advanced by the planner, not the physics.
    """
    pos: np.ndarray          # (N, 2)  world x, y  [m]
    heading: np.ndarray      # (N,)    yaw, rad CCW from +x (= east)
    speed: np.ndarray        # (N,)    forward speed [m/s]
    route_idx: np.ndarray    # (N,)    int  (carried from the planner)
    wp_ptr: np.ndarray       # (N,)    int
    lane: np.ndarray         # (N,)    int
    t: int = 0

    @property
    def n(self) -> int:
        return self.pos.shape[0]


# --------------------------------------------------------------------------
# Extraction: sim -> PhysicsState  (the one place physics meets the 2D schema)
# --------------------------------------------------------------------------
def extract_mujoco(model, data, n_cars: int, *,
                   route_idx=None, wp_ptr=None, lane=None, t: int = 0) -> PhysicsState:
    """Read each car's (pos, heading, speed) from a local mujoco MjData.

    Speed is the FORWARD (body-x) component of linear velocity, matching the 2D
    `speed` (signed along heading), not raw |v|."""
    import mujoco
    pos = np.zeros((n_cars, 2))
    heading = np.zeros(n_cars)
    speed = np.zeros(n_cars)
    for i in range(n_cars):
        name = "chassis" if n_cars == 1 else f"c{i}_chassis"
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        pos[i] = data.xpos[bid][:2]
        yaw = yaw_from_quat(data.xquat[bid])
        heading[i] = yaw
        v_world = data.cvel[bid][3:5]                      # linear vel (x, y)
        fwd = np.array([np.cos(yaw), np.sin(yaw)])
        speed[i] = float(v_world @ fwd)                   # signed forward speed
    return _fill(pos, heading, speed, n_cars, route_idx, wp_ptr, lane, t)


def extract_worldsim(sensors: dict, n_cars: int, *,
                     route_idx=None, wp_ptr=None, lane=None, t: int = 0) -> PhysicsState:
    """Same, from the worldsim sim's per-car sensor dict (cN_pos/_quat/_linvel),
    e.g. assembled from get_object_state / get_scene_info over MCP on the GPU box."""
    pos = np.zeros((n_cars, 2))
    heading = np.zeros(n_cars)
    speed = np.zeros(n_cars)
    for i in range(n_cars):
        p = np.asarray(sensors[f"c{i}_pos"], float)
        q = np.asarray(sensors[f"c{i}_quat"], float)
        v = np.asarray(sensors[f"c{i}_linvel"], float)
        pos[i] = p[:2]
        yaw = yaw_from_quat(q)
        heading[i] = yaw
        speed[i] = float(v[:2] @ np.array([np.cos(yaw), np.sin(yaw)]))
    return _fill(pos, heading, speed, n_cars, route_idx, wp_ptr, lane, t)


def _fill(pos, heading, speed, n, route_idx, wp_ptr, lane, t) -> PhysicsState:
    z = lambda a: np.zeros(n, np.int32) if a is None else np.asarray(a, np.int32)
    return PhysicsState(pos=pos, heading=heading, speed=speed,
                        route_idx=z(route_idx), wp_ptr=z(wp_ptr), lane=z(lane), t=t)


# --------------------------------------------------------------------------
# Rollout: drive the physics scene, accumulate a `tr` dict == the 2D one
# --------------------------------------------------------------------------
def rollout_mujoco(model, data, n_cars: int, plan, *, steps: int, dt_ctrl: float,
                   settle: int = 200, controller_kw: dict | None = None) -> dict:
    """Step Newtonian physics under a planner and return a 2D-shaped `tr`.

    plan(t, state) -> (targets (N,2), target_speeds (N,)): the coordination layer
    (trained policy setpoints or a replayed waypoint stream). `tr` has the same
    keys/shapes the kinematic rollout produces, so export_web._pack_world consumes
    it unchanged -> straight into the Cesium 3D view.

    dt_ctrl is the control/sample period (sub-stepped at the scene timestep), and
    becomes `tr["dt"]` so the viewer clock matches.
    """
    import mujoco
    mc = MultiCarController(n_cars, **(controller_kw or {}))
    sub = max(1, int(round(dt_ctrl / model.opt.timestep)))

    for _ in range(settle):
        mujoco.mj_step(model, data)

    pos_T, head_T, spd_T = [], [], []
    st = extract_mujoco(model, data, n_cars, t=0)
    for t in range(steps):
        targets, target_speeds = plan(t, st)
        action = mc.action(st.pos, st.heading, st.speed, targets, target_speeds)
        data.ctrl[:len(action)] = action
        for _ in range(sub):
            mujoco.mj_step(model, data)
        st = extract_mujoco(model, data, n_cars, route_idx=st.route_idx,
                            wp_ptr=st.wp_ptr, lane=st.lane, t=t + 1)
        pos_T.append(st.pos.copy()); head_T.append(st.heading.copy())
        spd_T.append(st.speed.copy())

    pos = np.stack(pos_T)            # (T, N, 2)
    heading = np.stack(head_T)       # (T, N)
    speed = np.stack(spd_T)          # (T, N)
    return {
        "pos": pos, "heading": heading, "speed": speed,
        "crashed": np.zeros((steps, n_cars), bool),     # contact-based crash TODO
        "goals": np.zeros((steps, n_cars), np.int32),
        "ped": np.zeros((steps, 0, 2)),
        "dt": dt_ctrl,
    }


# --------------------------------------------------------------------------
# Self-test: load car-v2 (the MESH car), drive it, prove state extraction.
#   python -m smoothride.worldsim.physics_state
# --------------------------------------------------------------------------
def _selftest():
    import math
    import os
    import mujoco

    scene = os.path.join(os.path.dirname(__file__), "scenes", "car-v2", "scene.xml")
    model = mujoco.MjModel.from_xml_path(scene)   # also proves the mesh asset loads
    data = mujoco.MjData(model)
    print(f"loaded car-v2 (mesh car): {model.ngeom} geoms, "
          f"{model.nmesh} mesh assets, {model.nu} actuators")

    def plan(t, st):
        # one car: aim 30 m ahead of the spawn heading, hold 8 m/s
        yaw0 = 0.0
        target = st.pos[0] + 30.0 * np.array([math.cos(yaw0), math.sin(yaw0)])
        return target[None, :], np.array([8.0])

    tr = rollout_mujoco(model, data, n_cars=1, plan=plan, steps=120,
                        dt_ctrl=0.1, settle=200)
    p0, p1 = tr["pos"][0, 0], tr["pos"][-1, 0]
    travelled = float(np.linalg.norm(p1 - p0))
    print(f"tr keys: {sorted(tr.keys())}")
    print(f"shapes: pos{tr['pos'].shape} heading{tr['heading'].shape} "
          f"speed{tr['speed'].shape}")
    print(f"car drove {travelled:.1f} m, final speed {tr['speed'][-1,0]:.1f} m/s "
          f"({'PHYSICS + STATE OK' if travelled > 3 else 'check gains'})")

    # prove it's the 2D schema: these are the exact keys export_web._pack_world reads
    need = {"pos", "heading", "speed", "crashed", "goals"}
    assert need <= set(tr), f"missing 2D-state keys: {need - set(tr)}"
    print("matches the 2D `tr` schema -> drop-in for export_web -> Cesium ✓")


if __name__ == "__main__":
    _selftest()
