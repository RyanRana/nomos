"""Shared fixtures for the RL-side (trace + verifier) tests.

Pure/offline — no JAX, no env, no network. Fabricate a small `Trace` by hand and
override only the fields a test cares about. Defaults describe a clean run: each car
sits on the lane-0 centerline of a straight +x segment, facing forward, stationary.
"""
from __future__ import annotations

import numpy as np
import pytest

from smoothride.rl.trace import Trace, TraceManifest


@pytest.fixture
def make_trace():
    def _make(n_steps: int = 3, n_agents: int = 2, n_peds: int = 0,
              dt: float = 0.2, collision_radius: float = 2.2,
              lane_width: float = 3.5, **overrides):
        T, N = n_steps, n_agents
        # Straight unit segment along +x; lane-0 centerline is offset right by
        # lane_width*0.5. right-normal of +x is (0,-1), so the centerline sits at
        # y = -lane_width*0.5. Place each car there → lateral offset 0 (on-lane).
        pos = np.zeros((T, N, 2), np.float32)
        pos[..., 1] = -lane_width * 0.5
        seg_start = np.zeros((T, N, 2), np.float32)
        seg_end = np.zeros((T, N, 2), np.float32)
        seg_end[..., 0] = 1.0
        fields = dict(
            pos=pos,
            z=np.zeros((T, N), np.float32),
            heading=np.zeros((T, N), np.float32),     # facing +x = route direction
            speed=np.zeros((T, N), np.float32),       # stationary → no wrong-way
            lane=np.zeros((T, N), np.int32),
            action=np.zeros((T, N, 3), np.float32),
            wp_ptr=np.zeros((T, N), np.int32),
            dist_remaining=np.zeros((T, N), np.float32),
            seg_start=seg_start,
            seg_end=seg_end,
            lane_count=np.ones((T, N), np.int32),
            spawn_grace=np.zeros((T, N), np.int32),
            crashed=np.zeros((T, N), bool),
            arrived=np.zeros((T, N), bool),
            speed_limit=np.full((T, N), 1e9, np.float32),
        )
        fields.update(overrides)
        manifest = TraceManifest(
            run_id="test-run", seed=0, scenario_id="test", policy_checkpoint_id="ckpt",
            config_hash="hash", dt=dt, n_steps=T, n_agents=N, n_peds=n_peds,
        )
        return Trace(manifest=manifest, collision_radius=collision_radius,
                     lane_width=lane_width, **fields)

    return _make
