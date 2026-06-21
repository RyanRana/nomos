"""Run trace schema — the contract the deterministic verifier consumes (handoff §7).

A rollout is logged to a `Trace`: a `TraceManifest` (makes the run replayable
bit-for-bit) plus per-step/per-car arrays. The trace carries *measurements* only —
positions, headings, the road segment each car is on — and the verifier makes *all*
judgments from them. It never touches the env, never replays physics, never calls an
LLM, so the same trace always yields the same verdict (handoff §8/§10).

Coordinates are the sim's metric frame (UTM, origin-shifted); units are SI
(meters, radians CCW from +x, m/s, seconds).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Per-car, per-step fields with shape (T, N). pos/action/seg_start/seg_end carry an
# extra trailing axis and are checked separately.
_TIMELINE_2D = (
    "z", "heading", "speed", "lane", "wp_ptr", "dist_remaining",
    "lane_count", "spawn_grace", "crashed", "arrived", "speed_limit",
)
_TIMELINE_XY = ("pos", "seg_start", "seg_end")


@dataclass(frozen=True)
class TraceManifest:
    """Identity of a run — the four IDs make any run replayable (handoff §10)."""

    run_id: str
    seed: int
    scenario_id: str
    policy_checkpoint_id: str
    config_hash: str            # env params + map version + code version
    dt: float
    n_steps: int
    n_agents: int
    n_peds: int


@dataclass(frozen=True)
class Trace:
    """Immutable recorded trajectory. Validates its own shapes on construction."""

    manifest: TraceManifest
    # timeline, shape (T, N) unless noted
    pos: np.ndarray             # (T, N, 2) meters
    z: np.ndarray               # (T, N) ground elevation, meters
    heading: np.ndarray         # (T, N) radians, CCW from +x
    speed: np.ndarray           # (T, N) m/s
    lane: np.ndarray            # (T, N) i32 discrete lane index
    action: np.ndarray          # (T, N, 3) accel/brake, steer, lane-change
    wp_ptr: np.ndarray          # (T, N) i32 current waypoint along the route
    dist_remaining: np.ndarray  # (T, N) meters left to destination
    # road geometry the verifier judges lane-keeping against
    seg_start: np.ndarray       # (T, N, 2) start of current road segment
    seg_end: np.ndarray         # (T, N, 2) end / current target waypoint
    lane_count: np.ndarray      # (T, N) i32 lanes on the current segment
    spawn_grace: np.ndarray     # (T, N) i32 merge-in immunity countdown
    # events
    crashed: np.ndarray         # (T, N) bool — collision this step (cars + peds)
    arrived: np.ndarray         # (T, N) bool — reached destination; latches True (§0②)
    # static
    speed_limit: np.ndarray     # (T, N) m/s — edge limit under each car
    collision_radius: float
    lane_width: float           # meters — lane offset geometry

    @property
    def n_steps(self) -> int:
        return self.manifest.n_steps

    @property
    def n_agents(self) -> int:
        return self.manifest.n_agents

    def __post_init__(self) -> None:
        T, N = self.manifest.n_steps, self.manifest.n_agents
        for name in _TIMELINE_2D:
            arr = getattr(self, name)
            if arr.shape != (T, N):
                raise ValueError(
                    f"trace.{name} must have shape {(T, N)}, got {arr.shape}")
        for name in _TIMELINE_XY:
            arr = getattr(self, name)
            if arr.shape != (T, N, 2):
                raise ValueError(
                    f"trace.{name} must have shape {(T, N, 2)}, got {arr.shape}")
        if self.action.shape != (T, N, 3):
            raise ValueError(
                f"trace.action must have shape {(T, N, 3)}, got {self.action.shape}")
