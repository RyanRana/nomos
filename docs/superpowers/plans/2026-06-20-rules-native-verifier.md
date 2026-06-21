# Rules-native deterministic verifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the deterministic verifier *own* the traffic rules — derive lane-keeping, wrong-way, speed, and collision verdicts from logged geometry instead of trusting sim-computed verdict flags.

**Architecture:** The `Trace` carries raw measurements (positions, headings, the road segment each car is on, lane counts); `verify()` applies pure-numpy geometric/arithmetic predicates to produce per-car and run-level verdicts. No env import, no physics replay, no randomness — same trace → same verdict (handoff §8/§10). The lane geometry math is `env/legality.py`'s off-lane/wrong-way logic re-homed in numpy over a `(T, N)` timeline.

**Tech Stack:** Python 3.13, numpy 2.x, pytest. Pure-numpy layer (no JAX); tests use hand-built traces, no sim.

## Global Constraints

- **Pure verifier:** `smoothride/rl/verifier.py` and `smoothride/rl/trace.py` MUST NOT import `smoothride.env` (or any JAX). numpy + stdlib only.
- **Immutability:** `Trace`, `CarVerdict`, `RunVerdict` are `@dataclass(frozen=True)`.
- **Determinism:** no randomness, no wall-clock, no network, no LLM.
- **Constants (verifier-owned, faithful to env defaults):** `OFFLANE_THRESH = 5.0`, `WRONGWAY_COS = -0.25`, `IDLE_SPEED = 0.5`, `MAX_LANES = 8`, `SPEED_EPS = 1e-6`. Default `lane_width = 3.5`.
- **TDD:** every behavior gets a failing test first. Run `python3 -m pytest` from the worktree root.
- **Frame/units:** metric frame, meters, radians CCW from +x wrapped to (−π, π], m/s.

---

### Task 1: Trace schema v2 (measurements in, verdicts out)

Replace the sim-computed verdict fields (`off_road`, `rule_violation`, `road_polygon_ref`) with the raw geometry the verifier needs (`seg_start`, `seg_end`, `lane_count`, `spawn_grace`, `lane_width`).

**Files:**
- Modify: `smoothride/rl/trace.py`
- Modify: `tests/rl/conftest.py`
- Modify: `tests/rl/test_trace.py`

**Interfaces:**
- Consumes: nothing (foundation task).
- Produces:
  - `Trace` frozen dataclass with fields: `manifest, pos (T,N,2), z (T,N), heading (T,N), speed (T,N), lane (T,N) i32, action (T,N,3), wp_ptr (T,N) i32, dist_remaining (T,N), seg_start (T,N,2), seg_end (T,N,2), lane_count (T,N) i32, spawn_grace (T,N) i32, crashed (T,N) bool, arrived (T,N) bool, speed_limit (T,N), collision_radius: float, lane_width: float`. Properties `n_steps`, `n_agents`.
  - `TraceManifest` unchanged.
  - `make_trace(n_steps=3, n_agents=2, n_peds=0, dt=0.2, collision_radius=2.2, lane_width=3.5, **overrides) -> Trace` test factory; defaults describe a clean, on-lane, stationary run.

- [ ] **Step 1: Rewrite `smoothride/rl/trace.py`**

```python
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
```

- [ ] **Step 2: Update `tests/rl/conftest.py` factory to the v2 schema**

```python
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
```

- [ ] **Step 3: Rewrite `tests/rl/test_trace.py`**

```python
"""Trace schema (handoff §7) — the contract the verifier reads.

The trace validates its own array shapes on construction (validate at boundaries).
"""
import numpy as np
import pytest


def test_trace_exposes_step_and_agent_counts(make_trace):
    tr = make_trace(n_steps=4, n_agents=3)
    assert tr.n_steps == 4
    assert tr.n_agents == 3


def test_trace_is_immutable(make_trace):
    tr = make_trace()
    with pytest.raises(Exception):
        tr.pos = np.zeros_like(tr.pos)


def test_trace_rejects_bad_timeline_shape(make_trace):
    with pytest.raises(ValueError, match="speed"):
        make_trace(n_steps=3, n_agents=2, speed=np.zeros((3, 5), np.float32))


def test_trace_rejects_bad_lane_count_shape(make_trace):
    with pytest.raises(ValueError, match="lane_count"):
        make_trace(n_steps=3, n_agents=2, lane_count=np.zeros((3, 5), np.int32))


def test_trace_rejects_bad_seg_shape(make_trace):
    with pytest.raises(ValueError, match="seg_start"):
        make_trace(n_steps=2, n_agents=1, seg_start=np.zeros((2, 1), np.float32))


def test_trace_rejects_bad_action_width(make_trace):
    with pytest.raises(ValueError, match="action"):
        make_trace(n_steps=2, n_agents=1, action=np.zeros((2, 1, 2), np.float32))
```

- [ ] **Step 4: Run trace tests — they should pass**

Run: `python3 -m pytest tests/rl/test_trace.py -q`
Expected: 6 passed. (Note: `test_verifier.py` will be RED until Task 3 — run just `test_trace.py` here.)

- [ ] **Step 5: Commit**

```bash
git add smoothride/rl/trace.py tests/rl/conftest.py tests/rl/test_trace.py
git commit -m "feat: Trace schema v2 — log road geometry, drop sim-computed verdicts"
```

---

### Task 2: Lane geometry helpers (off-lane distance + wrong-way)

Re-home `env/legality.py`'s point-to-segment lane-distance and heading-vs-route math as pure-numpy helpers over a `(T, N)` timeline.

**Files:**
- Modify: `smoothride/rl/verifier.py` (create the module with constants + helpers)
- Create: `tests/rl/test_geometry.py`

**Interfaces:**
- Consumes: nothing from other tasks (operates on raw numpy arrays).
- Produces:
  - Constants `OFFLANE_THRESH, WRONGWAY_COS, IDLE_SPEED, MAX_LANES, SPEED_EPS`.
  - `_wrap(angle: np.ndarray) -> np.ndarray` — wrap to (−π, π].
  - `lateral_offset(pos, seg_start, seg_end, lane_count, lane_width, max_lanes=MAX_LANES) -> np.ndarray` — `(T, N)` distance to the **nearest** valid lane centerline.
  - `wrong_way(heading, seg_start, seg_end, speed, spawn_grace, wrongway_cos=WRONGWAY_COS, idle_speed=IDLE_SPEED) -> np.ndarray` — `(T, N)` bool.

- [ ] **Step 1: Write the failing tests in `tests/rl/test_geometry.py`**

```python
"""Pure-numpy geometry helpers behind the lane rules (mirror env/legality.py)."""
import numpy as np

from smoothride.rl.verifier import _wrap, lateral_offset, wrong_way

LW = 3.5


def _straight_x(T, N):
    """Straight unit segment along +x for every car/step."""
    seg_start = np.zeros((T, N, 2), np.float32)
    seg_end = np.zeros((T, N, 2), np.float32)
    seg_end[..., 0] = 1.0
    return seg_start, seg_end


def test_wrap_brings_angle_into_pi_range():
    out = _wrap(np.array([0.0, 3 * np.pi, -3 * np.pi, np.pi]))
    assert np.allclose(out, [0.0, np.pi, np.pi, np.pi], atol=1e-6)


def test_on_lane_centerline_offset_is_zero():
    seg_start, seg_end = _straight_x(1, 1)
    pos = np.array([[[0.0, -LW * 0.5]]], np.float32)   # lane-0 centerline (y=-1.75)
    lane_count = np.ones((1, 1), np.int32)
    d = lateral_offset(pos, seg_start, seg_end, lane_count, LW)
    assert d.shape == (1, 1)
    assert np.allclose(d, 0.0, atol=1e-5)


def test_far_off_road_offset_is_large():
    seg_start, seg_end = _straight_x(1, 1)
    pos = np.array([[[0.0, 10.0]]], np.float32)        # ~11.75 m from lane-0 center
    lane_count = np.ones((1, 1), np.int32)
    d = lateral_offset(pos, seg_start, seg_end, lane_count, LW)
    assert d[0, 0] > 5.0


def test_nearest_lane_is_chosen_on_multilane_road():
    # 3-lane road, car sits on lane-2 centerline (offset LW*2.5 = 8.75 → y=-8.75).
    # Distance to lane-0 centerline is 7.0 (>thresh); nearest (lane-2) is ~0.
    seg_start, seg_end = _straight_x(1, 1)
    pos = np.array([[[0.0, -LW * 2.5]]], np.float32)
    lane_count = np.full((1, 1), 3, np.int32)
    d = lateral_offset(pos, seg_start, seg_end, lane_count, LW)
    assert np.allclose(d, 0.0, atol=1e-5)


def test_wrong_way_true_when_heading_reversed_and_moving():
    seg_start, seg_end = _straight_x(1, 1)
    heading = np.full((1, 1), np.pi, np.float32)        # facing -x against +x route
    speed = np.full((1, 1), 5.0, np.float32)
    grace = np.zeros((1, 1), np.int32)
    assert wrong_way(heading, seg_start, seg_end, speed, grace)[0, 0]


def test_wrong_way_false_when_stationary():
    seg_start, seg_end = _straight_x(1, 1)
    heading = np.full((1, 1), np.pi, np.float32)
    speed = np.zeros((1, 1), np.float32)                # ≤ idle → not moving
    grace = np.zeros((1, 1), np.int32)
    assert not wrong_way(heading, seg_start, seg_end, speed, grace)[0, 0]


def test_wrong_way_false_during_spawn_grace():
    seg_start, seg_end = _straight_x(1, 1)
    heading = np.full((1, 1), np.pi, np.float32)
    speed = np.full((1, 1), 5.0, np.float32)
    grace = np.ones((1, 1), np.int32)                   # immune
    assert not wrong_way(heading, seg_start, seg_end, speed, grace)[0, 0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/rl/test_geometry.py -q`
Expected: FAIL — `ImportError: cannot import name '_wrap'` (module/helpers not defined yet).

- [ ] **Step 3: Create `smoothride/rl/verifier.py` with constants + helpers**

```python
"""Deterministic verifier — the reward/validity source of truth (handoff §8, §10).

Principle: *verify the trace, don't re-simulate.* The verifier OWNS the rules: it
derives lane-keeping, wrong-way, speed, and collision verdicts from logged geometry
with pure geometric/arithmetic predicates, so the same trace yields the same verdict
regardless of GPU/float non-determinism.

Hard constraints (this module is pure):
  * no randomness, no wall-clock, no network, no LLM (Cosmos-Reason is NOT here)
  * no physics replay, never imports the env
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .trace import Trace

# Rule constants — the verifier owns these (faithful to env defaults).
OFFLANE_THRESH = 5.0     # m from nearest lane centerline; ~1.5 lane-widths
WRONGWAY_COS = -0.25     # heading-vs-route cosine below this == wrong way (~>105°)
IDLE_SPEED = 0.5         # m/s; below this a car isn't "moving" (no wrong-way)
MAX_LANES = 8            # generous per-segment lane bound; extra slots masked out
SPEED_EPS = 1e-6         # absorbs float noise in the speed-limit cross-check


def _wrap(angle: np.ndarray) -> np.ndarray:
    """Wrap radians to (−π, π]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def lateral_offset(pos: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray,
                   lane_count: np.ndarray, lane_width: float,
                   max_lanes: int = MAX_LANES) -> np.ndarray:
    """(T, N) distance from each car to the NEAREST valid lane centerline of the
    segment it is on. Point-to-segment, nearest-lane: legal lane changes and
    corner-cuts read as legal; only leaving the roadway grows it (mirrors
    env/legality.py)."""
    seg = seg_end - seg_start
    seglen = np.linalg.norm(seg, axis=-1, keepdims=True)
    u = seg / (seglen + 1e-6)                                  # (T,N,2) along-segment
    right = np.stack([u[..., 1], -u[..., 0]], axis=-1)        # (T,N,2) right-normal

    ls = np.arange(max_lanes)
    offs = lane_width * (ls + 0.5)                            # (L,) lane offsets
    valid = ls < np.maximum(lane_count, 1)[..., None]        # (T,N,L)

    # lane lines: segment shifted right by each lane offset → endpoints a, b
    a = seg_start[:, :, None, :] + right[:, :, None, :] * offs[None, None, :, None]
    b = seg_end[:, :, None, :] + right[:, :, None, :] * offs[None, None, :, None]
    ab = b - a                                                # (T,N,L,2)
    p = pos[:, :, None, :]                                    # (T,N,1,2)
    t = np.clip(np.sum((p - a) * ab, axis=-1)
                / (np.sum(ab * ab, axis=-1) + 1e-6), 0.0, 1.0)  # (T,N,L)
    proj = a + t[..., None] * ab
    d = np.linalg.norm(p - proj, axis=-1)                     # (T,N,L)
    d = np.where(valid, d, 1e9)
    return d.min(axis=-1)                                     # (T,N)


def wrong_way(heading: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray,
              speed: np.ndarray, spawn_grace: np.ndarray,
              wrongway_cos: float = WRONGWAY_COS,
              idle_speed: float = IDLE_SPEED) -> np.ndarray:
    """(T, N) bool: heading points against the route direction while moving and not
    spawn-immune (mirrors env/legality.py)."""
    seg = seg_end - seg_start
    u = seg / (np.linalg.norm(seg, axis=-1, keepdims=True) + 1e-6)
    route_head = np.arctan2(u[..., 1], u[..., 0])
    herr = _wrap(heading - route_head)
    return (np.cos(herr) < wrongway_cos) & (speed > idle_speed) & (spawn_grace == 0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/rl/test_geometry.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add smoothride/rl/verifier.py tests/rl/test_geometry.py
git commit -m "feat: pure-numpy lane geometry helpers (off-lane distance, wrong-way)"
```

---

### Task 3: Verdicts and `verify()` — assemble the four constraints

Add the verdict dataclasses and `verify()`, combining the geometry helpers (off-lane, wrong-way) with the trace-native rules (collision via logged event, speed cross-check, arrival latching).

**Files:**
- Modify: `smoothride/rl/verifier.py`
- Create: `tests/rl/test_verifier.py`

**Interfaces:**
- Consumes: `lateral_offset`, `wrong_way`, constants from Task 2; `Trace` from Task 1.
- Produces:
  - `CarVerdict(arrived: bool, travel_time: float | None, collided: bool, off_lane: bool, wrong_way: bool, over_speed: bool, max_lateral_offset: float, valid: bool)`.
  - `RunVerdict(valid_run: bool, throughput: int, mean_travel_time: float, crash_count: int, off_lane_count: int, wrong_way_count: int, speed_violation_count: int, per_car: list[CarVerdict])`.
  - `verify(trace: Trace) -> RunVerdict`.

- [ ] **Step 1: Write the failing tests in `tests/rl/test_verifier.py`**

```python
"""Deterministic verifier (handoff §8) — reward/validity source of truth.

Pure function over a Trace: geometric/arithmetic predicates over logged arrays.
No randomness, no wall-clock, no network, no physics replay, no env import.
"""
import numpy as np

from smoothride.rl.verifier import CarVerdict, RunVerdict, verify

LW = 3.5


def test_clean_run_is_valid(make_trace):
    v = verify(make_trace(n_steps=3, n_agents=2))
    assert v.valid_run is True
    assert v.crash_count == 0
    assert v.off_lane_count == 0
    assert v.wrong_way_count == 0
    assert v.speed_violation_count == 0
    assert v.throughput == 0
    assert all(c.valid for c in v.per_car)


def test_off_lane_trips_when_far_from_lane(make_trace):
    pos = np.zeros((2, 1, 2), np.float32)
    pos[..., 1] = 10.0                          # ~11.75 m from lane-0 center
    v = verify(make_trace(n_steps=2, n_agents=1, pos=pos))
    assert v.per_car[0].off_lane is True
    assert v.per_car[0].valid is False
    assert v.off_lane_count == 1


def test_legal_position_on_outer_lane_not_flagged(make_trace):
    # 3-lane road, car on lane-2 centerline (y = -LW*2.5). Nearest-lane → ~0.
    pos = np.zeros((2, 1, 2), np.float32)
    pos[..., 1] = -LW * 2.5
    lane_count = np.full((2, 1), 3, np.int32)
    v = verify(make_trace(n_steps=2, n_agents=1, pos=pos, lane_count=lane_count))
    assert v.per_car[0].off_lane is False
    assert v.per_car[0].valid is True


def test_off_lane_exempt_during_spawn_grace(make_trace):
    pos = np.zeros((2, 1, 2), np.float32)
    pos[..., 1] = 10.0                          # off-road...
    grace = np.ones((2, 1), np.int32)           # ...but spawn-immune
    v = verify(make_trace(n_steps=2, n_agents=1, pos=pos, spawn_grace=grace))
    assert v.per_car[0].off_lane is False


def test_wrong_way_trips_while_moving(make_trace):
    heading = np.full((2, 1), np.pi, np.float32)
    speed = np.full((2, 1), 5.0, np.float32)
    v = verify(make_trace(n_steps=2, n_agents=1, heading=heading, speed=speed))
    assert v.per_car[0].wrong_way is True
    assert v.per_car[0].valid is False
    assert v.wrong_way_count == 1


def test_wrong_way_not_flagged_when_stationary(make_trace):
    heading = np.full((2, 1), np.pi, np.float32)   # reversed but speed default 0
    v = verify(make_trace(n_steps=2, n_agents=1, heading=heading))
    assert v.per_car[0].wrong_way is False


def test_over_speed_trips_even_though_env_clips(make_trace):
    speed = np.array([[10.0], [25.0]], np.float32)
    limit = np.array([[20.0], [20.0]], np.float32)
    v = verify(make_trace(n_steps=2, n_agents=1, speed=speed, speed_limit=limit))
    assert v.per_car[0].over_speed is True
    assert v.per_car[0].valid is False
    assert v.speed_violation_count == 1


def test_collision_from_logged_event_invalidates_offending_car_only(make_trace):
    crashed = np.zeros((3, 2), bool)
    crashed[1, 0] = True
    v = verify(make_trace(n_steps=3, n_agents=2, crashed=crashed))
    assert v.per_car[0].collided is True
    assert v.per_car[0].valid is False
    assert v.per_car[1].valid is True
    assert v.valid_run is False
    assert v.crash_count == 1


def test_arrival_latches_and_throughput_counts_cars(make_trace):
    arrived = np.zeros((5, 2), bool)
    arrived[2:, 0] = True                        # car 0 arrives at step 2, latches
    v = verify(make_trace(n_steps=5, n_agents=2, dt=0.5, arrived=arrived))
    assert v.per_car[0].arrived is True
    assert v.per_car[0].travel_time == 1.0       # 2 steps * 0.5 s
    assert v.per_car[1].arrived is False
    assert v.throughput == 1                      # one car arrived, not 3 latched cells


def test_never_arrived_has_none_travel_time(make_trace):
    v = verify(make_trace(n_steps=3, n_agents=1))
    assert v.per_car[0].travel_time is None
    assert v.mean_travel_time == 0.0             # no arrivals -> 0.0, not NaN


def test_mean_travel_time_over_arrived_cars(make_trace):
    arrived = np.zeros((4, 2), bool)
    arrived[1:, 0] = True                        # arrives step 1 -> 1.0 s
    arrived[3:, 1] = True                        # arrives step 3 -> 3.0 s
    v = verify(make_trace(n_steps=4, n_agents=2, dt=1.0, arrived=arrived))
    assert v.mean_travel_time == 2.0


def test_max_lateral_offset_reported(make_trace):
    assert verify(make_trace(n_steps=2, n_agents=1)).per_car[0].max_lateral_offset == 0.0
    pos = np.zeros((2, 1, 2), np.float32)
    pos[..., 1] = -LW * 0.5 + 3.0                # 3 m off the lane-0 centerline
    off = verify(make_trace(n_steps=2, n_agents=1, pos=pos)).per_car[0]
    assert abs(off.max_lateral_offset - 3.0) < 1e-4


def test_verify_is_deterministic(make_trace):
    crashed = np.zeros((3, 2), bool)
    crashed[2, 1] = True
    tr = make_trace(n_steps=3, n_agents=2, crashed=crashed)
    assert verify(tr) == verify(tr)


def test_returns_run_and_car_verdict_types(make_trace):
    v = verify(make_trace(n_steps=2, n_agents=2))
    assert isinstance(v, RunVerdict)
    assert len(v.per_car) == 2
    assert all(isinstance(c, CarVerdict) for c in v.per_car)


def test_verifier_module_does_not_import_env():
    import smoothride.rl.trace as trace_mod
    import smoothride.rl.verifier as verifier_mod
    for mod in (trace_mod, verifier_mod):
        src = open(mod.__file__).read()
        assert "smoothride.env" not in src
        assert "import jax" not in src
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/rl/test_verifier.py -q`
Expected: FAIL — `ImportError: cannot import name 'CarVerdict'`.

- [ ] **Step 3: Append the verdicts and `verify()` to `smoothride/rl/verifier.py`**

```python
@dataclass(frozen=True)
class CarVerdict:
    arrived: bool
    travel_time: float | None     # seconds, None if never arrived
    collided: bool
    off_lane: bool
    wrong_way: bool
    over_speed: bool
    max_lateral_offset: float     # meters; eval metric + hinged-cost basis (§ decision ③)
    valid: bool                   # no collision/off-lane/wrong-way/over-speed any step


@dataclass(frozen=True)
class RunVerdict:
    valid_run: bool               # all cars valid (the eval headline)
    throughput: int               # distinct cars that arrived
    mean_travel_time: float       # mean first-arrival time over arrived cars
    crash_count: int              # cars that collided
    off_lane_count: int           # cars that left their lane at any step
    wrong_way_count: int          # cars that drove against the route at any step
    speed_violation_count: int    # cars that exceeded the speed limit at any step
    per_car: list[CarVerdict]


def _arrival(trace: Trace, i: int) -> tuple[bool, float | None]:
    """(arrived?, first-arrival travel time in seconds) for car i. `arrived` latches
    under remove-on-arrival (§0②), so the first set step is the arrival step."""
    steps = np.flatnonzero(trace.arrived[:, i])
    if steps.size == 0:
        return False, None
    return True, float(steps[0] * trace.manifest.dt)


def verify(trace: Trace) -> RunVerdict:
    """Reduce a recorded `Trace` to per-car and run-level verdicts (handoff §8)."""
    lateral = lateral_offset(trace.pos, trace.seg_start, trace.seg_end,
                             trace.lane_count, trace.lane_width)       # (T,N)
    ww = wrong_way(trace.heading, trace.seg_start, trace.seg_end,
                   trace.speed, trace.spawn_grace)                      # (T,N) bool
    over = trace.speed > trace.speed_limit + SPEED_EPS                  # (T,N) bool
    off_lane_steps = (lateral > OFFLANE_THRESH) & (trace.spawn_grace == 0)

    per_car: list[CarVerdict] = []
    for i in range(trace.n_agents):
        collided = bool(trace.crashed[:, i].any())
        off_lane = bool(off_lane_steps[:, i].any())
        wrong = bool(ww[:, i].any())
        over_speed = bool(over[:, i].any())
        arrived, travel_time = _arrival(trace, i)
        valid = not (collided or off_lane or wrong or over_speed)
        per_car.append(CarVerdict(
            arrived=arrived, travel_time=travel_time, collided=collided,
            off_lane=off_lane, wrong_way=wrong, over_speed=over_speed,
            max_lateral_offset=float(lateral[:, i].max()), valid=valid))

    arrived_times = [c.travel_time for c in per_car if c.travel_time is not None]
    return RunVerdict(
        valid_run=all(c.valid for c in per_car),
        throughput=sum(1 for c in per_car if c.arrived),
        mean_travel_time=float(np.mean(arrived_times)) if arrived_times else 0.0,
        crash_count=sum(1 for c in per_car if c.collided),
        off_lane_count=sum(1 for c in per_car if c.off_lane),
        wrong_way_count=sum(1 for c in per_car if c.wrong_way),
        speed_violation_count=sum(1 for c in per_car if c.over_speed),
        per_car=per_car,
    )
```

- [ ] **Step 4: Run the full suite to verify it passes**

Run: `python3 -m pytest -q`
Expected: all pass (6 trace + 7 geometry + 15 verifier = 28).

- [ ] **Step 5: Lint and commit**

```bash
python3 -m ruff check smoothride/rl/trace.py smoothride/rl/verifier.py
git add smoothride/rl/verifier.py tests/rl/test_verifier.py
git commit -m "feat: rules-native verify() — lane/wrong-way/speed/collision verdicts"
```

---

### Task 4: Verifier drives training (per-rollout relabeling + §9 reward strip)

Make the verifier produce the PPO cost (reward-model pattern), and strip the env reward to
efficiency-only so all constraint signal flows through the verifier.

**Files:**
- Modify: `smoothride/rl/verifier.py` (add `_lane_flags`, `step_cost`, `cost_signal`; refactor
  `verify()` to share `_lane_flags`)
- Modify: `smoothride/rl/ppo.py` (`collect` emits raw State fields; add `verifier_cost(env, batch)`)
- Modify: `smoothride/env/kinematic.py` (add `w_time`; reward = `w_progress·progress +
  w_goal·new_goal − w_time`; drop the prox/ped_prox/yield/idle/collision reward terms)
- Modify: `tests/rl/test_verifier.py` (cost_signal tests)
- Create: `scripts/smoke_train_verifier.py`

**Interfaces:**
- `step_cost(pos, seg_start, seg_end, lane_count, lane_width, heading, speed, spawn_grace, crashed, speed_limit=None) -> (… ,N) float32` — per-step cost = crash + off_lane + wrong_way (+ over_speed); works on `(T,N)` or `(B*T,N)`.
- `cost_signal(trace: Trace) -> (T,N)` — `step_cost` over a Trace.
- `verifier_cost(env, batch) -> (B,T,N)` — host adapter; gathers geometry for each `(route_idx, wp_ptr)`, calls `step_cost`.

- [ ] **Step 1: Add `_lane_flags`/`step_cost`/`cost_signal`; refactor `verify()` to use `_lane_flags`.** Run `pytest tests/rl/ -q` (30 pass — refactor is behavior-preserving; 2 new cost_signal tests).
- [ ] **Step 2: `collect` out-dict gains `pos/heading/speed/route_idx/wp_ptr/spawn_grace/crashed`; add `verifier_cost`.** (`update()` reads only its own keys, so extra keys are inert.)
- [ ] **Step 3: §9 reward strip in `kinematic.step`** — add `w_time` field; reward = efficiency only; remove the dead penalty computations (`prox_pen`, `ped_prox`, the yield block, `idle_pen`).
- [ ] **Step 4: `scripts/smoke_train_verifier.py`** — per iter: `collect → verifier_cost → update`, dual-ascend `lam`. Run it: verifier cost ≫ crash-only, `lam` rises, loss finite.
- [ ] **Step 5: Commit.**

---

## Self-Review

**Spec coverage:**
- Four constraints → off_lane/wrong_way (Task 2 helpers + Task 3 wiring), arrival (Task 3 `_arrival`), collision (Task 3 logged `crashed`), speed (Task 3 cross-check). ✅
- Decision ① wrong-way → Task 2 `wrong_way` + Task 3. ✅
- Decision ② collision via logged event → Task 3 uses `trace.crashed`, no pos re-derivation. ✅
- Decision ③ offset → metric + cost, not reward → `max_lateral_offset` reported; verifier does not touch reward. ✅
- Trace schema v2 (add seg/lane geometry, drop verdict fields) → Task 1. ✅
- Verifier output shape → Task 3 dataclasses match spec. ✅
- Geometry math (nearest-lane point-to-segment, wrong-way) → Task 2. ✅
- Purity (no env import) → `test_verifier_module_does_not_import_env`. ✅
- Testing matrix (off-lane near/far/grace, wrong-way moving/stationary/grace, speed, collision, arrival/throughput, metric, determinism) → Tasks 2–3. ✅

**Placeholder scan:** none — every step has full code/commands.

**Type consistency:** `lateral_offset`/`wrong_way` signatures defined in Task 2 are called with matching args in Task 3 `verify()`. `CarVerdict`/`RunVerdict` field names used in tests match the dataclass definitions. `make_trace` overrides use real field names from Task 1.
