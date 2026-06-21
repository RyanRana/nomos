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
    """Wrap radians to [−π, π)."""
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
