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
    assert verify(make_trace(n_steps=2, n_agents=1)).per_car[0].max_lateral_offset < 1e-4
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
