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
    inp = np.array([0.0, 3 * np.pi, -3 * np.pi, np.pi, 0.5])
    out = _wrap(inp)
    assert np.all(out > -np.pi - 1e-9) and np.all(out <= np.pi + 1e-9)
    # ±π are the same angle; verify equivalence via trig (convention-agnostic).
    assert np.allclose(np.cos(out), np.cos(inp), atol=1e-6)
    assert np.allclose(np.sin(out), np.sin(inp), atol=1e-6)


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
