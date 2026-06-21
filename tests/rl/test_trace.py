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
