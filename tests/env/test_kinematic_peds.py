"""Deterministic pedestrian motion, cruise cap, and layered collision radii tests."""
import jax
import jax.numpy as jnp
import numpy as np

from smoothride.env import kinematic as K
from smoothride.env.routing import RoutePool


def _pool():
    # 2 straight routes, 3 waypoints, 2 lanes, speed 10 m/s
    xy = np.array([[[0, 0], [50, 0], [100, 0]],
                   [[0, 20], [50, 20], [100, 20]]], np.float32)
    n = np.array([3, 3], np.int32)
    node = np.zeros((2, 3), np.int32)
    junc = np.zeros((2, 3), bool)
    lanes = np.full((2, 3), 2, np.int32)
    speed = np.full((2, 3), 10.0, np.float32)
    return RoutePool(xy=xy, n=n, node=node, junc=junc, lanes=lanes, speed=speed)


def _env(**kw):
    return K.make_env(_pool(), world_min=[-10, -10], world_max=[110, 40],
                      n_agents=4, n_peds=6, seed=0, **kw)


def test_peds_are_deterministic_across_resets():
    env = _env()
    s1, _ = K.reset(env, jax.random.PRNGKey(0))
    s2, _ = K.reset(env, jax.random.PRNGKey(999))   # different key
    # peds do not depend on the reset key (paths are prebuilt, motion is f(t))
    np.testing.assert_allclose(np.asarray(s1.ped_pos), np.asarray(s2.ped_pos), atol=1e-5)


def test_ped_waits_before_start_then_moves():
    env = _env()
    st, _ = K.reset(env, jax.random.PRNGKey(0))
    # a ped with start > 0 sits at path[0] until its start step
    late = int(np.argmax(np.asarray(env.ped_starts) > 0))
    assert env.ped_starts[late] > 0, "fixture must have at least one delayed-start ped"
    p0 = np.asarray(env.ped_paths[late, 0])
    np.testing.assert_allclose(np.asarray(st.ped_pos[late]), p0, atol=1e-4)
    # step until just past its start, then it should have moved
    act = jnp.zeros((env.n_agents, env.act_dim))
    s = st
    for _ in range(int(env.ped_starts[late]) + 5):
        s, *_ = K.step(env, s, act, jax.random.PRNGKey(0))
    assert np.linalg.norm(np.asarray(s.ped_pos[late]) - p0) > 0.1


def test_cruise_cap_clamps_speed():
    env = _env(cruise_cap=4.0)
    st, _ = K.reset(env, jax.random.PRNGKey(0))
    act = jnp.zeros((env.n_agents, env.act_dim)).at[:, 0].set(1.0)  # full throttle
    s = st
    for _ in range(30):
        s, *_ = K.step(env, s, act, jax.random.PRNGKey(0))
    assert float(jnp.max(s.speed)) <= 4.0 + 1e-4


def test_ped_collision_uses_raised_radius():
    env = _env(ped_radius=3.5)
    assert env.ped_radius == 3.5
    assert env.collision_radius < env.ped_radius   # asymmetric: wider berth for people
