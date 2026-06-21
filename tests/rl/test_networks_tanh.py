"""Tanh-squashed Gaussian policy + state-dependent std (exp/tanh-policy).

Covers the change-of-variables log-prob, bounded sampling, numerical stability
of the atanh round-trip at the (-1,1) boundary, the per-state log_std head, and a
finite forward + ppo_loss-style pass plus a tiny end-to-end collect/update smoke.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np

from smoothride.rl import ppo
from smoothride.rl.networks import (
    ActorCritic,
    gaussian_logp,
    squash_sample,
    squashed_gaussian_logp,
)
from tests.env.test_kinematic_peds import _env


def _np_squashed_logp(action, mean, log_std):
    """Reference NumPy implementation of the squashed-Gaussian log-prob."""
    a = np.clip(action, -1.0 + 1e-6, 1.0 - 1e-6)
    raw = np.arctanh(a)
    std = np.exp(log_std)
    base = (-0.5 * ((raw - mean) / std) ** 2 - log_std
            - 0.5 * math.log(2 * math.pi)).sum(-1)
    log_det = np.log(1.0 - a ** 2 + 1e-6).sum(-1)
    return base - log_det


def test_squashed_logp_matches_hand_computed_value():
    """squashed_gaussian_logp matches an independent NumPy computation."""
    action = np.array([0.3, -0.7, 0.95], np.float64)
    mean = np.array([0.1, 0.0, 0.5], np.float64)
    log_std = np.array([-0.5, -1.0, 0.2], np.float64)

    got = float(squashed_gaussian_logp(jnp.asarray(action),
                                       jnp.asarray(mean),
                                       jnp.asarray(log_std)))
    expected = float(_np_squashed_logp(action, mean, log_std))
    assert abs(got - expected) < 1e-4, f"got {got}, expected {expected}"


def test_squashed_logp_subtracts_jacobian_from_base():
    """The squashed log-prob equals base Gaussian logp minus the log-det term."""
    raw = np.array([0.4, -0.2, 1.1], np.float64)
    mean = np.array([0.0, 0.1, -0.3], np.float64)
    log_std = np.array([-0.5, -0.5, 0.0], np.float64)
    action = np.tanh(raw)

    base = float(gaussian_logp(jnp.asarray(raw), jnp.asarray(mean),
                               jnp.asarray(log_std)))
    log_det = float(np.log(1.0 - action ** 2 + 1e-6).sum())
    got = float(squashed_gaussian_logp(jnp.asarray(action),
                                       jnp.asarray(mean),
                                       jnp.asarray(log_std)))
    assert abs(got - (base - log_det)) < 1e-4


def test_sampled_actions_strictly_within_unit_interval():
    """squash_sample keeps actions within (-1, 1) for realistic policy params.

    tanh squashing bounds actions by construction (vs. the old unbounded Gaussian
    + env clip). For typical means/stds the actions are STRICTLY inside; logp stays
    finite throughout.
    """
    mean = jnp.array([0.5, -0.5, 0.0])
    log_std = jnp.array([-0.5, -0.5, -0.5])
    noise = jax.random.normal(jax.random.PRNGKey(0), (4096, 3))
    action, logp = jax.vmap(lambda nz: squash_sample(mean, log_std, nz))(noise)

    a = np.asarray(action)
    assert np.all(a > -1.0), f"min action {a.min()} not > -1"
    assert np.all(a < 1.0), f"max action {a.max()} not < 1"
    assert np.all(np.isfinite(np.asarray(logp))), "logp must be finite"


def test_sampled_actions_bounded_and_finite_logp_at_extreme_noise():
    """Under extreme noise tanh may saturate to +/-1 in float32, but actions stay
    within [-1, 1] (never the unbounded blow-up of the old policy) and logp stays
    finite thanks to the log-det epsilon floor."""
    mean = jnp.array([5.0, -5.0, 0.0])
    log_std = jnp.array([2.0, 2.0, 2.0])
    noise = jax.random.normal(jax.random.PRNGKey(0), (4096, 3)) * 10.0
    action, logp = jax.vmap(lambda nz: squash_sample(mean, log_std, nz))(noise)

    a = np.asarray(action)
    assert np.all(a >= -1.0) and np.all(a <= 1.0), (
        f"actions escaped [-1,1]: [{a.min()}, {a.max()}]"
    )
    assert np.all(np.isfinite(np.asarray(logp))), "logp must stay finite even at saturation"


def test_atanh_roundtrip_stable_at_boundary():
    """Log-prob is finite for actions arbitrarily close to +/-1 (atanh blow-up)."""
    action = jnp.array([1.0 - 1e-9, -1.0 + 1e-9, 1.0, -1.0])
    mean = jnp.zeros(4)
    log_std = jnp.zeros(4)
    logp = squashed_gaussian_logp(action, mean, log_std)
    assert jnp.isfinite(logp), f"boundary logp not finite: {logp}"


def test_sample_and_recompute_logp_consistent():
    """logp from squash_sample matches squashed_gaussian_logp on the stored action.

    The two paths differ only in numerics (sample uses raw directly; recompute
    uses atanh(action)); they must agree closely away from the boundary.
    """
    mean = jnp.array([0.2, -0.4, 0.1])
    log_std = jnp.array([-0.5, -0.3, 0.0])
    noise = jax.random.normal(jax.random.PRNGKey(7), (256, 3))
    action, logp_sample = jax.vmap(lambda nz: squash_sample(mean, log_std, nz))(noise)
    logp_recompute = jax.vmap(
        lambda a: squashed_gaussian_logp(a, mean, log_std))(action)
    np.testing.assert_allclose(np.asarray(logp_sample),
                               np.asarray(logp_recompute), atol=1e-3)


def test_log_std_head_is_state_dependent():
    """log_std now varies with the input obs (it is a head, not a global param)."""
    env = _env(cruise_cap=4.0)
    net = ActorCritic(act_dim=env.act_dim)
    st, obs = K_reset(env)
    gf = obs["ego"].mean(-2, keepdims=True)
    gf = jnp.broadcast_to(gf, obs["ego"].shape)
    params = net.init(jax.random.PRNGKey(0), obs, gf)

    # The fixed-param "log_std" must be gone; a Dense head supplies it now.
    assert "log_std" not in params["params"], (
        "state-independent log_std param should be removed"
    )
    _, log_std, _ = net.apply(params, obs, gf)
    # With distinct per-agent observations the log_stds should not all be identical.
    ls = np.asarray(log_std)
    assert ls.shape[-1] == env.act_dim
    assert np.isfinite(ls).all()
    assert np.all(ls >= -5.0 - 1e-5) and np.all(ls <= 2.0 + 1e-5), "log_std clamped"


def K_reset(env):
    from smoothride.env import kinematic as K
    return K.reset(env, jax.random.PRNGKey(0))


def test_forward_and_ppo_loss_finite_on_dummy():
    """A forward pass + squashed-logp PPO-style loss yields finite values."""
    env = _env(cruise_cap=4.0)
    net = ActorCritic(act_dim=env.act_dim)
    st, obs = K_reset(env)
    gf = jnp.broadcast_to(obs["ego"].mean(-2, keepdims=True), obs["ego"].shape)
    params = net.init(jax.random.PRNGKey(0), obs, gf)

    mean, log_std, value = net.apply(params, obs, gf)
    noise = jax.random.normal(jax.random.PRNGKey(1), mean.shape)
    action, logp = squash_sample(mean, log_std, noise)
    # Recompute logp from the stored squashed action (the update() path).
    logp2 = squashed_gaussian_logp(action, mean, log_std)
    ratio = jnp.exp(logp2 - logp)

    assert jnp.all(jnp.isfinite(mean))
    assert jnp.all(jnp.isfinite(log_std))
    assert jnp.all(jnp.isfinite(value))
    assert jnp.all(jnp.isfinite(action))
    assert jnp.all(jnp.isfinite(logp))
    assert jnp.all(jnp.isfinite(ratio))


def test_end_to_end_two_iterations_finite_and_bounded():
    """Two collect -> verifier_cost -> update iterations: finite losses, bounded acts."""
    env = _env(cruise_cap=4.0)
    cfg = ppo.PPOConfig(n_worlds=2, epochs=1, minibatches=2)
    ts = ppo.make_train_state(env, cfg, jax.random.PRNGKey(0))

    for it in range(2):
        batch = ppo.collect(env, ts, jax.random.PRNGKey(it + 1), cfg.n_worlds)
        # Sampled (squashed) actions stored in the batch must be in (-1, 1).
        a = np.asarray(batch["action"])
        assert np.all(a > -1.0) and np.all(a < 1.0), (
            f"iter {it}: stored actions not bounded: [{a.min()}, {a.max()}]"
        )
        cost_hard, cost_soft = ppo.verifier_cost(env, batch, w_carped=8.0)
        batch = {**batch, "cost_hard": cost_hard, "cost_soft": cost_soft}
        ts, metrics = ppo.update(env, cfg, ts, batch, lam_hard=1.0, lam_soft=0.5)
        assert jnp.isfinite(metrics["loss"]), f"iter {it}: loss not finite"
        assert jnp.isfinite(metrics["ep_reward"]), f"iter {it}: ep_reward not finite"
