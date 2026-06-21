"""Tests for AttentionPool encoder and selectable encoder in ActorCritic.

TDD: these tests are written FIRST and should FAIL until AttentionPool is
implemented, ActorCritic gains the `encoder` field, PPOConfig gains `encoder`,
and make_train_state wires it through.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from smoothride.rl.networks import AttentionPool, ActorCritic
from smoothride.rl import ppo
from smoothride.env import kinematic as K
from tests.env.test_kinematic_peds import _env


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_attn(mod: AttentionPool, ents: jnp.ndarray,
                mask: jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    """Init and apply an AttentionPool module, returning (output, params)."""
    params = mod.init(jax.random.PRNGKey(0), ents, mask)
    return mod.apply(params, ents, mask), params


# ---------------------------------------------------------------------------
# AttentionPool tests
# ---------------------------------------------------------------------------

def test_attention_pool_output_shape() -> None:
    """Output must be (..., hidden) — NOT 2*hidden like DeepSets."""
    mod = AttentionPool(hidden=8, num_heads=4)
    ents = jnp.ones((3, 5, 4))         # batch 3, cap 5, feat_dim 4
    mask = jnp.ones((3, 5), bool)      # all valid
    out, _ = _apply_attn(mod, ents, mask)
    assert out.shape == (3, 8), f"expected (3, 8), got {out.shape}"


def test_attention_pool_empty_set_is_zero_and_finite() -> None:
    """NaN-safety: all-False mask (empty set) must produce zeros, not NaN/inf.

    This guards the masked-softmax edge case: when every logit is masked out
    with -1e9, softmax still sums to 1 but to a finite uniform; the guard
    `mask.sum(-1) == 0` zeroes the entire output row.
    """
    mod = AttentionPool(hidden=8, num_heads=4)
    ents = jnp.ones((3, 5, 4))
    mask = jnp.zeros((3, 5), bool)     # ALL slots masked — empty set
    out, _ = _apply_attn(mod, ents, mask)
    assert jnp.all(jnp.isfinite(out)), f"output contains NaN/inf for empty set: {out}"
    np.testing.assert_allclose(np.asarray(out), 0.0, atol=1e-6,
                               err_msg="empty-set output must be all zeros")


def test_attention_pool_permutation_invariance() -> None:
    """Permuting the entity sequence (+ mask consistently) must not change output.

    AttentionPool is a set encoder: output is permutation-invariant by design.
    """
    mod = AttentionPool(hidden=8, num_heads=4)
    ents = jax.random.normal(jax.random.PRNGKey(1), (1, 5, 4))
    mask = jnp.array([[True, True, True, False, False]])
    out_a, params = _apply_attn(mod, ents, mask)

    perm = jnp.array([2, 0, 1, 4, 3])          # shuffle valid + invalid slots
    out_b = mod.apply(params, ents[:, perm], mask[:, perm])
    np.testing.assert_allclose(np.asarray(out_a), np.asarray(out_b), atol=1e-5,
                               err_msg="AttentionPool output changed under permutation")


def test_attention_pool_masked_slots_ignored() -> None:
    """Garbage values in masked slots must not affect the output.

    Sets padding slots to 999.0 and verifies the output is unchanged.
    """
    mod = AttentionPool(hidden=8, num_heads=4)
    ents = jax.random.normal(jax.random.PRNGKey(2), (1, 5, 4))
    mask = jnp.array([[True, True, False, False, False]])
    out_a, params = _apply_attn(mod, ents, mask)

    ents_garbage = ents.at[:, 2:].set(999.0)    # corrupt the masked slots
    out_b = mod.apply(params, ents_garbage, mask)
    np.testing.assert_allclose(np.asarray(out_a), np.asarray(out_b), atol=1e-5,
                               err_msg="masked slots affected the output")


# ---------------------------------------------------------------------------
# ActorCritic with encoder selection
# ---------------------------------------------------------------------------

def test_actor_critic_attention_encoder_forward() -> None:
    """ActorCritic with encoder='attention' must return (mean, log_std, value)
    with correct shapes and all-finite values.
    """
    env = _env(cruise_cap=4.0)
    _, obs = env_reset_obs(env)
    net = ActorCritic(act_dim=env.act_dim, encoder="attention")
    gf = ppo._global_feat(obs)
    params = net.init(jax.random.PRNGKey(0), obs, gf)
    mean, log_std, value = net.apply(params, obs, gf)

    # shapes (state-dependent log_std head: per-agent, same shape as mean)
    assert mean.shape[-1] == env.act_dim, f"mean.shape[-1]={mean.shape[-1]} != {env.act_dim}"
    assert log_std.shape == mean.shape, f"log_std.shape={log_std.shape} != {mean.shape}"
    assert value.shape == mean.shape[:-1], f"value.shape={value.shape} != {mean.shape[:-1]}"

    # finiteness
    assert jnp.all(jnp.isfinite(mean)), "mean contains NaN/inf"
    assert jnp.all(jnp.isfinite(log_std)), "log_std contains NaN/inf"
    assert jnp.all(jnp.isfinite(value)), "value contains NaN/inf"


def test_actor_critic_deepsets_encoder_regression() -> None:
    """Regression guard: encoder='deepsets' (default) must still work correctly.

    Shapes and finiteness check identical to the attention variant.
    """
    env = _env(cruise_cap=4.0)
    _, obs = env_reset_obs(env)
    net = ActorCritic(act_dim=env.act_dim, encoder="deepsets")
    gf = ppo._global_feat(obs)
    params = net.init(jax.random.PRNGKey(0), obs, gf)
    mean, log_std, value = net.apply(params, obs, gf)

    assert mean.shape[-1] == env.act_dim
    assert log_std.shape == mean.shape
    assert value.shape == mean.shape[:-1]
    assert jnp.all(jnp.isfinite(mean))
    assert jnp.all(jnp.isfinite(log_std))
    assert jnp.all(jnp.isfinite(value))


# ---------------------------------------------------------------------------
# PPO smoke test with attention encoder
# ---------------------------------------------------------------------------

def test_ppo_smoke_attention_encoder() -> None:
    """Full collect → verifier_cost → update loop with encoder='attention'.

    Uses a minimal config (n_worlds=2, epochs=1, minibatches=2) to keep
    compile time short. Metrics must be finite.
    """
    env = _env(cruise_cap=4.0)
    cfg = ppo.PPOConfig(n_worlds=2, epochs=1, minibatches=2, encoder="attention")
    ts = ppo.make_train_state(env, cfg, jax.random.PRNGKey(0))

    batch = ppo.collect(env, ts, jax.random.PRNGKey(1), cfg.n_worlds)
    cost_hard, cost_soft = ppo.verifier_cost(env, batch, w_carped=3.0)
    batch = {**batch, "cost_hard": cost_hard, "cost_soft": cost_soft}

    ts2, metrics = ppo.update(env, cfg, ts, batch, lam_hard=1.0, lam_soft=0.5)

    assert jnp.isfinite(metrics["loss"]), f"loss not finite: {metrics['loss']}"
    assert jnp.isfinite(metrics["ep_reward"]), f"ep_reward not finite: {metrics['ep_reward']}"
    assert ts2 is not ts, "update must return a new TrainState"


# ---------------------------------------------------------------------------
# Shared helper (module-level, not a test)
# ---------------------------------------------------------------------------

def env_reset_obs(env: K.Env) -> dict:
    """Reset env and return (state, obs). Used in attention encoder tests."""
    return K.reset(env, jax.random.PRNGKey(0))
