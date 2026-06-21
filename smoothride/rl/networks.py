"""Shared-parameter actor + centralized critic (CTDE).

All agents share one policy (homogeneous cars). The critic is centralized: it
sees each agent's local obs plus a pooled summary of the whole scene, which is
what lets agents learn to anticipate/avoid each other (the multi-agent point).

Encoder selection (v2 Task 6):
  - ``encoder="deepsets"`` (default): masked mean+max pool (DeepSets).
    Output per set: ``(..., 2*set_hidden)``.
  - ``encoder="attention"``: ego-query masked attention (AttentionPool).
    Output per set: ``(..., set_hidden)``.
The trunk Dense layers accept whichever concatenated size arrives; Flax's
``@nn.compact`` fixes the shapes on first forward call.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn


class ActorCritic(nn.Module):
    """Shared-parameter actor + centralized critic.

    Args:
        act_dim: Dimension of the continuous action space.
        hidden: Width of the MLP trunk layers.
        set_hidden: Hidden size for the set encoder (DeepSets or AttentionPool).
        encoder: Which set encoder to use — ``"deepsets"`` (default) or
            ``"attention"``.  The DeepSets encoder returns ``2*set_hidden``
            features (mean + max); AttentionPool returns ``set_hidden``.
            Both are permutation-invariant.

    Call signature (unchanged):
        Returns ``(mean, log_std, value)`` — do NOT modify this signature.
    """

    act_dim: int
    hidden: int = 128
    set_hidden: int = 64
    encoder: str = "deepsets"

    @nn.compact
    def __call__(self, obs, global_feat):
        # --- structured obs -> per-agent local feature vector ---
        # Encode the masked car/ped sets with the chosen permutation-invariant
        # encoder, then concat with the ego vector to form the local feature.
        if self.encoder == "attention":
            car_enc = AttentionPool(hidden=self.set_hidden)(
                obs["cars"], obs["cars_mask"])           # (..., set_hidden)
            ped_enc = AttentionPool(hidden=self.set_hidden)(
                obs["peds"], obs["peds_mask"])           # (..., set_hidden)
        elif self.encoder == "deepsets":
            car_enc = DeepSets(feat_dim=4, hidden=self.set_hidden)(
                obs["cars"], obs["cars_mask"])           # (..., 2*set_hidden)
            ped_enc = DeepSets(feat_dim=5, hidden=self.set_hidden)(
                obs["peds"], obs["peds_mask"])           # (..., 2*set_hidden)
        else:
            raise ValueError(
                f"Unknown encoder {self.encoder!r}. Choose 'deepsets' or 'attention'."
            )
        feat = jnp.concatenate([obs["ego"], car_enc, ped_enc], axis=-1)

        # --- actor: decentralized, local feature only ---
        x = nn.tanh(nn.Dense(self.hidden)(feat))
        x = nn.tanh(nn.Dense(self.hidden)(x))
        mean = nn.Dense(self.act_dim,
                        kernel_init=nn.initializers.orthogonal(0.01))(x)
        # State-DEPENDENT log_std: a head off the same actor trunk, so exploration
        # noise adapts to context (cruising an empty road vs. threading between a
        # pedestrian and a car). Clamped to a safe range to keep std numerically
        # bounded — exp(-5)≈0.007 (near-deterministic) .. exp(2)≈7.4 (very broad).
        log_std = nn.Dense(self.act_dim,
                           kernel_init=nn.initializers.orthogonal(0.01),
                           bias_init=nn.initializers.constant(-0.5))(x)
        log_std = jnp.clip(log_std, -5.0, 2.0)

        # --- critic: centralized, local feature + pooled scene summary ---
        c = jnp.concatenate([feat, global_feat], axis=-1)
        c = nn.tanh(nn.Dense(self.hidden)(c))
        c = nn.tanh(nn.Dense(self.hidden)(c))
        value = nn.Dense(1)(c)[..., 0]
        return mean, log_std, value


class DeepSets(nn.Module):
    """Permutation-invariant set encoder: per-element MLP phi, then masked
    mean+max pool. Empty set -> zeros. Density-agnostic, handles padded slots."""

    feat_dim: int
    hidden: int = 64

    @nn.compact
    def __call__(self, entities: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        """Encode a set of entities with validity mask.

        Args:
            entities: (..., C, feat_dim) — one row per slot.
            mask: (..., C) bool — True for valid slots, False for padding.

        Returns:
            (..., 2*hidden) — concat of masked mean-pool and max-pool.
        """
        h = nn.relu(nn.Dense(self.hidden)(entities))
        h = nn.relu(nn.Dense(self.hidden)(h))          # (..., C, hidden)
        m = mask[..., None].astype(h.dtype)            # (..., C, 1) float
        h = h * m                                      # zero invalid slots
        summed = jnp.sum(h, axis=-2)                   # (..., hidden)
        count = jnp.clip(jnp.sum(m, axis=-2), 1.0)    # live count >= 1
        mean = summed / count
        # masked max: push invalid slots to large-negative so they never win
        neg = jnp.where(m > 0, h, -1e9)
        mx = jnp.max(neg, axis=-2)                     # (..., hidden)
        any_valid = jnp.sum(m, axis=-2) > 0            # (..., 1) bool
        mx = jnp.where(any_valid, mx, 0.0)             # guard empty set
        return jnp.concatenate([mean, mx], axis=-1)    # (..., 2*hidden)


class AttentionPool(nn.Module):
    """Masked ego-query attention set encoder (social-attention / PMA-style).

    Projects each entity to key/value space, then attends via a single learned
    query vector.  Permutation-invariant by construction.

    NaN-safety: when ALL slots are masked (empty set), the softmax over
    all-``-1e9`` logits produces a finite uniform distribution instead of NaN,
    but we additionally zero the entire output row where ``mask.sum(-1) == 0``
    to guarantee a clean zero for empty sets.

    Output: ``(..., hidden)`` — NOT ``2*hidden`` like DeepSets.

    Args:
        hidden: Width of the key/value projection and the output dimension.
        num_heads: Stored for API parity; the implementation uses a single
            effective head (the head axis is folded into ``hidden``).
    """

    hidden: int = 64
    num_heads: int = 4

    @nn.compact
    def __call__(self, entities: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        """Encode a padded set of entities via masked attention.

        Args:
            entities: ``(..., C, feat_dim)`` — one row per slot.
            mask: ``(..., C)`` bool — True for valid slots.

        Returns:
            ``(..., hidden)``
        """
        # Project entities to key and value spaces.
        keys = nn.Dense(self.hidden)(entities)      # (..., C, hidden)
        vals = nn.Dense(self.hidden)(entities)      # (..., C, hidden)

        # Learned query vector: shape (hidden,).  Use normal(stddev=0.1) —
        # lecun_normal requires >=2D tensors and would raise for a 1D param.
        query = self.param(
            "query",
            nn.initializers.normal(stddev=0.1),
            (self.hidden,),
        )

        # Scaled dot-product attention: query (hidden,) vs keys (..., C, hidden).
        scale = self.hidden ** 0.5
        logits = jnp.sum(query * keys, axis=-1) / scale   # (..., C)

        # Mask: set padding logits to -1e9 (not -inf to avoid NaN in softmax).
        masked_logits = jnp.where(mask, logits, -1e9)      # (..., C)
        weights = jax.nn.softmax(masked_logits, axis=-1)    # (..., C)

        # Weighted sum over value vectors.
        out = jnp.sum(weights[..., None] * vals, axis=-2)  # (..., hidden)

        # Guard empty sets: zero the output where no slot is valid.
        any_valid = jnp.sum(mask.astype(out.dtype), axis=-1, keepdims=True) > 0
        return jnp.where(any_valid, out, 0.0)


def gaussian_logp(actions, mean, log_std):
    std = jnp.exp(log_std)
    pre = -0.5 * (((actions - mean) / std) ** 2) - log_std - 0.5 * jnp.log(2 * jnp.pi)
    return pre.sum(-1)


def gaussian_entropy(log_std):
    """Closed-form differential entropy of a diagonal Gaussian, summed over dims.

    Used as the exploration bonus for the tanh-squashed policy as well: we keep
    the entropy of the PRE-squash Gaussian rather than the (intractable in closed
    form) entropy of the squashed distribution. This is a standard, accepted
    approximation (the tanh Jacobian term shifts entropy but the pre-squash
    entropy is a smooth, well-behaved proxy that still rewards wider std).
    """
    return (log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e)).sum(-1)


# Numerical-stability constants for the tanh change-of-variables.
_ATANH_EPS = 1e-6   # clamp |a| away from 1 before atanh (atanh(±1) = ±inf)
_LOGDET_EPS = 1e-6  # floor on (1 - a^2) inside the log Jacobian term


def squashed_gaussian_logp(action, mean, log_std):
    """Log-prob of a tanh-squashed diagonal Gaussian at a SQUASHED action.

    The policy samples ``raw ~ Normal(mean, exp(log_std))`` and emits
    ``a = tanh(raw)`` as the env action, so actions live in (-1, 1) by
    construction. The change-of-variables (Jacobian) correction is::

        log pi(a) = sum_d [ normal_logp(raw_d; mean_d, std_d)
                            - log(1 - tanh(raw_d)^2 + eps) ]

    Given a stored squashed action ``a`` we recover the pre-squash sample with
    ``raw = atanh(clip(a, -1+eps, 1-eps))`` (numerically stable at the boundary)
    and use ``1 - a^2`` directly for the log-det term.

    Args:
        action: Squashed actions in (-1, 1), shape ``(..., act_dim)``.
        mean: Pre-squash Gaussian mean, shape ``(..., act_dim)``.
        log_std: Pre-squash Gaussian log-std, shape ``(..., act_dim)``.

    Returns:
        Log-prob per sample, shape ``(...,)`` (summed over the action dim).
    """
    a = jnp.clip(action, -1.0 + _ATANH_EPS, 1.0 - _ATANH_EPS)
    raw = jnp.arctanh(a)
    base = gaussian_logp(raw, mean, log_std)
    log_det = jnp.log(1.0 - a ** 2 + _LOGDET_EPS).sum(-1)
    return base - log_det


def squash_sample(mean, log_std, noise):
    """Reparameterised sample of the tanh-squashed policy + its log-prob.

    Args:
        mean: Pre-squash Gaussian mean, shape ``(..., act_dim)``.
        log_std: Pre-squash Gaussian log-std, shape ``(..., act_dim)``.
        noise: Standard-normal noise, same shape as ``mean``.

    Returns:
        ``(action, logp)`` where ``action = tanh(mean + std*noise)`` lies strictly
        in (-1, 1) and ``logp`` is the squashed log-prob of that action.
    """
    raw = mean + jnp.exp(log_std) * noise
    action = jnp.tanh(raw)
    # Use the raw sample directly (no atanh round-trip) for sampling-time logp:
    # this is the exact pre-squash value, so the Jacobian uses tanh(raw)^2.
    base = gaussian_logp(raw, mean, log_std)
    log_det = jnp.log(1.0 - jnp.tanh(raw) ** 2 + _LOGDET_EPS).sum(-1)
    logp = base - log_det
    return action, logp
