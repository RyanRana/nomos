# Pedestrian-Yield Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A signal-free, dense-pedestrian driving environment where cars learn to *slow* (not stop dead) for crossing pedestrians, via a continuous CMDP ped-yield cost and a permutation-invariant Deep Sets perception layer.

**Architecture:** Pedestrians follow deterministic hard-coded polylines (sidewalk run + one perpendicular crossing, staggered starts). The policy observes radius-capped, padded+masked *sets* of nearby cars and peds through Deep Sets encoders (replacing the flat fixed-K MLP). A continuous ped-yield hinge cost (proximity × speed, gated to crossing peds) is folded into the existing single-channel CMDP cost; reward stays §9 efficiency-only. Cars cruise at a low configurable speed cap and brake below it near crossers.

**Tech Stack:** JAX + Flax (`flax.struct`, `flax.linen`), Optax, NumPy, pytest, Modal (training).

## Global Constraints

- **Immutability:** never mutate JAX arrays or `@struct.dataclass` instances in place; always return new copies (`.replace(...)`, `jnp.where`, fresh arrays). Verifier module (`rl/verifier.py`) stays pure NumPy: no randomness, no wall-clock, no network, no LLM, never imports the env.
- **JAX shape discipline:** all per-step arrays are fixed-shape so `jax.jit`/`vmap` trace cleanly. Variable entity counts are handled by padding to a fixed cap + boolean mask, never ragged arrays. No `Date.now`-style host nondeterminism inside traced code.
- **Determinism:** pedestrian motion uses NO per-step RNG — position is a pure function of `t`, the prebuilt path, and `ped_starts`. Same seed → same paths.
- **Reward unchanged:** reward stays §9 efficiency-only (`w_progress·progress + w_goal·arrival − w_time`). ALL constraints flow through the verifier cost channel. Do not add anything to `reward` in `kinematic.step`.
- **One rulebook:** the ped-yield predicate lives in `verifier.step_cost` and is reused by both the training relabel (`ppo.verifier_cost`) and the offline grader (`verifier.cost_signal`). No divergent copies.
- **Python style:** PEP 8, type annotations on signatures, black/isort/ruff clean, functions < 50 lines, files < 800 lines.
- **Spec:** `docs/superpowers/specs/2026-06-20-pedestrian-yield-env-design.md` is the source of truth.

---

## File Structure

- `smoothride/env/kinematic.py` (modify) — host `build_ped_paths`, deterministic `_ped_step` via arc interpolation, ped velocity + crossing state, layered safety radii, cruise cap, `Movers` view, radius-capped candidate gather, structured observation.
- `smoothride/env/ped_paths.py` (create) — pure host-side path construction + arc-interpolation helpers (keeps `kinematic.py` from growing unwieldy; imported by it).
- `smoothride/rl/verifier.py` (modify) — `ped_yield_cost` predicate + integration into `step_cost`/`cost_signal`.
- `smoothride/rl/trace.py` (modify) — add ped fields (`ped_pos`, `ped_crossing`) to `Trace`.
- `smoothride/rl/networks.py` (modify) — Deep Sets front-end in `ActorCritic`.
- `smoothride/rl/ppo.py` (modify) — structured-obs plumbing, ped logging in `collect`, ped-yield in `verifier_cost`.
- `smoothride/rl/modal_train.py` (modify) — `--peds`, cruise-cap, ped-radius, candidate-cap flags.
- `tests/env/test_ped_paths.py` (create), `tests/env/test_kinematic_peds.py` (create), `tests/rl/test_verifier_pedyield.py` (create), `tests/rl/test_networks_deepsets.py` (create), `tests/rl/conftest.py` (modify) — tests.

**Test runner:** `python -m pytest <path> -v` from the worktree root. JAX runs CPU-only for tests (fast enough at small sizes).

---

## Part A — Pedestrian behavior & yield cost

### Task 1: Deterministic pedestrian paths (host build + arc interpolation)

**Files:**
- Create: `smoothride/env/ped_paths.py`
- Test: `tests/env/test_ped_paths.py`

**Interfaces:**
- Produces:
  - `build_ped_paths(routes_xy: np.ndarray, routes_n: np.ndarray, routes_lanes: np.ndarray, lane_width: float, n_peds: int, seed: int, *, sidewalk_offset: float = 1.5, run_len: float = 12.0, max_start: int = 60) -> PedPaths`
  - `PedPaths` = `@dataclass(frozen=True)` with fields `paths: np.ndarray (M,4,2)`, `cum: np.ndarray (M,4)`, `starts: np.ndarray (M,) int32`, `cross_lo: np.ndarray (M,) f32`, `cross_hi: np.ndarray (M,) f32` (arc-length bounds of the crossing leg).
  - `arc_interp(paths: jnp.ndarray, cum: jnp.ndarray, walked: jnp.ndarray) -> jnp.ndarray` — batched `(M,4,2),(M,4),(M,) -> (M,2)`; clamps `walked` to `[0, cum[...,-1]]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/env/test_ped_paths.py
import jax.numpy as jnp
import numpy as np
import pytest

from smoothride.env.ped_paths import PedPaths, arc_interp, build_ped_paths


@pytest.fixture
def simple_net():
    # one straight 2-lane route: 3 waypoints along +x at y=0
    routes_xy = np.array([[[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]]], np.float32)
    routes_n = np.array([3], np.int32)
    routes_lanes = np.array([[2, 2, 2]], np.int32)
    return routes_xy, routes_n, routes_lanes


def test_build_shapes_and_determinism(simple_net):
    xy, n, lanes = simple_net
    a = build_ped_paths(xy, n, lanes, lane_width=3.5, n_peds=5, seed=0)
    b = build_ped_paths(xy, n, lanes, lane_width=3.5, n_peds=5, seed=0)
    assert isinstance(a, PedPaths)
    assert a.paths.shape == (5, 4, 2)
    assert a.cum.shape == (5, 4)
    assert a.starts.shape == (5,)
    assert a.cross_lo.shape == (5,) and a.cross_hi.shape == (5,)
    # deterministic for a fixed seed
    np.testing.assert_array_equal(a.paths, b.paths)
    np.testing.assert_array_equal(a.starts, b.starts)
    # cumulative arc length is monotonic non-decreasing, starts at 0
    assert np.all(a.cum[:, 0] == 0.0)
    assert np.all(np.diff(a.cum, axis=1) >= -1e-4)
    # crossing leg is a real interval inside the path
    assert np.all(a.cross_hi > a.cross_lo)
    assert np.all(a.cross_hi <= a.cum[:, -1] + 1e-4)


def test_starts_staggered_and_bounded(simple_net):
    xy, n, lanes = simple_net
    p = build_ped_paths(xy, n, lanes, lane_width=3.5, n_peds=50, seed=1, max_start=60)
    assert p.starts.min() >= 0 and p.starts.max() < 60
    assert len(np.unique(p.starts)) > 1  # actually staggered


def test_arc_interp_endpoints_and_midpoint(simple_net):
    xy, n, lanes = simple_net
    p = build_ped_paths(xy, n, lanes, lane_width=3.5, n_peds=3, seed=2)
    paths, cum = jnp.asarray(p.paths), jnp.asarray(p.cum)
    # walked=0 -> path start; walked>=total -> path end (clamped)
    at_start = arc_interp(paths, cum, jnp.zeros(3))
    at_end = arc_interp(paths, cum, cum[:, -1] + 100.0)
    np.testing.assert_allclose(np.asarray(at_start), p.paths[:, 0, :], atol=1e-4)
    np.testing.assert_allclose(np.asarray(at_end), p.paths[:, -1, :], atol=1e-4)
    # halfway along total arc length lies on the polyline (finite, within bbox)
    mid = arc_interp(paths, cum, cum[:, -1] * 0.5)
    assert np.all(np.isfinite(np.asarray(mid)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/env/test_ped_paths.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'smoothride.env.ped_paths'`.

- [ ] **Step 3: Write minimal implementation**

```python
# smoothride/env/ped_paths.py
"""Deterministic pedestrian paths: a sidewalk run + one perpendicular crossing.

Built once on the host (NumPy); the env interpolates position along the polyline
as a pure function of time (no per-step RNG), so peds are reproducible and
JAX/vmap-friendly. Each path has 4 points: start sidewalk -> walk -> cross to the
far sidewalk -> walk. The crossing leg (point 1 -> point 2) is the moment the ped
is in the roadway, which cars must negotiate.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class PedPaths:
    paths: np.ndarray      # (M, 4, 2) polyline points, meters
    cum: np.ndarray        # (M, 4) cumulative arc length per point
    starts: np.ndarray     # (M,) int32 start step (staggered)
    cross_lo: np.ndarray   # (M,) f32 arc length where the crossing leg begins
    cross_hi: np.ndarray   # (M,) f32 arc length where the crossing leg ends


def build_ped_paths(routes_xy: np.ndarray, routes_n: np.ndarray,
                    routes_lanes: np.ndarray, lane_width: float, n_peds: int,
                    seed: int, *, sidewalk_offset: float = 1.5,
                    run_len: float = 12.0, max_start: int = 60) -> PedPaths:
    rng = np.random.default_rng(seed)
    R = routes_xy.shape[0]
    paths = np.zeros((n_peds, 4, 2), np.float32)
    for m in range(n_peds):
        r = int(rng.integers(0, R))
        nwp = max(int(routes_n[r]), 2)
        w = int(rng.integers(0, nwp - 1))           # segment [w, w+1]
        a, b = routes_xy[r, w], routes_xy[r, w + 1]
        u = b - a
        u = u / (np.linalg.norm(u) + 1e-6)          # along-segment unit
        nrm = np.array([u[1], -u[0]], np.float32)   # right-normal
        lanes = max(int(routes_lanes[r, w]), 1)
        half = lanes * lane_width / 2.0
        s = half + sidewalk_offset                  # sidewalk distance from centerline
        side = 1.0 if rng.random() < 0.5 else -1.0
        mid = (a + b) / 2.0
        p0 = mid + nrm * (s * side)                 # near sidewalk
        p1 = p0 + u * run_len                       # walk along sidewalk
        p2 = p1 - nrm * (2.0 * s * side)            # CROSS to far sidewalk (leg 1->2)
        p3 = p2 + u * run_len                       # walk along far sidewalk
        paths[m] = np.stack([p0, p1, p2, p3]).astype(np.float32)
    seg = np.linalg.norm(np.diff(paths, axis=1), axis=-1)          # (M, 3)
    cum = np.concatenate([np.zeros((n_peds, 1), np.float32),
                          np.cumsum(seg, axis=1)], axis=1).astype(np.float32)
    starts = rng.integers(0, max_start, size=n_peds).astype(np.int32)
    cross_lo = cum[:, 1].copy()
    cross_hi = cum[:, 2].copy()
    return PedPaths(paths=paths, cum=cum, starts=starts,
                    cross_lo=cross_lo, cross_hi=cross_hi)


def arc_interp(paths: jnp.ndarray, cum: jnp.ndarray,
               walked: jnp.ndarray) -> jnp.ndarray:
    """Batched position along each polyline at arc length `walked`. Clamped to ends."""
    total = cum[:, -1]
    s = jnp.clip(walked, 0.0, total)
    # segment index: number of cumulative breakpoints strictly below s, in [0, 2]
    seg = jnp.clip(jnp.sum(cum[:, 1:] <= s[:, None], axis=1), 0, paths.shape[1] - 2)
    lo = jnp.take_along_axis(cum, seg[:, None], axis=1)[:, 0]       # (M,)
    hi = jnp.take_along_axis(cum, (seg + 1)[:, None], axis=1)[:, 0]
    frac = jnp.clip((s - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    a = jnp.take_along_axis(paths, seg[:, None, None].repeat(2, 2), axis=1)[:, 0, :]
    b = jnp.take_along_axis(paths, (seg + 1)[:, None, None].repeat(2, 2), axis=1)[:, 0, :]
    return a + frac[:, None] * (b - a)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/env/test_ped_paths.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add smoothride/env/ped_paths.py tests/env/test_ped_paths.py
git commit -m "feat(env): deterministic pedestrian paths (sidewalk run + crossing) + arc interp"
```

---

### Task 2: Wire deterministic peds + layered radii + cruise cap into the env

**Files:**
- Modify: `smoothride/env/kinematic.py` (Env fields ~`30-89`, `State` ~`92-107`, `_ped_step` `219-229`, `reset` `287-305`, `step` `308-401`, `make_env` `404-418`)
- Test: `tests/env/test_kinematic_peds.py`

**Interfaces:**
- Consumes: `build_ped_paths`, `arc_interp` from Task 1.
- Produces:
  - `Env` gains static-ish array fields `ped_paths (M,4,2)`, `ped_cum (M,4)`, `ped_starts (M,)`, `cross_lo (M,)`, `cross_hi (M,)`, and scalar params `cruise_cap: float = 7.0`, `ped_radius` default raised to `3.5`, `r_yield: float = 9.0` (replaces the use of vestigial `prox_radius`).
  - `State` gains `ped_vel: jnp.ndarray (M,2)` and `ped_crossing: jnp.ndarray (M,) bool`; keeps `ped_pos`; `ped_dir` retained (derived from `ped_vel` for rendering).
  - `_ped_step(env, st) -> (ped_pos, ped_vel, ped_dir, ped_crossing)` — deterministic, NO key argument.
  - `make_env(..., n_peds=..., seed=0, **kw)` builds ped paths and stores them on `Env`.

- [ ] **Step 1: Write the failing test**

```python
# tests/env/test_kinematic_peds.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/env/test_kinematic_peds.py -v`
Expected: FAIL — `make_env` rejects `seed`/`cruise_cap` kwargs or `State` lacks `ped_crossing`.

- [ ] **Step 3: Write minimal implementation**

In `kinematic.py`, add Env fields (after `ped_speed`, near line 64):

```python
    cruise_cap: float = struct.field(pytree_node=False, default=7.0)
    r_yield: float = struct.field(pytree_node=False, default=9.0)
    # prebuilt deterministic pedestrian paths (host-built in make_env)
    ped_paths: jnp.ndarray = None
    ped_cum: jnp.ndarray = None
    ped_starts: jnp.ndarray = None
    cross_lo: jnp.ndarray = None
    cross_hi: jnp.ndarray = None
```

Raise the `ped_radius` default:

```python
    ped_radius: float = struct.field(pytree_node=False, default=3.5)
```

Add `State` fields (after `ped_dir`, line ~106):

```python
    ped_vel: jnp.ndarray
    ped_crossing: jnp.ndarray
```

Replace `_ped_step` (lines 219-229) with the deterministic version:

```python
def _ped_step(env: Env, st: State):
    """Deterministic ped motion: position is a pure function of time along the
    prebuilt polyline. No RNG. Returns (pos, vel, dir, crossing)."""
    from .ped_paths import arc_interp
    walked = (jnp.maximum(0, st.t - env.ped_starts).astype(jnp.float32)
              * env.ped_speed * env.dt)
    ped_pos = arc_interp(env.ped_paths, env.ped_cum, walked)
    # velocity from a small finite-difference lookahead along the arc
    ahead = arc_interp(env.ped_paths, env.ped_cum, walked + env.ped_speed * env.dt)
    delta = ahead - ped_pos
    moving = (walked > 0) & (walked < env.ped_cum[:, -1])
    ped_vel = jnp.where(moving[:, None], delta / env.dt, 0.0)
    ped_dir = jnp.arctan2(ped_vel[:, 1], ped_vel[:, 0])
    crossing = (walked >= env.cross_lo) & (walked <= env.cross_hi) & moving
    return ped_pos, ped_vel, _wrap(ped_dir), crossing
```

In `reset` (lines 287-305): drop the random ped placement, compute ped state at `t=0`, and set the new State fields. Replace the ped lines:

```python
def reset(env: Env, key: jax.Array):
    kc = key
    n = env.n_agents
    route_idx, pos, heading, wp = _place_cars(env, kc)
    st0 = State(
        pos=pos, heading=heading, speed=jnp.zeros(n),
        route_idx=route_idx, wp_ptr=wp, lane=jnp.zeros(n, jnp.int32),
        just_crashed=jnp.zeros(n, bool), crashes=jnp.zeros(n, jnp.int32),
        spawn_grace=jnp.full(n, SPAWN_GRACE, jnp.int32),
        arrived=jnp.zeros(n, bool), goals=jnp.zeros(n, jnp.int32),
        ped_pos=env.ped_paths[:, 0, :], ped_dir=jnp.zeros(env.n_peds),
        ped_vel=jnp.zeros((env.n_peds, 2)),
        ped_crossing=jnp.zeros(env.n_peds, bool),
        t=jnp.array(0, jnp.int32),
    )
    ped_pos, ped_vel, ped_dir, crossing = _ped_step(env, st0)
    st = st0.replace(ped_pos=ped_pos, ped_vel=ped_vel, ped_dir=ped_dir,
                     ped_crossing=crossing)
    return st, _observe(env, st, _candidates(env, st.pos))
```

In `step` (line 310): remove the `kped` split usage and update the ped-advance + speed clamp + State construction:

```python
    speed = jnp.clip(st.speed + accel * env.dt, 0.0,
                     jnp.minimum(vmax, env.cruise_cap))   # cruise cap
```

Replace `ped_pos, ped_dir = _ped_step(env, st, kped)` (line 380) with:

```python
    ped_pos, ped_vel, ped_dir, ped_crossing = _ped_step(env, st)
```

and the `nst = State(...)` (lines 392-395) ped fields:

```python
                goals=goals, ped_pos=ped_pos, ped_dir=ped_dir,
                ped_vel=ped_vel, ped_crossing=ped_crossing, t=t)
```

In `make_env` (404-418), build paths and pass them through:

```python
def make_env(pool: RoutePool, world_min, world_max, cell_size=35.0, cap=16,
             n_peds=12, seed=0, **kw) -> Env:
    import numpy as np
    from .ped_paths import build_ped_paths
    seg = np.linalg.norm(np.diff(pool.xy, axis=1), axis=-1)
    cum = np.concatenate([np.zeros((pool.xy.shape[0], 1)), np.cumsum(seg, axis=1)], 1)
    ncx, ncy = spatial.grid_dims(world_min, world_max, cell_size)
    lane_width = kw.get("lane_width", 3.5)
    pp = build_ped_paths(np.asarray(pool.xy), np.asarray(pool.n),
                         np.asarray(pool.lanes), lane_width, n_peds, seed)
    return Env(
        routes_xy=jnp.asarray(pool.xy), routes_n=jnp.asarray(pool.n),
        routes_node=jnp.asarray(pool.node), routes_junc=jnp.asarray(pool.junc),
        routes_lanes=jnp.asarray(pool.lanes), routes_speed=jnp.asarray(pool.speed),
        routes_cum=jnp.asarray(cum, jnp.float32),
        world_min=jnp.asarray(world_min, jnp.float32),
        world_max=jnp.asarray(world_max, jnp.float32),
        n_peds=n_peds,
        ped_paths=jnp.asarray(pp.paths), ped_cum=jnp.asarray(pp.cum),
        ped_starts=jnp.asarray(pp.starts), cross_lo=jnp.asarray(pp.cross_lo),
        cross_hi=jnp.asarray(pp.cross_hi),
        cell_size=cell_size, cap=cap, ncx=ncx, ncy=ncy,
        cand_C=spatial.candidate_count(cap), **kw,
    )
```

Delete `_place_peds` (lines 266-284) — peds are no longer randomly placed.

> NOTE: `step`'s signature still takes `key` (cars still need RNG for nothing now, but `collect` passes one); keep the parameter for call-site compatibility but it is unused for peds. The line `_, _, kped = jax.random.split(key, 3)` can be removed.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/env/test_kinematic_peds.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the existing env tests to catch regressions**

Run: `python -m pytest tests/env -v`
Expected: PASS. If any test referenced `_place_peds` or the old random ped motion, update it to the deterministic model (do not weaken assertions).

- [ ] **Step 6: Commit**

```bash
git add smoothride/env/kinematic.py tests/env/test_kinematic_peds.py
git commit -m "feat(env): deterministic peds + ped velocity/crossing state, cruise cap, raised ped radius"
```

---

### Task 3: Continuous ped-yield cost in the verifier

**Files:**
- Modify: `smoothride/rl/verifier.py` (constants ~`20-25`, new function, `step_cost` `151-165`)
- Test: `tests/rl/test_verifier_pedyield.py`

**Interfaces:**
- Produces:
  - `ped_yield_cost(pos, speed, ped_pos, ped_crossing, r_ped, r_yield, cruise_cap) -> np.ndarray` — `pos (T,N,2)`, `speed (T,N)`, `ped_pos (T,M,2)`, `ped_crossing (T,M) bool`; returns `(T,N)` continuous cost in ~`[0, 1]`.
  - `step_cost(...)` gains optional kwargs `ped_pos=None, ped_crossing=None, r_ped=3.5, r_yield=9.0, cruise_cap=7.0`; when `ped_pos is not None`, the ped-yield term is added to the existing cost sum.

**Hinge definition (from spec §7):** for car `i`, over crossing peds `j`:
`p_ij = clip((r_yield − d_ij) / (r_yield − r_ped), 0, 1)`, `yield_ij = p_ij · (speed_i / cruise_cap)`, take the **max over crossing peds**. Zero when stopped, far, or no ped crossing.

- [ ] **Step 1: Write the failing test**

```python
# tests/rl/test_verifier_pedyield.py
import numpy as np

from smoothride.rl.verifier import ped_yield_cost


def _one(pos, speed, ped, crossing, r_ped=3.5, r_yield=9.0, cap=7.0):
    # shape (1,1,2),(1,1),(1,1,2),(1,1) -> scalar
    return float(ped_yield_cost(
        np.array([[pos]], np.float32), np.array([[speed]], np.float32),
        np.array([[ped]], np.float32), np.array([[crossing]], bool),
        r_ped, r_yield, cap)[0, 0])


def test_zero_when_far():
    assert _one([0, 0], 7.0, [100, 0], True) == 0.0


def test_zero_when_stopped_even_if_adjacent():
    assert _one([0, 0], 0.0, [4.0, 0], True) == 0.0


def test_zero_when_ped_not_crossing():
    assert _one([0, 0], 7.0, [4.0, 0], False) == 0.0


def test_ramps_with_proximity():
    near = _one([0, 0], 7.0, [4.0, 0], True)   # just outside hard radius
    far = _one([0, 0], 7.0, [8.0, 0], True)    # near the outer edge
    assert near > far > 0.0


def test_ramps_with_speed():
    fast = _one([0, 0], 7.0, [5.0, 0], True)
    slow = _one([0, 0], 2.0, [5.0, 0], True)
    assert fast > slow > 0.0


def test_bounded_and_graded_not_binary():
    c = _one([0, 0], 7.0, [3.5, 0], True)      # at hard radius, full speed
    assert 0.9 <= c <= 1.0
    mid = _one([0, 0], 7.0, [6.25, 0], True)   # midpoint of [3.5, 9]
    assert 0.2 < mid < 0.8                       # continuous, not 0/1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/rl/test_verifier_pedyield.py -v`
Expected: FAIL — `ImportError: cannot import name 'ped_yield_cost'`.

- [ ] **Step 3: Write minimal implementation**

In `verifier.py`, add near the constants:

```python
PED_RADIUS = 3.5         # m; hard car-ped keep-out (asymmetric: wider than car-car)
PED_YIELD_RADIUS = 9.0   # m; outer yield zone where the continuous cost ramps
CRUISE_CAP = 7.0         # m/s; reference speed for normalizing the yield term
```

Add the function:

```python
def ped_yield_cost(pos: np.ndarray, speed: np.ndarray, ped_pos: np.ndarray,
                   ped_crossing: np.ndarray, r_ped: float = PED_RADIUS,
                   r_yield: float = PED_YIELD_RADIUS,
                   cruise_cap: float = CRUISE_CAP) -> np.ndarray:
    """(T,N) continuous yield cost: ramps with proximity x speed toward a CROSSING
    ped. Graded (a hinge), not a 0/1 flag, so the optimum is to SLOW, not freeze."""
    d = np.linalg.norm(pos[:, :, None, :] - ped_pos[:, None, :, :], axis=-1)  # (T,N,M)
    prox = np.clip((r_yield - d) / (r_yield - r_ped), 0.0, 1.0)
    prox = np.where(ped_crossing[:, None, :], prox, 0.0)                      # gate
    prox = prox.max(axis=-1)                                                  # (T,N)
    spd = np.clip(speed / cruise_cap, 0.0, 1.0)
    return (prox * spd).astype(np.float32)
```

Extend `step_cost` signature and body:

```python
def step_cost(pos, seg_start, seg_end, lane_count, lane_width, heading, speed,
              spawn_grace, crashed, speed_limit=None, *, ped_pos=None,
              ped_crossing=None, r_ped=PED_RADIUS, r_yield=PED_YIELD_RADIUS,
              cruise_cap=CRUISE_CAP) -> np.ndarray:
    _, off_lane, ww = _lane_flags(pos, seg_start, seg_end, lane_count, lane_width,
                                  heading, speed, spawn_grace)
    cost = (np.asarray(crashed, np.float32) + off_lane.astype(np.float32)
            + ww.astype(np.float32))
    if speed_limit is not None:
        cost = cost + (speed > speed_limit + SPEED_EPS).astype(np.float32)
    if ped_pos is not None:
        cost = cost + ped_yield_cost(pos, speed, ped_pos, ped_crossing,
                                     r_ped, r_yield, cruise_cap)
    return cost
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/rl/test_verifier_pedyield.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add smoothride/rl/verifier.py tests/rl/test_verifier_pedyield.py
git commit -m "feat(verifier): continuous ped-yield cost (proximity x speed, gated to crossing peds)"
```

---

### Task 4: Add ped fields to the Trace + wire cost_signal

**Files:**
- Modify: `smoothride/rl/trace.py` (timeline tuples ~`20-24`, dataclass `42-91`)
- Modify: `smoothride/rl/verifier.py` (`cost_signal` `168-173`)
- Modify: `tests/rl/conftest.py` (the `make_trace` fixture, ~`16-55`)
- Test: extend `tests/rl/test_verifier_pedyield.py`

**Interfaces:**
- Consumes: `ped_yield_cost`/`step_cost` (Task 3).
- Produces: `Trace` gains `ped_pos: np.ndarray (T,M,2)` and `ped_crossing: np.ndarray (T,M) bool`, validated in `__post_init__`. `cost_signal(trace)` passes them to `step_cost`. The `make_trace` fixture accepts `n_peds` (default 2) and synthesizes far-away non-crossing peds by default (so existing verifier tests are unaffected).

- [ ] **Step 1: Write the failing test (append)**

```python
# tests/rl/test_verifier_pedyield.py  (append)
from smoothride.rl.verifier import cost_signal


def test_cost_signal_includes_ped_yield(make_trace):
    # a car at origin moving at cruise speed, a crossing ped 4 m away -> cost > lane terms
    trace = make_trace(n_steps=1, n_agents=1, n_peds=1,
                       pos=[[[0.0, 0.0]]], speed=[[7.0]],
                       ped_pos=[[[4.0, 0.0]]], ped_crossing=[[True]])
    c = cost_signal(trace)
    assert c.shape == (1, 1)
    assert c[0, 0] > 0.5   # ped-yield term present and large
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/rl/test_verifier_pedyield.py::test_cost_signal_includes_ped_yield -v`
Expected: FAIL — `make_trace` got unexpected kwargs / `Trace` has no `ped_pos`.

- [ ] **Step 3: Write minimal implementation**

In `trace.py`, add to `_TIMELINE_2D`: `"ped_crossing"` is `(T,M)` not `(T,N)`, so handle it separately. Add new fields and a ped-shape check in `__post_init__`:

```python
    ped_pos: np.ndarray         # (T, M, 2) meters
    ped_crossing: np.ndarray    # (T, M) bool — ped is in its crossing leg
```

Add to `__post_init__` (after the existing checks):

```python
        M = self.manifest.n_peds
        if self.ped_pos.shape != (T, M, 2):
            raise ValueError(
                f"trace.ped_pos must have shape {(T, M, 2)}, got {self.ped_pos.shape}")
        if self.ped_crossing.shape != (T, M):
            raise ValueError(
                f"trace.ped_crossing must have shape {(T, M)}, "
                f"got {self.ped_crossing.shape}")
```

In `verifier.py`, update `cost_signal`:

```python
def cost_signal(trace: Trace) -> np.ndarray:
    return step_cost(trace.pos, trace.seg_start, trace.seg_end, trace.lane_count,
                     trace.lane_width, trace.heading, trace.speed, trace.spawn_grace,
                     trace.crashed, trace.speed_limit,
                     ped_pos=trace.ped_pos, ped_crossing=trace.ped_crossing)
```

In `tests/rl/conftest.py`, extend `make_trace` to accept `n_peds`, `ped_pos`, `ped_crossing` (default: peds parked far away, not crossing) and pass them into the `Trace` and `TraceManifest(n_peds=...)`. Example additions inside the fixture builder:

```python
        M = n_peds
        ped_pos = (np.asarray(ped_pos, np.float32)
                   if ped_pos is not None
                   else np.full((n_steps, M, 2), 1e6, np.float32))
        ped_crossing = (np.asarray(ped_crossing, bool)
                        if ped_crossing is not None
                        else np.zeros((n_steps, M), bool))
        # ... TraceManifest(..., n_peds=M)
        # ... Trace(..., ped_pos=ped_pos, ped_crossing=ped_crossing)
```

- [ ] **Step 4: Run the test + the full verifier suite**

Run: `python -m pytest tests/rl/test_verifier_pedyield.py tests/rl/test_verifier.py tests/rl/test_trace.py -v`
Expected: PASS (new test + all existing trace/verifier tests still green).

- [ ] **Step 5: Commit**

```bash
git add smoothride/rl/trace.py smoothride/rl/verifier.py tests/rl/conftest.py tests/rl/test_verifier_pedyield.py
git commit -m "feat(trace): ped_pos/ped_crossing fields; cost_signal applies ped-yield"
```

---

### Task 5: Log peds in rollouts + ped-yield in the training relabel

**Files:**
- Modify: `smoothride/rl/ppo.py` (`collect` out-dict `68-76`, `verifier_cost` `92-124`)
- Test: extend `tests/rl/test_verifier_pedyield.py` with a tiny integration check.

**Interfaces:**
- Consumes: env from Task 2 (`State.ped_pos`, `State.ped_crossing`), `step_cost` (Task 3).
- Produces: `collect` logs `ped_pos (B,T,M,2)` and `ped_crossing (B,T,M)` in the batch; `verifier_cost(env, batch)` adds the ped-yield term using `env.ped_radius`, `env.r_yield`, `env.cruise_cap`.

- [ ] **Step 1: Write the failing test**

```python
# tests/rl/test_verifier_pedyield.py  (append)
import jax
import jax.numpy as jnp
from smoothride.rl import ppo
from tests.env.test_kinematic_peds import _env  # reuse the tiny env


def test_collect_logs_peds_and_verifier_cost_runs():
    env = _env(cruise_cap=4.0)
    ts = ppo.make_train_state(env, ppo.PPOConfig(n_worlds=2), jax.random.PRNGKey(0))
    batch = ppo.collect(env, ts, jax.random.PRNGKey(1), 2)
    assert batch["ped_pos"].shape == (2, env.max_steps, env.n_peds, 2)
    assert batch["ped_crossing"].shape == (2, env.max_steps, env.n_peds)
    cost = ppo.verifier_cost(env, batch)
    assert cost.shape == (2, env.max_steps, env.n_agents)
    assert float(jnp.asarray(cost).max()) >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/rl/test_verifier_pedyield.py::test_collect_logs_peds_and_verifier_cost_runs -v`
Expected: FAIL — `KeyError: 'ped_pos'`.

- [ ] **Step 3: Write minimal implementation**

In `collect`'s `out = dict(...)` (ppo.py 68-76) add:

```python
                       ped_pos=st.ped_pos, ped_crossing=st.ped_crossing,
```

In `verifier_cost`, after computing `seg_*`/limits, pass peds to `step_cost`. Peds are shared across agents within a world, so broadcast per (B,T): reshape `ped_pos (B,T,M,2)` → `(B*T,M,2)` and `ped_crossing` → `(B*T,M)`:

```python
    pp = r2(np.asarray(batch["ped_pos"]))          # (B*T, M, 2)
    pc = r2(np.asarray(batch["ped_crossing"]))     # (B*T, M)
    cost = step_cost(
        r2(np.asarray(batch["pos"])), r2(seg_start), r2(seg_end), r2(lane_count),
        float(env.lane_width), r2(np.asarray(batch["heading"])),
        r2(np.asarray(batch["speed"])), r2(np.asarray(batch["spawn_grace"])),
        r2(np.asarray(batch["crashed"])), r2(speed_limit),
        ped_pos=pp, ped_crossing=pc, r_ped=float(env.ped_radius),
        r_yield=float(env.r_yield), cruise_cap=float(env.cruise_cap))
    return jnp.asarray(cost.reshape(B, T, N))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/rl/test_verifier_pedyield.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add smoothride/rl/ppo.py tests/rl/test_verifier_pedyield.py
git commit -m "feat(ppo): log peds in rollouts; ped-yield term in verifier_cost relabel"
```

> **Checkpoint:** Part A is complete. The env now has deterministic crossing peds, a cruise cap, a layered ped safety radius, and a continuous ped-yield cost wired through both the training relabel and the offline grader — trainable with the *existing* observation. Run `python -m pytest tests/ -q` to confirm the whole suite is green before Part B.

---

## Part B — Deep Sets perception

> Part B restructures the observation from a flat `(N, 26)` array into a pytree of padded+masked entity sets, and adds a Deep Sets front-end to the policy. This cascades through `_observe`, `networks.ActorCritic`, and `ppo` together. Build the encoder first (Task 6, pure-network, independently testable), then switch the observation + plumbing (Task 7), then the integration smoke (Task 8).

### Task 6: Deep Sets encoder module

**Files:**
- Modify: `smoothride/rl/networks.py` (add `DeepSets` module + use it in `ActorCritic`)
- Test: `tests/rl/test_networks_deepsets.py`

**Interfaces:**
- Produces:
  - `DeepSets(feat_dim: int, hidden: int = 64)` — Flax module. `__call__(entities: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray`. `entities (..., C, feat_dim)`, `mask (..., C) bool`; returns `(..., 2*hidden)` = `concat[mean_pool, max_pool]` of `φ(entity)` over valid slots. Empty set (all-mask-false) → zeros. Permutation-invariant.

- [ ] **Step 1: Write the failing test**

```python
# tests/rl/test_networks_deepsets.py
import jax
import jax.numpy as jnp
import numpy as np

from smoothride.rl.networks import DeepSets


def _apply(mod, ents, mask):
    p = mod.init(jax.random.PRNGKey(0), ents, mask)
    return mod.apply(p, ents, mask), p


def test_output_shape_and_empty_set_is_zero():
    mod = DeepSets(feat_dim=4, hidden=8)
    ents = jnp.ones((3, 5, 4))                      # batch 3, cap 5, feat 4
    mask = jnp.zeros((3, 5), bool)                  # all empty
    out, p = _apply(mod, ents, mask)
    assert out.shape == (3, 16)                     # 2*hidden
    np.testing.assert_allclose(np.asarray(out), 0.0, atol=1e-6)


def test_permutation_invariance():
    mod = DeepSets(feat_dim=4, hidden=8)
    ents = jax.random.normal(jax.random.PRNGKey(1), (1, 5, 4))
    mask = jnp.array([[True, True, True, False, False]])
    out_a, p = _apply(mod, ents, mask)
    perm = jnp.array([2, 0, 1, 3, 4])              # shuffle only valid+padding consistently
    out_b = mod.apply(p, ents[:, perm], mask[:, perm])
    np.testing.assert_allclose(np.asarray(out_a), np.asarray(out_b), atol=1e-5)


def test_masked_slots_do_not_affect_output():
    mod = DeepSets(feat_dim=4, hidden=8)
    ents = jax.random.normal(jax.random.PRNGKey(2), (1, 5, 4))
    mask = jnp.array([[True, True, False, False, False]])
    out_a, p = _apply(mod, ents, mask)
    # garbage in the masked slots must not change the result
    ents2 = ents.at[:, 2:].set(999.0)
    out_b = mod.apply(p, ents2, mask)
    np.testing.assert_allclose(np.asarray(out_a), np.asarray(out_b), atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/rl/test_networks_deepsets.py -v`
Expected: FAIL — `ImportError: cannot import name 'DeepSets'`.

- [ ] **Step 3: Write minimal implementation**

Add to `networks.py`:

```python
import flax.linen as nn
import jax.numpy as jnp


class DeepSets(nn.Module):
    """Permutation-invariant set encoder: per-element MLP phi, then masked
    mean+max pool. Empty set -> zeros. Density-agnostic, handles padded slots."""
    feat_dim: int
    hidden: int = 64

    @nn.compact
    def __call__(self, entities: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        h = nn.relu(nn.Dense(self.hidden)(entities))
        h = nn.relu(nn.Dense(self.hidden)(h))           # (..., C, hidden)
        m = mask[..., None].astype(h.dtype)             # (..., C, 1)
        h = h * m                                       # zero invalid slots
        summed = jnp.sum(h, axis=-2)                    # (..., hidden)
        count = jnp.clip(jnp.sum(m, axis=-2), 1.0)      # live count, >=1
        mean = summed / count
        # masked max: set invalid slots to -inf-ish so they never win, then guard empty
        neg = jnp.where(m > 0, h, -1e9)
        mx = jnp.max(neg, axis=-2)
        any_valid = jnp.sum(m, axis=-2) > 0             # (..., 1)
        mx = jnp.where(any_valid, mx, 0.0)
        return jnp.concatenate([mean, mx], axis=-1)     # (..., 2*hidden)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/rl/test_networks_deepsets.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add smoothride/rl/networks.py tests/rl/test_networks_deepsets.py
git commit -m "feat(networks): permutation-invariant Deep Sets encoder (masked mean+max pool)"
```

---

### Task 7: Structured observation (ego + masked car/ped sets) + ActorCritic wiring

**Files:**
- Modify: `smoothride/env/kinematic.py` (`_observe` `160-216`, `obs_dim`/new shape constants, `Movers` helper)
- Modify: `smoothride/rl/networks.py` (`ActorCritic.__call__` consumes the structured obs via `DeepSets`)
- Modify: `smoothride/rl/ppo.py` (`make_train_state` dummy obs, `_global_feat`, `collect`/`update` flatten helpers handle a dict obs)
- Test: `tests/env/test_kinematic_peds.py` (append obs-structure tests)

**Interfaces:**
- Produces:
  - `_observe(env, st, cand) -> dict` with keys: `ego (N, E0)`, `cars (N, Cc, FC)`, `cars_mask (N, Cc)`, `peds (N, Cp, FP)`, `peds_mask (N, Cp)`. Constants: `E0 = 7` (6 ego + lane_frac), `FC = 5` (rel x, rel y, rel vx, rel vy, is-neighbor-valid≡via mask so 4 features + nothing; use 4), `FP = 5` (rel x, rel y, rel vx, rel vy, crossing-bit). Set `FC = 4`, `FP = 5`.
  - `Env` gains `cand_cap_car: int = 16`, `cand_cap_ped: int = 16` (static).
  - `ActorCritic(act_dim)` now takes `obs: dict, gf: dict` (global feat is the pooled ego/embedding broadcast). Output unchanged: `(mean, log_std, value)`.

- [ ] **Step 1: Write the failing test (append)**

```python
# tests/env/test_kinematic_peds.py  (append)
def test_observation_is_structured_with_masks():
    env = _env()
    st, obs = K.reset(env, jax.random.PRNGKey(0))
    assert set(obs) == {"ego", "cars", "cars_mask", "peds", "peds_mask"}
    N = env.n_agents
    assert obs["ego"].shape == (N, 7)
    assert obs["cars"].shape == (N, env.cand_cap_car, 4)
    assert obs["cars_mask"].shape == (N, env.cand_cap_car)
    assert obs["peds"].shape == (N, env.cand_cap_ped, 5)
    assert obs["peds_mask"].shape == (N, env.cand_cap_ped)
    # masks are boolean and self is never a neighbor of itself
    assert obs["cars_mask"].dtype == jnp.bool_


def test_ped_crossing_bit_present_in_obs():
    env = _env()
    st, obs = K.reset(env, jax.random.PRNGKey(0))
    # 5th ped-feature is the crossing bit in {0,1}
    bit = np.asarray(obs["peds"][..., 4])
    assert set(np.unique(bit[np.asarray(obs["peds_mask"])])) <= {0.0, 1.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/env/test_kinematic_peds.py::test_observation_is_structured_with_masks -v`
Expected: FAIL — obs is an array, not a dict.

- [ ] **Step 3: Write minimal implementation**

Rewrite `_observe` (kinematic.py 160-216) to return the structured dict. Keep the ego block (6 dims) + `lane_frac`. Build a radius-capped car set from `cand` (already the spatial-hash candidates) by taking the nearest `cand_cap_car`; build a ped set by nearest `cand_cap_ped`. All entity features are ego-relative (rotate into ego heading frame as the existing code does).

```python
def _observe(env: Env, st: State, cand) -> dict:
    N = env.n_agents
    vmax = jnp.minimum(env.v_max, env.routes_speed[st.route_idx, st.wp_ptr])
    tgt = _target_wp(env, st)
    to_wp = tgt - st.pos
    dist = jnp.linalg.norm(to_wp, axis=-1)
    herr = _wrap(jnp.arctan2(to_wp[:, 1], to_wp[:, 0]) - st.heading)
    n = env.routes_n[st.route_idx]
    progress = st.wp_ptr / jnp.maximum(n - 1, 1)
    fdir = jnp.stack([jnp.cos(st.heading), jnp.sin(st.heading)], -1)
    c, s = jnp.cos(-st.heading), jnp.sin(-st.heading)

    # ----- lead gap (kept for the ego block) -----
    valid = (cand >= 0) & (cand != jnp.arange(N)[:, None])
    rel = st.pos[cand] - st.pos[:, None, :]
    forward = jnp.sum(rel * fdir[:, None, :], -1)
    lateral = jnp.abs(rel[..., 0] * fdir[:, None, 1] - rel[..., 1] * fdir[:, None, 0])
    ahead = valid & (forward > 0.5) & (lateral < env.lane_width) & (forward < env.lead_cone)
    lead_gap = jnp.min(jnp.where(ahead, forward, env.lead_cone), axis=-1)

    lane_frac = st.lane.astype(jnp.float32) / jnp.maximum(
        env.routes_lanes[st.route_idx, st.wp_ptr] - 1, 1)
    ego = jnp.stack([
        st.speed / jnp.maximum(vmax, 1.0), jnp.sin(herr), jnp.cos(herr),
        jnp.clip(dist / 100.0, 0, 1), progress,
        jnp.clip(lead_gap / env.lead_cone, 0, 1),
    ], axis=-1)
    ego = jnp.concatenate([ego, lane_frac[:, None]], -1)              # (N, 7)

    # ----- car set: nearest cand_cap_car candidates, ego-frame, masked -----
    cd = jnp.where(valid, jnp.linalg.norm(rel, axis=-1), 1e9)
    _, kk = jax.lax.top_k(-cd, env.cand_cap_car)                       # (N, Cc)
    nbr = jnp.take_along_axis(cand, kk, axis=1)
    nbr_valid = jnp.take_along_axis(valid, kk, axis=1)
    nrel = st.pos[nbr] - st.pos[:, None, :]
    nx = (nrel[..., 0] * c[:, None] - nrel[..., 1] * s[:, None])
    ny = (nrel[..., 0] * s[:, None] + nrel[..., 1] * c[:, None])
    vel = st.speed[:, None] * fdir
    nvel = vel[nbr] - vel[:, None, :]
    nvx = nvel[..., 0] * c[:, None] - nvel[..., 1] * s[:, None]
    nvy = nvel[..., 0] * s[:, None] + nvel[..., 1] * c[:, None]
    cars = jnp.stack([nx / 50, ny / 50, nvx / env.v_max, nvy / env.v_max], -1)
    cars = jnp.clip(cars, -1, 1)

    # ----- ped set: nearest cand_cap_ped peds, ego-frame, masked, + crossing bit -----
    pd = st.pos[:, None, :] - st.ped_pos[None, :, :]
    pdist = jnp.linalg.norm(pd, axis=-1)                              # (N, M)
    _, pk = jax.lax.top_k(-pdist, env.cand_cap_ped)                   # (N, Cp)
    prel = st.ped_pos[pk] - st.pos[:, None, :]
    px = prel[..., 0] * c[:, None] - prel[..., 1] * s[:, None]
    py = prel[..., 0] * s[:, None] + prel[..., 1] * c[:, None]
    pvel = st.ped_vel[pk]
    pvx = pvel[..., 0] * c[:, None] - pvel[..., 1] * s[:, None]
    pvy = pvel[..., 0] * s[:, None] + pvel[..., 1] * c[:, None]
    cross = st.ped_crossing[pk].astype(jnp.float32)
    peds = jnp.stack([jnp.clip(px / 50, -1, 1), jnp.clip(py / 50, -1, 1),
                      jnp.clip(pvx / env.v_max, -1, 1),
                      jnp.clip(pvy / env.v_max, -1, 1), cross], -1)
    peds_mask = jnp.take_along_axis(pdist, pk, axis=1) < env.r_yield * 3.0  # in-range gate

    return {"ego": ego, "cars": cars, "cars_mask": nbr_valid,
            "peds": peds, "peds_mask": peds_mask}
```

Update `obs_dim` property — replace it with explicit feature-dim constants used by the network:

```python
    CAR_FEAT = 4
    PED_FEAT = 5
    EGO_FEAT = 7

    @property
    def obs_dim(self) -> int:               # retained for back-compat callers
        return self.EGO_FEAT
```

Add the candidate caps to `Env` fields:

```python
    cand_cap_car: int = struct.field(pytree_node=False, default=16)
    cand_cap_ped: int = struct.field(pytree_node=False, default=16)
```

In `networks.py`, rewrite `ActorCritic` to consume the dict:

```python
class ActorCritic(nn.Module):
    act_dim: int
    hidden: int = 64

    @nn.compact
    def __call__(self, obs, gf):
        car_enc = DeepSets(feat_dim=4, hidden=self.hidden)(obs["cars"], obs["cars_mask"])
        ped_enc = DeepSets(feat_dim=5, hidden=self.hidden)(obs["peds"], obs["peds_mask"])
        x = jnp.concatenate([obs["ego"], car_enc, ped_enc, gf], axis=-1)
        h = nn.relu(nn.Dense(self.hidden)(x))
        h = nn.relu(nn.Dense(self.hidden)(h))
        mean = nn.Dense(self.act_dim)(h)
        log_std = self.param("log_std", nn.initializers.zeros, (self.act_dim,))
        value = nn.Dense(1)(jax.lax.stop_gradient(x) if False else x)[..., 0]
        return mean, jnp.broadcast_to(log_std, mean.shape), value
```

> Keep whatever existing trunk/critic structure `networks.py` already uses if it differs — the only required change is: encode `obs["cars"]`/`obs["peds"]` with `DeepSets`, concat with `obs["ego"]` and `gf`, feed the existing MLP head. Preserve the existing `gaussian_logp`/`gaussian_entropy` contract.

In `ppo.py`:
- `_global_feat(obs)`: pool the **ego** vector across agents and broadcast — `return jnp.broadcast_to(obs["ego"].mean(-2, keepdims=True), obs["ego"].shape)`.
- `make_train_state`: build a dummy structured obs matching the shapes (use `K.reset` on a 1-world env, or construct zeros of the right shapes) to `net.init`.
- `collect`: `gf = _global_feat(obs)` already works; storing `obs` as a dict in the trajectory is fine (scan handles pytrees). The `flat`/`flatten` helpers in `update` must map over dict leaves:

```python
def _flat_obs(obs):                 # dict of (B,T,N,...) -> dict of (B*T*N, ...)
    return {k: v.reshape((-1,) + v.shape[3:]) for k, v in obs.items()}
```

Replace `obs = flat(batch["obs"])` / `gf = flat(batch["gf"])` with `obs = _flat_obs(batch["obs"])` and `gf = flat(batch["gf"])` (gf is still an array since `_global_feat` returns the ego-shaped array; store it as the ego broadcast). Index dict obs in the minibatch step with `{k: v[idx] for k, v in obs.items()}`.

- [ ] **Step 2 (re-run): Run obs tests + a network init smoke**

Run: `python -m pytest tests/env/test_kinematic_peds.py tests/rl/test_networks_deepsets.py -v`
Expected: PASS.

- [ ] **Step 3: Implement until green** (iterate on the shape plumbing above).

- [ ] **Step 4: Commit**

```bash
git add smoothride/env/kinematic.py smoothride/rl/networks.py smoothride/rl/ppo.py tests/env/test_kinematic_peds.py
git commit -m "feat(perception): structured obs (ego + masked car/ped sets) + Deep Sets ActorCritic"
```

---

### Task 8: End-to-end training smoke (one update step)

**Files:**
- Test: `tests/rl/test_ppo_smoke.py` (create)

**Interfaces:**
- Consumes: everything above. Verifies `collect → verifier_cost → update` runs and the loss is finite.

- [ ] **Step 1: Write the failing test**

```python
# tests/rl/test_ppo_smoke.py
import jax
import jax.numpy as jnp

from smoothride.rl import ppo
from tests.env.test_kinematic_peds import _env


def test_one_ppo_iteration_runs_end_to_end():
    env = _env(cruise_cap=4.0)
    cfg = ppo.PPOConfig(n_worlds=2, epochs=1, minibatches=2)
    ts = ppo.make_train_state(env, cfg, jax.random.PRNGKey(0))
    batch = ppo.collect(env, ts, jax.random.PRNGKey(1), cfg.n_worlds)
    batch = {**batch, "cost": ppo.verifier_cost(env, batch)}
    ts2, metrics = ppo.update(env, cfg, ts, batch, lam=1.0)
    assert jnp.isfinite(metrics["loss"])
    assert jnp.isfinite(metrics["ep_reward"])
```

- [ ] **Step 2: Run test to verify it fails / passes**

Run: `python -m pytest tests/rl/test_ppo_smoke.py -v`
Expected: initially may FAIL on a shape mismatch in `update`; fix the dict-obs flattening until PASS.

- [ ] **Step 3: Fix plumbing until green.**

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add tests/rl/test_ppo_smoke.py
git commit -m "test(ppo): end-to-end smoke — collect/relabel/update with Deep Sets + ped-yield"
```

> **Checkpoint:** Part B complete. The policy now perceives radius-capped masked sets of cars and peds through Deep Sets. Update `scripts/smoke_verifier.py` and `scripts/eval_policy.py` if they construct `Trace`/obs directly (they build traces — add `ped_pos`/`ped_crossing` there; see Task 9).

---

## Part C — Train, eval, render

### Task 9: Modal/eval flags + trace builders + short training run

**Files:**
- Modify: `smoothride/rl/modal_train.py` (`train` signature ~`58`, `main` ~`144`)
- Modify: `scripts/eval_policy.py` (`policy_trace` builds a `Trace` — add ped fields), `scripts/smoke_verifier.py` (same)
- Test: manual smoke (documented commands).

**Interfaces:**
- Produces: `modal_train.train(..., n_peds: int = 300, cruise_cap: float = 7.0, ped_radius: float = 3.5, cand_cap: int = 16, ...)` threaded into `make_env`. `eval_policy`/`smoke_verifier` populate `Trace.ped_pos`/`ped_crossing` from the rollout `State`.

- [ ] **Step 1: Add flags + thread through make_env**

In `modal_train.py`, add params to `train` and `main` and pass to `make_env`:

```python
def train(..., n_peds: int = 300, cruise_cap: float = 7.0,
          ped_radius: float = 3.5, cand_cap: int = 16, ...):
    ...
    env = make_env(pool, world_min, world_max, n_peds=n_peds, seed=seed,
                   cruise_cap=cruise_cap, ped_radius=ped_radius,
                   cand_cap_car=cand_cap, cand_cap_ped=cand_cap, ...)
```

- [ ] **Step 2: Update the trace builders**

In `scripts/eval_policy.py::policy_trace` and `scripts/smoke_verifier.py::rollout_to_trace`, capture `st.ped_pos`/`st.ped_crossing` per step into stacked `(T,M,2)`/`(T,M)` arrays and pass them to `Trace(...)`, and set `TraceManifest(n_peds=env.n_peds)`.

- [ ] **Step 3: Run the local smokes**

Run: `python scripts/smoke_verifier.py`
Expected: runs clean, prints a verdict including ped-aware cost.

- [ ] **Step 4: Short Modal training run (verifier-driven, dense peds, low cap)**

Run:
```bash
modal run --detach smoothride/rl/modal_train.py --iters 200 --verifier \
  --cost-target 0.1 --region downtown --n-peds 300 --cruise-cap 7.0 --tag _peds
```
Expected: `lam` ascends; `verifier_cost` trends down; `crashes/car` low; over iterations cars begin braking near crossing peds (visible later in the viewer). Save produces `trained_peds.msgpack` in the volume.

- [ ] **Step 5: Eval on a held-out region + render**

```bash
modal volume get smoothride-nav-ckpts trained_peds.msgpack runs/trained_peds.msgpack
python scripts/eval_policy.py --region mission --trained runs/trained_peds.msgpack
```
Expected report adds ped-aware behavior; confirm arrivals stay reasonable while car-ped crashes ≈ 0. Then export a scene and view in the Cesium viewer (peds already render).

- [ ] **Step 6: Commit**

```bash
git add smoothride/rl/modal_train.py scripts/eval_policy.py scripts/smoke_verifier.py
git commit -m "feat(train): peds/cruise-cap/ped-radius/cand-cap flags; trace builders log peds"
```

---

## Self-Review

**Spec coverage:**
- §1 vision / staged thesis → encoded as constraints (reward unchanged, lanes kept, cruise cap as dial). ✓
- §3 deterministic ped paths → Task 1 + Task 2. ✓
- §4 Deep Sets perception (radius-capped, masked, ped vel + crossing bit) → Task 6 + Task 7. ✓
- §5 cruise cap → Task 2. ✓
- §6 layered asymmetric radii (`r_ped`, `r_yield`) → Task 2 (env), Task 3 (cost). ✓
- §7 single-channel continuous ped-yield cost → Task 3 + Task 5. ✓
- §8 retrain + eval → Task 9. ✓
- §10 change surface — every listed file has a task. ✓

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to" — every code step has concrete code. One acknowledged soft spot: Task 7 says "preserve the existing trunk/critic if it differs" — the implementer must read the current `ActorCritic` first; the required change (DeepSets over `obs["cars"]`/`obs["peds"]`, concat ego+gf) is fully specified.

**Type consistency:** `ped_yield_cost` signature identical in Tasks 3/4/5. `DeepSets(feat_dim, hidden)` and its `(entities, mask)` call identical in Tasks 6/7. Obs dict keys (`ego/cars/cars_mask/peds/peds_mask`) identical in Tasks 7/8. `PedPaths` fields identical in Tasks 1/2. `r_yield`/`ped_radius`/`cruise_cap` names consistent env↔verifier↔ppo.

**Known integration risks (flagged for the implementer, not gaps):**
- `_global_feat` now pools a dict obs — Task 7 redefines it over `obs["ego"]`; confirm `collect` stores `gf` as the ego-shaped array.
- `make_train_state` dummy obs must match structured shapes — build via a 1-world `reset` rather than hand-zeros to avoid drift.
- Existing tests using the old flat obs / `_place_peds` / `_ped_step(env, st, key)` signature will break — update them in the same task that changes the contract (Tasks 2, 7).

---

## Execution & Parallelization

**Dependency graph** (task → what it needs):

```
1 ─┐                          (1 ped_paths.py [new])
   ├─> 2 ─┐                   (2 kinematic.py)
3 ─┼─> 4  ├─> 5 ─┐            (3 verifier.py)(4 trace.py)(5 ppo.py)
   │      │      ├─> 8 ─> 9   (8 smoke test)(9 modal/eval)
6 ─┴──────┴─> 7 ─┘            (6 networks.py)(7 kinematic+networks+ppo)
```

**File-conflict map** (which tasks edit the same file — these MUST NOT run concurrently):
- `kinematic.py`: Tasks **2, 7**
- `verifier.py`: Tasks **3, 4**
- `networks.py`: Tasks **6, 7**
- `ppo.py`: Tasks **5, 7**

Task 7 is the convergence point — it touches three shared files, so nothing that edits `kinematic.py`/`networks.py`/`ppo.py` can run alongside it.

**Wave schedule** (each wave's tasks touch disjoint files → safe to run in parallel):

| Wave | Tasks (parallel) | Why parallel-safe |
|---|---|---|
| 1 | **1, 3, 6** | disjoint files: `ped_paths.py` (new), `verifier.py`, `networks.py` |
| 2 | **2, 4** | `kinematic.py` vs `trace.py`+`conftest.py`; deps (1, 3) done |
| 3 | **5**, then **7** | both edit `ppo.py` → **sequential**; 7 also re-edits `kinematic.py`/`networks.py` |
| 4 | **8** | test-only, needs 5 + 7 |
| 5 | **9** | needs everything |

**Critical path:** `1 → 2 → 7 → 8 → 9` (5 stages). So parallelism collapses 9 serial tasks into ~5 waves — the win is front-loaded (Waves 1–2), and the integration tail (7→8→9) stays serial no matter what.

**Mechanism for safe parallel agents:** dispatch each parallel task to a subagent with **`isolation: "worktree"`** (separate git worktree per agent) and merge the diffs after the wave. Wave 1 merges trivially (three disjoint files, one brand-new). Wave 2's two diffs are disjoint too. From Wave 3 on, run single-agent in the main worktree — the files overlap and the per-task review gate matters most there.

**Recommendation:** parallelize **Wave 1 only** (best ROI — three independent units, clean merge), then execute Waves 2–5 with the standard fresh-subagent-per-task + review gate (subagent-driven-development). Parallelizing Wave 2 saves little (two quick tasks) and adds merge overhead; the tail can't be parallelized. Net: ~30–40% wall-clock saving concentrated up front, without sacrificing review discipline on the risky integration tasks.
