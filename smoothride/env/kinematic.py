"""Vectorized kinematic multi-agent driving environment (JAX) — v3 (scales).

The behavioral layer of SmoothRide. Cars follow routes on the real SF graph under
a kinematic-bicycle model and learn to coordinate, at realistic city density.

Models the actual environment:
  * per-edge SPEED LIMITS (highways fast, surface slow)
  * MULTI-LANE roads + lane-change action
  * lead-vehicle GAP -> car-following + stop-and-go
  * INTERSECTION yielding (right-of-way without traffic lights)
  * unruly PEDESTRIANS to avoid (severe penalty)
  * continuous respawn -> persistent traffic (throughput = headline metric)
  * SPATIAL-HASH neighbor search -> O(N*C), scales to thousands of cars

Pure `reset`/`step` over one world of N cars + M peds; `vmap` over worlds.
"""
from __future__ import annotations

import flax.struct as struct
import jax
import jax.numpy as jnp

from . import spatial
from .routing import RoutePool


@struct.dataclass
class Env:
    routes_xy: jnp.ndarray
    routes_n: jnp.ndarray
    routes_node: jnp.ndarray
    routes_junc: jnp.ndarray
    routes_lanes: jnp.ndarray
    routes_speed: jnp.ndarray
    routes_cum: jnp.ndarray     # (P, W) cumulative arc length (for spread spawning)
    world_min: jnp.ndarray
    world_max: jnp.ndarray

    n_agents: int = struct.field(pytree_node=False, default=24)
    n_peds: int = struct.field(pytree_node=False, default=12)
    k_neighbors: int = struct.field(pytree_node=False, default=4)
    max_steps: int = struct.field(pytree_node=False, default=300)
    cell_size: float = struct.field(pytree_node=False, default=35.0)
    cap: int = struct.field(pytree_node=False, default=16)
    ncx: int = struct.field(pytree_node=False, default=1)
    ncy: int = struct.field(pytree_node=False, default=1)
    cand_C: int = struct.field(pytree_node=False, default=144)
    spread_spawn: bool = struct.field(pytree_node=False, default=True)
    dt: float = struct.field(pytree_node=False, default=0.2)
    v_max: float = struct.field(pytree_node=False, default=16.0)
    accel_max: float = struct.field(pytree_node=False, default=3.0)
    steer_max: float = struct.field(pytree_node=False, default=0.5)
    wheelbase: float = struct.field(pytree_node=False, default=2.7)
    lane_width: float = struct.field(pytree_node=False, default=3.5)
    wp_radius: float = struct.field(pytree_node=False, default=9.0)
    # collision_radius MUST be < lane_width, else adjacent-lane cars "collide"
    collision_radius: float = struct.field(pytree_node=False, default=2.2)
    ped_radius: float = struct.field(pytree_node=False, default=2.2)
    ped_speed: float = struct.field(pytree_node=False, default=1.4)
    prox_radius: float = struct.field(pytree_node=False, default=6.0)
    lead_cone: float = struct.field(pytree_node=False, default=30.0)
    junc_zone: float = struct.field(pytree_node=False, default=14.0)
    idle_speed: float = struct.field(pytree_node=False, default=0.5)
    w_progress: float = struct.field(pytree_node=False, default=1.0)
    w_idle: float = struct.field(pytree_node=False, default=0.05)
    w_prox: float = struct.field(pytree_node=False, default=1.0)
    w_collision: float = struct.field(pytree_node=False, default=40.0)
    w_ped_prox: float = struct.field(pytree_node=False, default=2.0)
    w_ped: float = struct.field(pytree_node=False, default=80.0)
    w_yield: float = struct.field(pytree_node=False, default=1.5)
    w_goal: float = struct.field(pytree_node=False, default=15.0)
    # traffic-law shaping (off by default; turned on to train LAWFUL driving):
    #   w_offlane  -> continuous penalty ramping up as the car leaves its lane
    #   w_wrongway -> penalty for heading against the route while moving
    w_offlane: float = struct.field(pytree_node=False, default=0.0)
    w_wrongway: float = struct.field(pytree_node=False, default=0.0)

    @property
    def obs_dim(self) -> int:
        return 6 + 1 + self.k_neighbors * 4 + 3

    @property
    def act_dim(self) -> int:
        return 3


@struct.dataclass
class State:
    pos: jnp.ndarray
    heading: jnp.ndarray
    speed: jnp.ndarray
    route_idx: jnp.ndarray
    wp_ptr: jnp.ndarray
    lane: jnp.ndarray
    just_crashed: jnp.ndarray  # (N,) bool: collided THIS step (then respawns)
    crashes: jnp.ndarray       # (N,) int: cumulative collisions
    spawn_grace: jnp.ndarray   # (N,) int: countdown of merge-in immunity steps
    goals: jnp.ndarray
    ped_pos: jnp.ndarray
    ped_dir: jnp.ndarray
    t: jnp.ndarray


def _wrap(a):
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


def _route_dir(env: Env, route_idx, wp_ptr):
    n = env.routes_n[route_idx]
    cur = env.routes_xy[route_idx, wp_ptr]
    nxt = env.routes_xy[route_idx, jnp.minimum(wp_ptr + 1, n - 1)]
    d = nxt - cur
    return d / (jnp.linalg.norm(d, axis=-1, keepdims=True) + 1e-6), cur


def _target_wp(env: Env, st: State) -> jnp.ndarray:
    dn, cur = _route_dir(env, st.route_idx, st.wp_ptr)
    right = jnp.stack([dn[..., 1], -dn[..., 0]], axis=-1)
    offset = env.lane_width * (st.lane.astype(jnp.float32) + 0.5)
    return cur + right * offset[..., None]


def _spawn(env: Env, idx):
    dn, cur = _route_dir(env, idx, jnp.ones_like(idx))
    return cur, jnp.arctan2(dn[..., 1], dn[..., 0])


def _spread_spawn(env: Env, idx, key):
    """Spawn at a random fraction ALONG the route -> spread across the network."""
    nwp = env.routes_n[idx]
    frac = jax.random.uniform(key, idx.shape)
    wp = jnp.clip((frac * (nwp - 1)).astype(jnp.int32), 1, jnp.maximum(nwp - 1, 1))
    dn, cur = _route_dir(env, idx, wp)
    return cur, jnp.arctan2(dn[..., 1], dn[..., 0]), wp


def _entry_spawn(env: Env, idx, key):
    """spread_spawn=True: spread along route (city). False: enter at the route
    START (boundary edge) and drive THROUGH the junction (scenario windows)."""
    if env.spread_spawn:
        return _spread_spawn(env, idx, key)
    dn, cur = _route_dir(env, idx, jnp.ones_like(idx))
    return cur, jnp.arctan2(dn[..., 1], dn[..., 0]), jnp.ones_like(idx, jnp.int32)


def _candidates(env: Env, pos):
    return spatial.grid_candidates(pos, env.world_min, env.cell_size,
                                   env.ncx, env.ncy, env.cap, env.cand_C)


def _observe(env: Env, st: State, cand) -> jnp.ndarray:
    N = env.n_agents
    vmax = jnp.minimum(env.v_max, env.routes_speed[st.route_idx, st.wp_ptr])
    tgt = _target_wp(env, st)
    to_wp = tgt - st.pos
    dist = jnp.linalg.norm(to_wp, axis=-1)
    herr = _wrap(jnp.arctan2(to_wp[:, 1], to_wp[:, 0]) - st.heading)
    n = env.routes_n[st.route_idx]
    progress = st.wp_ptr / jnp.maximum(n - 1, 1)
    fdir = jnp.stack([jnp.cos(st.heading), jnp.sin(st.heading)], -1)

    # candidate relative geometry (N, C, 2)
    valid = (cand >= 0) & (cand != jnp.arange(N)[:, None])
    rel = st.pos[cand] - st.pos[:, None, :]
    cd = jnp.where(valid, jnp.linalg.norm(rel, axis=-1), 1e9)
    forward = jnp.sum(rel * fdir[:, None, :], -1)
    lateral = jnp.abs(rel[..., 0] * fdir[:, None, 1] - rel[..., 1] * fdir[:, None, 0])
    ahead = valid & (forward > 0.5) & (lateral < env.lane_width) & (forward < env.lead_cone)
    lead_gap = jnp.min(jnp.where(ahead, forward, env.lead_cone), axis=-1)

    lane_frac = st.lane.astype(jnp.float32) / jnp.maximum(
        env.routes_lanes[st.route_idx, st.wp_ptr] - 1, 1)
    ego = jnp.stack([
        st.speed / jnp.maximum(vmax, 1.0),
        jnp.sin(herr), jnp.cos(herr),
        jnp.clip(dist / 100.0, 0, 1), progress,
        jnp.clip(lead_gap / env.lead_cone, 0, 1),
    ], axis=-1)

    # K nearest among candidates (ego frame)
    _, kk = jax.lax.top_k(-cd, env.k_neighbors)             # (N,K) into candidate axis
    nbr = jnp.take_along_axis(cand, kk, axis=1)             # (N,K) agent idx
    nbr_valid = jnp.take_along_axis(valid, kk, axis=1)
    c, s = jnp.cos(-st.heading), jnp.sin(-st.heading)
    nrel = st.pos[nbr] - st.pos[:, None, :]
    nx = (nrel[..., 0] * c[:, None] - nrel[..., 1] * s[:, None]) * nbr_valid
    ny = (nrel[..., 0] * s[:, None] + nrel[..., 1] * c[:, None]) * nbr_valid
    vel = st.speed[:, None] * fdir
    nvel = (vel[nbr] - vel[:, None, :]) * nbr_valid[..., None]
    nbr_feat = jnp.concatenate([
        jnp.clip(nx[..., None] / 50, -1, 1),
        jnp.clip(ny[..., None] / 50, -1, 1),
        jnp.clip(nvel / env.v_max, -1, 1),
    ], -1).reshape(N, -1)

    # nearest pedestrian (peds are few -> brute force)
    pd = st.pos[:, None, :] - st.ped_pos[None, :, :]
    pdist = jnp.linalg.norm(pd, axis=-1)
    pj = jnp.argmin(pdist, axis=-1)
    prel = st.ped_pos[pj] - st.pos
    px = prel[:, 0] * c - prel[:, 1] * s
    py = prel[:, 0] * s + prel[:, 1] * c
    pmin = pdist[jnp.arange(N), pj]
    ped_feat = jnp.stack([jnp.clip(px / 50, -1, 1), jnp.clip(py / 50, -1, 1),
                          jnp.clip(pmin / 50, 0, 1)], -1)

    return jnp.concatenate([ego, lane_frac[:, None], nbr_feat, ped_feat], -1)


def _ped_step(env: Env, st: State, key):
    kd, kj = jax.random.split(key)
    turn = jax.random.normal(kd, (env.n_peds,)) * 0.3
    dart = (jax.random.uniform(kj, (env.n_peds,)) < 0.02).astype(jnp.float32)
    ped_dir = _wrap(st.ped_dir + turn + dart * jax.random.normal(kd, (env.n_peds,)))
    step = env.ped_speed * env.dt * jnp.stack([jnp.cos(ped_dir), jnp.sin(ped_dir)], -1)
    ped_pos = jnp.clip(st.ped_pos + step, env.world_min, env.world_max)
    flip = (st.ped_pos + step < env.world_min) | (st.ped_pos + step > env.world_max)
    ped_dir = jnp.where(flip[:, 0], jnp.pi - ped_dir, ped_dir)
    ped_dir = jnp.where(flip[:, 1], -ped_dir, ped_dir)
    return ped_pos, _wrap(ped_dir)


def reset(env: Env, key: jax.Array):
    kr, kp, kpd, kf = jax.random.split(key, 4)
    n = env.n_agents
    route_idx = jax.random.randint(kr, (n,), 0, env.routes_xy.shape[0])
    pos, heading, wp = _entry_spawn(env, route_idx, kf)
    st = State(
        pos=pos, heading=heading, speed=jnp.zeros(n),
        route_idx=route_idx, wp_ptr=wp, lane=jnp.zeros(n, jnp.int32),
        just_crashed=jnp.zeros(n, bool), crashes=jnp.zeros(n, jnp.int32),
        spawn_grace=jnp.zeros(n, jnp.int32), goals=jnp.zeros(n, jnp.int32),
        ped_pos=jax.random.uniform(kp, (env.n_peds, 2),
                                   minval=env.world_min, maxval=env.world_max),
        ped_dir=jax.random.uniform(kpd, (env.n_peds,), minval=-jnp.pi, maxval=jnp.pi),
        t=jnp.array(0, jnp.int32),
    )
    return st, _observe(env, st, _candidates(env, st.pos))


def step(env: Env, st: State, action: jnp.ndarray, key: jax.Array):
    N = env.n_agents
    kidx, kfrac, kped = jax.random.split(key, 3)

    accel = jnp.clip(action[:, 0], -1, 1) * env.accel_max
    delta = jnp.clip(action[:, 1], -1, 1) * env.steer_max
    lane_cmd = action[:, 2]

    vmax = jnp.minimum(env.v_max, env.routes_speed[st.route_idx, st.wp_ptr])
    speed = jnp.clip(st.speed + accel * env.dt, 0.0, vmax)
    heading = _wrap(st.heading + (speed / env.wheelbase) * jnp.tan(delta) * env.dt)
    pos = st.pos + speed[:, None] * jnp.stack([jnp.cos(heading), jnp.sin(heading)], -1) * env.dt

    L = env.routes_lanes[st.route_idx, st.wp_ptr]
    shift = jnp.where(lane_cmd > 0.5, 1, jnp.where(lane_cmd < -0.5, -1, 0))
    lane = jnp.clip(st.lane + shift, 0, L - 1)

    nst0 = st.replace(pos=pos, heading=heading, lane=lane)
    tgt = _target_wp(env, nst0)
    to_wp = tgt - pos
    dist = jnp.linalg.norm(to_wp, axis=-1)
    herr = _wrap(jnp.arctan2(to_wp[:, 1], to_wp[:, 0]) - heading)
    progress = speed * jnp.cos(herr) * env.dt

    n_wp = env.routes_n[st.route_idx]
    hit = dist < env.wp_radius
    new_goal = hit & (st.wp_ptr >= n_wp - 1)
    advance = hit & (st.wp_ptr < n_wp - 1)
    wp_ptr = jnp.where(advance, st.wp_ptr + 1, st.wp_ptr)

    # spatial-hash neighbor reductions
    cand = _candidates(env, pos)
    valid = (cand >= 0) & (cand != jnp.arange(N)[:, None])
    rel = pos[cand] - pos[:, None, :]
    # spawn_grace: a car that just entered ("merged in") is immune for a few steps.
    # A just-spawned car is also EXCLUDED as a collision partner, so dropping a fresh
    # car near a moving one never registers a (spurious) crash for EITHER side — only
    # genuine car-to-car contact between two established cars counts.
    immune = st.spawn_grace > 0
    neighbor_ok = valid & ~immune[cand]
    cd = jnp.where(neighbor_ok, jnp.linalg.norm(rel, axis=-1), 1e9)
    min_d = cd.min(1)
    car_crash = (min_d < env.collision_radius) & ~immune
    prox_pen = jnp.clip((env.prox_radius - min_d) / env.prox_radius, 0, 1)

    # pedestrians (severe)
    pd = jnp.linalg.norm(pos[:, None, :] - st.ped_pos[None, :, :], axis=-1)
    ped_min = pd.min(axis=-1)
    ped_hit = (ped_min < env.ped_radius) & ~immune
    ped_prox = jnp.clip((env.prox_radius - ped_min) / env.prox_radius, 0, 1)
    crash_event = car_crash | ped_hit

    # intersection yielding over candidates (same junction node, closer car first)
    cur_node = env.routes_node[st.route_idx, st.wp_ptr]
    is_junc = env.routes_junc[st.route_idx, st.wp_ptr]
    near_junc = is_junc & (dist < env.junc_zone)
    same = valid & (cur_node[cand] == cur_node[:, None]) & (cur_node[:, None] >= 0)
    has_priority = same & near_junc[:, None] & near_junc[cand] & (dist[cand] < dist[:, None])
    should_yield = near_junc & jnp.any(has_priority, axis=1)
    yield_pen = (should_yield & (speed > env.idle_speed)).astype(jnp.float32)

    # traffic-law shaping: penalize leaving the lane / driving the wrong way, on the
    # post-dynamics (pre-respawn) state. Lazy import avoids a kinematic<->legality
    # import cycle. off-lane is CONTINUOUS (ramps from half-a-lane off to fully off)
    # so the policy gets a dense "stay in lane" gradient instead of a sparse flag.
    from . import legality as _LG
    law_st = st.replace(pos=pos, heading=heading, speed=speed, wp_ptr=wp_ptr)
    law = _LG.evaluate(env, law_st)
    offlane_pen = jnp.clip((law["lateral"] - 0.5 * env.lane_width) / env.lane_width,
                           0.0, 1.0)
    wrongway_pen = law["wrong_way"].astype(jnp.float32)

    # respawn cars that finished a trip OR crashed (a crash clears, not freezes)
    respawn = new_goal | crash_event
    new_idx = jax.random.randint(kidx, (N,), 0, env.routes_xy.shape[0])
    rs_pos, rs_head, rs_wp = _entry_spawn(env, new_idx, kfrac)
    route_idx = jnp.where(respawn, new_idx, st.route_idx)
    pos = jnp.where(respawn[:, None], rs_pos, pos)
    heading = jnp.where(respawn, rs_head, heading)
    wp_ptr = jnp.where(respawn, rs_wp, wp_ptr)
    lane = jnp.where(respawn, 0, lane)
    # merge INTO traffic at a sensible cruising speed instead of appearing as a dead
    # stop in a moving lane (which got the fresh car rear-ended the moment its grace
    # ended). Speed is capped by the spawn edge's limit.
    merge_speed = 0.6 * jnp.minimum(env.v_max, env.routes_speed[new_idx, rs_wp])
    speed = jnp.where(respawn, merge_speed, speed)
    goals = st.goals + new_goal.astype(jnp.int32)
    crashes = st.crashes + crash_event.astype(jnp.int32)
    ped_pos, ped_dir = _ped_step(env, st, kped)

    idle_pen = (speed < env.idle_speed)
    reward = (
        env.w_progress * progress
        - env.w_idle * idle_pen.astype(jnp.float32)
        - env.w_prox * prox_pen - env.w_ped_prox * ped_prox
        - env.w_yield * yield_pen
        - env.w_collision * car_crash.astype(jnp.float32)
        - env.w_ped * ped_hit.astype(jnp.float32)
        - env.w_offlane * offlane_pen - env.w_wrongway * wrongway_pen
        + env.w_goal * new_goal.astype(jnp.float32)
    )

    t = st.t + 1
    nst = State(pos=pos, heading=heading, speed=speed, route_idx=route_idx,
                wp_ptr=wp_ptr, lane=lane, just_crashed=crash_event, crashes=crashes,
                spawn_grace=jnp.where(respawn, 4, jnp.maximum(st.spawn_grace - 1, 0)),
                goals=goals, ped_pos=ped_pos, ped_dir=ped_dir, t=t)
    info = {"just_crashed": crash_event, "crashes": crashes,
            "goals": goals, "total_goals": goals.sum(),
            "crashes_per_car": crashes.mean(), "ped_hits": ped_hit.sum(),
            "mean_speed": jnp.mean(speed),
            "offlane": law["off_lane"].astype(jnp.float32),     # (N,) 1=off-lane
            "wrongway": wrongway_pen}                            # (N,) 1=wrong-way
    return nst, _observe(env, nst, _candidates(env, pos)), reward, t >= env.max_steps, info


def make_env(pool: RoutePool, world_min, world_max, cell_size=35.0, cap=16, **kw) -> Env:
    import numpy as np
    seg = np.linalg.norm(np.diff(pool.xy, axis=1), axis=-1)          # (P, W-1)
    cum = np.concatenate([np.zeros((pool.xy.shape[0], 1)), np.cumsum(seg, axis=1)], 1)
    ncx, ncy = spatial.grid_dims(world_min, world_max, cell_size)
    return Env(
        routes_xy=jnp.asarray(pool.xy), routes_n=jnp.asarray(pool.n),
        routes_node=jnp.asarray(pool.node), routes_junc=jnp.asarray(pool.junc),
        routes_lanes=jnp.asarray(pool.lanes), routes_speed=jnp.asarray(pool.speed),
        routes_cum=jnp.asarray(cum, jnp.float32),
        world_min=jnp.asarray(world_min, jnp.float32),
        world_max=jnp.asarray(world_max, jnp.float32),
        cell_size=cell_size, cap=cap, ncx=ncx, ncy=ncy,
        cand_C=spatial.candidate_count(cap), **kw,
    )
