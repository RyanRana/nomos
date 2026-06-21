"""Held-out evaluation — does the policy GENERALIZE, or did it memorize the map?

Drive a trained policy on a DIFFERENT part of San Francisco than it was trained on
(training = the downtown box, DOWNTOWN_SF_BBOX) and grade it with the deterministic
verifier. The policy is decentralized (local 26-dim obs only), so it can drive any
road network — this checks it actually does. Also evaluates the untrained checkpoint
on the same region as a baseline, so the numbers mean something.

    python scripts/eval_policy.py                       # default held-out region
    python scripts/eval_policy.py --bbox W S E N --agents 60 --steps 300

Network call on first run (downloads + caches the held-out OSM graph). No GPU.
"""
from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from smoothride.data.map_loader import DOWNTOWN_SF_BBOX, load_road_network
from smoothride.demo.render import load_params
from smoothride.env import kinematic as K
from smoothride.env.routing import build_route_pool
from smoothride.rl.networks import ActorCritic
from smoothride.rl.trace import Trace, TraceManifest
from smoothride.rl.verifier import _lane_flags, verify

# A held-out SF region (Western Addition / Alamo Square / NoPa) — well clear of the
# downtown training box, same dense grid character.
HELDOUT_BBOX = (-122.4450, 37.7680, -122.4270, 37.7810)


def policy_trace(env: K.Env, params, key, n_steps: int, sample: bool = False) -> Trace:
    """Roll out the trained policy (deterministic mean action by default) -> Trace."""
    net = ActorCritic(act_dim=env.act_dim)
    rxy = np.asarray(env.routes_xy)
    rlanes = np.asarray(env.routes_lanes)
    rspeed = np.asarray(env.routes_speed)
    step = jax.jit(lambda s, a, k: K.step(env, s, a, k))

    @jax.jit
    def act(obs):
        gf = jnp.broadcast_to(obs.mean(-2, keepdims=True), obs.shape)
        mean, log_std, _ = net.apply(params, obs, gf)
        return mean, log_std

    key, kr = jax.random.split(key)
    st, obs = K.reset(env, kr)
    recs: list[dict] = []
    for _ in range(n_steps):
        mean, log_std = act(obs)
        key, ka, ks = jax.random.split(key, 3)
        action = mean if not sample else mean + jnp.exp(log_std) * jax.random.normal(ka, mean.shape)
        ri, wp = np.asarray(st.route_idx), np.asarray(st.wp_ptr)
        nst, nobs, _, _, info = step(st, action, ks)
        recs.append(dict(
            pos=np.asarray(st.pos, np.float32), heading=np.asarray(st.heading, np.float32),
            speed=np.asarray(st.speed, np.float32), lane=np.asarray(st.lane, np.int32),
            action=np.asarray(action, np.float32), wp_ptr=wp.astype(np.int32),
            seg_start=rxy[ri, np.maximum(wp - 1, 0)].astype(np.float32),
            seg_end=rxy[ri, wp].astype(np.float32), lane_count=rlanes[ri, wp].astype(np.int32),
            spawn_grace=np.asarray(st.spawn_grace, np.int32),
            crashed=np.asarray(info["just_crashed"], bool),
            arrived=np.asarray(info["arrived"], bool),
            speed_limit=rspeed[ri, wp].astype(np.float32),
        ))
        st, obs = nst, nobs

    stack = lambda f: np.stack([r[f] for r in recs])
    T, N = n_steps, env.n_agents
    manifest = TraceManifest(
        run_id="heldout-eval", seed=0, scenario_id="heldout", policy_checkpoint_id="trained",
        config_hash="eval", dt=float(env.dt), n_steps=T, n_agents=N, n_peds=env.n_peds)
    return Trace(
        manifest=manifest, pos=stack("pos"), z=np.zeros((T, N), np.float32),
        heading=stack("heading"), speed=stack("speed"), lane=stack("lane"),
        action=stack("action"), wp_ptr=stack("wp_ptr"),
        dist_remaining=np.zeros((T, N), np.float32),
        seg_start=stack("seg_start"), seg_end=stack("seg_end"),
        lane_count=stack("lane_count"), spawn_grace=stack("spawn_grace"),
        crashed=stack("crashed"), arrived=stack("arrived"), speed_limit=stack("speed_limit"),
        collision_radius=float(env.collision_radius), lane_width=float(env.lane_width))


def report(name: str, v, n: int, trace: Trace) -> None:
    valid_cars = sum(c.valid for c in v.per_car)
    # per-STEP rates over active (not-yet-arrived) car-steps, so a single brief
    # excursion doesn't read like chronic violation the way the any-step counts do.
    lat, off, ww = _lane_flags(trace.pos, trace.seg_start, trace.seg_end, trace.lane_count,
                               trace.lane_width, trace.heading, trace.speed, trace.spawn_grace)
    active = ~trace.arrived
    denom = max(int(active.sum()), 1)
    print(f"\n[{name}]")
    print(f"  arrivals (throughput) : {v.throughput}/{n}  ({100*v.throughput/n:.0f}%)   "
          f"mean travel time: {v.mean_travel_time:.1f}s")
    print(f"  crashes               : {v.crash_count}/{n}  ({100*v.crash_count/n:.0f}%)")
    print(f"  ANY-STEP counts (strict / safety-gate): off-lane {v.off_lane_count}/{n}  "
          f"wrong-way {v.wrong_way_count}/{n}  over-speed {v.speed_violation_count}/{n}")
    print(f"  PER-STEP rates (policy quality): off-lane {100*(off&active).sum()/denom:.1f}%  "
          f"wrong-way {100*(ww&active).sum()/denom:.1f}%  mean lateral {lat[active].mean():.2f}m "
          f"(lane {trace.lane_width}m)")
    print(f"  fully-valid (no violation any step): {valid_cars}/{n}  valid_run={v.valid_run}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", type=float, nargs=4, default=list(HELDOUT_BBOX),
                    metavar=("W", "S", "E", "N"))
    ap.add_argument("--agents", type=int, default=60)
    ap.add_argument("--peds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trained", default="runs/trained.msgpack")
    ap.add_argument("--untrained", default="runs/untrained.msgpack")
    a = ap.parse_args()

    bbox = tuple(a.bbox)
    print(f"TRAINED on downtown box {DOWNTOWN_SF_BBOX}")
    print(f"EVALUATING on HELD-OUT box {bbox}")
    net = load_road_network(bbox=bbox)
    x0, y0, x1, y1 = net.bounds()
    pool = build_route_pool(net, n_routes=512, seed=a.seed)
    env = K.make_env(pool, (x0, y0), (x1, y1), n_agents=a.agents, n_peds=a.peds, max_steps=a.steps)
    print(f"held-out map: {net.n_nodes} nodes, {len(env.routes_xy)} routes | "
          f"env: agents={a.agents} obs={env.obs_dim} steps={a.steps}")

    for name, ckpt in [("untrained (baseline)", a.untrained), ("TRAINED policy", a.trained)]:
        params = load_params(env, ckpt)
        tr = policy_trace(env, params, jax.random.PRNGKey(a.seed), a.steps, sample=False)
        report(name, verify(tr), a.agents, tr)


if __name__ == "__main__":
    main()
