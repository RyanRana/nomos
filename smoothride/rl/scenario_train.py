"""Scenario-curriculum trainer: train ONE shared MAPPO policy across many SF
edge-case junction windows IN PARALLEL, with the runtime safety shield active
during rollouts (shielded RL). The policy learns to drive each topology safely;
the shield catches the residual. Saves a policy that, with the shield, gets low
crashes on the hard scenarios.

Parallel = the K scenario windows are stacked and vmapped as the batch dimension,
so every gradient step sees all edge cases at once.
"""
from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from ..data.map_loader import load_sf_graph, to_road_network
from ..data.scenarios import BIG_BBOX, mine, pick_representatives, window_net
from ..env import kinematic as K
from ..env.routing import RoutePool, build_route_pool
from . import ppo
from .networks import squash_sample
from .safety import safe_action

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "runs"))
# scenario types to train on (the edge cases)
KINDS = ["four_way", "three_way", "ramp_merge", "highway_uturn", "bridge", "complex"]


def _pad_pool(p: RoutePool, P: int, W: int) -> RoutePool:
    def padW(a, fill):
        out = np.full((a.shape[0], W) + a.shape[2:], fill, a.dtype)
        out[:, :a.shape[1]] = a[:, :W]
        return out
    xy = padW(p.xy, 0.0); node = padW(p.node, 0); junc = padW(p.junc, False)
    lanes = padW(p.lanes, 1); speed = padW(p.speed, 11.0); n = np.minimum(p.n, W)
    # tail-pad each route by repeating its last valid waypoint
    for i in range(xy.shape[0]):
        k = max(int(n[i]), 1)
        xy[i, k:] = xy[i, k - 1]; node[i, k:] = node[i, k - 1]
    def padP(a):
        if a.shape[0] >= P:
            return a[:P]
        reps = (P + a.shape[0] - 1) // a.shape[0]
        return np.concatenate([a] * reps, 0)[:P]
    return RoutePool(xy=padP(xy), n=padP(n), node=padP(node), junc=padP(junc),
                     lanes=padP(lanes), speed=padP(speed))


def build_scenario_envs(net, reps, half, n_agents, n_peds, steps, vmax, n_routes):
    pools, sizes, kinds = [], [], []
    for kind in KINDS:
        for s in reps.get(kind, [])[:2]:               # up to 2 windows per kind
            wn = window_net(net, s["x"], s["y"], half=half)
            if wn is None:
                continue
            wnet, size = wn
            try:
                pool = build_route_pool(wnet, n_routes=n_routes, min_waypoints=4,
                                        max_length_m=None, seed=0)
            except RuntimeError:
                continue
            pools.append(pool); sizes.append(size); kinds.append(kind)
    if not pools:
        raise RuntimeError("no scenario windows could be built")
    P = max(p.xy.shape[0] for p in pools)
    W = max(p.xy.shape[1] for p in pools)
    envs = []
    for pool, size in zip(pools, sizes):
        # spread_spawn=False -> cars ENTER at boundary edges and drive THROUGH the
        # junction (realistic load); w_collision=0 -> Lagrangian owns the crash cost.
        env = K.make_env(_pad_pool(pool, P, W), (0.0, 0.0), size,
                         n_agents=n_agents, n_peds=n_peds, max_steps=steps, v_max=vmax,
                         spread_spawn=False, w_collision=0.0)
        envs.append(env)
    batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *envs)
    return batched, envs, kinds


def collect_scenarios(env_b, ts, key, shield=True):
    def one(env, wkey):
        def step_fn(carry, k):
            st, obs = carry
            gf = ppo._global_feat(obs)
            mean, log_std, value = ts.apply_fn(ts.params, obs, gf)
            ka, kn = jax.random.split(k)
            noise = jax.random.normal(ka, mean.shape)
            # Tanh-squashed sample (action in (-1,1)) + change-of-variables logp.
            action, logp = squash_sample(mean, log_std, noise)
            exec_a = safe_action(env, st, action) if shield else action
            nst, nobs, r, done, info = K.step(env, st, exec_a, kn)
            return (nst, nobs), dict(obs=obs, gf=gf, action=action, logp=logp,
                                     value=value, reward=r,
                                     cost=info["just_crashed"].astype(jnp.float32))
        kr, ks = jax.random.split(wkey)
        st, obs = K.reset(env, kr)
        (lst, lobs), traj = jax.lax.scan(step_fn, (st, obs),
                                         jax.random.split(ks, env.max_steps))
        _, _, lv = ts.apply_fn(ts.params, lobs, ppo._global_feat(lobs))
        traj["last_value"] = lv
        traj["final_crashes"] = lst.crashes
        traj["final_goals"] = lst.goals
        return traj
    return jax.vmap(one)(env_b, jax.random.split(key, _n_scenarios(env_b)))


def _n_scenarios(env_b):
    return env_b.routes_xy.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--peds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=220)
    ap.add_argument("--half", type=float, default=200.0)
    ap.add_argument("--vmax", type=float, default=18.0)
    ap.add_argument("--crash-target", type=float, default=0.4, dest="crash_target")
    ap.add_argument("--init", default=os.path.join(OUT, "trained_lag.msgpack"),
                    help="warm-start policy")
    ap.add_argument("--tag", default="_scn")
    args = ap.parse_args()

    G = load_sf_graph(bbox=BIG_BBOX, cache_name="sf_huge_drive.graphml")
    net = to_road_network(G)
    scn = mine(G, net)
    reps = pick_representatives(scn, per_kind=2, seed=1)
    env_b, envs, kinds = build_scenario_envs(
        net, reps, args.half, args.agents, args.peds, args.steps, args.vmax, 256)
    nS = len(kinds)
    print(f"scenarios (parallel worlds): {nS} -> {kinds}")

    cfg = ppo.PPOConfig(n_worlds=nS)
    key = jax.random.PRNGKey(0)
    ts = ppo.make_train_state(envs[0], cfg, jax.random.PRNGKey(1))
    if args.init and os.path.exists(args.init):
        with open(args.init, "rb") as f:
            ts = ts.replace(params=serialization.from_bytes(ts.params, f.read()))
        print(f"warm-started from {os.path.basename(args.init)}")

    lam = 10.0  # Lagrangian crash-constraint multiplier
    for it in range(args.iters):
        key, kc = jax.random.split(key)
        batch = collect_scenarios(env_b, ts, kc, shield=False)
        ts, m = ppo.update(envs[0], cfg, ts, batch, lam)
        cpc = float(m["crashes_per_car"])
        lam = min(400.0, max(0.0, lam + 3.0 * (cpc - args.crash_target)))
        if it % 10 == 0 or it == args.iters - 1:
            print(f"it {it:3d} | reward {float(m['ep_reward']):7.1f} | lam {lam:5.0f} | "
                  f"crashes/car {cpc:.2f} | goals/agent {float(m['goals_per_agent']):.2f}")

    path = os.path.join(OUT, f"trained{args.tag}.msgpack")
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(ts.params))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
