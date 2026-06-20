"""Local MAPPO training on the kinematic SF env. Saves checkpoints + metrics.

Produces, by design, the demo's core artifact: an UNTRAINED baseline checkpoint
and a TRAINED checkpoint, so the renderer can show the learning delta.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import jax
from flax import serialization

from ..data.map_loader import load_road_network
from ..env import kinematic as K
from ..env.routing import build_route_pool
from . import ppo

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "runs"))


def save_params(ts, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(ts.params))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--worlds", type=int, default=32)
    ap.add_argument("--agents", type=int, default=24)
    ap.add_argument("--peds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--vmax", type=float, default=16.0)
    ap.add_argument("--routes", type=int, default=1024)
    ap.add_argument("--big", action="store_true",
                    help="train on the huge SF map (highways) instead of downtown")
    ap.add_argument("--tag", default="", help="suffix for checkpoint names")
    ap.add_argument("--save-every", type=int, default=0, dest="save_every")
    ap.add_argument("--lagrangian", action="store_true",
                    help="PPO-Lagrangian: adaptive crash-constraint multiplier")
    ap.add_argument("--crash-target", type=float, default=0.3, dest="crash_target")
    ap.add_argument("--w-offlane", type=float, default=0.0, dest="w_offlane",
                    help="reward penalty for leaving the lane (train lawful driving)")
    ap.add_argument("--w-wrongway", type=float, default=0.0, dest="w_wrongway",
                    help="reward penalty for driving against the route direction")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.big:
        from ..data.map_loader import load_sf_graph, to_road_network
        BIG = (-122.4300, 37.7250, -122.3800, 37.8050)
        net = to_road_network(load_sf_graph(bbox=BIG, cache_name="sf_huge_drive.graphml"))
    else:
        net = load_road_network()
    x0, y0, x1, y1 = net.bounds()
    pool = build_route_pool(net, n_routes=args.routes,
                            max_length_m=2500.0 if args.big else 700.0, seed=args.seed)
    # Lagrangian: zero the fixed crash penalty so the adaptive multiplier owns it
    extra = {"w_collision": 0.0} if args.lagrangian else {}
    # traffic-law shaping weights (0 => identical to legacy training)
    extra.update(w_offlane=args.w_offlane, w_wrongway=args.w_wrongway)
    env = K.make_env(pool, (x0, y0), (x1, y1), n_agents=args.agents,
                     n_peds=args.peds, max_steps=args.steps, v_max=args.vmax, **extra)
    print(f"map: nodes={net.n_nodes} edges={net.n_edges}")
    cfg = ppo.PPOConfig(n_worlds=args.worlds)
    print(f"env: agents={env.n_agents} obs={env.obs_dim} steps={env.max_steps} "
          f"worlds={cfg.n_worlds}")

    key = jax.random.PRNGKey(args.seed)
    key, kinit = jax.random.split(key)
    ts = ppo.make_train_state(env, cfg, kinit)

    tag = args.tag
    # baseline checkpoint BEFORE any learning -> the "today's traffic" shadow world
    save_params(ts, os.path.join(OUT, f"untrained{tag}.msgpack"))

    history = []
    lam = 10.0  # Lagrange multiplier on the crash constraint
    for it in range(args.iters):
        key, kc = jax.random.split(key)
        t0 = time.time()
        batch = ppo.collect(env, ts, kc, cfg.n_worlds)
        ts, m = ppo.update(env, cfg, ts, batch, lam if args.lagrangian else 0.0)
        m = {k: float(v) for k, v in m.items()}
        m["iter"], m["sec"] = it, round(time.time() - t0, 2)
        if args.lagrangian:  # dual ascent toward the crash target
            lam = min(400.0, max(0.0, lam + 3.0 * (m["crashes_per_car"] - args.crash_target)))
            m["lam"] = round(lam, 1)
        history.append(m)
        lam_s = f"lam {lam:6.1f} | " if args.lagrangian else ""
        law_s = (f"offlane {m['offlane_rate']*100:4.1f}% | "
                 f"wrongway {m['wrongway_rate']*100:4.1f}% | ")
        print(f"it {it:3d} | reward {m['ep_reward']:8.1f} | {lam_s}{law_s}"
              f"crashes/car {m['crashes_per_car']:.2f} | "
              f"goals/agent {m['goals_per_agent']:.2f} | {m['sec']}s", flush=True)
        # periodic checkpoint so artifacts can be rendered mid-training
        if args.save_every and it and it % args.save_every == 0:
            save_params(ts, os.path.join(OUT, f"trained{tag}.msgpack"))
            with open(os.path.join(OUT, f"history{tag}.json"), "w") as f:
                json.dump(history, f, indent=2)

    save_params(ts, os.path.join(OUT, f"trained{tag}.msgpack"))
    with open(os.path.join(OUT, f"history{tag}.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nsaved untrained{tag}.msgpack, trained{tag}.msgpack -> {OUT}")


if __name__ == "__main__":
    main()
