"""Render rollouts to video (GIF) + screenshots over the real SF road network.

Loads a checkpoint, replays one world, and draws cars on the OSM streets:
  blue = driving, red = crashed, green = reached goal.
This is the artifact pipeline: every checkpoint -> a video + stills.
"""
from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from flax import serialization

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from ..data.map_loader import RoadNetwork, load_road_network  # noqa: E402
from ..env import kinematic as K  # noqa: E402
from ..env.routing import build_route_pool  # noqa: E402
from ..rl import ppo  # noqa: E402

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "runs"))


def load_params(env, path):
    ts = ppo.make_train_state(env, ppo.PPOConfig(), jax.random.PRNGKey(0))
    with open(path, "rb") as f:
        return serialization.from_bytes(ts.params, f.read())


def rollout(env: K.Env, params, key, sample=True, safe=False, filt="vo"):
    """Replay one world; return numpy (T,N,...) trajectories.

    safe=True wraps each action with a runtime safety filter: filt='vo' (velocity
    obstacle / ORCA) or filt='cbf' (higher-order CBF-QP)."""
    from ..rl.networks import ActorCritic
    from ..rl.ppo import _global_feat
    from ..rl.safety import safe_action
    from ..rl.cbf import cbf_action
    filter_fn = cbf_action if filt == "cbf" else safe_action
    net = ActorCritic(act_dim=env.act_dim)

    def step_fn(carry, k):
        st, obs = carry
        gf = _global_feat(obs)
        mean, log_std, _ = net.apply(params, obs, gf)
        ka, kn = jax.random.split(k)
        # Tanh-squashed policy: squash the (optionally sampled) pre-tanh value so
        # the rendered actions match the distribution the policy was trained on.
        raw = mean + (jnp.exp(log_std) * jax.random.normal(ka, mean.shape)
                      if sample else 0.0)
        action = jnp.tanh(raw)
        if safe:
            action = filter_fn(env, st, action)
        nst, nobs, r, done, info = K.step(env, st, action, kn)
        rec = (st.pos, st.heading, st.speed, nst.just_crashed, nst.goals,
               st.ped_pos, nst.arrived)
        return (nst, nobs), rec

    kr, ks = jax.random.split(key)
    st, obs = K.reset(env, kr)
    keys = jax.random.split(ks, env.max_steps)
    _, (pos, heading, speed, crashed, goals, ped, arrived) = jax.lax.scan(
        step_fn, (st, obs), keys)
    return {"pos": np.asarray(pos), "heading": np.asarray(heading),
            "speed": np.asarray(speed), "crashed": np.asarray(crashed),
            "goals": np.asarray(goals), "ped": np.asarray(ped),
            "arrived": np.asarray(arrived)}


def render(net: RoadNetwork, pos, crashed, goals, ped, out_prefix, title,
           stride=2, fps=15):
    T, N, _ = pos.shape
    x0, y0, x1, y1 = net.bounds()
    fig, ax = plt.subplots(figsize=(8, 7), dpi=100)
    ax.set_facecolor("#0e1116")
    fig.patch.set_facecolor("#0e1116")

    # road network background
    segs = net.node_xy[net.edges]  # (E, 2, 2)
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(segs, colors="#39404d", linewidths=0.8))
    ax.set_xlim(x0 - 30, x1 + 30)
    ax.set_ylim(y0 - 30, y1 + 30)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, color="white", fontsize=13, pad=10)
    hud = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                  color="white", fontsize=10, family="monospace")

    # pedestrians (orange diamonds) under cars
    peds = ax.scatter(ped[0, :, 0], ped[0, :, 1], s=18, c="#f59e0b",
                      marker="D", edgecolors="none", zorder=2, alpha=0.9)
    scat = ax.scatter(pos[0, :, 0], pos[0, :, 1], s=30, c="#3b82f6",
                      edgecolors="white", linewidths=0.4, zorder=3)
    ax.scatter([], [], c="#f59e0b", marker="D", s=18, label="pedestrians")
    ax.scatter([], [], c="#3b82f6", s=30, label="cars")
    ax.scatter([], [], c="#ef4444", s=30, label="crashed")
    ax.legend(loc="lower left", facecolor="#0e1116", edgecolor="#39404d",
              labelcolor="white", fontsize=8, framealpha=0.6)
    frames = range(0, T, stride)

    def update(t):
        p, cr, g = pos[t], crashed[t], goals[t]
        colors = np.where(cr, "#ef4444", "#3b82f6")
        scat.set_offsets(p)
        scat.set_color(colors)
        peds.set_offsets(ped[t])
        hud.set_text(f"t={t:3d}/{T}   crashed={int(cr.sum()):2d}/{N}   "
                     f"trips done={int(g.sum()):3d}   peds={ped.shape[1]}")
        return scat, peds, hud

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    gif = out_prefix + ".gif"
    anim.save(gif, writer=animation.PillowWriter(fps=fps))

    # screenshots: start / middle / end
    for tag, t in [("start", 0), ("mid", T // 2), ("end", T - 1)]:
        update(t)
        fig.savefig(f"{out_prefix}_{tag}.png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return gif


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(OUT, "trained.msgpack"))
    ap.add_argument("--name", default="trained")
    ap.add_argument("--title", default="SmoothRide — trained policy")
    ap.add_argument("--agents", type=int, default=24)
    ap.add_argument("--peds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    net = load_road_network()
    x0, y0, x1, y1 = net.bounds()
    pool = build_route_pool(net, n_routes=1024, seed=0)
    env = K.make_env(pool, (x0, y0), (x1, y1), n_agents=args.agents,
                     n_peds=args.peds, max_steps=args.steps)
    params = load_params(env, args.ckpt)
    tr = rollout(env, params, jax.random.PRNGKey(args.seed), sample=args.sample)
    crashed, goals = tr["crashed"], tr["goals"]
    out = os.path.join(OUT, "artifacts", args.name)
    gif = render(net, tr["pos"], crashed, goals, tr["ped"], out, args.title)
    print(f"crashed_end={int(crashed[-1].sum())}/{args.agents}  "
          f"trips_done={int(goals[-1].sum())}")
    print(f"saved: {gif} (+ _start/_mid/_end.png)")


if __name__ == "__main__":
    main()
