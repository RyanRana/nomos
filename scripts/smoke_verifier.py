"""Full-loop smoke: real JAX env rollout → Trace → deterministic verifier.

Proves the verifier works on *genuine sim output*, not just hand-built fixtures —
the env steps a random policy, the rollout is adapted into a `Trace`, and `verify()`
grades it. Runs locally on CPU (training proper runs on Modal). No network.

The Trace adapter is the seam the eventual production rollout wrapper will own; here
it is inline so the smoke is self-contained. This worktree's env predates the
remove-on-arrival latch, so `arrived` is synthesized from `goals` increments.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from smoothride.data.map_loader import load_road_network
from smoothride.env import kinematic as K
from smoothride.env.routing import build_route_pool
from smoothride.rl.trace import Trace, TraceManifest
from smoothride.rl.verifier import verify

N_AGENTS, N_PEDS, N_STEPS, SEED = 16, 8, 80, 0


def rollout_to_trace(env: K.Env, key, n_steps: int) -> Trace:
    """Step a random policy for n_steps and log the per-step State into a Trace."""
    routes_xy = np.asarray(env.routes_xy)          # (R, Wmax, 2) metric
    routes_lanes = np.asarray(env.routes_lanes)    # (R, Wmax)
    routes_speed = np.asarray(env.routes_speed)    # (R, Wmax) m/s
    step = jax.jit(lambda s, a, k: K.step(env, s, a, k))

    key, kr = jax.random.split(key)
    st, _ = K.reset(env, kr)
    prev_goals = np.asarray(st.goals)
    recs: list[dict] = []
    for _ in range(n_steps):
        key, ka, ks = jax.random.split(key, 3)
        act = jax.random.uniform(ka, (env.n_agents, env.act_dim), minval=-1, maxval=1)
        ri = np.asarray(st.route_idx)
        wp = np.asarray(st.wp_ptr)
        nst, _, _, _, info = step(st, act, ks)
        new_goals = np.asarray(nst.goals)
        recs.append(dict(
            pos=np.asarray(st.pos, np.float32),
            heading=np.asarray(st.heading, np.float32),
            speed=np.asarray(st.speed, np.float32),
            lane=np.asarray(st.lane, np.int32),
            action=np.asarray(act, np.float32),
            wp_ptr=wp.astype(np.int32),
            seg_start=routes_xy[ri, np.maximum(wp - 1, 0)].astype(np.float32),
            seg_end=routes_xy[ri, wp].astype(np.float32),
            lane_count=routes_lanes[ri, wp].astype(np.int32),
            spawn_grace=np.asarray(st.spawn_grace, np.int32),
            crashed=np.asarray(info["just_crashed"], bool),
            arrived=(new_goals > prev_goals),          # synthesized (no latch here)
            speed_limit=routes_speed[ri, wp].astype(np.float32),
        ))
        prev_goals = new_goals
        st = nst

    def stack(field):
        return np.stack([r[field] for r in recs])

    T, N = n_steps, env.n_agents
    manifest = TraceManifest(
        run_id="smoke", seed=SEED, scenario_id="random-policy",
        policy_checkpoint_id="none", config_hash="smoke", dt=float(env.dt),
        n_steps=T, n_agents=N, n_peds=env.n_peds,
    )
    return Trace(
        manifest=manifest,
        pos=stack("pos"), z=np.zeros((T, N), np.float32),
        heading=stack("heading"), speed=stack("speed"), lane=stack("lane"),
        action=stack("action"), wp_ptr=stack("wp_ptr"),
        dist_remaining=np.zeros((T, N), np.float32),
        seg_start=stack("seg_start"), seg_end=stack("seg_end"),
        lane_count=stack("lane_count"), spawn_grace=stack("spawn_grace"),
        crashed=stack("crashed"), arrived=stack("arrived"),
        speed_limit=stack("speed_limit"),
        collision_radius=float(env.collision_radius), lane_width=float(env.lane_width),
    )


def main() -> None:
    net = load_road_network()
    x0, y0, x1, y1 = net.bounds()
    pool = build_route_pool(net, n_routes=256)
    env = K.make_env(pool, (x0, y0), (x1, y1),
                     n_agents=N_AGENTS, n_peds=N_PEDS, max_steps=N_STEPS)
    print(f"env: agents={env.n_agents} peds={env.n_peds} steps={N_STEPS} "
          f"lane_width={env.lane_width} collision_radius={env.collision_radius}")

    trace = rollout_to_trace(env, jax.random.PRNGKey(SEED), N_STEPS)
    print(f"trace: pos{trace.pos.shape} — built from a real JAX rollout")

    v = verify(trace)
    print("\n=== RunVerdict (random policy → expect many violations) ===")
    print(f"valid_run={v.valid_run}  throughput={v.throughput}  "
          f"mean_travel_time={v.mean_travel_time:.2f}s")
    print(f"crash_count={v.crash_count}  off_lane_count={v.off_lane_count}  "
          f"wrong_way_count={v.wrong_way_count}  speed_violation_count={v.speed_violation_count}")
    valid_cars = sum(c.valid for c in v.per_car)
    worst = max(v.per_car, key=lambda c: c.max_lateral_offset)
    print(f"valid cars: {valid_cars}/{len(v.per_car)}  "
          f"worst max_lateral_offset={worst.max_lateral_offset:.1f} m")

    again = verify(trace)
    print(f"\ndeterministic (same trace → same verdict): {again == v}")
    print("OK")


if __name__ == "__main__":
    main()
