# Nomos — reinforcement-learned traffic coordination for a driverless San Francisco

> *nomos* (νόμος) — Greek for "law." When every car is autonomous, the law of the
> road stops being painted lines and signal timing and becomes a *learned* policy.

When every vehicle drives itself, traffic lights and lane law are legacy overhead for
human limitations. Nomos rips them out and learns the **behavioral layer** instead — how
a whole fleet shares the real San Francisco road graph without a central controller and
without crashing. Low-level autonomy (perception, throttle/brake/steer) is assumed
solved; the one irreducible human element is **pedestrians**, whom cars must yield to.

![Nomos — orbiting the trained fleet over downtown San Francisco](docs/media/nomos-aerial.gif)

![Nomos — street-level dolly through the fleet](docs/media/nomos-street-dolly.gif)

*The trained policy driving the live [Cesium viewer](smoothride/demo/cesium/) on the real
SF grid — 172 cars, per-car colors, building facades, and a live telemetry dashboard.*

---

## The RL, in brief

**Formulation — decentralized multi-agent control.** Each car is an agent with a *local*
observation (its own kinematic state, goal, and nearby neighbors/pedestrians). Actions are
**setpoints** — waypoint / target velocity / heading — not raw torque, so the same policy
drives the training env and the downstream low-level controller. It's a Dec-POMDP: no agent
sees global state at run time.

**Algorithm — MAPPO/IPPO under CTDE.** *Centralized training, decentralized execution.* A
centralized critic sees all cars during training; each car acts from local obs at run time.
The policy is **shared-parameter** across homogeneous agents, so it scales to many cars
cheaply, and is trained with PPO + GAE across many parallel JAX worlds. CTDE is the standard
answer to the core multi-agent problem — **non-stationarity**, where the environment shifts
under each agent as the others learn.

**Reward — multi-objective with a curriculum.** A weighted sum the policy optimizes:

| component | sign | why |
|---|---|---|
| no-crash | − (dominant) | footprint-overlap / near-miss penalty |
| keep-moving | + | reward **progress toward goal**, not raw speed |
| shortest-path | − detour | Dijkstra baseline on the OSM graph |
| smooth turns | + | gentler driving → smaller sim-to-sim gap |

Weights are **annealed on a curriculum**: collision avoidance dominates first, efficiency
rewards fade in once crash-rate drops. Reward shaping guards against the classic hacks —
idle-forever (idle penalty) and circle-driving to farm speed (progress, not velocity).

**Safety — a certifiable backstop.** Learning alone won't guarantee zero crashes, so the
policy is filtered by a **Higher-Order Control Barrier Function (HOCBF)** QP that both
*steers* and *brakes* to stay in the safe set, plus a **dual-Lagrangian constrained-PPO**
objective that treats crash cost as a hard constraint rather than just another reward term.

**Network.** Permutation-invariant **Deep Sets + attention** over a variable neighbor set,
so the policy is agnostic to how many cars are nearby.

The reasoning behind each choice lives in [`components/`](components/) (one JSON per
decision) and [`docs/`](docs/).

---

## Run it

```bash
pip install -e .            # JAX + deps
python -m smoothride.rl.train_local      # train the coordination policy locally
python scripts/eval_policy.py            # held-out evaluation
pytest                                   # ~170 tests (env, rl, data, demo)
```

The 3D viewer is static — open `smoothride/demo/cesium/index.html`, or visit the deployed
root (`/`). See [`smoothride/demo/cesium/SCENES.md`](smoothride/demo/cesium/SCENES.md).

---

## Repo map

- `smoothride/env/` — JAX kinematic env, routing, spatial hash, pedestrian paths.
- `smoothride/rl/` — `ppo.py` (MAPPO + dual-Lagrangian), `cbf.py` (HOCBF safety filter),
  `networks.py` (Deep Sets + attention), `verifier.py`, `modal_train.py`.
- `smoothride/data/` — OSM map loader + SF travel-demand model.
- `smoothride/demo/` — the Cesium 3D viewer, landing page, and scene exporters.
- `smoothride/worldsim/` — experimental rigid-body physics path.
- `components/` — the design decisions, one JSON per component.
- `docs/` — [`DEVLOG.md`](docs/DEVLOG.md) (full build log), `REALISM.md`,
  `RESEARCH_SAFETY.md`, `STACK.md`, and `internal/` handoff notes.
- `scripts/`, `tests/` — eval/export utilities and the pytest suite.
