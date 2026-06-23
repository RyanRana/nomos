# Nomos — stack, build order & the honesty check

Companion to `unstructured.md` (annotated idea), `components/*.json` (structured
specs), and `graph.json` (dataflow graph). All findings web-researched & cited
(June 2026).

## The architecture: hierarchical control

> **TRAIN** the multi-agent coordination policy on a fast **kinematic env** (Modal GPU).
> **EXECUTE/DEMO** it on top of a **frozen low-level locomotion controller**
> (WheeledLab-derived) running rigid-body physics in **Isaac**.

The low-level controller is trained once and frozen — it is *never* wired into the
coordination training loop. That single decision dissolves the body-transfer risk:
we don't train on a simple body and swap to car physics; we learn coordination *on
top of* a fixed locomotion skill, whose action interface (velocity/heading setpoints)
is identical in the kinematic env and in Isaac.

## The stack (by layer)

| Layer | Component | Status | One-liner |
|---|---|---|---|
| data-map | **OSMnx** | ✅ | SF drivable road graph from OSM (training + routing + demo geometry) |
| data-map | **OSM proximity** (Overpass/features/Nominatim) | ➕ | real "near schools / what's near (lat,lng)" → reward caution zones |
| data-map | **Traffic data** (PeMS/AADT/LODES) | ➕ | seed #cars & density; freeway counts are the only live source |
| simulation | **kinematic-env** (JAX) | ➕ | the training substrate — many cars on the real graph, vectorized |
| rl | **MARL** (JaxMARL/PettingZoo + MAPPO/IPPO) | ➕ | CTDE coordination policy → setpoint actions ("one continuous model") |
| rl | **reward-system** | ➕ | no-crash ≫ keep-moving > shortest-path > smooth turns; curriculum weights |
| compute | **Modal** | ✅ | GPU training: rollout workers + learner + checkpoints (any GPU) |
| simulation | **lowlevel-controller** (WheeledLab-derived) | ➕ | frozen setpoint→steering/throttle; trained once, runs in Isaac |
| simulation | **WheeledLab** | ✅ | car asset (MuSHR/HOUND) + dynamics recipe to train the controller |
| simulation | **Isaac Lab** | ✅ | execution/demo only — rigid-body hero shot (RTX/L40S box) |
| simulation | **road-mesh-builder** | ⛔gap | OSM→drivable geometry for the Isaac demo (demo-only, keep cheap) |
| viz | **viz-demo** | ⛔gap | untrained gridlock vs trained smooth flow on the real SF map |

Critical path:
`OSMnx → kinematic-env → MARL (+reward) → Modal → trained policy → lowlevel-controller → Isaac → viz`

## The honesty check (your three questions)

- **Is this the best way to demonstrate RL?** Yes now. The claim ("zero crashes, no
  gridlock, no lights") lives at the behavioral level, so we train there. The demo is
  the *learning delta*: untrained → gridlock/crashes; trained → smooth flow.
- **Are we composing a real visual demo?** Yes. We render *actual logged rollouts*
  (the policy's real actions, replayed) — deck.gl on the SF map as the always-works
  path, Isaac rigid-body render as the cinematic upgrade. Nothing is scripted.
- **Are we training something real?** Yes — *two* real trainings: (1) the single-agent
  low-level controller, and (2) the multi-agent coordination policy with a measurable,
  reproducible learning curve. The kinematic env makes (2) fast enough to actually
  finish in a hackathon.

## Build order

1. **Map ground truth** — OSMnx pull of a downtown-SF bbox; cache graphml; Dijkstra baseline. (hours)
2. **Proximity + density** — Overpass/features school masks; PeMS/AADT to set car counts. (hours; register PeMS account early)
3. **kinematic-env** — JAX multi-agent env on the graph; kinematic-bicycle w/ accel & steering-rate limits matched to the car; calibrated collision footprint. (2–3 days)
4. **MARL + reward v1** — JaxMARL MAPPO, shared-param policy, start 4–8 cars, collision-dominant reward. (2–3 days)
5. **Scale on Modal** — `@app.function(gpu=...)`, fan out rollout workers, checkpoint to Volume; curriculum-anneal efficiency rewards → learning curve. (hours to wire)
6. **Low-level controller** — single-agent, Isaac + WheeledLab car/recipe, track velocity/heading setpoints, freeze. (1–2 days; start Isaac setup EARLY)
7. **Demo** — deck.gl two-worlds overlay (untrained vs trained) first; then Isaac rigid-body hero shot + live reward counters (crashes=0). (1–2 days)

Long poles to start on day 1: **Isaac install** (RTX/L40S box) and the **kinematic-env**.

## Open items to verify

- WheeledLab pretrained weights / a usable setpoint interface — assume you'll train the
  low-level controller yourself from its configs.
- Kinematic↔physics tracking gap — match dynamics limits; keep learned driving gentle.
- An actual RTX/L40S machine for the Isaac demo (else ship the deck.gl render).
