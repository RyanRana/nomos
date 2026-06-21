# Rules-native deterministic verifier — design

**Date:** 2026-06-20
**Branch:** `design/rl-env-reframe` (worktree `rl-verifier`, off `dfc67b9`)
**Companion to:** `docs/HANDOFF-sim-contract.md` (§0③, §7, §8), the RL-env-reframe design spec, the 3D-sim plan.
**Status:** approved design, pending implementation plan.

## Problem

The deterministic verifier (`smoothride/rl/verifier.py`) is the source of truth for run
validity and the eval metrics. Today it *trusts pre-computed verdict flags* logged by the sim
(`off_road`, `rule_violation`) for the geometry-heavy traffic rules. The sim's reference
implementation of those rules is `env/legality.py` (off-lane + wrong-way), which on `origin/main`
is baked into the PPO reward.

Per handoff §0③, consuming `legality.py` vs. re-deriving the geometry is **the verifier author's
call**. The decision: **re-derive**. The verifier should *own* the rules — compute them
deterministically from logged geometry — so the trace carries **measurements** and the verifier
makes **all judgments**. This keeps the verifier a pure, portable, hardware-independent judge
(handoff §8/§10: no env import, no physics replay, no randomness/network/LLM) that can score any
trace from any policy or env version with one consistent rulebook.

## Scope — the four constraints

The environment trains cars to self-organize at **unsignalized** intersections — no traffic
lights, no signs (zebra crossings a possible future exception). Collision-avoidance and
state-estimation are assumed (real AV stacks do them); the research target is the maneuvering
policy. So the verifier judges exactly four constraints:

| Constraint | Verifier predicate | Data source |
|---|---|---|
| **Stay in lane** | `off_lane` = lateral dist to *nearest* lane centerline > `OFFLANE_THRESH`; **`wrong_way`** = heading vs route direction while moving | derived from logged geometry |
| **Origin → destination** | `arrived` (latches), `travel_time` = first-arrival step × dt | existing `arrived` |
| **Avoid collisions** | `collided` = logged crash event (cars + pedestrians) | existing `crashed` |
| **Reasonable speed** | `over_speed` = `speed > speed_limit + ε` | existing `speed_limit` |

**Dropped entirely** (contradict the no-signals premise): yield-at-junction (old bitmask bit `4`),
traffic-light/sign compliance.

### Decision ① — wrong-way is in scope
Distance-to-nearest-lane does **not** catch a car driving the wrong way on a two-way road — it is
still near *a* centerline (the oncoming one). `legality.py` treats wrong-way as a separate check
for this reason. Since we already log the segment geometry, wrong-way is nearly free, and lane
discipline ("stay in your lane") means both *in the lane* and *going the right way*.

### Decision ② — collisions trust the logged crash event, not from-scratch re-derivation
Under remove-on-arrival (handoff §0②), a car that crashes (or arrives) **freezes at that spot and
is masked out of collision**. A frozen crashed car is a *phantom* — excluded as a collision
partner. A naive pos-based re-derivation (`pairwise dist < collision_radius`) would **false-flag**
every moving car that later passes through the frozen car's location, and would not know which
cars are spawn-immune. The logged `crashed` event already encodes the freeze/mask/grace semantics
and folds in pedestrian hits, so it is both the convenient *and* the correct primary signal.
"Don't use `legality.py`" is about the **lane/speed traffic rules** — crash is not `legality.py`'s
domain. A pos-based cross-check that replays the freeze/mask state is possible future work.

### Decision ③ — lateral offset → cost (hinge), not reward; verifier reports it as a metric
Lane-keeping is a **constraint**, not an efficiency objective. Per the CMDP reframe, constraints
go through the **cost channel** (Lagrangian in `rl/ppo.py`), never folded into reward as a fixed
penalty. The verifier computes the continuous lateral offset anyway (for `off_lane`), so it
reports `max_lateral_offset` per car as an eval metric, and that same number is the basis for a
**hinged** cost `max(0, offset − OFFLANE_THRESH)`. **Do not minimize raw offset**: that
would punish legal lane changes, corner-cutting, and intersection weaving — the very maneuvering
we are training. The hinge is zero inside the lane corridor and only grows once a car has left it.
The shipped per-step cost (`step_cost`) uses the **binary** off-lane indicator (handoff §6); the
continuous hinge is a drop-in refinement (the verifier already reports `max_lateral_offset`).

## How the verifier drives training (the reward model, not a report card)

The verifier is the **source of the training cost**, not an after-the-fact grader. It is "offline"
only in that it reads a logged trace instead of re-running physics — it runs **synchronously inside
every training iteration**, the reward-model pattern:

```
collect()        roll out on device (Modal GPU)        → trajectory (+ raw State fields)
verifier_cost()  gather road geometry, run step_cost    → per-step cost (B,T,N)   [host]
update()         reward_eff = reward − lam·cost          → GAE → PPO grad step
```

`rl/verifier.step_cost` / `cost_signal` produce the per-step `(T,N)` cost from the same
`_lane_flags` core `verify()` uses, so the signal the policy optimizes is exactly the rulebook the
verifier certifies — no divergence. `rl/ppo.verifier_cost` is the host adapter that gathers
`seg_start/seg_end/lane_count/speed_limit` for each logged `(route_idx, wp_ptr)` and calls
`step_cost`, returning `(B,T,N)` to drop into `batch["cost"]`. Paired with the §9 reward strip,
**every** constraint signal now reaches the policy through the verifier; the env reward is
efficiency-only. (A faster Option 2 — porting the predicates to JAX inside `collect`'s scan with
the numpy verifier as a cross-check — is deferred until the per-iteration host round-trip is a
measured bottleneck.)

## Trace schema v2 — measurements in, verdicts out

The trace stops carrying sim-computed verdicts and carries the raw geometry the rules need. The
sim-side rollout wrapper (not yet built) fills these; `off_road`/`rule_violation` were never built
sim-side (still `🔜` in handoff §4), so this is *choosing the derive path*, not removing working
code.

**Add — per car/step:**
- `seg_start: (T, N, 2) f32` — start of the road segment the car is on (`routes_xy[ri, wp-1]`).
- `seg_end:   (T, N, 2) f32` — end / current target waypoint (`routes_xy[ri, wp]`).
- `lane_count: (T, N) i32` — number of lanes on the current segment.
- `spawn_grace: (T, N) i32` — merge-in immunity countdown; exempts fresh spawns from lane/
  wrong-way (parity with `legality.py`). Less load-bearing since non-overlapping spawns (§0④) but
  cheap to keep.

**Add — scalar (static):**
- `lane_width: float` — meters; lane offset geometry.

**Remove** (sim-computed verdicts, now derived by the verifier):
- `off_road`, `rule_violation`, `road_polygon_ref`.

**Keep:** `pos, z, heading, speed, lane, action, wp_ptr, dist_remaining, crashed, arrived,
speed_limit, collision_radius`.

The trace remains immutable and self-validating (shape checks in `__post_init__`). This is a
**handoff §7 schema change**; flagged as a coordination item for the sim-side rollout wrapper.

## Verifier output

```python
@dataclass(frozen=True)
class CarVerdict:
    arrived: bool
    travel_time: float | None       # seconds, None if never arrived
    collided: bool
    off_lane: bool
    wrong_way: bool
    over_speed: bool
    max_lateral_offset: float       # meters; eval metric + cost basis (Decision ③)
    valid: bool                     # not (collided or off_lane or wrong_way or over_speed)

@dataclass(frozen=True)
class RunVerdict:
    valid_run: bool                 # all cars valid (eval headline)
    throughput: int                 # distinct cars that arrived
    mean_travel_time: float         # mean first-arrival time over arrived cars (0.0 if none)
    crash_count: int                # cars that collided
    off_lane_count: int             # cars that left their lane at any step
    wrong_way_count: int            # cars that drove against the route at any step
    speed_violation_count: int      # cars that exceeded the speed limit at any step
    per_car: list[CarVerdict]

def verify(trace: Trace) -> RunVerdict: ...
```

Counts are **per-car** ("how many cars ever violated"), matching the finite-cohort, one-trip-per-car
model; this is unambiguous under latching `arrived` and freeze-on-crash. `valid_run` is the eval
headline; per-car flags drive the training cost channel.

## Geometry math (the `legality.py` logic, re-homed in pure numpy)

For each car/step, vectorized over `(T, N)`:
1. `seg = seg_end − seg_start`; `u = seg / (|seg| + ε)` (unit along-segment);
   `right = [u_y, −u_x]` (unit right-normal).
2. Lane lines: offset the segment right by `lane_width * (l + 0.5)` for `l in 0..lane_count-1`.
   Point-to-segment distance from `pos` to each lane line; take the **nearest valid lane** →
   `lateral`. Nearest-lane (not assigned-lane) means legal lane changes and corner-cuts read as
   legal; only leaving the roadway trips it.
3. `off_lane = (lateral > OFFLANE_THRESH) & (spawn_grace == 0)`.
4. `route_heading = atan2(u_y, u_x)`; `herr = wrap(heading − route_heading)`;
   `wrong_way = (cos(herr) < WRONGWAY_COS) & (speed > IDLE_SPEED) & (spawn_grace == 0)`.
5. `over_speed = speed > speed_limit + SPEED_EPS`.

Constants live in the verifier (it owns the rule): `OFFLANE_THRESH ≈ 5.0` m (~1.5 lane widths),
`WRONGWAY_COS ≈ −0.25` (~>105° off), `IDLE_SPEED` (small, matches env idle threshold),
`SPEED_EPS = 1e-6`. `_wrap` to `(−π, π]` is a local pure helper (no env import).

## Components / files

- `smoothride/rl/trace.py` — schema v2 (add geometry fields, remove verdict fields). Keeps the
  immutable, shape-validating dataclass.
- `smoothride/rl/verifier.py` — rule engine: `verify(trace)` plus small pure helpers
  (`_lateral_offset`, `_wrong_way`, `_over_speed`, `_arrival`, `_wrap`). Pure numpy, no env import.
- `tests/rl/conftest.py` — `make_trace` factory updated to the v2 schema (defaults describe a
  clean, on-lane, forward-driving run; geometry defaults place each car on a simple straight
  segment at lane center).
- `tests/rl/test_trace.py` — shape/immutability for the v2 fields.
- `tests/rl/test_verifier.py` — rewritten around the four constraints.

## Testing (TDD, hand-built traces — no sim, no JAX)

- **off_lane:** nearest-lane picks the correct lane mid lane-change → no false trip; corner-cut
  stays legal; a car a full ~2 lanes off → trips; `spawn_grace > 0` exempts.
- **wrong_way:** heading reversed while moving → trips; stationary (`speed ≤ idle`) → no trip;
  during `spawn_grace` → no trip.
- **over_speed:** `speed > speed_limit` → trips even though the env clips speed (defensive
  ground-truth check, handoff §0 note).
- **collision:** `crashed[t,i]` → `collided`, invalidates only that car.
- **arrival/throughput:** latching `arrived`; `travel_time` = first-arrival × dt; throughput =
  distinct arrived cars; never-arrived → `travel_time=None`, `mean_travel_time=0.0`.
- **aggregates:** per-car counts; `valid_run` is the AND of per-car validity.
- **metric:** `max_lateral_offset` equals the max over steps; zero for an on-center run.
- **determinism:** `verify(tr) == verify(tr)`.
- **purity:** module imports only numpy + stdlib; no `smoothride.env` import.

## Done in this work

- Trace schema v2 + `verify()` (per-car/run verdicts).
- `step_cost`/`cost_signal` (per-step training cost) + `ppo.verifier_cost` host adapter.
- §9 reward strip in `kinematic.step` (efficiency-only reward; `w_time` added).
- `collect` emits raw State fields for relabeling.
- Smokes: `scripts/smoke_verifier.py` (env→trace→verify), `scripts/smoke_train_verifier.py`
  (verifier-driven PPO end to end).

## Out of scope (named, not silently dropped)

- **Option 2** — porting the predicates to JAX inside `collect`'s scan (verifier becomes a
  cross-check). Deferred until the per-iteration host round-trip is a measured bottleneck.
- Continuous **hinge** cost for off-lane (shipped cost is the binary §6 indicator).
- A **cross-test** asserting `verifier_cost` (host) == an in-sim JAX cost (only needed once
  Option 2 exists).
- A production rollout→Trace wrapper (the smokes' inline adapters cover this for now).
- Pedestrian-specific logic and zebra-crossing rules (future; ped collisions are already covered
  by the logged `crashed` event).
- A pos-based collision cross-check that replays freeze/mask state.
- Yield / traffic-light / sign rules (excluded by the no-signals premise).
