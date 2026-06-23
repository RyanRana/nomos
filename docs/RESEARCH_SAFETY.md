# Zero-crash framework — research + plan (safety layer, scenario curriculum, diffusion)

Goal: a framework strapped **on top of** our multi-agent RL policy (MAPPO, kinematic
cars on the real SF graph) that drives collisions toward zero and defeats specific
road-topology edge cases. Grounded in two research passes (deep-research workflow +
diffusion probe) and in our own measurements. Where the auto-synthesis of the deep
workflow returned junk, this is reconstructed from its verified sources + domain
knowledge.

---

## 0. What our own experiments already proved (read first)

- **Most "crashes" were sim artifacts.** Fixing three bugs took the measured rate from
  ~10 → **1.27 crashes/car** at 2,000-car density: (a) crashed cars froze into
  permanent obstacles → cascades; (b) `collision_radius` (3.5 m) exceeded `lane_width`
  (3.2 m) so adjacent-lane cars "collided"; (c) respawn teleported cars onto others.
- **A bolt-on runtime filter has a hard ceiling for cars.** A greedy brake-only shield
  was *net-negative* (cascading stops cause crashes). A reciprocal velocity-obstacle
  (ORCA-style) filter stopped the cascade but still didn't cut crashes — because
  **cars are non-holonomic**: at low speed they can't sidestep, so the lateral
  avoidance the filter wants isn't physically executable, leaving only braking.
- **Conclusion:** the runtime filter is a *last-resort backstop*. The policy itself
  must learn not to enter conflict states (anticipatory / safe-RL), and the hard
  cases must be *trained on*, not just filtered. This is exactly what the literature
  says, and it reorders our plan below.

---

## 1. Runtime safety layer — pick: CBF-QP (minimal-deviation), with RSS distances

The filter sits between policy and env: `a_exec = filter(a_policy, state)`.

| Option | Idea | Guarantee | Cost @ many agents | Verdict |
|---|---|---|---|---|
| **CBF-QP** | per car, smallest action change s.t. barrier `ḣ+αh≥0` to each neighbor | strong (if QP feasible) | small QP/agent, vectorizable | **recommended** runtime filter |
| **RSS** (Mobileye) | rule-based safe long/lat distances per scenario | "AV never the cause" | trivial | **use to set the CBF barrier / as audit** |
| **Sampling-MPC** (MPPI/CEM) | sample action rollouts, pick lowest-cost collision-free | soft | heavier, parallelizes | good alt if QP infeasible a lot |
| **HJ reachability** | precomputed safe sets | strongest | precompute-heavy, hard for N interacting | not practical here |
| greedy brake (what we tried) | slow for any threat | none | cheap | **net-negative — don't** |

- **Why CBF-QP over what we built:** the QP returns the action *closest to the policy's*
  subject to the barrier, using **steer + brake jointly** and staying feasible — the
  documented cure for brake cascades. For non-holonomic cars use a **higher-order CBF**
  (relative degree 2) or a kinematic-bicycle barrier so steering enters the constraint.
- **Honest caveat (our finding + literature):** even CBF-QP only guarantees safety
  *while feasible*; at high density/speed the feasible set can be empty (no safe action
  exists because the policy already drove into a trap). So the filter's value is bounded
  by how good the policy is — hence §2–§3.
- Sources: CBF safety filters arXiv 1812.05506, 2204.12507, 2012.01010; SafeDiffuser's
  per-step QP (1903→ ICLR'25, 2306.00148) confirms the minimal-deviation QP cost (~5–15×).

## 2. Safe / constrained RL — PPO-Lagrangian as the inner policy (NOT a guarantee)

- Train the base policy with the **crash as a constraint** (PPO-Lagrangian): a learned
  multiplier raises the collision cost until the constraint is met, instead of a fixed
  reward weight we hand-tuned. This yields a base policy that *enters fewer* conflicts,
  keeping the CBF-QP feasible.
- **Verified-false claim (deep-research killed it, 0/3):** MAPPO-Lagrangian / MACPO do
  **not** guarantee monotonic reward improvement *and* per-iteration constraint
  satisfaction. So safe-RL **reduces** violations but provides **no guarantee** — the
  guarantee must come from the runtime filter (§1). Source: arXiv 2110.02793 (refuted).

## 3. Scenario curriculum — mine SF + diffusion-generate adversarial conflicts

This is where edge cases get *defeated*, and where diffusion earns its place.

- **Mining (built):** `data/scenarios.py` catalogs from real SF OSM — 1,718 four-way,
  1,590 three-way (1,430 *uncontrolled* = the yield-only hard case), 226 ramps, 210
  U-turn/turning-circles, 161 bridges — by control type, with trainable window locations.
  Confirms the OSM approach (street_count degree, give_way/stop, bridge, motorway_link).
- **Adversarial generation (diffusion, recommended role):** generate **hard, plausible**
  junction conflicts *offline* to train/stress-test against — cut-ins, merge conflicts,
  left-turn-across-path yields. This is diffusion's highest-leverage, lowest-risk use.
  - **Safe-Sim** (ECCV'24, code) — adversarial term in denoising; explicit control of
    collision type + aggressiveness, closed-loop. Best fit. arXiv 2401.00391.
  - **CCDiff** (Cruise, code) — causal-structure-guided coherent long-tail scenarios.
  - **DiffScene** (AAAI'25) — guides diffusion into rare/critical regions; **training on
    its scenarios measurably improved AV safety** — direct evidence the loop works.
  - **CTG/CTG++** (NVIDIA, code) — rule-following (STL) controllable generation; CTG++
    compiles a language query into the guidance loss.
- **Adaptive curriculum:** **MATS-Gym** (ICRA'25, code) — UED/auto-curriculum for
  multi-agent driving; raises scenario difficulty to match policy competence.
- **Cheap alternative (be honest):** diffusion is overkill for *online* in-loop
  generation (high cost; no scenario paper even reports inference time). Scripted/
  parameterized junction conflicts + replay-perturbation + adversarial-RL near-misses
  (RARL 2406.02865) often suffice. **Use diffusion only as an offline generator.**

## 4. Diffusion as the policy (heavier — optional, not for zero-crash)

Diffusion *planners* are real and now real-time (Diffusion-Planner ~20 Hz A6000,
2501.15564; DiffusionDrive 2-step 45 FPS, 2411.15139; OneDP 1-step 10–700×, 2410.21257),
and they fix Gaussian-PPO's unimodal collapse at forks. But for *runtime* collision
avoidance, a CBF/MPC filter is simpler and certifiable. If we ever want the multimodal
planner: **proposer + MAPPO-critic selector** (IDQL, 2304.10573) keeps MAPPO intact;
**DPPO** (2409.00588, code) fine-tunes a diffusion policy with PPO directly; distill to
1 step for deployment. Safety still enters via cost-gradient guidance (MotionDiffuser's
pairwise-distance "repeller", 2306.03083) or a per-step QP (SafeDiffuser). **Deferred.**

---

## Recommended architecture (concrete, cheapest-first)

```
            ┌──────────────── scenario curriculum (offline) ────────────────┐
            │  SF OSM miner (built) ─┐                                       │
            │  diffusion adversarial │→ bank of hard junction conflicts ─┐   │
            │   gen (Safe-Sim/CCDiff)│   + MATS-Gym adaptive difficulty   │   │
            └────────────────────────┴───────────────────────────────────┼───┘
                                                                          ▼
   TRAIN:   MAPPO  +  PPO-Lagrangian (crash as constraint)  on mined + generated scenarios
                                                                          │  (shared CTDE policy)
                                                                          ▼
   RUNTIME: policy action ─► CBF-QP filter (RSS-set barrier, HOCBF for bicycle) ─► env/Isaac
                              └ backstop only; guarantee holds while feasible ┘
```

**Build order (what's done / next):**
1. ✅ SF scenario miner (`data/scenarios.py`).
2. ✅ Spatial-hash env scaling to thousands; bug fixes → true 1.27 baseline.
3. ✅ Runtime filter scaffold (`rl/safety.py`) — currently VO/ORCA; **upgrade to CBF-QP (HOCBF)**.
4. ⏳ PPO-Lagrangian inner policy (replace fixed crash weight with a learned multiplier).
5. ⏳ Scenario-curriculum trainer with realistic edge-entry load (fix over-saturated windows).
6. ⏳ Diffusion adversarial scenario bank (Safe-Sim) — offline, feeds step 5.
7. ◻︎ (optional) diffusion planner via IDQL/DPPO if multimodal junction behavior is needed.

## Key sources
CBF filters: arXiv 1812.05506 · 2204.12507 · 2012.01010 · 2306.00148(SafeDiffuser).
Safe-RL: 2110.02793 (the refuted-guarantee paper).
Diffusion policy/planner: 2205.09991(Diffuser) · 2303.04137(Diffusion Policy) ·
2501.15564(Diffusion-Planner) · 2411.15139(DiffusionDrive) · 2410.21257(OneDP) ·
2306.03083(MotionDiffuser) · 2304.10573(IDQL) · 2409.00588(DPPO).
Diffusion scenarios: 2401.00391(Safe-Sim) · 2412.17920(CCDiff) · 2210.17366(CTG) ·
2306.06344(CTG++) · DiffScene(AAAI'25) · 2403.17805(MATS-Gym).
OSM: osmnx graph-simplify notebook; OSM wiki give_way / motorway_link / key:turn.
