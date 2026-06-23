# Nomos — Development Log

Running record of the build. Each entry is ~two sentences capturing a **learning**,
an **architectural change**, or an **improvement**, in the order it happened, so the
flow of development is legible. Newest at the bottom.

---

### 1. Environment provisioning — learning
The system Python was 3.9.6, too old for the modern JAX/Flax stack we need. Found Homebrew's `python3.12` and built the project venv on that instead, keeping everything off the system interpreter.

### 2. Layered dependency install — improvement
Split dependencies into `data` / `rl` / `demo` extras in `pyproject.toml` so each vertical slice installs and verifies on its own. This let me confirm the geo stack (osmnx 2.1.0) worked before pulling in JAX, isolating failures.

### 3. Data layer on real OSM — learning
`osmnx.graph_from_bbox` with the v2 single-tuple bbox pulls a real downtown-SF drivable network (200 nodes, 418 edges, ~1.5 km²) and caches to graphml. Projecting to a UTM metric frame and origin-shifting keeps coordinates small and in meters, which the kinematic bicycle model needs.

### 4. Host-side routing → JAX arrays — architectural change
Dijkstra/shortest-path search is awkward inside JAX, so routing happens once on the host in networkx and is handed to the env as fixed-size padded waypoint arrays (a "route pool"). This keeps the env's `reset`/`step` pure and `vmap`-able — the env just gathers a route by index.

### 5. Kinematic env as the training substrate — architectural change
Built a vectorized JAX kinematic-bicycle env where cars follow route waypoints, observe their K nearest neighbors in an ego frame, and collide by footprint distance — no rigid-body physics. Actions are bounded (accel, steer) setpoints, deliberately matching the interface the frozen low-level controller will consume at demo time.

### 6. Vectorization contract verified — learning
A smoke test confirmed `reset`/`step` run under `jax.jit` and `jax.vmap` over 64 parallel worlds, producing the expected `(B, N, obs)` shapes. Random actions crashed ~2/16 cars and earned negative reward, confirming the dynamics and collision signal are wired correctly.

### 7. Shared-policy MAPPO with centralized critic — architectural change
All cars share one actor (homogeneous agents), while the critic is centralized: it sees each agent's local obs plus a mean-pooled scene summary (CTDE). This is what lets the policy implicitly learn to anticipate and avoid others without a separate prediction model.

### 8. JIT static-argument bug — learning
JIT-ing with the `Env` marked static failed because it carries JAX arrays (the route pool) and is unhashable. Fix: pass `Env` as a normal traced pytree — its `pytree_node=False` scalar fields (n_agents, max_steps) stay static automatically — and only mark `n_worlds`/`cfg` static (with `PPOConfig` made `frozen=True` to be hashable).

### 9. First training run — learning
The loop learns: episode reward climbed 35 → 260 and reach-rate ticked up, at only ~0.13 s/iter after JIT warmup. But routes averaged ~1184 m while a 150-step horizon only covers ~360 m, so cars couldn't finish trips — the horizon and route lengths were mismatched.

### 10. Renderer on the real map — improvement
Built a renderer that replays a checkpoint and draws cars on the actual SF street network, saving a GIF plus start/mid/end PNG screenshots (Pillow writer, since no ffmpeg). Every checkpoint now produces a video + stills artifact automatically.

### 11. Continuous respawn — architectural change
Changed the env so a car that reaches its goal respawns onto a fresh route instead of freezing, turning episodes into a persistent traffic flow and making **throughput (total trips completed)** the headline metric. Crashes still freeze a car in place, realistically modeling a crash that blocks the road.

### 12. Route-length cap — improvement
Added a `max_length_m` filter (default 700 m) to the route pool so trips are completable within a 300-step horizon. This made `goals/agent` a meaningful, moving signal instead of staying pinned at zero.

### 13. Reward imbalance discovered — learning
With respawn + longer horizon, reward jumped to ~547 and cars completed trips (goals/agent 0 → 0.83), but **crash rate rose** (0.19 → 0.27): the policy traded collisions for progress/goal reward. This is the classic multi-objective reward-hacking failure — the sparse crash penalty couldn't outweigh dense progress rewards.

### 14. Artificial-potential-field spacing penalty — improvement
Added a continuous proximity penalty (`w_prox` over a `prox_radius`) plus a higher collision weight, so cars get a smooth gradient pushing them to keep distance *before* a hard crash. This operationalizes the APF-as-reward-shaping idea from the plan, giving the policy a learnable spacing signal the sparse crash term lacked.

### 15. Crashes plateaued — learning
Stronger APF shaping barely moved the crash rate (~0.24): the limit wasn't the reward weights but the geometry. Pure reward tuning has diminishing returns once the environment makes some collisions physically unavoidable.

### 16. Centerline head-on is structural — learning
Diagnosed the real cause: every car tracks the road centerline, so two cars on a two-way street going opposite directions aim down the identical line — a guaranteed head-on no policy can dodge. This is an environment-modeling flaw, not a policy failure, which is why no amount of training fixed it.

### 17. Lateral lane offset — architectural change
Made each car track a point offset to the **right** of the route direction (a right-hand normal × `lane_offset`), separating opposing flows by ~5 m. Crash rate dropped to ~0.20 and throughput rose to 0.90 goals/agent — the best result yet, from a geometry fix rather than reward tuning.

### 18. Clean learning-delta artifact — improvement
Rendered untrained vs trained at 20 cars/300 steps: untrained completes **0 trips** (cars wander and stall), trained completes **18 trips** through downtown with the same crash count. Throughput, not crash rate alone, is the legible headline of the demo.

### 19. Realism roadmap thought through — architectural change
Worked through scale, highways, lanes, dynamics, stop-and-go, pedestrians, and SF hills, and confirmed each maps cleanly onto the existing two-layer split: behavioral realism (cars/lanes/intersections/pedestrians/timing) in the cheap kinematic env, physical realism (tire forces, terrain) in Isaac (see `REALISM.md`). Added two genuinely new components — `terrain-dem` (drape roads on a DEM for hills, feeding both layers) and `pedestrians` (a second agent class reusing the APF machinery) — plus a `kinematic-env` roadmap for lanes/intersections/spatial-hashing/highways/grade.

### 20. RoutePool carries road attributes — architectural change
Extended the route pool so every waypoint carries its node index, junction flag (OSM `street_count>=3`), lane count, and speed limit (m/s). This is the data backbone that lets lanes, intersections, and highways all read real per-segment attributes instead of constants.

### 21. Env v2: per-edge speed + lead-vehicle gap — improvement
Cars now cap at the current edge's real speed limit (highways fast, surface slow), and observe the gap to the nearest car ahead in a forward cone. The lead-gap signal is what makes car-following — and emergent stop-and-go — learnable rather than just collision-driven.

### 22. Env v2: multi-lane + lane-change — architectural change
Each car holds a discrete lane index; its target is offset right of the centerline by `lane_width*(lane+0.5)` using the edge's real lane count, and a 3rd action dim shifts lanes when intent is strong. This generalizes the earlier single-offset fix to true multi-lane roads with overtaking.

### 23. Env v2: intersection yielding — architectural change
Cars approaching the same junction node compute right-of-way (closer car goes first) and a yielding car moving too fast into an occupied junction is penalized. This is the "function without traffic lights" thesis encoded as a learnable negotiation, exactly where the centralized critic helps.

### 24. Env v2: unruly pedestrians — architectural change
Added M scripted pedestrians that random-walk and dart across roads, observed by cars through the neighbor channel, with an asymmetric severe penalty (and APF term) for hitting one. This is the hotel-robot predict-and-avoid problem made literal, and it reuses the spacing machinery already in place.

### 25. Trained on the richer world — learning
On env v2 the policy reaches ~0.79 goals/agent but crash rate sits higher (~0.28) than the toy env (0.20), because pedestrians, junctions, and lane dynamics make the task genuinely harder. The harder env is the point — the crash rate is now a real, honest tuning target rather than an artifact of an oversimplified world.

### 26. Zero-shot scale-up — learning
Because the policy is decentralized (local obs only) and shared across agents, the 24-car model runs unchanged on a 3.5×3.9 km SF area with 200 cars + 40 pedestrians — it transfers to any count for free. Crash rate climbs on the much denser/larger map (it would want a fine-tune), but it confirms the architecture scales without retraining the world.

### 27. 300 cars was ~3am — learning (demand calc)
Grounded car count in real SF travel data (`smoothride/data/demand.py`): instantaneous moving vehicles = daily_VMT × hour_fraction / avg_speed, distributed by lane-km. Our 39 km² / 912 lane-km chunk holds ~977 cars at 3am, ~6,800 at midday, and ~16,000–19,000 at AM/PM peak — so 300 cars was below deep-night, confirming the "empty city" instinct.

### 28. Spatial-hash neighbors — architectural change
Replaced the O(N²) neighbor/collision/lead/junction logic with a uniform-grid spatial hash (`smoothride/env/spatial.py`): each agent only compares against the ~144 agents in its 3×3 cell block, verified to match brute force within the cell radius. This took the env from ~hundreds of cars to thousands — measured 2,000 cars at 5 ms/step and 6,000 at 20 ms/step on CPU.

### 29. Spread spawning — improvement
Cars now spawn at a random fraction ALONG their route rather than all at the origin, so at t=0 traffic is distributed across the whole network like real conditions instead of clustered at route starts. This makes the "evenly distributed at realistic density" picture correct from the first frame.

### 30. Zoom + speed-colored renderer — improvement
Added `render_zoom.py`: a city view (every car a speed-colored dot, red=stopped→green=fast, so jams are visible) and a ~180 m zoom view drawing cars as oriented rectangles, so individual lanes, turn radius, and stop-and-go are legible. Auto-centers the zoom on the busiest spot in the rollout.

### 31. Train at density, deploy at scale — learning
The decentralized policy only needs to experience the right DENSITY (cars/lane-km), not the full city count, so training a few thousand cars on the big map (with highways via per-edge speed) should transfer zero-shot to the full ~6,800-car midday render. The 24-car policy crashes 76% at 2,000-car density, which is exactly why density-aware training is required rather than optional.

### 32. RL alone plateaued — learning
Density training dropped crashes fast (10→4 crashes/car by iter 20) then flattened and would not improve further on CPU. Pure MAPPO hit a local optimum: it drives and completes trips but can't reach low crashes, so we pivoted to adding a principled safety layer + targeted scenario training on top.

### 33. Deep research + scenario taxonomy — architectural change
Launched a deep-research workflow on zero-crash safety frameworks (CBF/MPC/RSS, safe-RL, scenario curricula) and committed to a two-part plan: a runtime safety filter wrapping the policy, plus per-scenario training on edge-case junctions. This reframes "zero crash" from "train longer" to "shield + curriculum."

### 34. Scenario miner — improvement
Built `data/scenarios.py` to mine edge-case topologies straight from SF OSM: 1,718 four-way and 1,590 three-way junctions, 226 ramps, 210 U-turn/turning-circles, 161 bridges, by control type (signalized/stop/uncontrolled). The 1,430 *uncontrolled* T-junctions are the real "yield-only" hard case, and each kind comes with representative trainable locations.

### 35. Crashes froze into obstacles — learning (bug)
At density the 73%-crash wall was a bug: crashed cars stayed frozen forever and became a minefield that cascaded into pile-ups. Fixed by making a crash CLEAR (penalty + respawn) instead of freezing — both more learnable and a cleaner demo.

### 36. collision_radius > lane_width — learning (bug)
Crashes wouldn't drop because `collision_radius` (3.5 m) exceeded `lane_width` (3.2 m), so two cars driving safely side-by-side in adjacent lanes registered as colliding. Set collision 2.2 m < lane 3.5 m (plus a tighter proximity penalty), which let dense multi-lane traffic exist at all.

### 37. Respawn teleport-overlaps inflated crashes — learning (bug)
Continuous respawn was teleporting cars in near others; those instant overlaps were being counted as "crashes," inflating the number ~3×. Adding a few-step merge-in grace (real cars enter from edges, not on top of others) revealed the TRUE driving-crash baseline is ~1.3 crashes/car, not ~4.

### 38. Runtime safety shield (CBF-style) — architectural change
Built `rl/safety.py`: a vectorized control-barrier filter that caps each car's speed by its stopping distance to the nearest in-path obstacle, adds time-to-collision braking with a yield-to-the-right rule, and a last-resort omnidirectional emergency brake — a classical control layer wrapping the RL policy, scaling to thousands via the same spatial hash.

### 39. Naive brake-shield is net-negative at density — learning (important)
Honest result: at 2,000-car density the brake-only shield did NOT reduce crashes (1.27 → 1.38) — sudden stops propagate and cause collisions, a documented failure mode of hand-rolled filters in dense uncoordinated traffic. Getting to真 zero needs a principled solver (CBF-QP with feasibility, RSS, or reciprocal ORCA-style avoidance) rather than greedy braking — which is what the deep research is for.

### 40. Scenario-curriculum trainer runs but windows over-saturate — learning
Built `rl/scenario_train.py`: stacks K edge-case junction windows as parallel vmap worlds and trains one shared policy across all of them with the shield active (shielded RL). It runs (12 parallel scenarios), but small windows funnel all traffic through one junction and over-saturate it, so the load model needs bigger windows / edge-entry traffic before scenario training pays off.

### 41. Velocity-obstacle filter + the non-holonomic ceiling — learning (important)
Replaced the brake-only shield with a reciprocal velocity-obstacle (ORCA-style) filter: it stopped the cascade (throughput preserved, 723→729 trips) but still didn't cut crashes (1.27→1.31). The reason is fundamental — cars are NON-HOLONOMIC, so at low speed they can't execute the lateral avoidance the filter computes; a runtime filter is therefore a last-resort backstop, and the policy itself must learn to avoid conflict states.

### 42. Deep-research verdict (despite junk synthesis) — architectural change
The deep-research workflow's auto-synthesis returned placeholder text (API drops mid-run), but its verified signal was decisive: it KILLED the claim that MAPPO-Lagrangian guarantees per-iteration safety, so safe-RL reduces but never guarantees — the guarantee must come from a runtime filter. Recommendation locked: CBF-QP (minimal-deviation, HOCBF for the bicycle) with RSS distances, over the greedy/ORCA filters.

### 43. Diffusion's role = offline adversarial scenario generation — architectural change
Diffusion probe (well-cited) concluded diffusion is highest-leverage as an OFFLINE generator of hard junction conflicts (Safe-Sim/CCDiff/DiffScene), feeding the scenario curriculum (MATS-Gym for adaptive difficulty) — DiffScene even showed training on generated scenarios improves AV safety. Diffusion as the runtime policy is heavier and not needed for zero-crash (CBF/MPC is faster + certifiable); if ever wanted, use IDQL proposer+MAPPO-critic or DPPO. Full plan written to `RESEARCH_SAFETY.md`.

### 44. PPO-Lagrangian breaks the plateau — improvement (the lever)
Treated crashes as a CONSTRAINT (PPO-Lagrangian): zeroed the fixed crash weight and added an adaptive multiplier λ updated by dual ascent toward a target crash rate. Unlike fixed-penalty PPO (stuck), crashes/car dropped steadily 1.96→1.27 in 50 iters while throughput held (goals/agent 0.34) and λ auto-climbed 15→203 — the policy itself learning to avoid conflicts, which is the root-cause fix the non-holonomic runtime filter couldn't provide. (Floor ~1.24 at 2,000-car highway density even with λ→400.)

### 45. Diffusion scenario generator — architectural change
Built `diffusion/traj_diffusion.py`: a real DDPM (Flax MLP denoiser, 50 steps) trained on ego-relative trajectory snippets collected from policy rollouts (loss 1.0→0.09), plus classifier-style cost guidance that pulls a generated "challenger" trajectory toward the ego path to synthesize plausible near-misses. This is the Safe-Sim/MotionDiffuser idea in miniature — an offline adversarial scenario generator for the curriculum, demonstrated end-to-end with a real/generated/adversarial viz.

### 46. CBF-QP filter is the first filter that helps — improvement
Built `rl/cbf.py`: a higher-order CBF (barrier h=‖Δp‖²−d_safe², relative degree 2) whose HOCBF condition is linear in [accel, tan δ], solved as a minimal-deviation half-space projection — so it STEERS and brakes jointly (the cure for the non-holonomic ceiling that sank the brake/ORCA filters). On the PPO-Lagrangian policy it cut crashes 1.24→1.13 (first filter to actually reduce them) at ~9% throughput cost. Full stack now layers: chaos ~10 → RL+bugfixes 1.27 → Lagrangian 1.24 → +CBF-QP 1.13.

### 47. Edge-entry traffic fixes the scenario curriculum — improvement
Added a `spread_spawn=False` mode so scenario-window cars ENTER at boundary edges and drive THROUGH the junction (realistic load) instead of all respawning inside a tiny box. This took the scenario trainer from degenerate (53→11 crashes/car, no learning) to clean and learnable: 12 parallel edge-case junctions, ~1.2 crashes/car with cars completing trips, warm-started from the Lagrangian policy with the constraint active. On the full city it generalized back without catastrophic forgetting (1.25→1.24 alone, 1.14→1.13 with CBF-QP) — the curriculum is correct now; pushing the number lower needs more iters + the diffusion-fed adversarial conflicts.

### 44. Web viewer: deck.gl + Mapbox on the real SF map — improvement
Built the planned demo viewer (`smoothride/demo/web/`): `export_web.py` replays the trained + untrained checkpoints, reprojects the metric-frame trajectories back to lon/lat (inverting the UTM project + origin-shift via the graph's stored CRS), and packs them into a ~690 KB JSON; a deck.gl app renders WheeledLab-style RC-car meshes (procedural, in `carmesh.js`) on a Mapbox SF basemap with speed coloring, motion trails, and a live HUD. The trained world (full opacity) overlaid on the untrained shadow world (faint) shows the delta directly — 41 trips vs 2 — with a graceful no-basemap fallback when no Mapbox token is present.

### 45. Isaac/WheeledLab physics path scaffolded — architectural change
Scoped the cinematic rigid-body path (`smoothride/demo/isaac/`) as a real handoff rather than a promise: `export_setpoints.py` dumps the per-car target waypoint/velocity/heading stream (plus georef origin+CRS) the frozen low-level controller tracks, and runs on any machine; `run_isaac_demo.py` is a structured stub that preflights for Isaac Sim/IsaacLab/CUDA and, off-GPU, explains exactly what's missing and what it would render instead of crashing. The setpoint contract and `config.yaml` are concrete now, so the remaining GPU-box work is wiring the `TODO(isaac)` calls, not redesigning the boundary.

### 48. Low-level controller: Isaac Lab + WheeledLab training on Modal — architectural change
Scaffolded the frozen locomotion layer (`smoothride/lowlevel/`) as a real Isaac Lab external task: a manager-based command-tracking env (`isaac_task/track_env_cfg.py`) where a WheeledLab Ackermann car learns to track a (forward-velocity, heading) setpoint and emit [throttle, steer] — the exact interface the high-level coordination policy emits — trained headless with RSL-RL PPO on a Modal A100 (PhysX only, no RT Cores, which is why no RTX box is needed for *training*). `train_isaac.py`/`export.py` keep all Isaac imports inside the remote Modal function so `modal run` works from the laptop, and `export.py` freezes the actor to a CPU-runnable `controller.pt` that the Isaac demo runner consumes. Honest status: not runnable from this Mac (no Isaac/GPU); APIs + Modal wiring are faithful, with `TODO(wheeledlab)` only on the car asset USD path + joint names that need the cloned WheeledLab repo on the first run.

### 49. Pivot: 3D physics on Antim/HUD worldsim (Newton), not Isaac — architectural change
Read the hud-evals worldsim-template and pivoted the physics/3D demo onto it: Newton physics (Warp + mujoco-warp) over MJCF scenes, where a car is just MJCF + the generic `step(action)` tool — no Isaac Lab, no nvcr.io, no RTX gating. Built and MuJoCo-validated `smoothride/worldsim/`: an Ackermann physics car (`scenes/car-v1`), an OSMnx-graph→3D-SF scene generator (`build_sf_scene.py`, roads as visual decals + N cars + optional extruded buildings), and a weight-free control bridge (pure-pursuit + P throttle) that closed the loop in a local rollout (drove a car 25m→8.4m to a waypoint). The whole chain is verifiable with stock `mujoco` on the Mac; only the Newton/Warp runtime (likely a GPU box) is unverified. Supersedes the Isaac path (`demo/isaac`, `lowlevel`) unless WheeledLab PhysX dynamics / sim-to-real are specifically needed. Full API map + build-on-top plan in `smoothride/worldsim/HANDOFF.md`.

### 50. Worldsim demo on Modal GPU + route-following planner — improvement
Made the Newton/worldsim 3D path turnkey on Modal (Newton needs CUDA): `render_modal.py` builds the worldsim-template + bundled Newton wheel image, generates the SF scene, drives every car along a real route, renders frames via the worldsim MCP tool API, and writes an mp4 to a Volume. Added `planner.py` (RoutePlanner) reusing the kinematic env's `build_route_pool` so cars follow real SF streets, with the scene and planner sharing one deterministic route assignment via scene metadata (cars spawn ON their routes). Validated the full planner+control_bridge loop locally in stock MuJoCo: 6/8 cars follow their routes ~75m with waypoint progress (2 stall on sharp initial turns — a documented actuator-gain knob). Only the Newton runtime + Modal image build remain unverified from this Mac.

### 51. Traffic-law compliance: pass/fail check + ground-up lawful RL — improvement
Added a legality layer that scores whether cars obey the law and trains a policy that does. `env/legality.py` is a pure (env, state)→per-car check: OFF-LANE = >1.5 lane-widths from the nearest lane centerline of the segment the car is *currently* driving (point-to-segment, nearest-lane, so legal lane changes / corner-cutting don't false-trip — key on this map where ~77% of route waypoints are intersections), and WRONG-WAY = heading against the route while moving; respawn grace exempt. `render_zoom` overlays it on the zoom view — magenta outline on law-breakers, live legal-% HUD, standing PASS/FAIL banner (legal-rate >=99% to pass) — and `render.rollout` records the per-step violations. Then I baked it into the reward (continuous off-lane ramp `w_offlane`, wrong-way flag `w_wrongway` in `kinematic.step`, logged through PPO) and trained from scratch on the big SF map (`--big --w-offlane 6 --w-wrongway 8`, 150 iters): off-lane 53%→~6%, wrong-way 5.8%→~0%, reward −1207→+390, goals/agent ↑. On the 6000-car render the learned (not filtered) policy cut violations roughly in half vs the baseline checkpoint: legal 88.9%→94.6%, off-lane-steps 166.5k→80.4k, wrong-way 957→589 — trails now hug the lanes. Still FAILs the strict 99% bar at full city density (trained at 24 agents/world); closing it wants density-matched training + higher law weight. Checkpoint: `trained_law.msgpack`.

### 52. Respawn no longer manufactures crashes — bugfix / realism
Fixed the "cars pop in and randomly crash" artifact. Two causes in `kinematic.step`: (1) spawn-grace only made the *fresh* car immune, so an existing moving car still got flagged for crashing into a just-teleported one — now spawn-immune cars are EXCLUDED as collision/proximity partners, so a respawn near traffic never counts as a crash for either side; (2) respawned cars appeared as a dead stop (speed 0) in a possibly-fast lane and got rear-ended the instant their grace expired — they now merge in at 0.6× the spawn edge's speed limit. (Also tried re-entering every respawn at the route START instead of a random midpoint, but at 6000 cars that piled cars onto shared start nodes and made crashes *worse* — reverted; spread-spawn + the immune mask is the right combo.) Retrained `trained_law` under the corrected dynamics. On the 6000-car lawful render: crashes/car 3.9→2.9 (below the 3.2 baseline), trips 1948→2226, all 6000 cars moving at the end — the churn of red cars blinking in/out is gone and traffic flows.
