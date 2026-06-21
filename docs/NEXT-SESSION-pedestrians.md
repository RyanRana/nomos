# Next-session pickup — pedestrians that cars slow for (no traffic signals)

**Branch:** `worktree-3d-sim-setup` (pushed to `origin/worktree-3d-sim-setup`). Built this session; resume here.

---

## The task (DECIDED — build next session)

Goal: a dense, **signal-free** pedestrian environment where cars **SLOW (not stop)** for crossing pedestrians.

**Spec (from the user):**
1. **Hard-code pedestrian paths** — deterministic, NOT the current random walk.
2. **Footpaths**: peds stick to sidewalks (offset from the road centerline ≈ road-half-width + ~1.5 m), not in the roadway.
3. **Cross the street**: each ped crosses perpendicular at a point — the moment it's *in* the road is what cars must negotiate.
4. **Randomize START TIME** (staggered): each ped has a random start step; waits, then walks its path.
5. Cars should **SLOW** for peds, **NOT stop**.
6. **Retrain** (this is "option 2"): peds → trace → verifier pedestrian-yield predicate → PPO-Lagrangian. Reward stays §9 (efficiency only).

## Implementation sketch

**A. Replace the ped motion model** (`smoothride/env/kinematic.py::_ped_step`, currently random-walk bouncing off `world_min/max`):
- Host-side `build_ped_paths(net, n_peds, seed)`: per-ped polyline = a sidewalk run + one perpendicular crossing. Emit fixed arrays: `ped_paths (M,P,2)`, cumulative arc length, `ped_starts (M,)`.
- Env: `ped_pos = arc_interpolate(path, walked)` where `walked = max(0, t - ped_start) * ped_speed * dt`. Before start → `path[0]`; after end → `path[-1]`. Deterministic, reproducible, JAX/vmap-friendly (no per-step RNG). Derive ped heading from the path tangent for rendering.

**B. Pedestrian-yield cost** (the brake incentive — cars currently NEVER brake, see Findings):
- Add **ped positions to the `Trace`** (`smoothride/rl/trace.py`) and log them in `ppo.collect` (it already logs car State; add `ped_pos`).
- `smoothride/rl/verifier.py::step_cost`: add a **CONTINUOUS ped-yield term** — cost ramps with `speed × proximity` to a nearby crossing ped (a hinge), so the optimum is to **slow** near peds, not freeze. ⚠️ Make it soft/graded, NOT a hard collision flag, or cars learn to stop dead.
- Keep it in the cost channel (CMDP); reward stays §9.

**C. Retrain** on Modal (verifier-driven) with the new peds + ped-yield cost, dense peds (`--peds 300-400`). Then eval/render.

**D. Viewer** already renders peds (amber cylinders, `app.js::addPed`) — no change needed.

---

## Where things stand (done this session)

- **Verifier merged** into this branch: full CMDP loop runs — §9 efficiency-only reward (`w_progress·progress + w_goal·arrival − w_time`) + **deterministic verifier cost** (off-lane/wrong-way/crash/speed) + PPO-Lagrangian. 61 tests green; 3 smokes pass.
- **Finite-cohort env**: remove-on-arrival (freeze+mask, no respawn) + **non-overlapping spawns** (root-cause spawn fix). Viewer car colors: red=crashed / green=arrived / blue=en-route.
- **Cesium viewer**: World Terrain + OSM Buildings (ion token in **gitignored** `config.js`), renders pedestrians, `no-store` fetch.
- **`modal_train.py`**: verifier-driven (`--verifier --cost-target`), **`--region`** (named SF regions in `map_loader.SF_REGIONS`: downtown / chinatown_fidi / mission / nopa), **`.spawn()`** so a flaky local connection can't cancel the run (run with `modal run --detach`).
- **`scripts/eval_policy.py`**: held-out generalization harness (`--region`), reports arrivals / crashes / **per-step AND any-step** off-lane/wrong-way.
- **Checkpoints** in Modal volume `smoothride-nav-ckpts`: `trained.msgpack` (downtown, 150 cars) and `trained_ctfidi.msgpack` (chinatown_fidi, 150 cars).

## Key findings (carry forward)

- **Cars CAN modulate speed but the trained policy floors it 100% of steps (0% braking).** §9 reward pays for progress + penalizes time → no reason to slow. **This is exactly why the ped-yield cost is needed** to create a brake incentive.
- Per-edge speed limits cap cars at **8.9–13.4 m/s** (`v_max=16` is non-binding) — realistic city speeds; don't raise to highway.
- **Observation is 26-dim, fully local/ego-relative** (6 ego + 1 lane + 16 for 4 neighbors + 3 nearest-ped). This is why it generalizes — and why lane discipline degrades cross-map (reflexes calibrated to the training grid's turns).
- **Generalization**: trained-downtown → NoPa held-out = **90% arrivals**, but strict per-car "valid" drops (brief corner-overshoot off-lane; per-step off-lane only ~5%).

## Other pending (not blocking peds)

- **Chinatown/FiD → Mission held-out eval** — `trained_ctfidi.msgpack` is ready in the volume. Run:
  `modal volume get smoothride-nav-ckpts trained_ctfidi.msgpack runs/trained_ctfidi.msgpack`
  `python scripts/eval_policy.py --region mission --trained runs/trained_ctfidi.msgpack`
- **Perf (for big overnight runs)**: the verifier cost relabels on the **host** (~4.5 s/iter vs ~0.2 s GPU-only). "Option 2" = port `step_cost` into the JAX scan so it runs on-device.
- Optional: `--cost-target 0.02` for tighter lane discipline.
