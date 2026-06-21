# ⚠️ SOURCE OF TRUTH — Overnight autonomous run (2026-06-21 → ~09:00)

**If you are resuming after a compaction: READ THIS FILE FIRST. It — plus `git log` and the Modal volume — is authoritative over any conversation summary.** Update it every cycle and `git commit && git push` after each update.

## Mission (from the user, leaving overnight)
- **Minimize crashes** (especially car–pedestrian) as low as possible across SF regions. **Do NOT stop early** — keep iterating until ~09:00; report the best at wake-up.
- **Find the density frontier:** the max car density that still holds **≤0.5% crash rate** (≈1–2 crashes per 300+ cars).
- **Validate via leave-one-out (LOO)** across regions `{downtown, mission, nopa, chinatown_fidi}` (train on subset, eval on held-out).
- **Free to test architectures** (Deep Sets vs ego-query attention / social-attention) — time + Modal compute are plentiful.
- **Ops rules:** frequent `git commit && git push`; **light review** (tests pass + documented, NO reviewer subagents); event-driven Modal monitoring; if near usage limit, sleep until reset; performance is NOT a priority (we just run a policy in sim).

## How to operate (the loop)
1. Launch Modal training jobs **detached** (`modal run --detach -m smoothride.rl.modal_train …`) — they run server-side and survive disconnects. Distinct `--tag` per experiment.
2. Monitor by polling the volume `smoothride-nav-ckpts` for `trained<TAG>_it*.msgpack` + `history<TAG>.json` (use a `Monitor` that breaks on the final snapshot / errors). Idle between (low token burn).
3. When a run finishes: pull `history<TAG>.json`, record metrics here, decide next experiment, launch it. Commit+push this file each cycle.
4. For eval/LOO: `scripts/eval_policy.py --region <held-out> --trained runs/<ckpt>` (reports arrivals + per-step/any-step crash rates). NOTE the per-bbox cache fix (99a9f9e) means `--region` now truly loads that region.

## Key facts / gotchas
- **Branch:** `worktree-3d-sim-setup`, pushed to `origin`. **Worktree:** `/Users/aineshmohan/Developer/driving/.claude/worktrees/3d-sim-setup`. Python = `python3`.
- **Modal:** authed; volume `smoothride-nav-ckpts` at `/ckpts`. App name prefix from `modal_train.APP_NAME`.
- **`--region` cache bug FIXED** (99a9f9e): graph cached per-bbox. Before that, all regions silently loaded downtown.
- **Cost redesign (v2):** `verifier.step_cost_components()` → dict; `hard_cost()` = `w_carcar·car_crash + w_carped·ped_hit` (collisions → drive to 0); `soft_cost()` = off_lane+wrong_way+over_cap+ped_yield+**car_risk** (graded). `car_risk_cost()` is the new dense "back off" hinge. Env `info` now exposes `car_crash` and `ped_hit` separately.
- **v1 baseline** (`trained_peds`, downtown, single cost target 0.08): crash/car 0.068, arrived 74%. Held-out Mission: 67% arrived / **12% crash** (the gap to close).

## Build status (foundation for the sweeps)
- v2-T1 intersection crossings — ✅ DONE (663ee7f)
- v2-T2 graded car-risk + cost components + car_crash/ped_hit — ✅ DONE (a0efd8a)
- v2-T3 dual-channel PPO + dual-Lagrangian (crash_target→0) — ✅ DONE (536f724, 159 tests)
- v2-T4 end-on-all-done trim (eval honesty) — ⬜ pending
- v2-T5 multi-region round-robin (--regions) — ✅ DONE (8825398, 174 tests)
- v2-T6 attention encoder (--arch attention) — ✅ DONE (5adc7bf, 166 tests)

## Experiment results (append every run)
| tag | region(s) train | eval region | arch | cars | peds | iters | crash/car | car-ped | car-car | arrived% | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| _peds (v1) | downtown | downtown | deepsets | 96 | 300 | 300 | 0.068 | – | – | 74 | pre-v2, single cost |
| _peds (v1) | downtown | mission | deepsets | 96 | 300 | 300 | 0.12 | – | – | 67 | held-out, cache-fixed |
| _v2val | downtown | – | deepsets | 96 | 300 | 150 | ~0.46@it60 | car-ped HIGH | ~0.000 | – | **FINDING: car-risk hinge KILLED car-car; intersection crossings spiked car-PED. crash worse than v1.** |
| _v2slow | downtown | – | deepsets | 96 | 300 | 200 | (running) | – | – | – | corrective: cruise_cap 4, w_carped 8 — slow+yield hard for intersection peds |

## RESUME HERE
- **Next action:** VALIDATION run `_v2val` launched (dual-channel, downtown, 96 cars/300 peds, 150 iters, crash-target 0.0). Watching lam_hard for saturation. If crashes drop & stable → build T5(multi-region)/T6(attention)/T4 and start LOO+density+arch sweeps. v2 run flags: --crash-target/--soft-target/--w-carped.
- **Running Modal jobs:** none yet.
- **KEY INSIGHT:** graded car-risk hinge => car-car ~0. Remaining crashes are car-PED, worsened by intersection crossings (peds at conflict points). Lever = lower cruise_cap + higher w_carped + earlier yield. May also need a density/speed frontier (fewer peds or slower).
- **Running (round 1, downtown, 96 cars/300 peds):**
  - _v2val  deepsets baseline (150it)  app ap-qtE0yvBZlX6fmJotmdRJbm  — trending crash/car ~0.44, car-car~0, car-ped HIGH
  - _v2slow cruise_cap4 w_carped8 (200it) app ap-h53Ftfqja2ROtt54TyBIpm
  - _v2att  attention arch (150it)      app ap-S5IpYn8sBuKFokOvfOAgyW
  - Monitors: poll volume for trained_<tag>_it*.msgpack + history_<tag>.json; quiet (snapshot+crash, complete, error).
- **Round 2 plan (after round1):** take the lever(s) that cut car-ped most (slow? attention? both), then sweep density/cruise_cap to find the frontier, then LOO across regions via --regions (multi-region T5 still UNBUILT — build it before LOO).
- NOTE: v2-T4 (end-on-done eval trim) still UNBUILT (cosmetic, low priority). v2-T5 multi-region DONE (8825398).
- _v2val (deepsets) plateaued ~0.44 crash/car (it100) — baseline arch insufficient for intersection-ped task. Awaiting _v2slow (slow+heavy-ped) & _v2att (attention).
- **Best config so far:** v1 `trained_peds` (0.068, but that was mid-block peds). v2 car-car solved; car-ped is the open problem.
