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
- v2-T3 dual-channel PPO + dual-Lagrangian (crash_target→0) — ⏳ IN PROGRESS
- v2-T4 end-on-all-done trim (eval honesty) — ⬜ pending
- v2-T5 multi-region round-robin (`--regions`) for LOO — ⬜ pending
- v2-T6 attention/social-attention encoder variant (selectable) — ⬜ pending

## Experiment results (append every run)
| tag | region(s) train | eval region | arch | cars | peds | iters | crash/car | car-ped | car-car | arrived% | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| _peds (v1) | downtown | downtown | deepsets | 96 | 300 | 300 | 0.068 | – | – | 74 | pre-v2, single cost |
| _peds (v1) | downtown | mission | deepsets | 96 | 300 | 300 | 0.12 | – | – | 67 | held-out, cache-fixed |

## RESUME HERE
- **Next action:** finish v2-T3 (dual-channel training), then a validation run on downtown to confirm the new cost lowers crashes vs v1; then build T5/T6/T4 and start the LOO + density + arch sweeps.
- **Running Modal jobs:** none yet.
- **Best config so far:** v1 `trained_peds` (downtown). Target: beat 0.068 crash/car and close the 12% held-out gap.
