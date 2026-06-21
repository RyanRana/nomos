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


## ROUND 3 (relaunched cleanly) — frontier sweep, downtown cruise4/wped8 worlds16 150it:
  _v3c60p10 ap-HhiUzpgjiMmMJhI8GkmQn6 | _v3c96p20 ap-V5QgMKInL9BPNUjok0jWUs | _v3c96p10 ap-SHo9ajkCQrJoH325R9O12K | _v3c150p20 ap-reMfoFDIvWedmurN5qMnYA
## Confirmed numbers so far:
- _v2slow FINAL it199: crash/car 0.199 (96cars/300peds, fully trained, slow+wped8) — still 40x target => 300 peds too dense.
- _v2att FINAL: 0.485 (attention NOT better than deepsets ~0.44). Drop attention.
- round2 @300 cars: car_car_rate ~0.0019/step (~0.47 cumulative), car_ped~0 => CAR-CAR wall at 300 cars.
- FRONTIER HYPOTHESIS: near-zero needs cars<=~100 AND peds<=~20. round3 tests this corner.


## *** FRONTIER FOUND (round 3) *** crash/car (downtown, cruise4/wped8):
  60c/10p -> 0.021 (final) | 96c/10p -> 0.018 (it100, falling) | 96c/20p -> 0.033 | 150c/20p -> 0.16 (car-car emerging)
  => SAFE CORNER ~ <=96 cars + ~10 peds gives ~2% crash (car-car=0, residual car-ped), trending to ~1%. Target 0.5% is close.
  => 300+ cars NOT achievable near-zero (car-car saturates). Honest frontier: ~80-100 cars max for low crash in this bbox.
## ROUND 4 (push to target + LOO model):
  _v4c96p10x ap-scYK8ObAZ9iV9KqoY3sVDy (96/10, 300it floor)
  _v4c96p5   ap-KlJUaByKKSl0ClfBlG6MZA (96/5, 200it)
  _v4c60p5   ap-VF7rFY4TXXQ6qyRhClEQfC (60/5, 200it)
  _v4loo     ap-nUwSf6DzQeBdONlaKSbDIV (96/10, 300it, --regions downtown,nopa,chinatown_fidi = LOO, eval HELD-OUT mission)
  pull numbers: for tg in ...; do modal volume get smoothride-nav-ckpts history$tg.json /tmp/h$tg.json --force; python3 -c "import json,sys;m=json.load(open('/tmp/h'+sys.argv[1]+'.json'))[-1];print(sys.argv[1],m['iter'],round(m['crashes_per_car'],3))" $tg; done
## NEXT after round4: eval _v4loo on held-out mission (export_cesium/eval); pick best safe config; render a clean demo scene; final summary in this file.


## *** HEADLINE: LOO GENERALIZES — held-out Mission eval of _v4loo ***
  TRAINED (downtown+nopa+chinatown, NEVER saw mission): crashes 1/96 = 1% | arrivals 71% | wrong-way 0% | off-lane 11%/step (soft, not collisions)
  vs v1 downtown->mission = 12% crashes. => ~12x cross-map SAFETY improvement. 1/96 ~= 3/300 ~ target band.
  THESIS RESULT: graded car-risk + dual-channel(crash_target0) + slow cruise + low ped density + MULTI-REGION training => ~1% crash on an UNSEEN SF neighborhood.

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
- ROUND 1 (downtown, 96 cars/300 peds) it50: _v2val 0.48, _v2slow(cruise4/wped8) 0.40, _v2att(attention) 0.50. ALL ~0.4-0.5 crash/car. car-car ~0 everywhere (car-risk hinge works). Conclusion: cost/speed/ARCH do NOT crack car-ped at 300 intersection-peds. It's a DENSITY problem (target 0.005/car is ~90x below current).
- ROUND 2 = density frontier at TARGET 300 cars (downtown, cruise4/wped8, worlds16, steps250, 150it). Find max peds holding <=0.5% crash:
  - _v2c300p150 ap-TSrKU4ZN8nAJ7YaERwaFoR
  - _v2c300p80  ap-3eGrU0SXLFlgZNjlSDzhO8
  - _v2c300p40  ap-uu8beReHqceyUVZ3c8Tskj
  - _v2c300p15  ap-ELAdZ94LyqsGWxSJgYRXQL
  consolidated monitor; expect ~1.5-2h (300 cars slow). Round1 (val/slow/att) also finishing for the 96-car points.
- ROUND 3 (after frontier): take the safe (cars,peds) point, run LOO via --regions across {downtown,mission,nopa,chinatown_fidi}, eval held-out.
- **BEST SO FAR:** _v4loo (multi-region) => 1% crash on HELD-OUT mission (generalizes). In-distribution: ~0.9% @96/5. Round5 pushing to <=0.5% + longer LOO (_v5loolong). NEXT: when round5 done, pull numbers, eval _v5loolong on held-out mission, render a demo scene, write FINAL SUMMARY here.
