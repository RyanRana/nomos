# v2 Task 5 — Multi-Region Round-Robin Training Report

## Implementation

### Files changed
- `smoothride/rl/modal_train.py` — added `region_for_iter`, `--regions` flag to `train()` and `main()`, multi-env build loop, per-iter round-robin dispatch.
- `tests/rl/test_region_for_iter.py` — 7 unit tests for `region_for_iter` (new file).

### Round-Robin Design

`region_for_iter(it, regions)` is a pure function: `regions[it % len(regions)]`. No state, no mutation. Before the training loop, one env is built per distinct region (calls `load_road_network` + `build_route_pool` + `K.make_env`) and stored in `env_map: dict[str, env]`. Each loop iteration picks the env via `env_map[region_for_iter(it, active_regions)]`.

The shared policy `ts` and dual-Lagrangian multipliers (`lam_hard`, `lam_soft`) are **global** — they are not reset between regions. This is the correct behaviour: the policy optimises a single constraint budget across all neighborhoods.

### JAX-Recompile Note

The first `ppo.collect` call on each distinct env instance triggers a JAX JIT recompile. With N regions this means ~N recompiles at the start of training (not per-iteration). Since all envs are built with identical `n_agents`, `n_peds`, `max_steps`, and `worlds`, the compiled shapes match — subsequent iters on the same env reuse the compiled kernel. Overhead is a one-time cost at iteration 0, 1, ..., N-1 (first time each env is seen).

### Backward Compatibility

`--regions ""` (empty string, the default) falls back to `--region` (default `"downtown"`), preserving single-region behaviour exactly. `make_train_state` always uses `env_map[active_regions[0]]` (first region) to initialise parameters, since obs/act dims are identical across regions.

### Per-Iter Logging

Each history entry and progress line now includes `m["region"] = iter_region` so training curves can be split by region in post-analysis.

### `region_for_iter` Test Coverage

7 tests in `tests/rl/test_region_for_iter.py`:
- `test_two_regions_alternates` — 0→A, 1→B, 2→A, 3→B
- `test_three_regions_cycles_correctly` — full 3×3 cycle verified
- `test_large_iter_wraps` — iter 1000, 1001 with modulo check
- `test_single_region_constant` — iters 0-9 always return same key
- `test_single_region_large_iter` — iter 9999 with 1-element list
- `test_empty_list_raises` — `ValueError` with "non-empty" message
- `test_iter_zero_returns_first` — first call always returns `regions[0]`

## Example LOO Commands

**Train on 3 regions (hold out `mission`):**
```bash
modal run -m smoothride.rl.modal_train \
  --regions downtown,nopa,chinatown_fidi \
  --iters 400 --tag _loo
```

**Eval on the held-out `mission` region** (pull checkpoint, run eval scene export):
```bash
modal volume get smoothride-nav-ckpts trained_loo.msgpack runs/trained_loo.msgpack
python -m smoothride.demo.export_cesium --elevation synthetic --agents 96 \
  --region mission --out smoothride/demo/cesium/public/scene_loo_eval.json
```

## Concerns

1. **JAX recompile at startup**: With 3 regions you get 3 recompiles in the first 3 iters. On A100 this is ~30-60 s overhead per region, acceptable for multi-hundred-iter runs.
2. **Imbalanced region coverage**: Round-robin gives equal iters per region only when `iters % len(regions) == 0`. Off-by-one is harmless for large `iters`, but worth noting if comparing per-region metrics.
3. **lam ascent speed with fewer samples per region**: With N regions the effective per-region sample rate is `1/N`. If `soft_target` convergence is slow, increase `--iters` proportionally (e.g. 3× for 3 regions vs single-region).
4. **eval path unchanged**: The evaluation / export pipeline (`export_cesium`, scene viewer) reads a checkpoint file and accepts `--region` for a single eval region. No changes needed there.
