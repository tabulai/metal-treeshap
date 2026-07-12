# metal-treeshap

An exact TreeSHAP implementation for Apple GPUs — a Metal port of NVIDIA's
[GPUTreeShap](https://github.com/rapidsai/gputreeshap) (Mitchell, Frank & Holmes,
[arXiv:2010.13972](https://arxiv.org/abs/2010.13972)) from CUDA to Metal.
Apache-2.0, with attribution to upstream (see LICENSE, NOTICE).

Start with these documents in `docs/`:

1. **[docs/01-cuda-acceleration-assessment.md](docs/01-cuda-acceleration-assessment.md)** — how
   the CUDA implementation actually accelerates TreeSHAP (path decomposition, SIMD-cooperative
   extend/unwind recurrences, bin packing, mixed precision), from a close read of
   `gpu_treeshap.h`.
2. **[docs/02-apple-gpu-project-proposal.md](docs/02-apple-gpu-project-proposal.md)** — the
   project plan for the Apple GPU port: CUDA→Metal mapping, the fp64 problem and accumulation
   strategies, compiled-model architecture, phased milestones, benchmark plan, risks.
3. **[docs/03-phase2-deterministic-design.md](docs/03-phase2-deterministic-design.md)** — the
   bounded-memory, fixed-order two-stage accumulation path.
4. **[docs/04-phase2-performance-results.md](docs/04-phase2-performance-results.md)** — the
   reproducible M4 Max tuning, accuracy, repeatability, CPU comparison, and limitations.
5. **[docs/05-phase21-production-results.md](docs/05-phase21-production-results.md)** — the
   atomic-tiling result, precise Kahan reducer, wide/multiclass measurements, Python API,
   optional SHAP comparison, and final production defaults.

## Status (Phase 2.1 production path complete on M4 Max)

The portable pipeline, checked-in metal-cpp runner, and Metal kernel pass the local M4 Max
differential suite in all three accumulation modes. On the checked-in 500-tree/depth-8 stress
generator, 8,192 rows take **0.6206 s** with the selected Metal configuration versus
**12.0345 s** for 16-thread XGBoost CPU `pred_contribs`: **19.39× steady-state API speedup**.
Adding separately measured setup components gives a **6.72× derived setup-plus-call estimate**,
not direct model-to-first-answer latency. These figures apply to the documented M4 Max workload,
not every model or Apple GPU; see the Phase 2 results for raw sample dispersion and limitations.
Phase 2.1 additionally measured **22.63×** on a 256-feature regression workload and **21.98×**
on an eight-class workload. A blocked finalist experiment did not reproduce a benefit from
atomic row tiling, so full dispatch remains the default. Precise Kahan reduction lowers the
deterministic stress error about 10.8×, and a tested nanobind wheel now exposes
`MetalTreeExplainer`.

| component | file(s) | state |
|---|---|---|
| Path representation + 32-B GPU layout | `include/metal_treeshap/paths.h` | **built & tested** (any platform) |
| Host preprocessing + two-layer validation (raw checks BEFORE dedup so merging can't launder malformed input, structural checks after; BFD/FFD/NF packing, sort, segments, fp64 bias) | `include/metal_treeshap/preprocess.h` | **built & tested**, ASAN/UBSAN clean locally |
| Scalar reference oracle (lane-faithful float recurrences, fp64/fp32 accumulation, order-shuffle mode) | `reference/reference_shap.h` | **built & tested vs xgboost** |
| XGBoost extractor: raw-JSON based — `tree_info` groups, vector base_score intercepts (3.1+), empirically-verified objective link table with explicit allowlist, DART `weight_drop`, categorical/multi-target rejection; works from a model file without xgboost | `tools/extract_paths.py` | **tested on xgboost 2.0.3 AND 3.1.2**, incl. `num_parallel_tree`, DART, and 9 objective-link cases |
| Golden tests (16 cases + rejection check; **non-mutating** — fixtures/results only change under explicit flags) | `tests/test_vs_xgboost.py` → `tests/RESULTS.md` | **passing on XGBoost 2.0.3 and 3.1.2** at the 1e-3 gate |
| Property-based additivity tests (stumps, structurally guaranteed depth-31/32-element groups, deterministic comb, repeated features, ~1e-4 covers, NaNs, independent exact-Shapley vector oracle) | `tests/test_property_additivity.cpp` | **passing**, incl. asserted 32-lane group execution |
| Frozen fixtures replayable without xgboost (6 model cases, including a real missing-only path, plus synthetic `deep31` 32-lane comb) | `tests/fixtures/*/`, `tests/test_fixture.py` | **passing** |
| Metal kernels: atomic, SIMD pre-aggregation, deterministic partials+precise Kahan reduction | `shaders/treeshap.metal` | **validated on M4 Max**: Metal 3 compile, width 32, lane-31, missing-only NaN routing, every frozen fixture ≤ 6.51e-6; separate fast recurrence and precise reducer pipelines |
| metal-cpp compiled-model host: persistent shared/private model buffers, three accumulation modes, bounded private deterministic scratch, atomic/deterministic row tiling, timing and tuning controls, hardened ownership/shape checks | `src/metal_host.hpp` | **locally built and exercised on M4 Max** across all fixtures and modes; deterministic output bit-stable across 100 repeats and tile sizes |
| Metal fixture runner (repository-reproducible validation; runtime-compiles the shader when the offline toolchain is absent) | `src/main_metal.cpp`, Metal CTests | **passing locally on M4 Max**; pinned metal-cpp headers are included |
| Persistent benchmark, hashed stress/wide/multiclass generators, blocked-shuffle runner, CPU/SHAP/power tooling, schemas and raw results | `src/main_benchmark.cpp`, `benchmarks/phase2_*`, `benchmarks/results/` | **verified**; 19.39× original stress result, 22× wide/multiclass results; privileged power trace still pending |
| Pip-installable Python API | `pyproject.toml`, `bindings/python_module.cpp`, `python/metal_treeshap/` | **wheel built and tested on M4 Max**; JSON/dict/Booster/sklearn sources, NumPy output, packaged extractor and shader |

## Quickstart (any platform — the portable core)

```bash
cmake -B build && cmake --build build      # add -DMETAL_TREESHAP_SANITIZE=ON for ASAN+UBSAN
./build/test_preprocess                                  # unit tests
./build/test_property                                    # additivity property tests
pip install "xgboost>=2.0" numpy
python tests/test_vs_xgboost.py ./build/reference_cli    # golden tests (non-mutating)
python tests/test_fixture.py    ./build/reference_cli    # frozen fixtures (no xgboost needed)
```

Fixtures and `tests/RESULTS.md` are regenerated only explicitly:
`python tests/test_vs_xgboost.py ./build/reference_cli --update-fixtures --write-results`
and `python tools/make_deep_fixture.py ./build/reference_cli` for the synthetic deep31 case.

Supported model sources: XGBoost `gbtree` and `dart` (weight_drop applied),
`num_parallel_tree` ≥ 1, missing values, and exactly these objectives (each link verified
empirically end-to-end against `pred_contribs`): identity-link `reg:squarederror`,
`reg:squaredlogerror`, `reg:absoluteerror`, `reg:quantileerror`, `reg:pseudohubererror`,
`binary:logitraw`, `binary:hinge`, `multi:softmax`, `multi:softprob`; logit-link
`binary:logistic`, `reg:logistic`; log-link `count:poisson`, `reg:gamma`, `reg:tweedie`.
Anything else (survival, ranking, multi-target, categorical splits) is **rejected with a
clear error** rather than silently mis-linked. Verified against xgboost 2.0.3 and 3.1.2.

## Quickstart (Apple Silicon — Metal differential run)

```bash
# The pinned Apple metal-cpp headers are included in third_party/metal-cpp.
cmake -B build && cmake --build build
# CMake probe-compiles a tiny kernel to detect the OFFLINE Metal toolchain; without it
# (Command Line Tools only), metal_cli runtime-compiles shaders/treeshap.metal instead —
# the path exercised by the local M4 Max validation.
ctest --test-dir build -R 'metal|fixture' --output-on-failure
# Direct equivalent for all fixtures at one row-bank setting:
python tests/test_fixture.py build/reference_cli --metal-cli build/metal_cli \
  --metal-rows-per-simdgroup 256
```

The production throughput default is plain float atomics, shared model storage, 256 rows per
SIMD-group, 256 threads per threadgroup, and full dispatch (`atomic_tile_rows=0`). Use
`--accumulation deterministic` for precise-Kahan, fixed-order, bit-repeatable output (256 MiB
scratch budget by default), or `--accumulation simdgroup` to test explicit pre-aggregation on
another GPU/model. No model-shape auto-switch is enabled because atomics won every measured
workload family.

## Python `MetalTreeExplainer`

```bash
python3 -m pip install build
python3 -m build --wheel
python3 -m pip install dist/metal_treeshap-*.whl
```

```python
from metal_treeshap import MetalTreeExplainer

explainer = MetalTreeExplainer.from_xgboost("model.json")
phis = explainer.explain(X)
```

`from_paths` is also available and requires explicit per-group intercepts. The package targets
macOS 13+ on ARM64 and runtime-compiles the bundled shader when an offline metallib is absent.

## Reproduce the Phase 2 benchmark

```bash
python3 benchmarks/phase2_workloads.py stress /tmp/metal-treeshap-phase2/stress --force
python3 benchmarks/phase2_run.py build/phase2_benchmark \
  /tmp/metal-treeshap-phase2/stress --kernel shaders/treeshap.metal \
  --output /tmp/metal-treeshap-phase2/results.json \
  --rows-per-simdgroup 256,1024 --threads-per-threadgroup 64,256 \
  --accumulations atomic,simdgroup,deterministic --atomic-tiling-sweep --rounds 3
```

## Repository hygiene

This directory is designed to be a standalone git repository (a `.gitignore` for build
products is included). Do **not** commit it from a parent repository whose index may
contain unrelated files or credentials.

## How the pieces fit

```
xgboost model (or saved .json) ──tools/extract_paths.py──► paths + per-group intercepts
                                             │  Preprocess() [host, portable, validated]
                                             ▼
                              dedup → validate → BFD bin-pack → sort → segments (+fp64 bias)
                                             │
                    ┌────────────────────────┴──────────────────────┐
                    ▼ (any platform)                                ▼ (Apple Silicon)
        reference/reference_shap.h                       shaders/treeshap.metal
        scalar oracle, fp64/fp32 accum,                  atomic/SIMD/deterministic kernels,
        shuffled-order mode                              compiled-model host (persistent buffers)
                    │                                                │
                    └──────── numerically matched phis ─────────────┘
                    (golden + fixture + property + performance + wheel suites; Phase 2.1)
```
