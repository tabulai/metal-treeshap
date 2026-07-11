# metal-treeshap

GPU-accelerated exact TreeSHAP for Apple Silicon — a port of NVIDIA's
[GPUTreeShap](https://github.com/rapidsai/gputreeshap) (Mitchell, Frank & Holmes,
[arXiv:2010.13972](https://arxiv.org/abs/2010.13972)) from CUDA to Metal.
Apache-2.0, with attribution to upstream (see LICENSE, NOTICE).

Start with the two documents in `docs/`:

1. **[docs/01-cuda-acceleration-assessment.md](docs/01-cuda-acceleration-assessment.md)** — how
   the CUDA implementation actually accelerates TreeSHAP (path decomposition, SIMD-cooperative
   extend/unwind recurrences, bin packing, mixed precision), from a close read of
   `gpu_treeshap.h`.
2. **[docs/02-apple-gpu-project-proposal.md](docs/02-apple-gpu-project-proposal.md)** — the
   project plan for the Apple GPU port: CUDA→Metal mapping, the fp64 problem and accumulation
   strategies, compiled-model architecture, phased milestones, benchmark plan, risks.

## Status (post Phase 0.5)

| component | file(s) | state |
|---|---|---|
| Path representation + 32-B GPU layout | `include/metal_treeshap/paths.h` | **built & tested** (any platform) |
| Host preprocessing + two-layer validation (raw checks BEFORE dedup so merging can't launder malformed input, structural checks after; BFD/FFD/NF packing, sort, segments, fp64 bias) | `include/metal_treeshap/preprocess.h` | **built & tested**, ASAN/UBSAN clean (external validation) |
| Scalar reference oracle (lane-faithful float recurrences, fp64/fp32 accumulation, order-shuffle mode) | `reference/reference_shap.h` | **built & tested vs xgboost** |
| XGBoost extractor: raw-JSON based — `tree_info` groups, vector base_score intercepts (3.1+), empirically-verified objective link table with explicit allowlist, DART `weight_drop`, categorical/multi-target rejection; works from a model file without xgboost | `tools/extract_paths.py` | **tested on xgboost 2.0.3 AND 3.1.2**, incl. `num_parallel_tree`, DART, and 9 objective-link cases |
| Golden tests (15 cases + rejection check; **non-mutating** — fixtures/results only change under explicit flags) | `tests/test_vs_xgboost.py` → `tests/RESULTS.md` | **passing on 2.0.3 and 3.1.2** at the 1e-3 gate |
| Property-based additivity tests (stumps, genuine long cooperative groups, deterministic 32-element comb path, repeated features, ~1e-4 covers, NaNs) | `tests/test_property_additivity.cpp` | **passing**, incl. asserted 32-lane group execution |
| Frozen fixtures replayable without xgboost (5 model cases + synthetic `deep31` 32-lane comb — the Phase-1 Metal differential target) | `tests/fixtures/*/`, `tests/test_fixture.py` | **passing** |
| Metal kernel (first-order, 64-bit work math) | `shaders/treeshap.metal` | **externally validated on M4 Max (3 rounds)**: Metal 3 compile, width 32, exact simple + multi-row-bank fixtures, and **all six frozen fixtures ≤ 6.5e-6 through the compiled-model host logic** |
| metal-cpp host: compiled-model design, hardened (exception-safe RAII construction, non-copyable owners, autorelease pools, zero-work paths, checked narrowing/products + 32-bit-grid dispatch guard, finite required intercepts, mutex-serialized Explain/tuning, runtime-source compilation) | `src/metal_host.hpp` | **host logic exercised externally on all six fixtures (v3)**; checked-in runner pending one on-device confirmation |
| Metal fixture runner (repository-reproducible validation; runtime-compiles the shader when the offline toolchain is absent) | `src/main_metal.cpp`, `fixture_metal` CTest | checked in; needs macOS + vendored metal-cpp to run |
| Benchmarks (phase-separated timings; adult ordinal-encoded with comparability caveat) | `benchmarks/benchmark_mac.py` | CPU baselines runnable; Metal hook raises until Phase 2-3 — **no acceleration claims yet** |

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
# 1. Vendor metal-cpp: download https://developer.apple.com/metal/cpp/ and unpack so
#    third_party/metal-cpp/Metal/Metal.hpp exists.
cmake -B build && cmake --build build
# CMake probe-compiles a tiny kernel to detect the OFFLINE Metal toolchain; without it
# (Command Line Tools only), metal_cli runtime-compiles shaders/treeshap.metal instead —
# the exact path used in the external M4 Max validation.
python tests/test_fixture.py build/reference_cli --metal-cli build/metal_cli
# or: ctest --test-dir build -R fixture
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
        scalar oracle, fp64/fp32 accum,                  SIMD-group kernel, atomic_float,
        shuffled-order mode                              compiled-model host (persistent buffers)
                    │                                                │
                    └───────────── identical phis ───────────────────┘
                          (golden + fixture + property suites, Phase 1 exit)
```
