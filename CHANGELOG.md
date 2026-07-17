# Changelog

All notable changes to MetalTreeShap are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## Unreleased

### Performance

- `SortPathsByBin` now decorates each element with its bin once and sorts flat keys
  instead of doing two `std::map` lookups inside the sort comparator, and
  `GetPathLengths` counts path runs in O(1) on the sorted dedup output. Measured on the
  M4 Max at the stress scale (65,536 paths, ~310K deduplicated elements): the sort drops
  from 205 ms to 16 ms (12.8×), full `Preprocess` from 284 ms to 93 ms (3.0×), and
  `MetalTreeExplainer.from_xgboost` on the 500-tree stress model from 1.30 s to 1.09 s.
  Output order is unchanged (pinned by an exact-equivalence regression test).

### Added

- CTest coverage for the compiled-metallib loader: on machines with the offline Metal
  toolchain, the all-fixture differential now also runs through `treeshap.metallib`
  (atomic) and its no-fast-math `treeshap_precise.metallib` sibling (deterministic),
  which previously had no test anywhere; a unit test pins the missing-file error path.

### Fixed

- Missing-value routing in the Metal kernel no longer depends on `isnan()` surviving
  fast math. The recurrence kernels compile with fast math, whose no-NaN assumption is
  demonstrably active (`x != x` folds to false under the default options); `isnan()`
  currently works only because the builtin is special-cased, which a future OS Metal
  compiler need not preserve. The NaN test is now an integer bit compare that no float
  math mode can fold.
- The CLI loaders now require the documented `paths.csv` header row instead of blindly
  discarding the first line. A headerless file — a plausible mistake, since `X.csv` in
  the same command is headerless — used to lose its first path element silently and exit
  0 with numerically wrong attributions; it is now rejected with an error naming the
  expected header.
- Rejected `Explain` calls now raise a catchable exception instead of aborting the
  process. Oversized dispatches, undersized deterministic scratch budgets, and scratch
  allocation failures previously threw between `computeCommandEncoder()` and
  `endEncoding()`; draining the autorelease pool during unwinding then released the
  un-ended encoder and tripped Metal's hard "Command encoder released without
  endEncoding" abort — killing the host process, including the Python interpreter behind
  the wheel. All throwing validation and allocation now runs before the encoder opens,
  and a new `EndEncodingGuard` closes the encoder during unwinding as defense in depth.
- `+inf` feature values now follow the branch XGBoost takes for any value above every
  finite threshold. Previously `+inf` satisfied no half-open split interval
  (`inf < inf` is false) in both the CPU reference and the Metal kernel, so affected rows
  silently produced non-additive attributions that matched no model prediction; the
  CPU/GPU differential suite could not catch it because both sides agreed. The Metal
  kernel uses an integer bit compare so fast math cannot fold the infinity test. Pinned by
  unit, Metal-differential, and sentinel-based golden tests (`-inf` and NaN routing were
  already correct and are now pinned too).

## 0.1.0 — 2026-07-12

Initial alpha release for Apple Silicon Macs.

### Added

- Metal implementation of first-order TreeSHAP using 32-lane SIMD-group cooperation.
- Compile-once, explain-many `MetalTreeExplainer` Python API for XGBoost JSON models,
  `Booster` objects, sklearn wrappers, and pre-extracted paths.
- Familiar `shap_values` compatibility method, bias-only/zero-tree model support, and nullable
  pandas missing-value conversion without making pandas a required dependency.
- Atomic throughput mode, SIMD pre-aggregation experiments, and a bit-repeatable
  deterministic mode with a precise Kahan reducer.
- CPU preprocessing, path packing, frozen model fixtures, analytic and double-precision
  references, and Metal differential tests.
- Persistent performance harnesses with hashed workloads, blocked/shuffled execution,
  XGBoost and optional `shap.TreeExplainer` baselines, and `powermetrics` integration.
- Paired real-data CPU/Metal runner with one shared model per cell, randomized call order,
  provenance hashes, exact power windows, and elementwise correctness gates.
- Linux portability tests, XGBoost compatibility tests, Apple-GPU tests, macOS ARM64
  wheel builds, and trusted-publishing release automation.

### Performance

- On the measured M4 Max workloads, atomic accumulation was 19.39× faster than the
  original paired 16-thread XGBoost 3.1.2 stress baseline, 22.63× on a 256-feature
  regression workload, and 21.98× on an eight-class workload.
- These are device- and workload-specific measurements, not cross-device guarantees.
  Raw artifacts and limitations are recorded under `benchmarks/results/` and `docs/`.

### Limitations

- Apple Silicon/macOS 13 or newer is required for the native package.
- First-order contributions are supported; interaction values are not yet implemented.
- XGBoost is the only model extractor in this release. Unsupported objectives and
  categorical or multi-target models are rejected explicitly.
- GPU floating-point accumulation is FP32. Deterministic mode improves repeatability and
  reduction accuracy but cannot make the FP32 path recurrence equivalent to FP64.
