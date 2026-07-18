# Changelog

All notable changes to MetalTreeShap are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## Unreleased

### Performance

- `from_xgboost` model loading is 2.1× faster at stress scale (1.09 s → 0.53 s for the
  500-tree/depth-8 model, 521K path elements, M4 Max): `_pack_paths` now packs flat
  path-element attributes with one comprehension per column (6.9× faster than the
  per-element/per-field dispatch, bit-identical output) and falls back to the generic
  mapping/nested/structured-array packing on the first element that differs, and the
  packaged extractor's `PathElement` dataclass uses `slots=True` to cheapen the one
  instance created per (leaf, ancestor).
- The deterministic reduction is now a fixed-shape two-stage pass: every cell's slot
  segment is split into model-defined 256-slot chunks, stage A Kahan-sums one chunk per
  thread, and stage B combines each cell's chunk sums in fixed chunk order. This
  replaces the fully serial per-cell chain, which left only rows×cells threads in flight
  with chains of tens of thousands of dependent adds on large models. Measured on the M4
  Max stress workload (8,192 rows, 296K partials): deterministic GPU time drops from
  0.553 s to 0.228 s (2.4×), within 1.24× of atomic throughput mode. Output remains
  bitwise stable across repeats and tile sizes (hash-identical at 225-row and 56-row
  tiles), and is bit-identical to the previous reducer for cells that fit one chunk
  (verified on the deep31 fixture).

- `SortPathsByBin` now decorates each element with its bin once and sorts flat keys
  instead of doing two `std::map` lookups inside the sort comparator, and
  `GetPathLengths` counts path runs in O(1) on the sorted dedup output. Measured on the
  M4 Max at the stress scale (65,536 paths, ~310K deduplicated elements): the sort drops
  from 205 ms to 16 ms (12.8×), full `Preprocess` from 284 ms to 93 ms (3.0×), and
  `MetalTreeExplainer.from_xgboost` on the 500-tree stress model from 1.30 s to 1.09 s.
  Output order is unchanged (pinned by an exact-equivalence regression test).

### Added

- `MetalTreeExplainer.last_timings` exposes the native timing/dispatch metadata of the
  most recent call (GPU time, zero-copy status, atomic/deterministic tiling) — the
  signals needed to actually use the tuning knobs — and `trim_buffers()` releases the
  persistent native buffers a long-lived explainer retains after a peak batch.
- `from_xgboost` accepts raw JSON model text/bytes (`booster.save_raw("json")` output)
  in addition to Boosters, file paths, and parsed dictionaries.
- float64/pandas/non-contiguous inputs are converted into a page-padded buffer that the
  Metal host wraps zero-copy (`bytesNoCopy` needs a page-multiple length), removing the
  per-call staging copy of X that previously applied to essentially every real shape.
- CTest coverage for the compiled-metallib loader: on machines with the offline Metal
  toolchain, the all-fixture differential now also runs through `treeshap.metallib`
  (atomic) and its no-fast-math `treeshap_precise.metallib` sibling (deterministic),
  which previously had no test anywhere; a unit test pins the missing-file error path.

### Fixed

- Robustness batch from the repository audit: the native explainer no longer holds the
  GIL through shader compilation, preprocessing, and model upload; using an explainer in
  a forked child raises a clear `RuntimeError` instead of crashing in the Metal driver;
  `np.ma.MaskedArray` input treats masked cells as missing instead of silently using the
  backing storage; polars/xarray-style `to_numpy()` without keyword support is accepted;
  GPU and pipeline failures append the underlying `NSError` description; CSV parsing
  accepts valid subnormal values (macOS `strtof` sets `ERANGE` on underflow) and CRLF
  blank lines; the benchmark's accuracy gate serializes at full precision instead of
  `%f`'s six decimals; and `phase2_run.py` validates native results against
  `phase2_schema.json` when jsonschema is installed.
- Test hardening: golden tests now gate the fp32 accumulation error and work-order
  spread they previously only printed; the property suite asserts elementwise fp32-vs-
  fp64 deviation; fixture differentials add a 1e-4 regression tripwire under the 1e-3
  product gate; `reg:pseudohubererror` and `multi:softprob` are trained end-to-end like
  every other allowlisted objective; and the Python API suite adds negative-validation,
  concurrency, fork, masked-input, raw-JSON, and zero-copy/timings tests.
- Build hygiene: `-Wall -Wextra` everywhere (vendored metal-cpp included as SYSTEM),
  declared `.air` byproducts for the metallib rule, SPDX license identifiers on the
  wheel-shipped sources, and a CI step that installs from the sdist on the Metal runner.
- The README Python quickstart no longer instructs `pip install metal-treeshap`: the
  name is not yet registered on PyPI (RELEASING.md records the 0.1.0 check), so the
  command failed for every reader. The quickstart now leads with the source-checkout
  wheel build until the first publish lands.
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
