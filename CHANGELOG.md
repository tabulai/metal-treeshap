# Changelog

All notable changes to MetalTreeShap are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## 0.1.0 — 2026-07-12

Initial alpha release for Apple Silicon Macs.

### Added

- Metal implementation of first-order TreeSHAP using 32-lane SIMD-group cooperation.
- Compile-once, explain-many `MetalTreeExplainer` Python API for XGBoost JSON models,
  `Booster` objects, sklearn wrappers, and pre-extracted paths.
- Atomic throughput mode, SIMD pre-aggregation experiments, and a bit-repeatable
  deterministic mode with a precise Kahan reducer.
- CPU preprocessing, path packing, frozen model fixtures, analytic and double-precision
  references, and Metal differential tests.
- Persistent performance harnesses with hashed workloads, blocked/shuffled execution,
  XGBoost and optional `shap.TreeExplainer` baselines, and `powermetrics` integration.
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
