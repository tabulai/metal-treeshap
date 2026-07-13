# Phase 2.1 production results

Phase 2.1 tested the two strongest follow-ups to the original 19.39× M4 Max result, widened
the workload matrix, and delivered the first pip-installable `MetalTreeExplainer`. The
production throughput default remains direct float atomics with a full-batch dispatch.

## Material Passport

- Origin Skill: `academic-research-suite/experiment-agent`
- Origin Mode: `run + validate`
- Origin Date: `2026-07-12`
- Verification Status: `VERIFIED`
- Version Label: `phase21_production_v1`

The machine was an Apple M4 Max running macOS 26.2. The source baseline was commit `d9775da`,
with Phase 2.1 changes on `codex/phase2-production`. The offline Metal compiler was unavailable,
so the tested host compiled `treeshap.metal` at runtime. The recurrence kernels used fast math;
the deterministic reducer came from a second library compiled with fast math disabled.

## Production defaults

| Control | Default | Evidence |
|---|---:|---|
| accumulation | `atomic` | Won stress, wide-feature, and multiclass throughput tests |
| model storage | `shared` | Original Phase 2 selection; unchanged |
| rows per SIMD-group | `256` | Original Phase 2 selection; unchanged |
| threads per threadgroup | `256` | Original Phase 2 selection; unchanged |
| atomic tile rows | `0` (full dispatch) | Tiled finalists did not win blocked rounds |
| deterministic scratch | 256 MiB | Preserves the established bounded-memory plan |

There is deliberately no model-shape heuristic that switches to SIMD pre-aggregation or the
deterministic path. Atomic accumulation won all three measured workload families. The
alternative modes remain explicit API controls: SIMD for future-device experiments, and
deterministic Kahan when repeatability and lower accumulation error matter more than latency.

## Atomic row tiling

The initial six-way sweep covered full dispatch and 256, 512, 1,024, 2,048, and 4,096-row
tiles. It showed a possible 1.2× gain but also a strong machine-state trend. A focused rerun
therefore used seven temporal blocks; each block contained full, 2,048, and 4,096-row jobs in
an independently shuffled order. Every job used three warmups and five timed calls.

| Dispatch | Median of seven job medians | Median paired speedup | Block wins |
|---|---:|---:|---:|
| full | 0.4964 s | 1.000× | — |
| 2,048 rows | 0.5073 s | 0.993× | 3/7 |
| 4,096 rows | 0.5045 s | 0.983× | 1/7 |

The first block was substantially slower for every configuration, which explains much of the
earlier apparent cache cliff. Row tiling did not recover a stable advantage. It remains
implemented as `atomic_tile_rows` for other devices, but `0` is the production default.

## Precise deterministic reduction

Kahan compensation is safe because one reducer thread exclusively owns each
`(row, group, feature)` segment. A naïve implementation under Metal fast math is ineffective:
the compiler can reassociate the correction away. The host now builds only the reduction
pipeline from a separate precise library; offline builds produce a sibling
`treeshap_precise.metallib`.

On the 500-tree stress workload:

| Reducer | Wall median | Max absolute error | Repeatability |
|---|---:|---:|---:|
| original serial sum | 1.2905 s¹ | 1.078e-4 | one hash |
| precise Kahan | 1.3599 s | 1.001e-5 | one hash in every job |

¹ The original time is from the Phase 2 session, not a same-process A/B run. The accuracy
comparison is stable: Kahan reduces worst-case error about 10.8× and mean error to 1.11e-7.

## Wider workload matrix

Both new workloads used 8,192 rows, three shuffled outer-round blocks, three warmups, and five
timed calls per job. CPU baselines used 16-thread XGBoost 3.1.2 with model and `DMatrix` setup
outside the timed call.

| Workload | Atomic | SIMD | Deterministic | XGBoost CPU | Atomic speedup |
|---|---:|---:|---:|---:|---:|
| 400 depth-6 trees, 256 features | 0.2046 s | 0.4752 s | 0.3078 s | 4.6290 s | 22.63× |
| 150 rounds, 8 classes, 32 features | 0.6447 s | 1.1795 s | 0.9149 s | 14.1686 s | 21.98× |

SIMD pre-aggregation was 2.32× slower than atomics on the wide model and 1.83× slower on
multiclass. Deterministic Kahan was 1.50× and 1.42× slower, respectively, while lowering max
absolute error from roughly 8e-6 to 1–3e-6. These results support atomics as the throughput
default and deterministic Kahan as the explicit accuracy/repeatability option.

## Optional SHAP comparison

The optional baseline used SHAP 0.52.0 with XGBoost 3.3.0, 16 CPU threads, one warmup, and
three timed 8,192-row calls on the stress workload. Backend provenance identifies
`shap.explainers._tree.TreeExplainer` with an XGBoost `TreeEnsemble`; this is a compiled
XGBoost-specific path, not an independent pure-Python algorithm.

| Implementation | Median |
|---|---:|
| `shap.TreeExplainer.shap_values` | 7.6039 s |
| native XGBoost 3.3 `pred_contribs` | 7.3767 s |
| Metal atomic, full dispatch | 0.4964 s |

This gives a 15.3× comparison to `TreeExplainer` and 14.9× to the contemporaneous XGBoost 3.3
call. It does not replace the original 19.39× headline, which used XGBoost 3.1.2 and its own
paired measurement session.

## Power measurement status

`phase2_run.py` can launch `powermetrics` without ever prompting for a password, record each
job's UTC interval, and attach per-job CPU/GPU mean power and estimated energy. The separate
`phase2_power.py` parser also handles NUL-separated plist traces and reports missing coverage.
The CPU XGBoost and optional SHAP baselines emit aggregate envelopes plus exact per-call UTC
windows using the same parser contract. `benchmark_mac.py` additionally uses homogeneous
repeated-call CPU and Metal blocks, idle guards, and one sampler interval of the same workload
before and after each recorded block; this prevents a boundary sample from mixing engines.
The software instrumentation is complete and tested with synthetic traces. A real privileged
trace is still required before making any performance-per-watt claim.

This session had no passwordless `sudo`, so power capture was explicitly recorded as skipped.
No performance-per-watt claim is made. To capture it on an authorized Mac:

```bash
sudo -v
# Controlled CPU-versus-Metal energy comparison: homogeneous engine blocks,
# randomized block order, and sampler-interval boundary conditioning.
python3 benchmarks/benchmark_mac.py \
  --datasets cal_housing,adult --sizes small,med \
  --nrows 10000 --niter 5 --warmup 2 --nthread 16 --device both \
  --output realdata-power.json --power-output realdata-power.plist \
  --power-sudo

# Kernel-configuration-only power comparison.
python3 benchmarks/phase2_run.py build/phase2_benchmark WORKLOAD \
  --kernel shaders/treeshap.metal --output suite.json \
  --power-output powermetrics.plist --power-sudo
python3 benchmarks/phase2_power.py suite.json powermetrics.plist \
  --output power-summary.json
```

## Python package

The scikit-build-core/nanobind package builds a macOS ARM64 wheel and exposes:

```python
import xgboost as xgb
from metal_treeshap import MetalTreeExplainer

booster = xgb.Booster(model_file="model.json")
explainer = MetalTreeExplainer.from_xgboost(booster)
phis = explainer.explain(X)  # (rows, features + 1); group axis retained for multiclass
```

`from_paths` requires explicit intercepts. `from_xgboost` accepts JSON paths, parsed model
dictionaries, `Booster` objects, and sklearn wrappers exposing `get_booster()`. The wheel
includes the exact validated extractor and Metal shader, and exposes accumulation, storage,
row-bank, threadgroup, deterministic-scratch, and atomic-tiling controls.

## Reproduction and artifacts

The concise machine-readable result is
`benchmarks/results/phase21_m4max_20260712_summary.json`. Raw suites, CPU comparisons, the
optional SHAP result, and the tested shader snapshot are in
`benchmarks/results/phase21_m4max_20260712_raw/`; their SHA-256 hashes are recorded in the
summary.

```bash
# Generate the additional workloads.
python3 benchmarks/phase2_workloads.py wide /tmp/mtshap/wide --force
python3 benchmarks/phase2_workloads.py multiclass /tmp/mtshap/multiclass --force

# Re-run the bounded tiling matrix with blocked randomized order.
python3 benchmarks/phase2_run.py build/phase2_benchmark /tmp/mtshap/stress \
  --kernel shaders/treeshap.metal --output tiling.json \
  --accumulations atomic --atomic-tiling-sweep --rounds 3

# Build and test the wheel.
python3 -m build --wheel
python3 -m pip install dist/metal_treeshap-*.whl
python3 tests/test_python_api.py
```

Timing is environment-sensitive. Configuration order was blocked and shuffled, but GPU clocks,
temperature, and system load were not directly observed because privileged telemetry was
unavailable. Treat the mode ordering as robust on this M4 Max; do not treat the exact medians as
cross-device predictions.

## Validation and fallacy scan

All 11 experiment-agent fallacy categories were checked. Simpson's paradox, ecological
inference, Berkson selection, collider adjustment, base-rate neglect, regression to the mean,
survivorship bias, correlation/causation, and reverse causality are not applicable to these
direct executable comparisons. The look-elsewhere effect and garden-of-forking-paths risks are
material: the initial tuning matrix was exploratory, so production tiling was decided only from
the separately blocked finalist run, and no p-value is claimed. The result remains
environment-sensitive and device-specific.
