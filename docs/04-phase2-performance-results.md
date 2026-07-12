# Phase 2 performance results

> This document preserves the original Phase 2 baseline and its raw artifacts. Phase 2.1
> subsequently added atomic row-tiling experiments, a precise Kahan deterministic reducer,
> wide-feature and multiclass workloads, Python bindings, and an optional SHAP baseline. The
> current production decisions and measurements are in
> [`05-phase21-production-results.md`](05-phase21-production-results.md).

## Material Passport

- Origin Skill: `academic-research-suite/experiment-agent`
- Mode: run + validate
- Date: 2026-07-12
- Verification Status: **VERIFIED**
- Version: `phase2_m4max_v1`

## Outcome

Phase 2 clears the project's performance gate on the tested M4 Max workload. For a
500-tree, depth-8 XGBoost model and 8,192 rows, the selected Metal configuration takes
**0.6206 s** per steady-state API call versus **12.0345 s** for XGBoost CPU
`pred_contribs=True`: **19.39× faster**, or 13,199 versus 681 rows/s. The corresponding
GPU interval is 0.6197 s; input wrapping/copy, command submission, output copy, and other
host API work are included in the 0.6206 s wall result.

Adding separately measured setup components to a warmed call gives 1.817 s for Metal versus
12.204 s for XGBoost CPU, a **6.72× setup-plus-call estimate**. This is not a direct or symmetric
model-to-first-answer measurement: Metal path extraction is excluded, while CPU load includes
the expected file. The 19.39× wall/API result clears the steady-state gate; the derived setup
estimate is orientation only. Both figures apply to this model, batch, machine, and API
comparison—not every TreeSHAP workload.

The accepted throughput default is:

```text
accumulation          atomic_float
model storage        shared
rows/SIMD-group      256
threads/threadgroup  256
```

The machine-readable summary, including workload and raw-result hashes, is
[`benchmarks/results/phase2_m4max_20260712_summary.json`](../benchmarks/results/phase2_m4max_20260712_summary.json).
The hashed raw suite JSON and row-sweep results are preserved beside it in
[`benchmarks/results/phase2_m4max_20260712_raw/`](../benchmarks/results/phase2_m4max_20260712_raw/).
That directory also preserves the exact tested shader snapshot (SHA-256
`73fffeca…1628a51`). The delivered shader differs only in status/default comments; its executable
kernel code is unchanged, and every benchmark command set the tuning values explicitly.

## Environment and workload

| item | value |
|---|---:|
| Machine | Apple M4 Max, 40-core GPU |
| CPU | 16 cores (12 performance + 4 efficiency) |
| Memory | 128 GiB |
| OS | macOS 26.2 (25C56) |
| Metal | Metal 4 device; MSL compiled at runtime with Metal 3 language mode |
| CPU baseline | XGBoost 3.1.2, OpenMP enabled, `nthread=16` |
| Trees / max depth | 500 / 8 |
| Explain rows / features / groups | 8,192 / 12 / 1 |
| Raw path elements | 521,503 |
| Packed SIMD bins | 11,618 |
| Non-root contributions per row | 296,034 |

The deterministic workload generator produced byte-identical model, paths, data, and
oracle files in two independent runs. Their SHA-256 values are pinned in the summary
JSON. Missing values occur in both training and explanation data.

## Primary CPU/Metal comparison

The Booster, DMatrix, Metal pipeline, and `CompiledModel` persist outside the measured
loops. XGBoost is invoked through Python and its timed call allocates the returned prediction
array. Metal uses the native C++ harness and reuses caller output storage, but still wraps or
copies the input on each call; the full-size input met the zero-copy alignment contract in these
runs. Both APIs include their normal output movement. Python/native wrapper overhead is most
relevant to the 1–32-row results and negligible relative to the 8,192-row compute time.

| implementation | samples | median | observed p10–p90 | rows/s |
|---|---:|---:|---:|---:|
| XGBoost CPU, 16 threads | 7 | 12.0345 s | 11.5453–12.4843 s | 681 |
| Metal wall/API | 50 across 5 shuffled rounds | 0.6206 s | 0.5975–0.6427 s | 13,199 |
| Metal GPU interval | 50 across 5 shuffled rounds | 0.6197 s | — | 13,219 |

The p10/p90 values are observed dispersion, not confidence intervals. Earlier sequential
A/B probes reversed their apparent winner when order was reversed; those probes were
discarded. Configuration order was therefore seeded and shuffled, complete sweeps were
retained, and the selected launch shape received a separate five-round confirmation.

### Batch-size crossover

These are steady-state API medians for the same 500-tree model. Setup remains excluded.

| rows | XGBoost CPU | Metal | speedup |
|---:|---:|---:|---:|
| 1 | 6.892 ms | 0.502 ms | 13.73× |
| 8 | 9.232 ms | 0.512 ms | 18.02× |
| 32 | 36.591 ms | 0.878 ms | 41.69× |
| 128 | 267.828 ms | 3.149 ms | 85.06× |
| 512 | 780.301 ms | 14.792 ms | 52.75× |
| 2,048 | 3.0331 s | 0.1605 s | 18.89× |
| 8,192 | 12.0345 s | 0.6155 s | 19.55× |

XGBoost parallelizes primarily across rows, while Metal exposes path/bin parallelism even
for a small row count; that is consistent with this large ensemble having no observed CPU
crossover in the tested range. Python/native wrapper overhead is also present, so these small-row
ratios are not pure-kernel comparisons. They do not imply that small models behave the same.

## Accumulation strategies

### Plain float atomics: accepted default

The stress-model focused run has worst max absolute deviation **7.49e-5**, worst
per-row/output-group sum deviation **8.25e-5**, and max observed pairwise repeat spread
**6.15e-5**. It passes the 1e-3 correctness gate comfortably. Atomic scheduling is not
formally deterministic: stress runs produced multiple bit hashes, even though the
adversarial hot workload happened to be bit-stable across 100 runs on this GPU.

### SIMD-group pre-aggregation: lower traffic, slower execution

On the stress model, SIMD pre-aggregation reduces the estimated output atomic count from
296,034 to 86,383 per row (**3.43×**), improves the worst paired-run absolute error from
7.20e-5 to 3.15e-5, and roughly halves repeat spread. Nonetheless, in the same shuffled
suite it takes 1.0378 s versus 0.5463 s for atomics—**1.90× slower**. On the synthetic
hot-cell workload it reduces atomics 16× and is more accurate, but still does not beat the
best atomic configuration. Apple's float-atomic path is efficient enough that the
key-matching ballots and repeated `simd_sum` operations cost more than they save here.

The mode remains exposed for future devices/workloads, but it is not the default.

### Deterministic two-stage reduction: repeatable fallback

The deterministic path writes one canonical partial per non-root element, then reduces
each `(row, group, feature)` segment in fixed path order. With a 256 MiB budget it uses
255.2 MiB active scratch, 226 rows/tile, and 37 tiles. Its stress median is 1.2905 s,
**2.08× slower** than selected atomics but still about 9.3× faster than XGBoost CPU.

It produces one bit hash and zero pairwise spread across repeated stress runs, 100 hot
runs, and host tests that vary tile size. This original plain reducer's max absolute
deviation is 1.08e-4; fixed ordering guarantees repeatability, not equality to an fp64
oracle. Phase 2.1 replaces the final summation with precise-math Kahan compensation and
reduces that deviation to 1.001e-5. Reducing the scratch budget to 64 MiB raises the tile
count from 37 to 147 and the original median to 2.2472 s, so the 256 MiB default is retained.

## Reproduction

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
ctest --test-dir build --output-on-failure

python3 benchmarks/phase2_workloads.py hot /tmp/metal-treeshap-phase2/hot --force
python3 benchmarks/phase2_workloads.py stress /tmp/metal-treeshap-phase2/stress --force

python3 benchmarks/phase2_run.py build/phase2_benchmark \
  /tmp/metal-treeshap-phase2/stress \
  --kernel shaders/treeshap.metal \
  --output /tmp/metal-treeshap-phase2/stress-suite.json \
  --rows-per-simdgroup 256,1024 \
  --threads-per-threadgroup 64,256 \
  --accumulations atomic,simdgroup,deterministic \
  --model-storage shared,private \
  --warmup 3 --iterations 7 --rounds 3

python3 benchmarks/phase2_cpu_xgboost.py \
  /tmp/metal-treeshap-phase2/stress/model.json \
  /tmp/metal-treeshap-phase2/stress/X.csv \
  --expected /tmp/metal-treeshap-phase2/stress/expected.csv \
  --output /tmp/metal-treeshap-phase2/cpu.json \
  --warmup 2 --iterations 7 --nthread 16
```

`phase2_run.py` verifies every workload hash before launching, validates the returned
configuration and shape, shuffles configuration order, records raw samples and binary /
kernel hashes, and applies both an elementwise and row/group-sum 1e-3 gate.

## Limitations and remaining work

- One M4 Max was measured. M1–M4 base/Pro and future devices may select different tuning.
- The result uses deterministic synthetic stress and hot workloads. The full upstream
  adult/covtype/cal-housing/fashion-MNIST matrix remains Phase 4 work.
- No `shap.TreeExplainer` timing was collected because the `shap` package is not installed;
  XGBoost's native exact `pred_contribs` is the current CPU baseline.
- Privileged `powermetrics` energy/temperature telemetry was unavailable. Seeded shuffling,
  outer rounds, raw samples, and explicit thermal caveats mitigate but do not erase this.
- Runtime source compilation is measured separately. Full-Xcode offline metallib packaging
  remains part of the release workflow.

## Statistical fallacy scan

Coverage: **11/11 checked**.

- Simpson's paradox, ecological fallacy, Berkson/collider bias, base-rate neglect,
  regression to the mean, and survivorship bias do not apply to this deterministic
  systems benchmark; no population or individual inference is made.
- Look-elsewhere and garden-of-forking-paths risks do apply to tuning. The complete sweep,
  selection rule, raw artifact hashes, and focused confirmation are disclosed.
- Correlation/causation and reverse causality are not invoked. Results are descriptive for
  the measured hardware/workloads.
