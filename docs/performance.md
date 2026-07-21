# Performance: results, methodology, reproduction

## Headline results

Sustained, paired, randomized A/B measurements on an Apple M4 Max (macOS 26), against
16-thread XGBoost 3.1.2 `pred_contribs` on identical models and inputs:

| workload | shape | XGBoost CPU | Metal (atomic) | speedup |
|---|---|---|---|---|
| stress | 500 trees × depth 8, 12 features, 8,192 rows | 8.31 s | 0.449 s | **18.5×** |
| wide | 256 features | 3.06 s | 0.213 s | **14.4×** |
| multiclass | 8 classes | 9.27 s | 0.594 s | **15.6×** |

The deterministic mode (bit-repeatable output) measures ~1.8× the atomic mode's time
on the stress workload and much closer on wide/multiclass models, whose many output
cells keep its reducer parallel.

Smaller models at batch-scoring volume can measure far higher ratios — the executed
[examples](../examples/README.md) notebooks show 20–80× on lighter ensembles — because
CPU cost scales with rows × trees × depth² while the GPU stays latency-bound longer.

**These are measurements, not guarantees.** They are specific to the device, the
thermal state, and the workload shape.

## Methodology — and why it matters

Two lessons this repository learned the hard way, both baked into its harnesses:

1. **Pair and interleave, or your ratios lie.** Reruns of the *same* CPU baseline on
   byte-identical inputs have measured 31–35% apart across sessions (clocks, thermal
   state, background load), and GPU clocks drift within a sustained run. Every
   benchmark here runs the engines back-to-back in randomized order inside each
   iteration, so drift hits all engines equally. Only paired, same-session ratios are
   treated as meaningful; absolute numbers from different sessions are never compared.
2. **Burst numbers overstate.** Short idle-machine runs ride boosted clocks; the
   figures above come from sustained runs. Where a burst figure is quoted anywhere in
   this repository, it is labeled as such.

Additional hygiene: one warmup per configuration; medians reported with raw samples
retained; `DMatrix` construction excluded from CPU timings; elementwise CPU↔GPU
agreement asserted (≤1e-3 gate, ~1e-5 observed) in the same run that produces any
timing; deterministic runs additionally assert one output hash per configuration.

## Tuning

The defaults (`atomic` accumulation, 256 rows per SIMD-group, 256-thread
threadgroups, shared model storage, full dispatch) were selected by measurement on M4
Max and are the right starting point everywhere. If you tune:

- `explainer.last_timings` reports GPU time, zero-copy state, and the tiling actually
  used — measure before and after.
- `accumulation="deterministic"` buys bit-identical reruns and tighter reduction
  accuracy for ~1.8× time; its `deterministic_scratch_mib` budget only changes
  tiling, never output bits.
- Row-tiling experiments (`atomic_tile_rows`) did not beat full dispatch on any
  measured workload family; the knob remains for other hardware.

## Reproducing

Generate a workload and run the persistent benchmark (all three modes, tiling sweep):

```bash
python3 benchmarks/phase2_workloads.py stress /tmp/mts-bench/stress --force
python3 benchmarks/phase2_run.py build/phase2_benchmark /tmp/mts-bench/stress \
  --kernel shaders/treeshap.metal --output /tmp/mts-bench/results.json \
  --rows-per-simdgroup 256,1024 --threads-per-threadgroup 64,256 \
  --accumulations atomic,simdgroup,deterministic --rounds 3
```

The real-data runner trains models on OpenML-style datasets, interleaves paired
CPU/Metal calls, and records hashes, additivity, elementwise error, and exact UTC
power windows:

```bash
python3 benchmarks/benchmark_mac.py \
  --datasets adult,covtype,cal_housing,fashion_mnist \
  --sizes small,med --nrows 10000 --niter 5 --warmup 2 \
  --nthread 16 --device both --output realdata.json
```

Artifacts are schema-validated (`benchmarks/phase2_schema.json`,
`benchmarks/realdata_schema.json`); the raw archives behind the table above live under
`benchmarks/results/`.
