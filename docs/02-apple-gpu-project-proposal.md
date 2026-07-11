# MetalTreeShap — GPU-Accelerated TreeSHAP on Apple Silicon

*A project proposal for porting the GPUTreeShap algorithm (Mitchell, Frank & Holmes) from CUDA to
Apple GPUs via Metal. Companion to `01-cuda-acceleration-assessment.md`, which analyzes the CUDA
implementation this project ports.*

---

## 1. Motivation

There is currently **no GPU-accelerated SHAP implementation for Apple Silicon**. On a Mac,
`xgboost.predict(pred_contribs=True)`, `shap.TreeExplainer`, and LightGBM's contribution mode all
run on CPU; `shap.explainers.GPUTree` and XGBoost's `device="gpu"` path hard-require CUDA. XGBoost
has explicitly declined Metal support (GitHub issue #2440), and a search for prior Metal/MLX
TreeSHAP work turns up nothing. Meanwhile a large fraction of practicing data scientists do daily
work on M-series MacBooks with a capable, mostly idle GPU: an M4 Max's 40-core GPU delivers on the
order of ~15-18 TFLOPS FP32 with 546 GB/s of unified-memory bandwidth — the same compute class as
the V100 on which GPUTreeShap published 13-19× speedups over 40 Xeon cores. (Unified memory
removes the PCIe staging that discrete GPUs pay; it does not make buffer handling free — see §5.)

The opportunity is attractive for three reasons. First, the algorithm is a *proven* GPU win — the
hard science (path decomposition, warp-cooperative recurrences, bin packing) is done and
published; this is a porting-and-engineering project, not a research gamble. Second, the port is
unusually well-matched: Apple GPUs execute in **32-wide SIMD-groups — exactly CUDA's warp width**
— and the algorithm's one hard constraint (path length ≤ 32) transfers unchanged. Third, Apple's
unified memory removes GPUTreeShap's main real-world tax: there is no PCIe transfer of the dataset
or the phis output at all.

The result would be a genuinely novel open-source contribution: `pip install metaltreeshap`, and
every M-series Mac explains tree models several times faster, at a fraction of the energy.

## 2. Goals, non-goals, success criteria

**Goal.** A production-quality, pip-installable library computing exact first-order SHAP values
(and later interactions / Taylor / interventional variants) for XGBoost, LightGBM and sklearn tree
ensembles on Apple-Silicon GPUs, numerically validated against the CPU reference implementations.

**Non-goals (initially).** Training acceleration; categorical-split support beyond what interval
conditions express (XGBoost one-hot/partition categoricals can be added later exactly as the
CUDA templated `SplitConditionT` anticipates); Intel-Mac AMD GPUs (Metal-capable but different
SIMD width economics; possible later since MSL code is family-gated, not impossible); Windows/Linux.

**Success criteria.**

1. *Correctness*: max |Δphi| ≤ 1e-3 (and sum-to-margin residual ≤ 1e-3) vs `xgboost`
   `pred_contribs` on the four benchmark datasets (adult, covtype, cal_housing, fashion_mnist) at
   small/med/large model sizes — same acceptance style the upstream repo uses.
2. *Performance*: **hard gate ≥ 2× end-to-end** (compiled-model steady state, setup amortized)
   over multithreaded CPU `pred_contribs` on the same machine (M-series Pro/Max) for medium and
   large models at 10K rows — with 5×+ as the kernel-throughput target; document energy via
   `powermetrics`. No acceleration claims before the Metal benchmark path actually runs.
3. *Usability*: `MetalTreeExplainer(model).shap_values(X)` one-liner; wheels on PyPI; CI on an
   Apple-Silicon runner.

## 3. Feasibility: mapping the CUDA machinery onto Metal

The assessment doc identified six acceleration levers. Five carry over almost mechanically; one
(fp64 accumulation) needs real design work. The table below is the Rosetta stone for the port —
"MSL" is the Metal Shading Language; all listed MSL features are supported on every Apple-Silicon
Mac (Apple7 GPU family / M1 and newer, Metal 3, macOS 13+).

| CUDA construct (as used in gpu_treeshap.h) | Metal equivalent | Notes |
|---|---|---|
| warp = 32 threads | SIMD-group = 32 threads | Exact width match on Apple GPUs; `thread_index_in_simdgroup` |
| `__shfl_sync(mask, v, lane)` | `simd_shuffle(v, lane)` | No mask arg; inactive-source reads are undefined → only shuffle within the (active) group, which the algorithm already guarantees |
| `__shfl_up_sync(mask, v, 1)` | `simd_shuffle(v, max(lane-1, group_start))` | MSL `simd_shuffle_up` is undefined for low lanes (CUDA returns own value) → use clamped-index shuffle |
| `__ballot_sync` | `simd_ballot()` → `simd_vote` cast to uint64/uint32 | |
| `__match_any_sync(mask, label)` (labeled partition) | Not available — replace with boundary trick | Elements are sorted by path within a bin, so groups are contiguous: ballot the lanes where `path_idx != prev`, then each lane finds its group start/end with `clz`/`ctz` bit math. Cheaper than the CUDA pre-Volta fallback loop |
| `__popc`, `__ffs`, `__clz` | `popcount()`, `ctz()`, `clz()` | Native MSL integer functions |
| `lanemask_lt` (inline PTX) | `(1u << lane) - 1` | Trivial |
| 64-bit packed 2×float shuffle | `simd_shuffle` on `float2` | MSL shuffles support vector types |
| `atomicAdd(double*)` | **No fp64 on Apple GPUs at all** | The one hard problem — see §4 |
| `atomicAdd(float*)` | `atomic_fetch_add_explicit(device atomic_float*, …)` | Native since Metal 3 (macOS 13); hardware float atomics |
| `__shared__` + `__syncthreads()` | `threadgroup` memory + `threadgroup_barrier()` | ~32 KB/threadgroup; PathElement staging (256 × ~32 B = 8 KB) fits easily |
| `__launch_bounds__(256)` | `maxTotalThreadsPerThreadgroup` / `[[max_total_threads_per_threadgroup(256)]]` | Also query `threadExecutionWidth` (=32) at PSO creation |
| `__fmul_rn/__fmaf_rn/__fdividef` | `fma()`, `fast::divide` or `-ffast-math` per-file | MSL fast-math is default; pin precise where needed |
| kernel launch `<<<grid, block>>>` | `dispatchThreadgroups` on a compute command encoder | |
| `thrust::sort/reduce_by_key/scan`, `cub::ReduceByKey` (device preprocessing) | **Move to host CPU** (`std::sort`, hand-rolled reduce/scan) | Justified below; MPSGraph/MLX primitives are a later option |
| `cudaMallocHost` / H2D copies | No PCIe staging — `MTLBuffer` `storageModeShared`; zero-copy only via `newBuffer(bytesNoCopy:)` (page-aligned, page-multiple) | Beware: `newBuffer(bytes:)` **copies**; the host falls back to a persistent staging buffer when alignment isn't met |
| CUDA streams | `MTLCommandQueue` + command buffers | |
| nvcc header-only template library | `.metal` → `.metallib` (offline) or runtime-compiled source; host in C++ via **metal-cpp** | Template split conditions become function-constant / preprocessor specializations of the kernel |

Moving preprocessing to the CPU deserves a word: the CUDA version preprocesses on-device chiefly
because the paths often already live in GPU memory (XGBoost trains there) and PCIe round-trips are
expensive. On Apple Silicon neither reason applies — memory is shared and the performance cores
are excellent at a one-shot O(model) sort/dedup/pack. This deletes the thrust/cub dependency, the
single largest source of porting surface area, at near-zero cost. (BFD bin packing already runs on
the host in the CUDA version.) If profiling ever shows preprocessing to matter for huge ensembles,
it can move to MPSGraph/MLX sort-scan primitives later.

Two Apple-specific execution-model facts derisk the kernel semantics: Apple GPUs run SIMD-groups
in lockstep with divergence handled by masking (no independent thread scheduling), and the
algorithm is already written for explicit-mask lockstep execution — it predates and never relies
on Volta ITS. And early-returning inactive lanes (`thread_active = false`) are safe because every
subsequent shuffle reads only from lanes that remained active, which MSL permits.

## 4. The one hard problem: no fp64

Apple GPUs have no double type in shaders (fp64 emulation exists as a third-party curiosity at
~1:68 throughput; not viable). GPUTreeShap uses fp64 in exactly three places, each with a
different right answer on Metal:

**(a) `phis` accumulation (`atomicAddDouble`).** The real issue. A SHAP value is a sum of many
signed, partially-cancelling path contributions; fp32 accumulation error grows with model size.
Three implementations to build and compare in Phase 2:

1. **Plain fp32 `atomic_float` (fast mode, initial default candidate).** Hardware-accelerated on
   Apple Silicon. Collisions on a given `phis[row, feature]` cell come only from distinct
   path-groups that share the feature (within a SIMD-group each lane writes a different feature;
   across the row loop each iteration writes a different row), which limits contention — but
   *"limited" is not "zero"*: root-level and other high-level split features recur across most
   paths of a tree, so hot features exist by construction.
2. **SIMD-group pre-aggregation before atomics.** Because packed paths within a bin frequently
   share those hot root/high-level features, aggregating contributions by `(group, feature)`
   inside the SIMD-group (shuffle-based segmented reduction) before touching global memory can
   meaningfully cut atomic traffic on exactly the cells most prone to contention. Whether the
   shuffle cost pays for the atomic savings is an empirical Phase-2 question.
3. **Two-stage deterministic reduction (deterministic mode).** Kernel writes privatized partials
   (per threadgroup or per bin-slice), a second fixed-order reduction pass combines them.
   Run-to-run bit-stable — the right mode for CI and regulated settings, at extra memory and a
   second dispatch.

A **fixed-point split-hi/lo accumulator** (scale by 2^k, `atomic_fetch_add` on a low `atomic_uint`
word, propagate carries into a high word via overflow detection on the returned old value) remains
a sound bit-exact alternative to (3) if its quantization (2^-k · range) is acceptable. A
Neumaier/two-float compensated *atomic* variant, floated in an earlier draft, is **withdrawn as
unsound**: a 32-bit CAS cannot atomically update a value-plus-compensation pair, and interleaved
writers corrupt the compensation term.

The evidence so far (§9): sequential-CPU fp32 accumulation on a 500-tree depth-8 model shows max
error 3.8e-5 absolute / 6.5e-5 elementwise-relative (floored), and shuffling the accumulation
order across 5 seeds moves results by ≤4.2e-5 — four orders of magnitude below upstream's own
`np.allclose(…, 1e-1, 1e-1)` acceptance bar. Encouraging, but it is a *CPU proxy*: it does not
measure true concurrent interleaving, sustained hot-cell contention, or run-to-run variance on
device. The paper's history is also friendly — early GPUTreeShap accumulated in fp32; upstream
moved to fp64 because on CUDA it was nearly free, not because fp32 was catastrophic. **Status:
fp32 atomics are the initial fast mode; the accepted production default is decided in Phase 2
from the three-way on-device comparison (accuracy, determinism, throughput), including a
hot-feature adversarial model and ≥100-run repeatability.**

**(b) `zero_fraction` stored as double.** It's a probability (cover ratio); computed on host in
fp64, *stored* fp32 in the GPU path buffer. The per-path product that matters (bias) never runs on
GPU (see c). Kernel math already ran fp32 in CUDA. Low risk; validated by the golden tests.

**(c) Bias / expected value (`ComputeBias`, fp64 reduce_by_key).** Runs on host CPU in fp64
(it's O(model), trivially cheap) and is *added* into the phis output buffer before kernel launch,
exactly as upstream does with its `temp_phi` initialization. Zero GPU involvement. Solved by the
architecture.

**(d) `W(s,n)` table for interventional SHAP** (`lgamma`-based). Precompute the 33×33 table on
host in fp64, truncate to fp32, upload as a constant buffer (upstream already truncates to fp32 in
shared memory). Solved.

## 5. Architecture

```
metal-treeshap/
├── include/metal_treeshap/        # portable C++20, no Metal/CUDA types
│   ├── paths.h                    #   PathElement, XgboostSplitCondition
│   └── preprocess.h               #   dedup, BFD/NF bin packing, sort, segments, bias
├── reference/reference_shap.h     # scalar CPU oracle (float compute, fp32/fp64 accumulation)
├── shaders/treeshap.metal         # ShapKernel port (SIMD-group cooperative)
├── src/metal_host.hpp             # metal-cpp: device, PSO, buffers, dispatch
├── tools/extract_paths.py         # XGBoost JSON dump → path elements (covers → zero_fractions)
├── tests/                         # preprocess unit tests; golden tests vs xgboost pred_contribs
├── benchmarks/benchmark_mac.py    # upstream benchmark.py adapted: CPU baselines vs Metal
└── python/ (Phase 3)              # nanobind bindings → metaltreeshap wheel
```

Data flow: `Booster` → (Python/C++ extractor) raw path elements + per-group intercepts → host
preprocess (validate raw → dedup → validate merged → BFD pack → sort → segments; bias in fp64) →
persistent shared `MTLBuffer`s (elements, segments; dataset zero-copy only when page-aligned and
page-multiple, else a persistent staging copy) → one compute dispatch of `ceil(bins·banks/8)`
threadgroups × 256 threads → result copied **once** from the shared output buffer into the
caller's array (a zero-copy buffer-view API is a Phase-3 option; do not call this path
"no copy").

The kernel keeps upstream's exact decomposition — lane = path element, SIMD-group = bin of paths ×
bank of rows, `rows_per_simdgroup` as a function constant (default 1,024, tuned in Phase 2) — and
replaces `active_labeled_partition` with the sorted-contiguity boundary computation:

```metal
// lanes with a new path_idx mark group boundaries (lane 0 always does). This mirrors
// shaders/treeshap.metal exactly — note the masks are built from (1u << (lane+1)) - 1,
// i.e. INCLUSIVE of the current lane (an exclusive mask would make a path's first lane
// select the previous group's boundary), with lane 31 special-cased because 1u << 32 is
// undefined.
bool boundary   = (lane == 0) || (path_idx != simd_shuffle(path_idx, lane - 1));
uint bmask      = uint(uint64_t(simd_ballot(boundary)));
uint below_inc  = bmask &  ((lane == 31) ? 0xFFFFFFFFu : ((1u << (lane + 1)) - 1u));
uint above      = bmask & ~((lane == 31) ? 0xFFFFFFFFu : ((1u << (lane + 1)) - 1u));
uint start      = 31 - clz(below_inc);                  // highest boundary at or below me
uint end        = (above != 0) ? ctz(above) : n_active; // next boundary above me, or end
ContiguousGroup g { start, end - start, lane - start }; // shfl(v,i) = simd_shuffle(v, start+i)
```

`GroupPath::Extend` / `UnwoundPathSum` / `ComputePhi` port line-for-line (float2 broadcast for the
packed pair, `fma`, clamped-index shuffle for `shfl_up`). Split-condition "templates" become
preprocessor variants of the kernel source (one specialization — XGBoost intervals — at first).

**Host/tooling choice — metal-cpp vs MLX vs Swift.** Compared:

- *metal-cpp* (Apple's official C++ wrapper): closest to the CUDA host code, zero Objective-C,
  easy to wrap with nanobind for Python, easy CMake integration, full control over PSOs, function
  constants, `bytesNoCopy` buffers. The **production core**.
- *MLX `mx.fast.metal_kernel`*: JIT-compiles a kernel body from Python, manages buffers, supports
  `atomic_outputs` — a convenient optional Python experimentation harness.
- *Swift + Metal*: most native, worst Python-ecosystem fit; pass.

**Recommendation (revised after external validation): go directly to metal-cpp.** The kernel has
since been compiled and executed on an M4 Max (§9) — Metal 3 compilation, 32-lane execution
width, exact results on a simple fixture and 5.96e-7 on a packed multi-path fixture — so the
original motivation for an MLX intermediate step (validate MSL semantics before writing host
code) is spent. Implementing the kernel twice buys nothing; MLX remains an optional harness for
quick Python-side experiments, not a required phase.

The host follows a **compiled-model design** (implemented in draft in `src/metal_host.hpp`):
`Explainer::Compile()` does all O(model) work once — preprocess, validate, pack, upload
persistent element/segment buffers, fold path bias + model intercept per group — and
`Explain()` reuses a growable output buffer and reports per-phase timings (upload, encode, GPU,
total). One correction to an earlier claim: unified memory removes PCIe *staging*, but it does
not make buffer handling free — `newBuffer(bytes:length:options:)` **copies** its input; only
`newBuffer(bytesNoCopy:)` wraps caller memory, and it requires a page-aligned pointer and
page-multiple length (the host takes the zero-copy path when eligible and stages through a
persistent upload buffer otherwise). Explanation-time benchmarks must therefore measure the
compiled-model steady state, not per-call setup.

## 6. Phased plan

**Phase 0 — Portable core + oracle. DONE.**
Repo scaffold; `PathElement`/split conditions and full host preprocessing (dedup, BFD/FFD/NF
packing, sort, segments, fp64 bias) in portable C++; scalar reference implementation of the
extend/unwind algorithm; XGBoost path extractor; golden test harness proving
extractor + preprocess + reference match `pred_contribs` (sum-to-margin and elementwise); fp32
accumulation-error study. *Everything in this phase runs on any machine — it was built and tested
on Linux — because it deliberately contains no Metal.*

**Phase 0.5 — Correctness hardening (added after external review). DONE in this revision.**
Fixes and additions from the first validation round, all verified in-sandbox:
extractor rewritten on the raw JSON model (works from a file with no xgboost installed):
`tree_info` as the authoritative tree→group mapping (round-robin by index is wrong under
`num_parallel_tree`), vector-valued `base_score` intercepts (XGBoost 3.1+ serializes e.g.
`'[3.3E-1,3.775E-1,2.925E-1]'`), DART `weight_drop` leaf scaling, explicit categorical-split
rejection; per-group model intercepts plumbed through preprocessing, reference CLI and host so
the public API returns complete contributions with no post-hoc patching; validation hardened
(exactly-one-root, feature/group ranges, fraction/leaf finiteness and bounds, uint32/int32
narrowing) so malformed input cannot reach unchecked output indexing; property-based additivity
tests (random ensembles: stumps, depth-31 paths, repeated features, ~1e-4 covers, NaNs); frozen
fixtures (`tests/fixtures/`) replayable without xgboost; golden suite green on **both xgboost
2.0.3 and 3.1.2**, including `num_parallel_tree` and DART cases; Apache-2.0 LICENSE + NOTICE.

**Phase 0.6 — Second-round hardening (added after validation_v2). DONE in this revision.**
Portable gaps the second external validation exposed, all fixed and verified in-sandbox:
objective-link table determined *empirically* (identity / logit / log per objective; probes
showed >1.0 bias errors for Poisson/Gamma/Tweedie under the old logit-only logic) with
objectives outside the tested allowlist **rejected**, plus nine objective-link golden cases and
a rejection test; raw-path validation moved **before** deduplication (merging laundered
duplicate roots, conflicting group/leaf metadata, and individually invalid fractions such as
2×0.25 or −0.5×−0.5 — all now unit-tested); the property generator's feature pool widened so
deep trials produce genuinely long deduplicated groups, plus a deterministic 31-distinct-feature
comb tree asserting a full 32-element cooperative group executes (the old pool capped groups at
5); golden runs made non-mutating (`--update-fixtures` / `--write-results` are explicit); the
correctness gate tightened to the stated 1e-3; order-spread redefined as max *pairwise* spread
across natural + seeded orders (environment-dependent by nature — magnitudes, not exact values,
are the signal); fixtures extended to five xgboost cases plus the synthetic `deep31` comb
fixture (the Phase-1 Metal differential target); Metal host hardened (non-copyable/movable
owners, scoped autorelease pools, zero-work paths, checked uint32 narrowing and size products,
required intercepts, internal serialization, RAII buffer guards); CMake shader step made
optional behind toolchain detection so the portable quickstart works without full Xcode; ASAN/
UBSAN option added (validation runs were clean).

**Phase 1 — End-to-end Metal correctness (~1-2 weeks). SUBSTANTIALLY MET (validation_v3).**
The compiled-model host logic has now run all six frozen fixtures on an M4 Max (§9 table) at
three `rows_per_simdgroup` settings, covering multi-bin, multi-bank, partial-SIMD bins, the
`deep31` genuine lane-31 case, empty-model/zero-row paths and repeated calls — max error 6.5e-6.
Since then the host gained the v3 hardening round (exception-safe constructors via ownership
guards, checked `num_cols+1` and byte products, finite-intercept requirement, mutex-protected
tuning, 64-bit shader work math + 32-bit-grid dispatch guard, `maxTotalThreadsPerThreadgroup`
check) and the validation harness is checked in (`src/main_metal.cpp`, `fixture_metal` CTest).
Remaining to close Phase 1: one on-device run of the checked-in runner (`ctest -R fixture`, or
`python tests/test_fixture.py build/reference_cli --metal-cli build/metal_cli`) to confirm the
repository-reproducible form, ideally in CI. Exit: that run green; then Phase 2.

**Phase 2 — Accumulation + tuning (~2 weeks).**
Implement and compare the three accumulation strategies (§4a): plain `atomic_float`, SIMD-group
pre-aggregation by (group, feature), two-stage deterministic reduction — on accuracy vs fp64
reference, run-to-run repeatability (≥100 runs), a hot-feature adversarial model, and throughput.
Sweep `rows_per_simdgroup` / threadgroup size via function constants; Xcode GPU capture + Metal
System Trace for occupancy, atomic throughput, shuffle density (~2 cycles/shuffle sustained is
the expected ceiling); shared- vs private-storage ablation for the model buffers. Exit: accepted
default accumulation mode with published evidence; kernel within sight of the §7 gate.

**Phase 3 — Persistent-model API + packaging (~2 weeks).**
Harden the compiled-model host (persistent buffers, batching over large row counts, zero-copy
input path); nanobind bindings; `MetalTreeExplainer` with the `shap` package's Explainer
interface; model parsers for LightGBM and sklearn (reuse `shap`'s `TreeEnsemble` normalization;
covers exist for all of them); wheels (cibuildwheel, macOS-14 arm64 runner); CI running the
golden + fixture suites on an Apple-Silicon GitHub runner.

**Phase 4 — Benchmarks, variants + upstreaming (~2-3 weeks).**
Reproducible CPU/Metal benchmark suite (§7) with phase-separated timings and energy; then
interaction values, Shapley-Taylor, interventional kernels (the ballot/popcount tricks port 1:1 —
`simd_ballot`+`popcount` are native; W table from host); propose `MetalTree` explainer upstream
to `shap`; write-up comparing M-series vs the published V100 results normalized per watt — the
energy story (powermetrics vs nvidia-smi) is likely the headline.

**Stretch.** fp16 path arithmetic for shallow models (Apple GPUs double fp16 rate); WebGPU port
(subgroups are now standard — would cover all vendors from one codebase); categorical split
conditions; direct XGBoost plugin so `device="metal"` works natively.

## 7. Benchmark plan

Mirror upstream `benchmark/benchmark.py` (same datasets: adult, covtype, cal_housing,
fashion_mnist; same 12 model configs: 10-1,000 rounds × depth 3/8/16; same 10K explain rows; 5
repetitions) so results are directly comparable to the published V100 table — with two honest
deviations: adult/fashion_mnist categoricals are ordinal-encoded to numeric (the extractor
rejects categorical splits for now; CPU and Metal see identical encoded data, so the comparison
is fair), and timing is phase-separated — extraction+preprocess+buffer setup reported once as
compiled-model cost, per-call explain time measured at steady state; only the latter enters the
speedup. Baselines on the *same* Mac: `xgboost` CPU `pred_contribs` (all cores),
`shap.TreeExplainer`, single-thread as reference point. Metrics: wall time, rows/s, speedup,
max|Δ| vs fp64 CPU, peak memory, and Joules (`sudo powermetrics --samplers gpu_power,cpu_power`).
Report M4 Pro/Max (and whatever other M-series are at hand) — plus the V100 numbers from the
README as context.

Honest expectations: the V100 achieved 13-19× against 40 Xeon cores. An M4 Max GPU is in the same
FP32-TFLOPS class as V100 but its CPU competitor (12 performance cores) is far stronger per-core
than 2015 Xeons, and memory bandwidth (546 GB/s vs 900 GB/s) is lower — though this workload is
compute/shuffle-bound with a tiny working set, not bandwidth-bound. A defensible target is
**5-10× vs on-device CPU for med/large models**, with small models remaining CPU-territory
(same as CUDA). Perf-per-watt should be dramatic (~40-60 W package vs 300 W+ for the DGX-1 setup).

## 8. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| fp32 accumulation error or run-to-run variance unacceptable on huge ensembles | Medium | CPU-proxy evidence gathered (§9: error and order-spread both ~4e-5 on 500-tree stress); three-way on-device comparison in Phase 2 with hot-feature adversarial model and ≥100-run repeatability; deterministic two-stage mode as the fallback |
| Atomic contention on hot features (root splits shared by most paths) | Medium | Hardware float atomics; contention diluted across rows; SIMD-group pre-aggregation by (group, feature) designed in §4a; per-threadgroup privatization as further fallback |
| XGBoost serialization churn (base_score became a vector in 3.1; future schema drift) | Medium | Extractor reads the raw JSON model with version-robust parsing; suite verified on 2.0.3 **and** 3.1.2; frozen fixtures (`tests/fixtures/`) catch drift independently of an installed xgboost |
| `simd_shuffle` undefined-lane semantics differ from CUDA (`shfl_up` low lanes) | Low (retired on fixtures) | Clamped-index shuffles; kernel executed correctly on M4 Max fixtures; model-scale differential tests in Phase 1 |
| MSL struct layout/alignment mismatch with C++ | Low | Explicit 32-B `GpuPathElement` layout, `static_assert` on the C++ side, fixture-verified on device |
| Malformed/hostile path input reaching out-of-bounds writes | Low (closed) | Phase 0.5 validation: one-root, range, finiteness, narrowing checks before packing |
| Apple-Silicon CI availability for wheels/tests | Low | GitHub macOS-14 arm64 runners exist; fixture tests need no xgboost |
| Depth > 32 models (rare; XGBoost defaults ≤ 10) | Low | Same upstream constraint; clear error at validation; fall back to CPU |
| Divergence when a bin packs paths of very different lengths | Low | Same behavior as CUDA (idle lanes during longer groups' loops); BFD already minimizes; measurable in capture |

## 9. Evidence gathered during proposal preparation

Two rounds of evidence exist: the Phase 0/0.5 CPU pipeline executed in a Linux sandbox (no Metal
required by design), and an **independent external validation on an M4 Max** that compiled and
ran the draft kernel.

**External M4 Max validation (independent reviewer, three rounds).** Rounds 1-2:
`shaders/treeshap.metal` compiled under Metal 3; execution width 32, 256-thread threadgroup
honored; exact results on simple and multi-row-bank fixtures, 5.96e-7 on a packed multi-path
fixture; portable suites reproduced independently (ASAN+UBSAN clean, fixtures byte-identical,
expected contributions within 6e-8 across environments). **Round 3 ran the compiled-model host
logic itself** (CompiledModel + Explain; shader runtime-compiled via newLibraryWithSource) over
all six frozen fixtures, at `rows_per_simdgroup` ∈ {1, 7, 1024}, plus empty-model, zero-row,
intercept, repeated-call and invalid-tuning behavior:

| fixture | rows | raw elements | bins | max Metal error |
|---|---:|---:|---:|---:|
| binary-depth6 | 300 | 17,053 | 552 | 4.29e-6 |
| dart | 200 | 2,400 | 72 | 1.55e-6 |
| deep31 (32-lane comb) | 8 | 559 | 18 | 6.50e-6 |
| multiclass-3 | 200 | 6,781 | 201 | 1.01e-6 |
| parallel-trees | 200 | 4,668 | 132 | 5.96e-7 |
| regression-missing | 300 | 800 | 24 | 1.19e-6 |

That is most of Phase 1's technical exit criterion. The remaining gap was repository
reproducibility — the harness used was the reviewer's own — which `src/main_metal.cpp`
(same CSV contract as reference_cli, runtime-source fallback) + `tests/test_fixture.py
--metal-cli` + the `fixture_metal` CTest entry now close.

**CPU pipeline results**, from `tests/RESULTS.md` — golden suite green on **xgboost 2.0.3 and
3.1.2** (both built from source in-sandbox); table from the 3.1.2 run:

| model | booster | trees | depth | paths | max\|phi − xgb\| | sum-to-margin | fp32 abs | fp32 rel (elem., floored) | order spread |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| regression + 15% missing | gbtree | 25 | 3 | 200 | 4.2e-7 | 7.5e-7 | 9.1e-7 | 1.0e-5 | — |
| binary logistic | gbtree | 50 | 6 | 2,511 | 7.1e-7 | 1.5e-6 | 3.9e-6 | 2.6e-5 | — |
| multiclass (3 groups) | gbtree | 90 | 4 | 1,364 | 2.7e-7 | 4.8e-7 | 1.1e-6 | 2.2e-5 | — |
| multiclass, num_parallel_tree=2 | gbtree | 60 | 4 | 1,364 | 2.1e-7 | 2.9e-7 | 5.7e-7 | 1.2e-5 | — |
| DART (rate_drop 0.2) | dart | 30 | 4 | 380 | 4.1e-7 | 5.8e-7 | 1.0e-6 | 1.4e-5 | — |
| stress regression | gbtree | 500 | 8 | 65,374 | 8.0e-6 | 8.3e-6 | 3.8e-5 | 6.5e-5 | 4.2e-5 |

1. **Golden correctness, cross-version.** Extractor (raw-JSON based: tree_info groups, vector
   intercepts, DART weight_drop) + preprocess + scalar reference reproduce
   `xgboost.predict(pred_contribs=True)` elementwise to ~1e-6, intercept included — no post-hoc
   patching — on both a pre-3.1 and a post-3.1 xgboost. The `num_parallel_tree` and DART rows are
   regression tests for the two extractor bugs the external review demonstrated.
2. **fp32 accumulation study (CPU proxy).** Absolute error ≤3.8e-5; elementwise-relative
   (floored at 1e-3·max|phi|, per review) ≤6.5e-5; max pairwise spread across shuffled
   accumulation orders ~4-6e-5 on the stress model (environment-dependent — the stdlib shuffle
   differs across platforms; magnitudes, not exact values, are the signal). Informative, not
   decisive: true concurrent atomics are measured on-device in Phase 2. Terminology note: the
   "reference" throughout is *float recurrences with selectable fp64/fp32 accumulation* — kernel
   arithmetic at kernel precision, by design — not a full-double computation; the full-double
   oracle exists separately in the comb test below.
3. **Property-based additivity + depth-31 conditioning.** Random constraint-aware ensembles —
   stumps, spine-guaranteed deep trees (cooperative groups ≥26 post-dedup), repeated features,
   ~1e-4 cover fractions, NaN routing — satisfy sum(phis)+bias = margin-by-traversal in both
   accumulation modes. The deterministic 32-element comb additionally compares against an
   independent full-double implementation: double-oracle additivity 1.9e-13 (logic exact); float
   recurrences deviate from the double oracle by ≤~9e-6 **per attribution** (elementwise), and
   those signed ~e-6 deviations accumulate to a **row-sum residual** of ~1.3e-4 — an additivity
   residual, not a per-attribution error bound. The CUDA kernel uses the same float recurrence,
   so similar deep-path sensitivity is a reasonable inference, though unmeasured on CUDA. All
   figures sit well below the 1e-3 gate (`tests/test_property_additivity.cpp`).
4. **Bin-packing sanity.** BFD/FFD/NF produce valid packings (no bin > 32, all paths packed); on
   5,000 random path lengths BFD achieved 2,611 bins vs a 2,578 theoretical lower bound (98.7%
   lane efficiency), matching the paper's near-optimality claim (`tests/test_preprocess.cpp`).
5. **Reproducibility.** `tests/fixtures/` freezes a model + data + expected contributions;
   `tests/test_fixture.py` replays them with **no xgboost installed** (the extractor parses the
   model file directly), pinning the pipeline against future interface drift.

The Metal kernel (`shaders/treeshap.metal`) and metal-cpp host (`src/metal_host.hpp`) are drafted
against these verified semantics but remain **uncompiled/untested** until Phase 1 hardware time —
they are starting points, not shipped code.

## 10. References

- GPUTreeShap paper: https://arxiv.org/abs/2010.13972 (PeerJ CS version:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9044362/)
- Upstream source: the `gputreeshap` repo (this analysis: `GPUTreeShap/gpu_treeshap.h`)
- Lundberg et al., "Consistent Individualized Feature Attribution for Tree Ensembles"
  (arXiv:1802.03888) — the TreeSHAP algorithm and extend/unwind recurrences
- Metal Shading Language Specification: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- Metal Feature Set Tables: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
- Float atomics on Metal 3 (via MoltenVK maintainers): https://github.com/KhronosGroup/MoltenVK/discussions/1616
- Apple GPU microarchitecture measurements (SIMD width 32, shuffle throughput, no fp64):
  https://github.com/philipturner/metal-benchmarks ; fp64 emulation:
  https://github.com/philipturner/metal-float64
- MLX custom Metal kernels: https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- metal-cpp: https://developer.apple.com/metal/cpp/
- `makeBuffer(bytes:length:options:)` copies its input (vs `bytesNoCopy`):
  https://developer.apple.com/documentation/metal/mtldevice/1433375-makebuffer
- XGBoost 3.x parameters incl. vector-valued base_score/intercept:
  https://xgboost.readthedocs.io/en/release_3.2.0/parameter.html
- XGBoost GPU SHAP integration: https://xgboost.readthedocs.io/en/stable/gpu/index.html ;
  Metal declined upstream: https://github.com/dmlc/xgboost/issues/2440
- M4 Max GPU specs (bandwidth/TFLOPS class): https://www.notebookcheck.net/Apple-M4-Max-40-core-GPU-Benchmarks-and-Specs.920457.0.html
