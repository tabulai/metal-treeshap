# How GPUTreeShap Accelerates TreeSHAP with CUDA

*An assessment of NVIDIA's GPUTreeShap library (`gputreeshap`, Apache 2.0), based on a read of
`GPUTreeShap/gpu_treeshap.h` (~1,574 lines, header-only), the example, tests, and benchmark code,
and the companion paper: Mitchell, Frank & Holmes, "GPUTreeShap: Massively Parallel Exact
Calculation of SHAP Scores for Tree Ensembles" (arXiv:2010.13972, PeerJ CS 2022).*

---

## 1. The problem being accelerated

TreeSHAP (Lundberg et al., 2018) computes exact Shapley values for tree-ensemble predictions. For
a single instance and a single tree, the classic recursive algorithm costs **O(L·D²)** where L is
the number of leaves and D the maximum depth, and it must be repeated for every row to be
explained. Explaining 10K rows against a 1,000-tree, depth-16 XGBoost model is minutes-to-hours of
CPU time (the repo's own benchmark measures 930 s on 40 Xeon cores for `covtype-large`). The
computation is embarrassingly parallel across rows and trees, but the inner recursion — the
"extend/unwind" bookkeeping of permutation weights along a root-to-leaf path — is sequential and
branchy, which is exactly what GPUs hate.

GPUTreeShap's contribution is a **reformulation of TreeSHAP as a massively parallel, non-recursive
computation over decomposed tree paths**, with a warp-level cooperative implementation of the
extend/unwind recurrences. It is the backend behind `xgboost.predict(pred_contribs=True)` on GPU
(XGBoost ≥ 1.3), `shap.explainers.GPUTree`, and cuML's explainability module.

## 2. Input representation: trees become paths, not nodes

The library never sees a tree. The host application (e.g. XGBoost) decomposes every tree into its
**unique root-to-leaf paths** and hands over a flat array of `PathElement<SplitConditionT>`
(gpu_treeshap.h:95-130). Each element carries:

| field | meaning |
|---|---|
| `path_idx` | globally unique id of the path (leaf) it belongs to |
| `feature_idx` | feature split on at this step; `-1` marks the path's root element |
| `group` | output class (multiclass models emit one path set per class) |
| `split_condition` | templated; for XGBoost an interval `[lower, upper)` + `is_missing_branch` flag |
| `zero_fraction` | fraction of *training* data that follows this branch (cover ratio child/parent), stored as `double` |
| `v` | leaf value at the end of the path (`float`) |

Two design choices here do a lot of work:

**Split conditions are intervals, not node references.** A path that passes through several splits
on the same feature is *deduplicated* by intersecting intervals (`Merge`, gpu_treeshap.h:77-82)
and multiplying the branch probabilities. After dedup, each path contains each feature at most
once, so "path length" ≤ min(depth, num_features) ≤ 32. Evaluating whether a row follows a path
element is a single interval test `lower ≤ x < upper` with a NaN check for missing values
(`EvaluateSplit`, gpu_treeshap.h:68-74) — no tree traversal, no pointer chasing.

**The representation is duplication-friendly.** Path decomposition inflates the model (a leaf at
depth D becomes D+1 elements), but the total is still only Σ(leaf depths) elements ≈ a few tens of
MB for even huge ensembles. The README's claim that "memory usage is proportional to the model,
not the dataset" follows directly. This trades memory for a *completely regular, flat* data
structure — the canonical GPU-friendly transformation.

The mathematical justification (paper, Theorem/derivation around "path-dependent feature
perturbation") is that TreeSHAP's per-tree Shapley value decomposes into an independent sum over
root-to-leaf paths: each path contributes `phi_i += UnwoundPathSum_i · (one_fraction_i −
zero_fraction_i) · v` for every feature i on the path, where the unwound sum is computed from the
path's permutation-weight vector. Path contributions are order-independent and can be computed in
any order, by any thread, and summed — which is precisely what makes the rest of the design
possible.

## 3. The parallel decomposition: (rows × paths) grid, warp-cooperative paths

The work is a 2-D grid: **every (row, unique-path) pair is an independently computable task** —
independent in computation, but not in output: many (row, path) tasks converge on the same
`phis[row, feature]` cell, which is exactly why the accumulation strategy (§3.4) is a first-class
design concern rather than an implementation detail. GPUTreeShap maps this hierarchy onto CUDA
hardware as follows (`ConfigureThread`, gpu_treeshap.h:424-453 and
`ComputeShap`, gpu_treeshap.h:493-513):

- **One warp (32 threads) owns one *bin* of paths and one *bank* of rows.** Warps needed =
  `bins_per_row × ceil(num_rows / kRowsPerWarp)` with `kRowsPerWarp = 1024` for first-order values.
  Blocks are 256 threads = 8 warps (`GPUTREESHAP_MAX_THREADS_PER_BLOCK`, `__launch_bounds__`).
- **One thread owns one path element.** Thread ranks within the warp index into the bin's slice of
  the sorted element array; each thread loads its element once into shared memory (dynamic,
  alignment-cast `char` storage to keep nvcc from spilling the structs to local memory,
  gpu_treeshap.h:465-469) and keeps its running state in registers.
- **The warp is subdivided into one cooperative group per path** via
  `active_labeled_partition(mask, e.path_idx)` (gpu_treeshap.h:259-282): threads with equal
  `path_idx` form a `ContiguousGroup`. On Volta+ this is a single `__match_any_sync`; pre-Volta a
  ballot loop emulates it. Because elements were pre-sorted, groups are guaranteed contiguous
  lanes, which makes all subsequent shuffles cheap rank arithmetic.

So a warp is a little SIMD container holding several *whole paths* side by side — e.g. a warp
might hold paths of length 11, 9, 7 and 5 packed into its 32 lanes. Each group then runs the
Shapley recurrence *cooperatively across its lanes* while iterating rows serially.

### 3.1 Bin packing: making 32 lanes earn their keep

Paths have variable length (1 + number of unique features on the path). Naively assigning one
path per warp would idle most lanes (average depth in the benchmarks is 3-14). GPUTreeShap treats
lane allocation as a **bin-packing problem**: pack path lengths into bins of capacity 32 so that
few lanes are wasted.

- `BFDBinPacking` (gpu_treeshap.h:976-1015) — Best Fit Decreasing with a `std::set` acting as a
  balanced tree over remaining capacities, O(n log n), run **on the host**. This is the default
  used by `PreprocessPaths`. BFD is provably within 11/9·OPT + 6/9 bins, and in the paper's
  measurements is essentially optimal for real models.
- `FFDBinPacking` (First Fit Decreasing, O(n²)) and `NFBinPacking` (Next Fit, O(n)) exist as
  alternatives/tests.

After packing, `SortPaths` orders elements by (bin, path_idx, feature_idx) and `GetBinSegments`
computes each bin's [start, end) offsets with an atomic histogram + `thrust::exclusive_scan`. The
kernel then reads `bin_segments[bin_idx]`/`[bin_idx+1]` to find its lanes' elements. Lanes with
`thread_rank ≥ bin size` simply deactivate for the whole kernel (`thread_active = false`).

This is the library's occupancy story: instead of load-balancing dynamically, it *statically*
packs irregular work into fixed 32-lane vessels once per model, amortized over all future rows.

### 3.2 The warp-cooperative Shapley recurrence

`ContiguousGroup` (gpu_treeshap.h:222-253) re-implements a mini cooperative-groups API on top of
raw warp intrinsics, re-based to the group's first lane (found with `__ffs(mask)`), with
`thread_rank()` from `__popc(mask & lanemask_lt())` (the lanemask via inline PTX,
gpu_treeshap.h:214-218). It provides `shfl`, `shfl_up`, `ballot`, and a shuffle-based scan
`reduce`.

**Extend.** `GroupPath::Extend()` (gpu_treeshap.h:304-329) is the heart of the port. In Lundberg's
CPU algorithm, EXTEND grows an array `m[0..d]` of permutation weights one path element at a time,
in a length-d loop nested inside the depth-d path walk — O(D²) serial work per (row, path). Here
the array is **distributed one element per lane** (`pweight_` in a register). Adding element k is
a *single cooperative step*:

```
// lane r holds m[r]; element k's (zero_fraction, one_fraction) broadcast from lane k
pweight_[r] = pweight_[r] · z_k · (k − r)/(k+1)     // "feature absent" term, max(k-r,0) branch-free
            + o_k · pweight_[r−1] · r/(k+1)          // "feature present" term via shfl_up(1)
```

Two micro-optimizations are notable: the element's `(zero_fraction, one_fraction)` pair is packed
into one 64-bit word so the broadcast costs **one** shuffle instead of two (gpu_treeshap.h:312-315),
and the update is written with explicit `__fmul_rn`/`__fmaf_rn`/`__fdividef` intrinsics and a
`max(k−r, 0)` in place of a branch. The full extend of a depth-D path is D cooperative steps —
**O(D) span instead of O(D²) serial work**, with the O(D²) total work spread across D lanes.

**Unwind-and-sum.** Computing feature i's contribution classically requires *unwinding* element i
out of the weight array and summing the result — again O(D) per feature, O(D²) per path serially.
`UnwoundPathSum()` (gpu_treeshap.h:332-353) has **each lane unwind its own feature
simultaneously**: a length-D loop of two shuffles and a handful of FMAs per lane, using an
algebraic rearrangement that never materializes the unwound array and is branch-free given that
`one_fraction ∈ {0,1}` for decision splits (the `if (precomputed > 0)` guard handles the z=0/o=0
degeneracies). All lanes reuse the *same* broadcasts `shfl(pweight_, i)`, so the warp-wide cost of
unwinding all features is the same O(D) loop, not D separate loops.

Per (row, path), the group therefore does: 1 interval test per lane (`one_fraction`), D extend
steps, D unwind steps, then `phi = sum · (one_fraction − zero_fraction) · v` per lane —
`ComputePhi`, gpu_treeshap.h:401-418.

### 3.3 Row banking: amortizing setup

Each warp loops serially over up to 1,024 rows (`kRowsPerWarp`), reusing its loaded path elements,
group partition, and `zero_fraction`s; only the per-row `one_fraction` (one dataset read + interval
test per lane) and the recurrences are recomputed (gpu_treeshap.h:482-490). This amortizes the
setup (element loads, labeled partition) and — more importantly — keeps the number of resident
warps proportional to the *model*, not rows × model, while still exposing `bins × row-banks`-way
parallelism, easily tens of thousands of warps for real workloads.

### 3.4 Accumulation and precision policy

Results land in a global `phis[row, group, feature]` array via **atomic adds in double precision**
(`atomicAddDouble`, native `atomicAdd(double*)` on CC ≥ 6.0, CAS loop otherwise;
gpu_treeshap.h:190-212). The policy is deliberately mixed-precision:

- Per-path arithmetic (extend/unwind) runs in **float32 registers**. That is fast and normally
  accurate for typical shallow trees, but it is not universally well-conditioned: the port's
  depth-31 comb test measures a ~1.1e-4 row residual originating in these float recurrences.
- Cross-path **accumulation runs in float64**. A SHAP value can sum many signed path
  contributions, so wider accumulation reduces rounding and order sensitivity. Historically,
  commit `2b0ba96` ("Determinism") changed the temporary/output array and atomics from float to
  double while adding a determinism test. The source does not quantify the preceding fp32 error or
  say that double atomics were free, so stronger causal claims would be speculation. The code
  today uses `double* phis` throughout the kernels and converts when copying to the caller's type.
- The **bias term** (expected value; column F+1 of the output) is computed separately on the
  device in full double precision by two `reduce_by_key` passes over the *raw* paths — product of
  `zero_fraction` per path, then Σ (leaf-probability · v) per class — explicitly "to avoid
  numerical stability issues" (`ComputeBias`, gpu_treeshap.h:1172-1231).

Atomic contention is diluted by design: at any instant the writers to a given `(row, feature)`
cell are only the groups whose paths contain that feature and whose bank covers that row, and
Maxwell+ hardware handles same-address double atomics in L2 reasonably. The tests
(`test_gpu_treeshap.cu`) validate against a CPU reference implementation and against the
sum-to-margin invariant (Σ phis + bias = margin prediction).

### 3.5 GPU-resident preprocessing pipeline

Everything between "raw paths in" and "kernel launch" runs on the device with thrust/cub
(`PreprocessPaths`, gpu_treeshap.h:1131-1149):

1. `DeduplicatePaths` — `thrust::sort` by (path_idx, feature_idx), then
   `cub::DeviceReduce::ReduceByKey` merges same-feature elements within a path (interval
   intersection, zero_fraction product) (gpu_treeshap.h:890-941).
2. `GetPathLengths` — atomic histogram of elements per path (gpu_treeshap.h:1074-1089).
3. `ValidatePaths` — rejects depth > 32 ("Tree depth must be < 32") and inconsistent leaf values
   via `thrust::any_of` (gpu_treeshap.h:1105-1129).
4. `BFDBinPacking` on host (the only host step; the lengths array is tiny).
5. `SortPaths` by (bin, path, feature) + `GetBinSegments` (atomic count + exclusive scan).

The preprocessing is O(model) once per explain call, negligible next to the O(rows × model) kernel.

## 4. The three variant kernels

The same skeleton (ConfigureThread → labeled partition → row loop) powers four kernels:

**First-order SHAP** (`ShapKernel`, gpu_treeshap.h:458-491) — as described above. 1,024
rows/warp.

**SHAP interaction values** (`ShapInteractionsKernel`, gpu_treeshap.h:563-617) — for each path and
each "condition feature" j on it, the element for j is swapped to the logical end of the path
(`SwapConditionedElement`), the path is re-extended *without* j, and each remaining feature i
accumulates `0.5 · (sum_cond_on − sum_cond_off)`-style terms into `phis[row, i, j]` and subtracts
from the diagonal (Lundberg's conditional-difference formulation). Work rises by a factor of
group-size (O(D) recomputations of an O(D) recurrence per row) → 100 rows/warp. Output is the
(F+1)×(F+1) matrix per row/class.

**Shapley-Taylor interactions** (`ShapTaylorInteractionsKernel`, gpu_treeshap.h:641-708) — same
conditioning loop but the unwind uses `TaylorGroupPath` (gpu_treeshap.h:358-399), a different
permutation weighting ("as if the total number of features was one larger", result × 2), plus a
shuffle-scan `reduce` (multiplicative) for the diagonal terms, implementing the
Sundararajan-Dhamdhere-Agarwal index.

**Interventional SHAP** (`ShapInterventionalKernel`, gpu_treeshap.h:747-823) — a genuinely
different algorithm (Chen et al.'s "true to the data" formulation): for every (foreground row,
background row, path) triple, lanes ballot which splits each row satisfies, and the Shapley
combinatorics reduce to **population counts on ballot masks**: `s = popc(x & ~r)`,
`n = popc(x ^ r)`, weight `W(s,n) = s!(n−s−1)!/n!` looked up from a 33×33 table precomputed in
shared memory via `lgamma` (log-space to dodge overflow, gpu_treeshap.h:742-745, 754-766). The
warp bit-trickery turns an exponential-looking sum into a few `popc`s per (x, r, path) — an
elegant, hardware-native reformulation. Cost scales with background size (O(|X|·|R|·paths)).

## 5. Why it's fast: a summary of the acceleration levers

1. **Algorithmic reshaping.** Recursive tree walk → flat independent path tasks; O(D²) sequential
   inner loops → O(D)-**span** cooperative recurrences on ≤32 lanes. To be precise: total
   arithmetic across the lanes remains O(D²) — the win is critical-path length and hardware
   occupancy, not operation count. No recursion, no stacks, no dynamic allocation on device.
2. **Work-to-hardware fit.** Max supported depth (32) == warp width. Paths are packed into warps
   by near-optimal bin packing, so lanes are ~fully utilized despite irregular path lengths;
   within a group there is *zero divergence* by construction (all lanes run the same D-step loops).
3. **Memory behavior.** Path elements staged once into shared memory per warp and reused across
   1,024 rows; all recurrence state lives in registers; dataset reads are the only per-row global
   traffic (one float per lane per row, coalesced-ish across the warp's features); output writes
   are atomics diluted across a huge phis array. Working set ∝ model, not data.
4. **Warp-primitive engineering.** match_any/ballot partitions, rank-rebased shuffles, packed
   64-bit broadcasts, popc/ffs bit math, branch-free FMA-intrinsic arithmetic, `__launch_bounds__`
   pinning, shared-memory struct staging to defeat local-memory spills.
5. **Mixed precision.** fp32 cooperative recurrences for throughput, fp64 cross-path atomics and
   a separate double bias reduction to reduce accumulation and expected-value error.
6. **Device-side preprocessing** with thrust/cub so the whole explain call (minus BFD packing)
   stays on-GPU, launched once per model.

Measured effect (repo README, DGX-1 V100 vs 2× 20-core Xeon E5-2698, 10K rows): **13-19× speedup**
on medium/large models (e.g. `adult-large` 88.1 s → 4.67 s; `covtype-large` 930 s → 50.9 s), and
~1-2× on tiny models where kernel launch and preprocessing overheads dominate — i.e. the
acceleration comes from throughput, and it needs enough (rows × paths) work to shine. The paper
additionally reports speedups of up to **340× for SHAP *interaction* values** against the same
40-core CPU baseline (not single-core; interactions multiply the per-row work by O(D), which the
GPU absorbs far better than the CPU).

## 6. Constraints and assumptions worth carrying to any port

- **Depth ≤ 32** (after per-path feature dedup) — hard requirement, enforced at preprocess time.
- Leaf value constant per path; paths must include a root element (`feature_idx = −1`,
  `zero_fraction = 1`).
- `one_fraction ∈ {0,1}` is assumed by the branch-free unwind (true for hard decision splits).
- Dataset adapter must be a trivially-copyable struct exposing `NumRows/NumCols/GetElement`
  (missing = NaN); XGBoost passes dense or sparse wrappers.
- Double-precision atomics (CC ≥ 6.0 for native; CAS fallback provided). This is the one primitive
  with **no Apple-GPU equivalent** — see the companion proposal document.
- Everything else is warp-32 arithmetic, shuffles, ballots, popcounts, shared memory, and
  sort/scan/reduce_by_key preprocessing — all of which have direct Metal analogs.
