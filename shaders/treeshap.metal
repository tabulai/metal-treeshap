// SPDX-License-Identifier: Apache-2.0
// Part of metal-treeshap (see LICENSE and NOTICE); ported from RAPIDS GPUTreeShap.
//
// metal-treeshap: first-order TreeSHAP kernel for Apple GPUs.
//
// Port of gpu_treeshap.h's ShapKernel (CUDA) to Metal Shading Language. See
// docs/01-cuda-acceleration-assessment.md for the algorithm and
// docs/02-apple-gpu-project-proposal.md §5 for the CUDA->Metal mapping decisions.
//
// STATUS: locally validated on an M4 Max through the checked-in metal-cpp host and CTests.
// All seven frozen fixtures pass across atomic, SIMD-group and deterministic accumulation,
// shared/private model storage, 32/64/128/256-thread threadgroups, and broad row-bank sweeps;
// worst observed Metal error is 6.505e-6. Execution width is 32.
// Requires Apple Silicon (Apple7+/M1+, Metal 3 for atomic_float).
//
// Design notes (vs the CUDA original):
//  * warp -> simdgroup: both 32-wide on Apple GPUs; max path length 32 unchanged.
//  * active_labeled_partition/__match_any_sync -> sorted-contiguity boundary trick:
//    elements are pre-sorted by (bin, path, feature), so path groups are contiguous lanes;
//    boundaries come from one ballot + clz/ctz bit math.
//  * __shfl_up_sync -> clamped-index simd_shuffle (MSL simd_shuffle_up is undefined for
//    low lanes, whereas CUDA returns the lane's own value; the algorithm multiplies that
//    term by rank==0 so any *finite* value works, but NaN/Inf garbage would poison it —
//    hence the clamp).
//  * atomicAdd(double) -> atomic_float fetch_add (fp32), either directly per path element
//    or after SIMD-group aggregation by output key. Deterministic mode writes canonical
//    per-element partials and reduces them in fixed path-id order. Bias is host-prefilled.
//  * PathElement is a 32-byte explicit layout shared with include/metal_treeshap/paths.h
//    (GpuPathElement); zero_fraction is fp32 here (host keeps fp64 until packing).
//  * Dataset is dense row-major fp32; NaN encodes missing.

#include <metal_stdlib>
using namespace metal;

struct PathElement {          // must match GpuPathElement in paths.h (32 bytes)
  uint  path_idx;
  int   feature_idx;          // -1 == root
  int   group;
  float zero_fraction;
  float v;
  float lower;
  float upper;
  uint  is_missing_branch;
};

struct Params {
  uint num_rows;              // rows in the current tile
  uint row_offset;            // first row of the tile in X and phis
  uint num_cols;
  uint num_groups;
  uint num_bins;              // == bins_per_row in the CUDA code
  uint rows_per_simdgroup;    // CUDA kRowsPerWarp analogue (host default tuned to 256)
};

struct OutputFillParams {     // must match OutputFillParams in metal_host.hpp
  uint num_rows;
  uint num_groups;
  uint num_cols;
};

struct DeterministicParams {
  uint num_rows;              // rows in the current tile
  uint row_offset;            // first row of the tile in X and phis
  uint num_cols;
  uint num_groups;
  uint num_bins;
  uint rows_per_simdgroup;
  uint num_partials;
  uint num_active_cells;
  uint num_chunks;            // stage-A work items per row
};

struct ReductionCell {        // matches DeterministicReductionCell in deterministic.h
  uint group;
  uint feature;
  uint begin;
  uint end;
};

struct ReductionChunk {       // matches DeterministicReductionChunk in deterministic.h
  uint begin;                 // partial-slot range [begin, end), one cell, <= 256 slots
  uint end;
};

constant constexpr uint kSimdWidth = 32;

inline bool EvaluateSplit(thread const PathElement& e, float x) {
  // NaN routes to the missing branch. Integer bit test rather than isnan(): this library
  // compiles with fast math, whose no-NaN assumption is demonstrably active (x != x folds
  // to false under the default compile options), and isnan() survives only through builtin
  // special-casing that a future OS Metal compiler need not preserve.
  if ((as_type<uint>(x) & 0x7FFFFFFFu) > 0x7F800000u) return e.is_missing_branch != 0;
  if (x >= e.lower && x < e.upper) return true;
  // +inf satisfies no half-open interval (inf < inf is false); route it like any value
  // above every finite threshold: follow iff the interval extends to +infinity. Must match
  // XgboostSplitCondition::EvaluateSplit in paths.h. Integer bit compare so the fast-math
  // main library cannot fold the infinity test.
  return as_type<uint>(x) == 0x7F800000u && as_type<uint>(e.upper) == 0x7F800000u;
}

// Contiguous sub-simdgroup [start, start+size): all shuffles rebased to `start`.
struct ContiguousGroup {
  uint start;
  uint size;
  uint rank;  // this lane's rank within the group

  template <typename T>
  T Shfl(T val, uint src_rank) const { return simd_shuffle(val, start + src_rank); }

  // CUDA __shfl_up_sync(…, 1) semantics for our use: lanes with rank==0 read themselves
  // (the value is multiplied by rank==0 downstream, so it only needs to be finite).
  float ShflUp1(float val, uint lane) const {
    return simd_shuffle(val, (rank == 0) ? lane : (lane - 1));
  }
};

template <bool kSimdgroupAggregation>
struct OutputAccumulator;

template <>
struct OutputAccumulator<false> {
  static inline void Add(device atomic_float* phis, constant Params& p, ulong row,
                         thread const PathElement& e, bool is_root, float phi, uint lane) {
    (void)lane;
    if (is_root) return;
    const ulong out = (row * p.num_groups + ulong(e.group)) * (p.num_cols + 1) +
                      ulong(e.feature_idx);
    atomic_fetch_add_explicit(&phis[out], phi, memory_order_relaxed);
  }
};

template <>
struct OutputAccumulator<true> {
  static inline void Add(device atomic_float* phis, constant Params& p, ulong row,
                         thread const PathElement& e, bool is_root, float phi, uint lane) {
    // Coalesce all contributions in this SIMD-group that target the same output key.
    ulong remaining = ulong(simd_ballot(!is_root));
    while (remaining != 0) {
      const uint leader = ctz(remaining);
      const int key_group = simd_shuffle(e.group, leader);
      const int key_feature = simd_shuffle(e.feature_idx, leader);
      const bool matches = !is_root && e.group == key_group && e.feature_idx == key_feature;
      const ulong match_mask = ulong(simd_ballot(matches));
      const float aggregate = simd_sum(matches ? phi : 0.0f);
      if (lane == leader) {
        const ulong out = (row * p.num_groups + ulong(key_group)) * (p.num_cols + 1) +
                          ulong(key_feature);
        atomic_fetch_add_explicit(&phis[out], aggregate, memory_order_relaxed);
      }
      remaining &= ~match_mask;
    }
  }
};

template <bool kSimdgroupAggregation>
inline void ShapFirstOrderImpl(device const float* X, device const PathElement* elements,
                               device const uint* bin_segments,
                               device atomic_float* phis, constant Params& p, uint tid,
                               uint lane) {
  // ---- Work assignment (CUDA ConfigureThread) ----
  // 64-bit work-count arithmetic so this can never disagree with the host's 64-bit
  // dispatch math on extreme shapes (the host additionally rejects dispatches whose
  // thread count would exceed 32-bit grid coordinates).
  const uint simd_id = tid / kSimdWidth;  // grid is dispatched as a multiple of 256
  const ulong banks =
      (ulong(p.num_rows) + p.rows_per_simdgroup - 1) / p.rows_per_simdgroup;
  if (ulong(simd_id) >= ulong(p.num_bins) * banks) return;

  const uint bin_idx = uint(ulong(simd_id) % p.num_bins);
  const ulong bank = ulong(simd_id) / p.num_bins;
  const uint seg_start = bin_segments[bin_idx];
  const uint seg_end = bin_segments[bin_idx + 1];
  const uint n_active = seg_end - seg_start;

  // Inactive lanes exit; every simd_shuffle below reads only from lanes that remain active
  // (sources are always members of the reader's own group), which MSL permits.
  if (lane >= n_active) return;

  const PathElement e = elements[seg_start + lane];
  const bool is_root = e.feature_idx < 0;
  const float zero_fraction = e.zero_fraction;

  // ---- Contiguous labeled partition by path_idx (replaces __match_any_sync) ----
  const uint prev_path = simd_shuffle(e.path_idx, (lane == 0) ? 0u : (lane - 1));
  const bool boundary = (lane == 0) || (e.path_idx != prev_path);
  const uint bmask = uint(uint64_t(simd_ballot(boundary)));
  const uint below_inc = bmask & ((lane == 31) ? 0xFFFFFFFFu : ((1u << (lane + 1)) - 1u));
  const uint above = bmask & ~((lane == 31) ? 0xFFFFFFFFu : ((1u << (lane + 1)) - 1u));

  ContiguousGroup g;
  g.start = 31 - clz(below_inc);                    // highest boundary at or below this lane
  const uint g_end = (above != 0) ? ctz(above) : n_active;
  g.size = g_end - g.start;
  g.rank = lane - g.start;

  const uint D = g.size - 1;  // unique_depth after all extends
  const ulong row_begin = ulong(bank) * p.rows_per_simdgroup;
  const ulong row_end = min(row_begin + p.rows_per_simdgroup, ulong(p.num_rows));

  for (ulong row = row_begin; row < row_end; row++) {
    const ulong output_row = row + p.row_offset;
    // one_fraction: does this row satisfy my element's split? (root: always 1)
    const float one_fraction =
        is_root ? 1.0f
                : (EvaluateSplit(e, X[output_row * p.num_cols + uint(e.feature_idx)])
                       ? 1.0f
                       : 0.0f);

    // ---- GroupPath::Extend, d = 1 .. g.size-1 (cooperative across the group) ----
    float pweight = (g.rank == 0) ? 1.0f : 0.0f;
    for (uint d = 1; d < g.size; d++) {
      const float2 zo = g.Shfl(float2(zero_fraction, one_fraction), d);
      const float left_pweight = g.ShflUp1(pweight, lane);
      const float inv = 1.0f / float(d + 1);
      // pweight = pweight * z_d * max(d - rank, 0)/(d+1) + o_d * left * rank/(d+1)
      pweight = pweight * zo.x * float(max(int(d) - int(g.rank), 0)) * inv;
      pweight = fma(zo.y * left_pweight, float(g.rank) * inv, pweight);
    }

    // ---- UnwoundPathSum: every lane unwinds its own feature simultaneously ----
    float next_one_portion = g.Shfl(pweight, D);
    float total = 0.0f;
    const float zero_frac_div_depth = zero_fraction / float(D + 1);
    for (int j = int(D) - 1; j >= 0; j--) {
      const float ith_pweight = g.Shfl(pweight, uint(j));
      const float precomputed = float(int(D) - j) * zero_frac_div_depth;
      const float tmp = next_one_portion * float(D + 1) / float(j + 1);
      total = fma(tmp, one_fraction, total);
      next_one_portion = fma(-tmp, precomputed, ith_pweight);
      const float numerator = (1.0f - one_fraction) * ith_pweight;
      if (precomputed > 0.0f) total += numerator / precomputed;
    }

    const float phi = total * (one_fraction - zero_fraction) * e.v;
    OutputAccumulator<kSimdgroupAggregation>::Add(
        phis, p, output_row, e, is_root, phi, lane);
  }
}

// Output prefill: zeros in the feature columns, per-group bias in the trailing column.
// Replaces two full CPU passes over the output (memset+bias before dispatch, and — when
// the host can wrap the caller's buffer — the copy-back after), neither of which
// overlapped GPU work. The host falls back to the CPU fill for outputs whose element
// count exceeds this kernel's 32-bit grid coordinate.
[[max_total_threads_per_threadgroup(256)]]
kernel void fill_output_bias(
    device float* phis            [[buffer(0)]],
    device const float* bias      [[buffer(1)]],
    constant OutputFillParams& p  [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  const ulong total = ulong(p.num_rows) * p.num_groups * (p.num_cols + 1);
  if (ulong(tid) >= total) return;
  const uint stride = p.num_cols + 1;
  const uint col = tid % stride;
  phis[tid] = (col == p.num_cols) ? bias[(tid / stride) % p.num_groups] : 0.0f;
}

// Separate entrypoints keep the baseline pipeline free of the SIMD aggregation loop and
// its register footprint. This is deliberately not a runtime Params branch.
[[max_total_threads_per_threadgroup(256)]]
kernel void shap_first_order(
    device const float* X               [[buffer(0)]],
    device const PathElement* elements  [[buffer(1)]],
    device const uint* bin_segments     [[buffer(2)]],
    device atomic_float* phis           [[buffer(3)]],
    constant Params& p                  [[buffer(4)]],
    uint tid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  ShapFirstOrderImpl<false>(X, elements, bin_segments, phis, p, tid, lane);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void shap_first_order_simdgroup(
    device const float* X               [[buffer(0)]],
    device const PathElement* elements  [[buffer(1)]],
    device const uint* bin_segments     [[buffer(2)]],
    device atomic_float* phis           [[buffer(3)]],
    constant Params& p                  [[buffer(4)]],
    uint tid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  ShapFirstOrderImpl<true>(X, elements, bin_segments, phis, p, tid, lane);
}

// ---- Phase-2 deterministic accumulation ---------------------------------------
// Stage 1 computes the same float recurrence as shap_first_order, but every non-root
// element owns one canonical scratch slot. There are no atomics and every slot is fully
// written for every row in the tile. The two reduction stages below then sum each output
// cell's slots in fixed path-id order through model-defined 256-slot chunks. The host
// tiles rows so scratch stays within a configurable byte budget.
[[max_total_threads_per_threadgroup(256)]]
kernel void shap_partials(
    device const float* X                    [[buffer(0)]],
    device const PathElement* elements       [[buffer(1)]],
    device const uint* bin_segments          [[buffer(2)]],
    device const uint* partial_slot_by_elem  [[buffer(3)]],
    device float* partials                   [[buffer(4)]],
    constant DeterministicParams& p          [[buffer(5)]],
    uint tid                                 [[thread_position_in_grid]],
    uint lane                                [[thread_index_in_simdgroup]]) {
  const uint simd_id = tid / kSimdWidth;
  const ulong banks =
      (ulong(p.num_rows) + p.rows_per_simdgroup - 1) / p.rows_per_simdgroup;
  if (ulong(simd_id) >= ulong(p.num_bins) * banks) return;

  const uint bin_idx = uint(ulong(simd_id) % p.num_bins);
  const ulong bank = ulong(simd_id) / p.num_bins;
  const uint seg_start = bin_segments[bin_idx];
  const uint seg_end = bin_segments[bin_idx + 1];
  const uint n_active = seg_end - seg_start;
  if (lane >= n_active) return;

  const uint element_idx = seg_start + lane;
  const PathElement e = elements[element_idx];
  const bool is_root = e.feature_idx < 0;
  const uint slot = partial_slot_by_elem[element_idx];

  const uint prev_path = simd_shuffle(e.path_idx, (lane == 0) ? 0u : (lane - 1));
  const bool boundary = (lane == 0) || (e.path_idx != prev_path);
  const uint bmask = uint(uint64_t(simd_ballot(boundary)));
  const uint lane_mask =
      (lane == 31) ? 0xFFFFFFFFu : ((1u << (lane + 1)) - 1u);
  const uint below_inc = bmask & lane_mask;
  const uint above = bmask & ~lane_mask;

  ContiguousGroup g;
  g.start = 31 - clz(below_inc);
  const uint g_end = (above != 0) ? ctz(above) : n_active;
  g.size = g_end - g.start;
  g.rank = lane - g.start;

  const uint D = g.size - 1;
  const ulong row_begin = ulong(bank) * p.rows_per_simdgroup;
  const ulong row_end = min(row_begin + p.rows_per_simdgroup, ulong(p.num_rows));

  for (ulong row = row_begin; row < row_end; ++row) {
    const ulong input_row = row + p.row_offset;
    const float one_fraction =
        is_root ? 1.0f
                : (EvaluateSplit(e, X[input_row * p.num_cols + uint(e.feature_idx)])
                       ? 1.0f
                       : 0.0f);

    float pweight = (g.rank == 0) ? 1.0f : 0.0f;
    for (uint d = 1; d < g.size; ++d) {
      const float2 zo = g.Shfl(float2(e.zero_fraction, one_fraction), d);
      const float left_pweight = g.ShflUp1(pweight, lane);
      const float inv = 1.0f / float(d + 1);
      pweight = pweight * zo.x * float(max(int(d) - int(g.rank), 0)) * inv;
      pweight = fma(zo.y * left_pweight, float(g.rank) * inv, pweight);
    }

    float next_one_portion = g.Shfl(pweight, D);
    float total = 0.0f;
    const float zero_frac_div_depth = e.zero_fraction / float(D + 1);
    for (int j = int(D) - 1; j >= 0; --j) {
      const float ith_pweight = g.Shfl(pweight, uint(j));
      const float precomputed = float(int(D) - j) * zero_frac_div_depth;
      const float tmp = next_one_portion * float(D + 1) / float(j + 1);
      total = fma(tmp, one_fraction, total);
      next_one_portion = fma(-tmp, precomputed, ith_pweight);
      const float numerator = (1.0f - one_fraction) * ith_pweight;
      if (precomputed > 0.0f) total += numerator / precomputed;
    }

    if (!is_root) {
      const float phi = total * (one_fraction - e.zero_fraction) * e.v;
      partials[row * p.num_partials + slot] = phi;
    }
  }
}

// Two-stage fixed-shape reduction. A fully serial per-cell chain (the Phase-2.1 design)
// left only rows*cells threads in flight with chains of tens of thousands of dependent
// adds on large models; splitting every cell's slot segment into fixed 256-slot chunks
// multiplies the stage-A parallelism by ~chunks/cells while keeping the summation shape
// a pure function of the model. Kahan compensation is safe in both stages because each
// scratch/output word has exactly one writer. The host builds both entrypoints from a
// separate library with fast math disabled; otherwise reassociation legally collapses
// the compensation back to ordinary summation.
//
// Stage A: one thread exclusively owns one (row, chunk) and Kahan-sums its slot run.
[[max_total_threads_per_threadgroup(256)]]
kernel void reduce_partials_chunks(
    device const float* partials          [[buffer(0)]],
    device const ReductionChunk* chunks   [[buffer(1)]],
    device float* chunk_sums              [[buffer(2)]],
    constant DeterministicParams& p       [[buffer(3)]],
    uint tid                              [[thread_position_in_grid]]) {
  const ulong work = ulong(p.num_rows) * p.num_chunks;
  if (ulong(tid) >= work) return;
  const uint row = tid / p.num_chunks;
  const uint chunk_idx = tid % p.num_chunks;
  const ReductionChunk chunk = chunks[chunk_idx];
  float sum = 0.0f;
  float compensation = 0.0f;
  const ulong partial_base = ulong(row) * p.num_partials;
  for (uint i = chunk.begin; i < chunk.end; ++i) {
    const float y = partials[partial_base + i] - compensation;
    const float next = sum + y;
    compensation = (next - sum) - y;
    sum = next;
  }
  chunk_sums[ulong(row) * p.num_chunks + chunk_idx] = sum;
}

// Stage B: one thread exclusively owns one active (row, group, feature) cell and
// Kahan-combines its chunk sums in fixed chunk order (cells' begin/end are CHUNK
// indices here). For a cell that fits one chunk this is bit-identical to the previous
// fully-serial reducer.
[[max_total_threads_per_threadgroup(256)]]
kernel void reduce_chunks_serial(
    device const float* chunk_sums        [[buffer(0)]],
    device const ReductionCell* cells     [[buffer(1)]],
    device float* phis                    [[buffer(2)]],
    constant DeterministicParams& p       [[buffer(3)]],
    uint tid                              [[thread_position_in_grid]]) {
  const ulong work = ulong(p.num_rows) * p.num_active_cells;
  if (ulong(tid) >= work) return;
  const uint row = tid / p.num_active_cells;
  const uint cell_idx = tid % p.num_active_cells;
  const ReductionCell cell = cells[cell_idx];
  float sum = 0.0f;
  float compensation = 0.0f;
  const ulong chunk_base = ulong(row) * p.num_chunks;
  for (uint i = cell.begin; i < cell.end; ++i) {
    const float y = chunk_sums[chunk_base + i] - compensation;
    const float next = sum + y;
    compensation = (next - sum) - y;
    sum = next;
  }

  const ulong output_row = ulong(row) + p.row_offset;
  const ulong out = (output_row * p.num_groups + cell.group) * (p.num_cols + 1) +
                    cell.feature;
  phis[out] = sum;
}
