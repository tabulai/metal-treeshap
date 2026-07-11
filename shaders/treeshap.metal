// metal-treeshap: first-order TreeSHAP kernel for Apple GPUs.
//
// Port of gpu_treeshap.h's ShapKernel (CUDA) to Metal Shading Language. See
// docs/01-cuda-acceleration-assessment.md for the algorithm and
// docs/02-apple-gpu-project-proposal.md §5 for the CUDA->Metal mapping decisions.
//
// STATUS: fixture-validated on an M4 Max (external validation rounds v1/v2): compiles
// under Metal 3, execution width 32, max 256 threads/threadgroup honored; EXACT results
// on a simple single-split fixture and on multi-row-bank runs, 5.96e-7 max error on a
// packed multi-path fixture. Model-scale differential testing (full golden suite through
// the metal-cpp host, incl. the tests/fixtures/deep31 32-lane comb case) is Phase 1.
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
//  * atomicAdd(double) -> atomic_float fetch_add (fp32). Accumulation-mode study and
//    fallback designs in proposal §4. Bias is pre-added to the phis buffer by the host.
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
  uint num_rows;
  uint num_cols;
  uint num_groups;
  uint num_bins;              // == bins_per_row in the CUDA code
  uint rows_per_simdgroup;    // CUDA kRowsPerWarp (default 1024; tune in Phase 2)
};

constant constexpr uint kSimdWidth = 32;

inline bool EvaluateSplit(thread const PathElement& e, float x) {
  if (isnan(x)) return e.is_missing_branch != 0;
  return x >= e.lower && x < e.upper;
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

[[max_total_threads_per_threadgroup(256)]]
kernel void shap_first_order(
    device const float* X               [[buffer(0)]],
    device const PathElement* elements  [[buffer(1)]],
    device const uint* bin_segments     [[buffer(2)]],  // num_bins + 1 entries
    device atomic_float* phis           [[buffer(3)]],  // pre-initialized with bias by host
    constant Params& p                  [[buffer(4)]],
    uint tid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
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
    // one_fraction: does this row satisfy my element's split? (root: always 1)
    const float one_fraction =
        is_root ? 1.0f
                : (EvaluateSplit(e, X[row * p.num_cols + uint(e.feature_idx)]) ? 1.0f : 0.0f);

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
    if (!is_root) {
      const ulong out = (row * p.num_groups + ulong(e.group)) * (p.num_cols + 1) +
                        ulong(e.feature_idx);
      atomic_fetch_add_explicit(&phis[out], phi, memory_order_relaxed);
    }
  }
}
