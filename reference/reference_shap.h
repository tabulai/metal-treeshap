// metal-treeshap: scalar CPU reference ("oracle") for the SIMD-group kernel.
//
// This simulates, lane for lane, the exact float32 recurrences the Metal kernel implements
// (GroupPath::Extend / UnwoundPathSum / ComputePhi from gpu_treeshap.h), walking the same
// (bin -> contiguous path group -> row) structure produced by preprocess.h. Its purposes:
//   1. Golden model: validated against xgboost.predict(pred_contribs=True) (tests/).
//   2. Bit-faithful target for the Metal kernel bring-up (Phase 1): same inputs, same layout,
//      float arithmetic in the same order; only fast-math/atomic-order differences remain.
//   3. Accumulation-error studies: AccumT = double (upstream behavior) vs float (Metal default).
//
// Pure C++20 — no Metal, no CUDA.
#pragma once

#include <cstdint>
#include <vector>

#include "../include/metal_treeshap/paths.h"
#include "../include/metal_treeshap/preprocess.h"

namespace metal_treeshap {

// Dense row-major dataset; NaN = missing.
struct DenseDataset {
  const float* data = nullptr;
  size_t num_rows = 0;
  size_t num_cols = 0;
  float GetElement(size_t r, size_t c) const { return data[r * num_cols + c]; }
};

namespace detail {

// One (path group, row): cooperative extend across "lanes" (path elements), then each lane
// unwinds its own feature. `elems` points at the group's contiguous, root-first elements.
// Float arithmetic mirrors the kernel; contributions are accumulated into phis as AccumT.
template <typename AccumT>
void ComputeGroupRow(const PathElement* elems, size_t group_size, const DenseDataset& X,
                     size_t row, size_t num_groups, AccumT* phis) {
  constexpr size_t kMaxGroup = 32;
  float one_fraction[kMaxGroup];
  float zero_fraction[kMaxGroup];
  float pweight[kMaxGroup];

  for (size_t r = 0; r < group_size; r++) {
    const PathElement& e = elems[r];
    zero_fraction[r] = static_cast<float>(e.zero_fraction);
    one_fraction[r] =
        e.IsRoot() ? 1.0f
                   : (e.split_condition.EvaluateSplit(
                          X.GetElement(row, static_cast<size_t>(e.feature_idx)))
                          ? 1.0f
                          : 0.0f);
    pweight[r] = (r == 0) ? 1.0f : 0.0f;  // GroupPath constructor: rank 0 holds weight 1
  }

  // --- GroupPath::Extend, d = 1 .. group_size-1 ---
  // Kernel form (all lanes update simultaneously; shfl_up reads pre-update left neighbor):
  //   pweight[r] = pweight[r] * z_d * max(d - r, 0) / (d+1)  +  o_d * pweight[r-1] * r / (d+1)
  // Serially we sweep r downward so pweight[r-1] is still the pre-update value.
  for (size_t d = 1; d < group_size; d++) {
    const float z_d = zero_fraction[d];
    const float o_d = one_fraction[d];
    const float inv = 1.0f / static_cast<float>(d + 1);
    for (size_t r = d;; r--) {  // ranks > d are all zero; sweep d..0
      const float left = (r > 0) ? pweight[r - 1] : 0.0f;  // rank 0 term is multiplied by r==0
      const float absent =
          pweight[r] * z_d * static_cast<float>(d - r) * inv;  // max(d-r,0): r<=d here
      const float present = o_d * left * static_cast<float>(r) * inv;
      pweight[r] = absent + present;
      if (r == 0) break;
    }
  }

  // --- Per lane: UnwoundPathSum + phi ---
  const size_t D = group_size - 1;  // unique_depth_ after all extends
  for (size_t i = 0; i < group_size; i++) {
    const PathElement& e = elems[i];
    if (e.IsRoot()) continue;  // root's phi is never written (and (o-z)==0 anyway)
    const float o_i = one_fraction[i];
    const float z_i = zero_fraction[i];

    float next_one_portion = pweight[D];
    float total = 0.0f;
    const float zero_frac_div_unique_depth = z_i / static_cast<float>(D + 1);
    for (int j = static_cast<int>(D) - 1; j >= 0; j--) {
      const float ith_pweight = pweight[j];
      const float precomputed = static_cast<float>(static_cast<int>(D) - j) *
                                zero_frac_div_unique_depth;
      const float tmp =
          next_one_portion * static_cast<float>(D + 1) / static_cast<float>(j + 1);
      total += tmp * o_i;
      next_one_portion = ith_pweight - tmp * precomputed;
      const float numerator = (1.0f - o_i) * ith_pweight;
      if (precomputed > 0.0f) total += numerator / precomputed;
    }

    const float phi = total * (o_i - z_i) * e.v;
    phis[IndexPhi(row, num_groups, static_cast<size_t>(e.group), X.num_cols,
                  static_cast<size_t>(e.feature_idx))] += static_cast<AccumT>(phi);
  }
}

}  // namespace detail

// A contiguous run of one path's elements within a bin — the unit of cooperative work
// (one SIMD sub-group on the GPU).
struct GroupRange {
  size_t start;
  size_t end;
};

inline std::vector<GroupRange> CollectGroups(const Preprocessed& pp) {
  std::vector<GroupRange> groups;
  for (size_t bin = 0; bin < pp.num_bins; bin++) {
    size_t g_start = pp.bin_segments[bin];
    const size_t end = pp.bin_segments[bin + 1];
    while (g_start < end) {
      size_t g_end = g_start + 1;
      while (g_end < end && pp.elements[g_end].path_idx == pp.elements[g_start].path_idx) {
        g_end++;
      }
      groups.push_back({g_start, g_end});
      g_start = g_end;
    }
  }
  return groups;
}

// First-order SHAP values over the preprocessed (binned, sorted) elements.
// phis must have size num_rows * num_groups * (num_cols + 1), zero-initialized by caller.
//
// `intercepts` (optional, margin space, one per output group — e.g. XGBoost base_score)
// is added to the bias column together with the cover-derived path bias, so the output is
// complete, xgboost-comparable contributions with no post-hoc patching.
//
// `order` (optional) processes path groups in the given permutation instead of the natural
// bin order — a CPU proxy for the scheduling-dependent accumulation order of concurrent
// GPU atomics, used by the fp32 accumulation-order study.
template <typename AccumT>
void ShapReference(const DenseDataset& X, const Preprocessed& pp, size_t num_groups,
                   AccumT* phis, const std::vector<double>& intercepts = {},
                   const std::vector<GroupRange>* order = nullptr) {
  // Bias + intercept: added to every row's bias column, as the host does before the
  // kernel runs.
  for (size_t row = 0; row < X.num_rows; row++) {
    for (size_t g = 0; g < num_groups; g++) {
      const double b = pp.bias[g] + (intercepts.empty() ? 0.0 : intercepts.at(g));
      phis[IndexPhi(row, num_groups, g, X.num_cols, X.num_cols)] += static_cast<AccumT>(b);
    }
  }

  const std::vector<GroupRange> natural = order ? std::vector<GroupRange>{} : CollectGroups(pp);
  const std::vector<GroupRange>& groups = order ? *order : natural;
  for (const GroupRange& g : groups) {
    for (size_t row = 0; row < X.num_rows; row++) {
      detail::ComputeGroupRow(&pp.elements[g.start], g.end - g.start, X, row, num_groups,
                              phis);
    }
  }
}

}  // namespace metal_treeshap
