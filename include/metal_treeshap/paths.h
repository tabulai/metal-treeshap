// metal-treeshap: portable path representation.
// Mirrors gpu_treeshap::PathElement<XgboostSplitCondition> semantics (Apache-2.0 upstream),
// with an explicit 32-byte GPU-facing layout that matches shaders/treeshap.metal.
//
// This header is pure C++20 — no Metal, no CUDA — and is exercised by tests on any platform.
#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

namespace metal_treeshap {

// Feature values in [lower, upper) follow this path element; NaN follows it iff is_missing_branch.
struct XgboostSplitCondition {
  float feature_lower_bound = -std::numeric_limits<float>::infinity();
  float feature_upper_bound = std::numeric_limits<float>::infinity();
  bool is_missing_branch = true;

  XgboostSplitCondition() = default;
  XgboostSplitCondition(float lower, float upper, bool missing)
      : feature_lower_bound(lower), feature_upper_bound(upper), is_missing_branch(missing) {
    assert(feature_lower_bound <= feature_upper_bound);
  }

  bool EvaluateSplit(float x) const {
    if (std::isnan(x)) return is_missing_branch;
    return x >= feature_lower_bound && x < feature_upper_bound;
  }

  // Combine duplicate features on one path: intersect intervals, AND missing branches.
  void Merge(const XgboostSplitCondition& other) {
    feature_lower_bound = std::fmax(feature_lower_bound, other.feature_lower_bound);
    feature_upper_bound = std::fmin(feature_upper_bound, other.feature_upper_bound);
    is_missing_branch = is_missing_branch && other.is_missing_branch;
  }
};

// One element of a unique root-to-leaf path. Host-side: zero_fraction kept in double,
// exactly like upstream; truncated to float only when packed for the GPU.
struct PathElement {
  uint64_t path_idx = 0;   // unique path (leaf) id
  int64_t feature_idx = -1;  // -1 == root element
  int32_t group = 0;         // output class
  XgboostSplitCondition split_condition{};
  double zero_fraction = 1.0;  // P(follow this branch | feature unknown) = cover ratio
  float v = 0.0f;              // leaf value of the path

  bool IsRoot() const { return feature_idx == -1; }
};

// GPU-facing packed element. MUST match `struct PathElement` in shaders/treeshap.metal
// byte-for-byte: 32 bytes, 4-byte aligned, no doubles (Apple GPUs have no fp64).
struct GpuPathElement {
  uint32_t path_idx;
  int32_t feature_idx;  // -1 == root
  int32_t group;
  float zero_fraction;  // truncated from host double
  float v;
  float feature_lower_bound;
  float feature_upper_bound;
  uint32_t is_missing_branch;
};
static_assert(sizeof(GpuPathElement) == 32, "GPU struct layout must match MSL");
static_assert(alignof(GpuPathElement) == 4, "GPU struct layout must match MSL");

inline GpuPathElement Pack(const PathElement& e) {
  return GpuPathElement{
      static_cast<uint32_t>(e.path_idx),
      static_cast<int32_t>(e.feature_idx),
      e.group,
      static_cast<float>(e.zero_fraction),
      e.v,
      e.split_condition.feature_lower_bound,
      e.split_condition.feature_upper_bound,
      e.split_condition.is_missing_branch ? 1u : 0u};
}

// phis layout, identical to upstream: [row][group][feature 0..F-1, bias at F]
inline size_t IndexPhi(size_t row_idx, size_t num_groups, size_t group, size_t num_columns,
                       size_t column_idx) {
  return (row_idx * num_groups + group) * (num_columns + 1) + column_idx;
}

}  // namespace metal_treeshap
