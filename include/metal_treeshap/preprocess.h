// metal-treeshap: host-side preprocessing.
// CPU port of gpu_treeshap.h's device preprocessing pipeline (DeduplicatePaths, GetPathLengths,
// ValidatePaths, BFD/FFD/NF bin packing, SortPaths, GetBinSegments, ComputeBias).
//
// Rationale: on Apple Silicon unified memory there is no transfer cost to preprocess on the CPU,
// so the thrust/cub dependency is replaced with std:: algorithms. All O(model), run once per
// explain call. Pure C++20 — tested on any platform (see tests/test_preprocess.cpp).
#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

#include "paths.h"

namespace metal_treeshap {

inline constexpr int kBinLimit = 32;  // SIMD-group width on Apple GPUs == CUDA warp size

// Sort by (path_idx, feature_idx) and merge duplicate features within a path:
// intersect intervals, multiply zero_fractions. Mirrors upstream DeduplicatePaths.
inline std::vector<PathElement> DeduplicatePaths(std::vector<PathElement> paths) {
  std::sort(paths.begin(), paths.end(), [](const PathElement& a, const PathElement& b) {
    if (a.path_idx != b.path_idx) return a.path_idx < b.path_idx;
    return a.feature_idx < b.feature_idx;
  });
  std::vector<PathElement> out;
  out.reserve(paths.size());
  for (const auto& e : paths) {
    if (!out.empty() && out.back().path_idx == e.path_idx &&
        out.back().feature_idx == e.feature_idx) {
      out.back().split_condition.Merge(e.split_condition);
      out.back().zero_fraction *= e.zero_fraction;
    } else {
      out.push_back(e);
    }
  }
  return out;
}

// Length (element count, root included) of each unique path. Requires deduplicated input.
// Returns map keyed by path_idx (path ids need not be dense).
inline std::map<uint64_t, int> GetPathLengths(const std::vector<PathElement>& paths) {
  std::map<uint64_t, int> lengths;
  for (const auto& e : paths) lengths[e.path_idx]++;
  return lengths;
}

// Raw-element and per-path validation, run BEFORE deduplication. Dedup merges elements
// (multiplying fractions, intersecting intervals, keeping the first element's metadata),
// which can launder malformed input into plausible-looking merged elements: duplicate
// roots collapse to one, conflicting group/leaf metadata on same-feature duplicates
// vanishes, and individually invalid fractions can multiply into a valid product
// (2 x 0.25, or -0.5 x -0.5). Every check here is therefore on the raw, pre-merge input.
inline void ValidateRawPaths(const std::vector<PathElement>& raw, size_t num_cols,
                             size_t num_groups) {
  struct PathMeta {
    int roots = 0;
    float v = 0.0f;
    int32_t group = 0;
    bool seen = false;
  };
  std::map<uint64_t, PathMeta> meta;
  for (const PathElement& e : raw) {
    if (e.path_idx > 0xFFFFFFFFull) {
      throw std::invalid_argument("path_idx does not fit uint32 (GPU layout)");
    }
    if (e.group < 0 || static_cast<size_t>(e.group) >= num_groups) {
      throw std::invalid_argument("group out of range");
    }
    if (!std::isfinite(e.zero_fraction) || e.zero_fraction < 0.0 || e.zero_fraction > 1.0) {
      throw std::invalid_argument("zero_fraction must be finite and in [0, 1]");
    }
    if (!std::isfinite(e.v)) throw std::invalid_argument("leaf value must be finite");
    const auto& sc = e.split_condition;
    if (std::isnan(sc.feature_lower_bound) || std::isnan(sc.feature_upper_bound)) {
      throw std::invalid_argument("split interval bounds must not be NaN");
    }
    if (sc.feature_lower_bound > sc.feature_upper_bound) {
      throw std::invalid_argument("split interval is inverted (lower > upper)");
    }
    if (e.IsRoot()) {
      if (e.zero_fraction != 1.0) {
        throw std::invalid_argument("root element must have zero_fraction == 1.0");
      }
    } else if (e.feature_idx < 0 || static_cast<size_t>(e.feature_idx) >= num_cols ||
               e.feature_idx > 0x7FFFFFFFll) {
      throw std::invalid_argument("feature_idx out of range");
    }
    PathMeta& m = meta[e.path_idx];
    if (e.IsRoot()) m.roots++;
    if (!m.seen) {
      m.v = e.v;
      m.group = e.group;
      m.seen = true;
    } else {
      if (e.v != m.v) {
        throw std::invalid_argument("Leaf value v should be the same across a single path");
      }
      if (e.group != m.group) {
        throw std::invalid_argument("group must be the same across a single path");
      }
    }
  }
  for (const auto& [idx, m] : meta) {
    (void)idx;
    if (m.roots == 0) throw std::invalid_argument("path is missing its root element");
    if (m.roots > 1) throw std::invalid_argument("path has more than one root element");
  }
}

// Structural + range validation. Beyond upstream's depth/leaf-value checks, this rejects
// anything that could reach out-of-bounds output indexing on the GPU or CPU: bad
// feature/group ranges, missing or duplicated roots, non-finite or out-of-range fractions
// and leaf values, and values that don't survive narrowing to the 32-bit GPU layout.
// `deduplicated` must be sorted by (path_idx, feature_idx), as DeduplicatePaths produces.
inline void ValidatePaths(const std::vector<PathElement>& deduplicated,
                          const std::map<uint64_t, int>& lengths, size_t num_cols,
                          size_t num_groups) {
  for (const auto& [idx, len] : lengths) {
    (void)idx;
    if (len > kBinLimit) throw std::invalid_argument("Tree depth must be < 32");
  }
  std::map<uint64_t, int> roots;
  for (size_t i = 0; i < deduplicated.size(); i++) {
    const PathElement& e = deduplicated[i];
    if (e.path_idx > 0xFFFFFFFFull) {
      throw std::invalid_argument("path_idx does not fit uint32 (GPU layout)");
    }
    if (e.group < 0 || static_cast<size_t>(e.group) >= num_groups) {
      throw std::invalid_argument("group out of range");
    }
    if (e.IsRoot()) {
      roots[e.path_idx]++;
      if (e.zero_fraction != 1.0) {
        throw std::invalid_argument("root element must have zero_fraction == 1.0");
      }
    } else {
      if (e.feature_idx < 0 || static_cast<size_t>(e.feature_idx) >= num_cols ||
          e.feature_idx > 0x7FFFFFFFll) {
        throw std::invalid_argument("feature_idx out of range");
      }
      // MERGED-interval check: two individually valid raw intervals on a repeated
      // feature (e.g. [0,1) and [2,3)) intersect to an inverted/empty interval after
      // Merge. Real tree paths nest their conditions, so an empty merged interval means
      // corrupted input, not a legal model.
      if (e.split_condition.feature_lower_bound >= e.split_condition.feature_upper_bound) {
        throw std::invalid_argument(
            "merged split interval is empty (lower >= upper) — repeated-feature "
            "conditions on a real tree path must nest");
      }
    }
    if (!std::isfinite(e.zero_fraction) || e.zero_fraction < 0.0 || e.zero_fraction > 1.0) {
      throw std::invalid_argument("zero_fraction must be finite and in [0, 1]");
    }
    if (!std::isfinite(e.v)) throw std::invalid_argument("leaf value must be finite");
    // Leaf value and group must be constant along a path (input is sorted by path_idx).
    if (i > 0 && e.path_idx == deduplicated[i - 1].path_idx) {
      if (e.v != deduplicated[i - 1].v) {
        throw std::invalid_argument("Leaf value v should be the same across a single path");
      }
      if (e.group != deduplicated[i - 1].group) {
        throw std::invalid_argument("group must be the same across a single path");
      }
    }
  }
  for (const auto& [idx, len] : lengths) {
    auto it = roots.find(idx);
    (void)len;
    if (it == roots.end()) throw std::invalid_argument("path is missing its root element");
    if (it->second != 1) throw std::invalid_argument("path has more than one root element");
  }
}

// --- Bin packing: pack variable-length paths into 32-lane SIMD-groups. ---
// Faithful ports of upstream BFDBinPacking / FFDBinPacking / NFBinPacking.

using LengthMap = std::map<uint64_t, int>;  // path_idx -> length
using BinMap = std::map<uint64_t, size_t>;  // path_idx -> bin

// Best Fit Decreasing, O(n log n) via balanced tree over residual capacities. Upstream default.
inline BinMap BFDBinPacking(const LengthMap& lengths, int bin_limit = kBinLimit) {
  using kv = std::pair<size_t, int>;  // (bin id, residual capacity)
  struct Cmp {
    bool operator()(const kv& l, const kv& r) const {
      return l.second == r.second ? l.first < r.first : l.second < r.second;
    }
  };
  std::vector<std::pair<uint64_t, int>> sorted(lengths.begin(), lengths.end());
  std::stable_sort(sorted.begin(), sorted.end(),
                   [](const auto& a, const auto& b) { return a.second > b.second; });

  BinMap bin_map;
  std::set<kv, Cmp> bins;
  bins.insert({0, bin_limit});
  size_t num_bins = 1;
  for (const auto& [path_idx, len] : sorted) {
    auto itr = bins.lower_bound({0, len});  // tightest bin with capacity >= len
    if (itr == bins.end()) {
      bins.insert({num_bins, bin_limit - len});
      bin_map[path_idx] = num_bins++;
    } else {
      kv entry = *itr;
      bins.erase(itr);
      entry.second -= len;
      bins.insert(entry);
      bin_map[path_idx] = entry.first;
    }
  }
  return bin_map;
}

// First Fit Decreasing, O(n^2). Kept for tests/comparison, as upstream does.
inline BinMap FFDBinPacking(const LengthMap& lengths, int bin_limit = kBinLimit) {
  std::vector<std::pair<uint64_t, int>> sorted(lengths.begin(), lengths.end());
  std::stable_sort(sorted.begin(), sorted.end(),
                   [](const auto& a, const auto& b) { return a.second > b.second; });
  BinMap bin_map;
  std::vector<int> capacities;
  for (const auto& [path_idx, len] : sorted) {
    bool placed = false;
    for (size_t j = 0; j < capacities.size(); j++) {
      if (capacities[j] >= len) {
        capacities[j] -= len;
        bin_map[path_idx] = j;
        placed = true;
        break;
      }
    }
    if (!placed) {
      capacities.push_back(bin_limit - len);
      bin_map[path_idx] = capacities.size() - 1;
    }
  }
  return bin_map;
}

// Next Fit, O(n).
inline BinMap NFBinPacking(const LengthMap& lengths, int bin_limit = kBinLimit) {
  BinMap bin_map;
  size_t current_bin = 0;
  int capacity = bin_limit;
  for (const auto& [path_idx, len] : lengths) {
    if (len <= capacity) {
      capacity -= len;
      bin_map[path_idx] = current_bin;
    } else {
      capacity = bin_limit - len;
      bin_map[path_idx] = ++current_bin;
    }
  }
  return bin_map;
}

// Sort elements by (bin, path_idx, feature_idx) so each bin's elements are contiguous and each
// path's elements are contiguous within its bin, root (feature -1) first. Mirrors SortPaths.
inline void SortPathsByBin(std::vector<PathElement>* paths, const BinMap& bin_map) {
  std::sort(paths->begin(), paths->end(), [&](const PathElement& a, const PathElement& b) {
    size_t a_bin = bin_map.at(a.path_idx), b_bin = bin_map.at(b.path_idx);
    if (a_bin != b_bin) return a_bin < b_bin;
    if (a.path_idx != b.path_idx) return a.path_idx < b.path_idx;
    return a.feature_idx < b.feature_idx;
  });
}

// [start, end) offsets of each bin in the sorted element array. Mirrors GetBinSegments.
inline std::vector<uint32_t> GetBinSegments(const std::vector<PathElement>& sorted_paths,
                                            const BinMap& bin_map) {
  size_t num_bins = 0;
  for (const auto& [p, b] : bin_map) {
    (void)p;
    num_bins = std::max(num_bins, b + 1);
  }
  std::vector<uint32_t> counts(num_bins + 1, 0);
  for (const auto& e : sorted_paths) counts[bin_map.at(e.path_idx) + 1]++;
  for (size_t i = 1; i < counts.size(); i++) counts[i] += counts[i - 1];  // exclusive scan
  return counts;
}

// Expected value per group in fp64, from RAW (pre-dedup) paths:
// bias[group] = sum over paths of (prod zero_fraction) * v.  Mirrors ComputeBias.
inline std::vector<double> ComputeBias(const std::vector<PathElement>& raw_paths,
                                       size_t num_groups) {
  std::map<uint64_t, std::pair<double, const PathElement*>> per_path;  // prod zf, representative
  for (const auto& e : raw_paths) {
    auto [itr, inserted] = per_path.try_emplace(e.path_idx, 1.0, &e);
    (void)inserted;  // v and group are constant along a path; first element is representative
    itr->second.first *= e.zero_fraction;
  }
  std::vector<double> bias(num_groups, 0.0);
  for (const auto& [idx, pv] : per_path) {
    (void)idx;
    bias.at(static_cast<size_t>(pv.second->group)) += pv.first * pv.second->v;
  }
  return bias;
}

struct Preprocessed {
  std::vector<PathElement> elements;   // deduplicated, sorted by (bin, path, feature)
  std::vector<uint32_t> bin_segments;  // size num_bins+1
  std::vector<double> bias;            // per group, fp64
  size_t num_bins = 0;

  std::vector<GpuPathElement> PackForGpu() const {
    std::vector<GpuPathElement> out;
    out.reserve(elements.size());
    for (const auto& e : elements) out.push_back(Pack(e));
    return out;
  }
};

// Full pipeline, mirroring upstream PreprocessPaths + ComputeBias. num_cols/num_groups
// bound the validation so malformed input cannot reach unchecked output indexing.
// Validation runs in two layers: ValidateRawPaths BEFORE dedup (merging can launder
// malformed raw input), ValidatePaths after (depth <= 32 is a post-merge property, and
// re-checking the merged elements is cheap defense in depth).
inline Preprocessed Preprocess(const std::vector<PathElement>& raw_paths, size_t num_groups,
                               size_t num_cols) {
  Preprocessed result;
  ValidateRawPaths(raw_paths, num_cols, num_groups);
  result.elements = DeduplicatePaths(raw_paths);
  auto lengths = GetPathLengths(result.elements);
  ValidatePaths(result.elements, lengths, num_cols, num_groups);
  result.bias = ComputeBias(raw_paths, num_groups);
  auto bin_map = BFDBinPacking(lengths);
  SortPathsByBin(&result.elements, bin_map);
  result.bin_segments = GetBinSegments(result.elements, bin_map);
  result.num_bins = result.bin_segments.size() - 1;
  return result;
}

}  // namespace metal_treeshap
