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
#include <limits>
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
  for (const auto& e : paths) {
    // Dedup output is path-sorted, so the current run's path is the map's maximum key:
    // bump it in O(1) instead of tree-searching per element. Unsorted input still takes
    // the general branch and counts identically.
    if (!lengths.empty() && std::prev(lengths.end())->first == e.path_idx) {
      ++std::prev(lengths.end())->second;
    } else {
      lengths[e.path_idx]++;
    }
  }
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
      // A merged condition is unsatisfiable only when BOTH its numeric interval is empty
      // and NaN cannot follow it.  In particular, XGBoost can emit a missing-only leaf by
      // splitting the same feature at the same threshold twice with opposite numeric
      // branches while routing missing values down both edges.  That produces [t,t) with
      // is_missing_branch=true and is a legal, positive-cover path.
      if (e.split_condition.feature_lower_bound >= e.split_condition.feature_upper_bound &&
          !e.split_condition.is_missing_branch) {
        throw std::invalid_argument(
            "merged split condition is unsatisfiable (empty numeric interval and "
            "missing values do not follow the path)");
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
// Decorate-sort-undecorate: resolving each element's bin once and sorting flat keys replaced
// a comparator doing two O(log paths) map walks per comparison, which dominated Preprocess
// at stress scale (~11x faster sort, identical order — (path_idx, feature_idx) is unique
// after dedup, and the original index breaks any malformed ties deterministically).
inline void SortPathsByBin(std::vector<PathElement>* paths, const BinMap& bin_map) {
  struct SortKey {
    size_t bin;
    uint64_t path_idx;
    int64_t feature_idx;
    size_t index;
  };
  std::vector<SortKey> keys;
  keys.reserve(paths->size());
  bool have_prev = false;
  uint64_t prev_path = 0;
  size_t prev_bin = 0;
  for (size_t i = 0; i < paths->size(); ++i) {
    const PathElement& e = (*paths)[i];
    // Consecutive elements usually share a path (dedup output is path-sorted); reuse the
    // previous lookup so the map is walked once per path, not once per element.
    if (!have_prev || e.path_idx != prev_path) {
      prev_bin = bin_map.at(e.path_idx);
      prev_path = e.path_idx;
      have_prev = true;
    }
    keys.push_back(SortKey{prev_bin, e.path_idx, e.feature_idx, i});
  }
  std::sort(keys.begin(), keys.end(), [](const SortKey& a, const SortKey& b) {
    if (a.bin != b.bin) return a.bin < b.bin;
    if (a.path_idx != b.path_idx) return a.path_idx < b.path_idx;
    if (a.feature_idx != b.feature_idx) return a.feature_idx < b.feature_idx;
    return a.index < b.index;
  });
  std::vector<PathElement> sorted;
  sorted.reserve(keys.size());
  for (const SortKey& key : keys) sorted.push_back((*paths)[key.index]);
  *paths = std::move(sorted);
}

// Convert per-bin size_t counts to the uint32 prefix offsets consumed by the shader.  Kept
// separate so the overflow boundary can be tested without allocating UINT32_MAX elements.
inline std::vector<uint32_t> CheckedBinSegmentsFromCounts(
    const std::vector<size_t>& bin_counts) {
  std::vector<uint32_t> segments(bin_counts.size() + 1, 0);
  size_t total = 0;
  constexpr size_t kMaxOffset = std::numeric_limits<uint32_t>::max();
  for (size_t i = 0; i < bin_counts.size(); ++i) {
    if (total > kMaxOffset || bin_counts[i] > kMaxOffset - total) {
      throw std::overflow_error("deduplicated path element count does not fit uint32 offsets");
    }
    total += bin_counts[i];
    segments[i + 1] = static_cast<uint32_t>(total);
  }
  return segments;
}

// [start, end) offsets of each bin in the sorted element array. Mirrors GetBinSegments.
inline std::vector<uint32_t> GetBinSegments(const std::vector<PathElement>& sorted_paths,
                                            const BinMap& bin_map) {
  // The shader consumes uint32 offsets.  Guard the aggregate before counting, then retain
  // size_t counters until the checked conversion so neither count nor prefix sum can wrap.
  if (sorted_paths.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("deduplicated path element count does not fit uint32 offsets");
  }
  size_t num_bins = 0;
  for (const auto& [p, b] : bin_map) {
    (void)p;
    num_bins = std::max(num_bins, b + 1);
  }
  std::vector<size_t> counts(num_bins, 0);
  for (const auto& e : sorted_paths) counts[bin_map.at(e.path_idx)]++;
  return CheckedBinSegmentsFromCounts(counts);
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
