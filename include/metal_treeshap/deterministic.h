// metal-treeshap: host plan for the Phase-2 deterministic accumulation path.
//
// The fast kernel atomically scatters one contribution per non-root path element.
// Deterministic mode instead assigns every such element a canonical partial slot,
// writes the slots without atomics, then reduces each (group, feature) segment in a
// fixed order.  Rows are tiled so scratch use is bounded independently of row count.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "preprocess.h"

namespace metal_treeshap {

inline constexpr uint32_t kNoPartialSlot = std::numeric_limits<uint32_t>::max();

// GPU-facing layout consumed by the deterministic kernels in shaders/treeshap.metal.
struct DeterministicReductionCell {
  uint32_t group;
  uint32_t feature;
  uint32_t begin;          // canonical partial-slot segment [begin, end)
  uint32_t end;
};
static_assert(sizeof(DeterministicReductionCell) == 16);
static_assert(alignof(DeterministicReductionCell) == 4);

struct DeterministicPlan {
  // Indexed exactly like Preprocessed::elements. Roots carry kNoPartialSlot.
  std::vector<uint32_t> partial_slot_by_element;

  // One entry per (group, feature) that occurs in the model. Cells are sorted by
  // (group, feature). Slots in each segment are sorted by path_idx, then element index.
  std::vector<DeterministicReductionCell> active_cells;

  size_t num_partials = 0;

  size_t ScratchBytesPerRow() const {
    if (num_partials > std::numeric_limits<size_t>::max() / sizeof(float)) {
      throw std::overflow_error("deterministic scratch bytes per row overflow");
    }
    return num_partials * sizeof(float);
  }

  // Largest row tile that respects scratch_budget_bytes. A non-empty plan needs
  // room for at least one row; this explicit failure prevents an accidental
  // zero-row dispatch loop in the host integration.
  size_t TileRows(size_t num_rows, size_t scratch_budget_bytes) const {
    if (num_rows == 0) return 0;
    if (num_partials == 0) return num_rows;  // bias-only / no feature work
    const size_t bytes_per_row = ScratchBytesPerRow();
    size_t rows = scratch_budget_bytes / bytes_per_row;
    if (rows == 0) {
      throw std::invalid_argument(
          "deterministic scratch budget cannot hold one row of partials");
    }
    // Both reduction kernels use a 32-bit 1-D grid coordinate. Cap the tile so
    // row * active_cell cannot wrap even when scratch is effectively unlimited.
    if (!active_cells.empty()) {
      rows = std::min(rows, static_cast<size_t>(std::numeric_limits<uint32_t>::max()) /
                                active_cells.size());
      if (rows == 0) {
        throw std::invalid_argument(
            "one deterministic output row exceeds 32-bit reduction dispatch width");
      }
    }
    return std::min(num_rows, rows);
  }
};

// Final row-tile bound shared by the host and portable tests. It combines the configured
// scratch cap, Metal's maximum buffer length, and both kernels' uint grid coordinates.
inline size_t DeterministicTileRows(size_t num_rows, size_t bytes_per_row,
                                    size_t active_cells, size_t num_bins,
                                    uint32_t rows_per_simdgroup,
                                    uint32_t threads_per_threadgroup,
                                    size_t scratch_budget_bytes,
                                    size_t device_max_buffer_bytes) {
  if (num_rows == 0) return 0;
  if (bytes_per_row == 0 || active_cells == 0 || num_bins == 0 ||
      rows_per_simdgroup == 0 || threads_per_threadgroup < 32 ||
      threads_per_threadgroup % 32 != 0) {
    throw std::invalid_argument("invalid deterministic tile shape");
  }
  const size_t scratch_limit =
      std::min(scratch_budget_bytes, device_max_buffer_bytes);
  size_t tile_rows = scratch_limit / bytes_per_row;
  if (tile_rows == 0) {
    throw std::invalid_argument(
        "deterministic scratch budget/device buffer limit cannot hold one row of partials");
  }
  tile_rows = std::min(tile_rows, num_rows);

  constexpr uint64_t kMaxGridThreads =
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1;
  const uint64_t reduction_row_limit =
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) / active_cells;
  tile_rows = std::min(tile_rows, static_cast<size_t>(reduction_row_limit));

  const uint64_t simdgroups_per_tg = threads_per_threadgroup / 32;
  const uint64_t max_tg_count = kMaxGridThreads / threads_per_threadgroup;
  const uint64_t max_simdgroups = max_tg_count * simdgroups_per_tg;
  const uint64_t max_banks = max_simdgroups / num_bins;
  const uint64_t stage1_row_limit = std::min(
      max_banks * rows_per_simdgroup,
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
  tile_rows = std::min(tile_rows, static_cast<size_t>(stage1_row_limit));
  if (tile_rows == 0) {
    throw std::invalid_argument("deterministic workload exceeds 32-bit dispatch limits");
  }
  return tile_rows;
}

// Build a canonical scatter/reduction plan from the already validated, deduplicated,
// bin-sorted model. Canonical slot order is independent of the bin-packing order:
// (group, feature, path_idx, element_index). The element index only breaks malformed
// ties defensively; preprocessing guarantees at most one element per path/feature.
inline DeterministicPlan BuildDeterministicPlan(const Preprocessed& pp, size_t num_groups,
                                                size_t num_cols) {
  if (num_groups == 0 || num_cols == 0) {
    throw std::invalid_argument("deterministic plan requires positive groups and columns");
  }
  if (pp.elements.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("deterministic element count does not fit uint32");
  }
  if (num_cols >= std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("deterministic output stride does not fit uint32");
  }
  if (num_groups > std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("deterministic group count does not fit uint32");
  }

  struct Entry {
    uint32_t group;
    uint32_t feature;
    uint32_t path;
    uint32_t element;
  };
  std::vector<Entry> entries;
  entries.reserve(pp.elements.size());
  for (size_t i = 0; i < pp.elements.size(); ++i) {
    const PathElement& e = pp.elements[i];
    if (e.IsRoot()) continue;
    if (e.group < 0 || static_cast<size_t>(e.group) >= num_groups || e.feature_idx < 0 ||
        static_cast<size_t>(e.feature_idx) >= num_cols) {
      throw std::invalid_argument("preprocessed element is outside deterministic plan shape");
    }
    entries.push_back(Entry{static_cast<uint32_t>(e.group),
                            static_cast<uint32_t>(e.feature_idx),
                            static_cast<uint32_t>(e.path_idx), static_cast<uint32_t>(i)});
  }
  if (entries.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("deterministic partial count does not fit uint32");
  }
  std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
    return std::tie(a.group, a.feature, a.path, a.element) <
           std::tie(b.group, b.feature, b.path, b.element);
  });

  DeterministicPlan plan;
  plan.partial_slot_by_element.assign(pp.elements.size(), kNoPartialSlot);
  plan.num_partials = entries.size();
  size_t begin = 0;
  while (begin < entries.size()) {
    size_t end = begin + 1;
    while (end < entries.size() && entries[end].group == entries[begin].group &&
           entries[end].feature == entries[begin].feature) {
      ++end;
    }
    plan.active_cells.push_back(DeterministicReductionCell{
        entries[begin].group, entries[begin].feature, static_cast<uint32_t>(begin),
        static_cast<uint32_t>(end)});
    begin = end;
  }
  for (size_t slot = 0; slot < entries.size(); ++slot) {
    plan.partial_slot_by_element[entries[slot].element] = static_cast<uint32_t>(slot);
  }
  return plan;
}

}  // namespace metal_treeshap
