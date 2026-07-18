#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../include/metal_treeshap/deterministic.h"

using namespace metal_treeshap;

namespace {
int checks = 0;
void Check(bool value, const char* message) {
  ++checks;
  if (!value) throw std::runtime_error(message);
}

PathElement E(uint64_t path, int64_t feature, int32_t group, float leaf) {
  PathElement e;
  e.path_idx = path;
  e.feature_idx = feature;
  e.group = group;
  e.v = leaf;
  e.zero_fraction = feature < 0 ? 1.0 : 0.5;
  return e;
}
}  // namespace

int main() {
  // Deliberately interleave groups/features/path ids. Preprocess may bin-sort them in
  // another order; the deterministic slots must still be canonical by output cell/path.
  const std::vector<PathElement> raw{
      E(30, -1, 1, 3), E(30, 2, 1, 3), E(30, 0, 1, 3),
      E(10, -1, 0, 1), E(10, 2, 0, 1), E(10, 1, 0, 1),
      E(20, -1, 0, 2), E(20, 2, 0, 2),
  };
  const Preprocessed pp = Preprocess(raw, 2, 3);
  const DeterministicPlan plan = BuildDeterministicPlan(pp, 2, 3);
  Check(plan.partial_slot_by_element.size() == pp.elements.size(), "slot map size");
  Check(plan.num_partials == 5, "partial count excludes roots");
  Check(plan.active_cells.size() == 4, "active cell count");
  // Scratch: 5 partial slots + 4 stage-A chunk sums (every cell fits one chunk).
  Check(plan.chunks.size() == 4, "chunk count for sub-chunk cells");
  Check(plan.ScratchBytesPerRow() == 36, "scratch bytes per row");
  Check(plan.TileRows(1000, 256) == 7, "budgeted tile rows");
  Check(plan.TileRows(3, 4096) == 3, "tile capped to request");
  // Cells within one chunk: each chunk mirrors its cell's slot range, and chunk_cells
  // carries the same outputs with [begin, end) in chunk indices.
  Check(plan.chunk_cells.size() == plan.active_cells.size(), "chunk cell count");
  for (size_t i = 0; i < plan.active_cells.size(); ++i) {
    Check(plan.chunks[i].begin == plan.active_cells[i].begin &&
              plan.chunks[i].end == plan.active_cells[i].end,
          "single-chunk range mismatch");
    Check(plan.chunk_cells[i].group == plan.active_cells[i].group &&
              plan.chunk_cells[i].feature == plan.active_cells[i].feature,
          "chunk cell output mismatch");
    Check(plan.chunk_cells[i].begin == i && plan.chunk_cells[i].end == i + 1,
          "chunk-space cell range mismatch");
  }

  bool small_budget_threw = false;
  try {
    (void)plan.TileRows(1, 19);
  } catch (const std::invalid_argument&) {
    small_budget_threw = true;
  }
  Check(small_budget_threw, "undersized budget accepted");

  // A cell larger than kDeterministicChunkSlots must split into fixed-shape chunks:
  // 600 single-split paths on one (group, feature) -> chunks [0,256), [256,512), [512,600).
  {
    std::vector<PathElement> big_raw;
    for (uint64_t p = 0; p < 600; ++p) {
      big_raw.push_back(E(p, -1, 0, 1.0f));
      big_raw.push_back(E(p, 0, 0, 1.0f));
    }
    const Preprocessed big_pp = Preprocess(big_raw, 1, 1);
    const DeterministicPlan big_plan = BuildDeterministicPlan(big_pp, 1, 1);
    Check(big_plan.num_partials == 600, "big cell partial count");
    Check(big_plan.active_cells.size() == 1, "big cell count");
    Check(big_plan.chunks.size() == 3, "big cell chunk count");
    Check(big_plan.chunks[0].begin == 0 && big_plan.chunks[0].end == 256 &&
              big_plan.chunks[1].begin == 256 && big_plan.chunks[1].end == 512 &&
              big_plan.chunks[2].begin == 512 && big_plan.chunks[2].end == 600,
          "big cell chunk boundaries");
    Check(big_plan.chunk_cells.size() == 1 && big_plan.chunk_cells[0].begin == 0 &&
              big_plan.chunk_cells[0].end == 3,
          "big cell chunk-space range");
    Check(big_plan.ScratchBytesPerRow() == (600 + 3) * sizeof(float),
          "big cell scratch bytes");
  }

  // Active output cells are (g,f): (0,1), (0,2), (1,0), (1,2).
  const std::vector<std::pair<uint32_t, uint32_t>> cells{{0, 1}, {0, 2}, {1, 0}, {1, 2}};
  for (size_t i = 0; i < cells.size(); ++i) {
    Check(plan.active_cells[i].group == cells[i].first &&
              plan.active_cells[i].feature == cells[i].second,
          "cell output ordering");
    Check(plan.active_cells[i].begin < plan.active_cells[i].end, "empty active cell");
  }
  Check(plan.active_cells[1].end - plan.active_cells[1].begin == 2,
        "two paths sharing one output were not grouped");

  size_t roots = 0, assigned = 0;
  for (size_t i = 0; i < pp.elements.size(); ++i) {
    if (pp.elements[i].IsRoot()) {
      ++roots;
      Check(plan.partial_slot_by_element[i] == kNoPartialSlot, "root received a slot");
    } else {
      ++assigned;
      Check(plan.partial_slot_by_element[i] < plan.num_partials, "feature missing slot");
    }
  }
  Check(roots == 3 && assigned == 5, "root/feature accounting");

  DeterministicPlan empty;
  Check(empty.TileRows(17, 0) == 17, "zero-partial plan should not need scratch");
  Check(plan.TileRows(0, 0) == 0, "zero rows should not need scratch");

  Check(DeterministicTileRows(100, 20, 4, 2, 7, 256, 1000, 400) == 20,
        "device maxBufferLength did not cap tile rows");
  Check(DeterministicTileRows(100, 20, 4, 2, 7, 256, 200, 4000) == 10,
        "scratch budget did not cap tile rows");
  bool device_too_small_threw = false;
  try {
    (void)DeterministicTileRows(1, 20, 1, 1, 1, 32, 1000, 19);
  } catch (const std::invalid_argument&) {
    device_too_small_threw = true;
  }
  Check(device_too_small_threw, "sub-row device buffer limit accepted");

  // UINT32_MAX active cells make the serial reduction grid one row wide.
  Check(DeterministicTileRows(10, 4, std::numeric_limits<uint32_t>::max(), 1, 1,
                              32, 1024, 1024) == 1,
        "reduction grid limit did not cap tile rows");

  std::cout << "ALL " << checks << " DETERMINISTIC PLAN TESTS PASSED\n";
}
