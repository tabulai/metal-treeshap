#if !defined(__APPLE__)

#include <iostream>
int main() {
  std::cout << "SKIP: Metal host tests require macOS\n";
  return 0;
}

#else

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include "../src/metal_host.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../reference/reference_shap.h"

using namespace metal_treeshap;

namespace {

int checks = 0;

void Check(bool condition, const char* message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

void Throws(const std::function<void()>& fn, const char* message) {
  ++checks;
  try {
    fn();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(message);
}

std::string ReadFile(const char* path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error(std::string("cannot open shader: ") + path);
  std::stringstream out;
  out << in.rdbuf();
  return out.str();
}

PathElement Element(uint64_t path, int64_t feature, double zero_fraction, float leaf,
                    XgboostSplitCondition condition = {}) {
  PathElement e;
  e.path_idx = path;
  e.feature_idx = feature;
  e.group = 0;
  e.zero_fraction = zero_fraction;
  e.v = leaf;
  e.split_condition = condition;
  return e;
}

void CheckClose(const std::vector<float>& actual, const std::vector<float>& expected,
                float tolerance, const char* message) {
  Check(actual.size() == expected.size(), "output size mismatch");
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > tolerance) {
      throw std::runtime_error(std::string(message) + " at index " + std::to_string(i) +
                               ": got " + std::to_string(actual[i]) + ", expected " +
                               std::to_string(expected[i]));
    }
  }
  ++checks;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " shaders/treeshap.metal\n";
    return 2;
  }

  Explainer explainer(ReadFile(argv[1]), Explainer::LibraryKind::kSourceString);
  Throws([&] { explainer.set_rows_per_simdgroup(0); }, "zero tuning value accepted");
  Throws([&] { explainer.set_threads_per_threadgroup(31); },
         "non-SIMD threadgroup size accepted");
  Throws([&] { explainer.set_threads_per_threadgroup(96); },
         "unsupported threadgroup size accepted");
  Throws([&] {
    explainer.set_accumulation_mode(static_cast<AccumulationMode>(99));
  }, "invalid accumulation mode accepted");
  Throws([&] { explainer.set_deterministic_scratch_budget_bytes(0); },
         "zero deterministic scratch budget accepted");
  Check(explainer.deterministic_scratch_budget_bytes() ==
            Explainer::kDefaultDeterministicScratchBudgetBytes,
        "wrong default deterministic scratch budget");
  Check(explainer.atomic_tile_rows() == 0, "wrong default atomic tile rows");
  Throws(
      [&] {
        explainer.set_atomic_tile_rows(
            static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1);
      },
      "oversized atomic tile accepted");

  // Empty models are a valid bias-only fast path and never dispatch Metal work.
  auto empty = explainer.Compile({}, 2, 2, {0.25, -0.5});
  Check(empty->empty(), "empty model unexpectedly contains GPU work");
  std::vector<float> x_empty(4, 0.0f), out_empty(12, 99.0f);
  const ExplainTimings empty_tm =
      explainer.Explain(*empty, x_empty.data(), 2, out_empty.data());
  Check(!empty_tm.dispatched, "empty model dispatched GPU work");
  const std::vector<float> empty_expected{
      0, 0, 0.25f, 0, 0, -0.5f, 0, 0, 0.25f, 0, 0, -0.5f};
  CheckClose(out_empty, empty_expected, 0.0f, "bias-only output mismatch");

  // The zero-row API deliberately permits null pointers and performs no work. The CLI
  // rejects empty matrices separately because a CSV cannot communicate num_cols.
  const ExplainTimings zero_tm = explainer.Explain(*empty, nullptr, 0, nullptr);
  Check(!zero_tm.dispatched, "zero-row call dispatched GPU work");
  Throws([&] { (void)explainer.Explain(*empty, nullptr, 1, out_empty.data()); },
         "positive-row null X accepted");

  const double too_large = 2.0 * static_cast<double>(std::numeric_limits<float>::max());
  Throws([&] { (void)explainer.Compile({}, 1, 1, {too_large}); },
         "float-overflowing combined bias accepted");
  Throws([&] {
    (void)explainer.Compile({}, 1, 1, {std::numeric_limits<double>::quiet_NaN()});
  }, "NaN intercept accepted");

  // A one-feature two-leaf tree exercises a real dispatch. Compare every output to the
  // same float recurrence used by the portable oracle, then resize/reuse both persistent
  // buffers and repeat under all three row-bank settings used by fixture validation.
  const std::vector<PathElement> paths{
      Element(0, -1, 1.0, -1.0f),
      Element(0, 0, 0.4, -1.0f,
              XgboostSplitCondition(-std::numeric_limits<float>::infinity(), 0.0f, false)),
      Element(1, -1, 1.0, 2.0f),
      Element(1, 0, 0.6, 2.0f,
              XgboostSplitCondition(0.0f, std::numeric_limits<float>::infinity(), true)),
  };
  const std::vector<double> intercept{0.25};
  auto model = explainer.Compile(paths, 1, 1, intercept);
  Check(!model->empty(), "non-empty model lost its work");
  Check(model->storage_mode() == ModelStorageMode::kShared,
        "default model storage is not shared");
  Check(model->elements()->storageMode() == MTL::StorageModeShared &&
            model->segments()->storageMode() == MTL::StorageModeShared,
        "shared model buffers have the wrong Metal storage mode");
  Check(model->atomic_writes_per_row() == 2, "wrong baseline atomic-write estimate");
  Check(model->simdgroup_writes_per_row() == 1,
        "wrong SIMD-group atomic-write estimate");
  Check(model->deterministic_num_partials() == 2,
        "wrong deterministic partial count");
  Check(model->deterministic_num_active_cells() == 1,
        "wrong deterministic active-cell count");
  Check(model->deterministic_scratch_bytes_per_row() == 2 * sizeof(float),
        "wrong deterministic scratch bytes per row");
  Check(model->deterministic_slots()->storageMode() == MTL::StorageModeShared &&
            model->deterministic_cells()->storageMode() == MTL::StorageModeShared,
        "shared deterministic buffers have the wrong Metal storage mode");
  auto private_model =
      explainer.Compile(paths, 1, 1, intercept, ModelStorageMode::kPrivate);
  Check(private_model->storage_mode() == ModelStorageMode::kPrivate,
        "private model storage not retained");
  Check(private_model->elements()->storageMode() == MTL::StorageModePrivate &&
            private_model->segments()->storageMode() == MTL::StorageModePrivate,
        "private model buffers have the wrong Metal storage mode");
  Check(private_model->atomic_writes_per_row() == model->atomic_writes_per_row(),
        "storage mode changed baseline traffic estimate");
  Check(private_model->simdgroup_writes_per_row() ==
            model->simdgroup_writes_per_row(),
        "storage mode changed SIMD-group traffic estimate");
  Check(private_model->deterministic_slots()->storageMode() == MTL::StorageModePrivate &&
            private_model->deterministic_cells()->storageMode() == MTL::StorageModePrivate,
        "private deterministic buffers have the wrong Metal storage mode");
  Throws([&] {
    (void)explainer.Compile(paths, 1, 1, intercept,
                            static_cast<ModelStorageMode>(99));
  }, "invalid model storage mode accepted");

  const std::vector<float> x{-1.0f, 1.0f, std::numeric_limits<float>::quiet_NaN()};
  DenseDataset dataset{x.data(), 3, 1};
  const Preprocessed pp = Preprocess(paths, 1, 1);
  std::vector<float> expected(6, 0.0f);
  ShapReference(dataset, pp, 1, expected.data(), intercept);

  for (AccumulationMode accumulation : {AccumulationMode::kAtomic,
                                        AccumulationMode::kSimdgroup,
                                        AccumulationMode::kDeterministic}) {
    explainer.set_accumulation_mode(accumulation);
    for (uint32_t threads_per_tg : {32u, 64u, 128u, 256u}) {
      explainer.set_threads_per_threadgroup(threads_per_tg);
      for (uint32_t rows_per_sg : {1u, 7u, 1024u}) {
        explainer.set_rows_per_simdgroup(rows_per_sg);
        for (const CompiledModel* active_model : {model.get(), private_model.get()}) {
          std::vector<float> actual(6, -999.0f);
          const ExplainTimings tm =
              explainer.Explain(*active_model, x.data(), 3, actual.data());
          Check(tm.dispatched, "non-empty model did not dispatch");
          CheckClose(actual, expected, 3e-6f, "Metal/reference mismatch");
        }
      }
    }
  }

  // Atomic tiling reuses the deterministic row-offset convention, but writes directly
  // into disjoint output rows. Pin both partial final tiles and full-dispatch equality.
  explainer.set_accumulation_mode(AccumulationMode::kAtomic);
  explainer.set_threads_per_threadgroup(256);
  explainer.set_rows_per_simdgroup(7);
  std::vector<float> atomic_full(6, -5.0f);
  explainer.set_atomic_tile_rows(0);
  const ExplainTimings atomic_full_tm =
      explainer.Explain(*model, x.data(), 3, atomic_full.data());
  Check(atomic_full_tm.atomic_tile_rows == 3 && atomic_full_tm.atomic_tiles == 1,
        "full atomic dispatch metadata mismatch");
  std::vector<float> atomic_tiled(6, -6.0f);
  explainer.set_atomic_tile_rows(2);
  const ExplainTimings atomic_tiled_tm =
      explainer.Explain(*model, x.data(), 3, atomic_tiled.data());
  Check(atomic_tiled_tm.atomic_tile_rows == 2 && atomic_tiled_tm.atomic_tiles == 2,
        "tiled atomic dispatch metadata mismatch");
  Check(std::memcmp(atomic_full.data(), atomic_tiled.data(),
                    atomic_full.size() * sizeof(float)) == 0,
        "atomic output changed with row tiling");
  explainer.set_atomic_tile_rows(0);

  // Force exactly one row per deterministic tile, then verify bitwise repeatability
  // across 100 dispatches and invariance against a single-tile run. This exercises
  // scratch reuse and the buffer barriers inside one command buffer.
  explainer.set_accumulation_mode(AccumulationMode::kDeterministic);
  explainer.set_threads_per_threadgroup(256);
  explainer.set_rows_per_simdgroup(7);
  explainer.set_deterministic_scratch_budget_bytes(
      model->deterministic_scratch_bytes_per_row());
  Check(explainer.deterministic_scratch_capacity_bytes() == 0,
        "lowering deterministic budget retained an oversized scratch buffer");
  std::vector<float> tiled(6, -1.0f);
  const ExplainTimings tiled_tm =
      explainer.Explain(*model, x.data(), 3, tiled.data());
  Check(tiled_tm.deterministic_tile_rows == 1 && tiled_tm.deterministic_tiles == 3 &&
            tiled_tm.deterministic_scratch_bytes == 2 * sizeof(float) &&
            tiled_tm.deterministic_scratch_capacity_bytes <=
                explainer.deterministic_scratch_budget_bytes(),
        "deterministic one-row tiling metadata mismatch");
  CheckClose(tiled, expected, 3e-6f, "tiled deterministic/reference mismatch");
  for (int repeat = 0; repeat < 100; ++repeat) {
    std::vector<float> repeated(6, -2.0f);
    explainer.Explain(*model, x.data(), 3, repeated.data());
    Check(std::memcmp(repeated.data(), tiled.data(), tiled.size() * sizeof(float)) == 0,
          "deterministic output changed across identical runs");
  }
  explainer.set_deterministic_scratch_budget_bytes(
      Explainer::kDefaultDeterministicScratchBudgetBytes);
  std::vector<float> single_tile(6, -3.0f);
  const ExplainTimings single_tm =
      explainer.Explain(*model, x.data(), 3, single_tile.data());
  Check(single_tm.deterministic_tiles == 1 && single_tm.deterministic_tile_rows == 3,
        "deterministic default budget did not use one tile");
  Check(std::memcmp(single_tile.data(), tiled.data(), tiled.size() * sizeof(float)) == 0,
        "deterministic output changed with tile size");

  // Shrink then grow again after the tuning matrix to exercise persistent capacity reuse
  // and output reset with the non-default strategy.
  std::vector<float> one_row(2, -999.0f);
  explainer.Explain(*private_model, x.data() + 1, 1, one_row.data());
  CheckClose(one_row, {expected[2], expected[3]}, 3e-6f, "single-row reuse mismatch");
  std::vector<float> actual(6, -123.0f);
  explainer.Explain(*private_model, x.data(), 3, actual.data());
  CheckClose(actual, expected, 3e-6f, "repeated-call mismatch");

  std::cout << "ALL " << checks << " METAL HOST TESTS PASSED\n";
  return 0;
}

#endif
