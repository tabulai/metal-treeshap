#if !defined(__APPLE__)
#error "MetalTreeShap Python bindings require macOS"
#endif

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include "../src/metal_host.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <unistd.h>  // getpid, getpagesize

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>  // posix_memalign, free
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using namespace metal_treeshap;

namespace {

template <typename T>
using Input1D =
    nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Matrix =
    nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Output =
    nb::ndarray<nb::numpy, float, nb::ndim<3>, nb::c_contig, nb::device::cpu>;

size_t CheckedMul(size_t a, size_t b, const char* what) {
  if (b != 0 && a > std::numeric_limits<size_t>::max() / b) {
    throw std::overflow_error(std::string(what) + " size overflows");
  }
  return a * b;
}

AccumulationMode ParseAccumulation(const std::string& value) {
  if (value == "atomic") return AccumulationMode::kAtomic;
  if (value == "simdgroup") return AccumulationMode::kSimdgroup;
  if (value == "deterministic") return AccumulationMode::kDeterministic;
  throw std::invalid_argument(
      "accumulation must be 'atomic', 'simdgroup', or 'deterministic'");
}

ModelStorageMode ParseStorage(const std::string& value) {
  if (value == "shared") return ModelStorageMode::kShared;
  if (value == "private") return ModelStorageMode::kPrivate;
  throw std::invalid_argument("model_storage must be 'shared' or 'private'");
}

class NativeExplainer {
 public:
  NativeExplainer(Input1D<uint64_t> path_idx, Input1D<int64_t> feature_idx,
                  Input1D<int32_t> group, Input1D<float> lower,
                  Input1D<float> upper, Input1D<uint8_t> is_missing,
                  Input1D<double> zero_fraction, Input1D<float> leaf,
                  size_t num_groups, size_t num_features,
                  Input1D<double> intercepts, const std::string& kernel_spec,
                  bool kernel_is_metallib, const std::string& model_storage) {
    if (num_groups == 0 || num_features == 0) {
      throw std::invalid_argument("num_groups and num_features must be positive");
    }
    const size_t count = path_idx.shape(0);
    if (feature_idx.shape(0) != count || group.shape(0) != count ||
        lower.shape(0) != count || upper.shape(0) != count ||
        is_missing.shape(0) != count || zero_fraction.shape(0) != count ||
        leaf.shape(0) != count) {
      throw std::invalid_argument("all path columns must have the same length");
    }
    if (intercepts.shape(0) != num_groups) {
      throw std::invalid_argument("intercepts must contain one value per group");
    }

    std::vector<PathElement> paths;
    paths.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      const float lo = lower.data()[i], hi = upper.data()[i];
      if (std::isnan(lo) || std::isnan(hi) || lo > hi) {
        throw std::invalid_argument("path split bounds must be non-NaN with lower <= upper");
      }
      if (is_missing.data()[i] > 1) {
        throw std::invalid_argument("is_missing values must be 0 or 1");
      }
      PathElement element;
      element.path_idx = path_idx.data()[i];
      element.feature_idx = feature_idx.data()[i];
      element.group = group.data()[i];
      element.split_condition =
          XgboostSplitCondition(lo, hi, is_missing.data()[i] != 0);
      element.zero_fraction = zero_fraction.data()[i];
      element.v = leaf.data()[i];
      paths.push_back(element);
    }
    std::vector<double> intercept_values(intercepts.data(),
                                         intercepts.data() + num_groups);

    // Everything below is pure C++/Metal work on the host copies made above: shader
    // compilation, preprocessing, and (for private storage) a blocking blit can take
    // seconds on large models and must not stall other Python threads.
    nb::gil_scoped_release release;
    explainer_ = std::make_unique<Explainer>(
        kernel_spec, kernel_is_metallib ? Explainer::LibraryKind::kMetallibFile
                                        : Explainer::LibraryKind::kSourceString);
    model_ = explainer_->Compile(paths, num_groups, num_features, intercept_values,
                                 ParseStorage(model_storage));
  }

  void Configure(uint32_t rows_per_simdgroup, uint32_t threads_per_threadgroup,
                 const std::string& accumulation, size_t deterministic_scratch_mib,
                 size_t atomic_tile_rows) {
    EnsureSameProcess();
    if (deterministic_scratch_mib == 0) {
      throw std::invalid_argument("deterministic_scratch_mib must be positive");
    }
    const AccumulationMode mode = ParseAccumulation(accumulation);
    const size_t scratch_bytes = CheckedMul(
        deterministic_scratch_mib, size_t{1024 * 1024}, "deterministic scratch");
    nb::gil_scoped_release release;  // the setters wait on the Explain mutex
    explainer_->set_rows_per_simdgroup(rows_per_simdgroup);
    explainer_->set_threads_per_threadgroup(threads_per_threadgroup);
    explainer_->set_accumulation_mode(mode);
    explainer_->set_deterministic_scratch_budget_bytes(scratch_bytes);
    explainer_->set_atomic_tile_rows(atomic_tile_rows);
  }

  Output Explain(Matrix matrix, size_t x_capacity_bytes) {
    EnsureSameProcess();
    if (matrix.shape(1) != model_->num_cols()) {
      throw std::invalid_argument("X feature count does not match compiled model");
    }
    const size_t rows = matrix.shape(0);
    const size_t length = CheckedMul(
        CheckedMul(rows, model_->num_groups(), "output"), model_->num_cols() + 1,
        "output");
    const size_t bytes = CheckedMul(length, sizeof(float), "output bytes");
    const size_t page = static_cast<size_t>(getpagesize());
    if (bytes > std::numeric_limits<size_t>::max() - page) {
      throw std::overflow_error("output size overflows");
    }
    // Page-aligned and page-padded so the host wraps this exact allocation with
    // bytesNoCopy: the GPU prefills and writes the caller-visible memory directly
    // and the copy-back disappears.
    const size_t padded = std::max(((bytes + page - 1) / page) * page, page);
    void* raw = nullptr;
    if (posix_memalign(&raw, page, padded) != 0) throw std::bad_alloc();
    float* data = static_cast<float*>(raw);
    nb::capsule owner(raw, [](void* pointer) noexcept { std::free(pointer); });
    Output output(data,
                  {rows, model_->num_groups(), model_->num_cols() + 1}, owner);
    ExplainTimings timings;
    {
      nb::gil_scoped_release release;
      timings = explainer_->Explain(*model_, matrix.data(), rows, data,
                                    x_capacity_bytes, padded);
    }
    last_timings_ = timings;  // GIL held again: concurrent explains serialize here
    has_timings_ = true;
    return output;
  }

  void TrimBuffers() {
    EnsureSameProcess();
    nb::gil_scoped_release release;
    explainer_->TrimPersistentBuffers();
  }

  nb::object LastTimings() const {
    if (!has_timings_) return nb::none();
    nb::dict out;
    out["upload_s"] = last_timings_.upload_s;
    out["encode_s"] = last_timings_.encode_s;
    out["gpu_s"] = last_timings_.gpu_s;
    out["total_s"] = last_timings_.total_s;
    out["x_zero_copy"] = last_timings_.x_zero_copy;
    out["output_zero_copy"] = last_timings_.output_zero_copy;
    out["dispatched"] = last_timings_.dispatched;
    out["deterministic_scratch_bytes"] = last_timings_.deterministic_scratch_bytes;
    out["deterministic_scratch_capacity_bytes"] =
        last_timings_.deterministic_scratch_capacity_bytes;
    out["deterministic_tile_rows"] = last_timings_.deterministic_tile_rows;
    out["deterministic_tiles"] = last_timings_.deterministic_tiles;
    out["atomic_tile_rows"] = last_timings_.atomic_tile_rows;
    out["atomic_tiles"] = last_timings_.atomic_tiles;
    return out;
  }

  size_t num_bins() const { return model_->num_bins(); }
  std::string storage_mode() const {
    return model_->storage_mode() == ModelStorageMode::kPrivate ? "private" : "shared";
  }

 private:
  // Metal state does not survive fork(); using an inherited explainer in the child
  // crashes deep inside the driver. Fail with a catchable error instead.
  void EnsureSameProcess() const {
    if (getpid() != pid_) {
      throw std::runtime_error(
          "MetalTreeExplainer cannot be used in a forked child process; construct a "
          "new explainer there (multiprocessing: use the 'spawn' start method)");
    }
  }

  std::unique_ptr<Explainer> explainer_;
  std::unique_ptr<CompiledModel> model_;
  pid_t pid_ = getpid();
  ExplainTimings last_timings_{};
  bool has_timings_ = false;
};

}  // namespace

NB_MODULE(_native, module) {
  module.doc() = "Native MetalTreeShap compiled-model bindings";
  nb::class_<NativeExplainer>(module, "NativeExplainer")
      .def(nb::init<Input1D<uint64_t>, Input1D<int64_t>, Input1D<int32_t>,
                    Input1D<float>, Input1D<float>, Input1D<uint8_t>,
                    Input1D<double>, Input1D<float>, size_t, size_t,
                    Input1D<double>, const std::string&, bool,
                    const std::string&>(),
           "path_idx"_a, "feature_idx"_a, "group"_a, "lower"_a, "upper"_a,
           "is_missing"_a, "zero_fraction"_a, "leaf"_a, "num_groups"_a,
           "num_features"_a, "intercepts"_a, "kernel_spec"_a,
           "kernel_is_metallib"_a, "model_storage"_a)
      .def("configure", &NativeExplainer::Configure,
           "rows_per_simdgroup"_a = 256,
           "threads_per_threadgroup"_a = 256,
           "accumulation"_a = "atomic",
           "deterministic_scratch_mib"_a = 256,
           "atomic_tile_rows"_a = 0)
      .def("explain", &NativeExplainer::Explain, "X"_a.noconvert(),
           "x_capacity_bytes"_a = 0)
      .def("trim_buffers", &NativeExplainer::TrimBuffers)
      .def_prop_ro("last_timings", &NativeExplainer::LastTimings)
      .def_prop_ro("num_bins", &NativeExplainer::num_bins)
      .def_prop_ro("storage_mode", &NativeExplainer::storage_mode);
}
