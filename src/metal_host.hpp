// metal-treeshap: metal-cpp host — compiled-model design and Phase-2 tuning controls.
//
// STATUS: locally built and exercised on an M4 Max. All seven frozen fixtures run through
// this CompiledModel/Explain logic with the shader compiled from source at runtime, with max
// error 6.505e-6 across atomic, SIMD-group, and deterministic accumulation, both
// model-storage modes, 32/64/128/256-thread threadgroups, and broad row-bank sweeps.
// Deterministic mode is additionally pinned by 100 bitwise-identical reruns and one-row
// versus single-tile equality. M4 Max tuning selected atomic/shared/256 rows/256 threads.
//
// Design (review-driven):
//   * CompiledModel: all O(model) work once — validate, preprocess, pack, build canonical
//     deterministic scatter/reduction metadata, optionally blit model buffers private,
//     and compute per-group (path bias + REQUIRED finite model intercept) in fp64.
//   * Explainer: loads the kernel from a .metallib file OR compiles MSL source at
//     runtime (newLibraryWithSource) — the latter is the development path on Macs
//     without the offline Metal toolchain.
//   * Exception safety: constructors acquire every Metal object into local owning
//     guards and transfer to members only after full validation — a throwing
//     constructor never leaks earlier acquisitions (validation_v3 finding).
//   * Explain is serialized by an internal mutex (persistent buffers); all tuning
//     setters take the same mutex, so tuning cannot race a running explanation. Use one
//     Explainer per thread for parallelism.
//   * All uint32 narrowings, byte-size products, and the 1-D dispatch width are checked
//     before any Metal call (grid coordinates are 32-bit in MSL; oversized workloads are
//     rejected with a "batch rows" error rather than silently wrapped — matching the
//     shader's 64-bit work-count arithmetic).
//   * No-throw encoding window: every check or allocation that can throw runs before the
//     compute encoder opens, and EndEncodingGuard ends the encoder during unwinding —
//     releasing an un-ended encoder is a Metal process abort, not a catchable error.
//   * Input zero-copy: newBuffer(bytes:length:options:) COPIES its input; only
//     bytesNoCopy wraps caller memory (page-aligned pointer AND page-multiple length) —
//     used opportunistically, else a persistent staging copy.
//   * Output: the result is memcpy'd once from the shared output buffer into the
//     caller's array, by API contract.
//
// metal-cpp setup: https://developer.apple.com/metal/cpp/ — vendor the headers and define
// NS_PRIVATE_IMPLEMENTATION / MTL_PRIVATE_IMPLEMENTATION / CA_PRIVATE_IMPLEMENTATION in
// exactly one translation unit.
#pragma once
#if defined(__APPLE__)

#include <Metal/Metal.hpp>
#include <unistd.h>  // getpagesize

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/metal_treeshap/deterministic.h"
#include "../include/metal_treeshap/paths.h"
#include "../include/metal_treeshap/preprocess.h"

namespace metal_treeshap {

struct KernelParams {  // must match struct Params in treeshap.metal
  uint32_t num_rows;
  uint32_t row_offset;
  uint32_t num_cols;
  uint32_t num_groups;
  uint32_t num_bins;
  uint32_t rows_per_simdgroup;
};

struct DeterministicKernelParams {  // must match DeterministicParams in treeshap.metal
  uint32_t num_rows;
  uint32_t row_offset;
  uint32_t num_cols;
  uint32_t num_groups;
  uint32_t num_bins;
  uint32_t rows_per_simdgroup;
  uint32_t num_partials;
  uint32_t num_active_cells;
  uint32_t num_chunks;
};

enum class AccumulationMode : uint32_t {
  kAtomic = 0,
  kSimdgroup = 1,
  kDeterministic = 2,
};

enum class ModelStorageMode {
  kShared,
  kPrivate,
};

struct ExplainTimings {
  double upload_s = 0.0;   // X staging copy (0 when the zero-copy path is taken)
  double encode_s = 0.0;   // command encoding + commit
  double gpu_s = 0.0;      // GPUEndTime - GPUStartTime
  double total_s = 0.0;    // wall time of Explain()
  bool x_zero_copy = false;
  bool dispatched = false;  // false for zero-work (bias-only) fast paths
  size_t deterministic_scratch_bytes = 0;
  size_t deterministic_scratch_capacity_bytes = 0;
  size_t deterministic_tile_rows = 0;
  size_t deterministic_tiles = 0;
  size_t atomic_tile_rows = 0;
  size_t atomic_tiles = 0;
};

namespace detail_host {

inline void CheckU32(size_t v, const char* what) {
  if (v > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument(std::string(what) + " does not fit uint32");
  }
}

inline size_t CheckedMul(size_t a, size_t b, const char* what) {
  if (b != 0 && a > std::numeric_limits<size_t>::max() / b) {
    throw std::invalid_argument(std::string(what) + " size overflows");
  }
  return a * b;
}

// Metal reports the actionable diagnosis (page fault, IOGPU restart, compile detail)
// through NSError; append it so failures are debuggable from the exception alone.
inline std::string AppendNsError(std::string message, NS::Error* error) {
  if (error && error->localizedDescription()) {
    message += ": ";
    message += error->localizedDescription()->utf8String();
  }
  return message;
}

// Scoped autorelease pool (metal-cpp command buffers/encoders are autoreleased).
class ScopedPool {
 public:
  ScopedPool() : pool_(NS::AutoreleasePool::alloc()->init()) {}
  ~ScopedPool() {
    if (pool_) pool_->release();
  }
  ScopedPool(const ScopedPool&) = delete;
  ScopedPool& operator=(const ScopedPool&) = delete;

 private:
  NS::AutoreleasePool* pool_;
};

// Ends a command encoder on scope exit unless End() already ran. Releasing an un-ended
// encoder — e.g. the autorelease pool draining while an exception unwinds — is a Metal
// process abort, not a catchable error, so encoding must be closed on every exit path.
class EndEncodingGuard {
 public:
  explicit EndEncodingGuard(MTL::CommandEncoder* enc) : enc_(enc) {}
  ~EndEncodingGuard() {
    if (enc_) enc_->endEncoding();
  }
  void End() {
    enc_->endEncoding();
    enc_ = nullptr;
  }
  EndEncodingGuard(const EndEncodingGuard&) = delete;
  EndEncodingGuard& operator=(const EndEncodingGuard&) = delete;

 private:
  MTL::CommandEncoder* enc_;
};

// Owns an NS/MTL object during construction; releases it unless Transfer()red. Makes
// throwing constructors leak-free (a throwing ctor never runs the class destructor, so
// members must not own anything until construction can no longer fail).
template <typename T>
class OwnGuard {
 public:
  explicit OwnGuard(T* p = nullptr) : p_(p) {}
  ~OwnGuard() {
    if (p_) p_->release();
  }
  OwnGuard(const OwnGuard&) = delete;
  OwnGuard& operator=(const OwnGuard&) = delete;
  OwnGuard(OwnGuard&& other) noexcept : p_(other.Transfer()) {}
  OwnGuard& operator=(OwnGuard&& other) noexcept {
    if (this != &other) {
      if (p_) p_->release();
      p_ = other.Transfer();
    }
    return *this;
  }
  T* get() const { return p_; }
  T* operator->() const { return p_; }
  explicit operator bool() const { return p_ != nullptr; }
  T* Transfer() {
    T* p = p_;
    p_ = nullptr;
    return p;
  }

 private:
  T* p_;
};

}  // namespace detail_host

// All O(model) state, built once and reused across Explain calls. Immutable after
// construction. `intercepts` is REQUIRED (margin-space, finite, one per output group,
// e.g. extract_paths.ExtractedModel.intercepts): omitting it silently truncates XGBoost
// contributions, so there is deliberately no default. Pass explicit zeros for a model
// that truly has no intercept.
class CompiledModel {
 public:
  CompiledModel(MTL::Device* device, MTL::CommandQueue* queue,
                const std::vector<PathElement>& raw_paths, size_t num_groups,
                size_t num_cols, const std::vector<double>& intercepts,
                ModelStorageMode storage_mode = ModelStorageMode::kShared)
      : num_groups_(num_groups), num_cols_(num_cols), storage_mode_(storage_mode) {
    if (!device) throw std::invalid_argument("null Metal device");
    if (storage_mode != ModelStorageMode::kShared &&
        storage_mode != ModelStorageMode::kPrivate) {
      throw std::invalid_argument("invalid model storage mode");
    }
    if (num_cols == 0) throw std::invalid_argument("num_cols must be > 0");
    if (num_groups == 0) throw std::invalid_argument("num_groups must be > 0");
    // Pre-check BEFORE computing num_cols + 1: at SIZE_MAX the +1 wraps to 0 first.
    if (num_cols >= std::numeric_limits<uint32_t>::max()) {
      throw std::invalid_argument("num_cols does not fit uint32");
    }
    detail_host::CheckU32(num_groups, "num_groups");
    if (intercepts.size() != num_groups) {
      throw std::invalid_argument("intercepts.size() must equal num_groups (pass explicit "
                                  "zeros for an intercept-free model)");
    }
    for (double v : intercepts) {
      if (!std::isfinite(v)) throw std::invalid_argument("intercepts must be finite");
    }

    Preprocessed pp = Preprocess(raw_paths, num_groups, num_cols);  // validates paths
    num_bins_ = pp.num_bins;
    detail_host::CheckU32(num_bins_, "num_bins");
    // Per-row output traffic implied by each strategy. The baseline writes once per
    // non-root element. SIMD-group mode writes once per distinct (group, feature) key in
    // each independently dispatched bin; the same key in different bins still needs a
    // separate atomic because SIMD-groups cannot communicate before the output update.
    for (const auto& e : pp.elements) {
      if (!e.IsRoot()) ++atomic_writes_per_row_;
    }
    for (size_t b = 0; b < pp.num_bins; ++b) {
      std::set<std::pair<int32_t, int64_t>> keys;
      for (size_t i = pp.bin_segments[b]; i < pp.bin_segments[b + 1]; ++i) {
        const auto& e = pp.elements[i];
        if (!e.IsRoot()) keys.emplace(e.group, e.feature_idx);
      }
      simdgroup_writes_per_row_ += keys.size();
    }
    DeterministicPlan deterministic_plan =
        BuildDeterministicPlan(pp, num_groups, num_cols);
    deterministic_num_partials_ = deterministic_plan.num_partials;
    deterministic_num_active_cells_ = deterministic_plan.active_cells.size();
    deterministic_num_chunks_ = deterministic_plan.chunks.size();
    detail_host::CheckU32(deterministic_num_chunks_, "deterministic chunks");
    deterministic_scratch_bytes_per_row_ = deterministic_plan.ScratchBytesPerRow();
    bias_.resize(num_groups);
    for (size_t g = 0; g < num_groups; g++) {
      const double combined = pp.bias[g] + intercepts[g];
      // The GPU output is float.  Reject an infinite fp64 sum and a finite value that
      // would become +/-inf during the bias prefill instead of silently poisoning every
      // attribution row for this group.
      if (!std::isfinite(combined) ||
          std::fabs(combined) > static_cast<double>(std::numeric_limits<float>::max())) {
        throw std::invalid_argument(
            "path bias + intercept must be finite and representable as float");
      }
      bias_[g] = combined;
    }

    if (!pp.elements.empty()) {
      std::vector<GpuPathElement> packed = pp.PackForGpu();
      const size_t elem_bytes =
          detail_host::CheckedMul(packed.size(), sizeof(GpuPathElement), "elements");
      const size_t seg_bytes = detail_host::CheckedMul(
          pp.bin_segments.size(), sizeof(uint32_t), "segments");
      const size_t slot_bytes = detail_host::CheckedMul(
          deterministic_plan.partial_slot_by_element.size(), sizeof(uint32_t),
          "deterministic slots");
      // The GPU consumes the CHUNK-space cell ranges (stage B) plus the chunk table
      // (stage A); the slot-space active_cells stay host-side statistics only.
      const size_t cell_bytes = detail_host::CheckedMul(
          deterministic_plan.chunk_cells.size(), sizeof(DeterministicReductionCell),
          "deterministic cells");
      const size_t chunk_bytes = detail_host::CheckedMul(
          deterministic_plan.chunks.size(), sizeof(DeterministicReductionChunk),
          "deterministic chunks");
      // Staging buffers also are the final buffers in shared mode. Keeping allocation in
      // guards makes every failure path leak-free.
      detail_host::OwnGuard<MTL::Buffer> elem_staging(device->newBuffer(
          packed.data(),
          elem_bytes,
          MTL::ResourceStorageModeShared));
      detail_host::OwnGuard<MTL::Buffer> seg_staging(device->newBuffer(
          pp.bin_segments.data(),
          seg_bytes,
          MTL::ResourceStorageModeShared));
      detail_host::OwnGuard<MTL::Buffer> slot_staging;
      detail_host::OwnGuard<MTL::Buffer> cell_staging;
      detail_host::OwnGuard<MTL::Buffer> chunk_staging;
      if (deterministic_num_partials_ != 0) {
        slot_staging = detail_host::OwnGuard<MTL::Buffer>(device->newBuffer(
            deterministic_plan.partial_slot_by_element.data(), slot_bytes,
            MTL::ResourceStorageModeShared));
        cell_staging = detail_host::OwnGuard<MTL::Buffer>(device->newBuffer(
            deterministic_plan.chunk_cells.data(), cell_bytes,
            MTL::ResourceStorageModeShared));
        chunk_staging = detail_host::OwnGuard<MTL::Buffer>(device->newBuffer(
            deterministic_plan.chunks.data(), chunk_bytes,
            MTL::ResourceStorageModeShared));
      }
      if (!elem_staging || !seg_staging) {
        throw std::runtime_error("model staging-buffer allocation failed");
      }
      if (deterministic_num_partials_ != 0 &&
          (!slot_staging || !cell_staging || !chunk_staging)) {
        throw std::runtime_error("deterministic model staging-buffer allocation failed");
      }

      if (storage_mode == ModelStorageMode::kShared) {
        elements_ = elem_staging.Transfer();
        segments_ = seg_staging.Transfer();
        if (deterministic_num_partials_ != 0) {
          deterministic_slots_ = slot_staging.Transfer();
          deterministic_cells_ = cell_staging.Transfer();
          deterministic_chunks_ = chunk_staging.Transfer();
        }
      } else {
        if (!queue) throw std::invalid_argument("null Metal command queue");
        detail_host::OwnGuard<MTL::Buffer> elem_private(
            device->newBuffer(elem_bytes, MTL::ResourceStorageModePrivate));
        detail_host::OwnGuard<MTL::Buffer> seg_private(
            device->newBuffer(seg_bytes, MTL::ResourceStorageModePrivate));
        detail_host::OwnGuard<MTL::Buffer> slot_private;
        detail_host::OwnGuard<MTL::Buffer> cell_private;
        detail_host::OwnGuard<MTL::Buffer> chunk_private;
        if (deterministic_num_partials_ != 0) {
          slot_private = detail_host::OwnGuard<MTL::Buffer>(
              device->newBuffer(slot_bytes, MTL::ResourceStorageModePrivate));
          cell_private = detail_host::OwnGuard<MTL::Buffer>(
              device->newBuffer(cell_bytes, MTL::ResourceStorageModePrivate));
          chunk_private = detail_host::OwnGuard<MTL::Buffer>(
              device->newBuffer(chunk_bytes, MTL::ResourceStorageModePrivate));
        }
        if (!elem_private || !seg_private) {
          throw std::runtime_error("private model-buffer allocation failed");
        }
        if (deterministic_num_partials_ != 0 &&
            (!slot_private || !cell_private || !chunk_private)) {
          throw std::runtime_error("private deterministic model-buffer allocation failed");
        }

        detail_host::ScopedPool pool;
        MTL::CommandBuffer* cmd = queue->commandBuffer();
        if (!cmd) throw std::runtime_error("failed to create model-upload command buffer");
        MTL::BlitCommandEncoder* blit = cmd->blitCommandEncoder();
        if (!blit) throw std::runtime_error("failed to create model-upload blit encoder");
        blit->copyFromBuffer(elem_staging.get(), 0, elem_private.get(), 0, elem_bytes);
        blit->copyFromBuffer(seg_staging.get(), 0, seg_private.get(), 0, seg_bytes);
        if (deterministic_num_partials_ != 0) {
          blit->copyFromBuffer(slot_staging.get(), 0, slot_private.get(), 0, slot_bytes);
          blit->copyFromBuffer(cell_staging.get(), 0, cell_private.get(), 0, cell_bytes);
          blit->copyFromBuffer(chunk_staging.get(), 0, chunk_private.get(), 0,
                               chunk_bytes);
        }
        blit->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
        if (cmd->status() == MTL::CommandBufferStatusError) {
          throw std::runtime_error(detail_host::AppendNsError(
              "private model-buffer upload failed", cmd->error()));
        }
        elements_ = elem_private.Transfer();
        segments_ = seg_private.Transfer();
        if (deterministic_num_partials_ != 0) {
          deterministic_slots_ = slot_private.Transfer();
          deterministic_cells_ = cell_private.Transfer();
          deterministic_chunks_ = chunk_private.Transfer();
        }
      }
    }
  }

  ~CompiledModel() {
    if (elements_) elements_->release();
    if (segments_) segments_->release();
    if (deterministic_slots_) deterministic_slots_->release();
    if (deterministic_cells_) deterministic_cells_->release();
    if (deterministic_chunks_) deterministic_chunks_->release();
  }
  CompiledModel(const CompiledModel&) = delete;
  CompiledModel& operator=(const CompiledModel&) = delete;
  CompiledModel(CompiledModel&&) = delete;
  CompiledModel& operator=(CompiledModel&&) = delete;

  size_t num_groups() const { return num_groups_; }
  size_t num_cols() const { return num_cols_; }
  size_t num_bins() const { return num_bins_; }
  bool empty() const { return elements_ == nullptr; }
  const std::vector<double>& bias() const { return bias_; }
  MTL::Buffer* elements() const { return elements_; }
  MTL::Buffer* segments() const { return segments_; }
  ModelStorageMode storage_mode() const { return storage_mode_; }
  size_t atomic_writes_per_row() const { return atomic_writes_per_row_; }
  size_t simdgroup_writes_per_row() const { return simdgroup_writes_per_row_; }
  size_t deterministic_num_partials() const { return deterministic_num_partials_; }
  size_t deterministic_num_active_cells() const {
    return deterministic_num_active_cells_;
  }
  size_t deterministic_num_chunks() const { return deterministic_num_chunks_; }
  // Partial slots plus one stage-A chunk sum per chunk (see DeterministicPlan).
  size_t deterministic_scratch_bytes_per_row() const {
    return deterministic_scratch_bytes_per_row_;
  }
  MTL::Buffer* deterministic_slots() const { return deterministic_slots_; }
  MTL::Buffer* deterministic_cells() const { return deterministic_cells_; }
  MTL::Buffer* deterministic_chunks() const { return deterministic_chunks_; }

 private:
  size_t num_groups_, num_cols_, num_bins_ = 0;
  std::vector<double> bias_;  // path bias + intercept, fp64, per group
  MTL::Buffer* elements_ = nullptr;
  MTL::Buffer* segments_ = nullptr;
  MTL::Buffer* deterministic_slots_ = nullptr;
  MTL::Buffer* deterministic_cells_ = nullptr;  // CHUNK-space [begin, end) per cell
  MTL::Buffer* deterministic_chunks_ = nullptr;
  ModelStorageMode storage_mode_ = ModelStorageMode::kShared;
  size_t atomic_writes_per_row_ = 0;
  size_t simdgroup_writes_per_row_ = 0;
  size_t deterministic_num_partials_ = 0;
  size_t deterministic_num_active_cells_ = 0;
  size_t deterministic_num_chunks_ = 0;
  size_t deterministic_scratch_bytes_per_row_ = 0;
};

class Explainer {
 public:
  static constexpr size_t kDefaultDeterministicScratchBudgetBytes =
      size_t{256} * 1024 * 1024;

  enum class LibraryKind {
    kMetallibFile,   // `spec` is a path to a compiled .metallib
    kSourceString,   // `spec` is MSL source (e.g. the contents of shaders/treeshap.metal)
  };

  explicit Explainer(const std::string& spec,
                     LibraryKind kind = LibraryKind::kMetallibFile) {
    // Acquire EVERYTHING into local guards; transfer to members only once nothing else
    // can throw (validation_v3: a throwing constructor leaked earlier acquisitions).
    detail_host::OwnGuard<MTL::Device> device_g(MTL::CreateSystemDefaultDevice());
    if (!device_g) throw std::runtime_error("no Metal device");
    detail_host::OwnGuard<MTL::CommandQueue> queue_g(device_g->newCommandQueue());
    if (!queue_g) throw std::runtime_error("failed to create command queue");

    detail_host::OwnGuard<MTL::ComputePipelineState> pso_g;
    detail_host::OwnGuard<MTL::ComputePipelineState> simdgroup_pso_g;
    detail_host::OwnGuard<MTL::ComputePipelineState> partial_pso_g;
    detail_host::OwnGuard<MTL::ComputePipelineState> chunk_reduce_pso_g;
    detail_host::OwnGuard<MTL::ComputePipelineState> cell_reduce_pso_g;
    {
      detail_host::ScopedPool pool;  // scope autoreleased strings/errors
      NS::Error* error = nullptr;
      detail_host::OwnGuard<MTL::Library> lib_g;
      detail_host::OwnGuard<MTL::Library> precise_lib_g;
      if (kind == LibraryKind::kMetallibFile) {
        auto* lib_path = NS::String::string(spec.c_str(), NS::UTF8StringEncoding);
        lib_g = detail_host::OwnGuard<MTL::Library>(device_g->newLibrary(lib_path, &error));
        if (!lib_g) {
          throw std::runtime_error(
              detail_host::AppendNsError("failed to load metallib: " + spec, error));
        }
        const size_t suffix = spec.rfind(".metallib");
        const std::string precise_path =
            suffix == std::string::npos
                ? spec + "_precise.metallib"
                : spec.substr(0, suffix) + "_precise.metallib";
        auto* precise_lib_path =
            NS::String::string(precise_path.c_str(), NS::UTF8StringEncoding);
        error = nullptr;
        precise_lib_g = detail_host::OwnGuard<MTL::Library>(
            device_g->newLibrary(precise_lib_path, &error));
        if (!precise_lib_g) {
          throw std::runtime_error(detail_host::AppendNsError(
              "failed to load precise reducer metallib: " + precise_path +
                  " (compile shaders/treeshap.metal with fast math disabled)",
              error));
        }
      } else {
        // Runtime compilation: the tested path on Macs without the offline Metal toolchain.
        auto* src = NS::String::string(spec.c_str(), NS::UTF8StringEncoding);
        detail_host::OwnGuard<MTL::CompileOptions> opts(MTL::CompileOptions::alloc()->init());
        if (!opts) throw std::runtime_error("failed to create Metal compile options");
        // Match the offline `metal -std=metal3.0` build exactly.  In particular this
        // makes float atomics an explicit requirement instead of relying on the SDK's
        // changing default language version.
        opts->setLanguageVersion(MTL::LanguageVersion3_0);
        lib_g = detail_host::OwnGuard<MTL::Library>(
            device_g->newLibrary(src, opts.get(), &error));
        if (!lib_g) {
          throw std::runtime_error(
              detail_host::AppendNsError("failed to compile MSL source", error));
        }

        // Compile a second copy only for the serial reducer. Metal fast math can
        // reassociate Kahan's correction to zero; keeping the precise pipeline separate
        // preserves the fast recurrence kernels and their Phase-2 performance.
        detail_host::OwnGuard<MTL::CompileOptions> precise_opts(
            MTL::CompileOptions::alloc()->init());
        if (!precise_opts) {
          throw std::runtime_error("failed to create precise Metal compile options");
        }
        precise_opts->setLanguageVersion(MTL::LanguageVersion3_0);
        precise_opts->setFastMathEnabled(false);
        error = nullptr;
        precise_lib_g = detail_host::OwnGuard<MTL::Library>(
            device_g->newLibrary(src, precise_opts.get(), &error));
        if (!precise_lib_g) {
          throw std::runtime_error(detail_host::AppendNsError(
              "failed to compile precise MSL reducer", error));
        }
      }
      auto make_pipeline = [&](MTL::Library* library, const char* name) {
        auto* fn_name = NS::String::string(name, NS::UTF8StringEncoding);
        detail_host::OwnGuard<MTL::Function> fn_g(library->newFunction(fn_name));
        if (!fn_g) throw std::runtime_error(std::string("kernel '") + name + "' not found");
        error = nullptr;
        detail_host::OwnGuard<MTL::ComputePipelineState> result(
            device_g->newComputePipelineState(fn_g.get(), &error));
        if (!result) {
          throw std::runtime_error(detail_host::AppendNsError(
              std::string("failed to create pipeline state for '") + name + "'", error));
        }
        return result;
      };
      pso_g = make_pipeline(lib_g.get(), "shap_first_order");
      simdgroup_pso_g = make_pipeline(lib_g.get(), "shap_first_order_simdgroup");
      partial_pso_g = make_pipeline(lib_g.get(), "shap_partials");
      chunk_reduce_pso_g =
          make_pipeline(precise_lib_g.get(), "reduce_partials_chunks");
      cell_reduce_pso_g =
          make_pipeline(precise_lib_g.get(), "reduce_chunks_serial");
    }
    if (pso_g->threadExecutionWidth() != 32 ||
        simdgroup_pso_g->threadExecutionWidth() != 32) {
      throw std::runtime_error("unexpected SIMD width: " +
                               std::to_string(pso_g->threadExecutionWidth()));
    }
    if (pso_g->maxTotalThreadsPerThreadgroup() < 256 ||
        simdgroup_pso_g->maxTotalThreadsPerThreadgroup() < 256) {
      throw std::runtime_error("device cannot run the default 256-thread threadgroup");
    }
    if (partial_pso_g->threadExecutionWidth() != 32) {
      throw std::runtime_error("unexpected deterministic SIMD width: " +
                               std::to_string(partial_pso_g->threadExecutionWidth()));
    }
    if (partial_pso_g->maxTotalThreadsPerThreadgroup() < 256 ||
        chunk_reduce_pso_g->maxTotalThreadsPerThreadgroup() < 256 ||
        cell_reduce_pso_g->maxTotalThreadsPerThreadgroup() < 256) {
      throw std::runtime_error(
          "device cannot run deterministic kernels with 256-thread threadgroups");
    }

    device_ = device_g.Transfer();  // nothing below can throw
    queue_ = queue_g.Transfer();
    pso_ = pso_g.Transfer();
    simdgroup_pso_ = simdgroup_pso_g.Transfer();
    partial_pso_ = partial_pso_g.Transfer();
    chunk_reduce_pso_ = chunk_reduce_pso_g.Transfer();
    cell_reduce_pso_ = cell_reduce_pso_g.Transfer();
  }

  // Owns raw Metal objects: neither copyable nor movable (hold via unique_ptr).
  Explainer(const Explainer&) = delete;
  Explainer& operator=(const Explainer&) = delete;
  Explainer(Explainer&&) = delete;
  Explainer& operator=(Explainer&&) = delete;

  std::unique_ptr<CompiledModel> Compile(const std::vector<PathElement>& raw_paths,
                                         size_t num_groups, size_t num_cols,
                                         const std::vector<double>& intercepts,
                                         ModelStorageMode storage_mode =
                                             ModelStorageMode::kShared) {
    return std::make_unique<CompiledModel>(device_, queue_, raw_paths, num_groups, num_cols,
                                           intercepts, storage_mode);
  }

  // phis_out: num_rows * num_groups * (num_cols + 1) floats, fully written (bias +
  // intercept included). The result is copied once from the shared output buffer.
  // Serialized internally (persistent buffers); one Explainer per thread to parallelize.
  // x_capacity_bytes (optional): the caller's actual allocation size behind X when it
  // extends past the logical num_rows*num_cols floats. bytesNoCopy needs a page-multiple
  // length, so a page-padded allocation lets arbitrary shapes take the zero-copy path.
  ExplainTimings Explain(const CompiledModel& model, const float* X, size_t num_rows,
                         float* phis_out, size_t x_capacity_bytes = 0) {
    std::lock_guard<std::mutex> lock(mu_);
    const uint32_t rows_per_sg = rows_per_simdgroup_;  // read under the same mutex
    const uint32_t threads_per_tg = threads_per_threadgroup_;
    const AccumulationMode accumulation = accumulation_mode_;
    const size_t atomic_tile_rows_requested = atomic_tile_rows_;
    const size_t deterministic_scratch_budget = deterministic_scratch_budget_bytes_;
    namespace chr = std::chrono;
    const auto t0 = chr::steady_clock::now();
    ExplainTimings t;

    const size_t num_cols = model.num_cols(), num_groups = model.num_groups();
    detail_host::CheckU32(num_rows, "num_rows");
    const size_t phis_len = detail_host::CheckedMul(
        detail_host::CheckedMul(num_rows, num_groups, "phis"), num_cols + 1, "phis");
    const size_t phis_bytes = detail_host::CheckedMul(phis_len, sizeof(float), "phis bytes");
    if (num_rows == 0) {
      t.total_s = chr::duration<double>(chr::steady_clock::now() - t0).count();
      return t;  // nothing to write
    }
    if (!X || !phis_out) throw std::invalid_argument("null input/output pointer");

    // Bias + intercept prefill is required in every mode.
    auto fill_bias = [&](float* phis) {
      std::memset(phis, 0, phis_bytes);
      for (size_t row = 0; row < num_rows; row++) {
        for (size_t gr = 0; gr < num_groups; gr++) {
          phis[IndexPhi(row, num_groups, gr, num_cols, num_cols)] =
              static_cast<float>(model.bias()[gr]);
        }
      }
    };

    if (model.empty()) {  // zero-work model (no paths): bias-only, no Metal calls
      fill_bias(phis_out);
      t.total_s = chr::duration<double>(chr::steady_clock::now() - t0).count();
      return t;
    }
    if (accumulation == AccumulationMode::kDeterministic &&
        model.deterministic_num_partials() == 0) {
      fill_bias(phis_out);  // root-only models have bias but no feature contributions
      t.total_s = chr::duration<double>(chr::steady_clock::now() - t0).count();
      return t;
    }

    detail_host::ScopedPool pool;  // scope all autoreleased command objects

    // ---- X buffer: zero-copy when eligible, else persistent staging copy ----
    const size_t x_bytes = detail_host::CheckedMul(
        detail_host::CheckedMul(num_rows, num_cols, "X"), sizeof(float), "X bytes");
    const auto tu0 = chr::steady_clock::now();
    MTL::Buffer* x_wrapped = nullptr;
    const size_t page = static_cast<size_t>(getpagesize());
    size_t wrap_bytes = 0;
    if ((reinterpret_cast<uintptr_t>(X) % page) == 0) {
      if (x_bytes % page == 0) {
        wrap_bytes = x_bytes;
      } else if (x_capacity_bytes >= x_bytes) {
        // A page-padded caller allocation admits a page-multiple wrap length that stays
        // inside the allocation; the kernels only ever read the leading x_bytes.
        const size_t padded = ((x_bytes + page - 1) / page) * page;
        if (padded <= x_capacity_bytes) wrap_bytes = padded;
      }
    }
    if (wrap_bytes != 0) {
      x_wrapped = device_->newBuffer(const_cast<float*>(X), wrap_bytes,
                                     MTL::ResourceStorageModeShared,
                                     nullptr /*no deallocator*/);
      t.x_zero_copy = (x_wrapped != nullptr);
    }
    detail_host::OwnGuard<MTL::Buffer> x_guard(x_wrapped);  // released on all exits
    MTL::Buffer* x_buf = x_wrapped;
    if (!x_buf) {
      EnsureCapacity(&x_staging_, x_bytes);
      std::memcpy(x_staging_->contents(), X, x_bytes);
      x_buf = x_staging_;
    }
    t.upload_s = chr::duration<double>(chr::steady_clock::now() - tu0).count();

    // ---- Output buffer: persistent, grown as needed; init = bias + intercept ----
    EnsureCapacity(&phis_buf_, phis_bytes);
    fill_bias(static_cast<float*>(phis_buf_->contents()));

    // ---- Dispatch geometry: every throwing check runs BEFORE the encoder opens ----
    // A throw while a command encoder is open must not unwind: the autorelease pool
    // would release the un-ended encoder and Metal aborts the whole process instead of
    // surfacing the exception. Tiling math, 32-bit dispatch validation, and scratch
    // allocation therefore all happen here; EndEncodingGuard below keeps any future
    // encoding-window throw fail-safe.
    constexpr uint64_t kMaxGridThreads =
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1;
    const uint64_t simdgroups_per_tg = threads_per_tg / 32;
    auto threadgroup_count = [&](size_t rows) {
      const uint64_t banks = (rows + rows_per_sg - 1) / rows_per_sg;
      const uint64_t simdgroups = static_cast<uint64_t>(model.num_bins()) * banks;
      const uint64_t groups =
          (simdgroups + simdgroups_per_tg - 1) / simdgroups_per_tg;
      // Include padded SIMD-groups in the last threadgroup: their grid positions are
      // visible to the shader before its early return and must not wrap uint32.
      if (groups > kMaxGridThreads / threads_per_tg) {
        throw std::invalid_argument("dispatch exceeds 32-bit grid coordinates");
      }
      return groups;
    };

    const bool deterministic = accumulation == AccumulationMode::kDeterministic;
    size_t tile_rows = num_rows;
    size_t det_partials = 0, det_active_cells = 0, det_chunks = 0;
    if (!deterministic) {
      const bool tile_atomic = accumulation == AccumulationMode::kAtomic;
      if (tile_atomic && atomic_tile_rows_requested != 0) {
        tile_rows = std::min(atomic_tile_rows_requested, num_rows);
      }
      if (tile_atomic) {
        t.atomic_tile_rows = tile_rows;
        t.atomic_tiles = (num_rows + tile_rows - 1) / tile_rows;
      }
    } else {
      det_partials = model.deterministic_num_partials();
      det_active_cells = model.deterministic_num_active_cells();
      det_chunks = model.deterministic_num_chunks();
      if (det_partials == 0 || det_active_cells == 0 || det_chunks == 0 ||
          !model.deterministic_slots() || !model.deterministic_cells() ||
          !model.deterministic_chunks()) {
        throw std::runtime_error("compiled model is missing deterministic metadata");
      }
      // bytes_per_row covers the partial slots plus the stage-A chunk sums; the chunk
      // count is the widest per-row reduction dispatch and therefore the 32-bit cap.
      const size_t bytes_per_row = model.deterministic_scratch_bytes_per_row();
      tile_rows = DeterministicTileRows(
          num_rows, bytes_per_row, det_chunks, model.num_bins(), rows_per_sg,
          threads_per_tg, deterministic_scratch_budget,
          static_cast<size_t>(device_->maxBufferLength()));
      const size_t partials_bytes = detail_host::CheckedMul(
          detail_host::CheckedMul(tile_rows, det_partials, "deterministic partials"),
          sizeof(float), "deterministic partials bytes");
      const size_t chunk_sum_bytes = detail_host::CheckedMul(
          detail_host::CheckedMul(tile_rows, det_chunks, "deterministic chunk sums"),
          sizeof(float), "deterministic chunk-sum bytes");
      EnsureCapacity(&deterministic_partials_buf_, partials_bytes,
                     MTL::ResourceStorageModePrivate);
      EnsureCapacity(&deterministic_chunk_sums_buf_, chunk_sum_bytes,
                     MTL::ResourceStorageModePrivate);
      t.deterministic_scratch_bytes = partials_bytes + chunk_sum_bytes;
      t.deterministic_scratch_capacity_bytes =
          deterministic_partials_buf_->length() +
          deterministic_chunk_sums_buf_->length();
      t.deterministic_tile_rows = tile_rows;
      t.deterministic_tiles = (num_rows + tile_rows - 1) / tile_rows;
    }
    // The first tile is the largest and the dispatch bound is monotone in rows, so this
    // one check rejects oversized workloads for every tile (deterministic tiles are
    // additionally pre-bounded by DeterministicTileRows' own 32-bit limits).
    (void)threadgroup_count(tile_rows);

    // ---- Encode + dispatch (no throw sites while the encoder is open) ----
    const auto te0 = chr::steady_clock::now();
    MTL::CommandBuffer* cmd = queue_->commandBuffer();
    if (!cmd) throw std::runtime_error("failed to create command buffer");
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
    if (!enc) throw std::runtime_error("failed to create compute encoder");
    detail_host::EndEncodingGuard encoding(enc);

    if (!deterministic) {
      enc->setComputePipelineState(accumulation == AccumulationMode::kSimdgroup
                                       ? simdgroup_pso_
                                       : pso_);
      enc->setBuffer(x_buf, 0, 0);
      enc->setBuffer(model.elements(), 0, 1);
      enc->setBuffer(model.segments(), 0, 2);
      enc->setBuffer(phis_buf_, 0, 3);
      for (size_t row_offset = 0; row_offset < num_rows; row_offset += tile_rows) {
        const size_t rows = std::min(tile_rows, num_rows - row_offset);
        KernelParams params{static_cast<uint32_t>(rows),
                            static_cast<uint32_t>(row_offset),
                            static_cast<uint32_t>(num_cols),
                            static_cast<uint32_t>(num_groups),
                            static_cast<uint32_t>(model.num_bins()), rows_per_sg};
        enc->setBytes(&params, sizeof(params), 4);
        enc->dispatchThreadgroups(MTL::Size::Make(threadgroup_count(rows), 1, 1),
                                  MTL::Size::Make(threads_per_tg, 1, 1));
      }
    } else {
      size_t row_offset = 0;
      while (row_offset < num_rows) {
        const size_t rows = std::min(tile_rows, num_rows - row_offset);
        DeterministicKernelParams params{
            static_cast<uint32_t>(rows), static_cast<uint32_t>(row_offset),
            static_cast<uint32_t>(num_cols), static_cast<uint32_t>(num_groups),
            static_cast<uint32_t>(model.num_bins()), rows_per_sg,
            static_cast<uint32_t>(det_partials),
            static_cast<uint32_t>(det_active_cells),
            static_cast<uint32_t>(det_chunks)};

        enc->setComputePipelineState(partial_pso_);
        enc->setBuffer(x_buf, 0, 0);
        enc->setBuffer(model.elements(), 0, 1);
        enc->setBuffer(model.segments(), 0, 2);
        enc->setBuffer(model.deterministic_slots(), 0, 3);
        enc->setBuffer(deterministic_partials_buf_, 0, 4);
        enc->setBytes(&params, sizeof(params), 5);
        enc->dispatchThreadgroups(MTL::Size::Make(threadgroup_count(rows), 1, 1),
                                  MTL::Size::Make(threads_per_tg, 1, 1));
        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        // Stage A: one thread per (row, chunk) — partials in, chunk sums out.
        // Cannot wrap: DeterministicTileRows capped tile_rows at UINT32_MAX / chunks.
        enc->setComputePipelineState(chunk_reduce_pso_);
        enc->setBuffer(deterministic_partials_buf_, 0, 0);
        enc->setBuffer(model.deterministic_chunks(), 0, 1);
        enc->setBuffer(deterministic_chunk_sums_buf_, 0, 2);
        enc->setBytes(&params, sizeof(params), 3);
        enc->dispatchThreads(MTL::Size::Make(rows * det_chunks, 1, 1),
                             MTL::Size::Make(threads_per_tg, 1, 1));
        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        // Stage B: one thread per (row, cell) — fixed-order combine of chunk sums.
        enc->setComputePipelineState(cell_reduce_pso_);
        enc->setBuffer(deterministic_chunk_sums_buf_, 0, 0);
        enc->setBuffer(model.deterministic_cells(), 0, 1);
        enc->setBuffer(phis_buf_, 0, 2);
        enc->setBytes(&params, sizeof(params), 3);
        enc->dispatchThreads(MTL::Size::Make(rows * det_active_cells, 1, 1),
                             MTL::Size::Make(threads_per_tg, 1, 1));
        row_offset += rows;
        if (row_offset < num_rows) {
          enc->memoryBarrier(MTL::BarrierScopeBuffers);  // scratch reuse by next tile
        }
      }
    }
    encoding.End();
    cmd->commit();
    t.encode_s = chr::duration<double>(chr::steady_clock::now() - te0).count();

    cmd->waitUntilCompleted();
    if (cmd->status() == MTL::CommandBufferStatusError) {
      throw std::runtime_error(
          detail_host::AppendNsError("GPU execution failed", cmd->error()));
    }
    t.gpu_s = cmd->GPUEndTime() - cmd->GPUStartTime();
    t.dispatched = true;

    std::memcpy(phis_out, phis_buf_->contents(), phis_bytes);

    t.total_s = chr::duration<double>(chr::steady_clock::now() - t0).count();
    return t;
  }

  void set_rows_per_simdgroup(uint32_t v) {
    if (v == 0) throw std::invalid_argument("rows_per_simdgroup must be > 0");
    std::lock_guard<std::mutex> lock(mu_);  // cannot race a running Explain
    rows_per_simdgroup_ = v;
  }

  void set_threads_per_threadgroup(uint32_t v) {
    if (v != 32 && v != 64 && v != 128 && v != 256) {
      throw std::invalid_argument(
          "threads_per_threadgroup must be one of 32, 64, 128, 256");
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (v > pso_->maxTotalThreadsPerThreadgroup()) {
      throw std::invalid_argument("threads_per_threadgroup exceeds pipeline maximum");
    }
    threads_per_threadgroup_ = v;
  }

  void set_accumulation_mode(AccumulationMode mode) {
    if (mode != AccumulationMode::kAtomic && mode != AccumulationMode::kSimdgroup &&
        mode != AccumulationMode::kDeterministic) {
      throw std::invalid_argument("invalid accumulation mode");
    }
    std::lock_guard<std::mutex> lock(mu_);
    accumulation_mode_ = mode;
  }

  // Zero retains the original single-dispatch behavior. A positive value splits atomic
  // accumulation into disjoint row dispatches in one command buffer. SIMD-group and
  // deterministic modes keep their independent dispatch policies.
  void set_atomic_tile_rows(size_t rows) {
    detail_host::CheckU32(rows, "atomic_tile_rows");
    std::lock_guard<std::mutex> lock(mu_);
    atomic_tile_rows_ = rows;
  }

  size_t atomic_tile_rows() const {
    std::lock_guard<std::mutex> lock(mu_);
    return atomic_tile_rows_;
  }

  void set_deterministic_scratch_budget_bytes(size_t bytes) {
    if (bytes == 0) {
      throw std::invalid_argument("deterministic scratch budget must be > 0 bytes");
    }
    std::lock_guard<std::mutex> lock(mu_);
    // The budget is a strict retained-buffer cap, not merely an active-range cap. The
    // partials and stage-A chunk-sum buffers count against it together.
    const size_t retained =
        (deterministic_partials_buf_ ? deterministic_partials_buf_->length() : 0) +
        (deterministic_chunk_sums_buf_ ? deterministic_chunk_sums_buf_->length() : 0);
    if (retained > bytes) {
      if (deterministic_partials_buf_) {
        deterministic_partials_buf_->release();
        deterministic_partials_buf_ = nullptr;
      }
      if (deterministic_chunk_sums_buf_) {
        deterministic_chunk_sums_buf_->release();
        deterministic_chunk_sums_buf_ = nullptr;
      }
    }
    deterministic_scratch_budget_bytes_ = bytes;
  }

  size_t deterministic_scratch_budget_bytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return deterministic_scratch_budget_bytes_;
  }

  size_t deterministic_scratch_capacity_bytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return (deterministic_partials_buf_ ? deterministic_partials_buf_->length() : 0) +
           (deterministic_chunk_sums_buf_ ? deterministic_chunk_sums_buf_->length()
                                          : 0);
  }

  // Release the persistent staging/output/scratch buffers retained across Explain
  // calls. They regrow on demand, so this only matters after an unusually large batch
  // whose peak allocation a long-lived explainer should not keep resident.
  void TrimPersistentBuffers() {
    std::lock_guard<std::mutex> lock(mu_);
    auto drop = [](MTL::Buffer** buffer) {
      if (*buffer) {
        (*buffer)->release();
        *buffer = nullptr;
      }
    };
    drop(&x_staging_);
    drop(&phis_buf_);
    drop(&deterministic_partials_buf_);
    drop(&deterministic_chunk_sums_buf_);
  }

  ~Explainer() {
    if (x_staging_) x_staging_->release();
    if (phis_buf_) phis_buf_->release();
    if (deterministic_partials_buf_) deterministic_partials_buf_->release();
    if (deterministic_chunk_sums_buf_) deterministic_chunk_sums_buf_->release();
    if (cell_reduce_pso_) cell_reduce_pso_->release();
    if (chunk_reduce_pso_) chunk_reduce_pso_->release();
    if (partial_pso_) partial_pso_->release();
    if (simdgroup_pso_) simdgroup_pso_->release();
    if (pso_) pso_->release();
    if (queue_) queue_->release();
    if (device_) device_->release();
  }

 private:
  void EnsureCapacity(MTL::Buffer** buf, size_t bytes,
                      MTL::ResourceOptions options = MTL::ResourceStorageModeShared) {
    if (*buf && (*buf)->length() >= bytes) return;
    // Allocate first so a failed growth leaves the previous reusable buffer intact.
    detail_host::OwnGuard<MTL::Buffer> replacement(
        device_->newBuffer(bytes, options));
    if (!replacement) throw std::runtime_error("buffer allocation failed");
    if (*buf) (*buf)->release();
    *buf = replacement.Transfer();
  }

  MTL::Device* device_ = nullptr;
  MTL::CommandQueue* queue_ = nullptr;
  MTL::ComputePipelineState* pso_ = nullptr;
  MTL::ComputePipelineState* simdgroup_pso_ = nullptr;
  MTL::ComputePipelineState* partial_pso_ = nullptr;
  MTL::ComputePipelineState* chunk_reduce_pso_ = nullptr;
  MTL::ComputePipelineState* cell_reduce_pso_ = nullptr;
  MTL::Buffer* x_staging_ = nullptr;  // persistent, grown as needed
  MTL::Buffer* phis_buf_ = nullptr;   // persistent, grown as needed
  MTL::Buffer* deterministic_partials_buf_ = nullptr;
  MTL::Buffer* deterministic_chunk_sums_buf_ = nullptr;
  // M4 Max Phase-2 tuning selected 256 for large ensembles; callers can retune.
  uint32_t rows_per_simdgroup_ = 256;
  uint32_t threads_per_threadgroup_ = 256;
  AccumulationMode accumulation_mode_ = AccumulationMode::kAtomic;
  size_t atomic_tile_rows_ = 0;  // 0 = full batch in one dispatch
  size_t deterministic_scratch_budget_bytes_ =
      kDefaultDeterministicScratchBudgetBytes;
  mutable std::mutex mu_;
};

}  // namespace metal_treeshap

#endif  // __APPLE__
