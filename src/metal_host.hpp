// metal-treeshap: metal-cpp host — compiled-model design (Phase 1).
//
// STATUS: externally exercised on an M4 Max (validation_v3): all six frozen fixtures ran
// through this CompiledModel/Explain logic (shader loaded from source at runtime there,
// since the offline Metal toolchain was absent) with max error <= 6.5e-6, across
// rows_per_simdgroup in {1, 7, 1024}, including empty-model, zero-row, intercept,
// repeated-call and invalid-tuning behavior. src/main_metal.cpp is the checked-in runner
// that makes that validation repository-reproducible (see tests/test_fixture.py).
//
// Design (review-driven):
//   * CompiledModel: all O(model) work once — validate, preprocess, pack, persistent
//     element/segment buffers, per-group (path bias + REQUIRED finite model intercept)
//     in fp64. Immutable after construction; safe to share across threads.
//   * Explainer: loads the kernel from a .metallib file OR compiles MSL source at
//     runtime (newLibraryWithSource) — the latter is the development path on Macs
//     without the offline Metal toolchain.
//   * Exception safety: constructors acquire every Metal object into local owning
//     guards and transfer to members only after full validation — a throwing
//     constructor never leaks earlier acquisitions (validation_v3 finding).
//   * Explain is serialized by an internal mutex (persistent buffers); the tuning
//     setter takes the same mutex, so tuning cannot race a running explanation. Use one
//     Explainer per thread for parallelism.
//   * All uint32 narrowings, byte-size products, and the 1-D dispatch width are checked
//     before any Metal call (grid coordinates are 32-bit in MSL; oversized workloads are
//     rejected with a "batch rows" error rather than silently wrapped — matching the
//     shader's 64-bit work-count arithmetic).
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

#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/metal_treeshap/paths.h"
#include "../include/metal_treeshap/preprocess.h"

namespace metal_treeshap {

struct KernelParams {  // must match struct Params in treeshap.metal
  uint32_t num_rows;
  uint32_t num_cols;
  uint32_t num_groups;
  uint32_t num_bins;
  uint32_t rows_per_simdgroup;
};

struct ExplainTimings {
  double upload_s = 0.0;   // X staging copy (0 when the zero-copy path is taken)
  double encode_s = 0.0;   // command encoding + commit
  double gpu_s = 0.0;      // GPUEndTime - GPUStartTime
  double total_s = 0.0;    // wall time of Explain()
  bool x_zero_copy = false;
  bool dispatched = false;  // false for zero-work (bias-only) fast paths
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
  CompiledModel(MTL::Device* device, const std::vector<PathElement>& raw_paths,
                size_t num_groups, size_t num_cols, const std::vector<double>& intercepts)
      : num_groups_(num_groups), num_cols_(num_cols) {
    if (!device) throw std::invalid_argument("null Metal device");
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
    bias_.resize(num_groups);
    for (size_t g = 0; g < num_groups; g++) bias_[g] = pp.bias[g] + intercepts[g];

    if (!pp.elements.empty()) {
      std::vector<GpuPathElement> packed = pp.PackForGpu();
      // Acquire both buffers into guards: if the second allocation throws/fails, the
      // first is released by its guard (the destructor won't run on a throwing ctor).
      detail_host::OwnGuard<MTL::Buffer> elem_g(device->newBuffer(
          packed.data(),
          detail_host::CheckedMul(packed.size(), sizeof(GpuPathElement), "elements"),
          MTL::ResourceStorageModeShared));
      detail_host::OwnGuard<MTL::Buffer> seg_g(device->newBuffer(
          pp.bin_segments.data(),
          detail_host::CheckedMul(pp.bin_segments.size(), sizeof(uint32_t), "segments"),
          MTL::ResourceStorageModeShared));
      if (!elem_g || !seg_g) throw std::runtime_error("model buffer allocation failed");
      elements_ = elem_g.Transfer();
      segments_ = seg_g.Transfer();
      // Phase 2 experiment: blit both into StorageModePrivate copies and benchmark.
    }
  }

  ~CompiledModel() {
    if (elements_) elements_->release();
    if (segments_) segments_->release();
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

 private:
  size_t num_groups_, num_cols_, num_bins_ = 0;
  std::vector<double> bias_;  // path bias + intercept, fp64, per group
  MTL::Buffer* elements_ = nullptr;
  MTL::Buffer* segments_ = nullptr;
};

class Explainer {
 public:
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
    {
      detail_host::ScopedPool pool;  // scope autoreleased strings/errors
      NS::Error* error = nullptr;
      detail_host::OwnGuard<MTL::Library> lib_g;
      if (kind == LibraryKind::kMetallibFile) {
        auto* lib_path = NS::String::string(spec.c_str(), NS::UTF8StringEncoding);
        lib_g = detail_host::OwnGuard<MTL::Library>(device_g->newLibrary(lib_path, &error));
        if (!lib_g) throw std::runtime_error("failed to load metallib: " + spec);
      } else {
        // Runtime compilation: the development path on Macs without the offline Metal
        // toolchain (validated externally in v3).
        auto* src = NS::String::string(spec.c_str(), NS::UTF8StringEncoding);
        detail_host::OwnGuard<MTL::CompileOptions> opts(MTL::CompileOptions::alloc()->init());
        lib_g = detail_host::OwnGuard<MTL::Library>(
            device_g->newLibrary(src, opts.get(), &error));
        if (!lib_g) {
          std::string msg = "failed to compile MSL source";
          if (error && error->localizedDescription()) {
            msg += ": ";
            msg += error->localizedDescription()->utf8String();
          }
          throw std::runtime_error(msg);
        }
      }
      auto* fn_name = NS::String::string("shap_first_order", NS::UTF8StringEncoding);
      detail_host::OwnGuard<MTL::Function> fn_g(lib_g->newFunction(fn_name));
      if (!fn_g) throw std::runtime_error("kernel 'shap_first_order' not found");
      pso_g = detail_host::OwnGuard<MTL::ComputePipelineState>(
          device_g->newComputePipelineState(fn_g.get(), &error));
      if (!pso_g) throw std::runtime_error("failed to create pipeline state");
    }
    if (pso_g->threadExecutionWidth() != 32) {  // the algorithm assumes 32-wide SIMD-groups
      throw std::runtime_error("unexpected SIMD width: " +
                               std::to_string(pso_g->threadExecutionWidth()));
    }
    if (pso_g->maxTotalThreadsPerThreadgroup() < kThreadsPerTg) {
      throw std::runtime_error("device cannot run 256-thread threadgroups for this kernel");
    }

    device_ = device_g.Transfer();  // nothing below can throw
    queue_ = queue_g.Transfer();
    pso_ = pso_g.Transfer();
  }

  // Owns raw Metal objects: neither copyable nor movable (hold via unique_ptr).
  Explainer(const Explainer&) = delete;
  Explainer& operator=(const Explainer&) = delete;
  Explainer(Explainer&&) = delete;
  Explainer& operator=(Explainer&&) = delete;

  std::unique_ptr<CompiledModel> Compile(const std::vector<PathElement>& raw_paths,
                                         size_t num_groups, size_t num_cols,
                                         const std::vector<double>& intercepts) {
    return std::make_unique<CompiledModel>(device_, raw_paths, num_groups, num_cols,
                                           intercepts);
  }

  // phis_out: num_rows * num_groups * (num_cols + 1) floats, fully written (bias +
  // intercept included). The result is copied once from the shared output buffer.
  // Serialized internally (persistent buffers); one Explainer per thread to parallelize.
  ExplainTimings Explain(const CompiledModel& model, const float* X, size_t num_rows,
                         float* phis_out) {
    std::lock_guard<std::mutex> lock(mu_);
    const uint32_t rows_per_sg = rows_per_simdgroup_;  // read under the same mutex
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

    // 1-D grid coordinates are 32-bit in MSL: reject dispatches whose thread count
    // would wrap rather than let host (64-bit) and shader work math disagree.
    const uint64_t banks = (num_rows + rows_per_sg - 1) / rows_per_sg;  // setter forbids 0
    const uint64_t simdgroups_needed = static_cast<uint64_t>(model.num_bins()) * banks;
    if (simdgroups_needed >
        (static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1) / 32) {
      throw std::invalid_argument("dispatch exceeds 32-bit grid coordinates — batch the "
                                  "rows across multiple Explain calls (Phase 3 adds "
                                  "automatic batching)");
    }

    detail_host::ScopedPool pool;  // scope all autoreleased command objects

    // ---- X buffer: zero-copy when eligible, else persistent staging copy ----
    const size_t x_bytes = detail_host::CheckedMul(
        detail_host::CheckedMul(num_rows, num_cols, "X"), sizeof(float), "X bytes");
    const auto tu0 = chr::steady_clock::now();
    MTL::Buffer* x_wrapped = nullptr;
    const size_t page = static_cast<size_t>(getpagesize());
    if ((reinterpret_cast<uintptr_t>(X) % page) == 0 && (x_bytes % page) == 0) {
      x_wrapped = device_->newBuffer(const_cast<float*>(X), x_bytes,
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

    // ---- Encode + dispatch ----
    const auto te0 = chr::steady_clock::now();
    KernelParams params{static_cast<uint32_t>(num_rows), static_cast<uint32_t>(num_cols),
                        static_cast<uint32_t>(num_groups),
                        static_cast<uint32_t>(model.num_bins()), rows_per_sg};
    const uint32_t simdgroups_per_tg = kThreadsPerTg / 32;
    const uint64_t tg_count = (simdgroups_needed + simdgroups_per_tg - 1) / simdgroups_per_tg;

    MTL::CommandBuffer* cmd = queue_->commandBuffer();
    if (!cmd) throw std::runtime_error("failed to create command buffer");
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
    if (!enc) throw std::runtime_error("failed to create compute encoder");
    enc->setComputePipelineState(pso_);
    enc->setBuffer(x_buf, 0, 0);
    enc->setBuffer(model.elements(), 0, 1);
    enc->setBuffer(model.segments(), 0, 2);
    enc->setBuffer(phis_buf_, 0, 3);
    enc->setBytes(&params, sizeof(params), 4);
    enc->dispatchThreadgroups(MTL::Size::Make(tg_count, 1, 1),
                              MTL::Size::Make(kThreadsPerTg, 1, 1));
    enc->endEncoding();
    cmd->commit();
    t.encode_s = chr::duration<double>(chr::steady_clock::now() - te0).count();

    cmd->waitUntilCompleted();
    if (cmd->status() == MTL::CommandBufferStatusError) {
      throw std::runtime_error("GPU execution failed");
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

  ~Explainer() {
    if (x_staging_) x_staging_->release();
    if (phis_buf_) phis_buf_->release();
    if (pso_) pso_->release();
    if (queue_) queue_->release();
    if (device_) device_->release();
  }

 private:
  static constexpr uint32_t kThreadsPerTg = 256;  // 8 SIMD-groups, as upstream's 256/32

  void EnsureCapacity(MTL::Buffer** buf, size_t bytes) {
    if (*buf && (*buf)->length() >= bytes) return;
    if (*buf) {
      (*buf)->release();
      *buf = nullptr;
    }
    *buf = device_->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (!*buf) throw std::runtime_error("buffer allocation failed");
  }

  MTL::Device* device_ = nullptr;
  MTL::CommandQueue* queue_ = nullptr;
  MTL::ComputePipelineState* pso_ = nullptr;
  MTL::Buffer* x_staging_ = nullptr;  // persistent, grown as needed
  MTL::Buffer* phis_buf_ = nullptr;   // persistent, grown as needed
  uint32_t rows_per_simdgroup_ = 1024;  // upstream kRowsPerWarp; tune in Phase 2
  std::mutex mu_;
};

}  // namespace metal_treeshap

#endif  // __APPLE__
