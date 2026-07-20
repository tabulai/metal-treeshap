// metal-treeshap: Metal CLI runner used by the repository-local differential CTests.
//
// Same CSV contract as reference_cli, so tests/test_fixture.py can run both engines
// over the frozen fixtures and diff them (set METAL_CLI=... or pass --metal-cli):
//
//   metal_cli <paths.csv> <X.csv> <num_groups> <out.csv> <intercepts>
//             [--kernel <treeshap.metallib | treeshap.metal>]
//             [--rows-per-simdgroup N]
//             [--threads-per-threadgroup 32|64|128|256]
//             [--atomic-tile-rows N]  # 0 means full dispatch
//             [--accumulation atomic|simdgroup|deterministic]
//             [--deterministic-scratch-mib N] [--model-storage shared|private]
//
// Kernel resolution when --kernel is omitted, in order: $METAL_TREESHAP_KERNEL,
// ./treeshap.metallib, ./shaders/treeshap.metal, ../shaders/treeshap.metal. A .metal
// path (or any non-.metallib file) is compiled AT RUNTIME with newLibraryWithSource —
// the development path on Macs without the offline Metal toolchain.
//
// macOS only. Build: see CMakeLists.txt (requires vendored third_party/metal-cpp).

#if !defined(__APPLE__)
#include <cstdio>
int main() {
  std::fprintf(stderr, "metal_cli requires macOS/Apple Silicon\n");
  return 2;
}
#else

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include "metal_host.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "csv_io.h"

using namespace metal_treeshap;

static bool FileExists(const std::string& p) { return std::ifstream(p).good(); }

static std::string ReadFile(const std::string& p) {
  std::ifstream in(p);
  if (!in) throw std::runtime_error("cannot open " + p);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

static bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(),
                                                suffix) == 0;
}

static std::string ResolveKernel(std::string kernel_arg) {
  if (!kernel_arg.empty()) return kernel_arg;
  if (const char* env = std::getenv("METAL_TREESHAP_KERNEL")) return env;
  for (const char* cand : {"treeshap.metallib", "shaders/treeshap.metal",
                           "../shaders/treeshap.metal"}) {
    if (FileExists(cand)) return cand;
  }
  throw std::runtime_error("no kernel found: pass --kernel or set METAL_TREESHAP_KERNEL");
}

int main(int argc, char** argv) {
  try {
    std::vector<std::string> pos;
    std::string kernel_arg;
    uint32_t rows_per_sg = 256;
    uint32_t threads_per_tg = 256;
    size_t atomic_tile_rows = 0;
    AccumulationMode accumulation = AccumulationMode::kAtomic;
    ModelStorageMode model_storage = ModelStorageMode::kShared;
    size_t deterministic_scratch_mib = 256;
    bool kernel_set = false, rows_per_set = false, threads_per_set = false;
    bool accumulation_set = false, storage_set = false, scratch_set = false;
    bool atomic_tile_set = false;
    for (int i = 1; i < argc; i++) {
      const std::string a = argv[i];
      if (a == "--kernel") {
        if (kernel_set) throw std::invalid_argument("--kernel specified more than once");
        if (i + 1 >= argc) throw std::invalid_argument("--kernel requires a path");
        kernel_arg = argv[++i];
        kernel_set = true;
      } else if (a == "--rows-per-simdgroup") {
        if (rows_per_set) {
          throw std::invalid_argument("--rows-per-simdgroup specified more than once");
        }
        if (i + 1 >= argc) {
          throw std::invalid_argument("--rows-per-simdgroup requires a value");
        }
        rows_per_sg = csv::ParseU32(argv[++i], "rows_per_simdgroup");
        if (rows_per_sg == 0) {
          throw std::invalid_argument("rows_per_simdgroup must be > 0");
        }
        rows_per_set = true;
      } else if (a == "--threads-per-threadgroup") {
        if (threads_per_set) {
          throw std::invalid_argument("--threads-per-threadgroup specified more than once");
        }
        if (i + 1 >= argc) {
          throw std::invalid_argument("--threads-per-threadgroup requires a value");
        }
        threads_per_tg = csv::ParseU32(argv[++i], "threads_per_threadgroup");
        if (threads_per_tg != 32 && threads_per_tg != 64 && threads_per_tg != 128 &&
            threads_per_tg != 256) {
          throw std::invalid_argument(
              "threads_per_threadgroup must be one of 32, 64, 128, 256");
        }
        threads_per_set = true;
      } else if (a == "--atomic-tile-rows") {
        if (atomic_tile_set) {
          throw std::invalid_argument("--atomic-tile-rows specified more than once");
        }
        if (i + 1 >= argc) {
          throw std::invalid_argument("--atomic-tile-rows requires a value");
        }
        atomic_tile_rows = csv::ParseSize(argv[++i], "atomic_tile_rows");
        if (atomic_tile_rows > std::numeric_limits<uint32_t>::max()) {
          throw std::invalid_argument("atomic_tile_rows does not fit uint32");
        }
        atomic_tile_set = true;
      } else if (a == "--accumulation") {
        if (accumulation_set) {
          throw std::invalid_argument("--accumulation specified more than once");
        }
        if (i + 1 >= argc) throw std::invalid_argument("--accumulation requires a value");
        const std::string value = argv[++i];
        if (value == "atomic") {
          accumulation = AccumulationMode::kAtomic;
        } else if (value == "simdgroup") {
          accumulation = AccumulationMode::kSimdgroup;
        } else if (value == "deterministic") {
          accumulation = AccumulationMode::kDeterministic;
        } else {
          throw std::invalid_argument(
              "--accumulation must be atomic, simdgroup, or deterministic");
        }
        accumulation_set = true;
      } else if (a == "--deterministic-scratch-mib") {
        if (scratch_set) {
          throw std::invalid_argument(
              "--deterministic-scratch-mib specified more than once");
        }
        if (i + 1 >= argc) {
          throw std::invalid_argument("--deterministic-scratch-mib requires a value");
        }
        deterministic_scratch_mib =
            csv::ParseSize(argv[++i], "deterministic_scratch_mib");
        if (deterministic_scratch_mib == 0) {
          throw std::invalid_argument("deterministic_scratch_mib must be > 0");
        }
        scratch_set = true;
      } else if (a == "--model-storage") {
        if (storage_set) {
          throw std::invalid_argument("--model-storage specified more than once");
        }
        if (i + 1 >= argc) throw std::invalid_argument("--model-storage requires a value");
        const std::string value = argv[++i];
        if (value == "shared") {
          model_storage = ModelStorageMode::kShared;
        } else if (value == "private") {
          model_storage = ModelStorageMode::kPrivate;
        } else {
          throw std::invalid_argument("--model-storage must be shared or private");
        }
        storage_set = true;
      } else if (a.rfind("--", 0) == 0) {
        throw std::invalid_argument("unknown option: " + a);
      } else {
        pos.push_back(a);
      }
    }
    if (pos.size() != 5) {
      std::cerr << "usage: " << argv[0]
                << " <paths.csv> <X.csv> <num_groups> <out.csv> <intercepts>"
                   " [--kernel <lib-or-source>] [--rows-per-simdgroup N]"
                   " [--threads-per-threadgroup 32|64|128|256]"
                   " [--atomic-tile-rows N]"
                   " [--accumulation atomic|simdgroup|deterministic]"
                   " [--deterministic-scratch-mib N]"
                   " [--model-storage shared|private]\n"
                   "  intercepts: comma-separated margin-space value per group; REQUIRED —"
                   " pass explicit zeros (e.g. \"0\") for an intercept-free model\n";
      return 2;
    }

    const auto raw_paths = csv::LoadPaths(pos[0]);
    size_t rows = 0, cols = 0;
    const auto data = csv::LoadMatrix(pos[1], &rows, &cols);
    if (rows == 0) throw std::invalid_argument("X.csv must contain at least one row");
    if (cols == 0) throw std::invalid_argument("X.csv must contain at least one column");
    const size_t num_groups = csv::ParseSize(pos[2], "num_groups");
    if (num_groups == 0) throw std::invalid_argument("num_groups must be > 0");
    const std::vector<double> intercepts = csv::ParseIntercepts(pos[4], num_groups);

    const std::string kernel = ResolveKernel(kernel_arg);
    const bool is_lib = EndsWith(kernel, ".metallib");
    Explainer explainer(is_lib ? kernel : ReadFile(kernel),
                        is_lib ? Explainer::LibraryKind::kMetallibFile
                               : Explainer::LibraryKind::kSourceString);
    explainer.set_rows_per_simdgroup(rows_per_sg);
    explainer.set_threads_per_threadgroup(threads_per_tg);
    explainer.set_atomic_tile_rows(atomic_tile_rows);
    explainer.set_accumulation_mode(accumulation);
    explainer.set_deterministic_scratch_budget_bytes(
        csv::CheckedMul(deterministic_scratch_mib, size_t{1024} * 1024,
                        "deterministic scratch"));

    auto model = explainer.Compile(raw_paths, num_groups, cols, intercepts, model_storage);
    const size_t phis_len = csv::CheckedMul(
        csv::CheckedMul(rows, num_groups, "output"), cols + 1, "output");
    std::vector<float> phis(phis_len, 0.0f);
    ExplainTimings tm = explainer.Explain(*model, data.data(), rows, phis.data());

    csv::WritePhis(pos[3], phis, rows, num_groups * (cols + 1));
    // Deterministic metadata builds lazily on first deterministic use; querying the
    // cell count here must not force that build (or its possible failure) onto a run
    // that never needed it — including root-only deterministic runs, whose Explain
    // fast path skips the build entirely. num_partials is eager and always exact,
    // so zero partials proves that the active-cell count is also zero; otherwise
    // the stable "unavailable" sentinel distinguishes an unbuilt plan from a
    // measured count.
    const size_t det_num_partials = model->deterministic_num_partials();
    const bool det_stats_built = model->deterministic_ready();
    const std::string det_active_cells =
        det_stats_built ? std::to_string(model->deterministic_num_active_cells())
                        : (det_num_partials == 0 ? "0" : "unavailable");
    std::fprintf(stderr,
                 "[metal_cli] kernel=%s (%s) rows=%zu cols=%zu groups=%zu bins=%zu "
                 "dispatched=%d zero_copy=%d output_zero_copy=%d upload=%.4fs "
                 "encode=%.4fs gpu=%.4fs "
                 "total=%.4fs accumulation=%s threads_per_tg=%u model_storage=%s "
                 "atomic_tile_rows_requested=%zu atomic_tile_rows=%zu atomic_tiles=%zu "
                 "atomic_writes_per_row=%zu simdgroup_writes_per_row=%zu "
                 "deterministic_partials_per_row=%zu deterministic_active_cells=%s "
                 "deterministic_scratch_mib=%zu deterministic_scratch_used=%zu "
                 "deterministic_scratch_capacity=%zu "
                 "deterministic_tile_rows=%zu deterministic_tiles=%zu\n",
                 kernel.c_str(), is_lib ? "metallib" : "runtime-compiled", rows, cols,
                 num_groups, model->num_bins(), int(tm.dispatched), int(tm.x_zero_copy),
                 int(tm.output_zero_copy),
                 tm.upload_s, tm.encode_s, tm.gpu_s, tm.total_s,
                 accumulation == AccumulationMode::kAtomic
                     ? "atomic"
                     : (accumulation == AccumulationMode::kSimdgroup ? "simdgroup"
                                                                     : "deterministic"),
                 threads_per_tg,
                 model_storage == ModelStorageMode::kShared ? "shared" : "private",
                 atomic_tile_rows, tm.atomic_tile_rows, tm.atomic_tiles,
                 model->atomic_writes_per_row(), model->simdgroup_writes_per_row(),
                 det_num_partials,
                 det_active_cells.c_str(), deterministic_scratch_mib,
                 tm.deterministic_scratch_bytes,
                 tm.deterministic_scratch_capacity_bytes,
                 tm.deterministic_tile_rows, tm.deterministic_tiles);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
#endif  // __APPLE__
