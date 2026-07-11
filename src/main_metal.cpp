// metal-treeshap: Metal CLI runner — the repository-reproducible version of the
// validation harness used externally in validation_v3.
//
// Same CSV contract as reference_cli, so tests/test_fixture.py can run both engines
// over the frozen fixtures and diff them (set METAL_CLI=... or pass --metal-cli):
//
//   metal_cli <paths.csv> <X.csv> <num_groups> <out.csv> [intercepts]
//             [--kernel <treeshap.metallib | treeshap.metal>]
//             [--rows-per-simdgroup N]
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
  std::vector<std::string> pos;
  std::string kernel_arg;
  uint32_t rows_per_sg = 0;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--kernel" && i + 1 < argc) {
      kernel_arg = argv[++i];
    } else if (a == "--rows-per-simdgroup" && i + 1 < argc) {
      rows_per_sg = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else {
      pos.push_back(a);
    }
  }
  if (pos.size() < 4 || pos.size() > 5) {
    std::cerr << "usage: " << argv[0]
              << " <paths.csv> <X.csv> <num_groups> <out.csv> [intercepts]"
                 " [--kernel <lib-or-source>] [--rows-per-simdgroup N]\n";
    return 2;
  }
  try {
    const auto raw_paths = csv::LoadPaths(pos[0]);
    size_t rows = 0, cols = 0;
    const auto data = csv::LoadMatrix(pos[1], &rows, &cols);
    const size_t num_groups = std::strtoull(pos[2].c_str(), nullptr, 10);
    const std::vector<double> intercepts =
        (pos.size() >= 5) ? csv::ParseIntercepts(pos[4], num_groups)
                          : std::vector<double>(num_groups, 0.0);

    const std::string kernel = ResolveKernel(kernel_arg);
    const bool is_lib = EndsWith(kernel, ".metallib");
    Explainer explainer(is_lib ? kernel : ReadFile(kernel),
                        is_lib ? Explainer::LibraryKind::kMetallibFile
                               : Explainer::LibraryKind::kSourceString);
    if (rows_per_sg != 0) explainer.set_rows_per_simdgroup(rows_per_sg);

    auto model = explainer.Compile(raw_paths, num_groups, cols, intercepts);
    std::vector<float> phis(rows * num_groups * (cols + 1), 0.0f);
    ExplainTimings tm = explainer.Explain(*model, data.data(), rows, phis.data());

    csv::WritePhis(pos[3], phis, rows, num_groups * (cols + 1));
    std::fprintf(stderr,
                 "[metal_cli] kernel=%s (%s) rows=%zu cols=%zu groups=%zu bins=%zu "
                 "dispatched=%d zero_copy=%d upload=%.4fs encode=%.4fs gpu=%.4fs "
                 "total=%.4fs\n",
                 kernel.c_str(), is_lib ? "metallib" : "runtime-compiled", rows, cols,
                 num_groups, model->num_bins(), int(tm.dispatched), int(tm.x_zero_copy),
                 tm.upload_s, tm.encode_s, tm.gpu_s, tm.total_s);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
#endif  // __APPLE__
