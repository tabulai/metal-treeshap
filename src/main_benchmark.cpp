// Persistent native benchmark runner for Phase 2 performance experiments.
//
// Loads paths and X once, creates the Metal pipeline once, compiles the model once,
// warms up, and then times repeated Explain calls on the same Explainer/CompiledModel.
// JSON is the only stdout output so orchestration scripts can consume it reliably.

#if !defined(__APPLE__)
#include <cstdio>
int main() {
  std::fprintf(stderr, "phase2_benchmark requires macOS/Apple Silicon\n");
  return 2;
}
#else

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include "metal_host.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "csv_io.h"

using namespace metal_treeshap;

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::string paths;
  std::string matrix;
  size_t num_groups = 0;
  std::vector<double> intercepts;
  std::string kernel;
  std::string expected;
  std::string output_json = "-";
  size_t warmups = 3;
  size_t iterations = 15;
  size_t row_limit = 0;
  uint32_t rows_per_simdgroup = 256;
  uint32_t threads_per_threadgroup = 256;
  size_t atomic_tile_rows = 0;
  std::string accumulation = "atomic";
  std::string storage = "shared";
  size_t deterministic_scratch_mib = 256;
  double relative_floor = 1e-6;
  double max_abs_error = std::numeric_limits<double>::infinity();
};

double Seconds(Clock::time_point begin, Clock::time_point end = Clock::now()) {
  return std::chrono::duration<double>(end - begin).count();
}

bool FileExists(const std::string& path) { return std::ifstream(path).good(); }

bool EndsWith(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string ReadFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::stringstream buffer;
  buffer << in.rdbuf();
  if (!in.good() && !in.eof()) throw std::runtime_error("failed reading " + path);
  return buffer.str();
}

std::string ResolveKernel(const std::string& requested) {
  if (!requested.empty()) return requested;
  if (const char* env = std::getenv("METAL_TREESHAP_KERNEL")) return env;
  for (const char* candidate : {"treeshap.metallib", "shaders/treeshap.metal",
                                "../shaders/treeshap.metal"}) {
    if (FileExists(candidate)) return candidate;
  }
  throw std::runtime_error("no kernel found: pass --kernel or set METAL_TREESHAP_KERNEL");
}

std::string JsonEscape(const std::string& value) {
  std::ostringstream out;
  for (unsigned char c : value) {
    switch (c) {
      case '\"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<unsigned>(c) << std::dec;
        } else {
          out << static_cast<char>(c);
        }
    }
  }
  return out.str();
}

std::string JsonString(const std::string& value) { return "\"" + JsonEscape(value) + "\""; }

size_t ParsePositiveSize(const std::string& value, const char* what) {
  const size_t parsed = csv::ParseSize(value, what);
  if (parsed == 0) throw std::invalid_argument(std::string(what) + " must be > 0");
  return parsed;
}

void Usage(const char* program) {
  std::cerr
      << "usage: " << program << " <paths.csv> <X.csv> <num_groups> [options]\n"
      << "  --intercepts v[,v...]       margin-space intercepts (default zeros)\n"
      << "  --kernel path               .metallib or .metal source\n"
      << "  --expected path             expected attribution CSV for error metrics\n"
      << "  --output-json path|-        destination (default stdout)\n"
      << "  --warmup N                  unmeasured calls (default 3)\n"
      << "  --iterations N              measured calls (default 15)\n"
      << "  --row-limit N               benchmark first N rows (0 means all)\n"
      << "  --rows-per-simdgroup N      row-bank size (default 256)\n"
      << "  --threads-per-threadgroup N 32,64,128,256 (default 256)\n"
      << "  --atomic-tile-rows N        rows/atomic dispatch; 0 means full (default 0)\n"
      << "  --accumulation MODE         atomic|simdgroup|deterministic (default atomic)\n"
      << "  --model-storage MODE        shared|private (default shared)\n"
      << "  --deterministic-scratch-mib N  scratch budget (default 256)\n"
      << "  --relative-floor X          denominator floor (default 1e-6)\n"
      << "  --max-abs-error X           gate element and row/group-sum errors\n";
}

Options ParseArgs(int argc, char** argv) {
  if (argc < 4) {
    Usage(argv[0]);
    throw std::invalid_argument("missing required arguments");
  }
  Options options;
  options.paths = argv[1];
  options.matrix = argv[2];
  options.num_groups = ParsePositiveSize(argv[3], "num_groups");
  std::string intercept_arg;
  bool intercept_set = false;
  auto need_value = [&](int& i, const std::string& name) -> std::string {
    if (i + 1 >= argc) throw std::invalid_argument(name + " requires a value");
    return argv[++i];
  };
  for (int i = 4; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--intercepts") {
      if (intercept_set) throw std::invalid_argument("--intercepts specified twice");
      intercept_arg = need_value(i, arg);
      intercept_set = true;
    } else if (arg == "--kernel") {
      options.kernel = need_value(i, arg);
    } else if (arg == "--expected") {
      options.expected = need_value(i, arg);
    } else if (arg == "--output-json") {
      options.output_json = need_value(i, arg);
    } else if (arg == "--warmup") {
      options.warmups = csv::ParseSize(need_value(i, arg), "warmup");
    } else if (arg == "--iterations") {
      options.iterations = ParsePositiveSize(need_value(i, arg), "iterations");
    } else if (arg == "--row-limit") {
      options.row_limit = csv::ParseSize(need_value(i, arg), "row_limit");
    } else if (arg == "--rows-per-simdgroup") {
      options.rows_per_simdgroup = csv::ParseU32(need_value(i, arg), "rows_per_simdgroup");
      if (options.rows_per_simdgroup == 0) {
        throw std::invalid_argument("rows_per_simdgroup must be > 0");
      }
    } else if (arg == "--threads-per-threadgroup") {
      options.threads_per_threadgroup =
          csv::ParseU32(need_value(i, arg), "threads_per_threadgroup");
    } else if (arg == "--atomic-tile-rows") {
      options.atomic_tile_rows = csv::ParseSize(need_value(i, arg), "atomic_tile_rows");
      if (options.atomic_tile_rows > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("atomic_tile_rows does not fit uint32");
      }
    } else if (arg == "--accumulation") {
      options.accumulation = need_value(i, arg);
      if (options.accumulation != "atomic" && options.accumulation != "simdgroup" &&
          options.accumulation != "deterministic") {
        throw std::invalid_argument(
            "accumulation must be atomic, simdgroup, or deterministic");
      }
    } else if (arg == "--model-storage") {
      options.storage = need_value(i, arg);
      if (options.storage != "shared" && options.storage != "private") {
        throw std::invalid_argument("model-storage must be shared or private");
      }
    } else if (arg == "--deterministic-scratch-mib") {
      options.deterministic_scratch_mib =
          ParsePositiveSize(need_value(i, arg), "deterministic_scratch_mib");
    } else if (arg == "--relative-floor") {
      options.relative_floor = csv::ParseDouble(need_value(i, arg), "relative_floor");
      if (!(options.relative_floor > 0.0) || !std::isfinite(options.relative_floor)) {
        throw std::invalid_argument("relative_floor must be finite and > 0");
      }
    } else if (arg == "--max-abs-error") {
      options.max_abs_error = csv::ParseDouble(need_value(i, arg), "max_abs_error");
      if (!(options.max_abs_error >= 0.0) || !std::isfinite(options.max_abs_error)) {
        throw std::invalid_argument("max_abs_error must be finite and >= 0");
      }
    } else if (arg == "--help" || arg == "-h") {
      Usage(argv[0]);
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown option: " + arg);
    }
  }
  options.intercepts = intercept_set
                           ? csv::ParseIntercepts(intercept_arg, options.num_groups)
                           : std::vector<double>(options.num_groups, 0.0);
  options.kernel = ResolveKernel(options.kernel);
  return options;
}

double Quantile(std::vector<double> values, double q) {
  if (values.empty()) throw std::invalid_argument("quantile of empty sample");
  std::sort(values.begin(), values.end());
  const double position = q * static_cast<double>(values.size() - 1);
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = static_cast<size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return values[lower] + fraction * (values[upper] - values[lower]);
}

std::string HashFloats(const std::vector<float>& values) {
  uint64_t hash = 1469598103934665603ULL;
  for (float value : values) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    for (unsigned shift = 0; shift < 32; shift += 8) {
      hash ^= static_cast<uint8_t>(bits >> shift);
      hash *= 1099511628211ULL;
    }
  }
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}

struct Differences {
  double max_abs = 0.0;
  double max_relative = 0.0;
  double mean_abs = 0.0;
  double max_row_group_sum_abs = 0.0;
};

Differences Compare(const std::vector<float>& actual, const std::vector<float>& expected,
                    double relative_floor, size_t group_width) {
  if (actual.size() != expected.size()) throw std::invalid_argument("comparison shape mismatch");
  if (group_width == 0 || actual.size() % group_width != 0) {
    throw std::invalid_argument("comparison group width mismatch");
  }
  Differences result;
  double sum_abs = 0.0;
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
      throw std::runtime_error("non-finite attribution at element " + std::to_string(i));
    }
    const double delta = std::fabs(static_cast<double>(actual[i]) - expected[i]);
    sum_abs += delta;
    result.max_abs = std::max(result.max_abs, delta);
    result.max_relative =
        std::max(result.max_relative,
                 delta / std::max(std::fabs(static_cast<double>(expected[i])), relative_floor));
  }
  result.mean_abs = sum_abs / static_cast<double>(actual.size());
  // Each group has an independent local-accuracy identity. Never sum across output
  // groups/classes, where opposite errors could cancel and hide a multiclass defect.
  for (size_t offset = 0; offset < actual.size(); offset += group_width) {
    double actual_sum = 0.0, expected_sum = 0.0;
    for (size_t col = 0; col < group_width; ++col) {
      actual_sum += actual[offset + col];
      expected_sum += expected[offset + col];
    }
    result.max_row_group_sum_abs =
        std::max(result.max_row_group_sum_abs, std::fabs(actual_sum - expected_sum));
  }
  return result;
}

void RequireFinite(const std::vector<float>& values) {
  for (size_t i = 0; i < values.size(); ++i) {
    if (!std::isfinite(values[i])) {
      throw std::runtime_error("non-finite attribution at element " +
                               std::to_string(i));
    }
  }
}

void WriteArray(std::ostream& out, const std::vector<double>& values) {
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    out << std::setprecision(17) << values[i];
  }
  out << ']';
}

void WriteStringArray(std::ostream& out, const std::vector<std::string>& values) {
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    out << JsonString(values[i]);
  }
  out << ']';
}

void WriteSizeArray(std::ostream& out, const std::vector<size_t>& values) {
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    out << values[i];
  }
  out << ']';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseArgs(argc, argv);

    const auto load_start = Clock::now();
    const auto raw_paths = csv::LoadPaths(options.paths);
    size_t rows = 0, cols = 0;
    std::vector<float> data = csv::LoadMatrix(options.matrix, &rows, &cols);
    if (rows == 0 || cols == 0) throw std::invalid_argument("X.csv must be non-empty");
    const size_t source_rows = rows;
    if (options.row_limit != 0 && options.row_limit < rows) {
      rows = options.row_limit;
      data.resize(csv::CheckedMul(rows, cols, "row-limited input"));
    }
    const double load_s = Seconds(load_start);

    const bool is_metallib = EndsWith(options.kernel, ".metallib");
    const auto pipeline_start = Clock::now();
    Explainer explainer(is_metallib ? options.kernel : ReadFile(options.kernel),
                        is_metallib ? Explainer::LibraryKind::kMetallibFile
                                    : Explainer::LibraryKind::kSourceString);
    const double pipeline_s = Seconds(pipeline_start);
    explainer.set_rows_per_simdgroup(options.rows_per_simdgroup);
    explainer.set_threads_per_threadgroup(options.threads_per_threadgroup);
    explainer.set_atomic_tile_rows(options.atomic_tile_rows);
    explainer.set_deterministic_scratch_budget_bytes(csv::CheckedMul(
        options.deterministic_scratch_mib, size_t{1024 * 1024}, "deterministic scratch"));
    const AccumulationMode accumulation =
        options.accumulation == "atomic"
            ? AccumulationMode::kAtomic
            : (options.accumulation == "simdgroup" ? AccumulationMode::kSimdgroup
                                                    : AccumulationMode::kDeterministic);
    explainer.set_accumulation_mode(accumulation);
    const ModelStorageMode storage = options.storage == "shared"
                                         ? ModelStorageMode::kShared
                                         : ModelStorageMode::kPrivate;

    const auto model_start = Clock::now();
    auto model = explainer.Compile(raw_paths, options.num_groups, cols, options.intercepts,
                                   storage);
    const double model_compile_s = Seconds(model_start);

    const size_t output_cols = csv::CheckedMul(options.num_groups, cols + 1, "output cols");
    const size_t output_len = csv::CheckedMul(rows, output_cols, "output");
    std::vector<float> phis(output_len, 0.0f);
    std::vector<float> repeat_min;
    std::vector<float> repeat_max;
    std::vector<float> expected;
    if (!options.expected.empty()) {
      size_t expected_rows = 0, expected_cols = 0;
      expected = csv::LoadMatrix(options.expected, &expected_rows, &expected_cols);
      if (expected_cols != output_cols || expected_rows < rows) {
        throw std::invalid_argument("expected CSV shape does not cover selected X rows");
      }
      expected.resize(output_len);
    }

    const auto warmup_start = Clock::now();
    for (size_t i = 0; i < options.warmups; ++i) {
      explainer.Explain(*model, data.data(), rows, phis.data());
    }
    const double warmup_s = Seconds(warmup_start);

    std::vector<double> wall, api_total, gpu, upload, encode;
    std::vector<size_t> active_scratch_bytes, retained_scratch_bytes, active_tile_rows,
        active_tile_counts, atomic_tile_rows, atomic_tile_counts;
    std::vector<size_t> x_zero_copy;
    std::vector<std::string> hashes;
    wall.reserve(options.iterations);
    api_total.reserve(options.iterations);
    gpu.reserve(options.iterations);
    upload.reserve(options.iterations);
    encode.reserve(options.iterations);
    active_scratch_bytes.reserve(options.iterations);
    retained_scratch_bytes.reserve(options.iterations);
    active_tile_rows.reserve(options.iterations);
    active_tile_counts.reserve(options.iterations);
    atomic_tile_rows.reserve(options.iterations);
    atomic_tile_counts.reserve(options.iterations);
    x_zero_copy.reserve(options.iterations);
    hashes.reserve(options.iterations);
    Differences repeatability;
    Differences accuracy_first;
    Differences accuracy_worst;
    for (size_t i = 0; i < options.iterations; ++i) {
      const auto begin = Clock::now();
      const ExplainTimings timing = explainer.Explain(*model, data.data(), rows, phis.data());
      wall.push_back(Seconds(begin));
      api_total.push_back(timing.total_s);
      gpu.push_back(timing.gpu_s);
      upload.push_back(timing.upload_s);
      encode.push_back(timing.encode_s);
      active_scratch_bytes.push_back(timing.deterministic_scratch_bytes);
      retained_scratch_bytes.push_back(timing.deterministic_scratch_capacity_bytes);
      active_tile_rows.push_back(timing.deterministic_tile_rows);
      active_tile_counts.push_back(timing.deterministic_tiles);
      atomic_tile_rows.push_back(timing.atomic_tile_rows);
      atomic_tile_counts.push_back(timing.atomic_tiles);
      x_zero_copy.push_back(timing.x_zero_copy ? 1 : 0);
      // Keep no-oracle performance runs from accepting NaN/Inf output. This scan is
      // deliberately outside the timed Explain call, alongside hashing and accuracy.
      RequireFinite(phis);
      hashes.push_back(HashFloats(phis));
      if (!expected.empty()) {
        const Differences current =
            Compare(phis, expected, options.relative_floor, cols + 1);
        if (i == 0) accuracy_first = current;
        accuracy_worst.max_abs = std::max(accuracy_worst.max_abs, current.max_abs);
        accuracy_worst.max_relative =
            std::max(accuracy_worst.max_relative, current.max_relative);
        accuracy_worst.mean_abs = std::max(accuracy_worst.mean_abs, current.mean_abs);
        accuracy_worst.max_row_group_sum_abs = std::max(
            accuracy_worst.max_row_group_sum_abs, current.max_row_group_sum_abs);
      }
      if (i == 0) {
        repeat_min = phis;
        repeat_max = phis;
      } else {
        for (size_t j = 0; j < phis.size(); ++j) {
          repeat_min[j] = std::min(repeat_min[j], phis[j]);
          repeat_max[j] = std::max(repeat_max[j], phis[j]);
        }
      }
    }

    // For each output element, the largest absolute difference between any two repeats
    // is its observed max minus min. This is the exact max-pairwise metric, unlike a
    // comparison only to the first run (which can understate spread by up to 2x).
    for (size_t i = 0; i < repeat_min.size(); ++i) {
      const double range = static_cast<double>(repeat_max[i]) - repeat_min[i];
      repeatability.max_abs = std::max(repeatability.max_abs, range);
      const double scale = std::max(
          {std::fabs(static_cast<double>(repeat_min[i])),
           std::fabs(static_cast<double>(repeat_max[i])), options.relative_floor});
      repeatability.max_relative = std::max(repeatability.max_relative, range / scale);
    }

    if (!expected.empty() &&
        (accuracy_worst.max_abs > options.max_abs_error ||
         accuracy_worst.max_row_group_sum_abs > options.max_abs_error)) {
      throw std::runtime_error(
          "accuracy gate failed: worst element error=" +
          std::to_string(accuracy_worst.max_abs) +
          ", worst row/group sum error=" +
          std::to_string(accuracy_worst.max_row_group_sum_abs) +
          ", --max-abs-error=" + std::to_string(options.max_abs_error));
    }
    const size_t unique_hashes = std::set<std::string>(hashes.begin(), hashes.end()).size();
    const double wall_median = Quantile(wall, 0.5);
    const double gpu_median = Quantile(gpu, 0.5);

    std::ostringstream json;
    json << "{\n"
         << "  \"schema\": \"metal_treeshap.phase2.benchmark.v1\",\n"
         << "  \"status\": \"ok\",\n"
         << "  \"workload\": {\"paths\": " << JsonString(options.paths)
         << ", \"matrix\": " << JsonString(options.matrix)
         << ", \"expected\": "
         << (options.expected.empty() ? "null" : JsonString(options.expected))
         << ", \"source_rows\": " << source_rows << ", \"rows\": " << rows
         << ", \"cols\": " << cols << ", \"groups\": " << options.num_groups
         << ", \"raw_path_elements\": " << raw_paths.size()
         << ", \"packed_bins\": " << model->num_bins()
         << ", \"atomic_writes_per_row\": " << model->atomic_writes_per_row()
         << ", \"simdgroup_writes_per_row\": " << model->simdgroup_writes_per_row()
         << ", \"deterministic_num_partials\": " << model->deterministic_num_partials()
         << ", \"deterministic_num_active_cells\": "
         << model->deterministic_num_active_cells()
         << ", \"deterministic_scratch_bytes_per_row\": "
         << model->deterministic_scratch_bytes_per_row()
         << ", \"theoretical_write_reduction\": "
         << (model->simdgroup_writes_per_row() == 0
                 ? 0.0
                 : static_cast<double>(model->atomic_writes_per_row()) /
                       static_cast<double>(model->simdgroup_writes_per_row()))
         << "},\n"
         << "  \"configuration\": {\"kernel\": " << JsonString(options.kernel)
         << ", \"kernel_kind\": " << JsonString(is_metallib ? "metallib" : "source")
         << ", \"rows_per_simdgroup\": " << options.rows_per_simdgroup
         << ", \"threads_per_threadgroup\": " << options.threads_per_threadgroup
         << ", \"atomic_tile_rows\": " << options.atomic_tile_rows
         << ", \"accumulation\": " << JsonString(options.accumulation)
         << ", \"model_storage\": " << JsonString(options.storage)
         << ", \"deterministic_scratch_mib\": " << options.deterministic_scratch_mib
         << ", \"warmups\": " << options.warmups
         << ", \"iterations\": " << options.iterations << "},\n"
         << "  \"setup_s\": {\"load\": " << std::setprecision(17) << load_s
         << ", \"pipeline\": " << pipeline_s << ", \"model_compile\": "
         << model_compile_s << ", \"warmup\": " << warmup_s
         << ", \"total_excluding_warmup\": " << (load_s + pipeline_s + model_compile_s)
         << "},\n"
         << "  \"timing_s\": {\n"
         << "    \"wall\": {\"median\": " << wall_median
         << ", \"p10\": " << Quantile(wall, 0.1) << ", \"p90\": "
         << Quantile(wall, 0.9) << ", \"samples\": ";
    WriteArray(json, wall);
    json << "},\n    \"gpu\": {\"median\": " << gpu_median
         << ", \"p10\": " << Quantile(gpu, 0.1) << ", \"p90\": "
         << Quantile(gpu, 0.9) << ", \"samples\": ";
    WriteArray(json, gpu);
    json << "},\n    \"api_total_samples\": ";
    WriteArray(json, api_total);
    json << ",\n    \"upload_samples\": ";
    WriteArray(json, upload);
    json << ",\n    \"encode_samples\": ";
    WriteArray(json, encode);
    json << ",\n    \"x_zero_copy_samples\": ";
    WriteSizeArray(json, x_zero_copy);
    json << ",\n    \"deterministic_runtime\": {\"active_scratch_bytes_samples\": ";
    WriteSizeArray(json, active_scratch_bytes);
    json << ", \"retained_scratch_bytes_samples\": ";
    WriteSizeArray(json, retained_scratch_bytes);
    json << ", \"tile_rows_samples\": ";
    WriteSizeArray(json, active_tile_rows);
    json << ", \"tile_count_samples\": ";
    WriteSizeArray(json, active_tile_counts);
    json << "},\n    \"atomic_runtime\": {\"tile_rows_samples\": ";
    WriteSizeArray(json, atomic_tile_rows);
    json << ", \"tile_count_samples\": ";
    WriteSizeArray(json, atomic_tile_counts);
    json << "}\n  },\n"
         << "  \"throughput\": {\"rows_per_wall_s\": "
         << (static_cast<double>(rows) / wall_median)
         << ", \"rows_per_gpu_s\": "
         << (gpu_median > 0.0 ? static_cast<double>(rows) / gpu_median : 0.0) << "},\n"
         << "  \"repeatability\": {\"hash_algorithm\": \"fnv1a64-float-bits\", "
         << "\"hashes\": ";
    WriteStringArray(json, hashes);
    json << ", \"unique_hashes\": " << unique_hashes
         << ", \"max_pairwise_abs\": " << repeatability.max_abs
         << ", \"max_pairwise_relative_symmetric\": " << repeatability.max_relative
         << ", \"relative_floor\": " << options.relative_floor << "},\n"
         << "  \"accuracy\": ";
    if (expected.empty()) {
      json << "null\n";
    } else {
      json << "{\"first_run_max_abs\": " << accuracy_first.max_abs
           << ", \"first_run_max_relative\": " << accuracy_first.max_relative
           << ", \"first_run_mean_abs\": " << accuracy_first.mean_abs
           << ", \"first_run_max_row_group_sum_abs\": "
           << accuracy_first.max_row_group_sum_abs
           << ", \"worst_run_max_abs\": " << accuracy_worst.max_abs
           << ", \"worst_run_max_relative\": " << accuracy_worst.max_relative
           << ", \"worst_run_mean_abs\": " << accuracy_worst.mean_abs
           << ", \"worst_run_max_row_group_sum_abs\": "
           << accuracy_worst.max_row_group_sum_abs
           << ", \"relative_floor\": " << options.relative_floor
           << ", \"max_abs_error_gate\": ";
      // Through the stream, not std::to_string: %f's fixed six decimals record any
      // gate below 5e-7 as 0.000000 in the artifact.
      if (std::isfinite(options.max_abs_error)) {
        json << options.max_abs_error;
      } else {
        json << "null";
      }
      json << "}\n";
    }
    json << "}\n";

    if (options.output_json == "-") {
      std::cout << json.str();
    } else {
      std::ofstream out(options.output_json);
      if (!out) throw std::runtime_error("cannot open output " + options.output_json);
      out << json.str();
      if (!out) throw std::runtime_error("failed writing " + options.output_json);
    }
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}

#endif  // __APPLE__
