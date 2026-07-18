// metal-treeshap: shared CSV I/O for the CLI runners (reference_cli and metal_cli).
// One implementation so the two engines can never drift in how they read fixtures.
#pragma once

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "../include/metal_treeshap/paths.h"

namespace metal_treeshap {
namespace csv {

inline std::vector<std::string> SplitLine(const std::string& line) {
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string tok;
  while (std::getline(ss, tok, ',')) out.push_back(tok);
  // std::getline does not emit the final empty field for "a,b,". Preserve it so the
  // strict numeric parser rejects malformed rows instead of silently dropping a column.
  if (!line.empty() && line.back() == ',') out.emplace_back();
  return out;
}

inline std::string TrimToken(std::string_view token) {
  constexpr std::string_view ws = " \t\r\n\f\v";
  const size_t first = token.find_first_not_of(ws);
  if (first == std::string_view::npos) return {};
  const size_t last = token.find_last_not_of(ws);
  return std::string(token.substr(first, last - first + 1));
}

inline uint64_t ParseU64(std::string_view token, const char* what) {
  const std::string s = TrimToken(token);
  if (s.empty() || s.front() == '-') {
    throw std::runtime_error(std::string("invalid ") + what + ": '" + s + "'");
  }
  errno = 0;
  char* end = nullptr;
  const unsigned long long value = std::strtoull(s.c_str(), &end, 10);
  if (errno == ERANGE || end == s.c_str() || *end != '\0') {
    throw std::runtime_error(std::string("invalid ") + what + ": '" + s + "'");
  }
  return static_cast<uint64_t>(value);
}

inline int64_t ParseI64(std::string_view token, const char* what) {
  const std::string s = TrimToken(token);
  errno = 0;
  char* end = nullptr;
  const long long value = std::strtoll(s.c_str(), &end, 10);
  if (s.empty() || errno == ERANGE || end == s.c_str() || *end != '\0') {
    throw std::runtime_error(std::string("invalid ") + what + ": '" + s + "'");
  }
  return static_cast<int64_t>(value);
}

inline size_t ParseSize(std::string_view token, const char* what) {
  const uint64_t value = ParseU64(token, what);
  if (value > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error(std::string(what) + " does not fit size_t");
  }
  return static_cast<size_t>(value);
}

inline uint32_t ParseU32(std::string_view token, const char* what) {
  const uint64_t value = ParseU64(token, what);
  if (value > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(std::string(what) + " does not fit uint32");
  }
  return static_cast<uint32_t>(value);
}

// ERANGE is fatal only for overflow (result +/-HUGE_VAL). macOS strtod/strtof also set
// it for gradual underflow while returning the correct subnormal; values like 1e-42 are
// representable input, not errors.
inline double ParseDouble(std::string_view token, const char* what) {
  const std::string s = TrimToken(token);
  errno = 0;
  char* end = nullptr;
  const double value = std::strtod(s.c_str(), &end);
  const bool overflow = errno == ERANGE && std::fabs(value) == HUGE_VAL;
  if (s.empty() || overflow || end == s.c_str() || *end != '\0') {
    throw std::runtime_error(std::string("invalid ") + what + ": '" + s + "'");
  }
  return value;
}

inline float ParseFloat(std::string_view token, const char* what) {
  const std::string s = TrimToken(token);
  errno = 0;
  char* end = nullptr;
  const float value = std::strtof(s.c_str(), &end);
  const bool overflow = errno == ERANGE && std::fabs(value) == HUGE_VALF;
  if (s.empty() || overflow || end == s.c_str() || *end != '\0') {
    throw std::runtime_error(std::string("invalid ") + what + ": '" + s + "'");
  }
  return value;
}

inline size_t CheckedMul(size_t a, size_t b, const char* what) {
  if (b != 0 && a > std::numeric_limits<size_t>::max() / b) {
    throw std::runtime_error(std::string(what) + " size overflows");
  }
  return a * b;
}

// paths.csv columns: path_idx,feature_idx,group,lower,upper,is_missing,zero_fraction,v
// (header row required).
inline std::vector<PathElement> LoadPaths(const std::string& file) {
  std::ifstream in(file);
  if (!in) throw std::runtime_error("cannot open " + file);
  std::vector<PathElement> paths;
  std::string line;
  if (!std::getline(in, line)) throw std::runtime_error("empty paths file: " + file);
  // Require the documented header row rather than discarding the first line blindly:
  // X.csv in the same CLI contract is headerless, so a headerless paths.csv would
  // otherwise lose its first element silently and exit 0 with wrong attributions.
  const std::string first_column = TrimToken(line.substr(0, line.find(',')));
  if (first_column != "path_idx") {
    throw std::runtime_error(
        "paths file " + file + " must start with the header row 'path_idx,feature_idx,"
        "group,lower,upper,is_missing,zero_fraction,v' (first column read: '" +
        first_column + "')");
  }
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();  // CRLF blank lines
    if (line.empty()) continue;
    auto f = SplitLine(line);
    if (f.size() != 8) throw std::runtime_error("bad paths row: " + line);
    PathElement e;
    e.path_idx = ParseU64(f[0], "path_idx");
    e.feature_idx = ParseI64(f[1], "feature_idx");
    const int64_t group = ParseI64(f[2], "group");
    if (group < std::numeric_limits<int32_t>::min() ||
        group > std::numeric_limits<int32_t>::max()) {
      throw std::runtime_error("group does not fit int32");
    }
    e.group = static_cast<int32_t>(group);
    const int64_t missing = ParseI64(f[5], "is_missing");
    if (missing != 0 && missing != 1) {
      throw std::runtime_error("is_missing must be 0 or 1");
    }
    const float lower = ParseFloat(f[3], "lower");
    const float upper = ParseFloat(f[4], "upper");
    // XgboostSplitCondition asserts this invariant in debug builds. Convert malformed
    // external data into a normal parse error instead of letting a CLI abort before the
    // preprocessing validator can report it. Infinities are valid open-ended bounds.
    if (std::isnan(lower) || std::isnan(upper) || lower > upper) {
      throw std::runtime_error("split bounds must be non-NaN with lower <= upper");
    }
    e.split_condition = XgboostSplitCondition(lower, upper, missing == 1);
    e.zero_fraction = ParseDouble(f[6], "zero_fraction");
    e.v = ParseFloat(f[7], "leaf value");
    paths.push_back(e);
  }
  return paths;
}

// Dense float matrix, no header; "nan" marks missing values.
inline std::vector<float> LoadMatrix(const std::string& file, size_t* rows, size_t* cols) {
  std::ifstream in(file);
  if (!in) throw std::runtime_error("cannot open " + file);
  std::vector<float> data;
  std::string line;
  *rows = 0;
  *cols = 0;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();  // CRLF blank lines
    if (line.empty()) continue;
    auto f = SplitLine(line);
    if (*cols == 0) {
      *cols = f.size();
    } else if (f.size() != *cols) {
      throw std::runtime_error("ragged X.csv");
    }
    for (auto& tok : f) data.push_back(ParseFloat(tok, "matrix value"));
    if (*rows == std::numeric_limits<size_t>::max()) {
      throw std::runtime_error("matrix row count overflows size_t");
    }
    (*rows)++;
  }
  return data;
}

// Comma-separated margin-space intercepts; a single value broadcasts across groups.
inline std::vector<double> ParseIntercepts(const std::string& arg, size_t num_groups) {
  std::vector<double> out;
  for (const auto& tok : SplitLine(arg)) {
    const double value = ParseDouble(tok, "intercept");
    if (!std::isfinite(value)) throw std::runtime_error("intercepts must be finite");
    out.push_back(value);
  }
  if (out.size() == 1 && num_groups > 1) out.assign(num_groups, out[0]);
  if (out.size() != num_groups) throw std::runtime_error("intercepts count != num_groups");
  return out;
}

template <typename T>
inline void WritePhis(const std::string& file, const std::vector<T>& phis, size_t rows,
                      size_t per_row) {
  const size_t expected = CheckedMul(rows, per_row, "output");
  if (phis.size() != expected) {
    throw std::runtime_error("output shape does not match attribution vector");
  }
  std::ofstream out(file);
  if (!out) throw std::runtime_error("cannot open output " + file);
  out.precision(12);
  for (size_t r = 0; r < rows; r++) {
    for (size_t c = 0; c < per_row; c++) {
      out << phis[r * per_row + c] << (c + 1 == per_row ? "" : ",");
    }
    out << "\n";
  }
  if (!out) throw std::runtime_error("failed writing output " + file);
}

}  // namespace csv
}  // namespace metal_treeshap
