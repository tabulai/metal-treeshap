// metal-treeshap: shared CSV I/O for the CLI runners (reference_cli and metal_cli).
// One implementation so the two engines can never drift in how they read fixtures.
#pragma once

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/metal_treeshap/paths.h"

namespace metal_treeshap {
namespace csv {

inline std::vector<std::string> SplitLine(const std::string& line) {
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string tok;
  while (std::getline(ss, tok, ',')) out.push_back(tok);
  return out;
}

// paths.csv columns: path_idx,feature_idx,group,lower,upper,is_missing,zero_fraction,v
// (header row required).
inline std::vector<PathElement> LoadPaths(const std::string& file) {
  std::ifstream in(file);
  if (!in) throw std::runtime_error("cannot open " + file);
  std::vector<PathElement> paths;
  std::string line;
  std::getline(in, line);  // header
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    auto f = SplitLine(line);
    if (f.size() != 8) throw std::runtime_error("bad paths row: " + line);
    PathElement e;
    e.path_idx = std::strtoull(f[0].c_str(), nullptr, 10);
    e.feature_idx = std::strtoll(f[1].c_str(), nullptr, 10);
    e.group = static_cast<int32_t>(std::strtol(f[2].c_str(), nullptr, 10));
    e.split_condition = XgboostSplitCondition(std::strtof(f[3].c_str(), nullptr),
                                              std::strtof(f[4].c_str(), nullptr),
                                              std::strtol(f[5].c_str(), nullptr, 10) != 0);
    e.zero_fraction = std::strtod(f[6].c_str(), nullptr);
    e.v = std::strtof(f[7].c_str(), nullptr);
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
    if (line.empty()) continue;
    auto f = SplitLine(line);
    if (*cols == 0) {
      *cols = f.size();
    } else if (f.size() != *cols) {
      throw std::runtime_error("ragged X.csv");
    }
    for (auto& tok : f) data.push_back(std::strtof(tok.c_str(), nullptr));
    (*rows)++;
  }
  return data;
}

// Comma-separated margin-space intercepts; a single value broadcasts across groups.
inline std::vector<double> ParseIntercepts(const std::string& arg, size_t num_groups) {
  std::vector<double> out;
  for (const auto& tok : SplitLine(arg)) out.push_back(std::strtod(tok.c_str(), nullptr));
  if (out.size() == 1 && num_groups > 1) out.assign(num_groups, out[0]);
  if (out.size() != num_groups) throw std::runtime_error("intercepts count != num_groups");
  return out;
}

template <typename T>
inline void WritePhis(const std::string& file, const std::vector<T>& phis, size_t rows,
                      size_t per_row) {
  std::ofstream out(file);
  out.precision(12);
  for (size_t r = 0; r < rows; r++) {
    for (size_t c = 0; c < per_row; c++) {
      out << phis[r * per_row + c] << (c + 1 == per_row ? "" : ",");
    }
    out << "\n";
  }
}

}  // namespace csv
}  // namespace metal_treeshap
