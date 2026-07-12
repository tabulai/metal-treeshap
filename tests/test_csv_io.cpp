#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/csv_io.h"

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

void Write(const std::filesystem::path& path, const std::string& contents) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("test setup could not open temporary file");
  out << contents;
  if (!out) throw std::runtime_error("test setup could not write temporary file");
}

}  // namespace

int main() {
  namespace fs = std::filesystem;
  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  const fs::path dir = fs::temp_directory_path() /
                       ("metal_treeshap_csv_test_" + std::to_string(nonce));
  fs::create_directories(dir);
  try {
    Check(csv::ParseU64(" 42 ", "value") == 42, "trimmed uint parse failed");
    Check(csv::ParseI64("-7", "value") == -7, "signed parse failed");
    Check(std::isnan(csv::ParseFloat("nan", "value")), "NaN matrix parse failed");
    Throws([] { (void)csv::ParseU64("-1", "value"); }, "negative uint accepted");
    Throws([] { (void)csv::ParseU64("1garbage", "value"); }, "numeric suffix accepted");
    Throws([] { (void)csv::ParseU32("4294967296", "value"); }, "uint32 overflow accepted");
    Throws([] { (void)csv::ParseDouble("", "value"); }, "empty number accepted");
    Throws([] { (void)csv::ParseIntercepts("nan", 1); }, "NaN intercept accepted");
    Throws([] { (void)csv::ParseIntercepts("inf", 1); }, "infinite intercept accepted");

    const std::string header =
        "path_idx,feature_idx,group,lower,upper,is_missing,zero_fraction,v\n";
    const fs::path valid = dir / "valid.csv";
    Write(valid, header + "0,-1,0,-inf,inf,1,1,2\n0,0,0,-inf,3,0,0.5,2\n");
    const auto paths = csv::LoadPaths(valid.string());
    Check(paths.size() == 2, "valid path count wrong");
    Check(paths[1].feature_idx == 0 && paths[1].zero_fraction == 0.5,
          "valid path values wrong");

    const fs::path garbage = dir / "garbage.csv";
    Write(garbage, header + "0,-1,0,-inf,inf,1,1x,2\n");
    Throws([&] { (void)csv::LoadPaths(garbage.string()); }, "garbage numeric field accepted");

    const fs::path overflow = dir / "overflow.csv";
    Write(overflow, header + "0,-1,2147483648,-inf,inf,1,1,2\n");
    Throws([&] { (void)csv::LoadPaths(overflow.string()); }, "int32 overflow accepted");

    const fs::path trailing = dir / "trailing.csv";
    Write(trailing, header + "0,-1,0,-inf,inf,1,1,2,\n");
    Throws([&] { (void)csv::LoadPaths(trailing.string()); }, "trailing CSV field accepted");

    const fs::path nan_bound = dir / "nan_bound.csv";
    Write(nan_bound, header + "0,0,0,nan,1,0,0.5,2\n");
    Throws([&] { (void)csv::LoadPaths(nan_bound.string()); }, "NaN split bound accepted");

    const fs::path inverted = dir / "inverted.csv";
    Write(inverted, header + "0,0,0,2,1,0,0.5,2\n");
    Throws([&] { (void)csv::LoadPaths(inverted.string()); }, "inverted split accepted");

    const fs::path bad_bool = dir / "bad_bool.csv";
    Write(bad_bool, header + "0,0,0,0,1,2,0.5,2\n");
    Throws([&] { (void)csv::LoadPaths(bad_bool.string()); }, "non-boolean missing flag accepted");

    const fs::path matrix = dir / "X.csv";
    Write(matrix, "1,nan\r\n2,3\r\n");
    size_t rows = 0, cols = 0;
    const auto values = csv::LoadMatrix(matrix.string(), &rows, &cols);
    Check(rows == 2 && cols == 2 && values.size() == 4, "valid matrix shape wrong");
    Check(std::isnan(values[1]), "matrix NaN lost");

    const fs::path bad_matrix = dir / "bad_X.csv";
    Write(bad_matrix, "1,2,\n");
    Throws([&] { (void)csv::LoadMatrix(bad_matrix.string(), &rows, &cols); },
           "empty trailing matrix field accepted");

    const std::vector<float> output{1.0f, 2.0f};
    csv::WritePhis((dir / "out.csv").string(), output, 1, 2);
    Throws([&] { csv::WritePhis((dir / "wrong.csv").string(), output, 2, 2); },
           "mismatched output shape accepted");
    Throws([&] { csv::WritePhis((dir / "missing" / "out.csv").string(), output, 1, 2); },
           "unopenable output path accepted");

    fs::remove_all(dir);
    std::cout << "ALL " << checks << " CSV I/O TESTS PASSED\n";
  } catch (...) {
    fs::remove_all(dir);
    throw;
  }
  return 0;
}
