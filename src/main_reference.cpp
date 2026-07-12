// metal-treeshap: reference CLI.
// Runs the full host pipeline (paths -> preprocess -> scalar reference kernel) on CSV inputs,
// producing complete, xgboost-comparable contributions (model intercept included) with BOTH
// fp64 accumulation (upstream CUDA behavior) and fp32 accumulation (Metal default candidate).
// Shares its CSV contract with metal_cli via src/csv_io.h, so tests/test_fixture.py can diff
// the two engines over identical inputs.
//
// Usage:
//   reference_cli <paths.csv> <X.csv> <num_groups> <out_fp64.csv> <out_fp32.csv>
//                 [intercepts] [shuffle_seed]
//
//   intercepts   comma-separated margin-space intercept per group (e.g. "0.5" or
//                "0.1,0.2,0.3"); default 0. Added to the bias column with the path bias.
//   shuffle_seed if nonzero, process path groups in a seeded random order — a CPU proxy
//                for GPU atomic scheduling order, used by the accumulation-order study.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/metal_treeshap/paths.h"
#include "../include/metal_treeshap/preprocess.h"
#include "../reference/reference_shap.h"
#include "csv_io.h"

using namespace metal_treeshap;

int main(int argc, char** argv) {
  if (argc < 6 || argc > 8) {
    std::cerr << "usage: " << argv[0]
              << " <paths.csv> <X.csv> <num_groups> <out_fp64.csv> <out_fp32.csv>"
                 " [intercepts] [shuffle_seed]\n";
    return 2;
  }
  try {
    const auto raw_paths = csv::LoadPaths(argv[1]);
    size_t rows = 0, cols = 0;
    const auto data = csv::LoadMatrix(argv[2], &rows, &cols);
    if (rows == 0) throw std::invalid_argument("X.csv must contain at least one row");
    if (cols == 0) throw std::invalid_argument("X.csv must contain at least one column");
    const size_t num_groups = csv::ParseSize(argv[3], "num_groups");
    if (num_groups == 0) throw std::invalid_argument("num_groups must be > 0");
    const std::vector<double> intercepts =
        (argc >= 7) ? csv::ParseIntercepts(argv[6], num_groups)
                    : std::vector<double>(num_groups, 0.0);
    const uint64_t shuffle_seed =
        (argc >= 8) ? csv::ParseU64(argv[7], "shuffle_seed") : 0;

    DenseDataset X{data.data(), rows, cols};
    const auto pp = Preprocess(raw_paths, num_groups, cols);

    std::vector<GroupRange> order = CollectGroups(pp);
    if (shuffle_seed != 0) {
      std::mt19937_64 rng(shuffle_seed);
      std::shuffle(order.begin(), order.end(), rng);
    }

    std::fprintf(stderr,
                 "[reference_cli] paths(raw)=%zu dedup=%zu bins=%zu groups_of_work=%zu "
                 "rows=%zu cols=%zu out_groups=%zu shuffle_seed=%llu\n",
                 raw_paths.size(), pp.elements.size(), pp.num_bins, order.size(), rows, cols,
                 num_groups, static_cast<unsigned long long>(shuffle_seed));

    if (cols == std::numeric_limits<size_t>::max()) {
      throw std::invalid_argument("num_cols + 1 overflows size_t");
    }
    const size_t out_size = csv::CheckedMul(
        csv::CheckedMul(rows, num_groups, "output"), cols + 1, "output");
    std::vector<double> phis64(out_size, 0.0);
    std::vector<float> phis32(out_size, 0.0f);
    ShapReference(X, pp, num_groups, phis64.data(), intercepts, &order);
    ShapReference(X, pp, num_groups, phis32.data(), intercepts, &order);

    csv::WritePhis(argv[4], phis64, rows, num_groups * (cols + 1));
    csv::WritePhis(argv[5], phis32, rows, num_groups * (cols + 1));
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
