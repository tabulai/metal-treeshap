// Unit tests for the portable host pipeline (no Metal required — runs anywhere).
// Build & run:  g++ -std=c++20 -O2 -o test_preprocess tests/test_preprocess.cpp && ./test_preprocess
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>

#include "../include/metal_treeshap/paths.h"
#include "../include/metal_treeshap/preprocess.h"
#include "../reference/reference_shap.h"

using namespace metal_treeshap;

static int checks = 0;
#define CHECK(cond)                                            \
  do {                                                         \
    checks++;                                                  \
    if (!(cond)) {                                             \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                \
    }                                                          \
  } while (0)

static PathElement MakeElement(uint64_t path, int64_t feat, double zf, float v, float lo = -1e30f,
                               float hi = 1e30f, bool missing = true, int group = 0) {
  PathElement e;
  e.path_idx = path;
  e.feature_idx = feat;
  e.group = group;
  e.split_condition = XgboostSplitCondition(lo, hi, missing);
  e.zero_fraction = zf;
  e.v = v;
  return e;
}

static int TestDedup() {
  // Path 0 splits twice on feature 3: intervals intersect, zero_fractions multiply.
  std::vector<PathElement> raw{
      MakeElement(0, 3, 0.5, 1.0f, 0.0f, 10.0f, true),
      MakeElement(0, -1, 1.0, 1.0f),
      MakeElement(0, 3, 0.4, 1.0f, 2.0f, 20.0f, false),
      MakeElement(0, 1, 0.9, 1.0f, -5.0f, 0.0f, true),
  };
  auto dd = DeduplicatePaths(raw);
  CHECK(dd.size() == 3);
  CHECK(dd[0].feature_idx == -1);  // root sorts first
  CHECK(dd[1].feature_idx == 1);
  CHECK(dd[2].feature_idx == 3);
  CHECK(std::abs(dd[2].zero_fraction - 0.2) < 1e-12);
  CHECK(dd[2].split_condition.feature_lower_bound == 2.0f);   // max of lowers
  CHECK(dd[2].split_condition.feature_upper_bound == 10.0f);  // min of uppers
  CHECK(dd[2].split_condition.is_missing_branch == false);    // AND
  return 0;
}

static int TestBinPacking() {
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(1, 32);
  LengthMap lengths;
  int total = 0;
  for (uint64_t i = 0; i < 5000; i++) {
    lengths[i] = dist(rng);
    total += lengths[i];
  }
  auto check_valid = [&](const BinMap& bm) {
    std::map<size_t, int> load;
    for (const auto& [p, b] : bm) load[b] += lengths.at(p);
    for (const auto& [b, l] : load) {
      (void)b;
      CHECK(l <= kBinLimit);
    }
    CHECK(bm.size() == lengths.size());
    return 0;
  };
  auto bfd = BFDBinPacking(lengths), ffd = FFDBinPacking(lengths), nf = NFBinPacking(lengths);
  if (check_valid(bfd) || check_valid(ffd) || check_valid(nf)) return 1;
  auto num_bins = [](const BinMap& bm) {
    size_t n = 0;
    for (const auto& [p, b] : bm) {
      (void)p;
      n = std::max(n, b + 1);
    }
    return n;
  };
  size_t lower_bound = (total + kBinLimit - 1) / kBinLimit;
  CHECK(num_bins(bfd) >= lower_bound);
  CHECK(num_bins(bfd) <= num_bins(ffd));
  CHECK(num_bins(ffd) <= num_bins(nf));
  std::printf("  bin packing: lower_bound=%zu BFD=%zu FFD=%zu NF=%zu\n", lower_bound,
              num_bins(bfd), num_bins(ffd), num_bins(nf));

  // Perfect-fit case
  LengthMap perfect{{0, 16}, {1, 16}, {2, 16}, {3, 16}};
  CHECK(num_bins(BFDBinPacking(perfect)) == 2);
  return 0;
}

static bool Throws(std::vector<PathElement> raw, size_t num_cols = 64,
                   size_t num_groups = 1) {
  // Mirrors Preprocess's two-layer validation: raw checks BEFORE dedup (merging can
  // launder malformed input), structural checks after.
  try {
    ValidateRawPaths(raw, num_cols, num_groups);
    auto dd = DeduplicatePaths(std::move(raw));
    ValidatePaths(dd, GetPathLengths(dd), num_cols, num_groups);
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

static int TestValidate() {
  // Length 33 (root + 32 splits) must throw; length 32 (root + 31, the depth-31 case)
  // must pass — the exact SIMD-width boundary.
  std::vector<PathElement> too_long{MakeElement(0, -1, 1.0, 1.0f)};
  for (int f = 0; f < 32; f++) too_long.push_back(MakeElement(0, f, 0.5, 1.0f));
  CHECK(Throws(too_long));
  std::vector<PathElement> max_ok{MakeElement(0, -1, 1.0, 1.0f)};
  for (int f = 0; f < 31; f++) max_ok.push_back(MakeElement(0, f, 0.5, 1.0f));
  CHECK(!Throws(max_ok));

  // Inconsistent leaf value.
  CHECK(Throws({MakeElement(1, -1, 1.0, 1.0f), MakeElement(1, 0, 0.5, 2.0f)}));
  // Missing root.
  CHECK(Throws({MakeElement(2, 0, 0.5, 1.0f)}));
  // Root with wrong zero_fraction.
  CHECK(Throws({MakeElement(3, -1, 0.7, 1.0f), MakeElement(3, 0, 0.5, 1.0f)}));
  // Feature index out of range for the dataset.
  CHECK(Throws({MakeElement(4, -1, 1.0, 1.0f), MakeElement(4, 99, 0.5, 1.0f)}, /*cols=*/8));
  // Group out of range.
  {
    PathElement e = MakeElement(5, 0, 0.5, 1.0f);
    e.group = 3;
    PathElement r = MakeElement(5, -1, 1.0, 1.0f);
    r.group = 3;
    CHECK(Throws({r, e}, 8, /*groups=*/2));
  }
  // zero_fraction out of [0,1] and non-finite leaf values.
  CHECK(Throws({MakeElement(6, -1, 1.0, 1.0f), MakeElement(6, 0, 1.5, 1.0f)}));
  CHECK(Throws({MakeElement(7, -1, 1.0, 1.0f),
                MakeElement(7, 0, std::numeric_limits<double>::quiet_NaN(), 1.0f)}));
  CHECK(Throws({MakeElement(8, -1, 1.0, std::numeric_limits<float>::infinity()),
                MakeElement(8, 0, 0.5, std::numeric_limits<float>::infinity())}));
  return 0;
}

// Cases dedup would LAUNDER if validation ran after merging (review finding: dedup
// multiplies fractions, intersects intervals, and keeps first-element metadata, hiding
// raw invalidity). All must throw because ValidateRawPaths runs first.
static int TestValidateRawLaundering() {
  // Duplicate roots: dedup would merge the two (path_idx, -1) elements into one.
  CHECK(Throws({MakeElement(0, -1, 1.0, 1.0f), MakeElement(0, -1, 1.0, 1.0f),
                MakeElement(0, 2, 0.5, 1.0f)}));
  // Individually invalid fractions whose product is plausible: 2.0 x 0.25 = 0.5.
  CHECK(Throws({MakeElement(1, -1, 1.0, 1.0f), MakeElement(1, 2, 2.0, 1.0f),
                MakeElement(1, 2, 0.25, 1.0f)}));
  // Negative fractions whose product is positive: -0.5 x -0.5 = 0.25.
  CHECK(Throws({MakeElement(2, -1, 1.0, 1.0f), MakeElement(2, 2, -0.5, 1.0f),
                MakeElement(2, 2, -0.5, 1.0f)}));
  // Conflicting leaf value on same-feature duplicates (dedup keeps the first).
  CHECK(Throws({MakeElement(3, -1, 1.0, 1.0f), MakeElement(3, 2, 0.5, 1.0f),
                MakeElement(3, 2, 0.5, 9.0f)}));
  // Conflicting group on same-feature duplicates (dedup keeps the first).
  {
    PathElement a = MakeElement(4, -1, 1.0, 1.0f);
    PathElement b = MakeElement(4, 2, 0.5, 1.0f);
    PathElement c = MakeElement(4, 2, 0.5, 1.0f);
    c.group = 1;
    CHECK(Throws({a, b, c}, 64, /*groups=*/2));
  }
  // NaN interval bound (bypass the ctor assert by setting fields directly).
  {
    PathElement e = MakeElement(5, 2, 0.5, 1.0f);
    e.split_condition.feature_lower_bound = std::numeric_limits<float>::quiet_NaN();
    CHECK(Throws({MakeElement(5, -1, 1.0, 1.0f), e}));
  }
  // Inverted interval bounds.
  {
    PathElement e = MakeElement(6, 2, 0.5, 1.0f);
    e.split_condition.feature_lower_bound = 3.0f;
    e.split_condition.feature_upper_bound = -3.0f;
    CHECK(Throws({MakeElement(6, -1, 1.0, 1.0f), e}));
  }
  // Sanity: a VALID duplicate-feature path still passes both layers and merges.
  {
    std::vector<PathElement> ok{MakeElement(7, -1, 1.0, 1.0f),
                                MakeElement(7, 2, 0.5, 1.0f, 0.0f, 10.0f, true),
                                MakeElement(7, 2, 0.8, 1.0f, 2.0f, 20.0f, true)};
    CHECK(!Throws(ok));
  }
  // Empty numeric intersection with at least one non-missing edge is unsatisfiable: NaN
  // cannot rescue the path after Merge ANDs the missing flags, so post-dedup must reject it.
  CHECK(Throws({MakeElement(8, -1, 1.0, 1.0f),
                MakeElement(8, 2, 0.5, 1.0f, 0.0f, 1.0f, true),
                MakeElement(8, 2, 0.5, 1.0f, 2.0f, 3.0f, false)}));
  // Degenerate touching intervals are likewise rejected when missing does not follow both.
  CHECK(Throws({MakeElement(9, -1, 1.0, 1.0f),
                MakeElement(9, 2, 0.5, 1.0f, 0.0f, 1.0f, true),
                MakeElement(9, 2, 0.5, 1.0f, 1.0f, 2.0f, false)}));

  // Real XGBoost trees can contain a missing-only path: both repeated-feature edges route
  // NaN to the leaf while their numeric intervals are complementary.  The merged [1,1)
  // interval is empty for finite values but EvaluateSplit(NaN) is true, so this must pass.
  std::vector<PathElement> missing_only{
      MakeElement(10, -1, 1.0, -3.0f),
      MakeElement(10, 0, 2.0 / 3.0, -3.0f,
                  -std::numeric_limits<float>::infinity(), 1.0f, true),
      MakeElement(10, 0, 0.5, -3.0f, 1.0f,
                  std::numeric_limits<float>::infinity(), true)};
  CHECK(!Throws(missing_only, /*num_cols=*/1));
  auto missing_only_dd = DeduplicatePaths(std::move(missing_only));
  CHECK(missing_only_dd.size() == 2);
  CHECK(missing_only_dd[1].split_condition.feature_lower_bound == 1.0f);
  CHECK(missing_only_dd[1].split_condition.feature_upper_bound == 1.0f);
  CHECK(missing_only_dd[1].split_condition.EvaluateSplit(
      std::numeric_limits<float>::quiet_NaN()));
  CHECK(!missing_only_dd[1].split_condition.EvaluateSplit(1.0f));
  return 0;
}

static int TestSegmentsAndLayout() {
  // Prefix offsets are accumulated in size_t and rejected before uint32 narrowing.
  CHECK(CheckedBinSegmentsFromCounts({2, 0, 3}) ==
        std::vector<uint32_t>({0, 2, 2, 5}));
  try {
    (void)CheckedBinSegmentsFromCounts(
        {static_cast<size_t>(std::numeric_limits<uint32_t>::max()), 1});
    CHECK(false);
  } catch (const std::overflow_error&) {
    CHECK(true);
  }

  std::mt19937 rng(7);
  std::uniform_int_distribution<int> depth_dist(1, 12);
  std::vector<PathElement> raw;
  for (uint64_t p = 0; p < 400; p++) {
    int depth = depth_dist(rng);
    float v = static_cast<float>(p % 5) - 2.0f;
    raw.push_back(MakeElement(p, -1, 1.0, v));
    for (int f = 0; f < depth; f++) raw.push_back(MakeElement(p, f, 0.5, v));
  }
  auto pp = Preprocess(raw, 1, /*num_cols=*/16);
  CHECK(pp.bin_segments.front() == 0);
  CHECK(pp.bin_segments.back() == pp.elements.size());
  for (size_t b = 0; b < pp.num_bins; b++) {
    size_t start = pp.bin_segments[b], end = pp.bin_segments[b + 1];
    CHECK(end - start <= 32);  // fits one SIMD-group
    CHECK(end > start);
    // Within a bin: path runs are contiguous, each starting with its root.
    for (size_t i = start; i < end; i++) {
      bool run_start = (i == start) || (pp.elements[i].path_idx != pp.elements[i - 1].path_idx);
      if (run_start) CHECK(pp.elements[i].IsRoot());
    }
  }
  return 0;
}

static int TestReferenceSingleSplit() {
  // Tree: f0 < 0.5 -> leaf -1 (cover 2), else leaf 0 (cover 3). Total cover 5.
  // E[f] = (2*(-1) + 3*0)/5 = -0.4. For x = 1.0: prediction 0, so phi_f0 = 0.4, bias = -0.4.
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<PathElement> raw{
      MakeElement(0, -1, 1.0, -1.0f),
      MakeElement(0, 0, 2.0 / 5.0, -1.0f, -inf, 0.5f, true),
      MakeElement(1, -1, 1.0, 0.0f),
      MakeElement(1, 0, 3.0 / 5.0, 0.0f, 0.5f, inf, false),
  };
  auto pp = Preprocess(raw, 1, /*num_cols=*/1);
  float x[2] = {1.0f, 0.0f};  // two rows, one feature
  DenseDataset X{x, 2, 1};
  std::vector<double> phis(2 * 1 * 2, 0.0);
  ShapReference(X, pp, 1, phis.data());
  CHECK(std::abs(phis[IndexPhi(0, 1, 0, 1, 0)] - 0.4) < 1e-6);   // row 0 phi_f0
  CHECK(std::abs(phis[IndexPhi(0, 1, 0, 1, 1)] + 0.4) < 1e-6);   // row 0 bias
  CHECK(std::abs(phis[IndexPhi(1, 1, 0, 1, 0)] + 0.6) < 1e-6);   // row 1: pred -1 => phi = -0.6
  CHECK(std::abs(phis[IndexPhi(1, 1, 0, 1, 1)] + 0.4) < 1e-6);
  // Local accuracy: contributions sum to the prediction.
  CHECK(std::abs(phis[0] + phis[1] - 0.0) < 1e-6);
  CHECK(std::abs(phis[2] + phis[3] + 1.0) < 1e-6);
  return 0;
}

static int TestReferenceMissingValues() {
  // Same tree; NaN goes down the "yes" (left) branch => same phis as x < 0.5.
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<PathElement> raw{
      MakeElement(0, -1, 1.0, -1.0f),
      MakeElement(0, 0, 0.4, -1.0f, -inf, 0.5f, true),
      MakeElement(1, -1, 1.0, 0.0f),
      MakeElement(1, 0, 0.6, 0.0f, 0.5f, inf, false),
  };
  auto pp = Preprocess(raw, 1, /*num_cols=*/1);
  float x[1] = {std::numeric_limits<float>::quiet_NaN()};
  DenseDataset X{x, 1, 1};
  std::vector<double> phis(2, 0.0);
  ShapReference(X, pp, 1, phis.data());
  CHECK(std::abs(phis[0] + 0.6) < 1e-6);  // NaN -> left leaf (-1): phi = -1 - (-0.4) = -0.6
  CHECK(std::abs(phis[1] + 0.4) < 1e-6);
  return 0;
}

static int TestStumpAndIntercept() {
  // A stump (single-leaf tree) is a root-only path: contributes only bias. With an
  // intercept passed through ShapReference, the bias column must be bias + intercept.
  std::vector<PathElement> raw{MakeElement(0, -1, 1.0, 2.5f)};
  auto pp = Preprocess(raw, 1, /*num_cols=*/1);
  float x[1] = {0.0f};
  DenseDataset X{x, 1, 1};
  std::vector<double> phis(2, 0.0);
  ShapReference(X, pp, 1, phis.data(), /*intercepts=*/{0.5});
  CHECK(std::abs(phis[0] - 0.0) < 1e-9);        // no feature contribution
  CHECK(std::abs(phis[1] - (2.5 + 0.5)) < 1e-9);  // bias = leaf + intercept
  return 0;
}

int main() {
  if (TestDedup()) return 1;
  std::printf("dedup ok\n");
  if (TestBinPacking()) return 1;
  std::printf("bin packing ok\n");
  if (TestValidate()) return 1;
  std::printf("validate ok\n");
  if (TestValidateRawLaundering()) return 1;
  std::printf("raw-laundering validation ok\n");
  if (TestSegmentsAndLayout()) return 1;
  std::printf("segments/layout ok\n");
  if (TestReferenceSingleSplit()) return 1;
  std::printf("reference single-split ok\n");
  if (TestReferenceMissingValues()) return 1;
  std::printf("reference missing-values ok\n");
  if (TestStumpAndIntercept()) return 1;
  std::printf("stump + intercept ok\n");
  std::printf("ALL OK (%d checks)\n", checks);
  return 0;
}
