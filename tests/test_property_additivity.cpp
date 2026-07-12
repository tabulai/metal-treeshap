// Property-based test: local accuracy (additivity) on random synthetic ensembles.
//
// TreeSHAP's defining invariant is that contributions plus bias sum exactly to the model's
// margin prediction for every row. This test builds random tree ensembles directly (no
// xgboost needed), including the stress cases the review asked for:
//   * depths up to 31 (root + 31 splits = the 32-lane boundary)
//   * stumps (single-leaf trees)
//   * repeated features along a path (exercises dedup/Merge)
//   * extreme cover ratios (zero_fractions down to ~1e-4)
//   * missing values routed by per-node default branches
// then checks sum(phis) == margin-by-traversal for both fp64 and fp32 accumulation.
//
// Build & run:  g++ -std=c++20 -O2 -o test_property tests/test_property_additivity.cpp
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../include/metal_treeshap/paths.h"
#include "../include/metal_treeshap/preprocess.h"
#include "../reference/reference_shap.h"

using namespace metal_treeshap;

struct Node {
  int left = -1, right = -1;
  int feature = -1;
  float threshold = 0.0f;
  bool default_left = true;
  double cover = 0.0;
  float leaf = 0.0f;
  bool IsLeaf() const { return left == -1; }
};

struct Tree {
  std::vector<Node> nodes;
};

// feature_pool controls how many distinct features the tree may split on. A SMALL pool
// (4) forces repeated features along paths, exercising dedup/Merge; a LARGE pool (all
// features) lets deep trees keep long deduplicated paths — without it, dedup caps every
// cooperative group at pool+1 elements and deep trials never exercise long recurrences
// (review finding: the original 0..3 pool capped groups at 5 even for raw depth 31).
//
// The generator is CONSTRAINT-AWARE: it tracks the feasible value interval per feature
// along each branch and draws thresholds strictly inside it, exactly as real trained
// trees do (a split threshold always separates data present in the node). Independent
// thresholds would create non-nesting repeated-feature conditions whose merged interval
// is empty — which preprocessing now correctly REJECTS (validation_v3 finding), so such
// trees are invalid inputs, not test cases.
static constexpr int kMaxGenFeatures = 64;

static Tree RandomTree(std::mt19937& rng, int max_depth, int feature_pool) {
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::uniform_real_distribution<float> leaf(-1.0f, 1.0f);
  std::uniform_int_distribution<int> feat(0, feature_pool - 1);

  Tree t;
  t.nodes.push_back({});
  t.nodes[0].cover = 1000.0;
  if (max_depth == 0) {  // stump
    t.nodes[0].leaf = leaf(rng);
    return t;
  }
  struct Item {
    int node, depth;
    bool spine;  // the forced-deep chain (see below)
    std::array<float, kMaxGenFeatures> lo, hi;  // feasible interval per feature
  };
  Item root{0, 0, true, {}, {}};
  root.lo.fill(-3.0f);
  root.hi.fill(3.0f);
  std::vector<Item> stack{root};
  while (!stack.empty()) {
    Item it = stack.back();
    stack.pop_back();
    const int n = it.node, d = it.depth;
    // The random leaf coin makes depth a branching process with substantial extinction
    // probability (~82% at p=0.45), so "does a deep path exist" would be RNG- and
    // stdlib-dependent. One SPINE (root -> rightmost chain) is therefore exempt from
    // the coin and always reaches max_depth. When the feature pool is at least the
    // requested depth, the spine also requires a fresh feature at every level; this makes
    // post-dedup cooperative length structural rather than a lucky RNG outcome. (The
    // deterministic 32-element boundary is separately pinned by TestComb31.)
    const double leaf_p = max_depth > 12 ? 0.45 : 0.25;
    bool make_leaf =
        d >= max_depth || (d > 0 && !it.spine && unit(rng) < leaf_p);
    int f = -1;
    if (!make_leaf) {  // pick a feature whose feasible interval is still wide enough
      const bool require_fresh = it.spine && feature_pool >= max_depth;
      const auto usable = [&](int cand) {
        const bool fresh = it.lo[cand] == -3.0f && it.hi[cand] == 3.0f;
        return it.hi[cand] - it.lo[cand] > 0.05f && (!require_fresh || fresh);
      };
      for (int tries = 0; tries < 4; tries++) {
        int cand = feat(rng);
        if (usable(cand)) {
          f = cand;
          break;
        }
      }
      // Four random probes are not evidence that no valid split remains. Scan the pool
      // before terminating so a feasible forced spine cannot die by bad sampling.
      if (f < 0) {
        for (int cand = 0; cand < feature_pool; cand++) {
          if (usable(cand)) {
            f = cand;
            break;
          }
        }
      }
      if (f < 0) make_leaf = true;
    }
    if (make_leaf) {
      t.nodes[n].leaf = leaf(rng);
      continue;
    }
    const float span = it.hi[f] - it.lo[f];
    const float threshold =
        it.lo[f] + span * static_cast<float>(0.1 + 0.8 * unit(rng));  // strictly inside
    t.nodes[n].feature = f;
    t.nodes[n].threshold = threshold;
    t.nodes[n].default_left = unit(rng) < 0.5;
    // Occasionally extreme cover splits (1e-4 fraction).
    double frac = (unit(rng) < 0.1) ? 1e-4 : (0.02 + 0.96 * unit(rng));
    const int l = static_cast<int>(t.nodes.size());
    t.nodes.push_back({});
    const int r = static_cast<int>(t.nodes.size());
    t.nodes.push_back({});
    t.nodes[n].left = l;
    t.nodes[n].right = r;
    t.nodes[l].cover = t.nodes[n].cover * frac;
    t.nodes[r].cover = t.nodes[n].cover * (1.0 - frac);
    Item left = it, right = it;
    left.node = l;
    left.depth = d + 1;
    left.spine = false;
    left.hi[f] = threshold;  // left branch: x < threshold
    right.node = r;
    right.depth = d + 1;
    right.spine = it.spine;  // the spine continues down the right chain
    right.lo[f] = threshold;  // right branch: x >= threshold
    stack.push_back(left);
    stack.push_back(right);
  }
  return t;
}

// Mirrors tools/extract_paths.py::_walk_tree for the synthetic trees.
static void ExtractPaths(const Tree& t, int group, uint64_t* next_pid,
                         std::vector<PathElement>* out) {
  const float inf = std::numeric_limits<float>::infinity();
  struct Item {
    int node;
    std::vector<PathElement> acc;
  };
  std::vector<Item> stack{{0, {}}};
  while (!stack.empty()) {
    Item it = std::move(stack.back());
    stack.pop_back();
    const Node& n = t.nodes[it.node];
    if (n.IsLeaf()) {
      const uint64_t pid = (*next_pid)++;
      for (const auto& e : it.acc) {
        PathElement pe = e;
        pe.path_idx = pid;
        pe.v = n.leaf;
        out->push_back(pe);
      }
      PathElement root;
      root.path_idx = pid;
      root.feature_idx = -1;
      root.group = group;
      root.split_condition = XgboostSplitCondition(-inf, inf, true);
      root.zero_fraction = 1.0;
      root.v = n.leaf;
      out->push_back(root);
      continue;
    }
    for (int side = 0; side < 2; side++) {
      const bool is_left = side == 0;
      const int child = is_left ? n.left : n.right;
      PathElement e;
      e.feature_idx = n.feature;
      e.group = group;
      e.split_condition = XgboostSplitCondition(is_left ? -inf : n.threshold,
                                                is_left ? n.threshold : inf,
                                                n.default_left == is_left);
      e.zero_fraction = t.nodes[child].cover / n.cover;
      Item next{child, it.acc};
      next.acc.push_back(e);
      stack.push_back(std::move(next));
    }
  }
}

static float PredictTree(const Tree& t, const float* row) {
  int n = 0;
  while (!t.nodes[n].IsLeaf()) {
    const Node& node = t.nodes[n];
    const float x = row[node.feature];
    bool go_left;
    if (std::isnan(x)) {
      go_left = node.default_left;
    } else {
      go_left = x < node.threshold;
    }
    n = go_left ? node.left : node.right;
  }
  return t.nodes[n].leaf;
}

// Longest deduplicated path group produced by preprocessing — the size of the
// cooperative recurrence actually executed.
static size_t MaxGroupLen(const Preprocessed& pp) {
  size_t best = 0;
  for (const auto& g : CollectGroups(pp)) best = std::max(best, g.end - g.start);
  return best;
}

// Independent exact-Shapley oracle for the comb's deepest (all-right) row, built directly
// from path probabilities. It does NOT use TreeSHAP's extend/unwind recurrence. For each
// target feature i it multiplies the polynomial
//
//     product_{j != i} (z_j + o_j t)
//
// with an ordinary subset-size coefficient DP. Coefficient k is the summed probability
// of all size-k coalitions; multiplying it by k!(D-k-1)!/D! and by (o_i-z_i) is the exact
// Shapley formula for that path. This is intentionally independent of the SIMD recurrence
// under test. It returns the COMPLETE phi vector (per-feature attributions + bias), so
// sum-preserving attribution redistribution cannot hide.
//
// What it establishes at depth 31 (32 distinct-feature elements):
//   * exact-oracle additivity residual ~e-16 -> the analytic SHAP vector is pinned;
//   * float recurrences vs this oracle: max ELEMENTWISE attribution deviation ~9.4e-6;
//   * float accumulated ROW-SUM residual ~1.3e-4 (many signed ~e-6 deviations adding up).
// The e-4 figure is the accumulated additivity residual, not a per-attribution error or
// a measured condition number. The CUDA kernel uses the same float recurrence, so
// similar deep-path sensitivity is a reasonable inference — unmeasured on CUDA so far.
// All of it sits far below the project's 1e-3 gate; typical models (depth <= 10) are ~e-6.
static std::vector<double> Comb31ExactShapleyPhis(const std::vector<double>& frac,
                                                  const std::vector<float>& leaf) {
  const int depth = 31;
  std::vector<double> phis(depth + 1, 0.0);  // features 0..30, bias at [depth]
  for (int p = 0; p <= depth; p++) {
    const int D = (p < depth) ? p + 1 : depth;  // split features on this path
    std::vector<double> z(D), o(D);
    for (int d = 0; d < std::min(p, depth); d++) {
      z[d] = 1.0 - frac[d];
      o[d] = 1.0;  // the all-right row satisfies every spine split
    }
    if (p < depth) {
      z[p] = frac[p];
      o[p] = 0.0;  // ...and fails every left exit
    }
    double pz = 1.0;
    for (double zi : z) pz *= zi;
    phis[depth] += pz * leaf[p];  // bias

    for (int i = 0; i < D; i++) {
      // coeff[k] is the path-probability sum over size-k coalitions that omit i.
      std::vector<double> coeff(D, 0.0);
      coeff[0] = 1.0;
      int seen = 0;
      for (int j = 0; j < D; j++) {
        if (j == i) continue;
        for (int k = seen + 1; k >= 0; k--) {
          const double excluded = coeff[k] * z[j];
          const double included = (k > 0) ? coeff[k - 1] * o[j] : 0.0;
          coeff[k] = excluded + included;
        }
        seen++;
      }

      double weighted = 0.0;
      double choose = 1.0;  // C(D-1, 0)
      for (int k = 0; k < D; k++) {
        // k!(D-k-1)!/D! = 1 / (D * C(D-1,k)).
        weighted += coeff[k] / (double(D) * choose);
        if (k + 1 < D) choose *= double(D - 1 - k) / double(k + 1);
      }
      phis[i] += weighted * (o[i] - z[i]) * leaf[p];
    }
  }
  return phis;
}

// Deterministic comb tree: node at depth d splits on feature d (all distinct), left
// child is a leaf, right child descends. The deepest path is root + 31 distinct
// features = 32 elements — the exact SIMD-width boundary — and MUST survive dedup
// intact, so the full 32-lane cooperative recurrence really executes.
static int TestComb31() {
  const int depth = 31, num_features = 31;
  Tree t;
  t.nodes.push_back({});
  t.nodes[0].cover = 1.0;
  int node = 0;
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> leaf(-1.0f, 1.0f);
  std::vector<double> fracs;               // for the exact-Shapley oracle
  std::vector<float> leaf_vals(depth + 1);  // leaf_vals[p] = leaf of the path exiting at depth p
  for (int d = 0; d < depth; d++) {
    Node& n = t.nodes[node];
    n.feature = d;
    n.threshold = 0.0f;
    n.default_left = (d % 2) == 0;
    const int l = static_cast<int>(t.nodes.size());
    t.nodes.push_back({});
    const int r = static_cast<int>(t.nodes.size());
    t.nodes.push_back({});
    t.nodes[node].left = l;
    t.nodes[node].right = r;
    const double frac = (d % 5 == 0) ? 1e-4 : 0.3;  // include extreme covers
    fracs.push_back(frac);
    t.nodes[l].cover = t.nodes[node].cover * frac;
    t.nodes[r].cover = t.nodes[node].cover * (1.0 - frac);
    t.nodes[l].leaf = leaf(rng);
    leaf_vals[d] = t.nodes[l].leaf;
    node = r;  // continue down the right spine
  }
  t.nodes[node].leaf = leaf(rng);
  leaf_vals[depth] = t.nodes[node].leaf;

  // Exactness at the 32-element boundary: the independent double-precision oracle must
  // satisfy additivity near machine precision.
  const std::vector<double> oracle = Comb31ExactShapleyPhis(fracs, leaf_vals);
  double oracle_sum = 0.0;
  for (double v : oracle) oracle_sum += v;
  const double oracle_resid = std::abs(oracle_sum - double(leaf_vals[depth]));
  if (oracle_resid > 1e-9) {
    std::fprintf(stderr, "comb31 exact-Shapley residual %.3e (oracle error!)\n", oracle_resid);
    return 1;
  }

  std::vector<PathElement> raw;
  uint64_t pid = 0;
  ExtractPaths(t, 0, &pid, &raw);
  auto pp = Preprocess(raw, 1, num_features);
  if (MaxGroupLen(pp) != 32) {
    std::fprintf(stderr, "comb: expected a 32-element cooperative group, got %zu\n",
                 MaxGroupLen(pp));
    return 1;
  }

  // Rows: all-right (hits the deepest leaf), all-left, all-NaN (default routing), random.
  const size_t rows = 32;
  std::mt19937 rng2(11);
  std::uniform_real_distribution<float> val(-3.0f, 3.0f);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::vector<float> X(rows * num_features);
  for (size_t r = 0; r < rows; r++) {
    for (int c = 0; c < num_features; c++) {
      float v = (r == 0) ? 1.0f
                : (r == 1) ? -1.0f
                : (r == 2) ? std::numeric_limits<float>::quiet_NaN()
                           : (unit(rng2) < 0.15
                                  ? std::numeric_limits<float>::quiet_NaN()
                                  : val(rng2));
      X[r * num_features + c] = v;
    }
  }
  DenseDataset ds{X.data(), rows, static_cast<size_t>(num_features)};
  std::vector<double> phis64(rows * (num_features + 1), 0.0);
  std::vector<float> phis32(rows * (num_features + 1), 0.0f);
  ShapReference(ds, pp, 1, phis64.data());
  ShapReference(ds, pp, 1, phis32.data());
  double resid64 = 0.0, resid32 = 0.0;
  for (size_t r = 0; r < rows; r++) {
    const double margin = PredictTree(t, &X[r * num_features]);
    double s64 = 0.0, s32 = 0.0;
    for (int c = 0; c <= num_features; c++) {
      s64 += phis64[r * (num_features + 1) + c];
      s32 += phis32[r * (num_features + 1) + c];
    }
    resid64 = std::max(resid64, std::abs(s64 - margin));
    resid32 = std::max(resid32, std::abs(s32 - margin));
  }
  // ELEMENTWISE attribution comparison for the all-right row (row 0) against the full
  // exact-Shapley oracle: catches sum-preserving redistribution the residual can't see.
  // Measured 9.41e-6 on this fixture; gate 2e-5.
  double elem_dev = 0.0;
  for (int c = 0; c <= num_features; c++) {
    elem_dev = std::max(elem_dev,
                        std::abs(phis64[0 * (num_features + 1) + c] - oracle[c]));
  }

  // Sum tolerance 5e-4: at 32 distinct-feature elements the fp32 recurrences carry ~e-6
  // elementwise deviations whose signed sum accumulates to ~1.3e-4 across the row
  // (identical in fp64-accumulation mode: the deviation lives in the per-path float
  // arithmetic, which the CUDA kernel shares). The exact-Shapley oracle pins the target
  // vector independently near machine precision. See Comb31ExactShapleyPhis and proposal §9.
  const bool ok = resid64 < 5e-4 && resid32 < 5e-4 && elem_dev < 2e-5;
  std::printf("comb31: group_len=32 exact_shapley_resid=%.1e elementwise_vs_oracle=%.2e "
              "additivity fp64=%.2e fp32=%.2e %s\n",
              oracle_resid, elem_dev, resid64, resid32, ok ? "ok" : "FAIL");
  return ok ? 0 : 1;
}

int main() {
  if (TestComb31()) return 1;

  std::mt19937 rng(2026);
  std::uniform_real_distribution<float> val(-3.0f, 3.0f);
  std::uniform_real_distribution<double> unit(0.0, 1.0);

  int failures = 0;
  size_t deep_trial_max_group = 0;
  for (int trial = 0; trial < 12; trial++) {
    const int num_features = 40;
    const int num_trees = 1 + trial * 2;
    std::vector<Tree> forest;
    std::vector<PathElement> raw;
    uint64_t pid = 0;
    for (int i = 0; i < num_trees; i++) {
      // Mix: stumps, shallow repeat-heavy trees (pool 4 exercises dedup/Merge), and in
      // later trials deep trees over the FULL feature space so long deduplicated
      // cooperative paths actually occur (see RandomTree comment).
      const bool deep = trial >= 9 && i == 1;
      int max_depth = (i == 0) ? 0 : deep ? 31 : 1 + (i * 7) % 9;
      forest.push_back(RandomTree(rng, max_depth, deep ? num_features : 4));
      ExtractPaths(forest.back(), 0, &pid, &raw);
    }

    const size_t rows = 64;
    std::vector<float> X(rows * num_features);
    for (auto& x : X) x = (unit(rng) < 0.1) ? std::numeric_limits<float>::quiet_NaN()
                                            : val(rng);

    auto pp = Preprocess(raw, 1, num_features);
    if (trial >= 9) deep_trial_max_group = std::max(deep_trial_max_group, MaxGroupLen(pp));
    DenseDataset ds{X.data(), rows, static_cast<size_t>(num_features)};
    std::vector<double> phis64(rows * (num_features + 1), 0.0);
    std::vector<float> phis32(rows * (num_features + 1), 0.0f);
    const std::vector<double> intercept{0.5};
    ShapReference(ds, pp, 1, phis64.data(), intercept);
    ShapReference(ds, pp, 1, phis32.data(), intercept);

    double max_resid64 = 0.0, max_resid32 = 0.0, max_dev = 0.0;
    for (size_t r = 0; r < rows; r++) {
      double margin = 0.5;
      for (const auto& t : forest) margin += PredictTree(t, &X[r * num_features]);
      double sum64 = 0.0, sum32 = 0.0;
      for (int c = 0; c <= num_features; c++) {
        sum64 += phis64[r * (num_features + 1) + c];
        sum32 += phis32[r * (num_features + 1) + c];
        max_dev = std::max(max_dev,
                           std::abs(phis64[r * (num_features + 1) + c] -
                                    static_cast<double>(phis32[r * (num_features + 1) + c])));
      }
      max_resid64 = std::max(max_resid64, std::abs(sum64 - margin));
      max_resid32 = std::max(max_resid32, std::abs(sum32 - margin));
    }
    // fp32 kernel arithmetic on deep paths: tolerance scales mildly with ensemble size.
    const double tol = 1e-4 * std::max(1, num_trees);
    const bool ok = max_resid64 < tol && max_resid32 < tol;
    std::printf("trial %2d: trees=%2d elements=%6zu max_group=%2zu additivity fp64=%.2e "
                "fp32=%.2e fp32-vs-fp64=%.2e %s\n",
                trial, num_trees, raw.size(), MaxGroupLen(pp), max_resid64, max_resid32,
                max_dev, ok ? "ok" : "FAIL");
    if (!ok) failures++;
  }
  // The deep trials must genuinely produce long cooperative groups post-dedup (the
  // review found the old generator capped them at 5). With a feature pool wider than
  // the requested depth, the forced spine makes 31 structurally distinct splits, so
  // root + features must survive dedup as an exact 32-element group.
  if (deep_trial_max_group != 32) {
    std::fprintf(stderr, "deep trials reached group length %zu (expected 32): generator "
                 "regression\n", deep_trial_max_group);
    failures++;
  }
  std::printf("deep-trial max cooperative group length: %zu\n", deep_trial_max_group);
  if (failures) {
    std::fprintf(stderr, "%d trials FAILED\n", failures);
    return 1;
  }
  std::printf("ALL ADDITIVITY TRIALS OK\n");
  return 0;
}
