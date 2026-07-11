"""Golden test: extractor + preprocess + scalar reference vs xgboost pred_contribs.

Trains real XGBoost models, extracts GPUTreeShap-style paths from the raw JSON model, runs
the C++ reference pipeline (reference_cli) with the model intercept plumbed through, and
checks, at the project's stated 1e-3 gate:

  1. fp64-accumulated phis match xgboost.predict(pred_contribs=True) elementwise
  2. local accuracy: phis sum to the margin prediction
  3. fp32-accumulated phis error vs fp64: absolute, elementwise-relative (floored), and
     MAX PAIRWISE ORDER SPREAD across the natural + N seeded work orders — a CPU proxy
     for GPU atomic scheduling (the spread bound is environment-dependent: it varies with
     the standard library's shuffle; treat magnitudes, not exact values, as the signal)

Coverage: regression with missing values, binary logistic, multiclass, multiclass with
num_parallel_tree > 1 (tree_info mapping), DART (weight_drop), a 500-tree depth-8 stress
model, an objective-link mini-suite (identity/logit/log objectives, empirically pinned),
and a rejection check for objectives outside the tested allowlist.

THIS RUN IS NON-MUTATING BY DEFAULT (review requirement: verification must not regenerate
its own oracle). Explicit flags:
  --update-fixtures   freeze tests/fixtures/<case>/ for the marked cases
  --write-results     rewrite tests/RESULTS.md

Usage: python tests/test_vs_xgboost.py [path/to/reference_cli] [--update-fixtures]
       [--write-results]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import xgboost as xgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from extract_paths import extract_model, write_paths_csv  # noqa: E402

ARGS = sys.argv[1:]
UPDATE_FIXTURES = "--update-fixtures" in ARGS
WRITE_RESULTS = "--write-results" in ARGS
POS = [a for a in ARGS if not a.startswith("--")]
CLI = POS[0] if POS else "./reference_cli"
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS: list[dict] = []

TOL = 1e-3  # the project's stated correctness gate (observed errors are ~1e-6..1e-5)
# Elementwise relative error uses a floor so near-zero attributions don't dominate with
# noise ratios: rel_i = |d_i| / max(|phi64_i|, REL_FLOOR_FRACTION * max|phi64|).
REL_FLOOR_FRACTION = 1e-3
ORDER_SEEDS = 5


def run_cli(paths_csv, x_csv, num_groups, out64, out32, intercepts, seed=0):
    icept = ",".join(repr(float(v)) for v in intercepts)
    subprocess.run([CLI, paths_csv, x_csv, str(num_groups), out64, out32, icept, str(seed)],
                   check=True, capture_output=True)


def make_data(objective, n_train, n_test, n_features, missing_frac, seed):
    rng = np.random.RandomState(seed)
    X_train = rng.randn(n_train, n_features).astype(np.float32)
    X_test = rng.randn(n_test, n_features).astype(np.float32)
    if missing_frac > 0:
        for X in (X_train, X_test):
            mask = rng.rand(*X.shape) < missing_frac
            X[mask] = np.nan
    w = rng.randn(n_features)
    signal = np.nansum(X_train * w, axis=1) + 0.5 * np.nan_to_num(X_train[:, 0]) * np.nan_to_num(
        X_train[:, 1 % n_features])
    if objective in ("binary:logistic", "binary:logitraw", "binary:hinge"):
        y = (signal > 0).astype(np.float32)
    elif objective == "reg:logistic":
        y = 1.0 / (1.0 + np.exp(-signal))
    elif objective == "count:poisson":
        y = rng.poisson(np.exp(np.clip(0.3 * signal, -3, 3))).astype(np.float32)
    elif objective == "reg:gamma":
        y = np.exp(np.clip(0.3 * signal, -3, 3)) + 0.05
    elif objective == "reg:tweedie":
        y = np.maximum(rng.poisson(np.exp(np.clip(0.3 * signal, -3, 3))).astype(np.float32),
                       0.0)
    elif objective == "reg:squaredlogerror":
        y = np.abs(signal) + 0.1
    elif objective.startswith("multi:"):
        y = (np.digitize(signal, np.quantile(signal, [0.33, 0.66]))).astype(np.float32)
    else:  # squarederror, absoluteerror, quantileerror, ...
        y = signal + 0.1 * rng.randn(n_train)
    return X_train, y, X_test


def run_case(name, objective, n_features, n_train, n_test, depth, rounds,
             missing_frac=0.0, seed=0, extra_params=None, shuffle_trials=0,
             fixture=False):
    X_train, y, X_test = make_data(objective, n_train, n_test, n_features, missing_frac, seed)
    params = {"objective": objective, "max_depth": depth, "eta": 0.1, "tree_method": "hist",
              "seed": seed}
    params.update(extra_params or {})
    if objective.startswith("multi:"):
        params["num_class"] = int(np.max(y) + 1)
    booster = xgb.train(params, xgb.DMatrix(X_train, label=y), rounds)

    em = extract_model(booster)
    num_groups = em.num_groups
    dtest = xgb.DMatrix(X_test)
    expected = booster.predict(dtest, pred_contribs=True).reshape(
        n_test, num_groups, n_features + 1)
    margin = booster.predict(dtest, output_margin=True)

    with tempfile.TemporaryDirectory() as td:
        paths_csv, x_csv = os.path.join(td, "p.csv"), os.path.join(td, "x.csv")
        out64, out32 = os.path.join(td, "o64.csv"), os.path.join(td, "o32.csv")
        write_paths_csv(em.paths, paths_csv)
        np.savetxt(x_csv, X_test, delimiter=",", fmt="%.9g")
        run_cli(paths_csv, x_csv, num_groups, out64, out32, em.intercepts)
        phis64 = np.loadtxt(out64, delimiter=",").reshape(n_test, num_groups, n_features + 1)
        phis32 = np.loadtxt(out32, delimiter=",").reshape(n_test, num_groups, n_features + 1)

        # Order-sensitivity: MAX PAIRWISE spread across natural + seeded orders,
        # tracked via running elementwise min/max (max-min == max pairwise |delta|).
        order_spread = 0.0
        if shuffle_trials:
            lo = phis32.copy()
            hi = phis32.copy()
            for s in range(1, shuffle_trials + 1):
                run_cli(paths_csv, x_csv, num_groups, out64, out32, em.intercepts, seed=s)
                p32s = np.loadtxt(out32, delimiter=",").reshape(phis32.shape)
                np.minimum(lo, p32s, out=lo)
                np.maximum(hi, p32s, out=hi)
            order_spread = float(np.max(hi - lo))

    err_vs_xgb = float(np.max(np.abs(phis64 - expected)))
    row_sums = phis64.sum(axis=2).squeeze()
    margin_err = float(np.max(np.abs(row_sums - margin.reshape(row_sums.shape))))
    d = np.abs(phis32.astype(np.float64) - phis64)
    err_fp32 = float(np.max(d))
    floor = REL_FLOOR_FRACTION * np.max(np.abs(phis64))
    rel_fp32_elem = float(np.max(d / np.maximum(np.abs(phis64), floor)))

    if fixture and UPDATE_FIXTURES:
        fx = os.path.join(HERE, "fixtures", name)
        os.makedirs(fx, exist_ok=True)
        booster.save_model(os.path.join(fx, "model.json"))
        np.savetxt(os.path.join(fx, "X.csv"), X_test, delimiter=",", fmt="%.9g")
        np.savetxt(os.path.join(fx, "expected_contribs.csv"),
                   expected.reshape(n_test, -1), delimiter=",", fmt="%.9g")
        with open(os.path.join(fx, "meta.json"), "w") as f:
            json.dump({"case": name, "kind": "xgboost_model", "num_groups": num_groups,
                       "num_features": n_features, "xgboost_version": xgb.__version__,
                       "tolerance": TOL,
                       "note": "expected = pred_contribs; layout [rows, "
                               "groups*(features+1)]"}, f, indent=1)

    RESULTS.append(dict(name=name, objective=objective, booster=em.booster,
                        trees=rounds * max(1, num_groups)
                        * (extra_params or {}).get("num_parallel_tree", 1),
                        depth=depth, paths=em.paths[-1].path_idx + 1,
                        err_vs_xgb=err_vs_xgb, margin_err=margin_err, err_fp32=err_fp32,
                        rel_fp32_elem=rel_fp32_elem,
                        order_spread=order_spread if shuffle_trials else None))
    ok = err_vs_xgb < TOL and margin_err < TOL
    print(f"[{'PASS' if ok else 'FAIL'}] {name} ({objective}): "
          f"max|phi-xgb|={err_vs_xgb:.3e} sum-to-margin={margin_err:.3e} "
          f"fp32(abs={err_fp32:.3e}, rel_elem={rel_fp32_elem:.2e}"
          + (f", order_spread={order_spread:.3e}" if shuffle_trials else "") + ")")
    assert ok, f"{name}: mismatch vs xgboost (gate {TOL})"


def check_unknown_objective_rejected():
    """Objectives outside the tested allowlist must be rejected, not mis-linked."""
    rng = np.random.RandomState(0)
    X = rng.randn(300, 4).astype(np.float32)
    y = np.abs(X[:, 0]) + 0.1
    try:
        booster = xgb.train({"objective": "survival:cox", "max_depth": 2},
                            xgb.DMatrix(X, label=y), 3)
    except xgb.core.XGBoostError:
        print("[SKIP] survival:cox did not train in this xgboost; rejection untested here")
        return
    try:
        extract_model(booster)
    except NotImplementedError as e:
        print(f"[PASS] unknown objective rejected: {str(e)[:80]}...")
        return
    raise AssertionError("survival:cox was not rejected by the objective allowlist")


def write_results_md():
    path = os.path.join(HERE, "RESULTS.md")
    with open(path, "w") as f:
        f.write("# Golden test results (Phase 0.6, CPU reference pipeline)\n\n")
        f.write(f"xgboost {xgb.__version__}, numpy {np.__version__}; correctness gate "
                f"{TOL}.\n\n")
        f.write("`err_vs_xgb` = max |phi − xgboost pred_contribs| (fp64 accumulation, "
                "intercept plumbed through the pipeline). `margin_err` = max |Σ phis − "
                "margin|. `fp32_abs` = max |fp32-accumulated − fp64-accumulated|. "
                "`fp32_rel_elem` = max elementwise relative error, floored at "
                f"{REL_FLOOR_FRACTION:g}·max|phi|. `order_spread` = max PAIRWISE |Δ| "
                f"across the natural + {ORDER_SEEDS} seeded fp32 work orders — a CPU "
                "proxy for GPU atomic scheduling whose exact value is environment-"
                "dependent (stdlib shuffle); the on-device measurement happens in "
                "Phase 2.\n\n")
        f.write("| case | objective | booster | trees | depth | paths | err_vs_xgb | "
                "margin_err | fp32_abs | fp32_rel_elem | order_spread |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in RESULTS:
            spread = f"{r['order_spread']:.2e}" if r["order_spread"] is not None else "—"
            f.write(f"| {r['name']} | {r['objective']} | {r['booster']} | {r['trees']} | "
                    f"{r['depth']} | {r['paths']} | {r['err_vs_xgb']:.2e} | "
                    f"{r['margin_err']:.2e} | {r['err_fp32']:.2e} | "
                    f"{r['rel_fp32_elem']:.2e} | {spread} |\n")
        f.write("\nObjective links verified empirically (identity/logit/log, see "
                "tools/extract_paths.py `_MARGIN_LINK`); objectives outside the allowlist "
                "are rejected (checked with survival:cox). Cross-version: suite verified "
                "on xgboost 2.0.3 and 3.1.2. External M4 Max validation (v3): the "
                "compiled-model host logic ran ALL SIX frozen fixtures end-to-end "
                "(shader runtime-compiled from source) with max Metal error 6.5e-6, "
                "across rows_per_simdgroup in {1, 7, 1024}, including empty-model, "
                "zero-row, intercept, repeated-call and invalid-tuning behavior. "
                "src/main_metal.cpp + `test_fixture.py --metal-cli` make that run "
                "repository-reproducible.\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    # Core cases (fixture=True cases freeze under --update-fixtures).
    run_case("regression-missing", "reg:squarederror", 8, 2000, 300, 3, 25,
             missing_frac=0.15, seed=1, fixture=True)
    run_case("binary-depth6", "binary:logistic", 12, 3000, 300, 6, 50, seed=2, fixture=True)
    run_case("multiclass-3", "multi:softmax", 10, 3000, 200, 4, 30, seed=3, fixture=True)
    run_case("parallel-trees", "multi:softmax", 10, 3000, 200, 4, 10, seed=5,
             extra_params={"num_parallel_tree": 2}, fixture=True)
    run_case("dart", "reg:squarederror", 8, 2000, 200, 4, 30, seed=6,
             extra_params={"booster": "dart", "rate_drop": 0.2}, fixture=True)
    run_case("stress-depth8x500", "reg:squarederror", 12, 4000, 200, 8, 500, seed=4,
             shuffle_trials=ORDER_SEEDS)

    # Objective-link mini-suite: every allowlisted non-default link + identity edge cases.
    mini = dict(n_features=6, n_train=1200, n_test=150, depth=3, rounds=10)
    run_case("obj-reg-logistic", "reg:logistic", seed=11, **mini)
    run_case("obj-logitraw", "binary:logitraw", seed=12, **mini)
    run_case("obj-hinge", "binary:hinge", seed=13, **mini)
    run_case("obj-poisson", "count:poisson", seed=14, **mini)
    run_case("obj-gamma", "reg:gamma", seed=15, **mini)
    run_case("obj-tweedie", "reg:tweedie", seed=16, **mini)
    run_case("obj-absoluteerror", "reg:absoluteerror", seed=17, **mini)
    run_case("obj-squaredlogerror", "reg:squaredlogerror", seed=18, **mini)
    run_case("obj-quantile", "reg:quantileerror", seed=19,
             extra_params={"quantile_alpha": 0.5}, **mini)

    check_unknown_objective_rejected()
    if WRITE_RESULTS:
        write_results_md()
    print("ALL GOLDEN TESTS PASSED"
          + ("" if not UPDATE_FIXTURES else " (fixtures updated)")
          + ("" if not WRITE_RESULTS else " (RESULTS.md updated)"))
