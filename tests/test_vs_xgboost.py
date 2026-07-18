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
model, XGBoost 3.3's flattened DART layout when available, an objective-link mini-suite
(identity/logit/log objectives, empirically pinned), a +/-inf routing check (pinned
against a finite sentinel because DMatrix rejects infinities), and a rejection check for
objectives outside the tested allowlist.

THIS RUN IS NON-MUTATING BY DEFAULT (review requirement: verification must not regenerate
its own oracle). Explicit flags:
  --update-fixtures   freeze tests/fixtures/<case>/ for the marked cases
  --write-results     rewrite tests/RESULTS.md

Usage: python tests/test_vs_xgboost.py [path/to/reference_cli] [--update-fixtures]
       [--write-results]
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
import xgboost as xgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from extract_paths import extract_model, write_paths_csv  # noqa: E402

ARGS = sys.argv[1:]
KNOWN_FLAGS = {"--update-fixtures", "--write-results"}
unknown_flags = [a for a in ARGS if a.startswith("--") and a not in KNOWN_FLAGS]
if unknown_flags:
    raise SystemExit(f"unknown option(s): {', '.join(unknown_flags)}")
UPDATE_FIXTURES = "--update-fixtures" in ARGS
WRITE_RESULTS = "--write-results" in ARGS
POS = [a for a in ARGS if not a.startswith("--")]
if len(POS) > 1:
    raise SystemExit(f"unexpected positional arguments: {POS[1:]}")
CLI = POS[0] if POS else "./reference_cli"
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS: list[dict] = []

TOL = 1e-3  # the project's stated correctness gate (observed errors are ~1e-6..1e-5)
# Elementwise relative error uses a floor so near-zero attributions don't dominate with
# noise ratios: rel_i = |d_i| / max(|phi64_i|, REL_FLOOR_FRACTION * max|phi64|).
REL_FLOOR_FRACTION = 1e-3
ORDER_SEEDS = 5


def xgboost_version() -> tuple[int, ...]:
    """Numeric release prefix, ignoring development/local suffixes."""
    out = []
    for part in xgb.__version__.split("."):
        digits = "".join(c for c in part if c.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)

EVIDENCE_SOURCES = (
    os.path.join(HERE, "test_vs_xgboost.py"),
    os.path.join(HERE, "..", "tools", "extract_paths.py"),
    os.path.join(HERE, "..", "include", "metal_treeshap", "paths.h"),
    os.path.join(HERE, "..", "include", "metal_treeshap", "preprocess.h"),
    os.path.join(HERE, "..", "reference", "reference_shap.h"),
    os.path.join(HERE, "..", "src", "main_reference.cpp"),
)


def run_cli(paths_csv, x_csv, num_groups, out64, out32, intercepts, seed=0):
    icept = ",".join(repr(float(v)) for v in intercepts)
    subprocess.run([CLI, paths_csv, x_csv, str(num_groups), out64, out32, icept, str(seed)],
                   check=True, capture_output=True)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_fingerprint() -> str:
    """Hash the exact portable implementation/test sources used by this invocation."""
    h = hashlib.sha256()
    root = os.path.realpath(os.path.join(HERE, ".."))
    for path in sorted(map(os.path.realpath, EVIDENCE_SOURCES)):
        h.update(os.path.relpath(path, root).encode("utf-8"))
        h.update(b"\0")
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


def git_provenance() -> tuple[str, bool]:
    """Best-effort source-repository state; the content hash above remains authoritative."""
    root = os.path.realpath(os.path.join(HERE, ".."))
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root, text=True).strip())
        return head, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", True


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


def has_missing_only_path(paths) -> bool:
    """Whether deduplication yields a positive-cover condition satisfied only by NaN."""
    merged: dict[tuple[int, int], list] = {}
    for e in paths:
        if e.feature_idx < 0:
            continue
        key = (e.path_idx, e.feature_idx)
        if key not in merged:
            merged[key] = [e.lower, e.upper, e.is_missing_branch, e.zero_fraction]
        else:
            m = merged[key]
            m[0] = max(m[0], e.lower)
            m[1] = min(m[1], e.upper)
            m[2] = m[2] and e.is_missing_branch
            m[3] *= e.zero_fraction
    return any(lo >= hi and missing and cover > 0.0
               for lo, hi, missing, cover in merged.values())


def run_case(name, objective, n_features, n_train, n_test, depth, rounds,
             missing_frac=0.0, seed=0, extra_params=None, shuffle_trials=0,
             fixture=False, data_override=None, require_missing_only_path=False):
    if data_override is None:
        X_train, y, X_test = make_data(
            objective, n_train, n_test, n_features, missing_frac, seed)
    else:
        X_train, y, X_test = data_override
        if X_train.shape != (n_train, n_features) or X_test.shape != (n_test, n_features):
            raise ValueError(f"{name}: data_override shapes do not match declared dimensions")
        if y.shape != (n_train,):
            raise ValueError(f"{name}: data_override labels do not match n_train")
    params = {"objective": objective, "max_depth": depth, "eta": 0.1, "tree_method": "hist",
              "seed": seed}
    params.update(extra_params or {})
    if objective.startswith("multi:"):
        params["num_class"] = int(np.max(y) + 1)
    booster = xgb.train(params, xgb.DMatrix(X_train, label=y), rounds)

    em = extract_model(booster)
    if require_missing_only_path:
        assert has_missing_only_path(em.paths), (
            f"{name}: trained model no longer contains the missing-only repeated-feature "
            "path this regression case is intended to pin")
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
    # err_fp32 and order_spread were previously computed and recorded but never gated:
    # a regression confined to fp32 accumulation (or an exploding work-order spread)
    # would have printed PASS. The 1e-3 gate is generous for both (observed ~1e-5).
    ok = (err_vs_xgb < TOL and margin_err < TOL and err_fp32 < TOL
          and (not shuffle_trials or order_spread < TOL))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} ({objective}): "
          f"max|phi-xgb|={err_vs_xgb:.3e} sum-to-margin={margin_err:.3e} "
          f"fp32(abs={err_fp32:.3e}, rel_elem={rel_fp32_elem:.2e}"
          + (f", order_spread={order_spread:.3e}" if shuffle_trials else "") + ")")
    assert ok, f"{name}: mismatch vs xgboost (gate {TOL})"


def check_infinity_routing():
    """+/-inf feature values must follow the branch XGBoost takes for any value beyond
    every finite threshold (fvalue < t false -> the [t, +inf) child, and vice versa).
    xgboost's DMatrix rejects non-NaN infinities outright, so equivalence is pinned
    against a huge finite sentinel that takes the same branch at every finite split."""
    sentinel = np.float32(3.0e38)
    X_train, y, X_test = make_data("reg:squarederror", 1500, 60, 6, 0.1, seed=21)
    rng = np.random.RandomState(22)
    pos = rng.rand(*X_test.shape) < 0.15
    neg = ~pos & (rng.rand(*X_test.shape) < 0.15)
    X_inf, X_sent = X_test.copy(), X_test.copy()
    X_inf[pos], X_inf[neg] = np.inf, -np.inf
    X_sent[pos], X_sent[neg] = sentinel, -sentinel
    booster = xgb.train({"objective": "reg:squarederror", "max_depth": 4, "eta": 0.1,
                         "tree_method": "hist", "seed": 21},
                        xgb.DMatrix(X_train, label=y), 30)
    em = extract_model(booster)
    max_finite_bound = max(abs(b) for e in em.paths for b in (e.lower, e.upper)
                           if np.isfinite(b))
    assert max_finite_bound < float(sentinel), \
        "sentinel does not dominate every finite threshold"
    dsent = xgb.DMatrix(X_sent)
    expected = booster.predict(dsent, pred_contribs=True)
    margin = booster.predict(dsent, output_margin=True)
    with tempfile.TemporaryDirectory() as td:
        paths_csv, x_csv = os.path.join(td, "p.csv"), os.path.join(td, "x.csv")
        out64, out32 = os.path.join(td, "o64.csv"), os.path.join(td, "o32.csv")
        write_paths_csv(em.paths, paths_csv)
        np.savetxt(x_csv, X_inf, delimiter=",", fmt="%.9g")
        run_cli(paths_csv, x_csv, 1, out64, out32, em.intercepts)
        phis64 = np.loadtxt(out64, delimiter=",")
    err = float(np.max(np.abs(phis64 - expected)))
    margin_err = float(np.max(np.abs(phis64.sum(axis=1) - margin)))
    ok = err < TOL and margin_err < TOL
    print(f"[{'PASS' if ok else 'FAIL'}] infinity-routing (+/-inf vs finite sentinel): "
          f"max|phi-xgb|={err:.3e} sum-to-margin={margin_err:.3e}")
    assert ok, f"infinity-routing: mismatch vs xgboost sentinel (gate {TOL})"


def check_unknown_objective_rejected() -> dict[str, str]:
    """Objectives outside the tested allowlist must be rejected, not mis-linked."""
    rng = np.random.RandomState(0)
    X = rng.randn(300, 4).astype(np.float32)
    y = np.abs(X[:, 0]) + 0.1
    try:
        booster = xgb.train({"objective": "survival:cox", "max_depth": 2},
                            xgb.DMatrix(X, label=y), 3)
    except xgb.core.XGBoostError:
        print("[SKIP] survival:cox did not train in this xgboost; rejection untested here")
        return {"status": "skipped", "detail": "survival:cox did not train"}
    try:
        extract_model(booster)
    except NotImplementedError as e:
        print(f"[PASS] unknown objective rejected: {str(e)[:80]}...")
        return {"status": "passed", "detail": "survival:cox rejected by extractor"}
    raise AssertionError("survival:cox was not rejected by the objective allowlist")


def write_results_md(objective_rejection: dict[str, str]):
    path = os.path.join(HERE, "RESULTS.md")
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    invocation = shlex.join([sys.executable, os.path.abspath(__file__), *sys.argv[1:]])
    git_head, git_dirty = git_provenance()
    source_sha = source_fingerprint()
    cli_path = os.path.realpath(CLI)
    cli_sha = sha256_file(cli_path)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Golden test results (Phase 0.6, CPU reference pipeline)\n\n")
        f.write("This file contains evidence produced by one local invocation only; it does "
                "not infer results for other XGBoost versions, machines, or the Metal engine.\n\n")
        f.write(f"- Generated (UTC): `{generated_at}`\n"
                f"- Platform: `{platform.platform()}`\n"
                f"- Python: `{platform.python_version()}`\n"
                f"- xgboost: `{xgb.__version__}`\n"
                f"- numpy: `{np.__version__}`\n"
                f"- Git HEAD: `{git_head}` (worktree dirty: `{str(git_dirty).lower()}`)\n"
                f"- Portable source fingerprint (SHA-256): `{source_sha}`\n"
                f"- Reference CLI: `{cli_path}`\n"
                f"- Reference CLI SHA-256: `{cli_sha}`\n"
                f"- Invocation: `{invocation}`\n"
                f"- Correctness gate: `{TOL}`\n\n")
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
        f.write("\nThis invocation exercised every identity/logit/log objective case listed "
                "above against XGBoost `pred_contribs`; see `tools/extract_paths.py` "
                "`_MARGIN_LINK`. Unsupported-objective check: "
                f"**{objective_rejection['status']}** "
                f"({objective_rejection['detail']}).\n")
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
    if xgboost_version() >= (3, 3):
        # In 3.3+, dropout lives directly on the tree booster.  The JSON still says
        # name=gbtree, with weight_drop adjacent to model; this is distinct from the
        # <=3.2 wrapper layout exercised by the same "dart" case on older releases.
        run_case("dart-xgb33", "reg:squarederror", 6, 600, 64, 3, 12, seed=33,
                 extra_params={"rate_drop": 0.25, "skip_drop": 0.0, "one_drop": 1},
                 fixture=True)
    else:
        print(f"[SKIP] dart-xgb33 requires xgboost >= 3.3 (found {xgb.__version__})")
    # Deterministic real-model regression for XGBoost's missing-only path representation.
    # The root and child split f0 at the same threshold with opposite numeric branches,
    # while both route NaN toward the -3 leaf.  Dedup therefore yields [1,1), missing=true.
    missing_X = np.array([0.0] * 5 + [1.0] * 5 + [np.nan] * 5,
                         dtype=np.float32).reshape(-1, 1)
    missing_y = np.array([-10.0] * 5 + [10.0] * 5 + [-3.0] * 5, dtype=np.float32)
    missing_test = np.array([[0.0], [1.0], [np.nan], [0.5], [2.0]], dtype=np.float32)
    run_case("missing-only-path", "reg:squarederror", 1, 15, 5, 2, 1, seed=0,
             extra_params={"eta": 1.0, "min_child_weight": 0.0, "lambda": 0.0,
                           "gamma": 0.0, "base_score": 0.0, "nthread": 1},
             fixture=True, data_override=(missing_X, missing_y, missing_test),
             require_missing_only_path=True)
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
    # Every allowlisted objective must actually be trained by this suite: these two were
    # in tools/extract_paths._MARGIN_LINK but previously untested end-to-end.
    run_case("obj-pseudohuber", "reg:pseudohubererror", seed=20, **mini)
    run_case("obj-softprob", "multi:softprob", seed=22, **mini)

    check_infinity_routing()
    objective_rejection = check_unknown_objective_rejected()
    if WRITE_RESULTS:
        write_results_md(objective_rejection)
    print("ALL GOLDEN TESTS PASSED"
          + ("" if not UPDATE_FIXTURES else " (fixtures updated)")
          + ("" if not WRITE_RESULTS else " (RESULTS.md updated)"))
