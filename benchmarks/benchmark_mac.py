"""Benchmark runner for MetalTreeShap (proposal §7).

Mirrors upstream gputreeshap/benchmark/benchmark.py: same datasets (adult, covtype,
cal_housing, fashion_mnist), same model grid (10/100/1000 rounds x depth 3/8/16), same 10K
explain rows, so results are directly comparable with the published V100 table.

Differences from upstream, per review:
  * adult / fashion_mnist categoricals are ORDINAL-ENCODED to numeric (the extractor does
    not support categorical splits yet). NOTE: ordinal encoding imposes an artificial
    order on nominal categories and therefore changes the attainable splits — the trained
    model is NOT the same model upstream trained with native categorical support, so
    these rows are not directly comparable to the published categorical V100 results.
    The CPU-vs-Metal comparison itself stays fair because both see the identical encoded
    model and data.
  * Timing is phase-separated: model extraction+preprocess (one-time, amortized) is
    reported apart from per-call explain time, and only explain time enters the speedup —
    a compiled-model API is pointless if benchmarks re-measure setup every call.
  * No acceleration numbers are claimed until the Metal path runs: the metal device raises
    until the Phase-1 bindings land.

Energy: run alongside `sudo powermetrics --samplers cpu_power,gpu_power -i 1000` and record
package/GPU watts for the perf-per-watt comparison.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import xgboost as xgb
from sklearn import datasets

MODEL_GRID = {"small": (10, 3), "med": (100, 8), "large": (1000, 16)}


def _encode_numeric(X):
    """Ordinal-encode object/category columns; keep NaN as missing."""
    import pandas as pd
    if not hasattr(X, "iloc"):
        return np.asarray(X, dtype=np.float32)
    X = X.copy()
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == "category":
            X[c] = X[c].astype("category").cat.codes.replace(-1, np.nan)
    return X.astype(np.float32).to_numpy()


def fetch(name: str):
    if name == "cal_housing":
        X, y = datasets.fetch_california_housing(return_X_y=True)
        return np.asarray(X, np.float32), y, "reg:squarederror"
    if name == "covtype":
        X, y = datasets.fetch_covtype(return_X_y=True)
        return np.asarray(X, np.float32), (y - 1), "multi:softmax"
    if name == "adult":
        X, y = datasets.fetch_openml("adult", return_X_y=True)
        return _encode_numeric(X), np.array([yi != "<=50K" for yi in y]), "binary:logistic"
    if name == "fashion_mnist":
        X, y = datasets.fetch_openml("Fashion-MNIST", return_X_y=True)
        return _encode_numeric(X), y.astype(np.int64), "multi:softmax"
    raise ValueError(name)


def bench(dataset: str, size: str, nrows: int, niter: int, device: str) -> dict:
    X, y, objective = fetch(dataset)
    rounds, depth = MODEL_GRID[size]
    params = {"objective": objective, "max_depth": depth, "eta": 0.01, "tree_method": "hist"}
    if objective == "multi:softmax":
        params["num_class"] = int(np.max(y) + 1)
    model = xgb.train(params, xgb.QuantileDMatrix(X, y), rounds)

    rs = np.random.RandomState(432)
    Xt = X[rs.randint(0, X.shape[0], size=nrows)]
    result = {"model": f"{dataset}-{size}", "device": device, "setup_s": 0.0}

    if device == "cpu":
        dtest = xgb.DMatrix(Xt)
        run = lambda: model.predict(dtest, pred_contribs=True)  # noqa: E731
    elif device == "metal":
        # Phase 1-3: compiled-model API — extraction+preprocess+buffer upload once (setup),
        # then repeated explains. Wire the bindings here when they exist:
        #   import metaltreeshap
        #   t0 = time.perf_counter()
        #   compiled = metaltreeshap.compile(model)          # extract + preprocess + buffers
        #   result["setup_s"] = time.perf_counter() - t0
        #   run = lambda: compiled.shap_values(Xt)
        raise NotImplementedError("metal backend lands in Phase 1-3 (see proposal §6)")
    else:
        raise ValueError(device)

    samples, shap_vals = [], None
    for _ in range(niter):
        t0 = time.perf_counter()
        shap_vals = run()
        samples.append(time.perf_counter() - t0)

    margin = model.predict(xgb.DMatrix(Xt), output_margin=True)
    ok = np.allclose(np.sum(shap_vals, axis=-1), margin, 1e-1, 1e-1)
    result.update(mean_s=float(np.mean(samples)), std_s=float(np.std(samples)),
                  rows_per_s=nrows / float(np.mean(samples)), local_accuracy=bool(ok))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="cal_housing,adult")
    ap.add_argument("--sizes", default="small,med")
    ap.add_argument("--nrows", type=int, default=10000)
    ap.add_argument("--niter", type=int, default=5)
    ap.add_argument("--device", default="cpu", choices=["cpu", "metal"])
    args = ap.parse_args()
    for d in args.datasets.split(","):
        for s in args.sizes.split(","):
            print(bench(d, s, args.nrows, args.niter, args.device))
