#!/usr/bin/env python3
"""Benchmark XGBoost's CPU TreeSHAP on a frozen Phase-2 workload.

The Booster and each DMatrix are constructed outside the measured loop, matching the
compiled-model Metal benchmark.  Each timed call still includes XGBoost's normal output
allocation.  JSON is written atomically so interrupted runs cannot masquerade as results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import xgboost as xgb


SCHEMA = "metal_treeshap.phase2.cpu_xgboost.v1"


def _positive_csv(value: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("row limits must be comma-separated integers") from error
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("row limits must be positive")
    return parsed


def _quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method="linear"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-limits", type=_positive_csv,
                        default=_positive_csv("1,8,32,128,512,2048,8192"))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--nthread", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.nthread <= 0:
        raise SystemExit("warmup must be nonnegative; iterations and nthread must be positive")

    load_start = time.perf_counter()
    matrix = np.loadtxt(args.matrix, delimiter=",", dtype=np.float32, ndmin=2)
    expected = (np.loadtxt(args.expected, delimiter=",", dtype=np.float32, ndmin=2)
                if args.expected else None)
    booster = xgb.Booster()
    booster.load_model(args.model)
    booster.set_param({"device": "cpu", "nthread": args.nthread})
    load_s = time.perf_counter() - load_start

    limits = []
    for value in args.row_limits:
        effective = min(value, matrix.shape[0])
        if effective not in limits:
            limits.append(effective)

    results = []
    for rows in limits:
        setup_start = time.perf_counter()
        dmatrix = xgb.DMatrix(matrix[:rows])
        dmatrix_s = time.perf_counter() - setup_start

        prediction = None
        for _ in range(args.warmup):
            prediction = booster.predict(dmatrix, pred_contribs=True)

        samples = []
        hashes = []
        for _ in range(args.iterations):
            begin = time.perf_counter()
            prediction = booster.predict(dmatrix, pred_contribs=True)
            samples.append(time.perf_counter() - begin)
            hashes.append(_hash_array(prediction))
        assert prediction is not None

        flat = np.asarray(prediction, dtype=np.float32).reshape(rows, -1)
        accuracy = None
        if expected is not None:
            if expected.shape[0] < rows or expected.shape[1] != flat.shape[1]:
                raise SystemExit("expected attribution shape does not match prediction")
            delta = np.abs(flat - expected[:rows])
            accuracy = {
                "max_abs": float(delta.max(initial=0.0)),
                "mean_abs": float(delta.mean()) if delta.size else 0.0,
            }
        median = _quantile(samples, 0.5)
        results.append({
            "rows": rows,
            "dmatrix_setup_s": dmatrix_s,
            "timing_s": {
                "median": median,
                "p10": _quantile(samples, 0.1),
                "p90": _quantile(samples, 0.9),
                "samples": samples,
            },
            "throughput_rows_per_s": rows / median,
            "repeatability": {
                "hash_algorithm": "sha256-array-bytes",
                "unique_hashes": len(set(hashes)),
                "hashes": hashes,
            },
            "accuracy": accuracy,
        })

    artifact = {
        "schema": SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "xgboost": xgb.__version__,
            "xgboost_build_info": xgb.build_info(),
            "nthread": args.nthread,
        },
        "inputs": {
            "model": str(args.model.resolve()),
            "matrix": str(args.matrix.resolve()),
            "expected": str(args.expected.resolve()) if args.expected else None,
            "sha256": {
                "model": _sha256(args.model),
                "matrix": _sha256(args.matrix),
                "expected": _sha256(args.expected) if args.expected else None,
            },
            "available_rows": int(matrix.shape[0]),
            "cols": int(matrix.shape[1]),
        },
        "design": {
            "warmups": args.warmup,
            "iterations": args.iterations,
            "setup_excluded": ["model_load", "DMatrix construction"],
            "timed_call": "Booster.predict(DMatrix, pred_contribs=True)",
        },
        "setup_s": {"load_model_matrix_expected": load_s},
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, delete=False,
                                     prefix=args.output.name + ".", suffix=".tmp") as temp:
        json.dump(artifact, temp, indent=2, sort_keys=True)
        temp.write("\n")
        temporary = Path(temp.name)
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
