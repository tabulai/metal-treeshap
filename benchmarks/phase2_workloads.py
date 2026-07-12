#!/usr/bin/env python3
"""Create deterministic Phase-2 benchmark workloads outside the frozen test oracles.

Three workload families are supported:

* ``hot`` creates many stumps that all update one feature/output cell.  It is a focused
  contention workload for comparing per-lane atomics with SIMD-group aggregation and has
  an analytic expected attribution file.
* ``fixture`` materializes any frozen/raw/model fixture into a separate directory, with
  optional deterministic row tiling.  Source fixtures are opened read-only and never
  modified.  Directories such as the generated stress500 dataset are accepted too.
* ``stress`` trains and freezes a deterministic 500-tree depth-8 XGBoost regression
  workload, including model, extracted paths, data, oracle output, and provenance hashes.

Every output has a workload.json consumed by phase2_run.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from extract_paths import extract_model, write_paths_csv  # noqa: E402

SCHEMA = "metal_treeshap.phase2.workload.v1"


def _prepare_output(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not force:
            raise SystemExit(f"output directory is non-empty: {path} (pass --force)")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(output: Path, manifest: dict) -> None:
    files = ["paths.csv", "X.csv"]
    if (output / "expected.csv").exists():
        files.append("expected.csv")
    if manifest.get("model") and (output / manifest["model"]).exists():
        files.append(manifest["model"])
    manifest.update(
        schema=SCHEMA,
        paths="paths.csv",
        matrix="X.csv",
        expected="expected.csv" if "expected.csv" in files else None,
        sha256={name: _sha256(output / name) for name in files},
    )
    with (output / "workload.json").open("w") as target:
        json.dump(manifest, target, indent=2, sort_keys=True)
        target.write("\n")


def generate_hot(args: argparse.Namespace) -> None:
    if args.trees <= 0 or args.rows <= 0:
        raise SystemExit("--trees and --rows must be positive")
    if not math.isfinite(args.leaf_scale) or args.leaf_scale <= 0:
        raise SystemExit("--leaf-scale must be finite and positive")
    output = Path(args.output).resolve()
    _prepare_output(output, args.force)

    paths = output / "paths.csv"
    with paths.open("w", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(("path_idx", "feature_idx", "group", "lower", "upper",
                         "is_missing", "zero_fraction", "v"))
        path_idx = 0
        for _ in range(args.trees):
            for lower, upper, missing, leaf in (
                ("-inf", "0", 1, -args.leaf_scale),
                ("0", "inf", 0, args.leaf_scale),
            ):
                writer.writerow((path_idx, 0, 0, lower, upper, missing, 0.5, repr(leaf)))
                writer.writerow((path_idx, -1, 0, "-inf", "inf", 1, 1.0, repr(leaf)))
                path_idx += 1

    rng = random.Random(args.seed)
    values: list[float | None] = []
    for row in range(args.rows):
        # Guarantee all three routing cases, then use a deterministic balanced stream.
        if row % 257 == 0:
            values.append(None)
        else:
            values.append(rng.uniform(-2.0, 2.0))
    with (output / "X.csv").open("w") as matrix, \
         (output / "expected.csv").open("w") as expected:
        magnitude = args.trees * args.leaf_scale
        for value in values:
            matrix.write("nan\n" if value is None else f"{value:.9g}\n")
            phi = -magnitude if value is None or value < 0 else magnitude
            expected.write(f"{phi:.12g},0\n")

    _write_manifest(
        output,
        {
            "name": args.name,
            "kind": "synthetic_hot_feature_stumps",
            "rows": args.rows,
            "cols": 1,
            "num_groups": 1,
            "intercepts": [0.0],
            "raw_path_elements": args.trees * 4,
            "trees": args.trees,
            "seed": args.seed,
            "leaf_scale": args.leaf_scale,
            "tolerance": 1e-3,
            "expected_atomic_reduction_per_full_bin": 16,
        },
    )
    print(output / "workload.json")


def _read_nonempty_rows(path: Path) -> list[list[str]]:
    with path.open(newline="") as source:
        rows = [row for row in csv.reader(source) if row]
    if not rows:
        raise SystemExit(f"empty CSV: {path}")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise SystemExit(f"ragged CSV: {path}")
    return rows


def _write_tiled(source: Path, target: Path, rows: int) -> tuple[int, int]:
    values = _read_nonempty_rows(source)
    count = rows or len(values)
    if count <= 0:
        raise SystemExit("--rows must be positive when supplied")
    with target.open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        for index in range(count):
            writer.writerow(values[index % len(values)])
    return count, len(values[0])


def materialize_fixture(args: argparse.Namespace) -> None:
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    if not source.is_dir():
        raise SystemExit(f"fixture directory does not exist: {source}")
    if output == source:
        raise SystemExit("output must differ from the source fixture")
    _prepare_output(output, args.force)

    meta_path = source / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    model_path = source / "model.json"
    source_paths = source / "paths.csv"
    if source_paths.exists():
        shutil.copyfile(source_paths, output / "paths.csv")
        num_groups = int(meta.get("num_groups", meta.get("groups", 1)))
        intercepts = [float(value) for value in meta.get("intercepts", [0.0] * num_groups)]
    elif model_path.exists():
        extracted = extract_model(str(model_path))
        write_paths_csv(extracted.paths, str(output / "paths.csv"))
        num_groups = extracted.num_groups
        intercepts = [float(value) for value in extracted.intercepts]
    else:
        raise SystemExit(f"{source} contains neither paths.csv nor model.json")

    matrix_source = source / "X.csv"
    if not matrix_source.exists():
        raise SystemExit(f"missing {matrix_source}")
    rows, cols = _write_tiled(matrix_source, output / "X.csv", args.rows)

    expected_source = source / "expected_contribs.csv"
    if not expected_source.exists():
        expected_source = source / "expected.csv"
    if expected_source.exists():
        expected_rows, expected_cols = _write_tiled(expected_source, output / "expected.csv",
                                                     rows)
        if expected_rows != rows or expected_cols != num_groups * (cols + 1):
            raise SystemExit("expected attribution shape does not match X/groups")

    _write_manifest(
        output,
        {
            "name": args.name or source.name,
            "kind": "materialized_fixture",
            "rows": rows,
            "cols": cols,
            "num_groups": num_groups,
            "intercepts": intercepts,
            "source": str(source),
            "source_meta_sha256": _sha256(meta_path) if meta_path.exists() else None,
            "source_model_sha256": _sha256(model_path) if model_path.exists() else None,
            "row_tiling": rows != len(_read_nonempty_rows(matrix_source)),
            "tolerance": float(meta.get("tolerance", 1e-3)),
        },
    )
    print(output / "workload.json")


def generate_stress(args: argparse.Namespace) -> None:
    """Train and freeze a deterministic, moderately large XGBoost regression workload."""
    try:
        import numpy as np
        import xgboost as xgb
    except ImportError as error:
        raise SystemExit("stress generation requires numpy and xgboost") from error
    if min(args.trees, args.depth, args.features, args.train_rows, args.rows) <= 0:
        raise SystemExit("stress dimensions must all be positive")
    if args.features < 5:
        raise SystemExit("--features must be at least 5 for the fixed nonlinear target")
    if not math.isfinite(args.eta) or not 0 < args.eta <= 1:
        raise SystemExit("--eta must be finite and in (0, 1]")
    if not math.isfinite(args.missing_rate) or not 0 <= args.missing_rate < 1:
        raise SystemExit("--missing-rate must be finite and in [0, 1)")

    output = Path(args.output).resolve()
    _prepare_output(output, args.force)
    rng = np.random.default_rng(args.seed)
    train_x = rng.normal(size=(args.train_rows, args.features)).astype(np.float32)
    train_missing = rng.random(train_x.shape) < args.missing_rate
    train_x[train_missing] = np.nan
    # Fixed nonlinear signal gives deep trees useful structure without external data.
    safe = np.nan_to_num(train_x, nan=0.0)
    y = (1.7 * safe[:, 0] - 1.1 * safe[:, 1] ** 2 +
         0.8 * safe[:, 2] * safe[:, 3] + np.sin(safe[:, 4]) +
         rng.normal(0.0, 0.15, size=args.train_rows)).astype(np.float32)
    params = {
        "objective": "reg:squarederror",
        "max_depth": args.depth,
        "eta": args.eta,
        "tree_method": "hist",
        "seed": args.seed,
        "nthread": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    }
    booster = xgb.train(params, xgb.DMatrix(train_x, label=y),
                        num_boost_round=args.trees)
    model_path = output / "model.json"
    booster.save_model(model_path)
    extracted = extract_model(str(model_path))
    write_paths_csv(extracted.paths, str(output / "paths.csv"))

    explain_x = rng.normal(size=(args.rows, args.features)).astype(np.float32)
    explain_x[rng.random(explain_x.shape) < args.missing_rate] = np.nan
    # Training stays single-threaded for a stable model; contribution prediction is a
    # pure read of that frozen model and can use the host without changing its result.
    booster.set_param({"nthread": os.cpu_count() or 1})
    expected = booster.predict(xgb.DMatrix(explain_x), pred_contribs=True)
    np.savetxt(output / "X.csv", explain_x, delimiter=",", fmt="%.9g")
    np.savetxt(output / "expected.csv", expected.reshape(args.rows, -1),
               delimiter=",", fmt="%.12g")

    _write_manifest(
        output,
        {
            "name": args.name,
            "kind": "deterministic_xgboost_stress",
            "rows": args.rows,
            "cols": args.features,
            "num_groups": extracted.num_groups,
            "intercepts": [float(value) for value in extracted.intercepts],
            "raw_path_elements": len(extracted.paths),
            "trees": args.trees,
            "depth": args.depth,
            "train_rows": args.train_rows,
            "seed": args.seed,
            "eta": args.eta,
            "missing_rate": args.missing_rate,
            "xgboost_version": xgb.__version__,
            "tolerance": 1e-3,
            "model": "model.json",
            "model_sha256": _sha256(model_path),
        },
    )
    print(output / "workload.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    hot = subparsers.add_parser("hot", help="generate a deterministic hot-cell workload")
    hot.add_argument("output")
    hot.add_argument("--name", default="hot2000")
    hot.add_argument("--trees", type=int, default=2000)
    hot.add_argument("--rows", type=int, default=32768)
    hot.add_argument("--leaf-scale", type=float, default=0.0005)
    hot.add_argument("--seed", type=int, default=20260712)
    hot.add_argument("--force", action="store_true")
    hot.set_defaults(func=generate_hot)

    fixture = subparsers.add_parser("fixture", help="copy/extract and row-tile a fixture")
    fixture.add_argument("source")
    fixture.add_argument("output")
    fixture.add_argument("--name")
    fixture.add_argument("--rows", type=int, default=0,
                         help="rows in output; 0 preserves the source row count")
    fixture.add_argument("--force", action="store_true")
    fixture.set_defaults(func=materialize_fixture)

    stress = subparsers.add_parser(
        "stress", help="train and freeze a deterministic XGBoost stress workload")
    stress.add_argument("output")
    stress.add_argument("--name", default="stress500")
    stress.add_argument("--trees", type=int, default=500)
    stress.add_argument("--depth", type=int, default=8)
    stress.add_argument("--features", type=int, default=12)
    stress.add_argument("--train-rows", type=int, default=4000)
    stress.add_argument("--rows", type=int, default=8192)
    stress.add_argument("--seed", type=int, default=20260712)
    stress.add_argument("--eta", type=float, default=0.03)
    stress.add_argument("--missing-rate", type=float, default=0.03)
    stress.add_argument("--force", action="store_true")
    stress.set_defaults(func=generate_stress)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
