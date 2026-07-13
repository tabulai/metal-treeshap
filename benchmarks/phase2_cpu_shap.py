#!/usr/bin/env python3
"""Optionally benchmark shap.TreeExplainer on a frozen Phase-2 workload.

This is a comparison baseline, not a required dependency. If SHAP (or one of its binary
dependencies) cannot be imported, the command writes a provenance-rich ``skipped``
artifact and exits successfully. Timings use ``TreeExplainer.shap_values`` with model
loading and explainer construction excluded, matching the persistent Metal benchmark.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

SCHEMA = "metal_treeshap.phase2.cpu_shap.v1"


def _utc_now() -> str:
    """Return the ISO-8601 UTC timestamp contract used by phase2_power.py."""
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _positive_csv(value: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "row limits must be comma-separated integers"
        ) from error
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("row limits must be positive")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_array(values: Any) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def normalize_shap_values(values: Any, rows: int, cols: int, groups: int):
    """Return group-major ``[row, group, feature]`` values across SHAP API layouts."""
    import numpy as np

    if isinstance(values, list):
        if len(values) != groups:
            raise ValueError(
                f"SHAP returned {len(values)} class arrays, expected {groups}"
            )
        arrays = [np.asarray(value, dtype=np.float32) for value in values]
        if any(array.shape != (rows, cols) for array in arrays):
            raise ValueError("SHAP list output has an unexpected shape")
        return np.stack(arrays, axis=1)

    array = np.asarray(values, dtype=np.float32)
    if groups == 1 and array.shape == (rows, cols):
        return array.reshape(rows, 1, cols)
    if array.shape == (rows, groups, cols):
        return array
    if array.shape == (rows, cols, groups):
        return np.transpose(array, (0, 2, 1))
    raise ValueError(
        f"unsupported SHAP output shape {array.shape}; "
        f"expected ({rows},{groups},{cols}) or ({rows},{cols},{groups})"
    )


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _write_artifact(output: Path, artifact: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=output.parent, delete=False, prefix=output.name + ".", suffix=".tmp"
    ) as temp:
        json.dump(artifact, temp, indent=2, sort_keys=True)
        temp.write("\n")
        temporary = Path(temp.name)
    temporary.replace(output)


def _base_artifact(args: argparse.Namespace) -> dict:
    return {
        "schema": SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "pending",
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "shap_distribution": _version("shap"),
            "xgboost_distribution": _version("xgboost"),
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
        },
        "design": {
            "warmups": args.warmup,
            "iterations": args.iterations,
            "feature_perturbation": args.feature_perturbation,
            "model_output": args.model_output,
            "check_additivity": False,
            "setup_excluded": [
                "model load",
                "matrix load",
                "TreeExplainer construction",
            ],
            "timed_call": "TreeExplainer.shap_values(X, check_additivity=False)",
            "power_window": (
                "each result started_utc/finished_utc is an envelope around its "
                "measured calls; sample_windows_utc records the exact timed calls "
                "and excludes intervening gaps"
            ),
        },
        "results": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--row-limits", type=_positive_csv, default=_positive_csv("128,512,2048,8192")
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--nthread", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--feature-perturbation", default="tree_path_dependent")
    parser.add_argument("--model-output", default="raw")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.nthread <= 0:
        raise SystemExit(
            "warmup must be nonnegative; iterations and nthread must be positive"
        )
    for path in (args.model, args.matrix, args.expected):
        if path is not None and not path.is_file():
            raise SystemExit(f"input does not exist: {path}")

    artifact = _base_artifact(args)
    if os.environ.get("METAL_TREESHAP_DISABLE_SHAP"):
        artifact.update(status="skipped", skip_reason="disabled by environment")
        _write_artifact(args.output, artifact)
        print(args.output)
        return
    try:
        import numpy as np
        import shap
        import xgboost as xgb
    except Exception as error:  # Binary dependency failures are not always ImportError.
        artifact.update(
            status="skipped",
            skip_reason=f"SHAP baseline unavailable: {type(error).__name__}: {error}",
        )
        _write_artifact(args.output, artifact)
        print(args.output)
        return

    setup_start = time.perf_counter()
    matrix = np.loadtxt(args.matrix, delimiter=",", dtype=np.float32, ndmin=2)
    expected = (
        np.loadtxt(args.expected, delimiter=",", dtype=np.float32, ndmin=2)
        if args.expected
        else None
    )
    booster = xgb.Booster()
    booster.load_model(args.model)
    booster.set_param({"device": "cpu", "nthread": args.nthread})
    explainer = shap.TreeExplainer(
        booster,
        feature_perturbation=args.feature_perturbation,
        model_output=args.model_output,
    )
    setup_s = time.perf_counter() - setup_start

    cols = int(matrix.shape[1])
    if expected is not None:
        if expected.shape[1] % (cols + 1):
            raise SystemExit(
                "expected attribution width is not groups * (features + 1)"
            )
        groups = expected.shape[1] // (cols + 1)
    else:
        groups = int(getattr(explainer.model, "num_outputs", 1) or 1)
    limits: list[int] = []
    for value in args.row_limits:
        effective = min(value, matrix.shape[0])
        if effective not in limits:
            limits.append(effective)

    method = explainer.shap_values
    artifact["environment"].update(
        shap=shap.__version__, xgboost=xgb.__version__, nthread=args.nthread
    )
    artifact["backend_provenance"] = {
        "public_explainer": f"{type(explainer).__module__}.{type(explainer).__qualname__}",
        "internal_model": f"{type(explainer.model).__module__}."
        f"{type(explainer.model).__qualname__}",
        "internal_model_type": getattr(explainer.model, "model_type", None),
        "timed_method_module": getattr(inspect.getmodule(method), "__name__", None),
        "timed_method_qualname": getattr(method, "__qualname__", None),
        "note": (
            "TreeExplainer may dispatch to compiled SHAP/XGBoost-specific code; the "
            "recorded classes and method identify the observed path without claiming "
            "an independent pure-Python implementation."
        ),
    }

    for rows in limits:
        values = None
        for _ in range(args.warmup):
            values = method(matrix[:rows], check_additivity=False)
        samples: list[float] = []
        sample_windows: list[dict[str, str | float]] = []
        hashes: list[str] = []
        normalized = None
        for _ in range(args.iterations):
            started_utc = _utc_now()
            start = time.perf_counter()
            values = method(matrix[:rows], check_additivity=False)
            elapsed_s = time.perf_counter() - start
            finished_utc = _utc_now()
            samples.append(elapsed_s)
            sample_windows.append(
                {
                    "started_utc": started_utc,
                    "finished_utc": finished_utc,
                    "elapsed_s": elapsed_s,
                }
            )
            normalized = normalize_shap_values(values, rows, cols, groups)
            hashes.append(_hash_array(normalized))
        assert normalized is not None
        samples_array = np.asarray(samples, dtype=np.float64)
        accuracy = None
        if expected is not None:
            expected_groups = expected[:rows].reshape(rows, groups, cols + 1)
            delta = np.abs(normalized - expected_groups[:, :, :cols])
            expected_value = np.asarray(
                explainer.expected_value, dtype=np.float32
            ).reshape(-1)
            if expected_value.size == 1 and groups > 1:
                expected_value = np.repeat(expected_value, groups)
            bias_delta = (
                np.abs(expected_value - expected_groups[0, :, cols])
                if expected_value.size == groups
                else np.asarray([np.nan])
            )
            accuracy = {
                "feature_max_abs": float(delta.max(initial=0.0)),
                "feature_mean_abs": float(delta.mean()) if delta.size else 0.0,
                "expected_value_max_abs": float(np.nanmax(bias_delta)),
            }
        median = float(np.quantile(samples_array, 0.5, method="linear"))
        artifact["results"].append(
            {
                "rows": rows,
                "started_utc": sample_windows[0]["started_utc"],
                "finished_utc": sample_windows[-1]["finished_utc"],
                "sample_windows_utc": sample_windows,
                "timing_s": {
                    "median": median,
                    "p10": float(np.quantile(samples_array, 0.1, method="linear")),
                    "p90": float(np.quantile(samples_array, 0.9, method="linear")),
                    "samples": samples,
                },
                "throughput_rows_per_s": rows / median,
                "repeatability": {
                    "hash_algorithm": "sha256-array-bytes",
                    "unique_hashes": len(set(hashes)),
                    "hashes": hashes,
                },
                "accuracy": accuracy,
            }
        )

    artifact.update(status="ok", setup_s={"load_and_explainer": setup_s})
    artifact["inputs"].update(
        available_rows=int(matrix.shape[0]), cols=cols, groups=groups
    )
    _write_artifact(args.output, artifact)
    print(args.output)


if __name__ == "__main__":
    main()
