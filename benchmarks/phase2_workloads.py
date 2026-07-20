#!/usr/bin/env python3
"""Create deterministic Phase-2 benchmark workloads outside the frozen test oracles.

Five workload families are supported:

* ``hot`` creates many stumps that all update one feature/output cell.  It is a focused
  contention workload for comparing per-lane atomics with SIMD-group aggregation and has
  an analytic expected attribution file.
* ``fixture`` materializes any frozen/raw/model fixture into a separate directory, with
  optional deterministic row tiling.  Source fixtures are opened read-only and never
  modified.  Directories such as the generated stress500 dataset are accepted too.
* ``stress`` trains and freezes a deterministic 500-tree depth-8 XGBoost regression
  workload, including model, extracted paths, data, oracle output, and provenance hashes.
* ``wide`` trains the same deterministic regression shape over many features, reducing
  output-cell contention and exercising a much wider attribution vector.
* ``multiclass`` trains a deterministic multi-output classifier, exercising tree-group
  routing and the group-major attribution layout used by the native benchmark.

Every output has a workload.json consumed by phase2_run.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import struct
import sys
import tempfile
from dataclasses import dataclass
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
        writer.writerow(
            (
                "path_idx",
                "feature_idx",
                "group",
                "lower",
                "upper",
                "is_missing",
                "zero_fraction",
                "v",
            )
        )
        path_idx = 0
        for _ in range(args.trees):
            for lower, upper, missing, leaf in (
                ("-inf", "0", 1, -args.leaf_scale),
                ("0", "inf", 0, args.leaf_scale),
            ):
                writer.writerow(
                    (path_idx, 0, 0, lower, upper, missing, 0.5, repr(leaf))
                )
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
    with (
        (output / "X.csv").open("w") as matrix,
        (output / "expected.csv").open("w") as expected,
    ):
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


_DECIMAL_INTEGER = re.compile(r"[+-]?[0-9]+", re.ASCII)
_DECIMAL_FLOAT = re.compile(
    r"[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
    r"|inf(?:inity)?|nan)",
    re.IGNORECASE | re.ASCII,
)


def _read_nonempty_rows(data: bytes, path: Path) -> list[list[str]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SystemExit(f"CSV is not valid UTF-8: {path}") from error
    rows = [row for row in csv.reader(io.StringIO(text, newline="")) if row]
    if not rows:
        raise SystemExit(f"empty CSV: {path}")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise SystemExit(f"ragged CSV: {path}")
    return rows


def _parse_integer(token: str, *, label: str, location: str) -> int:
    spelling = token.strip()
    if not _DECIMAL_INTEGER.fullmatch(spelling):
        raise SystemExit(f"invalid {label} at {location}: {token!r}")
    return int(spelling, 10)


def _parse_decimal(
    token: str,
    *,
    label: str,
    location: str,
    float32: bool = False,
    require_finite: bool = False,
) -> float:
    """Match the native decimal subset and, where applicable, strtof range."""
    spelling = token.strip()
    if not _DECIMAL_FLOAT.fullmatch(spelling):
        raise SystemExit(f"invalid {label} at {location}: {token!r}")
    value = float(spelling)
    special = spelling.lstrip("+-").lower() in {"inf", "infinity", "nan"}
    if not special and not math.isfinite(value):
        raise SystemExit(f"overflowing {label} at {location}: {token!r}")
    if require_finite and not math.isfinite(value):
        raise SystemExit(f"non-finite {label} at {location}")
    if float32 and math.isfinite(value):
        try:
            value = struct.unpack("=f", struct.pack("=f", value))[0]
        except (OverflowError, struct.error) as error:
            raise SystemExit(f"overflowing float32 {label} at {location}") from error
        if not math.isfinite(value):
            raise SystemExit(f"overflowing float32 {label} at {location}")
    return value


def _read_numeric_rows(
    data: bytes, path: Path, *, label: str, require_finite: bool = False
) -> list[list[str]]:
    """Capture a headerless float32 CSV after applying the native token contract."""
    rows = _read_nonempty_rows(data, path)
    for row_index, row in enumerate(rows, start=1):
        for column_index, token in enumerate(row, start=1):
            _parse_decimal(
                token,
                label=f"{label} value",
                location=f"{path}:{row_index}:{column_index}",
                float32=True,
                require_finite=require_finite,
            )
    return [[token.strip() for token in row] for row in rows]


def _write_tiled(values: list[list[str]], target: Path, rows: int) -> None:
    with target.open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        for index in range(rows):
            writer.writerow(values[index % len(values)])


def _positive_metadata_integer(meta: dict, key: str, source: Path) -> int:
    value = meta.get(key)
    if type(value) is int:
        parsed = value
    elif type(value) is float and math.isfinite(value) and value.is_integer():
        parsed = int(value)
    else:
        raise SystemExit(f"{source} field {key!r} must be a positive integer")
    if parsed <= 0:
        raise SystemExit(f"{source} field {key!r} must be a positive integer")
    if parsed > 0xFFFFFFFF:
        raise SystemExit(f"{source} field {key!r} does not fit uint32")
    return parsed


def _validate_intercepts(values: object, num_groups: int, source: Path) -> list[float]:
    if not isinstance(values, list):
        raise SystemExit(f"{source} field 'intercepts' must be an array")
    if not all(type(value) in (int, float) for value in values):
        raise SystemExit(f"{source} intercepts must be JSON numbers")
    intercepts = [float(value) for value in values]
    if len(intercepts) == 1 and num_groups > 1:
        intercepts *= num_groups
    if len(intercepts) != num_groups:
        raise SystemExit(
            f"{source} intercept count ({len(intercepts)}) does not match "
            f"num_groups ({num_groups})"
        )
    if not all(math.isfinite(value) for value in intercepts):
        raise SystemExit(f"{source} intercepts must be finite")
    return intercepts


@dataclass(frozen=True)
class _CapturedPath:
    path_idx: int
    feature_idx: int
    group: int
    lower: float
    upper: float
    is_missing: bool
    zero_fraction: float
    leaf_value: float


def _parse_path_row(row: list[str], path: Path, row_index: int) -> _CapturedPath:
    location = f"{path}:{row_index}"
    path_idx = _parse_integer(row[0], label="path_idx", location=location)
    feature_idx = _parse_integer(row[1], label="feature_idx", location=location)
    group = _parse_integer(row[2], label="group", location=location)
    lower = _parse_decimal(
        row[3], label="lower bound", location=location, float32=True
    )
    upper = _parse_decimal(
        row[4], label="upper bound", location=location, float32=True
    )
    missing = _parse_integer(row[5], label="is_missing", location=location)
    zero_fraction = _parse_decimal(
        row[6], label="zero_fraction", location=location, require_finite=True
    )
    leaf_value = _parse_decimal(
        row[7],
        label="leaf value",
        location=location,
        float32=True,
        require_finite=True,
    )
    if missing not in (0, 1):
        raise SystemExit(f"is_missing must be 0 or 1 at {location}")
    return _CapturedPath(
        path_idx,
        feature_idx,
        group,
        lower,
        upper,
        bool(missing),
        zero_fraction,
        leaf_value,
    )


def _validate_captured_paths(
    paths: list[_CapturedPath], num_groups: int, num_cols: int, source: Path
) -> None:
    per_path: dict[int, list[_CapturedPath]] = {}
    for row_index, element in enumerate(paths, start=2):
        location = f"{source}:{row_index}"
        if not 0 <= element.path_idx <= 0xFFFFFFFF:
            raise SystemExit(f"path_idx does not fit uint32 at {location}")
        if element.feature_idx != -1 and not 0 <= element.feature_idx < num_cols:
            raise SystemExit(f"feature_idx out of range at {location}")
        if not 0 <= element.group < num_groups:
            raise SystemExit(f"group out of range at {location}")
        if math.isnan(element.lower) or math.isnan(element.upper):
            raise SystemExit(f"split bounds must not be NaN at {location}")
        if element.lower > element.upper:
            raise SystemExit(f"split interval is inverted at {location}")
        if not 0 <= element.zero_fraction <= 1:
            raise SystemExit(f"zero_fraction must be in [0, 1] at {location}")
        if element.feature_idx == -1 and element.zero_fraction != 1.0:
            raise SystemExit(f"root element must have zero_fraction == 1.0 at {location}")
        per_path.setdefault(element.path_idx, []).append(element)

    for path_idx, elements in per_path.items():
        roots = sum(element.feature_idx == -1 for element in elements)
        if roots == 0:
            raise SystemExit(f"path {path_idx} is missing its root element")
        if roots > 1:
            raise SystemExit(f"path {path_idx} has more than one root element")
        group = elements[0].group
        leaf_value = elements[0].leaf_value
        if any(element.group != group for element in elements[1:]):
            raise SystemExit(f"group must be constant along path {path_idx}")
        if any(element.leaf_value != leaf_value for element in elements[1:]):
            raise SystemExit(f"leaf value must be constant along path {path_idx}")

        merged: dict[int, tuple[float, float, bool, float]] = {}
        for element in elements:
            previous = merged.get(element.feature_idx)
            if previous is None:
                merged[element.feature_idx] = (
                    element.lower,
                    element.upper,
                    element.is_missing,
                    element.zero_fraction,
                )
            else:
                merged[element.feature_idx] = (
                    max(previous[0], element.lower),
                    min(previous[1], element.upper),
                    previous[2] and element.is_missing,
                    previous[3] * element.zero_fraction,
                )
        if len(merged) > 32:
            raise SystemExit(f"path {path_idx} exceeds the 32-element depth limit")
        for feature_idx, (lower, upper, missing, zero_fraction) in merged.items():
            if not math.isfinite(zero_fraction) or not 0 <= zero_fraction <= 1:
                raise SystemExit(
                    f"merged zero_fraction is invalid on path {path_idx}"
                )
            if feature_idx != -1 and lower >= upper and not missing:
                raise SystemExit(
                    f"merged split condition is unsatisfiable on path {path_idx}"
                )


def _validate_combined_bias(
    paths: list[_CapturedPath],
    intercepts: list[float],
    num_groups: int,
    source: Path,
) -> None:
    """Mirror the native fp64 path-bias sum and its fp32 output contract."""
    per_path: dict[int, tuple[float, _CapturedPath]] = {}
    for element in paths:
        product, representative = per_path.get(element.path_idx, (1.0, element))
        per_path[element.path_idx] = (
            product * element.zero_fraction,
            representative,
        )
    bias = [0.0] * num_groups
    for path_idx in sorted(per_path):  # std::map order in ComputeBias
        product, representative = per_path[path_idx]
        bias[representative.group] += product * representative.leaf_value
    float32_max = float.fromhex("0x1.fffffep+127")
    for group, (path_bias, intercept) in enumerate(zip(bias, intercepts, strict=True)):
        combined = path_bias + intercept
        if not math.isfinite(combined) or abs(combined) > float32_max:
            raise SystemExit(
                f"{source} path bias + intercept for group {group} must be finite "
                "and representable as float32"
            )


def _capture_paths_csv(
    data: bytes, path: Path, num_groups: int, num_cols: int
) -> list[_CapturedPath]:
    rows = _read_nonempty_rows(data, path)
    expected_header = [
        "path_idx",
        "feature_idx",
        "group",
        "lower",
        "upper",
        "is_missing",
        "zero_fraction",
        "v",
    ]
    if [field.strip() for field in rows[0]] != expected_header:
        raise SystemExit(f"invalid paths.csv header: {path}")
    captured = [
        _parse_path_row(row, path, row_index)
        for row_index, row in enumerate(rows[1:], start=2)
    ]
    _validate_captured_paths(captured, num_groups, num_cols, path)
    return captured


def _capture_extracted_paths(
    paths: list, source: Path, num_groups: int, num_cols: int
) -> list[_CapturedPath]:
    captured = [
        _parse_path_row(
            [
                str(element.path_idx),
                str(element.feature_idx),
                str(element.group),
                repr(element.lower),
                repr(element.upper),
                str(int(element.is_missing_branch)),
                repr(element.zero_fraction),
                repr(element.v),
            ],
            source,
            row_index,
        )
        for row_index, element in enumerate(paths, start=2)
    ]
    _validate_captured_paths(captured, num_groups, num_cols, source)
    return captured


def _write_captured_paths(paths: list[_CapturedPath], target: Path) -> None:
    with target.open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(
            (
                "path_idx",
                "feature_idx",
                "group",
                "lower",
                "upper",
                "is_missing",
                "zero_fraction",
                "v",
            )
        )
        for element in paths:
            writer.writerow(
                (
                    element.path_idx,
                    element.feature_idx,
                    element.group,
                    repr(element.lower),
                    repr(element.upper),
                    int(element.is_missing),
                    repr(element.zero_fraction),
                    repr(element.leaf_value),
                )
            )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _capture_file(path: Path, label: str) -> tuple[Path, bytes]:
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")
    try:
        return path, path.read_bytes()
    except OSError as error:
        raise SystemExit(f"cannot read {label}: {path}") from error


def _resolve_artifact(source: Path, reference: object, label: str) -> Path:
    if not isinstance(reference, str) or not reference:
        raise SystemExit(f"workload.json field {label!r} must be a non-empty string")
    relative = Path(reference)
    if relative.is_absolute():
        raise SystemExit(f"workload.json field {label!r} must be relative")
    artifact = (source / relative).resolve()
    try:
        artifact.relative_to(source)
    except ValueError as error:
        raise SystemExit(
            f"workload.json field {label!r} escapes the source directory"
        ) from error
    if not artifact.is_file():
        raise SystemExit(f"missing workload artifact for {label!r}: {artifact}")
    return artifact


def _capture_workload_manifest(
    source: Path, manifest_path: Path, manifest_bytes: bytes
) -> tuple[dict, dict[str, tuple[Path, bytes]]]:
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid JSON manifest: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise SystemExit(f"{manifest_path} must contain a JSON object")
    required = {
        "schema",
        "name",
        "paths",
        "matrix",
        "expected",
        "rows",
        "cols",
        "num_groups",
        "intercepts",
        "sha256",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise SystemExit(f"{manifest_path} is missing required fields: {missing}")
    if manifest["schema"] != SCHEMA:
        raise SystemExit(f"unsupported workload schema: {manifest['schema']!r}")
    if not isinstance(manifest["name"], str) or not manifest["name"]:
        raise SystemExit(f"{manifest_path} field 'name' must be a non-empty string")
    for key in ("paths", "matrix"):
        if not isinstance(manifest[key], str) or not manifest[key]:
            raise SystemExit(
                f"{manifest_path} field {key!r} must be a non-empty string"
            )
    if manifest["expected"] is not None and not isinstance(
        manifest["expected"], str
    ):
        raise SystemExit(
            f"{manifest_path} field 'expected' must be a string or null"
        )
    if "model" in manifest and manifest["model"] is not None and not isinstance(
        manifest["model"], str
    ):
        raise SystemExit(f"{manifest_path} field 'model' must be a string or null")

    hashes = manifest["sha256"]
    if not isinstance(hashes, dict):
        raise SystemExit(f"{manifest_path} field 'sha256' must be an object")
    captured_by_reference: dict[str, tuple[Path, bytes]] = {}
    for reference, expected_hash in hashes.items():
        artifact = _resolve_artifact(source, reference, f"sha256[{reference!r}]")
        if not isinstance(expected_hash, str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", expected_hash
        ):
            raise SystemExit(
                f"invalid SHA-256 digest for workload artifact {reference!r}"
            )
        try:
            data = artifact.read_bytes()
        except OSError as error:
            raise SystemExit(f"cannot read workload artifact: {artifact}") from error
        actual_hash = _sha256_bytes(data)
        if actual_hash != expected_hash.lower():
            raise SystemExit(
                f"SHA-256 mismatch for workload artifact {reference!r}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        captured_by_reference[reference] = (artifact, data)

    artifacts: dict[str, tuple[Path, bytes]] = {}
    for key in ("paths", "matrix", "expected", "model"):
        reference = manifest.get(key)
        if reference is None:
            continue
        artifact = _resolve_artifact(source, reference, key)
        if reference not in captured_by_reference:
            raise SystemExit(
                f"workload artifact {key!r} ({reference!r}) has no SHA-256 entry"
            )
        captured_path, data = captured_by_reference[reference]
        if captured_path != artifact:
            raise SystemExit(f"workload artifact reference changed during validation: {key}")
        artifacts[key] = (artifact, data)
    return manifest, artifacts


def _transactional_output(path: Path, force: bool, build) -> None:
    """Build beside the destination, then swap with rollback of an existing output."""
    if path.exists():
        nonempty = not path.is_dir() or any(path.iterdir())
        if nonempty and not force:
            raise SystemExit(f"output directory is non-empty: {path} (pass --force)")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{path.name}.materializing-", dir=path.parent)
    )
    backup: Path | None = None
    try:
        build(staging)
        if path.exists():
            backup = Path(
                tempfile.mkdtemp(prefix=f".{path.name}.backup-", dir=path.parent)
            )
            backup.rmdir()
            os.replace(path, backup)
        try:
            os.replace(staging, path)
        except BaseException:
            if backup is not None:
                os.replace(backup, path)
                backup = None
            raise
        if backup is not None:
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink()
            backup = None
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup is not None and not path.exists():
            os.replace(backup, path)


def materialize_fixture(args: argparse.Namespace) -> None:
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    if not source.is_dir():
        raise SystemExit(f"fixture directory does not exist: {source}")
    # Reject ANY ancestor/descendant overlap, not just exact equality: --force
    # recursively deletes the output, so an output that contains the source would
    # delete the fixture before it is read. Compare FILESYSTEM IDENTITY (st_dev,
    # st_ino), not lexical paths: macOS APFS is case-insensitive by default, so
    # differently-cased spellings of the same directory defeat string comparison.
    def _fs_id(path: Path) -> tuple[int, int]:
        stat = path.stat()
        return (stat.st_dev, stat.st_ino)

    source_chain = {_fs_id(node) for node in (source, *source.parents)}
    probe = output
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent  # deepest existing ancestor: what rmtree/mkdir touch
    output_chain = ({_fs_id(node) for node in (probe, *probe.parents)}
                    if probe.exists() else set())
    # source inside (existing part of) output, or output equal to / containing source.
    if _fs_id(source) in output_chain or (
            output.exists() and _fs_id(output) in source_chain):
        raise SystemExit(
            f"output must not overlap the source fixture directory "
            f"(source={source}, output={output})")
    # Capture and validate every input before constructing a sibling staging directory.
    # Frozen fixtures use fixed artifact names plus meta.json; generated workloads use
    # the references and integrity hashes in workload.json as their authority.
    meta_path = source / "meta.json"
    manifest_path = source / "workload.json"
    source_meta_sha256 = None
    source_manifest_sha256 = None
    if meta_path.exists():
        meta_bytes = _capture_file(meta_path, "fixture metadata")[1]
        try:
            meta = json.loads(meta_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SystemExit(f"invalid JSON metadata: {meta_path}") from error
        meta_name = meta_path
        source_meta_sha256 = _sha256_bytes(meta_bytes)
        matrix_capture = _capture_file(source / "X.csv", "fixture matrix")
        paths_path = source / "paths.csv"
        paths_capture = (
            _capture_file(paths_path, "fixture paths") if paths_path.exists() else None
        )
        model_path = source / "model.json"
        model_capture = (
            _capture_file(model_path, "fixture model") if model_path.exists() else None
        )
        expected_path = source / "expected_contribs.csv"
        if not expected_path.exists():
            expected_path = source / "expected.csv"
        expected_capture = (
            _capture_file(expected_path, "fixture expected attributions")
            if expected_path.exists()
            else None
        )
    elif manifest_path.exists():
        manifest_bytes = _capture_file(manifest_path, "workload manifest")[1]
        meta, artifacts = _capture_workload_manifest(
            source, manifest_path, manifest_bytes
        )
        meta_name = manifest_path
        source_manifest_sha256 = _sha256_bytes(manifest_bytes)
        matrix_capture = artifacts["matrix"]
        paths_capture = artifacts["paths"]
        expected_capture = artifacts.get("expected")
        model_capture = artifacts.get("model")
    else:
        meta, meta_name = {}, meta_path
        matrix_capture = _capture_file(source / "X.csv", "fixture matrix")
        paths_path = source / "paths.csv"
        paths_capture = (
            _capture_file(paths_path, "fixture paths") if paths_path.exists() else None
        )
        model_path = source / "model.json"
        model_capture = (
            _capture_file(model_path, "fixture model") if model_path.exists() else None
        )
        expected_path = source / "expected_contribs.csv"
        if not expected_path.exists():
            expected_path = source / "expected.csv"
        expected_capture = (
            _capture_file(expected_path, "fixture expected attributions")
            if expected_path.exists()
            else None
        )
    if not isinstance(meta, dict):
        raise SystemExit(f"{meta_name} must contain a JSON object")
    if args.rows < 0:
        raise SystemExit("--rows must be positive when supplied")
    matrix_source, matrix_bytes = matrix_capture
    matrix_values = _read_numeric_rows(
        matrix_bytes, matrix_source, label="matrix"
    )
    source_rows = len(matrix_values)
    cols = len(matrix_values[0])
    rows = args.rows or source_rows
    if rows > 0xFFFFFFFF:
        raise SystemExit("output row count does not fit uint32")

    extracted = None
    if paths_capture is not None:
        # Mirror the CLI contract: intercepts must be explicit, never fabricated —
        # a silent zero default hides real bias errors in every downstream run.
        if "intercepts" not in meta:
            raise SystemExit(
                f"{meta_name} must carry explicit 'intercepts' (pass zeros for an "
                "intercept-free model) for a paths.csv fixture")
        if "num_groups" not in meta and "groups" not in meta:
            raise SystemExit(
                f"{meta_name} must carry 'num_groups' for a paths.csv fixture")
        group_key = "num_groups" if "num_groups" in meta else "groups"
        num_groups = _positive_metadata_integer(meta, group_key, meta_name)
        if "num_groups" in meta and "groups" in meta:
            legacy_groups = _positive_metadata_integer(meta, "groups", meta_name)
            if legacy_groups != num_groups:
                raise SystemExit(
                    f"{meta_name} fields 'num_groups' and 'groups' disagree"
                )
        intercepts = _validate_intercepts(meta["intercepts"], num_groups, meta_name)
        paths_source, paths_bytes = paths_capture
        captured_paths = _capture_paths_csv(
            paths_bytes, paths_source, num_groups, cols
        )
    elif model_capture is not None:
        model_path, model_bytes = model_capture
        extracted = extract_model(model_bytes)
        num_groups = extracted.num_groups
        if num_groups <= 0:
            raise SystemExit(f"model has invalid num_groups: {num_groups}")
        intercepts = _validate_intercepts(
            list(extracted.intercepts), num_groups, model_path
        )
        for key in ("num_groups", "groups"):
            if key in meta:
                declared_groups = _positive_metadata_integer(meta, key, meta_name)
                if declared_groups != num_groups:
                    raise SystemExit(
                        f"{meta_name} field {key!r} ({declared_groups}) does not "
                        f"match model group count ({num_groups})"
                    )
        if "intercepts" in meta:
            declared_intercepts = _validate_intercepts(
                meta["intercepts"], num_groups, meta_name
            )
            if declared_intercepts != intercepts:
                raise SystemExit(
                    f"{meta_name} intercepts do not match the extracted model"
                )
        if extracted.num_features != cols:
            raise SystemExit(
                f"X column count ({cols}) does not match model feature count "
                f"({extracted.num_features})"
            )
        captured_paths = _capture_extracted_paths(
            extracted.paths, model_path, num_groups, cols
        )
    else:
        raise SystemExit(f"{source} contains neither paths.csv nor model.json")

    _validate_combined_bias(captured_paths, intercepts, num_groups, meta_name)

    for key in ("cols", "num_features"):
        if key in meta:
            declared_cols = _positive_metadata_integer(meta, key, meta_name)
            if declared_cols != cols:
                raise SystemExit(
                    f"{meta_name} field {key!r} ({declared_cols}) does not match "
                    f"X column count ({cols})"
                )
    if "rows" in meta:
        declared_rows = _positive_metadata_integer(meta, "rows", meta_name)
        if declared_rows != source_rows:
            raise SystemExit(
                f"{meta_name} row count ({declared_rows}) does not match "
                f"X row count ({source_rows})"
            )
    raw_tolerance = meta.get("tolerance", 1e-3)
    if type(raw_tolerance) not in (int, float):
        raise SystemExit(f"{meta_name} tolerance must be a JSON number")
    tolerance = float(raw_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise SystemExit(f"{meta_name} tolerance must be finite and positive")

    expected_values = None
    if expected_capture is not None:
        expected_source, expected_bytes = expected_capture
        expected_values = _read_numeric_rows(
            expected_bytes,
            expected_source,
            label="expected attribution",
            require_finite=True,
        )
        if len(expected_values) != source_rows:
            raise SystemExit(
                f"expected attribution row count ({len(expected_values)}) does not "
                f"match X row count ({source_rows})"
            )
        expected_cols = num_groups * (cols + 1)
        if len(expected_values[0]) != expected_cols:
            raise SystemExit(
                f"expected attribution width ({len(expected_values[0])}) does not "
                f"match groups * (features + 1) ({expected_cols})"
            )

    source_model_sha256 = (
        _sha256_bytes(model_capture[1]) if model_capture is not None else None
    )

    def _build(staging: Path) -> None:
        _write_captured_paths(captured_paths, staging / "paths.csv")
        _write_tiled(matrix_values, staging / "X.csv", rows)
        if expected_values is not None:
            _write_tiled(expected_values, staging / "expected.csv", rows)
        output_model = None
        # A paths.csv source is already the complete native fixture contract. Preserve
        # a model only when it was actually selected and parsed to produce those paths;
        # do not advertise an unrelated optional model beside authoritative raw paths.
        if extracted is not None and model_capture is not None:
            output_model = "model.json"
            (staging / output_model).write_bytes(model_capture[1])
        output_manifest = {
            "name": args.name or source.name,
            "kind": "materialized_fixture",
            "rows": rows,
            "cols": cols,
            "num_groups": num_groups,
            "intercepts": intercepts,
            "source": str(source),
            "source_meta_sha256": source_meta_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "source_model_sha256": source_model_sha256,
            "row_tiling": rows != source_rows,
            "tolerance": tolerance,
        }
        if output_model is not None:
            output_manifest["model"] = output_model
        _write_manifest(staging, output_manifest)

    _transactional_output(output, args.force, _build)
    print(output / "workload.json")


def _validate_xgboost_args(args: argparse.Namespace, *, minimum_features: int) -> None:
    if min(args.trees, args.depth, args.features, args.train_rows, args.rows) <= 0:
        raise SystemExit("workload dimensions must all be positive")
    if args.features < minimum_features:
        raise SystemExit(
            f"--features must be at least {minimum_features} for this target"
        )
    if not math.isfinite(args.eta) or not 0 < args.eta <= 1:
        raise SystemExit("--eta must be finite and in (0, 1]")
    if not math.isfinite(args.missing_rate) or not 0 <= args.missing_rate < 1:
        raise SystemExit("--missing-rate must be finite and in [0, 1)")


def _freeze_booster(
    args: argparse.Namespace,
    *,
    params: dict,
    train_x,
    y,
    kind: str,
    extra_manifest: dict | None = None,
    explain_x=None,
) -> None:
    """Train and freeze an XGBoost benchmark with a common, hashed artifact contract."""
    import numpy as np
    import xgboost as xgb

    output = Path(args.output).resolve()
    _prepare_output(output, args.force)
    booster = xgb.train(
        params, xgb.DMatrix(train_x, label=y), num_boost_round=args.trees
    )
    model_path = output / "model.json"
    booster.save_model(model_path)
    extracted = extract_model(str(model_path))
    write_paths_csv(extracted.paths, str(output / "paths.csv"))

    if explain_x is None:
        rng = np.random.default_rng(args.seed + 1)
        explain_x = rng.normal(size=(args.rows, args.features)).astype(np.float32)
        explain_x[rng.random(explain_x.shape) < args.missing_rate] = np.nan
    booster.set_param({"nthread": os.cpu_count() or 1})
    expected = booster.predict(xgb.DMatrix(explain_x), pred_contribs=True)
    np.savetxt(output / "X.csv", explain_x, delimiter=",", fmt="%.9g")
    np.savetxt(
        output / "expected.csv",
        expected.reshape(args.rows, -1),
        delimiter=",",
        fmt="%.12g",
    )

    manifest = {
        "name": args.name,
        "kind": kind,
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
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    _write_manifest(output, manifest)
    print(output / "workload.json")


def generate_stress(args: argparse.Namespace) -> None:
    """Train and freeze a deterministic, moderately large XGBoost regression workload."""
    try:
        import numpy as np
        __import__("xgboost")
    except ImportError as error:
        raise SystemExit("stress generation requires numpy and xgboost") from error
    _validate_xgboost_args(args, minimum_features=5)
    rng = np.random.default_rng(args.seed)
    train_x = rng.normal(size=(args.train_rows, args.features)).astype(np.float32)
    train_missing = rng.random(train_x.shape) < args.missing_rate
    train_x[train_missing] = np.nan
    # Fixed nonlinear signal gives deep trees useful structure without external data.
    safe = np.nan_to_num(train_x, nan=0.0)
    y = (
        1.7 * safe[:, 0]
        - 1.1 * safe[:, 1] ** 2
        + 0.8 * safe[:, 2] * safe[:, 3]
        + np.sin(safe[:, 4])
        + rng.normal(0.0, 0.15, size=args.train_rows)
    ).astype(np.float32)
    # Preserve the original stress500 byte stream: its explain matrix continues the
    # training generator rather than starting a separate RNG. Existing hashes therefore
    # remain comparable across the Phase-2 and Phase-2.1 runners.
    explain_x = rng.normal(size=(args.rows, args.features)).astype(np.float32)
    explain_x[rng.random(explain_x.shape) < args.missing_rate] = np.nan
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
    _freeze_booster(
        args,
        params=params,
        train_x=train_x,
        y=y,
        kind="deterministic_xgboost_stress",
        explain_x=explain_x,
    )


def generate_wide(args: argparse.Namespace) -> None:
    """Generate a low-contention regression workload with a wide feature vector."""
    try:
        import numpy as np
    except ImportError as error:
        raise SystemExit("wide generation requires numpy and xgboost") from error
    try:
        import xgboost  # noqa: F401
    except ImportError as error:
        raise SystemExit("wide generation requires numpy and xgboost") from error
    _validate_xgboost_args(args, minimum_features=16)
    rng = np.random.default_rng(args.seed)
    train_x = rng.normal(size=(args.train_rows, args.features)).astype(np.float32)
    train_x[rng.random(train_x.shape) < args.missing_rate] = np.nan
    safe = np.nan_to_num(train_x[:, :16], nan=0.0)
    weights = np.linspace(1.4, 0.2, 16, dtype=np.float32)
    # Some Accelerate-backed NumPy builds emit spurious fp-status warnings after a
    # finite float32 matmul; inputs and output are explicitly checked below.
    with np.errstate(all="ignore"):
        y = (
            np.tanh(safe) @ weights
            + 0.7 * safe[:, 0] * safe[:, 8]
            - 0.5 * safe[:, 3] * safe[:, 12]
            + rng.normal(0.0, 0.2, size=args.train_rows)
        ).astype(np.float32)
    if not np.isfinite(y).all():
        raise SystemExit("wide target generation produced non-finite values")
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
    _freeze_booster(
        args,
        params=params,
        train_x=train_x,
        y=y,
        kind="deterministic_xgboost_wide_features",
        extra_manifest={
            "workload_axis": "feature_width",
            "explain_seed": args.seed + 1,
        },
    )


def generate_multiclass(args: argparse.Namespace) -> None:
    """Generate a deterministic multiclass workload with multiple output groups."""
    try:
        import numpy as np
    except ImportError as error:
        raise SystemExit("multiclass generation requires numpy and xgboost") from error
    try:
        import xgboost  # noqa: F401
    except ImportError as error:
        raise SystemExit("multiclass generation requires numpy and xgboost") from error
    _validate_xgboost_args(args, minimum_features=8)
    if args.classes < 3:
        raise SystemExit("--classes must be at least 3")
    if args.classes > args.features:
        raise SystemExit("--classes cannot exceed --features for this target")
    rng = np.random.default_rng(args.seed)
    train_x = rng.normal(size=(args.train_rows, args.features)).astype(np.float32)
    train_x[rng.random(train_x.shape) < args.missing_rate] = np.nan
    safe = np.nan_to_num(train_x, nan=0.0)
    projection = rng.normal(0.0, 0.65, size=(args.features, args.classes)).astype(
        np.float32
    )
    with np.errstate(all="ignore"):
        logits = safe @ projection
    if not np.isfinite(logits).all():
        raise SystemExit("multiclass target generation produced non-finite values")
    logits += 0.55 * np.sin(safe[:, : args.classes])
    logits += rng.normal(0.0, 0.25, size=logits.shape)
    y = np.argmax(logits, axis=1).astype(np.float32)
    # Guarantee every class is represented even for deliberately tiny smoke tests.
    if args.train_rows >= args.classes:
        y[: args.classes] = np.arange(args.classes, dtype=np.float32)
    params = {
        "objective": "multi:softprob",
        "num_class": args.classes,
        "max_depth": args.depth,
        "eta": args.eta,
        "tree_method": "hist",
        "seed": args.seed,
        "nthread": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    }
    _freeze_booster(
        args,
        params=params,
        train_x=train_x,
        y=y,
        kind="deterministic_xgboost_multiclass",
        extra_manifest={
            "workload_axis": "output_groups",
            "classes": args.classes,
            "explain_seed": args.seed + 1,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    hot = subparsers.add_parser(
        "hot", help="generate a deterministic hot-cell workload"
    )
    hot.add_argument("output")
    hot.add_argument("--name", default="hot2000")
    hot.add_argument("--trees", type=int, default=2000)
    hot.add_argument("--rows", type=int, default=32768)
    hot.add_argument("--leaf-scale", type=float, default=0.0005)
    hot.add_argument("--seed", type=int, default=20260712)
    hot.add_argument("--force", action="store_true")
    hot.set_defaults(func=generate_hot)

    fixture = subparsers.add_parser(
        "fixture", help="copy/extract and row-tile a fixture"
    )
    fixture.add_argument("source")
    fixture.add_argument("output")
    fixture.add_argument("--name")
    fixture.add_argument(
        "--rows",
        type=int,
        default=0,
        help="rows in output; 0 preserves the source row count",
    )
    fixture.add_argument("--force", action="store_true")
    fixture.set_defaults(func=materialize_fixture)

    stress = subparsers.add_parser(
        "stress", help="train and freeze a deterministic XGBoost stress workload"
    )
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

    wide = subparsers.add_parser(
        "wide", help="train a deterministic wide-feature XGBoost regression workload"
    )
    wide.add_argument("output")
    wide.add_argument("--name", default="wide256")
    wide.add_argument("--trees", type=int, default=400)
    wide.add_argument("--depth", type=int, default=6)
    wide.add_argument("--features", type=int, default=256)
    wide.add_argument("--train-rows", type=int, default=6000)
    wide.add_argument("--rows", type=int, default=8192)
    wide.add_argument("--seed", type=int, default=20260712)
    wide.add_argument("--eta", type=float, default=0.04)
    wide.add_argument("--missing-rate", type=float, default=0.03)
    wide.add_argument("--force", action="store_true")
    wide.set_defaults(func=generate_wide)

    multiclass = subparsers.add_parser(
        "multiclass", help="train a deterministic multiclass XGBoost workload"
    )
    multiclass.add_argument("output")
    multiclass.add_argument("--name", default="multiclass8")
    multiclass.add_argument(
        "--trees",
        type=int,
        default=150,
        help="boosting rounds; total trees are rounds times classes",
    )
    multiclass.add_argument("--depth", type=int, default=6)
    multiclass.add_argument("--features", type=int, default=32)
    multiclass.add_argument("--classes", type=int, default=8)
    multiclass.add_argument("--train-rows", type=int, default=6000)
    multiclass.add_argument("--rows", type=int, default=8192)
    multiclass.add_argument("--seed", type=int, default=20260712)
    multiclass.add_argument("--eta", type=float, default=0.05)
    multiclass.add_argument("--missing-rate", type=float, default=0.03)
    multiclass.add_argument("--force", action="store_true")
    multiclass.set_defaults(func=generate_multiclass)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
