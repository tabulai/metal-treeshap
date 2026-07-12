#!/usr/bin/env python3
"""Run the persistent native Phase-2 benchmark over a reproducible tuning matrix.

The native executable owns the timed loop; this script only schedules configurations,
captures environment/provenance, validates returned JSON, and writes one suite artifact.
Configuration order is deterministically shuffled to reduce monotonic thermal/order bias.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from pathlib import Path

SUITE_SCHEMA = "metal_treeshap.phase2.suite.v1"
RESULT_SCHEMA = "metal_treeshap.phase2.benchmark.v1"
WORKLOAD_SCHEMA = "metal_treeshap.phase2.workload.v1"


def csv_values(value: str, convert=str) -> list:
    result = [convert(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=repo, text=True, check=True,
                              capture_output=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def command_value(*args: str) -> str | None:
    try:
        return subprocess.run(list(args), text=True, check=True,
                              capture_output=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_workload(path: Path) -> dict:
    manifest_path = path / "workload.json" if path.is_dir() else path
    manifest_path = manifest_path.resolve()
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema") != WORKLOAD_SCHEMA:
        raise SystemExit(f"unsupported workload schema in {manifest_path}")
    base = manifest_path.parent
    hashes = payload.get("sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise SystemExit(f"workload has no provenance hashes: {manifest_path}")
    referenced = [payload.get("paths"), payload.get("matrix"), payload.get("expected"),
                  payload.get("model")]
    missing_hashes = [value for value in referenced if value and value not in hashes]
    if missing_hashes:
        raise SystemExit(
            f"workload lacks hashes for referenced files {missing_hashes}: {manifest_path}"
        )
    for relative, expected_hash in hashes.items():
        hashed_path = (base / relative).resolve()
        if not hashed_path.is_file():
            raise SystemExit(f"hashed workload file is missing: {hashed_path}")
        actual_hash = sha256(hashed_path)
        if actual_hash != expected_hash:
            raise SystemExit(
                f"workload hash mismatch for {hashed_path}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
    for key in ("paths", "matrix", "expected"):
        value = payload.get(key)
        payload[f"_{key}"] = str((base / value).resolve()) if value else None
    payload["_manifest"] = str(manifest_path)
    return payload


def validate_result(result: dict, workload: dict, configuration: dict, *, warmup: int,
                    iterations: int, row_limit: int) -> None:
    if result.get("schema") != RESULT_SCHEMA or result.get("status") != "ok":
        raise RuntimeError("native benchmark returned an unsupported/failed result")
    if result["workload"]["rows"] <= 0 or result["workload"]["cols"] <= 0:
        raise RuntimeError("native benchmark returned an invalid workload shape")
    if result["configuration"]["accumulation"] != configuration["accumulation"]:
        raise RuntimeError("native result configuration does not match requested mode")
    if result["configuration"]["model_storage"] != configuration["storage"]:
        raise RuntimeError("native result storage does not match requested mode")
    expected_rows = min(workload["rows"], row_limit) if row_limit else workload["rows"]
    native_workload = result["workload"]
    native_config = result["configuration"]
    if (native_workload["source_rows"] != workload["rows"] or
            native_workload["rows"] != expected_rows or
            native_workload["cols"] != workload["cols"] or
            native_workload["groups"] != workload["num_groups"]):
        raise RuntimeError("native result workload shape/groups do not match manifest")
    if (native_config["rows_per_simdgroup"] != configuration["rows_per_simdgroup"] or
            native_config["threads_per_threadgroup"] !=
            configuration["threads_per_threadgroup"] or
            native_config["deterministic_scratch_mib"] !=
            configuration["deterministic_scratch_mib"] or
            native_config["warmups"] != warmup or
            native_config["iterations"] != iterations):
        raise RuntimeError("native result tuning/iteration configuration does not match request")
    if workload.get("expected") and result.get("accuracy") is None:
        raise RuntimeError("expected workload produced no accuracy metrics")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path, help="phase2_benchmark executable")
    parser.add_argument("workloads", nargs="+", type=Path,
                        help="workload.json files or their containing directories")
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-simdgroup", default="256,1024,4096")
    parser.add_argument("--threads-per-threadgroup", default="64,128,256")
    parser.add_argument("--accumulations", default="atomic,simdgroup")
    parser.add_argument("--model-storage", default="shared,private")
    parser.add_argument("--deterministic-scratch-mib", default="256")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--row-limit", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=1,
                        help="outer repeats of each configuration")
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    benchmark = args.benchmark.resolve()
    kernel = args.kernel.resolve()
    if not benchmark.is_file() or not os.access(benchmark, os.X_OK):
        raise SystemExit(f"benchmark is not executable: {benchmark}")
    if not kernel.is_file():
        raise SystemExit(f"kernel does not exist: {kernel}")
    if args.warmup < 0 or args.iterations <= 0 or args.row_limit < 0 or args.rounds <= 0:
        raise SystemExit("warmup/row-limit must be nonnegative; iterations/rounds positive")

    rps_values = csv_values(args.rows_per_simdgroup, int)
    thread_values = csv_values(args.threads_per_threadgroup, int)
    accumulation_values = csv_values(args.accumulations)
    storage_values = csv_values(args.model_storage)
    scratch_values = csv_values(args.deterministic_scratch_mib, int)
    if any(value <= 0 for value in rps_values):
        raise SystemExit("rows-per-simdgroup values must be positive")
    if any(value not in (32, 64, 128, 256) for value in thread_values):
        raise SystemExit("threads-per-threadgroup values must be 32,64,128,256")
    if any(value not in ("atomic", "simdgroup", "deterministic")
           for value in accumulation_values):
        raise SystemExit("accumulations values must be atomic,simdgroup,deterministic")
    if any(value not in ("shared", "private") for value in storage_values):
        raise SystemExit("model-storage values must be shared,private")
    if any(value <= 0 for value in scratch_values):
        raise SystemExit("deterministic-scratch-mib values must be positive")

    workloads = [load_workload(path) for path in args.workloads]
    jobs = []
    for outer_round in range(args.rounds):
        for workload, (rps, threads, accumulation, storage, scratch_mib) in itertools.product(
            workloads,
            itertools.product(rps_values, thread_values, accumulation_values, storage_values,
                              scratch_values),
        ):
            # Scratch budget has no effect outside deterministic mode; do not duplicate
            # identical atomic/SIMD-group measurements when a budget sweep is requested.
            if accumulation != "deterministic" and scratch_mib != scratch_values[0]:
                continue
            jobs.append((outer_round, workload, {
                "rows_per_simdgroup": rps,
                "threads_per_threadgroup": threads,
                "accumulation": accumulation,
                "storage": storage,
                "deterministic_scratch_mib": scratch_mib,
            }))
    random.Random(args.seed).shuffle(jobs)

    results = []
    for index, (outer_round, workload, config) in enumerate(jobs, 1):
        command = [
            str(benchmark), workload["_paths"], workload["_matrix"],
            str(workload["num_groups"]), "--intercepts",
            ",".join(repr(float(value)) for value in workload["intercepts"]),
            "--kernel", str(kernel), "--warmup", str(args.warmup),
            "--iterations", str(args.iterations), "--rows-per-simdgroup",
            str(config["rows_per_simdgroup"]), "--threads-per-threadgroup",
            str(config["threads_per_threadgroup"]), "--accumulation",
            config["accumulation"], "--model-storage", config["storage"],
            "--deterministic-scratch-mib", str(config["deterministic_scratch_mib"]),
        ]
        if workload["_expected"]:
            command.extend(("--expected", workload["_expected"]))
            command.extend(("--max-abs-error",
                            str(float(workload.get("tolerance", 1e-3)))))
        if args.row_limit:
            command.extend(("--row-limit", str(args.row_limit)))
        print(f"[{index}/{len(jobs)}] {workload['name']} {config}", file=sys.stderr,
              flush=True)
        process = subprocess.run(command, text=True, capture_output=True)
        if process.returncode != 0:
            raise SystemExit(
                f"benchmark failed ({process.returncode})\ncommand={command!r}\n"
                f"stdout={process.stdout}\nstderr={process.stderr}"
            )
        try:
            native = json.loads(process.stdout)
        except json.JSONDecodeError as error:
            raise SystemExit(f"invalid native JSON: {error}\n{process.stdout}") from error
        validate_result(native, workload, config, warmup=args.warmup,
                        iterations=args.iterations, row_limit=args.row_limit)
        results.append({
            "sequence": index,
            "outer_round": outer_round,
            "workload_name": workload["name"],
            "workload_manifest": workload["_manifest"],
            "command": command,
            "native": native,
            "native_stderr": process.stderr,
        })

    repo = Path(__file__).resolve().parents[1]
    artifact = {
        "schema": SUITE_SCHEMA,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "macos": command_value("sw_vers"),
            "cpu_brand": command_value("sysctl", "-n", "machdep.cpu.brand_string"),
            "physical_cpus": command_value("sysctl", "-n", "hw.physicalcpu"),
            "logical_cpus": command_value("sysctl", "-n", "hw.logicalcpu"),
            "memory_bytes": command_value("sysctl", "-n", "hw.memsize"),
            "gpu_display_profile": command_value("system_profiler", "SPDisplaysDataType"),
            "power_settings": command_value("pmset", "-g", "custom"),
            "thermal_state": command_value("pmset", "-g", "therm"),
            "thermal_note": (
                "No privileged temperature/power trace was collected; configuration order "
                "is shuffled and outer rounds must be used to expose order/thermal drift."
            ),
            "git_commit": git_value(repo, "rev-parse", "HEAD"),
            "git_status_porcelain": git_value(repo, "status", "--porcelain"),
            "benchmark": str(benchmark),
            "benchmark_sha256": sha256(benchmark),
            "kernel": str(kernel),
            "kernel_sha256": sha256(kernel),
        },
        "design": {
            "warmup": args.warmup,
            "iterations_per_invocation": args.iterations,
            "outer_rounds": args.rounds,
            "row_limit": args.row_limit,
            "order": "deterministic_shuffle",
            "order_seed": args.seed,
            "quantiles": "linear interpolation at p*(n-1), native per invocation",
        },
        "workloads": [{key: value for key, value in workload.items()
                       if not key.startswith("_")} for workload in workloads],
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, delete=False,
                                     prefix=args.output.name + ".", suffix=".tmp") as temp:
        json.dump(artifact, temp, indent=2, sort_keys=True)
        temp.write("\n")
        temp_path = Path(temp.name)
    temp_path.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
