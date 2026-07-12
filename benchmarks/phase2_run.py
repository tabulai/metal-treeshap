#!/usr/bin/env python3
"""Run the persistent native Phase-2 benchmark over a reproducible tuning matrix.

The native executable owns the timed loop; this script only schedules configurations,
captures environment/provenance, validates returned JSON, and writes one suite artifact.
Configuration order is deterministically shuffled within outer-round blocks to reduce
monotonic thermal/order bias while retaining paired temporal replicates.
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
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from phase2_power import load_samples, summarize_jobs

SUITE_SCHEMA = "metal_treeshap.phase2.suite.v1"
RESULT_SCHEMA = "metal_treeshap.phase2.benchmark.v1"
WORKLOAD_SCHEMA = "metal_treeshap.phase2.workload.v1"
ATOMIC_TILING_SWEEP = [0, 256, 512, 1024, 2048, 4096]


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
        return subprocess.run(
            ["git", *args], cwd=repo, text=True, check=True, capture_output=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def command_value(*args: str) -> str | None:
    try:
        return subprocess.run(
            list(args), text=True, check=True, capture_output=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class PowerCapture:
    """Best-effort suite-wide powermetrics capture without an interactive sudo prompt."""

    def __init__(self, output: Path | None, interval_ms: int, use_sudo: bool):
        self.output = output.resolve() if output else None
        self.interval_ms = interval_ms
        self.use_sudo = use_sudo
        self.process: subprocess.Popen | None = None
        self.started_utc: str | None = None
        self.metadata: dict = {
            "requested": output is not None,
            "status": "disabled" if output is None else "pending",
            "scope": "entire shuffled benchmark suite; correlate samples with job timestamps",
            "format": "plist-nul-separated",
            "sample_interval_ms": interval_ms,
        }

    def _command(self) -> tuple[list[str] | None, str | None]:
        if self.output is None:
            return None, "not requested"
        executable = command_value("command", "-v", "powermetrics")
        # `command` is normally a shell builtin. Fall back to the standard macOS path.
        if not executable and Path("/usr/bin/powermetrics").is_file():
            executable = "/usr/bin/powermetrics"
        if not executable:
            return None, "powermetrics is unavailable"
        prefix: list[str] = []
        if os.geteuid() != 0:
            if not self.use_sudo:
                return None, "powermetrics requires root; rerun with --power-sudo"
            sudo = command_value("which", "sudo")
            authorized = subprocess.run(
                [sudo or "sudo", "-n", "true"], text=True, capture_output=True
            )
            if authorized.returncode != 0:
                return None, "passwordless sudo is unavailable; power capture skipped"
            prefix = [sudo or "sudo", "-n"]
        command = prefix + [
            executable,
            "--samplers",
            "cpu_power,gpu_power,thermal",
            "--sample-rate",
            str(self.interval_ms),
            "--format",
            "plist",
            "--buffer-size",
            "1",
            "--output-file",
            str(self.output),
        ]
        return command, None

    def start(self) -> None:
        command, reason = self._command()
        if command is None:
            if self.output is not None:
                self.metadata.update(status="skipped", reason=reason)
            return
        assert self.output is not None
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.unlink(missing_ok=True)
        self.started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
        self.process = subprocess.Popen(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )
        self.metadata.update(
            status="capturing",
            command=command,
            output=str(self.output),
            started_utc=self.started_utc,
        )
        # Detect immediate authorization/sampler failures without delaying benchmark runs.
        time.sleep(0.05)
        if self.process.poll() is not None:
            _, stderr = self.process.communicate()
            self.metadata.update(
                status="skipped",
                reason=(stderr.strip() or "powermetrics exited early"),
                returncode=self.process.returncode,
            )
            self.process = None

    def stop(self) -> None:
        if self.process is not None:
            self.process.send_signal(signal.SIGINT)
            try:
                _, stderr = self.process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                _, stderr = self.process.communicate()
            self.metadata.update(
                stopped_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                returncode=self.process.returncode,
                stderr=stderr.strip(),
            )
            self.process = None
        if (
            self.output is not None
            and self.output.is_file()
            and self.output.stat().st_size
        ):
            self.metadata.update(
                status="captured",
                bytes=self.output.stat().st_size,
                sha256=sha256(self.output),
            )
        elif self.metadata["status"] == "capturing":
            self.metadata.update(
                status="failed", reason="powermetrics produced no samples"
            )


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
    referenced = [
        payload.get("paths"),
        payload.get("matrix"),
        payload.get("expected"),
        payload.get("model"),
    ]
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


def validate_result(
    result: dict,
    workload: dict,
    configuration: dict,
    *,
    warmup: int,
    iterations: int,
    row_limit: int,
) -> None:
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
    if (
        native_workload["source_rows"] != workload["rows"]
        or native_workload["rows"] != expected_rows
        or native_workload["cols"] != workload["cols"]
        or native_workload["groups"] != workload["num_groups"]
    ):
        raise RuntimeError("native result workload shape/groups do not match manifest")
    if (
        native_config["rows_per_simdgroup"] != configuration["rows_per_simdgroup"]
        or native_config["threads_per_threadgroup"]
        != configuration["threads_per_threadgroup"]
        or native_config["deterministic_scratch_mib"]
        != configuration["deterministic_scratch_mib"]
        or native_config["atomic_tile_rows"] != configuration["atomic_tile_rows"]
        or native_config["warmups"] != warmup
        or native_config["iterations"] != iterations
    ):
        raise RuntimeError(
            "native result tuning/iteration configuration does not match request"
        )
    if workload.get("expected") and result.get("accuracy") is None:
        raise RuntimeError("expected workload produced no accuracy metrics")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path, help="phase2_benchmark executable")
    parser.add_argument(
        "workloads",
        nargs="+",
        type=Path,
        help="workload.json files or their containing directories",
    )
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-simdgroup", default="256,1024,4096")
    parser.add_argument("--threads-per-threadgroup", default="64,128,256")
    parser.add_argument("--accumulations", default="atomic,simdgroup")
    parser.add_argument("--model-storage", default="shared,private")
    parser.add_argument("--deterministic-scratch-mib", default="256")
    parser.add_argument(
        "--atomic-tile-rows",
        default="0",
        help="comma-separated atomic row-tile sizes; 0 is one full dispatch",
    )
    parser.add_argument(
        "--atomic-tiling-sweep",
        action="store_true",
        help="sweep full dispatch plus 256,512,1024,2048,4096-row tiles",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--row-limit", type=int, default=0)
    parser.add_argument(
        "--rounds", type=int, default=1, help="outer repeats of each configuration"
    )
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument(
        "--power-output",
        type=Path,
        help="optional raw powermetrics plist trace for the entire suite",
    )
    parser.add_argument("--power-interval-ms", type=int, default=500)
    parser.add_argument(
        "--power-sudo",
        action="store_true",
        help="allow non-interactive sudo -n for powermetrics; never prompts",
    )
    args = parser.parse_args()

    benchmark = args.benchmark.resolve()
    kernel = args.kernel.resolve()
    if not benchmark.is_file() or not os.access(benchmark, os.X_OK):
        raise SystemExit(f"benchmark is not executable: {benchmark}")
    if not kernel.is_file():
        raise SystemExit(f"kernel does not exist: {kernel}")
    if (
        args.warmup < 0
        or args.iterations <= 0
        or args.row_limit < 0
        or args.rounds <= 0
    ):
        raise SystemExit(
            "warmup/row-limit must be nonnegative; iterations/rounds positive"
        )

    rps_values = csv_values(args.rows_per_simdgroup, int)
    thread_values = csv_values(args.threads_per_threadgroup, int)
    accumulation_values = csv_values(args.accumulations)
    storage_values = csv_values(args.model_storage)
    scratch_values = csv_values(args.deterministic_scratch_mib, int)
    atomic_tile_values = (
        ATOMIC_TILING_SWEEP
        if args.atomic_tiling_sweep
        else csv_values(args.atomic_tile_rows, int)
    )
    if any(value <= 0 for value in rps_values):
        raise SystemExit("rows-per-simdgroup values must be positive")
    if any(value not in (32, 64, 128, 256) for value in thread_values):
        raise SystemExit("threads-per-threadgroup values must be 32,64,128,256")
    if any(
        value not in ("atomic", "simdgroup", "deterministic")
        for value in accumulation_values
    ):
        raise SystemExit("accumulations values must be atomic,simdgroup,deterministic")
    if any(value not in ("shared", "private") for value in storage_values):
        raise SystemExit("model-storage values must be shared,private")
    if any(value <= 0 for value in scratch_values):
        raise SystemExit("deterministic-scratch-mib values must be positive")
    if any(value < 0 for value in atomic_tile_values):
        raise SystemExit("atomic-tile-rows values must be nonnegative (0 means full)")
    if len(set(atomic_tile_values)) != len(atomic_tile_values):
        raise SystemExit("atomic-tile-rows values must be unique")
    if args.power_interval_ms <= 0:
        raise SystemExit("power-interval-ms must be positive")

    workloads = [load_workload(path) for path in args.workloads]
    jobs = []
    order_rng = random.Random(args.seed)
    for outer_round in range(args.rounds):
        round_jobs = []
        for workload, (
            rps,
            threads,
            accumulation,
            storage,
            scratch_mib,
        ) in itertools.product(
            workloads,
            itertools.product(
                rps_values,
                thread_values,
                accumulation_values,
                storage_values,
                scratch_values,
            ),
        ):
            # Scratch budget has no effect outside deterministic mode; do not duplicate
            # identical atomic/SIMD-group measurements when a budget sweep is requested.
            if accumulation != "deterministic" and scratch_mib != scratch_values[0]:
                continue
            # Row tiling is an atomic-only experiment. Other modes execute once at 0,
            # including when the caller requested only positive atomic tile sizes.
            tiles = atomic_tile_values if accumulation == "atomic" else [0]
            for atomic_tile_rows in tiles:
                round_jobs.append(
                    (
                        outer_round,
                        workload,
                        {
                            "rows_per_simdgroup": rps,
                            "threads_per_threadgroup": threads,
                            "accumulation": accumulation,
                            "storage": storage,
                            "deterministic_scratch_mib": scratch_mib,
                            "atomic_tile_rows": atomic_tile_rows,
                        },
                    )
                )
        # Treat every outer round as a temporal block and randomize only within that
        # block. A global shuffle can accidentally place most samples for one
        # configuration late in the run, confounding it with machine-wide drift.
        order_rng.shuffle(round_jobs)
        jobs.extend(round_jobs)

    results = []
    power = PowerCapture(args.power_output, args.power_interval_ms, args.power_sudo)
    power.start()
    try:
        for index, (outer_round, workload, config) in enumerate(jobs, 1):
            command = [
                str(benchmark),
                workload["_paths"],
                workload["_matrix"],
                str(workload["num_groups"]),
                "--intercepts",
                ",".join(repr(float(value)) for value in workload["intercepts"]),
                "--kernel",
                str(kernel),
                "--warmup",
                str(args.warmup),
                "--iterations",
                str(args.iterations),
                "--rows-per-simdgroup",
                str(config["rows_per_simdgroup"]),
                "--threads-per-threadgroup",
                str(config["threads_per_threadgroup"]),
                "--accumulation",
                config["accumulation"],
                "--model-storage",
                config["storage"],
                "--deterministic-scratch-mib",
                str(config["deterministic_scratch_mib"]),
                "--atomic-tile-rows",
                str(config["atomic_tile_rows"]),
            ]
            if workload["_expected"]:
                command.extend(("--expected", workload["_expected"]))
                command.extend(
                    ("--max-abs-error", str(float(workload.get("tolerance", 1e-3))))
                )
            if args.row_limit:
                command.extend(("--row-limit", str(args.row_limit)))
            print(
                f"[{index}/{len(jobs)}] {workload['name']} {config}",
                file=sys.stderr,
                flush=True,
            )
            started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
            process = subprocess.run(command, text=True, capture_output=True)
            finished_utc = dt.datetime.now(dt.timezone.utc).isoformat()
            if process.returncode != 0:
                raise SystemExit(
                    f"benchmark failed ({process.returncode})\ncommand={command!r}\n"
                    f"stdout={process.stdout}\nstderr={process.stderr}"
                )
            try:
                native = json.loads(process.stdout)
            except json.JSONDecodeError as error:
                raise SystemExit(
                    f"invalid native JSON: {error}\n{process.stdout}"
                ) from error
            validate_result(
                native,
                workload,
                config,
                warmup=args.warmup,
                iterations=args.iterations,
                row_limit=args.row_limit,
            )
            results.append(
                {
                    "sequence": index,
                    "outer_round": outer_round,
                    "workload_name": workload["name"],
                    "workload_manifest": workload["_manifest"],
                    "command": command,
                    "native": native,
                    "native_stderr": process.stderr,
                    "started_utc": started_utc,
                    "finished_utc": finished_utc,
                }
            )
    finally:
        power.stop()

    if power.metadata["status"] == "captured" and args.power_output is not None:
        try:
            summaries = summarize_jobs(
                load_samples(args.power_output.resolve()), results
            )
        except (OSError, ValueError) as error:
            power.metadata.update(parse_status="failed", parse_error=str(error))
            summaries = [
                {"status": "unavailable", "missing_reason": str(error)} for _ in results
            ]
        else:
            power.metadata["parse_status"] = "ok"
    else:
        reason = power.metadata.get("reason", "power capture was not requested")
        summaries = [
            {"status": "unavailable", "missing_reason": reason} for _ in results
        ]
    for result, summary in zip(results, summaries, strict=True):
        result["power"] = summary

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
            "gpu_display_profile": command_value(
                "system_profiler", "SPDisplaysDataType"
            ),
            "power_settings": command_value("pmset", "-g", "custom"),
            "thermal_state": command_value("pmset", "-g", "therm"),
            "thermal_note": (
                "Configuration order is shuffled and outer rounds must be used to expose "
                "order/thermal drift; see power_trace for optional suite-wide telemetry."
            ),
            "power_trace": power.metadata,
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
            "order": "deterministic_blocked_shuffle",
            "order_seed": args.seed,
            "quantiles": "linear interpolation at p*(n-1), native per invocation",
            "atomic_tile_rows": atomic_tile_values,
            "atomic_tile_rows_semantics": "0 is one full dispatch; positive values tile rows",
        },
        "workloads": [
            {key: value for key, value in workload.items() if not key.startswith("_")}
            for workload in workloads
        ],
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=args.output.parent,
        delete=False,
        prefix=args.output.name + ".",
        suffix=".tmp",
    ) as temp:
        json.dump(artifact, temp, indent=2, sort_keys=True)
        temp.write("\n")
        temp_path = Path(temp.name)
    temp_path.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
