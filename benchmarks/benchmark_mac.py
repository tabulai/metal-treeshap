"""Benchmark runner for MetalTreeShap (proposal §7).

Mirrors upstream gputreeshap/benchmark/benchmark.py: same datasets (adult, covtype,
cal_housing, fashion_mnist), same model grid (10/100/1000 rounds x depth 3/8/16), and same 10K
explain rows. This aligns the workload shape with the published V100 table, but categorical
encoding and hardware/software differences mean those numbers are contextual, not universally
direct comparisons.

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
  * The Metal path uses the public compile-once ``MetalTreeExplainer`` API. Model extraction,
    preprocessing, pipeline creation, and persistent-buffer setup are timed once; warmups and
    repeated explain calls are reported separately.

Energy: pass ``--power-output TRACE --power-sudo`` after ``sudo -v``. The runner uses
homogeneous CPU/Metal blocks with sampler-interval lead-in and tail conditioning so a
powermetrics interval cannot be contaminated by the other engine.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import importlib.resources
import json
import platform
import random
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import sklearn
import xgboost as xgb
from sklearn import datasets

try:
    from .phase2_power import load_samples, summarize_jobs
    from .phase2_run import PowerCapture
except ImportError:  # Direct ``python benchmarks/benchmark_mac.py`` execution.
    from phase2_power import load_samples, summarize_jobs
    from phase2_run import PowerCapture

MODEL_GRID = {"small": (10, 3), "med": (100, 8), "large": (1000, 16)}
DATASET_SOURCES = {
    "adult": "OpenML adult version 1",
    "covtype": "scikit-learn fetch_covtype",
    "cal_housing": "scikit-learn fetch_california_housing",
    "fashion_mnist": "OpenML Fashion-MNIST version 1",
}
ROOT = Path(__file__).resolve().parents[1]
_PROVENANCE: dict | None = None


def _encode_numeric(X):
    """Ordinal-encode object/category columns; keep NaN as missing."""
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
        X, y = datasets.fetch_openml("adult", version=1, return_X_y=True)
        return _encode_numeric(X), np.array([yi != "<=50K" for yi in y]), "binary:logistic"
    if name == "fashion_mnist":
        X, y = datasets.fetch_openml("Fashion-MNIST", version=1, return_X_y=True)
        return _encode_numeric(X), y.astype(np.int64), "multi:softmax"
    raise ValueError(name)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(repr(contiguous.shape).encode())
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_provenance() -> dict:
    global _PROVENANCE
    if _PROVENANCE is not None:
        return dict(_PROVENANCE)

    def command(*args: str) -> str | None:
        try:
            return subprocess.run(
                list(args), text=True, check=True, capture_output=True, cwd=ROOT
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    package_files: dict[str, str] = {}
    try:
        package = importlib.resources.files("metal_treeshap")
        for name in ("treeshap.metal", "explainer.py", "_extract_paths.py"):
            resource = package.joinpath(name)
            if resource.is_file():
                package_files[name] = hashlib.sha256(resource.read_bytes()).hexdigest()
        for resource in package.iterdir():
            if resource.name.startswith("_native") and resource.name.endswith(".so"):
                package_files[resource.name] = hashlib.sha256(
                    resource.read_bytes()
                ).hexdigest()
        package_version = importlib.metadata.version("metal-treeshap")
    except (ImportError, importlib.metadata.PackageNotFoundError, OSError):
        package_version = "unavailable"

    display_name = None
    display_json = command("system_profiler", "SPDisplaysDataType", "-json")
    if display_json:
        try:
            displays = json.loads(display_json).get("SPDisplaysDataType", [])
            if displays:
                display_name = displays[0].get("sppci_model") or displays[0].get("_name")
        except (AttributeError, json.JSONDecodeError):
            pass

    git_commit = command("git", "rev-parse", "HEAD")
    git_status = command("git", "status", "--porcelain", "--untracked-files=all")
    source_files = {
        "benchmark_mac.py": _sha256_path(Path(__file__).resolve()),
        "shaders/treeshap.metal": _sha256_path(ROOT / "shaders" / "treeshap.metal"),
    }
    provenance = {
        "package_version": package_version,
        "package_file_sha256": package_files,
        "source_file_sha256": source_files,
        "git_commit": git_commit or "unavailable",
        "git_dirty": git_status is None or bool(git_status),
        "platform": platform.platform(),
        "macos_version": platform.mac_ver()[0],
        "machine": platform.machine(),
        "metal_device": display_name or "unavailable",
    }
    # Dirty state is evidence, but not an implementation compatibility key: writing the
    # output inside the repository can itself make a subsequent resume process dirty.
    compatibility = {key: value for key, value in provenance.items() if key != "git_dirty"}
    provenance["fingerprint_sha256"] = hashlib.sha256(
        json.dumps(compatibility, sort_keys=True).encode()
    ).hexdigest()
    _PROVENANCE = provenance
    return dict(provenance)


def _prepare(dataset: str, size: str, nrows: int, nthread: int, seed: int):
    fetch_started = time.perf_counter()
    X, y, objective = fetch(dataset)
    fetch_s = time.perf_counter() - fetch_started
    rounds, depth = MODEL_GRID[size]
    params = {"objective": objective, "max_depth": depth, "eta": 0.01,
              "tree_method": "hist", "seed": seed}
    if nthread:
        params["nthread"] = nthread
    if objective == "multi:softmax":
        params["num_class"] = int(np.max(y) + 1)
    training_started = time.perf_counter()
    model = xgb.train(params, xgb.QuantileDMatrix(X, y), rounds)
    training_s = time.perf_counter() - training_started

    rs = np.random.RandomState(seed)
    Xt = X[rs.randint(0, X.shape[0], size=nrows)]
    model_json = bytes(model.save_raw(raw_format="json"))
    metadata = {
        "model": f"{dataset}-{size}",
        "dataset": dataset,
        "dataset_source": DATASET_SOURCES[dataset],
        "size": size,
        "rows": int(nrows),
        "training_rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "rounds": int(rounds),
        "depth": int(depth),
        "nthread": int(nthread),
        "seed": int(seed),
        "fetch_s": float(fetch_s),
        "training_s": float(training_s),
        "tree_count": len(model.get_dump()),
        "dataset_sha256": _hash_arrays(np.asarray(X), np.asarray(y)),
        "explain_matrix_sha256": _hash_arrays(np.asarray(Xt)),
        "model_json_sha256": hashlib.sha256(model_json).hexdigest(),
        "xgboost_version": xgb.__version__,
        "scikit_learn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "implementation": _implementation_provenance(),
    }
    return model, np.ascontiguousarray(Xt, dtype=np.float32), metadata


def _run_cell(dataset: str, size: str, nrows: int, niter: int,
              devices: tuple[str, ...], *, warmup: int, nthread: int,
              seed: int, power_block_s: float = 0.0,
              power_block_rounds: int = 0, power_guard_s: float = 0.0,
              power_sampler_interval_s: float = 0.0) -> dict:
    if nrows <= 0 or niter <= 0 or warmup < 0 or nthread < 0:
        raise ValueError("nrows/niter must be positive; warmup/nthread must be non-negative")
    if not devices or any(device not in {"cpu", "metal"} for device in devices):
        raise ValueError(f"unsupported devices: {devices}")
    if (power_block_s < 0 or power_block_rounds < 0 or power_guard_s < 0
            or power_sampler_interval_s < 0):
        raise ValueError("power block duration/rounds/guard must be non-negative")

    model, Xt, metadata = _prepare(dataset, size, nrows, nthread, seed)
    runners: dict[str, object] = {}
    device_metadata: dict[str, dict] = {}

    if "cpu" in devices:
        setup_started = time.perf_counter()
        dtest = xgb.DMatrix(Xt)
        setup_s = time.perf_counter() - setup_started
        runners["cpu"] = lambda: model.predict(dtest, pred_contribs=True)
        device_metadata["cpu"] = {"setup_s": float(setup_s)}

    if "metal" in devices:
        from metal_treeshap import MetalTreeExplainer

        setup_started = time.perf_counter()
        compiled = MetalTreeExplainer.from_xgboost(model)
        setup_s = time.perf_counter() - setup_started
        runners["metal"] = lambda: compiled.explain(Xt)
        device_metadata["metal"] = {
            "setup_s": float(setup_s),
            "num_bins": compiled.num_bins,
            "storage_mode": compiled.storage_mode,
        }

    outputs: dict[str, np.ndarray] = {}
    for _ in range(warmup):
        for device in devices:
            outputs[device] = np.asarray(runners[device]())

    samples: dict[str, list[dict]] = {device: [] for device in devices}
    rng = random.Random(seed)
    suite_started = _utc_now()
    for iteration in range(niter):
        order = list(devices)
        rng.shuffle(order)
        for order_index, device in enumerate(order):
            started_utc = _utc_now()
            started = time.perf_counter()
            outputs[device] = np.asarray(runners[device]())
            elapsed = time.perf_counter() - started
            samples[device].append({
                "iteration": iteration,
                "order_index": order_index,
                "started_utc": started_utc,
                "finished_utc": _utc_now(),
                "seconds": float(elapsed),
            })
    suite_finished = _utc_now()

    margin = np.asarray(model.predict(xgb.DMatrix(Xt), output_margin=True))
    results = []
    for device in devices:
        values = outputs[device]
        row_sums = np.sum(values, axis=-1)
        shaped_margin = margin.reshape(row_sums.shape)
        local_accuracy_error = float(np.max(np.abs(row_sums - shaped_margin)))
        seconds = [sample["seconds"] for sample in samples[device]]
        result = {
            **metadata,
            **device_metadata[device],
            "device": device,
            "warmup": int(warmup),
            "iterations": int(niter),
            "samples": samples[device],
            "mean_s": float(np.mean(seconds)),
            "median_s": float(np.median(seconds)),
            "std_s": float(np.std(seconds)),
            "rows_per_s": nrows / float(np.median(seconds)),
            "max_local_accuracy_error": local_accuracy_error,
            "local_accuracy": local_accuracy_error < 1e-3,
            "output_shape": list(values.shape),
            "output_sha256": _hash_arrays(values),
        }
        if not result["local_accuracy"]:
            raise RuntimeError(
                f"{dataset}-{size}-{device}: local-accuracy error "
                f"{local_accuracy_error:.6g} exceeds 1e-3"
            )
        results.append(result)

    comparison = None
    if set(devices) == {"cpu", "metal"}:
        cpu = outputs["cpu"].reshape(-1)
        metal = outputs["metal"].reshape(-1)
        if cpu.shape != metal.shape:
            raise RuntimeError(f"CPU/Metal output shapes differ: {cpu.shape} vs {metal.shape}")
        max_error = float(np.max(np.abs(cpu.astype(np.float64) - metal)))
        if max_error >= 1e-3:
            raise RuntimeError(f"CPU/Metal max attribution error {max_error:.6g} exceeds 1e-3")
        by_device = {result["device"]: result for result in results}
        comparison = {
            "max_abs_attribution_error": max_error,
            "metal_speedup_vs_cpu": (
                by_device["cpu"]["median_s"] / by_device["metal"]["median_s"]
            ),
        }

    power_jobs = []
    if power_block_s > 0 and power_block_rounds > 0:
        # Powermetrics reports interval averages. Short, interleaved calls would mix CPU
        # and GPU activity inside one sample. Measure homogeneous repeated-call blocks
        # with idle guards so each engine has multiple fully contained samples.
        power_rng = random.Random(seed ^ 0x5A17)
        for block_round in range(power_block_rounds):
            order = list(devices)
            power_rng.shuffle(order)
            for order_index, device in enumerate(order):
                if power_guard_s:
                    time.sleep(power_guard_s)
                conditioning_calls = 0
                conditioning_started = time.perf_counter()
                while time.perf_counter() - conditioning_started < power_sampler_interval_s:
                    runners[device]()
                    conditioning_calls += 1
                started_utc = _utc_now()
                started = time.perf_counter()
                calls = 0
                while time.perf_counter() - started < power_block_s:
                    runners[device]()
                    calls += 1
                seconds = time.perf_counter() - started
                power_jobs.append({
                    "dataset": dataset,
                    "size": size,
                    "device": device,
                    "rows": int(nrows),
                    "explained_rows": int(nrows * calls),
                    "iteration": block_round,
                    "block_round": block_round,
                    "order_index": order_index,
                    "started_utc": started_utc,
                    "finished_utc": _utc_now(),
                    "seconds": float(seconds),
                    "calls": calls,
                    "power_block": True,
                    "lead_in_s": power_sampler_interval_s,
                    "tail_s": power_sampler_interval_s,
                    "conditioning_calls": conditioning_calls,
                })
                tail_started = time.perf_counter()
                while time.perf_counter() - tail_started < power_sampler_interval_s:
                    runners[device]()
        if power_guard_s:
            time.sleep(power_guard_s)
    else:
        for result in results:
            for sample in result["samples"]:
                power_jobs.append({
                    "dataset": dataset,
                    "size": size,
                    "device": result["device"],
                    "rows": int(nrows),
                    "explained_rows": int(nrows),
                    **sample,
                })
    power_jobs.sort(key=lambda job: job["started_utc"])

    return {
        "schema": "metal_treeshap.realdata_cell.v1",
        "started_utc": suite_started,
        "finished_utc": suite_finished,
        "configuration_order": "seeded_random_within_iteration",
        "power_design": (
            "homogeneous_repeated_call_blocks_with_idle_guards"
            if power_block_s > 0 and power_block_rounds > 0
            else "timed_call_windows_only"
        ),
        "devices": results,
        "comparison": comparison,
        "power_jobs": power_jobs,
    }


def bench(dataset: str, size: str, nrows: int, niter: int, device: str,
          *, warmup: int = 1, nthread: int = 0, seed: int = 432) -> dict:
    """Backwards-compatible single-device entry point."""
    return _run_cell(dataset, size, nrows, niter, (device,), warmup=warmup,
                     nthread=nthread, seed=seed)["devices"][0]


def _suite_payload(cells: list[dict], *, power_trace: dict | None = None,
                   power_summary: list[dict] | None = None) -> dict:
    payload = {
        "schema": "metal_treeshap.realdata_suite.v1",
        "generated_utc": _utc_now(),
        "cells": cells,
        # Exact, non-overlapping timed-call windows consumed directly by
        # benchmarks/phase2_power.py when a privileged powermetrics trace exists.
        "results": [job for cell in cells for job in cell["power_jobs"]],
    }
    if power_trace is not None:
        payload["power_trace"] = power_trace
    if power_summary is not None:
        payload["power_summary"] = power_summary
    return payload


def _write_suite(path: Path, cells: list[dict], *, power_trace: dict | None = None,
                 power_summary: list[dict] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, prefix=path.name + ".", suffix=".tmp"
    ) as target:
        json.dump(
            _suite_payload(cells, power_trace=power_trace,
                           power_summary=power_summary),
            target,
            indent=2,
            sort_keys=True,
        )
        target.write("\n")
        temporary = Path(target.name)
    temporary.replace(path)


def _power_blocks_enabled(power_output: Path | None, metadata: dict) -> bool:
    """Run costly conditioning blocks only while a sampler is actually recording."""
    return power_output is not None and metadata.get("status") == "capturing"


def _resume_cells(path: Path, args: argparse.Namespace) -> list[dict]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "metal_treeshap.realdata_suite.v1":
        raise SystemExit(f"cannot resume unsupported artifact: {path}")
    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise SystemExit(f"cannot resume artifact without cells: {path}")
    if payload.get("power_trace", {}).get("requested") is True:
        raise SystemExit(
            "cannot resume an artifact that requested power evidence; "
            "start a new output/trace pair"
        )
    current_versions = {
        "xgboost_version": xgb.__version__,
        "scikit_learn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    prior_versions: dict[str, str] | None = None
    current_fingerprint = _implementation_provenance()["fingerprint_sha256"]
    for cell in cells:
        devices = cell.get("devices")
        if not isinstance(devices, list) or not devices:
            raise SystemExit(f"cannot resume malformed cell in {path}")
        example = devices[0]
        expected = {
            "rows": args.nrows,
            "warmup": args.warmup,
            "iterations": args.niter,
            "nthread": args.nthread,
            "seed": args.seed,
        }
        if any(example.get(key) != value for key, value in expected.items()):
            raise SystemExit(f"resume configuration differs for {example.get('model')}")
        expected_devices = {"cpu", "metal"} if args.device == "both" else {args.device}
        actual_devices = {result.get("device") for result in devices}
        if actual_devices != expected_devices:
            raise SystemExit(f"resume device set differs for {example.get('model')}")
        saved_fingerprint = example.get("implementation", {}).get("fingerprint_sha256")
        if saved_fingerprint != current_fingerprint:
            raise SystemExit(f"resume implementation differs for {example.get('model')}")
        saved_versions = {
            key: example.get(key) for key in current_versions
        }
        if saved_versions != current_versions:
            raise SystemExit(
                f"resume software versions differ for {example.get('model')}: "
                f"saved={saved_versions}, current={current_versions}"
            )
        if prior_versions is not None and saved_versions != prior_versions:
            raise SystemExit(f"resume artifact mixes software versions in {path}")
        prior_versions = saved_versions
    return cells


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="cal_housing,adult")
    ap.add_argument("--sizes", default="small,med")
    ap.add_argument("--nrows", type=int, default=10000)
    ap.add_argument("--niter", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=432)
    ap.add_argument("--nthread", type=int, default=0,
                    help="XGBoost CPU threads; 0 uses its default")
    ap.add_argument("--device", default="cpu", choices=["cpu", "metal", "both"])
    ap.add_argument("--output", type=Path,
                    help="optional JSON artifact; stdout always receives one JSON object/line")
    output_mode = ap.add_mutually_exclusive_group()
    output_mode.add_argument("--resume", action="store_true",
                             help="resume missing cells from a compatible output artifact")
    output_mode.add_argument("--force", action="store_true",
                             help="replace an existing output artifact")
    ap.add_argument("--power-output", type=Path,
                    help="optional suite-wide powermetrics plist trace")
    ap.add_argument("--power-interval-ms", type=int, default=250)
    ap.add_argument("--power-sudo", action="store_true",
                    help="use already-authorized non-interactive sudo for powermetrics")
    ap.add_argument("--power-block-seconds", type=float, default=2.0,
                    help="minimum homogeneous repeated-call block per engine")
    ap.add_argument("--power-block-rounds", type=int, default=3)
    ap.add_argument("--power-guard-seconds", type=float, default=0.5,
                    help="idle guard between engine blocks")
    args = ap.parse_args()
    if (args.resume or args.force) and args.output is None:
        raise SystemExit("--resume/--force requires --output")
    if args.power_output is not None and args.output is None:
        raise SystemExit("--power-output requires --output")
    if args.resume and args.power_output is not None:
        raise SystemExit("power capture cannot be resumed; use a new output/trace pair")
    if args.power_output is not None:
        if args.power_interval_ms <= 0:
            raise SystemExit("--power-interval-ms must be positive")
        if args.power_output.resolve() == args.output.resolve():
            raise SystemExit("--power-output and --output must be different paths")
        if args.power_output.exists() and not args.force:
            raise SystemExit("power output exists (use --force to replace it)")
        minimum_block_s = 4.0 * args.power_interval_ms / 1000.0
        if args.power_block_seconds < minimum_block_s:
            raise SystemExit(
                f"--power-block-seconds must be at least {minimum_block_s:g} "
                "(four sampler intervals)"
            )
        if args.power_block_rounds <= 0 or args.power_guard_seconds < 0:
            raise SystemExit("power block rounds must be positive and guard non-negative")
    if args.output and args.output.exists():
        if args.resume:
            cells = _resume_cells(args.output, args)
        elif args.force:
            cells = []
        else:
            raise SystemExit(f"output exists: {args.output} (use --resume or --force)")
    else:
        cells = []
    completed = {
        (cell["devices"][0]["dataset"], cell["devices"][0]["size"])
        for cell in cells
    }
    power = PowerCapture(args.power_output, args.power_interval_ms, args.power_sudo)
    power.start()
    power_blocks_enabled = _power_blocks_enabled(args.power_output, power.metadata)
    try:
        for d in args.datasets.split(","):
            for s in args.sizes.split(","):
                if (d, s) in completed:
                    print(json.dumps({"status": "resume-skip", "dataset": d, "size": s},
                                     sort_keys=True), flush=True)
                    continue
                devices = (("cpu", "metal") if args.device == "both" else (args.device,))
                cell = _run_cell(d, s, args.nrows, args.niter, devices,
                                 warmup=args.warmup, nthread=args.nthread, seed=args.seed,
                                 power_block_s=(args.power_block_seconds
                                                if power_blocks_enabled else 0.0),
                                 power_block_rounds=(args.power_block_rounds
                                                     if power_blocks_enabled else 0),
                                 power_guard_s=(args.power_guard_seconds
                                                if power_blocks_enabled else 0.0),
                                 power_sampler_interval_s=(args.power_interval_ms / 1000.0
                                                           if power_blocks_enabled else 0.0))
                cells.append(cell)
                for result in cell["devices"]:
                    print(json.dumps(result, sort_keys=True), flush=True)
                if args.output:
                    _write_suite(args.output, cells, power_trace=power.metadata)
    except BaseException:
        power.stop()
        if args.output:
            _write_suite(args.output, cells, power_trace=power.metadata)
        raise
    else:
        power.stop()

    if args.output:
        jobs = [job for cell in cells for job in cell["power_jobs"]]
        if power.metadata["status"] == "captured" and args.power_output is not None:
            try:
                power_summary = summarize_jobs(load_samples(args.power_output), jobs)
            except (OSError, ValueError) as error:
                power.metadata.update(parse_status="failed", parse_error=str(error))
                power_summary = [
                    {"status": "unavailable", "missing_reason": str(error)} for _ in jobs
                ]
            else:
                power.metadata["parse_status"] = "ok"
        else:
            reason = power.metadata.get("reason", "power capture was not requested")
            power_summary = [
                {"status": "unavailable", "missing_reason": reason} for _ in jobs
            ]
        _write_suite(
            args.output,
            cells,
            power_trace=power.metadata,
            power_summary=power_summary,
        )
