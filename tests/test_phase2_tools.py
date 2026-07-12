"""Portable contracts for Phase-2.1 workload, sweep, power, and optional SHAP tools."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import plistlib
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

from phase2_power import load_samples, summarize_jobs  # noqa: E402
from phase2_cpu_shap import normalize_shap_values  # noqa: E402


def run(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args], check=True, text=True, capture_output=True, env=env
    )


def assert_manifest(path: Path, *, kind: str, cols: int, groups: int) -> dict:
    manifest = json.loads((path / "workload.json").read_text())
    assert manifest["kind"] == kind
    assert manifest["cols"] == cols and manifest["num_groups"] == groups
    for name, expected in manifest["sha256"].items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == expected
    return manifest


def main() -> None:
    checks = 0
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        workload_tool = str(BENCHMARKS / "phase2_workloads.py")

        # The refactor must preserve stress's historical continued-RNG explain matrix.
        stress = root / "stress"
        run(
            workload_tool,
            "stress",
            str(stress),
            "--trees",
            "1",
            "--depth",
            "2",
            "--features",
            "5",
            "--train-rows",
            "32",
            "--rows",
            "7",
            "--seed",
            "42",
        )
        rng = np.random.default_rng(42)
        train = rng.normal(size=(32, 5)).astype(np.float32)
        train[rng.random(train.shape) < 0.03] = np.nan
        rng.normal(0.0, 0.15, size=32)  # target noise consumed before explain rows
        legacy_x = rng.normal(size=(7, 5)).astype(np.float32)
        legacy_x[rng.random(legacy_x.shape) < 0.03] = np.nan
        np.testing.assert_allclose(
            np.loadtxt(stress / "X.csv", delimiter=",", ndmin=2),
            legacy_x,
            rtol=5e-9,
            atol=5e-9,
        )
        checks += 1

        wide = root / "wide"
        run(
            workload_tool,
            "wide",
            str(wide),
            "--trees",
            "2",
            "--depth",
            "2",
            "--features",
            "16",
            "--train-rows",
            "64",
            "--rows",
            "8",
            "--seed",
            "7",
        )
        assert_manifest(
            wide, kind="deterministic_xgboost_wide_features", cols=16, groups=1
        )
        checks += 1

        multiclass = root / "multiclass"
        run(
            workload_tool,
            "multiclass",
            str(multiclass),
            "--trees",
            "2",
            "--depth",
            "2",
            "--features",
            "8",
            "--classes",
            "3",
            "--train-rows",
            "64",
            "--rows",
            "8",
            "--seed",
            "8",
        )
        manifest = assert_manifest(
            multiclass, kind="deterministic_xgboost_multiclass", cols=8, groups=3
        )
        assert np.loadtxt(
            multiclass / "expected.csv", delimiter=",", ndmin=2
        ).shape == (8, 27)
        assert manifest["classes"] == 3
        checks += 2

        # SHAP layout normalization covers old list and new feature-last APIs.
        old = [np.full((2, 3), 1, np.float32), np.full((2, 3), 2, np.float32)]
        normalized = normalize_shap_values(old, 2, 3, 2)
        assert normalized.shape == (2, 2, 3) and normalized[0, 1, 0] == 2
        new = np.stack(old, axis=2)
        np.testing.assert_array_equal(normalize_shap_values(new, 2, 3, 2), normalized)
        checks += 2

        # Optional SHAP absence/disable is a successful, structured skip.
        disabled = root / "shap-disabled.json"
        env = dict(os.environ, METAL_TREESHAP_DISABLE_SHAP="1")
        run(
            str(BENCHMARKS / "phase2_cpu_shap.py"),
            str(wide / "model.json"),
            str(wide / "X.csv"),
            "--output",
            str(disabled),
            env=env,
        )
        assert json.loads(disabled.read_text())["status"] == "skipped"
        checks += 1

        # Synthetic plist samples exercise interval overlap and missing-data reporting.
        power = root / "power.plist"
        sample_end = dt.datetime(2026, 7, 12, 12, 0, 1)
        samples = [
            {
                "timestamp": sample_end,
                "elapsed_ns": 1_000_000_000,
                "processor": {"gpu_power": 2000.0, "cpu_power": 1000.0},
                "thermal_pressure": "Nominal",
            },
            {
                "timestamp": sample_end + dt.timedelta(seconds=1),
                "elapsed_ns": 1_000_000_000,
                "processor": {"gpu_power": 4000.0, "cpu_power": 2000.0},
                "thermal_pressure": "Moderate",
            },
        ]
        power.write_bytes(
            b"\0".join(plistlib.dumps(sample) for sample in samples) + b"\0"
        )
        jobs = [
            {
                "started_utc": "2026-07-12T12:00:00.500000+00:00",
                "finished_utc": "2026-07-12T12:00:01.500000+00:00",
            },
            {
                "started_utc": "2026-07-12T12:00:03+00:00",
                "finished_utc": "2026-07-12T12:00:04+00:00",
            },
        ]
        summaries = summarize_jobs(load_samples(power), jobs)
        assert summaries[0]["gpu_mean_mw"] == 3000.0
        assert summaries[0]["gpu_estimated_energy_j"] == 3.0
        assert summaries[1]["status"] == "unavailable"
        checks += 3

        # End-to-end runner job construction: six atomic tiles, one deterministic full.
        fake = root / "fake_benchmark.py"
        fake.write_text("""#!/usr/bin/env python3
import csv,json,sys
a=sys.argv[1:]
def v(flag): return a[a.index(flag)+1]
rows=sum(1 for r in csv.reader(open(a[1])) if r)
cols=len(next(csv.reader(open(a[1]))))
print(json.dumps({'schema':'metal_treeshap.phase2.benchmark.v1','status':'ok',
'workload':{'source_rows':rows,'rows':rows,'cols':cols,'groups':int(a[2])},
'configuration':{'rows_per_simdgroup':int(v('--rows-per-simdgroup')),
'threads_per_threadgroup':int(v('--threads-per-threadgroup')),
'accumulation':v('--accumulation'),'model_storage':v('--model-storage'),
'deterministic_scratch_mib':int(v('--deterministic-scratch-mib')),
'atomic_tile_rows':int(v('--atomic-tile-rows')),'warmups':int(v('--warmup')),
'iterations':int(v('--iterations'))},'accuracy':{}}))
""")
        fake.chmod(0o755)
        kernel = root / "kernel.metal"
        kernel.write_text("// provenance only\n")
        suite = root / "suite.json"
        run(
            str(BENCHMARKS / "phase2_run.py"),
            str(fake),
            str(wide),
            "--kernel",
            str(kernel),
            "--output",
            str(suite),
            "--rows-per-simdgroup",
            "256",
            "--threads-per-threadgroup",
            "32",
            "--accumulations",
            "atomic,deterministic",
            "--model-storage",
            "shared",
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--atomic-tiling-sweep",
        )
        result = json.loads(suite.read_text())
        tiles = sorted(
            item["native"]["configuration"]["atomic_tile_rows"]
            for item in result["results"]
            if item["native"]["configuration"]["accumulation"] == "atomic"
        )
        assert tiles == [0, 256, 512, 1024, 2048, 4096]
        assert len(result["results"]) == 7
        assert all(
            item["power"]["status"] == "unavailable" for item in result["results"]
        )
        checks += 3

    print(f"ALL {checks} PHASE-2.1 TOOL TESTS PASSED")


if __name__ == "__main__":
    main()
