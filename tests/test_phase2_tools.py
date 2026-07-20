"""Portable contracts for Phase-2.1 workload, sweep, power, and optional SHAP tools."""

from __future__ import annotations

import datetime as dt
import argparse
import copy
import hashlib
import json
import os
import plistlib
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import numpy as np
    import jsonschema
except ImportError:
    if __name__ != "__main__":  # pytest without the tooling deps: skip, don't error
        import pytest

        pytest.skip(
            "the phase2 tooling suite requires numpy and jsonschema",
            allow_module_level=True,
        )
    raise

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

from phase2_power import load_samples, summarize_jobs  # noqa: E402
from phase2_cpu_shap import normalize_shap_values  # noqa: E402
import benchmark_mac  # noqa: E402


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


def write_raw_fixture(
    path: Path, *, x_csv: str, expected_csv: str, meta: dict
) -> None:
    path.mkdir()
    (path / "paths.csv").write_text(
        "path_idx,feature_idx,group,lower,upper,is_missing,zero_fraction,v\n"
        "0,-1,0,-inf,inf,1,1.0,0.5\n"
    )
    (path / "X.csv").write_text(x_csv)
    (path / "expected.csv").write_text(expected_csv)
    (path / "meta.json").write_text(json.dumps(meta))


def assert_fixture_rejected_without_touching_output(
    workload_tool: str,
    source: Path,
    output: Path,
    expected_error: str,
    *extra_args: str,
) -> None:
    output.mkdir()
    sentinel = output / "keep-me.txt"
    sentinel.write_text("existing output must survive source validation\n")
    result = subprocess.run(
        [
            sys.executable,
            workload_tool,
            "fixture",
            str(source),
            str(output),
            "--force",
            *extra_args,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, result.stdout
    assert expected_error in result.stderr, result.stderr
    assert sentinel.read_text() == "existing output must survive source validation\n"
    assert list(output.iterdir()) == [sentinel]


def assert_power_windows(result: dict, expected_samples: int) -> None:
    """Check the UTC interval contract consumed by phase2_power.summarize_jobs."""
    started = dt.datetime.fromisoformat(result["started_utc"])
    finished = dt.datetime.fromisoformat(result["finished_utc"])
    assert started.utcoffset() == dt.timedelta(0)
    assert finished.utcoffset() == dt.timedelta(0)
    assert started <= finished

    windows = result["sample_windows_utc"]
    assert len(windows) == expected_samples
    assert [window["elapsed_s"] for window in windows] == result["timing_s"][
        "samples"
    ]
    previous = started
    for window in windows:
        sample_start = dt.datetime.fromisoformat(window["started_utc"])
        sample_finish = dt.datetime.fromisoformat(window["finished_utc"])
        assert sample_start.utcoffset() == dt.timedelta(0)
        assert sample_finish.utcoffset() == dt.timedelta(0)
        assert previous <= sample_start <= sample_finish <= finished
        assert window["elapsed_s"] >= 0
        previous = sample_finish

    # The aggregate result can be passed directly to the existing power correlator.
    duration_s = max((finished - started).total_seconds(), 1e-6)
    summary = summarize_jobs(
        [
            {
                "timestamp": finished,
                "elapsed_ns": int((duration_s + 1.0) * 1e9),
                "processor": {"cpu_power": 1500.0},
                "thermal_pressure": "Nominal",
            }
        ],
        [result],
    )[0]
    assert summary["status"] == "ok"
    assert np.isclose(summary["cpu_mean_mw"], 1500.0)
    assert summary["job"]["rows"] == result["rows"]


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

        # CPU baselines expose aggregate and exact-call UTC intervals in the same
        # contract as the Metal suite, so one powermetrics trace can correlate both.
        cpu_xgboost = root / "cpu-xgboost.json"
        run(
            str(BENCHMARKS / "phase2_cpu_xgboost.py"),
            str(wide / "model.json"),
            str(wide / "X.csv"),
            "--expected",
            str(wide / "expected.csv"),
            "--output",
            str(cpu_xgboost),
            "--row-limits",
            "8",
            "--warmup",
            "0",
            "--iterations",
            "2",
            "--nthread",
            "1",
        )
        cpu_xgboost_payload = json.loads(cpu_xgboost.read_text())
        assert cpu_xgboost_payload["schema"] == "metal_treeshap.phase2.cpu_xgboost.v1"
        assert_power_windows(cpu_xgboost_payload["results"][0], 2)
        checks += 3

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

        # classes == features: the square ndarray must follow SHAP's documented
        # feature-last [samples, features, outputs] contract (the return TYPE is the
        # discriminator — legacy layouts arrive as per-class lists), so it is
        # transposed, never misread as group-major.
        square = np.arange(4 * 3 * 3, dtype=np.float32).reshape(4, 3, 3)
        np.testing.assert_array_equal(
            normalize_shap_values(square, 4, 3, 3), np.transpose(square, (0, 2, 1))
        )
        checks += 1

        # Fixture materialization must refuse ancestor/descendant source/output
        # overlap: --force recursively deletes the output, so an output containing
        # the source would destroy the fixture before it is ever read.
        fixture_src = root / "overlap-fixture" / "src"
        fixture_src.mkdir(parents=True)
        (fixture_src / "paths.csv").write_text(
            "path_idx,feature_idx,group,lower,upper,is_missing,zero_fraction,v\n"
            "0,-1,0,-inf,inf,1,1.0,0.5\n"
        )
        (fixture_src / "X.csv").write_text("0.0\n")
        bad_outputs = [fixture_src.parent, fixture_src, fixture_src / "nested"]
        # On case-insensitive filesystems (macOS APFS default), a case-variant
        # spelling of an ancestor is the SAME directory; the guard compares
        # filesystem identity so it must catch that spelling too.
        case_variant = fixture_src.parent.with_name(fixture_src.parent.name.upper())
        if case_variant.exists():
            bad_outputs.append(case_variant)
        for bad_output in bad_outputs:
            overlap = subprocess.run(
                [sys.executable, str(BENCHMARKS / "phase2_workloads.py"), "fixture",
                 str(fixture_src), str(bad_output), "--force"],
                capture_output=True, text=True)
            assert overlap.returncode != 0, bad_output
            assert "must not overlap" in overlap.stderr, (bad_output, overlap.stderr)
        assert (fixture_src / "paths.csv").exists(), "overlap check deleted the source"
        checks += len(bad_outputs) + 1

        # Generated workload directories carry their metadata in workload.json;
        # fixture materialization must consume it (regression: requiring meta.json
        # broke hot/stress/wide -> fixture and left partial outputs behind).
        materialized_out = root / "materialized-wide"
        run(
            str(BENCHMARKS / "phase2_workloads.py"), "fixture", str(wide),
            str(materialized_out), "--force",
        )
        materialized = json.loads((materialized_out / "workload.json").read_text())
        source_manifest = json.loads((wide / "workload.json").read_text())
        assert materialized["intercepts"] == source_manifest["intercepts"]
        assert (materialized_out / "paths.csv").exists()
        assert (materialized_out / "X.csv").exists()
        # A rejected fixture must leave NO partial output (validation precedes
        # _prepare_output): a paths.csv source without any manifest is refused cleanly.
        bare_src = root / "bare-fixture"
        bare_src.mkdir()
        (bare_src / "paths.csv").write_text(
            "path_idx,feature_idx,group,lower,upper,is_missing,zero_fraction,v\n"
        )
        (bare_src / "X.csv").write_text("0.0\n")
        bare_out = root / "bare-out"
        bare = subprocess.run(
            [sys.executable, str(BENCHMARKS / "phase2_workloads.py"), "fixture",
             str(bare_src), str(bare_out)],
            capture_output=True, text=True)
        assert bare.returncode != 0 and "intercepts" in bare.stderr
        assert not bare_out.exists(), "rejected fixture left a partial output"
        checks += 5

        # Generated workloads use workload.json rather than meta.json. Both an
        # analytic hot workload and an extracted-model stress workload remain valid
        # fixture sources, preserve their intercepts, and tile X/oracle rows in lockstep.
        hot = root / "hot"
        run(
            workload_tool,
            "hot",
            str(hot),
            "--trees",
            "2",
            "--rows",
            "3",
        )
        materialized_hot = root / "materialized-hot"
        run(
            workload_tool,
            "fixture",
            str(hot),
            str(materialized_hot),
            "--rows",
            "5",
        )
        hot_x = (hot / "X.csv").read_text().splitlines()
        hot_expected = (hot / "expected.csv").read_text().splitlines()
        assert (materialized_hot / "X.csv").read_text().splitlines() == [
            hot_x[index % 3] for index in range(5)
        ]
        assert (materialized_hot / "expected.csv").read_text().splitlines() == [
            hot_expected[index % 3] for index in range(5)
        ]
        hot_manifest = json.loads((hot / "workload.json").read_text())
        tiled_manifest = json.loads((materialized_hot / "workload.json").read_text())
        assert tiled_manifest["intercepts"] == hot_manifest["intercepts"]
        assert tiled_manifest["source_manifest_sha256"] == hashlib.sha256(
            (hot / "workload.json").read_bytes()
        ).hexdigest()

        materialized_stress = root / "materialized-stress"
        run(workload_tool, "fixture", str(stress), str(materialized_stress))
        stress_manifest = json.loads((stress / "workload.json").read_text())
        stress_copy_manifest = json.loads(
            (materialized_stress / "workload.json").read_text()
        )
        assert stress_copy_manifest["intercepts"] == stress_manifest["intercepts"]
        assert stress_copy_manifest["intercepts"] != [0.0]
        assert "model" not in stress_copy_manifest
        assert not (materialized_stress / "model.json").exists()
        checks += 7

        # workload.json references, not conventional filenames, are authoritative.
        # A null expected reference must also ignore a stale expected.csv beside it.
        referenced = root / "referenced-workload"
        run(workload_tool, "hot", str(referenced), "--trees", "1", "--rows", "2")
        referenced_manifest = json.loads((referenced / "workload.json").read_text())
        (referenced / "paths.csv").rename(referenced / "custom-paths.csv")
        (referenced / "X.csv").rename(referenced / "custom-matrix.csv")
        referenced_manifest["paths"] = "custom-paths.csv"
        referenced_manifest["matrix"] = "custom-matrix.csv"
        referenced_manifest["expected"] = None
        hashes = referenced_manifest["sha256"]
        hashes["custom-paths.csv"] = hashes.pop("paths.csv")
        hashes["custom-matrix.csv"] = hashes.pop("X.csv")
        hashes.pop("expected.csv")
        (referenced / "workload.json").write_text(json.dumps(referenced_manifest))
        referenced_output = root / "referenced-output"
        run(workload_tool, "fixture", str(referenced), str(referenced_output))
        assert not (referenced_output / "expected.csv").exists()
        assert json.loads((referenced_output / "workload.json").read_text())[
            "expected"
        ] is None
        checks += 2

        valid_meta = {
            "num_groups": 1,
            "num_features": 1,
            "intercepts": [0.0],
            "tolerance": 1e-3,
        }

        # X and expected rows must align before optional tiling. Independently cycling
        # a one-row oracle over two distinct inputs silently manufactures bad goldens.
        mismatched_rows = root / "mismatched-rows"
        write_raw_fixture(
            mismatched_rows,
            x_csv="-1\n1\n",
            expected_csv="-1,0\n",
            meta=valid_meta,
        )
        assert_fixture_rejected_without_touching_output(
            workload_tool,
            mismatched_rows,
            root / "mismatched-rows-output",
            "expected attribution row count",
        )

        # Every source CSV is parsed before --force can remove an existing output.
        ragged_x = root / "ragged-x"
        write_raw_fixture(
            ragged_x,
            x_csv="1,2\n3\n",
            expected_csv="0,0,0\n0,0,0\n",
            meta={**valid_meta, "num_features": 2},
        )
        assert_fixture_rejected_without_touching_output(
            workload_tool, ragged_x, root / "ragged-output", "ragged CSV"
        )

        wrong_width = root / "wrong-expected-width"
        write_raw_fixture(
            wrong_width,
            x_csv="-1\n1\n",
            expected_csv="-1,0,99\n1,0,99\n",
            meta=valid_meta,
        )
        assert_fixture_rejected_without_touching_output(
            workload_tool,
            wrong_width,
            root / "wrong-width-output",
            "expected attribution width",
        )
        checks += 3

        # Metadata is part of the same preflight contract: invalid group counts,
        # intercept cardinality, and non-finite intercepts cannot consume the output.
        invalid_metadata = [
            ({**valid_meta, "num_groups": 0}, "positive integer"),
            ({**valid_meta, "num_groups": 4294967296}, "does not fit uint32"),
            ({**valid_meta, "intercepts": [0.0, 1.0]}, "intercept count"),
            ({**valid_meta, "intercepts": [float("nan")]}, "must be finite"),
        ]
        for index, (bad_meta, expected_error) in enumerate(invalid_metadata):
            bad_source = root / f"invalid-metadata-{index}"
            write_raw_fixture(
                bad_source,
                x_csv="1\n",
                expected_csv="0,0\n",
                meta=bad_meta,
            )
            assert_fixture_rejected_without_touching_output(
                workload_tool,
                bad_source,
                root / f"invalid-metadata-output-{index}",
                expected_error,
            )
        checks += len(invalid_metadata)

        too_many_rows = root / "too-many-output-rows"
        write_raw_fixture(
            too_many_rows,
            x_csv="1\n",
            expected_csv="0,0\n",
            meta=valid_meta,
        )
        assert_fixture_rejected_without_touching_output(
            workload_tool,
            too_many_rows,
            root / "too-many-output-rows-output",
            "row count does not fit uint32",
            "--rows",
            "4294967296",
        )
        checks += 1

        # Finite fp64 metadata can still overflow the fp32 bias column consumed by
        # both native engines. Validate the combined path bias + intercept up front.
        overflowing_bias = root / "overflowing-bias"
        write_raw_fixture(
            overflowing_bias,
            x_csv="1\n",
            expected_csv="0,0\n",
            meta={**valid_meta, "intercepts": [1e39]},
        )
        (overflowing_bias / "paths.csv").write_text(
            "path_idx,feature_idx,group,lower,upper,is_missing,zero_fraction,v\n"
        )
        assert_fixture_rejected_without_touching_output(
            workload_tool,
            overflowing_bias,
            root / "overflowing-bias-output",
            "representable as float32",
        )
        checks += 1

        # Raw paths receive the same structural validation as native preprocessing,
        # including invariants that deduplication could otherwise hide.
        path_violations = [
            (
                "missing-root",
                "0,0,0,-inf,inf,1,0.5,0.5\n",
                valid_meta,
                "missing its root",
            ),
            (
                "bad-root-fraction",
                "0,-1,0,-inf,inf,1,0.5,0.5\n",
                valid_meta,
                "root element must have zero_fraction == 1.0",
            ),
            (
                "unsatisfiable-merge",
                "0,0,0,-inf,0,0,0.5,0.5\n"
                "0,0,0,0,inf,0,0.5,0.5\n"
                "0,-1,0,-inf,inf,1,1,0.5\n",
                valid_meta,
                "merged split condition is unsatisfiable",
            ),
            (
                "uint32-path-id",
                "4294967296,-1,0,-inf,inf,1,1,0.5\n",
                valid_meta,
                "does not fit uint32",
            ),
        ]
        path_header = (
            "path_idx,feature_idx,group,lower,upper,is_missing,zero_fraction,v\n"
        )
        for name, body, metadata, expected_error in path_violations:
            bad_source = root / name
            write_raw_fixture(
                bad_source, x_csv="1\n", expected_csv="0,0\n", meta=metadata
            )
            (bad_source / "paths.csv").write_text(path_header + body)
            assert_fixture_rejected_without_touching_output(
                workload_tool,
                bad_source,
                root / f"{name}-output",
                expected_error,
            )

        too_deep = root / "too-deep"
        deep_body = "".join(
            f"0,{feature},0,-inf,inf,1,0.5,0.5\n" for feature in range(32)
        ) + "0,-1,0,-inf,inf,1,1,0.5\n"
        write_raw_fixture(
            too_deep,
            x_csv=",".join("0" for _ in range(32)) + "\n",
            expected_csv=",".join("0" for _ in range(33)) + "\n",
            meta={**valid_meta, "num_features": 32},
        )
        (too_deep / "paths.csv").write_text(path_header + deep_body)
        assert_fixture_rejected_without_touching_output(
            workload_tool, too_deep, root / "too-deep-output", "depth limit"
        )
        checks += len(path_violations) + 1

        # Python's int/float accept underscores and doubles above float32 range; the
        # native CSV readers reject both, so materialization must not bless them.
        numeric_violations = [
            ("underscored-matrix", "1_0\n", "0,0\n", None, "invalid matrix value"),
            ("overflowing-matrix", "1e39\n", "0,0\n", None, "overflowing float32"),
            (
                "overflowing-expected",
                "1\n",
                "1e39,0\n",
                None,
                "overflowing float32",
            ),
            (
                "underscored-path-id",
                "1\n",
                "0,0\n",
                "1_0,-1,0,-inf,inf,1,1,0.5\n",
                "invalid path_idx",
            ),
        ]
        for name, x_csv, expected_csv, path_body, expected_error in numeric_violations:
            bad_source = root / name
            write_raw_fixture(
                bad_source,
                x_csv=x_csv,
                expected_csv=expected_csv,
                meta=valid_meta,
            )
            if path_body is not None:
                (bad_source / "paths.csv").write_text(path_header + path_body)
            assert_fixture_rejected_without_touching_output(
                workload_tool,
                bad_source,
                root / f"{name}-output",
                expected_error,
            )
        checks += len(numeric_violations)

        # The native intercept parser broadcasts one explicit value across groups.
        singleton = root / "singleton-intercept"
        write_raw_fixture(
            singleton,
            x_csv="1\n",
            expected_csv="0,0,0,0\n",
            meta={**valid_meta, "num_groups": 2, "intercepts": [0.25]},
        )
        (singleton / "paths.csv").write_text(
            path_header
            + "0,-1,0,-inf,inf,1,1,0.5\n"
            + "1,-1,1,-inf,inf,1,1,-0.25\n"
        )
        singleton_output = root / "singleton-output"
        run(workload_tool, "fixture", str(singleton), str(singleton_output))
        assert json.loads((singleton_output / "workload.json").read_text())[
            "intercepts"
        ] == [0.25, 0.25]
        checks += 1

        # Every digest in a generated workload is authoritative and checked before an
        # existing --force target is touched.
        tampered = root / "tampered-workload"
        run(workload_tool, "hot", str(tampered), "--trees", "1", "--rows", "1")
        with (tampered / "X.csv").open("a") as target:
            target.write("123\n")
        assert_fixture_rejected_without_touching_output(
            workload_tool,
            tampered,
            root / "tampered-output",
            "SHA-256 mismatch",
        )
        checks += 1

        # If the final staging->destination rename fails after the old output has been
        # backed up, rollback restores the old directory byte-for-byte.
        import phase2_workloads  # noqa: E402

        rollback_output = root / "rollback-output"
        rollback_output.mkdir()
        rollback_sentinel = rollback_output / "keep-me.txt"
        rollback_sentinel.write_text("old output\n")
        real_replace = phase2_workloads.os.replace
        replace_calls = 0

        def fail_staging_swap(source, destination):
            nonlocal replace_calls
            replace_calls += 1
            if replace_calls == 2:
                raise OSError("injected staging swap failure")
            return real_replace(source, destination)

        phase2_workloads.os.replace = fail_staging_swap
        try:
            try:
                phase2_workloads.materialize_fixture(
                    argparse.Namespace(
                        source=str(singleton),
                        output=str(rollback_output),
                        rows=0,
                        name=None,
                        force=True,
                    )
                )
            except OSError as error:
                assert "injected staging swap failure" in str(error)
            else:
                raise AssertionError("injected staging swap unexpectedly succeeded")
        finally:
            phase2_workloads.os.replace = real_replace
        assert rollback_sentinel.read_text() == "old output\n"
        assert list(rollback_output.iterdir()) == [rollback_sentinel]
        assert not list(root.glob(".rollback-output.*"))
        checks += 1

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

        # A tiny fake SHAP frontend exercises the successful artifact path without
        # adding SHAP's compiled dependency stack to the portable tool test.
        fake_modules = root / "fake-modules"
        fake_modules.mkdir()
        (fake_modules / "shap.py").write_text(
            """import numpy as np
__version__ = 'test-double'
class _Model:
    num_outputs = 1
    model_type = 'xgboost-test-double'
class TreeExplainer:
    def __init__(self, booster, **kwargs):
        self.model = _Model()
        self.expected_value = 0.0
    def shap_values(self, X, check_additivity=False):
        return np.zeros_like(X, dtype=np.float32)
"""
        )
        cpu_shap = root / "cpu-shap.json"
        shap_env = dict(os.environ)
        shap_env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(fake_modules), shap_env.get("PYTHONPATH")))
        )
        run(
            str(BENCHMARKS / "phase2_cpu_shap.py"),
            str(wide / "model.json"),
            str(wide / "X.csv"),
            "--output",
            str(cpu_shap),
            "--row-limits",
            "8",
            "--warmup",
            "0",
            "--iterations",
            "2",
            "--nthread",
            "1",
            env=shap_env,
        )
        cpu_shap_payload = json.loads(cpu_shap.read_text())
        assert cpu_shap_payload["status"] == "ok"
        assert_power_windows(cpu_shap_payload["results"][0], 2)
        checks += 3

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
                "explained_rows": 100,
            },
            {
                "started_utc": "2026-07-12T12:00:03+00:00",
                "finished_utc": "2026-07-12T12:00:04+00:00",
            },
        ]
        summaries = summarize_jobs(load_samples(power), jobs)
        assert np.isclose(summaries[0]["gpu_mean_mw"], 3000.0)
        assert np.isclose(summaries[0]["gpu_estimated_energy_j"], 3.0)
        assert np.isclose(
            summaries[0]["gpu_estimated_energy_j_per_explained_row"], 0.03
        )
        assert summaries[0]["job"]["explained_rows"] == 100
        assert summaries[1]["status"] == "unavailable"
        checks += 5

        # Exact per-call windows exclude idle/hash gaps inside an aggregate envelope.
        gap_job = {
            "started_utc": "2026-07-12T12:00:00+00:00",
            "finished_utc": "2026-07-12T12:00:02+00:00",
            "sample_windows_utc": [
                {
                    "started_utc": "2026-07-12T12:00:00+00:00",
                    "finished_utc": "2026-07-12T12:00:00.250000+00:00",
                },
                {
                    "started_utc": "2026-07-12T12:00:01.750000+00:00",
                    "finished_utc": "2026-07-12T12:00:02+00:00",
                },
            ],
        }
        gap_summary = summarize_jobs(load_samples(power), [gap_job])[0]
        assert gap_summary["job_window_count"] == 2
        assert np.isclose(gap_summary["job_duration_s"], 0.5)
        assert np.isclose(gap_summary["gpu_estimated_energy_j"], 1.5)
        checks += 3

        # The real-data artifact schema and atomic-resume compatibility are permanent
        # contracts. In particular, a partial matrix must never span software versions.
        sample = {
            "iteration": 0,
            "order_index": 0,
            "seconds": 0.1,
            "started_utc": "2026-07-12T12:00:00+00:00",
            "finished_utc": "2026-07-12T12:00:00.100000+00:00",
        }
        implementation = {"fingerprint_sha256": "test-fingerprint"}
        device = {
            "dataset": "cal_housing",
            "size": "small",
            "model": "cal_housing-small",
            "device": "cpu",
            "rows": 8,
            "features": 8,
            "rounds": 10,
            "depth": 3,
            "warmup": 0,
            "iterations": 1,
            "nthread": 1,
            "seed": 432,
            "samples": [sample],
            "median_s": 0.1,
            "rows_per_s": 80.0,
            "max_local_accuracy_error": 0.0,
            "local_accuracy": True,
            "dataset_sha256": "d",
            "model_json_sha256": "m",
            "explain_matrix_sha256": "x",
            "output_sha256": "o",
            "implementation": implementation,
            "xgboost_version": benchmark_mac.xgb.__version__,
            "scikit_learn_version": benchmark_mac.sklearn.__version__,
            "python_version": benchmark_mac.platform.python_version(),
        }
        cell = {
            "schema": "metal_treeshap.realdata_cell.v1",
            "started_utc": sample["started_utc"],
            "finished_utc": sample["finished_utc"],
            "configuration_order": "seeded_random_within_iteration",
            "power_design": "timed_call_windows_only",
            "devices": [device],
            "comparison": None,
            "power_jobs": [{"dataset": "cal_housing", "size": "small",
                            "device": "cpu", **sample}],
        }
        realdata = benchmark_mac._suite_payload([cell])
        realdata_schema = json.loads((BENCHMARKS / "realdata_schema.json").read_text())
        validator = jsonschema.Draft202012Validator(
            realdata_schema, format_checker=jsonschema.FormatChecker()
        )
        validator.validate(realdata)
        invalid = copy.deepcopy(realdata)
        invalid["cells"][0]["unexpected"] = True
        try:
            validator.validate(invalid)
        except jsonschema.ValidationError:
            pass
        else:
            raise AssertionError("real-data schema accepted an unknown cell property")
        checks += 2

        resume_path = root / "resume.json"
        resume_path.write_text(json.dumps(realdata))
        resume_args = argparse.Namespace(
            nrows=8, warmup=0, niter=1, nthread=1, seed=432, device="cpu"
        )
        saved_provenance = benchmark_mac._PROVENANCE
        benchmark_mac._PROVENANCE = {
            "fingerprint_sha256": implementation["fingerprint_sha256"]
        }
        try:
            assert benchmark_mac._resume_cells(resume_path, resume_args) == [cell]
            incompatible = copy.deepcopy(realdata)
            incompatible["cells"][0]["devices"][0]["xgboost_version"] = "different"
            resume_path.write_text(json.dumps(incompatible))
            try:
                benchmark_mac._resume_cells(resume_path, resume_args)
            except SystemExit as error:
                assert "software versions differ" in str(error)
            else:
                raise AssertionError("resume accepted an incompatible XGBoost version")
            incompatible = copy.deepcopy(realdata)
            incompatible["power_trace"] = {"requested": True, "status": "skipped"}
            resume_path.write_text(json.dumps(incompatible))
            try:
                benchmark_mac._resume_cells(resume_path, resume_args)
            except SystemExit as error:
                assert "power evidence" in str(error)
            else:
                raise AssertionError("resume accepted a prior power-capture request")
        finally:
            benchmark_mac._PROVENANCE = saved_provenance
        checks += 3

        # Failed or unauthorized telemetry must not trigger expensive conditioning
        # blocks that cannot produce power evidence.
        assert not benchmark_mac._power_blocks_enabled(
            root / "power.plist", {"status": "skipped"}
        )
        assert benchmark_mac._power_blocks_enabled(
            root / "power.plist", {"status": "capturing"}
        )
        checks += 2

        # End-to-end runner job construction: six atomic tiles, one deterministic full.
        fake = root / "fake_benchmark.py"
        # The fake emits a fully schema-conforming native result: phase2_run now
        # validates against phase2_schema.json's native_result branch, so a bare
        # hand-rolled subset (the previous fake) would be rejected by the runner.
        fake.write_text("""#!/usr/bin/env python3
import csv,json,sys
a=sys.argv[1:]
def v(flag): return a[a.index(flag)+1]
rows=sum(1 for r in csv.reader(open(a[1])) if r)
cols=len(next(csv.reader(open(a[1]))))
dist={'median':0.001,'p10':0.001,'p90':0.001,'samples':[0.001]}
print(json.dumps({'schema':'metal_treeshap.phase2.benchmark.v1','status':'ok',
'workload':{'source_rows':rows,'rows':rows,'cols':cols,'groups':int(a[2]),
'raw_path_elements':4,'packed_bins':1},
'configuration':{'rows_per_simdgroup':int(v('--rows-per-simdgroup')),
'threads_per_threadgroup':int(v('--threads-per-threadgroup')),
'accumulation':v('--accumulation'),'model_storage':v('--model-storage'),
'deterministic_scratch_mib':int(v('--deterministic-scratch-mib')),
'atomic_tile_rows':int(v('--atomic-tile-rows')),'warmups':int(v('--warmup')),
'iterations':int(v('--iterations'))},
'setup_s':{},
'timing_s':{'wall':dist,'gpu':dist,'x_zero_copy_samples':[1],
'deterministic_runtime':{'active_scratch_bytes_samples':[0],
'tile_rows_samples':[0],'tile_count_samples':[0]}},
'throughput':{},
'repeatability':{'hashes':['0'],'unique_hashes':1,'max_pairwise_abs':0,
'max_pairwise_relative_symmetric':0,'relative_floor':1e-06},
'accuracy':{'first_run_max_abs':0,'first_run_max_relative':0,'first_run_mean_abs':0,
'first_run_max_row_group_sum_abs':0,'worst_run_max_abs':0,'worst_run_max_relative':0,
'worst_run_mean_abs':0,'worst_run_max_row_group_sum_abs':0}}))
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

        # phase2_schema.json is enforced, not decorative: every embedded native result
        # must validate against its native_result branch, and the runner's validator
        # must reject nonconforming payloads.
        import phase2_run  # noqa: E402  (benchmarks/ already on sys.path)

        schema_doc = json.loads((BENCHMARKS / "phase2_schema.json").read_text())
        native_validator = jsonschema.Draft202012Validator(
            {"$defs": schema_doc["$defs"], "$ref": "#/$defs/native_result"}
        )
        for item in result["results"]:
            native_validator.validate(item["native"])
        runner_validator = phase2_run._native_schema_validator()
        assert runner_validator is not None, "jsonschema present but validator disabled"
        broken = copy.deepcopy(result["results"][0]["native"])
        del broken["timing_s"]
        assert next(runner_validator.iter_errors(broken), None) is not None
        checks += 3

    print(f"ALL {checks} PHASE-2.1 TOOL TESTS PASSED")


if __name__ == "__main__":
    main()
