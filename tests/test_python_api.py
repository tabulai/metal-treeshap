#!/usr/bin/env python3
"""Installed-package tests for the public MetalTreeExplainer API."""

from __future__ import annotations

import csv
import importlib.resources
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from metal_treeshap import MetalTreeExplainer
    from metal_treeshap.explainer import _require_native

    # A source checkout imports successfully with the native extension deliberately
    # absent (_native is None); detect that too, not just an ImportError.
    _require_native()
except (ImportError, RuntimeError):
    if __name__ != "__main__":  # pytest without a usable wheel: skip, don't error
        import pytest

        pytest.skip(
            "metal-treeshap with its native extension is not installed; build and "
            "install the wheel first (see README)",
            allow_module_level=True,
        )
    raise


ROOT = Path(os.environ.get("METAL_TREESHAP_SOURCE_DIR", Path(__file__).resolve().parents[1]))


def load_path_records(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as source:
        records = []
        for row in csv.DictReader(source):
            records.append({
                "path_idx": int(row["path_idx"]),
                "feature_idx": int(row["feature_idx"]),
                "group": int(row["group"]),
                "lower": float(row["lower"]),
                "upper": float(row["upper"]),
                "is_missing_branch": bool(int(row["is_missing"])),
                "zero_fraction": float(row["zero_fraction"]),
                "v": float(row["v"]),
            })
    return records


class MetalTreeExplainerTests(unittest.TestCase):
    def test_packaged_assets_exist(self) -> None:
        package = importlib.resources.files("metal_treeshap")
        self.assertTrue(package.joinpath("treeshap.metal").is_file())
        from metal_treeshap._extract_paths import extract_model
        self.assertTrue(callable(extract_model))

    def test_deep_paths_and_deterministic_repeatability(self) -> None:
        fixture = ROOT / "tests" / "fixtures" / "deep31"
        paths = load_path_records(fixture / "paths.csv")
        X = np.loadtxt(fixture / "X.csv", delimiter=",", dtype=np.float32, ndmin=2)[:4]
        expected = np.loadtxt(
            fixture / "expected_contribs.csv", delimiter=",", dtype=np.float32, ndmin=2
        )[:4]
        explainer = MetalTreeExplainer.from_paths(
            paths,
            num_groups=1,
            num_features=31,
            intercepts=[0.5],
            rows_per_simdgroup=7,
            threads_per_threadgroup=64,
            accumulation="deterministic",
            model_storage="private",
            deterministic_scratch_mib=1,
        )
        first = explainer.explain(X)
        second = explainer(X)
        compatible = explainer.shap_values(X)
        self.assertEqual(first.shape, (4, 32))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, compatible)
        self.assertLess(float(np.max(np.abs(first - expected))), 1e-3)
        self.assertGreater(explainer.num_bins, 0)
        self.assertEqual(explainer.storage_mode, "private")
        with self.assertRaisesRegex(ValueError, "features"):
            explainer.explain(X[:, :-1])

    def test_xgboost_json_without_runtime_xgboost_dependency(self) -> None:
        fixture = ROOT / "tests" / "fixtures" / "regression-missing"
        X = np.loadtxt(fixture / "X.csv", delimiter=",", dtype=np.float32, ndmin=2)[:4]
        expected = np.loadtxt(
            fixture / "expected_contribs.csv", delimiter=",", dtype=np.float32, ndmin=2
        )[:4]
        explainer = MetalTreeExplainer.from_xgboost(
            fixture / "model.json",
            rows_per_simdgroup=16,
            threads_per_threadgroup=64,
            accumulation="atomic",
            atomic_tile_rows=2,
        )
        actual = explainer.explain(X)
        self.assertEqual(actual.shape, expected.shape)
        self.assertLess(float(np.max(np.abs(actual - expected))), 1e-3)

    def test_model_dictionary_and_xgboost_style_wrapper(self) -> None:
        fixture = ROOT / "tests" / "fixtures" / "regression-missing"
        model = json.loads((fixture / "model.json").read_text(encoding="utf-8"))
        X = np.loadtxt(fixture / "X.csv", delimiter=",", dtype=np.float32, ndmin=2)[:4]
        expected = np.loadtxt(
            fixture / "expected_contribs.csv", delimiter=",", dtype=np.float32, ndmin=2
        )[:4]

        class XGBoostStyleWrapper:
            def get_booster(self):
                return model

        for source in (model, XGBoostStyleWrapper()):
            with self.subTest(source=type(source).__name__):
                explainer = MetalTreeExplainer.from_xgboost(source)
                actual = explainer.shap_values(X)
                self.assertEqual(actual.shape, expected.shape)
                self.assertLess(float(np.max(np.abs(actual - expected))), 1e-3)

    def test_bias_only_models_without_xgboost_runtime(self) -> None:
        fixtures = ROOT / "tests" / "fixtures"

        regression = json.loads(
            (fixtures / "regression-missing" / "model.json").read_text(
                encoding="utf-8"
            )
        )
        regression_learner = regression["learner"]
        regression_learner["gradient_booster"]["model"]["trees"] = []
        regression_learner["gradient_booster"]["model"]["tree_info"] = []
        regression_learner["learner_model_param"]["base_score"] = "[2.5E-1]"

        single = MetalTreeExplainer.from_xgboost(regression)
        single_actual = single.explain(np.zeros((3, 8), dtype=np.float32))
        single_expected = np.zeros((3, 9), dtype=np.float32)
        single_expected[:, -1] = 0.25
        self.assertEqual(single.num_bins, 0)
        np.testing.assert_array_equal(single_actual, single_expected)
        self.assertEqual(
            single.explain(np.empty((0, 8), dtype=np.float32)).shape,
            (0, 9),
        )

        multiclass = json.loads(
            (fixtures / "multiclass-3" / "model.json").read_text(encoding="utf-8")
        )
        multiclass_learner = multiclass["learner"]
        multiclass_learner["gradient_booster"]["model"]["trees"] = []
        multiclass_learner["gradient_booster"]["model"]["tree_info"] = []
        multiclass_learner["learner_model_param"]["base_score"] = (
            "[2E-1,3E-1,5E-1]"
        )

        # A saved JSON model uses only the extractor packaged in the wheel; XGBoost is
        # deliberately not imported by this test or by the loading path.
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "bias-only-multiclass.json"
            model_path.write_text(json.dumps(multiclass), encoding="utf-8")
            grouped = MetalTreeExplainer.from_xgboost(model_path)

        grouped_actual = grouped.explain(np.zeros((2, 10), dtype=np.float32))
        grouped_expected = np.zeros((2, 3, 11), dtype=np.float32)
        grouped_expected[:, :, -1] = np.array([0.2, 0.3, 0.5], dtype=np.float32)
        self.assertEqual(grouped.num_bins, 0)
        np.testing.assert_array_equal(grouped_actual, grouped_expected)
        self.assertEqual(
            grouped.explain(np.empty((0, 10), dtype=np.float32)).shape,
            (0, 3, 11),
        )

    def test_multiclass_group_axis_and_empty_batch(self) -> None:
        fixture = ROOT / "tests" / "fixtures" / "multiclass-3"
        X = np.loadtxt(fixture / "X.csv", delimiter=",", dtype=np.float32, ndmin=2)[:4]
        flat_expected = np.loadtxt(
            fixture / "expected_contribs.csv", delimiter=",", dtype=np.float32, ndmin=2
        )[:4]
        explainer = MetalTreeExplainer.from_xgboost(
            fixture / "model.json",
            accumulation="deterministic",
            deterministic_scratch_mib=1,
        )
        actual = explainer.explain(X)
        expected = flat_expected.reshape(4, 3, X.shape[1] + 1)
        self.assertEqual(actual.shape, expected.shape)
        self.assertLess(float(np.max(np.abs(actual - expected))), 1e-3)
        empty = explainer.explain(np.empty((0, X.shape[1]), dtype=np.float32))
        self.assertEqual(empty.shape, (0, 3, X.shape[1] + 1))

    def test_nullable_pandas_dataframe(self) -> None:
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas is an optional dependency")

        fixture = ROOT / "tests" / "fixtures" / "regression-missing"
        X = np.loadtxt(fixture / "X.csv", delimiter=",", dtype=np.float32, ndmin=2)[:4]
        expected = np.loadtxt(
            fixture / "expected_contribs.csv", delimiter=",", dtype=np.float32, ndmin=2
        )[:4]
        frame = pd.DataFrame(X).astype("Float32")
        frame.iloc[0, 0] = pd.NA
        expected_matrix = X.copy()
        expected_matrix[0, 0] = np.nan

        explainer = MetalTreeExplainer.from_xgboost(
            fixture / "model.json",
            accumulation="deterministic",
            deterministic_scratch_mib=1,
        )
        actual = explainer.explain(frame)
        direct = explainer.explain(expected_matrix)
        np.testing.assert_array_equal(actual, direct)
        self.assertEqual(actual.shape, expected.shape)

    def test_intercepts_are_required_for_raw_paths(self) -> None:
        fixture = ROOT / "tests" / "fixtures" / "deep31"
        with self.assertRaises(TypeError):
            MetalTreeExplainer.from_paths(
                load_path_records(fixture / "paths.csv"),
                num_groups=1,
                num_features=31,
            )

    def test_raw_json_model_sources(self) -> None:
        # save_raw("json") output (bytes/bytearray) and its decoded text are model
        # documents, not filenames; both must load like the parsed dict does.
        fixture = ROOT / "tests" / "fixtures" / "regression-missing"
        X = np.loadtxt(fixture / "X.csv", delimiter=",", dtype=np.float32, ndmin=2)[:4]
        text = (fixture / "model.json").read_text(encoding="utf-8")
        # Deterministic mode: bitwise-equal outputs across explainer instances, so the
        # equality below proves the sources parse to the identical model.
        baseline = MetalTreeExplainer.from_xgboost(
            json.loads(text), accumulation="deterministic"
        ).explain(X)
        for source in (text, text.encode("utf-8"), bytearray(text.encode("utf-8"))):
            with self.subTest(source=type(source).__name__):
                actual = MetalTreeExplainer.from_xgboost(
                    source, accumulation="deterministic"
                ).explain(X)
                np.testing.assert_array_equal(actual, baseline)

    def test_validation_errors_cross_the_binding(self) -> None:
        fixture = ROOT / "tests" / "fixtures" / "deep31"
        records = load_path_records(fixture / "paths.csv")
        kwargs = dict(num_groups=1, num_features=31, intercepts=[0.5])
        with self.assertRaisesRegex(ValueError, "one value per output group"):
            MetalTreeExplainer.from_paths(records, num_groups=1, num_features=31,
                                          intercepts=[0.5, 0.5])
        with self.assertRaisesRegex(ValueError, "accumulation"):
            MetalTreeExplainer.from_paths(records, accumulation="bogus", **kwargs)
        with self.assertRaisesRegex(ValueError, "threads_per_threadgroup"):
            MetalTreeExplainer.from_paths(records, threads_per_threadgroup=96, **kwargs)
        with self.assertRaisesRegex(ValueError, "deterministic_scratch_mib"):
            MetalTreeExplainer.from_paths(records, deterministic_scratch_mib=0, **kwargs)
        bad_bounds = [dict(record) for record in records]
        bad_bounds[1]["lower"] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-NaN"):
            MetalTreeExplainer.from_paths(bad_bounds, **kwargs)
        explainer = MetalTreeExplainer.from_paths(records, **kwargs)
        with self.assertRaisesRegex(TypeError, "complex"):
            # An unsafe cast would silently discard the imaginary parts.
            explainer.explain(np.zeros((2, 31), dtype=np.complex64))
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "fake.metallib"
            fake.write_bytes(b"not a metallib")
            with self.assertRaisesRegex(RuntimeError, "failed to load metallib"):
                MetalTreeExplainer.from_paths(records, kernel=fake, **kwargs)

    def test_masked_and_duck_typed_inputs(self) -> None:
        fixture = ROOT / "tests" / "fixtures" / "regression-missing"
        X = np.loadtxt(fixture / "X.csv", delimiter=",", dtype=np.float32, ndmin=2)[:4]
        # Deterministic mode makes the equalities below exact: inputs that coerce to
        # the same matrix must produce bitwise-identical attributions.
        explainer = MetalTreeExplainer.from_xgboost(
            fixture / "model.json", accumulation="deterministic"
        )
        nan_matrix = X.copy()
        nan_matrix[0, 0] = np.nan
        baseline = explainer.explain(nan_matrix)

        # A masked entry means "missing": it must route like NaN, never expose the
        # backing storage value.
        masked = np.ma.MaskedArray(X.copy(), mask=np.zeros_like(X, dtype=bool))
        masked[0, 0] = 999.0
        masked.mask[0, 0] = True
        np.testing.assert_array_equal(explainer.explain(masked), baseline)

        class PlainToNumpy:  # polars/xarray-style: to_numpy() takes no kwargs
            def to_numpy(self, **kwargs):
                if kwargs:
                    raise TypeError("to_numpy() got an unexpected keyword argument")
                return nan_matrix

        np.testing.assert_array_equal(explainer.explain(PlainToNumpy()), baseline)

    def test_last_timings_trim_and_zero_copy_conversion(self) -> None:
        fixture = ROOT / "tests" / "fixtures" / "regression-missing"
        X = np.loadtxt(fixture / "X.csv", delimiter=",", dtype=np.float32, ndmin=2)
        explainer = MetalTreeExplainer.from_xgboost(
            fixture / "model.json", accumulation="deterministic"
        )
        self.assertIsNone(explainer.last_timings)
        big = np.tile(X, (256, 1))  # large enough for a page-aligned numpy allocation
        base = explainer.explain(big)
        # float64 input takes the page-padded conversion, which the host wraps zero-copy.
        wide = explainer.explain(big.astype(np.float64))
        timings = explainer.last_timings
        self.assertIsInstance(timings, dict)
        self.assertTrue(timings["dispatched"])
        self.assertTrue(timings["x_zero_copy"])
        # The binding page-aligns and pads the output, so the GPU always writes the
        # caller-visible memory directly.
        self.assertTrue(timings["output_zero_copy"])
        self.assertGreater(timings["total_s"], 0.0)
        np.testing.assert_array_equal(wide, base)
        explainer.trim_buffers()
        np.testing.assert_array_equal(explainer.explain(big), base)  # regrows on demand

    def test_concurrent_explain_calls_are_safe(self) -> None:
        import concurrent.futures

        fixture = ROOT / "tests" / "fixtures" / "regression-missing"
        X = np.loadtxt(fixture / "X.csv", delimiter=",", dtype=np.float32, ndmin=2)
        explainer = MetalTreeExplainer.from_xgboost(
            fixture / "model.json", accumulation="deterministic"
        )
        baseline = explainer.explain(X)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: explainer.explain(X), range(16)))
        for result in results:  # deterministic mode: bitwise equality expected
            np.testing.assert_array_equal(result, baseline)

    def test_forked_child_is_rejected_cleanly(self) -> None:
        fixture = ROOT / "tests" / "fixtures" / "regression-missing"
        X = np.loadtxt(fixture / "X.csv", delimiter=",", dtype=np.float32, ndmin=2)[:2]
        explainer = MetalTreeExplainer.from_xgboost(fixture / "model.json")
        explainer.explain(X)  # fully initialized before forking
        pid = os.fork()
        if pid == 0:  # child: the guard must raise before any Metal call can crash
            code = 1
            try:
                explainer.explain(X)
            except RuntimeError as error:
                code = 42 if "forked child" in str(error) else 43
            except BaseException:
                code = 44
            finally:
                os._exit(code)
        _, status = os.waitpid(pid, 0)
        self.assertTrue(os.WIFEXITED(status))
        self.assertEqual(os.WEXITSTATUS(status), 42)


if __name__ == "__main__":
    unittest.main()
