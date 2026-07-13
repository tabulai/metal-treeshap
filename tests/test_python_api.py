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

from metal_treeshap import MetalTreeExplainer


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


if __name__ == "__main__":
    unittest.main()
