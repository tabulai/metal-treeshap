#!/usr/bin/env python3
"""Installed-package tests for the public MetalTreeExplainer API."""

from __future__ import annotations

import csv
import importlib.resources
import os
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
        self.assertEqual(first.shape, (4, 32))
        np.testing.assert_array_equal(first, second)
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

    def test_multiclass_group_axis_and_empty_batch(self) -> None:
        fixture = ROOT / "tests" / "fixtures" / "multiclass-3"
        X = np.loadtxt(fixture / "X.csv", delimiter=",", dtype=np.float32, ndmin=2)[:4]
        flat_expected = np.loadtxt(
            fixture / "expected_contribs.csv", delimiter=",", dtype=np.float32, ndmin=2
        )[:4]
        explainer = MetalTreeExplainer.from_xgboost(fixture / "model.json")
        actual = explainer.explain(X)
        expected = flat_expected.reshape(4, 3, X.shape[1] + 1)
        self.assertEqual(actual.shape, expected.shape)
        self.assertLess(float(np.max(np.abs(actual - expected))), 1e-3)
        empty = explainer.explain(np.empty((0, X.shape[1]), dtype=np.float32))
        self.assertEqual(empty.shape, (0, 3, X.shape[1] + 1))

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
