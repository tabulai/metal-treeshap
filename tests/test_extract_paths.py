#!/usr/bin/env python3
"""Focused, xgboost-free tests for raw-model extraction compatibility."""

from __future__ import annotations

import copy
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from extract_paths import extract_model  # noqa: E402


class DartLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (ROOT / "tests" / "fixtures" / "dart" / "model.json").open() as f:
            cls.old_dart = json.load(f)

    def flattened(self) -> dict:
        model = copy.deepcopy(self.old_dart)
        gb = model["learner"]["gradient_booster"]
        model["learner"]["gradient_booster"] = {
            "name": "gbtree",
            "model": gb["gbtree"]["model"],
            "weight_drop": gb["weight_drop"],
        }
        model["version"] = [3, 3, 0]
        return model

    def test_old_and_xgboost_33_layouts_are_equivalent(self) -> None:
        old = extract_model(self.old_dart)
        new = extract_model(self.flattened())
        self.assertEqual(old.booster, "dart")
        self.assertEqual(new.booster, "dart")
        self.assertEqual(old.paths, new.paths)
        self.assertEqual(old.intercepts, new.intercepts)
        self.assertEqual(new.extras["serialized_booster"], "gbtree")
        self.assertTrue(new.extras["uses_dropout_weights"])

    def test_plain_gbtree_has_no_dropout_scaling(self) -> None:
        model = self.flattened()
        del model["learner"]["gradient_booster"]["weight_drop"]
        extracted = extract_model(model)
        self.assertEqual(extracted.booster, "gbtree")
        self.assertFalse(extracted.extras["uses_dropout_weights"])

    def test_frozen_xgboost_33_model_uses_flattened_layout(self) -> None:
        path = ROOT / "tests" / "fixtures" / "dart-xgb33" / "model.json"
        with path.open() as f:
            model = json.load(f)
        gb = model["learner"]["gradient_booster"]
        self.assertEqual(model["version"][:2], [3, 3])
        self.assertEqual(gb["name"], "gbtree")
        self.assertIn("weight_drop", gb)
        extracted = extract_model(model)
        self.assertEqual(extracted.booster, "dart")
        self.assertTrue(extracted.paths)

    def test_weight_count_must_match_tree_count(self) -> None:
        model = self.flattened()
        model["learner"]["gradient_booster"]["weight_drop"].pop()
        with self.assertRaisesRegex(ValueError, r"weight_drop has 29 entries for 30 trees"):
            extract_model(model)

    def test_weight_vector_must_be_an_array(self) -> None:
        for bad in ("1.0", None, {}):
            with self.subTest(bad=bad):
                model = self.flattened()
                model["learner"]["gradient_booster"]["weight_drop"] = bad
                with self.assertRaisesRegex(ValueError, "must be a JSON array"):
                    extract_model(model)

    def test_old_dart_layout_requires_weights(self) -> None:
        model = copy.deepcopy(self.old_dart)
        del model["learner"]["gradient_booster"]["weight_drop"]
        with self.assertRaisesRegex(ValueError, "DART model is missing weight_drop"):
            extract_model(model)

    def test_weights_must_be_numeric_and_finite(self) -> None:
        for bad, message in (("not-a-number", "non-numeric"),
                             (math.nan, "must all be finite"),
                             (math.inf, "must all be finite")):
            with self.subTest(bad=bad):
                model = self.flattened()
                model["learner"]["gradient_booster"]["weight_drop"][0] = bad
                with self.assertRaisesRegex(ValueError, message):
                    extract_model(model)


if __name__ == "__main__":
    unittest.main()
