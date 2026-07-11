"""Frozen-fixture regression tests — run WITHOUT xgboost.

Iterates every tests/fixtures/<case>/ directory. Two fixture kinds (meta.json "kind"):

  xgboost_model  model.json + X.csv + expected_contribs.csv: the extractor parses the raw
                 model file directly (no xgboost import), the reference CLI computes
                 contributions, and they must match the frozen pred_contribs.
  raw_paths      paths.csv + X.csv + expected_contribs.csv: CLI consumes the paths
                 directly (e.g. the synthetic deep31 32-lane comb fixture, which is also
                 the Phase-1 Metal differential target).

Fixtures are only ever (re)generated explicitly — test_vs_xgboost.py --update-fixtures or
tools/make_deep_fixture.py — never by an ordinary verification run.

DUAL-ENGINE MODE (macOS): when METAL_CLI is set (env var, or --metal-cli <path>), every
fixture ALSO runs through the Metal engine (src/main_metal.cpp) and is compared against
the same frozen expectations — the repository-reproducible form of the validation_v3
on-device runs (observed Metal errors there: <= 6.5e-6 across all six fixtures).

Usage: python tests/test_fixture.py [path/to/reference_cli] [--metal-cli path/to/metal_cli]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from extract_paths import extract_model, write_paths_csv  # noqa: E402

ARGS = sys.argv[1:]
METAL_CLI = os.environ.get("METAL_CLI")
if "--metal-cli" in ARGS:
    METAL_CLI = ARGS[ARGS.index("--metal-cli") + 1]
    del ARGS[ARGS.index("--metal-cli"):ARGS.index("--metal-cli") + 2]
CLI = ARGS[0] if ARGS else "./reference_cli"
FX_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def run_fixture(case_dir: str) -> None:
    with open(os.path.join(case_dir, "meta.json")) as f:
        meta = json.load(f)
    kind = meta.get("kind", "xgboost_model")
    num_groups = meta["num_groups"]
    per_row = num_groups * (meta["num_features"] + 1)
    x_csv = os.path.join(case_dir, "X.csv")
    expected = np.loadtxt(os.path.join(case_dir, "expected_contribs.csv"), delimiter=",")
    n_test = expected.shape[0]

    with tempfile.TemporaryDirectory() as td:
        out64, out32 = os.path.join(td, "o64.csv"), os.path.join(td, "o32.csv")
        if kind == "xgboost_model":
            em = extract_model(os.path.join(case_dir, "model.json"))  # no xgboost import
            assert em.num_groups == num_groups, (em.num_groups, num_groups)
            paths_csv = os.path.join(td, "p.csv")
            write_paths_csv(em.paths, paths_csv)
            intercepts = em.intercepts
        elif kind == "raw_paths":
            paths_csv = os.path.join(case_dir, "paths.csv")
            intercepts = meta["intercepts"]
        else:
            raise ValueError(f"unknown fixture kind {kind!r} in {case_dir}")
        icept = ",".join(repr(float(v)) for v in intercepts)
        subprocess.run([CLI, paths_csv, x_csv, str(num_groups), out64, out32, icept],
                       check=True, capture_output=True)
        phis = np.loadtxt(out64, delimiter=",").reshape(n_test, per_row)

        metal_err = None
        if METAL_CLI:
            outm = os.path.join(td, "om.csv")
            subprocess.run([METAL_CLI, paths_csv, x_csv, str(num_groups), outm, icept],
                           check=True, capture_output=True)
            phism = np.loadtxt(outm, delimiter=",").reshape(n_test, per_row)
            metal_err = float(np.max(np.abs(phism - expected.reshape(n_test, per_row))))

    err = float(np.max(np.abs(phis - expected.reshape(n_test, per_row))))
    tol = meta["tolerance"]
    ok = err < tol and (metal_err is None or metal_err < tol)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] fixture '{meta['case']}' ({kind}"
          + (f", frozen with xgboost {meta['xgboost_version']}" if "xgboost_version" in meta
             else "") + f"): reference max|phi - expected| = {err:.3e}"
          + (f", METAL max|phi - expected| = {metal_err:.3e}" if metal_err is not None
             else "") + f" (tol {tol})")
    assert ok, f"fixture regression in {case_dir}"


if __name__ == "__main__":
    cases = sorted(d for d in os.listdir(FX_ROOT)
                   if os.path.isfile(os.path.join(FX_ROOT, d, "meta.json")))
    if not cases:
        raise SystemExit("no fixture directories found — run test_vs_xgboost.py "
                         "--update-fixtures and tools/make_deep_fixture.py first")
    for case in cases:
        run_fixture(os.path.join(FX_ROOT, case))
    engines = "reference" + (" + Metal" if METAL_CLI else "")
    print(f"ALL {len(cases)} FIXTURE TESTS PASSED ({engines})")
