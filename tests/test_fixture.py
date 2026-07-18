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
the same frozen expectations — the repository-reproducible form of the on-device validation
runs (observed Metal errors: <= 6.501e-6 across all eight current fixtures).

Usage: python tests/test_fixture.py [path/to/reference_cli] [--metal-cli path/to/metal_cli]
             [--metal-rows-per-simdgroup N]
             [--metal-atomic-tile-rows N]
             [--metal-accumulation atomic|simdgroup|deterministic]
             [--metal-deterministic-scratch-mib N]
             [--metal-model-storage shared|private]
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
METAL_ROWS_PER_SIMDGROUP = os.environ.get("METAL_ROWS_PER_SIMDGROUP", "1024")
METAL_ATOMIC_TILE_ROWS = os.environ.get("METAL_ATOMIC_TILE_ROWS", "0")
METAL_ACCUMULATION = os.environ.get("METAL_ACCUMULATION", "atomic")
METAL_DETERMINISTIC_SCRATCH_MIB = os.environ.get(
    "METAL_DETERMINISTIC_SCRATCH_MIB", "256")
METAL_MODEL_STORAGE = os.environ.get("METAL_MODEL_STORAGE", "shared")
if "--metal-cli" in ARGS:
    idx = ARGS.index("--metal-cli")
    if idx + 1 >= len(ARGS):
        raise SystemExit("--metal-cli requires a path")
    METAL_CLI = ARGS[idx + 1]
    del ARGS[idx:idx + 2]
if "--metal-rows-per-simdgroup" in ARGS:
    idx = ARGS.index("--metal-rows-per-simdgroup")
    if idx + 1 >= len(ARGS):
        raise SystemExit("--metal-rows-per-simdgroup requires a value")
    METAL_ROWS_PER_SIMDGROUP = ARGS[idx + 1]
    del ARGS[idx:idx + 2]
if "--metal-atomic-tile-rows" in ARGS:
    idx = ARGS.index("--metal-atomic-tile-rows")
    if idx + 1 >= len(ARGS):
        raise SystemExit("--metal-atomic-tile-rows requires a value")
    METAL_ATOMIC_TILE_ROWS = ARGS[idx + 1]
    del ARGS[idx:idx + 2]
if "--metal-accumulation" in ARGS:
    idx = ARGS.index("--metal-accumulation")
    if idx + 1 >= len(ARGS):
        raise SystemExit("--metal-accumulation requires a value")
    METAL_ACCUMULATION = ARGS[idx + 1]
    del ARGS[idx:idx + 2]
if "--metal-deterministic-scratch-mib" in ARGS:
    idx = ARGS.index("--metal-deterministic-scratch-mib")
    if idx + 1 >= len(ARGS):
        raise SystemExit("--metal-deterministic-scratch-mib requires a value")
    METAL_DETERMINISTIC_SCRATCH_MIB = ARGS[idx + 1]
    del ARGS[idx:idx + 2]
if "--metal-model-storage" in ARGS:
    idx = ARGS.index("--metal-model-storage")
    if idx + 1 >= len(ARGS):
        raise SystemExit("--metal-model-storage requires a value")
    METAL_MODEL_STORAGE = ARGS[idx + 1]
    del ARGS[idx:idx + 2]
try:
    METAL_ROWS_PER_SIMDGROUP_INT = int(METAL_ROWS_PER_SIMDGROUP)
except ValueError as exc:
    raise SystemExit("Metal rows_per_simdgroup must be an integer") from exc
if METAL_ROWS_PER_SIMDGROUP_INT <= 0 or METAL_ROWS_PER_SIMDGROUP_INT > 2**32 - 1:
    raise SystemExit("Metal rows_per_simdgroup must be in [1, 2^32-1]")
try:
    METAL_ATOMIC_TILE_ROWS_INT = int(METAL_ATOMIC_TILE_ROWS)
except ValueError as exc:
    raise SystemExit("Metal atomic tile rows must be an integer") from exc
if METAL_ATOMIC_TILE_ROWS_INT < 0 or METAL_ATOMIC_TILE_ROWS_INT > 2**32 - 1:
    raise SystemExit("Metal atomic tile rows must be in [0, 2^32-1]")
if METAL_ACCUMULATION not in {"atomic", "simdgroup", "deterministic"}:
    raise SystemExit("Metal accumulation must be atomic, simdgroup, or deterministic")
if METAL_MODEL_STORAGE not in {"shared", "private"}:
    raise SystemExit("Metal model storage must be shared or private")
try:
    METAL_DETERMINISTIC_SCRATCH_MIB_INT = int(METAL_DETERMINISTIC_SCRATCH_MIB)
except ValueError as exc:
    raise SystemExit("Metal deterministic scratch MiB must be an integer") from exc
if METAL_DETERMINISTIC_SCRATCH_MIB_INT <= 0:
    raise SystemExit("Metal deterministic scratch MiB must be > 0")
if len(ARGS) > 1:
    raise SystemExit(f"unexpected arguments: {ARGS[1:]}")
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
            subprocess.run([METAL_CLI, paths_csv, x_csv, str(num_groups), outm, icept,
                            "--rows-per-simdgroup", str(METAL_ROWS_PER_SIMDGROUP_INT),
                            "--atomic-tile-rows", str(METAL_ATOMIC_TILE_ROWS_INT),
                            "--accumulation", METAL_ACCUMULATION,
                            "--deterministic-scratch-mib",
                            str(METAL_DETERMINISTIC_SCRATCH_MIB_INT),
                            "--model-storage", METAL_MODEL_STORAGE],
                           check=True, capture_output=True)
            phism = np.loadtxt(outm, delimiter=",").reshape(n_test, per_row)
            metal_err = float(np.max(np.abs(phism - expected.reshape(n_test, per_row))))

    err = float(np.max(np.abs(phis - expected.reshape(n_test, per_row))))
    # The frozen product gate (meta.json, 1e-3) leaves 100-1000x headroom over the
    # observed errors (<= ~6.5e-6 across all current fixtures); the tripwire catches
    # moderate numeric regressions the product gate would wave through. Raise it only
    # for an intended numerical-contract change.
    REGRESSION_TRIPWIRE = 1e-4
    tol = min(meta["tolerance"], REGRESSION_TRIPWIRE)
    ok = err < tol and (metal_err is None or metal_err < tol)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] fixture '{meta['case']}' ({kind}"
          + (f", frozen with xgboost {meta['xgboost_version']}" if "xgboost_version" in meta
             else "") + f"): reference max|phi - expected| = {err:.3e}"
          + (f", METAL[{METAL_ACCUMULATION},{METAL_MODEL_STORAGE},"
             f"rps={METAL_ROWS_PER_SIMDGROUP_INT},tile={METAL_ATOMIC_TILE_ROWS_INT}] "
             f"max|phi - expected| = {metal_err:.3e}" if metal_err is not None
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
    engines = "reference" + (f" + Metal[{METAL_ACCUMULATION},{METAL_MODEL_STORAGE},"
                             f"rps={METAL_ROWS_PER_SIMDGROUP_INT},"
                             f"tile={METAL_ATOMIC_TILE_ROWS_INT}]"
                             if METAL_CLI else "")
    print(f"ALL {len(cases)} FIXTURE TESTS PASSED ({engines})")
