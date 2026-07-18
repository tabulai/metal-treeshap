"""Generate the deterministic 32-element cooperative-path fixture (tests/fixtures/deep31).

Builds a comb tree — node at depth d splits on feature d, left child is a leaf, right
child descends — whose deepest path is root + 31 DISTINCT features = 32 elements, the
exact SIMD-width boundary. Feature distinctness matters: deduplication cannot shorten the
path, so the full 32-lane cooperative recurrence executes in whatever engine consumes it.

This is the Phase-1 Metal differential target: the kernel must reproduce
expected_contribs.csv (produced by the fp64 CPU reference) on this fixture, covering the
lane-31 boundary, partial-SIMD bins, extreme covers (1e-4), and NaN default routing.

Usage: python tools/make_deep_fixture.py <reference_cli> [out_dir]
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

DEPTH = 31
INF = math.inf


def build_paths():
    """Mirror tests/test_property_additivity.cpp::TestComb31's tree shape."""
    # Spine covers: frac to the leaf child at depth d.
    leaf_vals = [((d * 37) % 19 - 9) / 10.0 for d in range(DEPTH + 1)]  # deterministic
    rows = []  # csv rows
    spine = []  # (feature, zero_fraction_right, default_left) along the right spine
    for d in range(DEPTH):
        frac = 1e-4 if d % 5 == 0 else 0.3
        spine.append((d, 1.0 - frac, frac, (d % 2) == 0))

    # Path p (p = 0..DEPTH-1): follows the right spine to depth p, then LEFT to a leaf.
    # Path DEPTH: the full right spine to the deepest leaf (32 elements with root).
    pid = 0
    for p in range(DEPTH + 1):
        v = leaf_vals[p]
        for d in range(min(p, DEPTH)):
            feat, zf_right, _, dleft = spine[d]
            # right branch on feature d: x >= 0 -> [0, inf); missing goes right iff not default_left
            rows.append((pid, feat, 0, 0.0, INF, int(not dleft), zf_right, v))
        if p < DEPTH:
            feat, _, zf_left, dleft = spine[p]
            # left branch on feature p: x < 0 -> (-inf, 0); missing goes left iff default_left
            rows.append((pid, feat, 0, -INF, 0.0, int(dleft), zf_left, v))
        rows.append((pid, -1, 0, -INF, INF, 1, 1.0, v))  # root
        pid += 1
    return rows


def build_X():
    n_rows, n_cols = 8, DEPTH
    X = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            if r == 0:
                row.append(1.0)  # all-right: reaches the deepest (32-element) leaf
            elif r == 1:
                row.append(-1.0)  # all-left: exits at depth 1
            elif r == 2:
                row.append(float("nan"))  # default routing everywhere
            elif r == 3:
                row.append(1.0 if c % 2 == 0 else -1.0)
            elif r == 4:
                row.append(float("nan") if c % 3 == 0 else 1.0)
            elif r == 5:
                row.append(1.0 if c < 30 else -1.0)  # exits at lane 30/31 boundary
            elif r == 6:
                row.append(1.0 if c < 16 else -1.0)  # mid-path exit
            else:
                row.append(-1.0 if c % 7 == 3 else 1.0)
        X.append(row)
    return X


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: make_deep_fixture.py <reference_cli> [out_dir]")
    cli = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "fixtures",
        "deep31")
    os.makedirs(out_dir, exist_ok=True)

    paths_csv = os.path.join(out_dir, "paths.csv")
    with open(paths_csv, "w") as f:
        f.write("path_idx,feature_idx,group,lower,upper,is_missing,zero_fraction,v\n")
        for row in build_paths():
            f.write(",".join(repr(x) if isinstance(x, float) else str(x) for x in row) + "\n")

    x_csv = os.path.join(out_dir, "X.csv")
    with open(x_csv, "w") as f:
        for row in build_X():
            f.write(",".join(repr(v) for v in row) + "\n")

    expected = os.path.join(out_dir, "expected_contribs.csv")
    fp32_out = os.path.join(out_dir, "_fp32_tmp.csv")
    subprocess.run([cli, paths_csv, x_csv, "1", expected, fp32_out, "0.5"], check=True)
    os.remove(fp32_out)

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"case": "deep31-comb", "kind": "raw_paths", "num_groups": 1,
                   "num_features": DEPTH, "intercepts": [0.5], "tolerance": 1e-3,
                   "note": "32-element cooperative path (root + 31 distinct features); "
                           "expected = fp64 CPU reference; Phase-1 Metal kernel must "
                           "match. Covers lane-31 boundary, partial exits, 1e-4 covers, "
                           "NaN default routing."}, f, indent=1)
    print(f"wrote deep31 fixture to {out_dir}")


if __name__ == "__main__":
    main()
