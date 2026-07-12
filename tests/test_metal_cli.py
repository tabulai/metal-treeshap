"""Fast argument/input validation checks for the checked-in Metal CLI.

These cases fail before device or shader creation, so they isolate the CLI contract from
the on-device differential tests in test_fixture.py.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile


def expect_failure(command: list[str], expected: str) -> None:
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode != 0, (command, result.stdout, result.stderr)
    assert expected in result.stderr, (expected, result.stderr)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} path/to/metal_cli")
    cli = sys.argv[1]
    with tempfile.TemporaryDirectory() as td:
        paths = os.path.join(td, "paths.csv")
        matrix = os.path.join(td, "X.csv")
        output = os.path.join(td, "out.csv")
        with open(paths, "w", encoding="utf-8") as stream:
            stream.write(
                "path_idx,feature_idx,group,lower,upper,is_missing,zero_fraction,v\n"
            )
        open(matrix, "w", encoding="utf-8").close()

        base = [cli, paths, matrix, "1", output]
        expect_failure(base + ["--rows-per-simdgroup", "0"],
                       "rows_per_simdgroup must be > 0")
        expect_failure(base + ["--rows-per-simdgroup", "7junk"],
                       "invalid rows_per_simdgroup")
        expect_failure(base + ["--rows-per-simdgroup", "4294967296"],
                       "does not fit uint32")
        expect_failure(base + ["--atomic-tile-rows", "7junk"],
                       "invalid atomic_tile_rows")
        expect_failure(base + ["--atomic-tile-rows", "4294967296"],
                       "atomic_tile_rows does not fit uint32")
        expect_failure(base + ["--atomic-tile-rows", "0",
                               "--atomic-tile-rows", "256"],
                       "specified more than once")
        expect_failure(base + ["--accumulation", "other"],
                       "atomic, simdgroup, or deterministic")
        expect_failure(base + ["--deterministic-scratch-mib", "0"],
                       "deterministic_scratch_mib must be > 0")
        expect_failure(base + ["--deterministic-scratch-mib", "1junk"],
                       "invalid deterministic_scratch_mib")
        expect_failure(base + ["--deterministic-scratch-mib", "1",
                               "--deterministic-scratch-mib", "2"],
                       "specified more than once")
        expect_failure(base + ["--unknown"], "unknown option")
        expect_failure(base + ["--accumulation", "deterministic",
                               "--deterministic-scratch-mib", "1"],
                       "X.csv must contain at least one row")
        expect_failure(base, "X.csv must contain at least one row")

    print("ALL 13 METAL CLI VALIDATION TESTS PASSED")


if __name__ == "__main__":
    main()
