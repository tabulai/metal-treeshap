"""Argument/input validation and focused reporting checks for the Metal CLI.

The validation cases fail before device or shader creation; the reporting regression
uses one frozen fixture to cover the successful on-device path.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def expect_failure(command: list[str], expected: str) -> None:
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode != 0, (command, result.stdout, result.stderr)
    assert expected in result.stderr, (expected, result.stderr)


def check_atomic_stats_do_not_force_deterministic_plan(
    cli: str, output: str
) -> None:
    """A non-root atomic run must report lazy deterministic stats as unavailable."""
    source_dir = Path(__file__).resolve().parent.parent
    fixture = source_dir / "tests" / "fixtures" / "deep31"
    result = subprocess.run(
        [
            cli,
            str(fixture / "paths.csv"),
            str(fixture / "X.csv"),
            "1",
            output,
            "0.5",
            "--kernel",
            str(source_dir / "shaders" / "treeshap.metal"),
            "--accumulation",
            "atomic",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    partials = re.search(r"deterministic_partials_per_row=(\d+)", result.stderr)
    assert partials and int(partials.group(1)) > 0, result.stderr
    assert "deterministic_active_cells=unavailable" in result.stderr, result.stderr


def check_root_only_stats_are_known_zero(cli: str, temp_dir: str) -> None:
    """A root-only deterministic run has a known zero-cell plan without building it."""
    source_dir = Path(__file__).resolve().parent.parent
    paths = Path(temp_dir) / "root-only-paths.csv"
    matrix = Path(temp_dir) / "root-only-X.csv"
    output = Path(temp_dir) / "root-only-out.csv"
    paths.write_text(
        "path_idx,feature_idx,group,lower,upper,is_missing,zero_fraction,v\n"
        "0,-1,0,-inf,inf,1,1.0,2.5\n",
        encoding="utf-8",
    )
    matrix.write_text("0\n", encoding="utf-8")
    result = subprocess.run(
        [
            cli,
            str(paths),
            str(matrix),
            "1",
            str(output),
            "0",
            "--kernel",
            str(source_dir / "shaders" / "treeshap.metal"),
            "--accumulation",
            "deterministic",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "deterministic_partials_per_row=0" in result.stderr, result.stderr
    assert "deterministic_active_cells=0" in result.stderr, result.stderr
    assert "deterministic_active_cells=unavailable" not in result.stderr, result.stderr
    assert [float(value) for value in output.read_text().strip().split(",")] == [
        0.0,
        2.5,
    ]


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

        base = [cli, paths, matrix, "1", output, "0"]
        # Intercepts are REQUIRED (the host API contract): omitting them used to
        # silently assume zeros and hide real bias errors.
        expect_failure([cli, paths, matrix, "1", output], "usage:")
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

        check_atomic_stats_do_not_force_deterministic_plan(cli, output)
        check_root_only_stats_are_known_zero(cli, td)

    print("ALL 16 METAL CLI VALIDATION TESTS PASSED")


if __name__ == "__main__":
    main()
