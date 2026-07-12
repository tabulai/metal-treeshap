"""Argument and output-contract checks for the portable reference CLI."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np


def expect_failure(command: list[str], expected: str) -> None:
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode != 0, (command, result.stdout, result.stderr)
    assert expected in result.stderr, (expected, result.stderr)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} path/to/reference_cli")
    cli = sys.argv[1]
    with tempfile.TemporaryDirectory() as td:
        paths = os.path.join(td, "paths.csv")
        matrix = os.path.join(td, "X.csv")
        empty_matrix = os.path.join(td, "empty.csv")
        out64 = os.path.join(td, "out64.csv")
        out32 = os.path.join(td, "out32.csv")
        with open(paths, "w", encoding="utf-8") as stream:
            stream.write(
                "path_idx,feature_idx,group,lower,upper,is_missing,zero_fraction,v\n"
            )
        with open(matrix, "w", encoding="utf-8") as stream:
            stream.write("0\n")
        open(empty_matrix, "w", encoding="utf-8").close()

        base = [cli, paths, matrix]
        expect_failure(base + ["junk", out64, out32], "invalid num_groups")
        expect_failure(base + ["0", out64, out32], "num_groups must be > 0")
        expect_failure(base + ["1", out64, out32, "nan"], "intercepts must be finite")
        expect_failure(base + ["1", out64, out32, "0", "7junk"],
                       "invalid shuffle_seed")
        expect_failure([cli, paths, empty_matrix, "1", out64, out32],
                       "X.csv must contain at least one row")
        expect_failure(base + ["1", os.path.join(td, "missing", "o64.csv"), out32],
                       "cannot open output")

        subprocess.run(base + ["1", out64, out32, "0.25"], check=True,
                       capture_output=True)
        expected = np.array([0.0, 0.25])
        np.testing.assert_allclose(np.loadtxt(out64, delimiter=","), expected,
                                   rtol=0.0, atol=0.0)
        np.testing.assert_allclose(np.loadtxt(out32, delimiter=","), expected,
                                   rtol=0.0, atol=0.0)

    print("ALL 7 REFERENCE CLI VALIDATION TESTS PASSED")


if __name__ == "__main__":
    main()
