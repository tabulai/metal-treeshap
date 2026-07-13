#!/usr/bin/env python3
"""Validate release archives without installing or importing their native extension."""

from __future__ import annotations

import argparse
import re
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


REQUIRED_SDIST_SUFFIXES = {
    "CMakeLists.txt",
    "LICENSE",
    "NOTICE",
    "README.md",
    "pyproject.toml",
    "python/metal_treeshap/__init__.py",
    "python/metal_treeshap/explainer.py",
    "python/metal_treeshap/py.typed",
    "shaders/treeshap.metal",
    "tools/extract_paths.py",
}

REQUIRED_WHEEL_SUFFIXES = {
    "metal_treeshap/__init__.py",
    "metal_treeshap/_extract_paths.py",
    "metal_treeshap/explainer.py",
    "metal_treeshap/py.typed",
    "metal_treeshap/treeshap.metal",
}


def _assert_clean(names: set[str], archive: Path) -> None:
    bad = sorted(
        name for name in names
        if "__pycache__" in PurePosixPath(name).parts
        or PurePosixPath(name).suffix in {".pyc", ".pyo"}
    )
    assert not bad, f"{archive.name} contains Python cache artifacts: {bad}"


def _assert_suffixes(names: set[str], required: set[str], archive: Path) -> None:
    missing = sorted(
        suffix for suffix in required
        if not any(name == suffix or name.endswith(f"/{suffix}") for name in names)
    )
    assert not missing, f"{archive.name} is missing required files: {missing}"


def check_sdist(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        names = {member.name for member in archive.getmembers() if member.isfile()}
        _assert_clean(names, path)
        _assert_suffixes(names, REQUIRED_SDIST_SUFFIXES, path)


def check_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = {name for name in archive.namelist() if not name.endswith("/")}
        _assert_clean(names, path)
        _assert_suffixes(names, REQUIRED_WHEEL_SUFFIXES, path)
        native = [
            name for name in names
            if re.search(r"metal_treeshap/_native[^/]*\.(?:so|dylib)$", name)
        ]
        assert native, f"{path.name} does not contain the native extension"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", nargs="?", type=Path, default=Path("dist"))
    args = parser.parse_args()

    sdists = sorted(args.dist.glob("metal_treeshap-*.tar.gz"))
    wheels = sorted(args.dist.glob("metal_treeshap-*.whl"))
    assert sdists, f"no metal-treeshap sdist found under {args.dist}"
    assert wheels, f"no metal-treeshap wheel found under {args.dist}"
    for path in sdists:
        check_sdist(path)
    for path in wheels:
        check_wheel(path)
    print(f"PACKAGE CONTENT TEST PASSED ({len(sdists)} sdist, {len(wheels)} wheel)")


if __name__ == "__main__":
    main()
