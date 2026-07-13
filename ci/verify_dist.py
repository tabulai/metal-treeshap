#!/usr/bin/env python3
"""Validate release archives without importing or executing their contents."""

from __future__ import annotations

import argparse
from email.parser import Parser
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


FORBIDDEN_PARTS = {"__pycache__", ".DS_Store"}
SDIST_REQUIRED = {
    "CMakeLists.txt",
    "LICENSE",
    "NOTICE",
    "pyproject.toml",
    "shaders/treeshap.metal",
    "python/metal_treeshap/explainer.py",
    "python/metal_treeshap/py.typed",
    "third_party/metal-cpp/Metal/Metal.hpp",
    "tools/extract_paths.py",
}
WHEEL_REQUIRED = {
    "metal_treeshap/_extract_paths.py",
    "metal_treeshap/explainer.py",
    "metal_treeshap/py.typed",
    "metal_treeshap/treeshap.metal",
}


def _clean_names(names: list[str], *, strip_sdist_root: bool) -> set[str]:
    paths = [PurePosixPath(name) for name in names if name and not name.endswith("/")]
    for path in paths:
        if FORBIDDEN_PARTS.intersection(path.parts) or path.suffix == ".pyc":
            raise AssertionError(f"forbidden generated file in distribution: {path}")
    if not strip_sdist_root:
        return {str(path) for path in paths}
    roots = {path.parts[0] for path in paths if path.parts}
    if len(roots) != 1:
        raise AssertionError(f"sdist must have one top-level directory, found {sorted(roots)}")
    return {str(PurePosixPath(*path.parts[1:])) for path in paths if len(path.parts) > 1}


def _metadata(text: str, archive: Path) -> tuple[str, str]:
    metadata = Parser().parsestr(text)
    name = metadata.get("Name", "").strip()
    version = metadata.get("Version", "").strip()
    requires_python = metadata.get("Requires-Python", "").strip()
    if name.lower().replace("_", "-") != "metal-treeshap":
        raise AssertionError(f"{archive}: unexpected project name {name!r}")
    if not version:
        raise AssertionError(f"{archive}: missing Version metadata")
    if not requires_python:
        raise AssertionError(f"{archive}: missing Requires-Python metadata")
    return name, version


def inspect_sdist(path: Path) -> tuple[str, str]:
    with tarfile.open(path, "r:*") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        normalized = _clean_names(names, strip_sdist_root=True)
        missing = SDIST_REQUIRED - normalized
        if missing:
            raise AssertionError(f"{path}: missing sdist files: {sorted(missing)}")
        metadata_members = [m for m in members if PurePosixPath(m.name).name == "PKG-INFO"]
        if len(metadata_members) != 1:
            raise AssertionError(f"{path}: expected exactly one PKG-INFO")
        stream = archive.extractfile(metadata_members[0])
        if stream is None:
            raise AssertionError(f"{path}: could not read PKG-INFO")
        return _metadata(stream.read().decode("utf-8"), path)


def inspect_wheel(path: Path) -> tuple[str, str]:
    if "macosx" not in path.name or "arm64" not in path.name:
        raise AssertionError(f"{path}: wheel must target macOS arm64")
    if "universal2" in path.name:
        raise AssertionError(f"{path}: universal2 wheel is outside the supported target")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        normalized = _clean_names(names, strip_sdist_root=False)
        missing = WHEEL_REQUIRED - normalized
        if missing:
            raise AssertionError(f"{path}: missing wheel files: {sorted(missing)}")
        native = [name for name in normalized
                  if name.startswith("metal_treeshap/_native") and name.endswith(".so")]
        if len(native) != 1:
            raise AssertionError(f"{path}: expected one native extension, found {native}")
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise AssertionError(f"{path}: expected exactly one METADATA file")
        return _metadata(archive.read(metadata_names[0]).decode("utf-8"), path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--require-sdist", action="store_true")
    parser.add_argument("--require-wheel", action="store_true")
    parser.add_argument("--expected-wheel-tag", action="append", default=[])
    parser.add_argument("--print-version", action="store_true")
    args = parser.parse_args()

    sdists = [path for path in args.archives if path.name.endswith(".tar.gz")]
    wheels = [path for path in args.archives if path.suffix == ".whl"]
    unknown = [path for path in args.archives if path not in sdists and path not in wheels]
    if unknown:
        raise AssertionError(f"unknown distribution archive(s): {unknown}")
    if args.require_sdist and len(sdists) != 1:
        raise AssertionError(f"expected one sdist, found {len(sdists)}")
    if args.require_wheel and not wheels:
        raise AssertionError("expected at least one wheel")
    if args.expected_wheel_tag and len(wheels) != len(args.expected_wheel_tag):
        raise AssertionError(
            f"expected {len(args.expected_wheel_tag)} wheels, found {len(wheels)}"
        )
    for tag in args.expected_wheel_tag:
        matches = [path for path in wheels if f"-{tag}-" in path.name]
        if len(matches) != 1:
            raise AssertionError(f"expected one {tag} wheel, found {matches}")

    records = [(path, inspect_sdist(path)) for path in sdists]
    records.extend((path, inspect_wheel(path)) for path in wheels)
    if not records:
        raise AssertionError("no distribution archives supplied")
    names = {record[1][0].lower().replace("_", "-") for record in records}
    versions = {record[1][1] for record in records}
    if names != {"metal-treeshap"} or len(versions) != 1:
        raise AssertionError(f"inconsistent distribution metadata: names={names}, versions={versions}")

    version = versions.pop()
    print(f"validated {len(sdists)} sdist(s) and {len(wheels)} wheel(s) for {version}")
    if args.print_version:
        print(version)


if __name__ == "__main__":
    main()
