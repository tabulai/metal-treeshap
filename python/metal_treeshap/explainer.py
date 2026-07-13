"""Public compiled-model API for MetalTreeShap."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from importlib import resources
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

_native_import_error: ImportError | None = None
try:
    from . import _native
except ImportError as error:  # Source-tree docs/imports remain usable.
    _native = None
    # Exception targets are cleared when an ``except`` suite exits. Keep a separate
    # reference so the delayed, user-facing error can retain the original cause.
    _native_import_error = error


def _require_native():
    if _native is None:
        raise RuntimeError(
            "MetalTreeShap's native extension is unavailable. Install the package from "
            "a wheel or build it on Apple Silicon with nanobind and the pinned metal-cpp "
            "headers."
        ) from _native_import_error
    return _native


def _value(element: Any, name: str) -> Any:
    if isinstance(element, Mapping):
        return element[name]
    dtype = getattr(element, "dtype", None)
    if dtype is not None and getattr(dtype, "names", None) and name in dtype.names:
        return element[name]
    return getattr(element, name)


def _condition_value(element: Any, flat_name: str, nested_name: str) -> Any:
    try:
        return _value(element, flat_name)
    except (AttributeError, KeyError, ValueError):
        condition = _value(element, "split_condition")
        return _value(condition, nested_name)


def _pack_paths(paths: Iterable[Any]) -> tuple[np.ndarray, ...]:
    columns: tuple[list[Any], ...] = tuple([] for _ in range(8))
    for element in paths:
        columns[0].append(_value(element, "path_idx"))
        columns[1].append(_value(element, "feature_idx"))
        columns[2].append(_value(element, "group"))
        columns[3].append(
            _condition_value(element, "lower", "feature_lower_bound")
        )
        columns[4].append(
            _condition_value(element, "upper", "feature_upper_bound")
        )
        columns[5].append(
            _condition_value(element, "is_missing_branch", "is_missing_branch")
        )
        columns[6].append(_value(element, "zero_fraction"))
        columns[7].append(_value(element, "v"))
    dtypes = (np.uint64, np.int64, np.int32, np.float32, np.float32, np.uint8,
              np.float64, np.float32)
    return tuple(np.ascontiguousarray(values, dtype=dtype)
                 for values, dtype in zip(columns, dtypes, strict=True))


def _resolve_kernel(kernel: str | PathLike[str] | None) -> tuple[str, bool]:
    if kernel is None:
        resource = resources.files("metal_treeshap").joinpath("treeshap.metal")
        if resource.is_file():
            return resource.read_text(encoding="utf-8"), False
        source_checkout = Path(__file__).resolve().parents[2] / "shaders" / "treeshap.metal"
        if source_checkout.is_file():
            return source_checkout.read_text(encoding="utf-8"), False
        raise FileNotFoundError("packaged treeshap.metal shader is missing")
    path = Path(kernel).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".metallib":
        return str(path), True
    return path.read_text(encoding="utf-8"), False


class MetalTreeExplainer:
    """A compiled TreeSHAP model backed by a reusable Metal pipeline and buffers.

    Construct with :meth:`from_paths` or :meth:`from_xgboost`. Single-output calls return
    ``(rows, features + 1)``; multi-output calls return
    ``(rows, groups, features + 1)``. The final column is the bias/intercept term.
    """

    def __init__(self, native: Any, *, num_groups: int, num_features: int) -> None:
        self._native = native
        self.num_groups = int(num_groups)
        self.num_features = int(num_features)

    @classmethod
    def from_paths(
        cls,
        paths: Iterable[Any],
        *,
        num_groups: int,
        num_features: int,
        intercepts: Iterable[float],
        kernel: str | PathLike[str] | None = None,
        rows_per_simdgroup: int = 256,
        threads_per_threadgroup: int = 256,
        accumulation: str = "atomic",
        model_storage: str = "shared",
        deterministic_scratch_mib: int = 256,
        atomic_tile_rows: int = 0,
    ) -> "MetalTreeExplainer":
        """Compile pre-extracted root-to-leaf paths once for repeated explanations."""
        if int(num_groups) <= 0 or int(num_features) <= 0:
            raise ValueError("num_groups and num_features must be positive")
        packed = _pack_paths(paths)
        intercept_array = np.ascontiguousarray(
            list(intercepts),
            dtype=np.float64,
        )
        if intercept_array.shape != (int(num_groups),):
            raise ValueError("intercepts must contain one value per output group")
        kernel_spec, is_metallib = _resolve_kernel(kernel)
        native_module = _require_native()
        native = native_module.NativeExplainer(
            *packed,
            int(num_groups),
            int(num_features),
            intercept_array,
            kernel_spec,
            is_metallib,
            model_storage,
        )
        native.configure(
            rows_per_simdgroup=int(rows_per_simdgroup),
            threads_per_threadgroup=int(threads_per_threadgroup),
            accumulation=accumulation,
            deterministic_scratch_mib=int(deterministic_scratch_mib),
            atomic_tile_rows=int(atomic_tile_rows),
        )
        return cls(native, num_groups=num_groups, num_features=num_features)

    @classmethod
    def from_xgboost(
        cls,
        model: Any,
        **kwargs: Any,
    ) -> "MetalTreeExplainer":
        """Extract and compile an XGBoost Booster, JSON model path, or model dictionary.

        Loading a saved JSON model does not require XGBoost to be installed. Passing a
        live Booster requires XGBoost only because that object owns the model.
        """
        try:
            from ._extract_paths import extract_model
        except ModuleNotFoundError as error:
            if error.name != f"{__package__}._extract_paths":
                raise
            # Development-source fallback; wheels install the same checked-in extractor
            # as ``metal_treeshap._extract_paths`` via CMake.
            from tools.extract_paths import extract_model
        if hasattr(model, "get_booster"):
            model = model.get_booster()
        if isinstance(model, PathLike):
            model = str(model)
        extracted = extract_model(model)
        return cls.from_paths(
            extracted.paths,
            num_groups=extracted.num_groups,
            num_features=extracted.num_features,
            intercepts=extracted.intercepts,
            **kwargs,
        )

    def explain(self, X: Any, *, keep_group_axis: bool = False) -> np.ndarray:
        """Return additive feature contributions for a 2-D dense input matrix."""
        to_numpy = getattr(X, "to_numpy", None)
        if callable(to_numpy):
            # pandas nullable dtypes expose missing values as ``pd.NA`` when coerced via
            # plain np.asarray.  Request the dense float representation explicitly while
            # keeping pandas an optional dependency.
            matrix = to_numpy(dtype=np.float32, na_value=np.nan)
        else:
            matrix = X
        matrix = np.asarray(matrix, dtype=np.float32, order="C")
        if matrix.ndim != 2:
            raise ValueError("X must be a 2-D array")
        if matrix.shape[1] != self.num_features:
            raise ValueError(
                f"X has {matrix.shape[1]} features; model expects {self.num_features}"
            )
        output = self._native.explain(matrix)
        if self.num_groups == 1 and not keep_group_axis:
            return output[:, 0, :]
        return output

    def shap_values(self, X: Any, *, keep_group_axis: bool = False) -> np.ndarray:
        """Return SHAP values using the familiar ``TreeExplainer`` method name.

        This is the same operation and has the same shape contract as :meth:`explain`;
        it is provided as a compatibility convenience for code that already calls
        ``shap_values`` on tree explainers.
        """
        return self.explain(X, keep_group_axis=keep_group_axis)

    __call__ = explain

    @property
    def num_bins(self) -> int:
        return int(self._native.num_bins)

    @property
    def storage_mode(self) -> str:
        return str(self._native.storage_mode)
