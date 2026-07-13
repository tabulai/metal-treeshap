#!/usr/bin/env python3
"""Source-checkout fallbacks that do not require a built native extension."""

import builtins

from metal_treeshap import MetalTreeExplainer
from metal_treeshap.explainer import _require_native, _resolve_kernel


try:
    _require_native()
except RuntimeError as error:
    assert "native extension is unavailable" in str(error)
    assert isinstance(error.__cause__, ImportError)
else:
    raise AssertionError("source-only import unexpectedly found the native extension")


source, is_metallib = _resolve_kernel(None)
assert not is_metallib
assert "kernel void shap_first_order" in source

# The wheel installs this module under metal_treeshap; source checkouts intentionally
# reuse the repository implementation directly.
from tools.extract_paths import extract_model

assert callable(extract_model)

# An ImportError raised *inside* the packaged extractor is a broken installation, not a
# signal to silently switch to the source-tree fallback.
real_import = builtins.__import__


def import_with_broken_extractor(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "_extract_paths" and level == 1:
        raise ModuleNotFoundError(
            "No module named 'extractor_internal_dependency'",
            name="extractor_internal_dependency",
        )
    return real_import(name, globals, locals, fromlist, level)


builtins.__import__ = import_with_broken_extractor
try:
    MetalTreeExplainer.from_xgboost({})
except ModuleNotFoundError as error:
    assert error.name == "extractor_internal_dependency"
else:
    raise AssertionError("internal extractor import failure was silently masked")
finally:
    builtins.__import__ = real_import

print("SOURCE FALLBACK TEST PASSED")
