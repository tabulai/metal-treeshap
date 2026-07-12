#!/usr/bin/env python3
"""Source-checkout fallbacks that do not require a built native extension."""

from metal_treeshap.explainer import _resolve_kernel


source, is_metallib = _resolve_kernel(None)
assert not is_metallib
assert "kernel void shap_first_order" in source

# The wheel installs this module under metal_treeshap; source checkouts intentionally
# reuse the repository implementation directly.
from tools.extract_paths import extract_model

assert callable(extract_model)
print("SOURCE FALLBACK TEST PASSED")
