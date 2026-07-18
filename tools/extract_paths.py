# SPDX-License-Identifier: Apache-2.0
# Part of metal-treeshap (see LICENSE and NOTICE); shipped in wheels as _extract_paths.
"""Extract GPUTreeShap-style path elements from an XGBoost model.

Parses the *raw JSON model* (booster.save_raw("json"), booster.save_model(...json), or an
already-loaded dict) rather than get_dump() text, which makes it robust to feature names,
independent of an installed xgboost at extraction time, and gives access to the fields the
text dump lacks:

  * tree_info            — authoritative tree -> output-group mapping (correct for
                           num_parallel_tree > 1, where round-robin by tree index is wrong)
  * sum_hessian          — node cover; zero_fraction = cover(child)/cover(parent)
  * default_left         — missing-value branch per node
  * weight_drop          — DART tree weights (leaf values are scaled by these),
                           including XGBoost 3.3+'s flattened ``name=gbtree`` layout
  * categories_nodes     — categorical splits (explicitly rejected: not yet supported)
  * base_score           — model intercept; scalar in xgboost <= 3.0, may be vector-valued
                           in 3.1+ — both are handled, returned per output group in
                           MARGIN space (logit applied for logistic objectives)

Split semantics: left/"yes" child takes x < split_condition, encoded as the interval
[-inf, t); right child takes [t, +inf); NaN follows the default_left branch. The root
element of each path has feature_idx = -1 and zero_fraction = 1.0.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

# base_score (output space) -> margin-space intercept, per objective. This mirrors
# XGBoost's objective-specific ProbToMargin and was determined EMPIRICALLY against
# xgboost 3.1.2 (train -> pred_contribs bias column minus cover-derived path bias; see
# tests/RESULTS.md): identity for plain regression/hinge/logitraw/multiclass priors,
# logit for logistic, log for the Poisson-family objectives. Objectives not in this
# table are REJECTED rather than guessed — survival:cox/aft, rank:*, and multi-target
# models are untested and previously produced bias errors > 1.0 when mis-linked.
_MARGIN_LINK: dict[str, str] = {
    "reg:squarederror": "identity",
    "reg:squaredlogerror": "identity",
    "reg:absoluteerror": "identity",
    "reg:quantileerror": "identity",
    "reg:pseudohubererror": "identity",
    "binary:logitraw": "identity",
    "binary:hinge": "identity",
    "multi:softmax": "identity",
    "multi:softprob": "identity",
    "binary:logistic": "logit",
    "reg:logistic": "logit",
    "count:poisson": "log",
    "reg:gamma": "log",
    "reg:tweedie": "log",
}


@dataclass(slots=True)  # created once per (leaf, ancestor): slots keep that cheap
class PathElement:
    path_idx: int
    feature_idx: int  # -1 == root
    group: int
    lower: float
    upper: float
    is_missing_branch: bool
    zero_fraction: float
    v: float


@dataclass
class ExtractedModel:
    paths: list[PathElement]
    num_groups: int
    intercepts: list[float]  # margin-space, one per output group
    objective: str
    booster: str  # "gbtree" | "dart"
    num_features: int
    extras: dict = field(default_factory=dict)


def load_model_dict(source) -> dict:
    """Accept an xgboost Booster, a path to a JSON model file, raw JSON text/bytes
    (``booster.save_raw("json")`` output), or a parsed dict."""
    if isinstance(source, dict):
        return source
    if isinstance(source, (str, bytes, bytearray)):
        # save_raw("json") hands back the model itself (a bytearray), and users also
        # pass its decoded text; treating either as a filename fails with a confusing
        # "File name too long". A model document always starts with '{'.
        stripped = source.lstrip()
        if stripped.startswith("{" if isinstance(stripped, str) else b"{"):
            return json.loads(source)
        with open(source, "rb") as f:
            return json.load(f)
    if hasattr(source, "save_raw"):  # xgboost.Booster without importing xgboost
        return json.loads(bytes(source.save_raw(raw_format="json")))
    raise TypeError(f"unsupported model source: {type(source)!r}")


def _parse_base_score(raw: str | float | list) -> list[float]:
    """base_score is a scalar string ('5E-1') up to xgboost 3.0; 3.1+ may serialize a
    vector-valued intercept (e.g. '[4.9E-1]' or a JSON list). Return a list."""
    if isinstance(raw, list):
        return [float(x) for x in raw]
    if isinstance(raw, (int, float)):
        return [float(raw)]
    s = str(raw).strip()
    if s.startswith("["):
        inner = s.strip("[]").strip()
        return [float(x) for x in inner.split(",")] if inner else []
    return [float(s)]


def _to_margin(values: list[float], objective: str) -> list[float]:
    """Convert base_score (output space) to margin-space intercepts via the objective's
    inverse link. Unknown objectives are rejected: a wrong link silently corrupts the
    bias column (observed errors > 1.0 for log-link objectives under identity)."""
    link = _MARGIN_LINK.get(objective)
    if link is None:
        raise NotImplementedError(
            f"objective {objective!r} is not in the tested allowlist "
            f"({sorted(_MARGIN_LINK)}); its base_score->margin link is unverified")
    if link == "logit":
        return [math.log(v / (1.0 - v)) for v in values]
    if link == "log":
        return [math.log(v) for v in values]
    return list(values)


def _walk_tree(tree: dict, group: int, leaf_scale: float, first_path_idx: int,
               paths: list[PathElement]) -> int:
    """Decompose one tree (raw JSON arrays) into unique root-to-leaf paths.
    Returns the number of paths (leaves) emitted."""
    if tree.get("categories_nodes"):
        raise NotImplementedError(
            "categorical splits are not supported yet; encode features numerically "
            "(proposal §2 non-goals / Phase 4 stretch)")
    left = tree["left_children"]
    right = tree["right_children"]
    split_index = tree["split_indices"]
    split_cond = tree["split_conditions"]
    default_left = tree["default_left"]
    cover = tree["sum_hessian"]

    n_emitted = 0
    # Iterative DFS carrying the edge stack; recursion depth is bounded but why risk it.
    stack: list[tuple[int, list[PathElement]]] = [(0, [])]
    while stack:
        node, acc = stack.pop()
        if left[node] == -1:  # leaf (stumps: node 0 itself)
            pid = first_path_idx + n_emitted
            n_emitted += 1
            v = float(split_cond[node]) * leaf_scale  # leaf value lives in split_conditions
            for e in acc:
                paths.append(PathElement(pid, e.feature_idx, group, e.lower, e.upper,
                                         e.is_missing_branch, e.zero_fraction, v))
            paths.append(PathElement(pid, -1, group, -math.inf, math.inf, True, 1.0, v))
            continue
        feat = int(split_index[node])
        thresh = float(split_cond[node])
        parent_cover = float(cover[node])
        for child, lower, upper, is_left in (
            (int(left[node]), -math.inf, thresh, True),
            (int(right[node]), thresh, math.inf, False),
        ):
            zf = float(cover[child]) / parent_cover
            missing_here = bool(default_left[node]) == is_left
            stack.append((child, acc + [PathElement(0, feat, group, lower, upper,
                                                    missing_here, zf, 0.0)]))
    return n_emitted


def extract_model(source) -> ExtractedModel:
    """Full extraction: paths + per-group margin intercepts + metadata."""
    m = load_model_dict(source)
    learner = m["learner"]
    gb = learner["gradient_booster"]
    serialized_booster_name = gb["name"]

    if serialized_booster_name == "dart":
        # XGBoost <= 3.2 serializes the DART wrapper explicitly and nests the
        # underlying tree model one level deeper.
        inner = gb["gbtree"]["model"]
        if "weight_drop" not in gb:
            raise ValueError("DART model is missing weight_drop")
        raw_weight_drop = gb["weight_drop"]
        has_dropout_weights = True
        booster_name = "dart"
    elif serialized_booster_name == "gbtree":
        # XGBoost 3.3 removed the DART wrapper.  A tree booster configured with
        # dropout now serializes as ``name=gbtree`` with ``weight_drop`` beside
        # ``model``.  Presence of that vector, not the name alone, distinguishes
        # dropout from an ordinary gbtree model.
        inner = gb["model"]
        raw_weight_drop = gb.get("weight_drop")
        has_dropout_weights = "weight_drop" in gb
        booster_name = "dart" if has_dropout_weights else "gbtree"
    else:
        raise NotImplementedError(f"unsupported booster type: {serialized_booster_name!r} "
                                  "(gblinear has no trees to explain)")

    trees = inner["trees"]
    weight_drop: list[float] | None = None
    if has_dropout_weights:
        if not isinstance(raw_weight_drop, list):
            raise ValueError("weight_drop must be a JSON array")
        try:
            weight_drop = [float(w) for w in raw_weight_drop]
        except (TypeError, ValueError) as e:
            raise ValueError("weight_drop contains a non-numeric value") from e
        if len(weight_drop) != len(trees):
            raise ValueError(
                f"weight_drop has {len(weight_drop)} entries for {len(trees)} trees")
        if not all(math.isfinite(w) for w in weight_drop):
            raise ValueError("weight_drop entries must all be finite")

    tree_info = [int(g) for g in inner["tree_info"]]
    if len(tree_info) != len(trees):
        raise ValueError("tree_info length does not match number of trees")

    num_class = int(learner["learner_model_param"].get("num_class", "0") or 0)
    num_target = int(learner["learner_model_param"].get("num_target", "1") or 1)
    if num_target > 1:
        raise NotImplementedError("multi-target models (num_target > 1) are not supported")
    num_groups = max(1, num_class, (max(tree_info) + 1) if tree_info else 1)
    num_features = int(learner["learner_model_param"].get("num_feature", "0") or 0)
    objective = learner["objective"]["name"]

    paths: list[PathElement] = []
    next_pid = 0
    for tree_idx, tree in enumerate(trees):
        group = tree_info[tree_idx]
        if not (0 <= group < num_groups):
            raise ValueError(f"tree {tree_idx} has out-of-range group {group}")
        scale = weight_drop[tree_idx] if weight_drop is not None else 1.0
        next_pid += _walk_tree(tree, group, scale, next_pid, paths)

    base = _parse_base_score(learner["learner_model_param"]["base_score"])
    intercepts = _to_margin(base, objective)
    if len(intercepts) == 1:
        intercepts = intercepts * num_groups
    elif len(intercepts) != num_groups:
        raise ValueError(f"base_score has {len(intercepts)} entries for {num_groups} groups")

    return ExtractedModel(paths=paths, num_groups=num_groups, intercepts=intercepts,
                          objective=objective, booster=booster_name,
                          num_features=num_features,
                          extras={"serialized_booster": serialized_booster_name,
                                  "uses_dropout_weights": weight_drop is not None})


# --- Backwards-compatible helpers -------------------------------------------------------

def extract_paths(source, num_groups: int | None = None) -> list[PathElement]:
    em = extract_model(source)
    if num_groups is not None and num_groups != em.num_groups:
        raise ValueError(f"model has {em.num_groups} output groups, caller expected {num_groups}")
    return em.paths


def write_paths_csv(paths: list[PathElement], filename: str) -> None:
    with open(filename, "w") as f:
        f.write("path_idx,feature_idx,group,lower,upper,is_missing,zero_fraction,v\n")
        for e in paths:
            f.write(f"{e.path_idx},{e.feature_idx},{e.group},{e.lower},{e.upper},"
                    f"{int(e.is_missing_branch)},{e.zero_fraction!r},{e.v!r}\n")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_json", help="xgboost JSON model file (save_model output)")
    ap.add_argument("out_csv")
    args = ap.parse_args()

    em = extract_model(args.model_json)
    write_paths_csv(em.paths, args.out_csv)
    print(f"booster={em.booster} objective={em.objective} groups={em.num_groups} "
          f"features={em.num_features} paths_elements={len(em.paths)} "
          f"intercepts={em.intercepts}")
