# metal-treeshap

**Exact SHAP values for XGBoost models, computed on Apple GPUs.**

`metal-treeshap` runs the TreeSHAP algorithm — the same exact attribution method behind
XGBoost's `pred_contribs` and the `shap` package's tree explainer — as Metal compute
kernels on Apple Silicon. Compile a model once, then explain batches of rows at GPU
speed: on an M4 Max, sustained paired benchmarks measure **14–19× over 16-thread
XGBoost CPU** on the same models and inputs, with elementwise agreement to ~1e-5.

```python
from metal_treeshap import MetalTreeExplainer

explainer = MetalTreeExplainer.from_xgboost("model.json")  # compile once
phis = explainer.shap_values(X)                            # explain many, fast
```

The output is a NumPy array with the standard contract: one column per feature plus a
final bias column, and each row sums to the model's margin prediction (*local
accuracy*). Single-output models return `(rows, features + 1)`; multiclass models
return `(rows, classes, features + 1)`.

## Why this exists

Exact TreeSHAP is expensive — its cost grows with rows × trees × depth², so explaining
a large ensemble over a real dataset can take minutes on a CPU. NVIDIA solved this for
CUDA with [GPUTreeShap](https://github.com/rapidsai/gputreeshap) (Mitchell, Frank &
Holmes, [arXiv:2010.13972](https://arxiv.org/abs/2010.13972)), which decomposes trees
into root-to-leaf paths and evaluates them cooperatively across GPU warps. That
acceleration never reached Macs, because CUDA doesn't run there.

`metal-treeshap` is a from-scratch Metal port of that algorithm for Apple GPUs. It
keeps GPUTreeShap's path decomposition and cooperative evaluation strategy, adapts them
to Metal's SIMD-groups and unified memory, and adds an accuracy-focused execution mode
that Apple's fp32-only GPUs make necessary. The portable core is continuously verified
against XGBoost's own attribution output. See [docs/how-it-works.md](docs/how-it-works.md)
for the design.

## Requirements

- Apple Silicon Mac (M1 or newer), macOS 13+
- Python 3.10+ and NumPy (pandas optional)
- Xcode Command Line Tools for building the wheel — the offline Metal shader
  toolchain is **not** required; kernels compile at runtime

## Installation

The package is not on PyPI yet. Build the wheel from a source checkout:

```bash
git clone https://github.com/tabulai/metal-treeshap.git
cd metal-treeshap
python3 -m pip install build
python3 -m build --wheel
python3 -m pip install dist/metal_treeshap-*.whl
```

## Usage

`from_xgboost` accepts a live `xgboost.Booster`, an sklearn-style wrapper, a saved
JSON model path, raw JSON text/bytes (`booster.save_raw("json")`), or a parsed dict.
Loading a saved model does **not** require xgboost to be installed.

```python
import xgboost as xgb
from metal_treeshap import MetalTreeExplainer

booster = xgb.train(params, dtrain, num_boost_round=500)
explainer = MetalTreeExplainer.from_xgboost(booster)

phis = explainer.shap_values(X_test)     # also: explainer.explain(X), explainer(X)
assert phis.shape == (len(X_test), X_test.shape[1] + 1)
```

**Inputs.** NumPy arrays and pandas DataFrames are accepted; any numeric dtype and
memory layout is converted as needed. DataFrame columns are consumed **by position** —
pass them in training order. `NaN` means missing and routes exactly as XGBoost routes
it; pandas nullable values (`pd.NA`) and masked arrays convert to `NaN`. Complex input
is rejected rather than silently truncated.

**Works with the `shap` package.** Wrap the output in a `shap.Explanation` and every
standard plot works, drawn from GPU-computed values:

```python
import shap

explanation = shap.Explanation(
    values=phis[:, :-1], base_values=phis[:, -1], data=X_test,
    feature_names=feature_names,
)
shap.plots.beeswarm(explanation)
```

**Supported models.** XGBoost `gbtree` and `dart` boosters (both weight-drop layouts),
`num_parallel_tree` ≥ 1, missing values, and exactly these objectives — each link
verified end-to-end against `pred_contribs`: `reg:squarederror`,
`reg:squaredlogerror`, `reg:absoluteerror`, `reg:quantileerror`,
`reg:pseudohubererror`, `binary:logitraw`, `binary:hinge`, `multi:softmax`,
`multi:softprob`, `binary:logistic`, `reg:logistic`, `count:poisson`, `reg:gamma`,
`reg:tweedie`. Anything else — survival, ranking, multi-target, categorical splits —
is **rejected with a clear error** rather than silently mis-attributed. Verified
against xgboost 2.0.3, 3.1.2, and 3.3.0.

**Introspection.** `explainer.last_timings` reports what the most recent call did
(GPU time, zero-copy state, dispatch shape); `explainer.trim_buffers()` releases the
persistent buffers a long-lived explainer retains after a peak batch.

## Execution modes

| mode | behavior | when to use |
|---|---|---|
| `atomic` *(default)* | fastest; float atomics make reruns differ at ~1e-6 | throughput |
| `deterministic` | fixed-order Kahan reduction; **bit-identical** reruns, tighter accuracy; ~1.8× the default mode's time | reproducible pipelines, caching, audits |
| `simdgroup` | SIMD pre-aggregation experiment | exploring other GPUs/models |

```python
explainer = MetalTreeExplainer.from_xgboost(model, accumulation="deterministic")
```

Advanced knobs (`rows_per_simdgroup`, `threads_per_threadgroup`,
`deterministic_scratch_mib`, `atomic_tile_rows`, `model_storage`) are keyword
arguments on the constructors; the defaults were tuned on M4 Max. The `from_paths`
constructor accepts pre-extracted path elements for non-XGBoost sources.

## Performance

Sustained, paired, randomized A/B measurements on an Apple M4 Max against 16-thread
XGBoost 3.1.2 `pred_contribs`, identical models and inputs:

| workload | model | speedup |
|---|---|---|
| stress | 500 trees × depth 8, 12 features, 8,192 rows | **18.5×** |
| wide | 256 features | **14.4×** |
| multiclass | 8 classes | **15.6×** |

Figures are device- and workload-specific, not guarantees; smaller models at batch
volume can measure considerably higher (see the executed
[examples/](examples/README.md) notebooks). Methodology, caveats, and reproduction
commands live in [docs/performance.md](docs/performance.md); the raw archived
artifacts are under `benchmarks/results/`.

## Examples

Three executed notebooks in [`examples/`](examples/README.md): a quickstart, a paired
CPU-vs-GPU benchmark, and a tour of the execution modes and tuning knobs.

## Development

```bash
cmake -B build && cmake --build build     # portable core + Metal targets
ctest --test-dir build --output-on-failure
python3 tests/test_vs_xgboost.py build/reference_cli   # golden tests vs xgboost
```

The portable C++ core (path preprocessing, CPU reference oracle) builds and tests on
any platform; the Metal targets and differential fixture tests require Apple Silicon.
`python -m pytest` runs the importable suites; script-style suites print their own
invocation instructions. See [docs/how-it-works.md](docs/how-it-works.md) for the
architecture and [RELEASING.md](RELEASING.md) for the release process.

## License and attribution

Apache-2.0. This project is a port of NVIDIA's
[GPUTreeShap](https://github.com/rapidsai/gputreeshap) and retains its algorithmic
structure (see [NOTICE](NOTICE)). The TreeSHAP algorithm is due to Lundberg, Erion &
Lee ([arXiv:1802.03888](https://arxiv.org/abs/1802.03888)); the GPU formulation to
Mitchell, Frank & Holmes ([arXiv:2010.13972](https://arxiv.org/abs/2010.13972)).
Apple's [metal-cpp](https://github.com/apple/metal-cpp) headers are vendored under
`third_party/`.

## Trademarks

Metal is a trademark of Apple Inc., registered in the U.S. and other countries and
regions. Other names and marks (including XGBoost) are the property of their respective
owners. These names are used solely to describe interoperability. This project is an
independent open-source work and is not affiliated with, sponsored, or endorsed by
Apple Inc. or any other trademark owner.
