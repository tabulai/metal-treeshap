# Examples

Executed Jupyter notebooks demonstrating `metal-treeshap`. The committed outputs come
from one run on an Apple M4 Max (macOS 26, XGBoost 3.1.2) so they are readable on
GitHub without running anything; re-executing on your machine will produce your own
numbers.

| notebook | what it shows | runtime |
|---|---|---|
| [`01-quickstart.ipynb`](01-quickstart.ipynb) | train → compile → explain, correctness vs `pred_contribs`, missing values, pandas input, a first attribution chart | ~15 s |
| [`02-cpu-vs-gpu-benchmark.ipynb`](02-cpu-vs-gpu-benchmark.ipynb) | **paired, interleaved benchmark of XGBoost CPU `pred_contribs` vs the Metal GPU** on one 400-tree model across batch sizes, with accuracy checks, setup-cost break-even, and honest-methodology caveats | ~3 min |
| [`03-accumulation-modes.ipynb`](03-accumulation-modes.ipynb) | atomic vs deterministic modes: bit-repeatability demo, the cost of determinism, `last_timings` introspection, scratch-budget tiling, tuning knobs | ~30 s |

## Setup

Apple Silicon Mac (macOS 13+). Build and install the wheel from the repository root
(the package is not on PyPI yet), then add the notebook dependencies:

```bash
python3 -m pip install build
python3 -m build --wheel
python3 -m pip install dist/metal_treeshap-*.whl
python3 -m pip install jupyter matplotlib pandas "xgboost>=2.0"
jupyter lab examples/
```

The benchmark notebook deliberately keeps batch sizes modest (the CPU baseline runs
at roughly a millisecond per row on the 400-tree model); for publication-grade
measurements use the repository harnesses in `benchmarks/`, which add hashing, power
capture, and schema-validated artifacts.
