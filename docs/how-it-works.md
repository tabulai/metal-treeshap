# How metal-treeshap works

A guided tour of the pipeline, for the curious and for contributors. The algorithm is
NVIDIA's GPUTreeShap (Mitchell, Frank & Holmes,
[arXiv:2010.13972](https://arxiv.org/abs/2010.13972)) reformulated for Apple GPUs; the
attribution method is TreeSHAP (Lundberg, Erion & Lee,
[arXiv:1802.03888](https://arxiv.org/abs/1802.03888)).

## The pipeline

```
xgboost model (or saved .json) ──tools/extract_paths.py──► paths + per-group intercepts
                                             │  Preprocess() [host, portable, validated]
                                             ▼
                              dedup → validate → BFD bin-pack → sort → segments (+fp64 bias)
                                             │
                    ┌────────────────────────┴──────────────────────┐
                    ▼ (any platform)                                ▼ (Apple Silicon)
        reference/reference_shap.h                       shaders/treeshap.metal
        scalar CPU oracle, fp64/fp32 accum,              atomic/SIMD/deterministic kernels,
        shuffled-order mode                              compiled-model host (persistent buffers)
                    │                                                │
                    └──────── numerically matched phis ─────────────┘
```

## Path decomposition

TreeSHAP's insight is that a tree's SHAP values decompose over its **root-to-leaf
paths**: each unique path contributes independently, weighted by how the explained row
and the "background" cover distribution flow through it. `tools/extract_paths.py`
walks the raw XGBoost JSON model (never the text dump) and emits one element per
(path, feature) edge: the split interval `[lower, upper)`, the missing-value routing
flag, the cover-derived `zero_fraction`, and the leaf value. It reads the
authoritative `tree_info` group mapping, both DART `weight_drop` layouts, and vector
base scores, and converts intercepts to margin space through an empirically verified
per-objective link table — unknown objectives are rejected, never guessed.

## Host preprocessing (portable C++)

`include/metal_treeshap/preprocess.h` mirrors GPUTreeShap's device preprocessing with
plain C++20 (on unified memory there is no transfer to hide, so the CPU does this
once per model):

1. **Deduplicate** repeated features along a path (intersect intervals, multiply
   fractions).
2. **Validate twice** — raw checks *before* dedup (merging can launder malformed
   input: duplicate roots collapse, invalid fractions can multiply into valid ones)
   and structural checks after. Anything that could reach out-of-bounds GPU indexing
   is rejected here.
3. **Bin-pack** variable-length paths into 32-lane bins (best-fit decreasing), so one
   GPU SIMD-group processes a full bin. Max path length is 32 (root + depth 31).
4. **Sort** elements so each bin's paths are contiguous, and compute per-group bias
   in fp64.

## The kernel

One SIMD-group (32 lanes, one per path element) evaluates a bin against a bank of
rows. The two TreeSHAP recurrences — *extend*, which builds permutation weights down
the path, and *unwind*, which each lane applies for its own feature — run
cooperatively via lane shuffles, exactly like the CUDA warp version but expressed
with Metal `simd_shuffle` and a sorted-contiguity trick replacing CUDA's labeled
partitions. NaN routing uses an integer bit test so the fast-math compiler can never
fold it away, and `+inf` explicitly follows the branch XGBoost would take.

Each element's contribution `phi` then has to reach the output row. That's where the
three **accumulation modes** differ:

- **atomic** (default): a plain `atomic_float` add per non-root element. Fastest;
  float addition order varies with scheduling, so reruns differ at ~1e-6.
- **deterministic**: every element owns a canonical scratch slot; a two-stage Kahan
  reduction (fixed 256-slot chunks, then a fixed-order combine per output cell) sums
  them with a shape that depends only on the model — **bit-identical output** across
  runs, threadgroup sizes, and row-tile sizes, plus better accuracy than plain fp32
  addition. Scratch is tiled under a configurable budget. Costs ~1.8× atomic.
- **simdgroup**: pre-aggregates matching output keys within the SIMD-group before one
  atomic per key. Kept as an experiment for other GPU/model shapes.

Apple GPUs have no fp64, so accuracy comes from structure instead: fp64 bias
computation on the host, Kahan compensation in the deterministic reducer (compiled
with fast math disabled), and continuous differential testing.

## The host (metal-cpp)

`src/metal_host.hpp` compiles a model once into persistent GPU buffers and reuses
them across calls. Design points worth knowing:

- **Zero-copy I/O**: page-aligned inputs and outputs are wrapped with `bytesNoCopy`;
  the Python layer allocates the result page-aligned so the GPU writes NumPy-owned
  memory directly, and the output prefill (zeros + bias) runs as a GPU kernel.
- **Lazy deterministic metadata**: the deterministic plan and its buffers build on
  first deterministic use, so the default atomic path never pays for them.
- **No-throw encoding window**: all validation and allocation happens before a
  command encoder opens (a throw with an open encoder is a process abort in Metal),
  with an RAII guard as defense in depth.
- Kernels compile **at runtime** from `shaders/treeshap.metal` when no offline
  metallib is present — no Xcode Metal toolchain required.

## How correctness is maintained

Three independent layers, all in CI:

1. **Golden tests** (`tests/test_vs_xgboost.py`): train real models across the
   objective allowlist and compare elementwise against
   `xgboost.predict(pred_contribs=True)` — plus local-accuracy (row sums equal the
   margin) and fp32-path gates.
2. **Property tests** (`tests/test_property_additivity.cpp`): randomized forests
   checked against an independent exact-Shapley oracle, including depth-31 bins,
   repeated features, tiny covers, and NaNs.
3. **Differential fixtures** (`tests/test_fixture.py`): frozen models replayed
   through both the CPU reference and every Metal mode; deterministic mode is pinned
   bit-stable across 100 reruns.

The CPU reference (`reference/reference_shap.h`) reproduces the kernel's float
arithmetic lane-for-lane, which is what makes CPU↔GPU differences attributable to
scheduling rather than logic.
