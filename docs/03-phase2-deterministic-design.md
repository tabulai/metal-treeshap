# Phase 2 deterministic accumulation design

## Outcome

The practical deterministic mode is a row-tiled, two-kernel pipeline:

1. `shap_partials` computes the existing float TreeSHAP recurrence but writes each
   non-root path-element contribution to a unique scratch slot. There are no atomics.
2. `reduce_partials_serial` gives one thread exclusive ownership of each active
   `(row, group, feature)` output and adds its slots in canonical `path_idx` order.

The production kernels are in `shaders/treeshap.metal`; the persistent scatter and
reduction metadata is produced by `BuildDeterministicPlan` in
`include/metal_treeshap/deterministic.h`. The on-device test compiles all experimental
kernels with Metal 3 and runs deterministic accumulation against the CPU float-recurrence
reference.

“Deterministic” here means bitwise repeatable for a fixed shader, model, and GPU
arithmetic mode. It does not mean equal to fp64 TreeSHAP: the cooperative recurrence and
the fixed reduction remain fp32.

## Memory bound

Let:

- `E` be packed path elements including roots;
- `P` be paths (one root each);
- `C = E - P` be non-root contributions per row;
- `A` be active `(group, feature)` cells;
- `T` be rows in the current tile.

The extra persistent model metadata is:

```text
4E + 16A bytes
```

The reusable scratch buffer is exactly:

```text
scratch(T) = 4TC bytes
T = max(1, floor(scratch_budget / (4C)))
```

`T` is additionally capped so `T * A` fits the shader's 32-bit reduction grid. This
makes scratch independent of the full row count. For example, if a model has 450,000
non-root elements, a 256 MiB budget holds 149 rows per tile; one million input rows still
uses the same scratch allocation. The normal input and final output allocations are not
included because every accumulation mode already needs them.

A dense `[row, bin, group, feature]` privatization was rejected: its memory grows as
`R * B * G * F`. A `[row, element]` scratch with a canonical slot per contribution is
both sparse and linear in the actual work.

## Canonical plan

Every non-root packed element receives one slot sorted by:

```text
(group, feature, path_idx, packed_element_index)
```

The last field is only a defensive tie-breaker; deduplication guarantees at most one
element for a `(path, feature)`. Slots for the same output cell are contiguous, so the
reducer needs only a 16-byte `{group, feature, begin, end}` record per active cell. No
indirection is needed while reducing.

Roots receive the sentinel `UINT32_MAX` and consume no scratch. Stage 1 nevertheless
keeps roots in the SIMD group because the TreeSHAP recurrence requires them.

## Host integration

The integrated `AccumulationMode::kDeterministic` path does the following:

1. During model compilation, build the deterministic plan and upload
   `partial_slot_by_element` and `active_cells` once.
2. Compile separate pipeline states for `shap_partials` and `reduce_partials_serial`.
3. Allocate one persistent private scratch buffer of `4 * T * C` bytes. The default budget
   is 256 MiB; lowering it releases oversized retained capacity. Both the configured budget
   and `MTLDevice::maxBufferLength` cap the tile, and one row must fit.
4. Initialize the full output. For each row tile, bind tile-relative X and output
   offsets, set `num_rows=T`, and dispatch stage 1 with
   `num_bins * ceil(T / rows_per_simdgroup)` SIMD-groups.
5. Issue `memoryBarrier(MTL::BarrierScopeBuffers)`, dispatch stage 2 over `T * A`
   threads, then issue another buffer barrier before reusing scratch for the next tile.
6. Copy the final output once after the command buffer completes.

The existing 32-bit stage-1 work guard remains required. The tile must satisfy both:

```text
num_bins * ceil(T / rows_per_simdgroup) * 32 <= UINT32_MAX
T * A <= UINT32_MAX
```

All tiles are encoded into one command buffer; separate command buffers would make small
scratch budgets disproportionately expensive.

## Reduction variants

`reduce_partials_serial` is the implemented initial strategy. Although one thread serializes a
single hot feature, rows expose many independent copies, while most output segments are
far shorter than 256 contributors.

A future parallel reducer could assign a complete threadgroup to one hot cell and use a
fixed reduction tree. It would remove the hot-cell critical path but waste most of a group
on short segments. Add it only if profiling justifies a compile-time segment-length split;
do not choose the threshold analytically.

The deterministic mode necessarily adds one scratch write and one scratch read per
contribution plus another dispatch. It is the accuracy/repeatability mode, not the throughput
default. Phase 2 selected plain float atomics for throughput; explicit SIMD pre-aggregation is
also available but slower on the tested M4 Max workloads.

## Completed validation

- All frozen fixtures versus the CPU float-recurrence/fp64-accumulation oracle.
- Bitwise equality across at least 100 identical runs.
- Bitwise equality across scratch budgets/tile sizes and row-bank settings.
- Deep-31/32-element, missing-only, DART, parallel-tree, multiclass, and 500-tree stress
  models.
- Peak scratch stays within the configured budget.
- End-to-end and GPU-only time against float atomics; report scratch allocation and
  output initialization separately.

All items above pass. Every frozen fixture runs through deterministic mode; the host test pins
100 bitwise-identical reruns and exact one-row/single-tile equality. On the 500-tree stress
workload, a 256 MiB budget uses 255.2 MiB active scratch across 37 tiles and takes 1.2905 s versus
0.6206 s for selected atomics. It has one output hash and zero repeat spread; max error is
1.08e-4. A 64 MiB budget increases the tile count to 147 and time to 2.2472 s. See
`docs/04-phase2-performance-results.md`.

## Split-hi/lo fixed-point assessment

An isolated Phase-2 prototype demonstrated valid modulo-`2^64` signed addition using two native
`atomic_uint` words. Each update atomically adds the low word, derives its carry from the
returned old value, and atomically adds `high + carry`. Because no reader observes the
pair until the kernel completes, the final 64 bits are order-independent. Reading the
pair during accumulation would be a torn and invalid snapshot.

This is technically feasible but is not recommended as the production deterministic
mode:

- it performs two contended atomics per contribution instead of one;
- a safe scale must be proven before dispatch because overflow wraps silently modulo
  `2^64`;
- quantization error is at most `N / (2S)` for `N` contributions and scale `S`, but a
  conservative no-overflow bound can force a small `S` on large ensembles;
- the per-value overflow flag in the prototype cannot detect final-sum signed overflow.

A host could derive a conservative bound from the sum of absolute leaf/path
contributions and choose the largest power-of-two scale satisfying

```text
S * bound + N/2 < 2^63.
```

That bound may be so loose that accuracy loses to fp32. Keep fixed64 as an experimental
bit-exact fallback and measure it, but prioritize the two-stage reduction because it has
no overflow mode and preserves the existing float contribution values exactly before
the final fixed-order sum.
