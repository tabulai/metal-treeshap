# Local program completion

This closes the local v0.1 implementation program on the M4 Max. Hosted GitHub/PyPI work is
deferred by request, and a real power trace remains an explicit privileged follow-up rather than
an inferred result. Neither item changes the local API, correctness, or acceleration result.

## Material Passport

- Origin skill: `academic-research-suite/experiment-agent`
- Origin mode: `run + validate`
- Machine: Apple M4 Max (40-core GPU), Mac16,6
- OS: macOS 26.2 (25C56)
- Verification status: `LOCAL_COMPLETE_EXTERNAL_DEFERRED`
- Version label: `v0.1.0-local-complete`

## Bounded real-data result

The runner trained each cell once, compiled both engines once, performed two excluded warmups,
then randomized five timed CPU/Metal calls within each iteration. Both engines used the same
model and 10,000-row matrix, verified by hashes. These are steady-state explain-call speedups,
not end-to-end training or first-call speedups.

| Dataset | Size (trees, depth, features) | CPU median | Metal median | Speedup | Max attribution difference |
|---|---:|---:|---:|---:|---:|
| Adult | small (10, 3, 14) | 0.006017 s | 0.001694 s | 3.55× | 1.19e-7 |
| Adult | medium (100, 8, 14) | 2.128081 s | 0.119221 s | 17.85× | 1.16e-5 |
| Covtype | small (70, 3, 54) | 0.060453 s | 0.005281 s | 11.45× | 7.15e-7 |
| Covtype | medium (700, 8, 54) | 11.410887 s | 1.059989 s | 10.77× | 1.98e-5 |
| California housing | small (10, 3, 8) | 0.008710 s | 0.001106 s | 7.88× | 7.45e-8 |
| California housing | medium (100, 8, 8) | 3.736883 s | 0.129581 s | 28.84× | 9.89e-6 |
| Fashion-MNIST | small (100, 3, 784) | 0.109052 s | 0.019996 s | 5.45× | 1.79e-7 |
| Fashion-MNIST | medium (1000, 8, 784) | 22.483277 s | 5.132889 s | 4.38× | 2.68e-5 |

All eight cells accelerated. The median speedup across cells was 9.32× and the geometric mean
was 8.98×. The maximum CPU/Metal attribution difference was 2.68e-5; the largest Metal
sum-to-margin residual was 2.73e-5. Both are well inside the predeclared 1e-3 gate.

The checked-in summary is
`benchmarks/results/realdata_m4max_20260712_summary.json`. The exact 97,174-byte raw artifact
had SHA-256 `3616849d6595ef011f353cafb5b974dac5acd3c1441245e7e33dc9373545280c`.
It began before the final `power_design` and implementation-provenance fields landed, so it is
reported as legacy evidence rather than silently rewritten to the final schema. A fresh
post-hardening paired smoke artifact passed `benchmarks/realdata_schema.json`, including exact
windows, provenance fingerprints, atomic checkpointing, and resume compatibility.

## What was closed

- The public Python API supports JSON/dict/Booster/sklearn-wrapper inputs, `explain`, direct
  calls, and the familiar `shap_values` spelling. Bias-only models and zero-row batches use the
  native zero-work path. Nullable pandas values become `NaN` without making pandas mandatory.
- The missing-native import path now preserves its diagnostic instead of raising `NameError`.
- Resume rejects changed arguments, device sets, implementation hashes, Python, XGBoost, or
  scikit-learn versions, and rejects artifacts associated with a prior power-capture request.
- CPU XGBoost and SHAP baselines expose exact call windows. The power correlator excludes gaps
  and retains job identities and explicit explained-row counts, including joules per explained
  row when energy is present. Real-data power runs use randomized homogeneous engine blocks,
  idle guards, and same-engine sampler-interval lead-in/tail conditioning.
- If `powermetrics` cannot start, the runner skips the expensive conditioning blocks and falls
  back to ordinary exact call windows. This path is tested; it cannot spend hours collecting no
  evidence.
- The final production default remains full-dispatch float atomics. Tested row tiling did not
  reproduce a benefit. Deterministic precise-Kahan mode remains the bit-repeatable accuracy mode.

## Validation boundary

Fresh Release and ASAN/UBSAN builds each passed 17/17 CTests on the actual Metal device. The
portable benchmark-tool suite passes 29 checks, including real-data schema rejection,
software-version-safe resume, power-capture guards, exact-window integration, and the optional
SHAP contract. The complete wheel matrix and installed public-API results are recorded in
`docs/06-v0.1-release-validation.md`.

The `large` real-data cells remain opt-in. They are not a v0.1 release gate: upstream-shaped
depth-16 ensembles can require multi-hour training and multi-GiB packed models, while the bounded
suite already spans 8–784 features, one to ten output groups, and 10–1000 actual trees.

Power software is complete, but `/usr/bin/powermetrics` requires local administrator authority.
This session had no passwordless authorization, so the recorded result is
`BLOCKED_BY_PRIVILEGE`; no performance-per-watt number is claimed. After `sudo -v`, the exact
reproduction command is in `docs/05-phase21-production-results.md`.

## Fallacy scan

No p-value or population-wide hardware claim is made. Look-elsewhere and garden-of-forking-paths
risks were controlled by keeping production choices tied to separate blocked finalist runs, not
the exploratory sweep. Regression-to-the-mean risk is reduced by paired randomized calls, but
five samples in one session do not justify a confidence interval. Simpson's paradox is avoided
by reporting every dataset/size cell instead of only an aggregate; the median and geometric mean
are descriptive. Ordinal encoding, training-row resampling, excluded setup, missing privileged
telemetry, and the legacy artifact fields are disclosed rather than treated as causal evidence.
