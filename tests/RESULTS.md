# Golden test results (Phase 0.6, CPU reference pipeline)

This file contains evidence produced by one local invocation only; it does not infer results for other XGBoost versions, machines, or the Metal engine.

- Generated (UTC): `2026-07-12T21:09:24+00:00`
- Platform: `macOS-26.2-arm64-arm-64bit-Mach-O`
- Python: `3.13.3`
- xgboost: `3.1.2`
- numpy: `2.1.3`
- Git HEAD: `897363d4f7400d2a52177d58a79970a11c6543c2` (worktree dirty: `true`)
- Portable source fingerprint (SHA-256): `8569dca2f2c68c7c334269af5ba5916a7e7c4c5086e75e86d0f22d63d5f336cc`
- Reference CLI: `/private/tmp/mtshap-final-release/reference_cli`
- Reference CLI SHA-256: `021d0fe8fa3278cf54f06352c4bac8ab62591321336364f406407c6c7cfb98ae`
- Invocation: `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 /Users/tunguz/Programming/GPU_Tree_Shap_explorer/metal-treeshap/tests/test_vs_xgboost.py /private/tmp/mtshap-final-release/reference_cli --write-results`
- Correctness gate: `0.001`

`err_vs_xgb` = max |phi − xgboost pred_contribs| (fp64 accumulation, intercept plumbed through the pipeline). `margin_err` = max |Σ phis − margin|. `fp32_abs` = max |fp32-accumulated − fp64-accumulated|. `fp32_rel_elem` = max elementwise relative error, floored at 0.001·max|phi|. `order_spread` = max PAIRWISE |Δ| across the natural + 5 seeded fp32 work orders — a CPU proxy for GPU atomic scheduling whose exact value is environment-dependent (stdlib shuffle); the on-device measurement happens in Phase 2.

| case | objective | booster | trees | depth | paths | err_vs_xgb | margin_err | fp32_abs | fp32_rel_elem | order_spread |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| regression-missing | reg:squarederror | gbtree | 25 | 3 | 200 | 4.17e-07 | 7.46e-07 | 9.04e-07 | 9.22e-06 | — |
| binary-depth6 | binary:logistic | gbtree | 50 | 6 | 2511 | 7.11e-07 | 1.49e-06 | 3.91e-06 | 2.65e-05 | — |
| multiclass-3 | multi:softmax | gbtree | 90 | 4 | 1369 | 2.70e-07 | 4.83e-07 | 1.07e-06 | 1.56e-05 | — |
| parallel-trees | multi:softmax | gbtree | 60 | 4 | 938 | 1.47e-07 | 2.94e-07 | 5.80e-07 | 1.34e-05 | — |
| dart | reg:squarederror | dart | 30 | 4 | 480 | 4.15e-07 | 5.83e-07 | 9.96e-07 | 1.25e-05 | — |
| missing-only-path | reg:squarederror | gbtree | 1 | 2 | 3 | 4.77e-07 | 4.77e-07 | 4.77e-07 | 5.96e-08 | — |
| stress-depth8x500 | reg:squarederror | gbtree | 500 | 8 | 65374 | 8.01e-06 | 8.32e-06 | 3.76e-05 | 6.27e-05 | 6.10e-05 |
| obj-reg-logistic | reg:logistic | gbtree | 10 | 3 | 80 | 1.04e-07 | 1.55e-07 | 1.37e-07 | 1.58e-06 | — |
| obj-logitraw | binary:logitraw | gbtree | 10 | 3 | 80 | 1.32e-07 | 2.04e-07 | 2.78e-07 | 1.30e-06 | — |
| obj-hinge | binary:hinge | gbtree | 10 | 3 | 73 | 5.64e-08 | 2.37e-07 | 1.51e-07 | 2.49e-06 | — |
| obj-poisson | count:poisson | gbtree | 10 | 3 | 80 | 4.68e-08 | 9.14e-08 | 2.58e-07 | 9.37e-06 | — |
| obj-gamma | reg:gamma | gbtree | 10 | 3 | 80 | 7.07e-08 | 1.24e-07 | 1.58e-07 | 3.52e-06 | — |
| obj-tweedie | reg:tweedie | gbtree | 10 | 3 | 80 | 8.46e-08 | 1.25e-07 | 1.32e-07 | 4.11e-06 | — |
| obj-absoluteerror | reg:absoluteerror | gbtree | 10 | 3 | 80 | 1.48e-07 | 2.45e-07 | 2.68e-07 | 1.84e-06 | — |
| obj-squaredlogerror | reg:squaredlogerror | gbtree | 10 | 3 | 70 | 7.95e-08 | 1.34e-07 | 1.03e-07 | 1.06e-05 | — |
| obj-quantile | reg:quantileerror | gbtree | 10 | 3 | 80 | 2.45e-07 | 1.75e-07 | 4.53e-07 | 1.68e-06 | — |

This invocation exercised every identity/logit/log objective case listed above against XGBoost `pred_contribs`; see `tools/extract_paths.py` `_MARGIN_LINK`. Unsupported-objective check: **passed** (survival:cox rejected by extractor).
