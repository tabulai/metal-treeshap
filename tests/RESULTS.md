# Golden test results (Phase 0.6, CPU reference pipeline)

xgboost 3.1.2, numpy 2.4.4; correctness gate 0.001.

`err_vs_xgb` = max |phi − xgboost pred_contribs| (fp64 accumulation, intercept plumbed through the pipeline). `margin_err` = max |Σ phis − margin|. `fp32_abs` = max |fp32-accumulated − fp64-accumulated|. `fp32_rel_elem` = max elementwise relative error, floored at 0.001·max|phi|. `order_spread` = max PAIRWISE |Δ| across the natural + 5 seeded fp32 work orders — a CPU proxy for GPU atomic scheduling whose exact value is environment-dependent (stdlib shuffle); the on-device measurement happens in Phase 2.

| case | objective | booster | trees | depth | paths | err_vs_xgb | margin_err | fp32_abs | fp32_rel_elem | order_spread |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| regression-missing | reg:squarederror | gbtree | 25 | 3 | 200 | 4.17e-07 | 7.52e-07 | 9.10e-07 | 9.95e-06 | — |
| binary-depth6 | binary:logistic | gbtree | 50 | 6 | 2511 | 7.06e-07 | 1.48e-06 | 3.91e-06 | 2.64e-05 | — |
| multiclass-3 | multi:softmax | gbtree | 90 | 4 | 1369 | 2.71e-07 | 4.77e-07 | 1.07e-06 | 2.22e-05 | — |
| parallel-trees | multi:softmax | gbtree | 60 | 4 | 938 | 2.05e-07 | 2.91e-07 | 5.75e-07 | 1.15e-05 | — |
| dart | reg:squarederror | dart | 30 | 4 | 480 | 4.15e-07 | 5.84e-07 | 1.00e-06 | 1.38e-05 | — |
| stress-depth8x500 | reg:squarederror | gbtree | 500 | 8 | 65374 | 8.02e-06 | 8.34e-06 | 3.76e-05 | 6.45e-05 | 4.20e-05 |
| obj-reg-logistic | reg:logistic | gbtree | 10 | 3 | 80 | 9.58e-08 | 1.61e-07 | 1.43e-07 | 1.63e-06 | — |
| obj-logitraw | binary:logitraw | gbtree | 10 | 3 | 80 | 1.32e-07 | 2.01e-07 | 2.78e-07 | 1.52e-06 | — |
| obj-hinge | binary:hinge | gbtree | 10 | 3 | 73 | 5.66e-08 | 2.42e-07 | 1.52e-07 | 2.69e-06 | — |
| obj-poisson | count:poisson | gbtree | 10 | 3 | 80 | 4.68e-08 | 9.15e-08 | 2.59e-07 | 9.38e-06 | — |
| obj-gamma | reg:gamma | gbtree | 10 | 3 | 80 | 6.94e-08 | 1.24e-07 | 1.57e-07 | 5.39e-06 | — |
| obj-tweedie | reg:tweedie | gbtree | 10 | 3 | 80 | 8.64e-08 | 1.25e-07 | 1.32e-07 | 1.97e-06 | — |
| obj-absoluteerror | reg:absoluteerror | gbtree | 10 | 3 | 80 | 1.47e-07 | 2.47e-07 | 2.64e-07 | 1.84e-06 | — |
| obj-squaredlogerror | reg:squaredlogerror | gbtree | 10 | 3 | 70 | 8.64e-08 | 1.24e-07 | 7.23e-08 | 1.33e-05 | — |
| obj-quantile | reg:quantileerror | gbtree | 10 | 3 | 80 | 2.45e-07 | 1.74e-07 | 4.53e-07 | 1.19e-06 | — |

Objective links verified empirically (identity/logit/log, see tools/extract_paths.py `_MARGIN_LINK`); objectives outside the allowlist are rejected (checked with survival:cox). Cross-version: suite verified on xgboost 2.0.3 and 3.1.2. External M4 Max validation (v3): the compiled-model host logic ran ALL SIX frozen fixtures end-to-end (shader runtime-compiled from source) with max Metal error 6.5e-6, across rows_per_simdgroup in {1, 7, 1024}, including empty-model, zero-row, intercept, repeated-call and invalid-tuning behavior. src/main_metal.cpp + `test_fixture.py --metal-cli` make that run repository-reproducible.
