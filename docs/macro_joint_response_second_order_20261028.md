# Analytic second-order macro response review, 2026-10-28

The first report and its original BFGS flags remain intact. This report uses the same three complete natural training captures and fixed feature/nuisance design. All raw/protocol reviews, full training reviews and macro extractions were exactly rederived again. The new report binds 598 receipts. No GPU or validation data was used. Native CUDA API numeric authentication remains **UNQUALIFIED**.

## Analytic Hessian

For each row let l=log V, d=-E_w[x], C=Cov_w[x], r=l-mean_group(l), v=mean(r²), a=mean(r d), and d_c=d-mean_group(d). For profile mean NLL:

```text
g = mean(d) + a/v + mean(endpoint_x)
H = mean(C) + [mean(d_c d_cᵀ) + mean(r C)]/v − 2 a aᵀ/v²
```

The endpoint Jacobian is linear in beta, so it contributes to g and zero to H. Group-mean derivatives cancel in a through centered residuals; d_c is essential in H. The dynamic-path test verifies this Hessian against central differences of the exact analytic gradient and verifies the objective/gradient against the separately implemented v1 profile likelihood.

## Solver and qualification

Trust-exact minimization uses starts 0, +0.25 and -0.25 in every coordinate, with exact analytic Hessian and gtol=1e-10. Each trust solution is then independently refined and verified using LM gradient-root solving with the analytic Hessian as its Jacobian. Root tolerance options are fixed at 1e-12; qualification requires score norm <1e-10, strictly positive Hessian eigenvalues, all three successful roots agreeing within 1e-7, all three positive-curvature trust endpoints agreeing with that root within 1e-7, and full nuisance-inclusive Jacobian rank. None of these thresholds uses heldout labels.

This is a separate gradient-root stationarity certificate. Original BFGS and trust-exact success=False flags remain False. Some trust attempts still report numerical approximation failure close to the solution; these are not relabelled as successful minimizers. Every retained gradient-root attempt reports its own genuine success=True.

Counterexample: a blind root search from the original starts can drift to |beta|≈1e16 where the score approaches zero but the profile Hessian is negative or zero. A solver success flag or small score alone cannot certify a minimum. Tests explicitly reject negative/zero curvature and nuisance-absorbed coefficients. The final solver verifies roots near independently minimized trust solutions. Matching three starts establishes local numerical agreement, not proof of a global minimum.

Full-fit qualification **and** all leaveout-fold qualifications are required for selection. A test forces the full fit to fail with good folds and confirms no model is eligible.

## Real numerical results

| Model / block | Beta | Score norm | Hessian eigenvalues | Root maximum disagreement | Original BFGS success | Trust success flags |
|---|---|---:|---|---:|---|---|
| visible_count / full | [0.19391108324433062] | 1.11e-16 | [10.645523555670007] | 8.33e-17 | True | [True, True, False] |
| visible_count / 75fce60e | [0.1891942764682469] | 1.11e-16 | [10.982953806915994] | 0 | False | [True, True, False] |
| visible_count / 96f5d8d1 | [0.1930618860357648] | 3.33e-16 | [10.105512096069287] | 2.78e-17 | True | [True, True, False] |
| visible_count / edb880c5 | [0.20082361286235365] | 7.77e-16 | [11.046800021140404] | 1.11e-16 | True | [True, True, False] |
| visible_phase_aware / full | [0.24604298490981574, -0.09497094056843268] | 6.75e-16 | [1.5111736588655762, 14.900011481022112] | 2.39e-16 | False | [True, False, False] |
| visible_phase_aware / 75fce60e | [0.23344235859774476, -0.08065068350067686] | 1.55e-15 | [1.5744592306893508, 15.27012311223124] | 1.69e-16 | False | [False, False, False] |
| visible_phase_aware / 96f5d8d1 | [0.24046161508291222, -0.08722011759763419] | 8.01e-16 | [1.4285541279367937, 13.998925013374382] | 2e-16 | False | [True, False, False] |
| visible_phase_aware / edb880c5 | [0.2649854166956484, -0.11586861993135873] | 2.22e-16 | [1.5573674540645213, 15.72970685108538] | 3.12e-16 | True | [False, False, False] |

Every full fit and fold is supported and passes the independent second-order qualification. Ranks remain baseline 48/48, count 49/49 and phase-aware 50/50. The baseline has an analytic profiled solution with no beta optimizer.

| Model | Heldout conditional mean NLL | Conditional MAE ms | Conditional RMSE ms | 90% interval coverage | Selection eligible |
|---|---:|---:|---:|---:|---|
| baseline | 3.841226414 | 36.303419 | 130.811161 | 0.872255 | True |
| visible_count | 3.676489515 | 25.343659 | 80.227212 | 0.897206 | True |
| visible_phase_aware | 3.668804704 | 25.755667 | 81.209862 | 0.899202 | True |

**Selected: visible_phase_aware**, by equal-capture heldout conditional NLL. Count has slightly lower wall MAE, which is not the predeclared selection objective. The second-order solutions differ from the original BFGS beta by only about 1e-8 or less; scores and errors agree to numerical precision. This supports a numerical stationarity explanation of the original precision-loss flags, without changing those flags.

Every individual macro error and every per-capture/per-macro summary is retained in review02/report.json. The response errors remain conditional on actual heldout peer paths. For 462 phase-aware rows, median inversion extends the last observed peer state beyond the actual macro endpoint; this assumption is explicit and is not a generated future path. All 1002 nodes are retained and supported.

## Tests and remaining gaps

8 v1 tests and 6 second-order tests passed. The latter cover exact dynamic Hessian, positive/negative beta recovery across starts, false stationarity at very large beta, negative/zero/saddle curvature, nuisance-confounded rank, and the full-fit eligibility gate.

Internal H/host-END timing, query outcomes, cross-macro delayed notification law, future work templates and residual dependence remain UNKNOWN_NOT_FITTED. Three whole training captures do not estimate capture-robust parameter uncertainty. Queue provider completeness, independent accuracy, effective SM service, native CUDA numerical authentication and final gain remain unqualified.

New source: `tools/predictor_validation_20261027/macro_joint_response_second_order_v1.py`. Tests: `tests/test_macro_joint_response_second_order_v1.py`. New result: `artifacts/predictor-macro-joint-response-fit-v1-natural-20261028-review02/report.json`. Original review01 report and original fit module were not changed.
