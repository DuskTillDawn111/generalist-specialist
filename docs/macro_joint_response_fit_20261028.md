# Macro response fit: three natural training captures, 2026-10-28

CPU-only training diagnostic. No frozen worker/collector/model was changed, no GPU was initialized, and no validation labels were read. Native CUDA API numerical authentication remains **UNQUALIFIED**.

Each capture has 50 requests and 334 disjoint macros: 1002 rows total. Entry, all forward-to-next-forward responses, the last forward-to-native-END response, native-END-to-READY tails, and coarse OCR are retained. All 1002 heldout rows have an exact training nuisance group. The final language forward includes selection/model-specific final work (including the possible TTS scan); this is not separately added as a wait.

## Source and clock checks

All three macro documents were extracted into the artifact directory. Capture01 equals the original response-macros-v1-03.json exactly. Before fitting, each complete source closure was hash checked; the raw/protocol verifier and training-evidence review were rerun into temporary exclusive directories and compared exactly; macros were then independently rederived and compared exactly. The final report binds 596 distinct source/evidence receipts, checked again after fitting.

The response uses worker occurrence timestamps START to READY. Features change only at parent collector processing timestamps and use the processed prefix at each interval left endpoint. A true stream query does not close a visible macro; selection/GPU tails persist until next forward BEGIN or native END. Internal raw events and ordinals remain training labels, including notifications processed after the macro has ended. They are not runtime covariates known at macro BEGIN.

## Likelihood

For macro i, define V_i(T; beta) = integral_0^T exp(-beta^T x_i(t)) dt in milliseconds. Let log W_i be Normal(mu_g, sigma^2), with g = owner/input/task/macro-id/semantic-phase/kind/forward-shape. Baseline means, beta and variance are estimated jointly by profiling mu_g = mean(log V_i) within each group and sigma^2 = mean squared residual. Coefficients are free real numbers; negative values are allowed.

The exact response NLL is sum[log V_i + log(sigma sqrt(2 pi)) + (log V_i - mu_g)^2/(2 sigma^2) + beta^T x_i(T-) ]. The last term is the endpoint density Jacobian. Omitting it changes the model. The optimizer minimizes this NLL divided by the number of training rows, preserving its argmin while reducing cancellation and gradient scaling. The reported optimizer flags are SciPy flags, without a post hoc success override.

Nonlinear response Jacobian ranks include every baseline nuisance mean plus beta; sigma is excluded from this particular rank diagnostic. It is not a capture-robust uncertainty estimate. A common sigma is used across all macro classes; no within-request dependence law is fitted.

## Whole capture leaveout

Each fold fits on two complete training captures and scores the third. No phase or request from the heldout capture enters fitting. Candidate selection uses the equal-capture mean heldout conditional NLL, among folds that are supported, full rank and report optimizer convergence. These three blocks are training repetitions, not independent final validation.

| Candidate | Heldout mean NLL | Conditional MAE ms | Conditional RMSE ms | 90% reference interval coverage | Selection eligible |
|---|---:|---:|---:|---:|---|
| baseline | 3.841226 | 36.303 | 130.811 | 0.8723 | True |
| visible_count | 3.676490 | 25.344 | 80.227 | 0.8972 | False |
| visible_phase_aware | 3.668805 | 25.756 | 81.210 | 0.8992 | False |

Phase-aware has the lowest conditional NLL, while count has the lowest conditional wall MAE. Count fold01 and phase-aware folds01/02 return precision loss, despite small mean-NLL gradient norms. The phase-aware full fit also returns precision loss. The strict optimizer gate therefore leaves **baseline** as the selected eligible candidate; this is not a claim that baseline has the best heldout score.

| Candidate | Full-fit beta | Full rank | Full optimizer flag | Fold coefficients |
|---|---|---|---|---|
| baseline | [] | 48/48 | True | [[], [], []] |
| visible_count | [0.19391108416809685] | 49/49 | True | [[0.18919427858716575], [0.19306188604162325], [0.20082361286235312]] |
| visible_phase_aware | [0.24604298454563775, -0.09497094301192988] | 50/50 | False | [[0.23344236396486248, -0.08065068850523147], [0.24046162760982115, -0.08722013236733545], [0.26498541705271106, -0.1158686215243327]] |

Count beta ranges 0.189194–0.200824 across folds (span 0.011629). Phase-aware count beta ranges 0.233442–0.264985, same-phase beta -0.115869–-0.080651. These are association diagnostics conditional on visible evidence, not causal hardware coefficients or calibrated parameter precision.

## Wall errors and their scope

For each heldout macro, the report retains its actual wall, conditional NLL, PIT, median reference-clock inversion and error. If the median reference work exceeds the work integrated over the actual macro path, inversion explicitly assumes the last observed peer state persists. This happens in 460 count rows and 462 phase-aware rows; it is a diagnostic continuation assumption, not a learned notification or future-path law. Baseline needs no future peer states because beta has dimension zero. Neither model provides a prospective whole queue simulation.

Below, errors pool the three heldout captures for each owner/macro ID. Units are milliseconds; every individual row and every capture-specific summary remains in report.json. decode_0004 is the final-forward response and includes final native work.

| Owner | Macro | Rows | Baseline MAE | Count MAE | Phase-aware MAE |
|---|---|---:|---:|---:|---:|
| generalist_2#0 | decode_0001 | 54 | 8.353 | 5.093 | 4.516 |
| generalist_2#0 | decode_0002 | 54 | 7.788 | 4.587 | 3.758 |
| generalist_2#0 | decode_0003 | 54 | 6.914 | 4.071 | 3.145 |
| generalist_2#0 | decode_0004 | 54 | 88.584 | 67.475 | 68.360 |
| generalist_2#0 | entry_to_first_forward | 54 | 5.514 | 4.833 | 4.647 |
| generalist_2#0 | native_end_to_worker_ready | 54 | 0.440 | 0.444 | 0.459 |
| generalist_2#0 | prefill_0000 | 54 | 136.050 | 85.130 | 92.042 |
| ocr#0 | entry_to_first_forward | 12 | 0.033 | 0.043 | 0.041 |
| ocr#0 | native_end_to_worker_ready | 12 | 0.017 | 0.052 | 0.063 |
| ocr#0 | native_ocr | 12 | 489.447 | 290.384 | 244.975 |
| text#0 | decode_0001 | 54 | 4.697 | 6.932 | 6.343 |
| text#0 | decode_0002 | 54 | 3.660 | 7.630 | 6.795 |
| text#0 | decode_0003 | 54 | 2.182 | 7.264 | 6.204 |
| text#0 | decode_0004 | 54 | 1.741 | 6.969 | 6.291 |
| text#0 | entry_to_first_forward | 54 | 2.766 | 3.074 | 2.846 |
| text#0 | native_end_to_worker_ready | 54 | 7.814 | 8.592 | 8.990 |
| text#0 | prefill_0000 | 54 | 155.600 | 99.389 | 109.742 |
| text#1 | decode_0001 | 30 | 4.238 | 7.194 | 6.624 |
| text#1 | decode_0002 | 30 | 2.949 | 7.777 | 7.116 |
| text#1 | decode_0003 | 30 | 2.160 | 9.061 | 8.286 |
| text#1 | decode_0004 | 30 | 1.190 | 8.306 | 7.678 |
| text#1 | entry_to_first_forward | 30 | 1.728 | 1.684 | 1.589 |
| text#1 | native_end_to_worker_ready | 30 | 11.033 | 11.281 | 11.417 |
| text#1 | prefill_0000 | 30 | 215.654 | 124.314 | 136.050 |

## Tests and remaining gaps

`python -m unittest tests.test_macro_joint_response_fit_v1 -v`: **8 passed**. Tests cover free positive/negative beta recovery with joint nuisance means, likelihood endpoint Jacobian, analytic versus numerical mean-NLL gradients for constant and dynamic paths, exact path integration and inversion, nuisance-confounded rank, entry/OCR/tail retention, unseen support as UNKNOWN, clock/domain rejection, exact-rederivation rejection of rehashed labels and source tampering, and whole-capture isolation.

Internal host-END timing, query outcomes, delayed notifications (including cross-macro delivery), future generated-work templates, and dependence among macro residuals remain **UNKNOWN_NOT_FITTED**. Future shapes/stops/new input support are not qualified. Current-stream observations do not prove all-stream completion or effective SM service. Queue provider eligibility, independent prediction accuracy and final gain eligibility all remain false.

Artifacts: `artifacts/predictor-macro-joint-response-fit-v1-natural-20261028-review01/{capture01-macros.json,capture02-macros.json,capture03-macros.json,report.json}`. New implementation: `tools/predictor_validation_20261027/macro_joint_response_fit_v1.py`. No frozen v1 source was edited.
