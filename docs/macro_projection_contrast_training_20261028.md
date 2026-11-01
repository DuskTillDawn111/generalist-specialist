# Partial decoder projection contrast in macro training, 2026-10-28

This candidate is a declared structural proxy, not measured GPU demand or a compute/memory classification. It excludes attention/KV, DRAM/cache, quantized kernel costs, LM head, embedding, norms/activation/RoPE, and the separate LoRA work/dtype uncertainty. Existing models, workers, captures, fit modules and reports are unchanged. Only CPU training data is used.

## Computable definition

For a source-bound language decoder with actual parent-processed BEGIN shape, let R=batch×sequence input rows, P=sum(in×out) across seven projection families/layers, and A=sum(in+out) across those projections. At activation dtype size b:

```text
F = 2 R P                         nominal dense-equivalent GEMM FLOPs
B = b P + b R A                   hypothetical dense weights + logical IO operand sum
I = F / B                        partial decoder projection intensity
c(i,j) = abs(log(I_i / I_j))
D(i,j) = c/(1+c)                  bounded partial projection intensity contrast
S(i,j) = 1/(1+c)                  bounded partial projection intensity similarity
```

B is not actual quantized weight storage, resident VRAM, memory capacity demand, live peak or DRAM traffic. It intentionally includes repeated and aliased per-affine operands. I is not a roofline ratio against matched hardware peaks. The existing static binding separately preserves LoRA declarations, but this candidate uses only the base dense-equivalent projection proxy. Its name/caption always says partial decoder projection contrast; neither D nor S establishes physical complementarity or interference benefit.

## Processed-prefix support and missingness

At each causal macro exposure time, own structure is supported only for a language macro whose exact owner/request/macro BEGIN has already been processed by the parent. It never uses the physically begun but not-yet-processed shape. Entry, READY tails and OCR own projections are unsupported. A peer's structure comes from the BEGIN corresponding to that peer's current active visible macro identity. A true stream query does not clear that macro, because further GPU/selection work may follow. This last visible descriptor is an association proxy, not actual current resource use.

Each time piece records the exact BEGIN identity, batch, sequence, phase, binding digest, occurrence and processing timestamps, raw-event ordinal, own_supported, every pair's support/reasons, and:

- N: all parent-visible peer macros.
- K: pairs for which both own and peer partial projection descriptors are supported.
- U=N−K: unknown pairs, including every peer if own is unsupported.
- sum_known D and sum_known S, sums over K supported pairs only.
- all-peer sum/mean is null when U>0; the mean is also null when N=0.

An empty known set may have sum_known D=0, but this is never an imputation for unknown pairs: K/U and own_supported are retained. Even if N=0, own_supported remains explicit. All attention/KV and quantized runtime costs remain UNKNOWN_EXCLUDED_FROM_PARTIAL_PROXY for otherwise supported descriptors.

Since S+D=1 for each supported pair, sum_known S+sum_known D=K=N−U. Including S and D together with N and U creates an exact alias. The candidates fit D only, with missingness and peer count explicitly included.

## Complete-domain candidates and inference

The prespecified candidates are baseline; visible phase-aware count; phase-aware count plus U and own_unsupported; and that missingness candidate plus sum_known D. All use the same owner/input/task/macro-id/phase/kind/forward-shape nuisance means and a profiled common lognormal variance. Every one of the 1002 macro responses is retained, including entry, final-forward work, READY and OCR. No language subset substitutes for this complete-domain comparison.

The time change and endpoint density Jacobian are identical to the previous macro fits. The independent analytic profile Hessian is used in trust-exact minimization from 0/+0.25/−0.25, followed by a separate gradient-root stationarity verification. Raw optimizer flags are retained; full-fit and all-fold numerical qualification, nuisance-inclusive full rank and exact support are required for selection. Numerical thresholds were fixed before heldout scoring. Candidate selection uses whole-training-capture leaveout conditional NLL, never validation labels.

Nuisance-adjusted derivative rank is reported in addition to full Jacobian rank. A full raw feature rank is insufficient if owner/input/macro nuisance absorbs a feature. Three whole captures are the replication units; 1002 macro rows and 230 forwards per capture are not independent capture replications.

## Conditional versus prospective

Training scores condition on the actual heldout processed peer path. A prospective provider must receive a qualified own predispatch shape/batch/phase template or generate its BEGIN processing notice; exact model/precision/source identities; current peer owner/request/phase and actually processed BEGIN shapes with notice ages; and causally generated future peer BEGIN/native END/processing events, including delayed cross-macro delivery. It also requires generation stop/shape, missingness and residual dependence laws, internal H/query/selection/RESULT/READY notification laws, and independent whole-capture queue accuracy validation. These inputs/laws remain unqualified here.

Main queue provider, effective SM service, physical compute/memory classification, measured DRAM/cache evidence, proven complementarity and final gain remain false. Native CUDA API numerical authentication remains UNQUALIFIED. Current source/header declarations do not independently certify historical runtime-loaded operator identity.

## Actual training results and the retained rank failure

The source-bound full response domain contains 1002 macros (690 language, 312 entry/READY/OCR) and 9697 causal time pieces. Summed request occupancy is 140.111 seconds; own partial projection support covers 124.794 seconds. Integrated visible-peer exposure is 203.902 owner-pair seconds: 152.548 known (74.814%) and 51.354 unknown. These are integrated evidence denominators, not GPU service or elapsed capture time. Own unsupported with no visible peers still occupies 4.724 seconds and remains explicitly flagged. Language BEGIN-not-yet-processed exposure is 96.169 ms across all 690 language macros.

Independent post-extraction checks traversed every piece: N=K+U, all-peer aggregate is null for U>0, own unsupported implies K=0, and every supported own/peer descriptor has processing time <= piece cutoff. There are nine observed (owner,phase,batch,sequence) templates. All observed batches are one; synthetic batch-change tests validate arithmetic, not prospective batch response accuracy.

The original own-unsupported-main-effect candidates are retained in review01. Their own coefficient grows to about 26–29, suppressing the short language pre-processing portion; host/OCR own support is constant within nuisance groups. The resulting near-flat direction has curvature around 1e-15, roots from different starts disagree, and joint ranks are 51/52 (missingness only) and 52/53 (with contrast). They remain ineligible regardless of their lower conditional scores. This is a weak/limiting identification failure; it is not a newly identified physical resource class.

A separate follow-up module preserves own_supported for all pieces but does not fit a standalone own-unsupported response main effect. The owner/input/macro nuisance still supplies the complete empirical response baseline, U models unknown pair interactions, and D sums only supported pair interactions. This does not assign an unknown own resource the value zero; no own structural service model or extrapolation is supplied. The new training design follows the observed training rank failure and is explicitly not an independent validation. The first failed report and module are unchanged.

| Candidate | Heldout mean NLL | Conditional MAE ms | Joint rank | Nuisance-adjusted rank | Eligible |
|---|---:|---:|---|---|---|
| baseline | 3.841226414 | 36.303419 | 48/48 | 0/0 | True |
| visible_phase_aware | 3.668804704 | 25.755667 | 50/50 | 2/2 | True |
| phase_and_projection_missingness | 3.624019544 | 25.595560 | 51/52 | 3/4 | False |
| phase_missingness_projection_contrast | 3.621676250 | 25.162031 | 52/53 | 4/5 | False |
| phase_and_unknown_projection_pairs | 3.624938136 | 25.580169 | 51/51 | 3/3 | True |
| phase_unknown_pairs_projection_contrast | 3.622455663 | 25.126852 | 52/52 | 4/4 | True |

Among the supported full-rank candidates, whole-training-capture conditional NLL selects **phase_unknown_pairs_projection_contrast**. Full fit beta in [N, same-phase count, U, sum_known D] order is [0.492810600, -0.270952957, -0.315928737, -0.125422068]. The free contrast coefficient is negative, but this is an association conditional on observed prefixes and other fitted counts, not proven complementarity.

Full-fit score norm is 5.58e-16, Hessian eigenvalues [0.165119, 1.365912, 2.970184, 19.343726], joint rank 52/52 and nuisance-adjusted derivative rank 4/4. All three full training-capture folds also pass rank/curvature/stationarity: score norms <=1.31e-15, minimum Hessian eigenvalues >=0.142091. All original trust/root flags and per-start solutions are retained.

| Heldout capture prefix | Contrast beta | Delta conditional NLL from adding D to phase+U |
|---|---:|---:|
| 75fce60e | -0.099902693 | -0.004617411 |
| 96f5d8d1 | -0.121353146 | -0.003976362 |
| edb880c5 | -0.149637533 | +0.001146354 |

The mean contrast-only improvement is -0.002482473 NLL: two folds improve and one worsens. Most improvement versus the original phase-aware candidate comes from U, not from D. Three paired capture differences do not establish significance, capture-robust uncertainty, causal interference reduction or a prospective queue benefit. The new candidate median error also uses explicit endpoint-state continuation in 476 rows.

All per-row and per-macro errors are retained in review02. The following table pools all three heldout captures and compares the original phase-aware model with phase+U and phase+U+D. Errors are conditional MAE in milliseconds.

| Owner | Macro | Rows | Phase-aware | Phase+U | Phase+U+D |
|---|---|---:|---:|---:|---:|
| generalist_2#0 | decode_0001 | 54 | 4.516 | 3.358 | 3.494 |
| generalist_2#0 | decode_0002 | 54 | 3.758 | 2.269 | 2.527 |
| generalist_2#0 | decode_0003 | 54 | 3.145 | 1.709 | 2.054 |
| generalist_2#0 | decode_0004 | 54 | 68.360 | 61.522 | 61.984 |
| generalist_2#0 | entry_to_first_forward | 54 | 4.647 | 4.938 | 4.917 |
| generalist_2#0 | native_end_to_worker_ready | 54 | 0.459 | 0.440 | 0.442 |
| generalist_2#0 | prefill_0000 | 54 | 92.042 | 85.946 | 84.015 |
| ocr#0 | entry_to_first_forward | 12 | 0.041 | 0.032 | 0.031 |
| ocr#0 | native_end_to_worker_ready | 12 | 0.063 | 0.046 | 0.050 |
| ocr#0 | native_ocr | 12 | 244.975 | 323.111 | 307.452 |
| text#0 | decode_0001 | 54 | 6.343 | 8.240 | 8.063 |
| text#0 | decode_0002 | 54 | 6.795 | 8.243 | 8.254 |
| text#0 | decode_0003 | 54 | 6.204 | 7.327 | 7.419 |
| text#0 | decode_0004 | 54 | 6.291 | 7.747 | 7.718 |
| text#0 | entry_to_first_forward | 54 | 2.846 | 2.354 | 2.215 |
| text#0 | native_end_to_worker_ready | 54 | 8.990 | 8.379 | 8.487 |
| text#0 | prefill_0000 | 54 | 109.742 | 105.706 | 102.253 |
| text#1 | decode_0001 | 30 | 6.624 | 8.168 | 8.417 |
| text#1 | decode_0002 | 30 | 7.116 | 8.183 | 8.511 |
| text#1 | decode_0003 | 30 | 8.286 | 9.449 | 9.781 |
| text#1 | decode_0004 | 30 | 7.678 | 8.611 | 8.876 |
| text#1 | entry_to_first_forward | 30 | 1.589 | 1.561 | 1.625 |
| text#1 | native_end_to_worker_ready | 30 | 11.417 | 11.199 | 11.241 |
| text#1 | prefill_0000 | 30 | 136.050 | 123.211 | 120.860 |

## Verification and files

`python -m unittest tests.test_macro_projection_contrast_training_v1 -v`: 7 tests passed. Tests cover D/S arithmetic and denominators, own unsupported with N>0 and N=0, unavailable/future BEGIN isolation, phase/shape/batch changes, real fitting of the structural contrast with joint nuisance, no-row-drop coverage and nuisance-confounded rank. The existing 14 macro clock/Hessian tests remain separate.

New sources are `macro_projection_contrast_training_v1.py` and `macro_projection_interaction_training_v1.py` under `tools/predictor_validation_20261027/`. The inventory bindings were exactly reconstructed from source/header declarations, each natural training BEGIN feature document was rederived, full raw/evidence reviews and macro extraction were rerun exactly, and review02 features were compared exactly with preserved review01 features. Review01 binds 618 receipts; review02 binds 620.

Reports: `artifacts/predictor-macro-projection-contrast-v1-natural-20261028-review01/report.json` (full feature paths and failed own-main-effect fits) and `artifacts/predictor-macro-projection-contrast-v1-natural-20261028-review02/report.json` (complete-domain interaction fits, source-bound path reference and all errors). Provider completeness and all physical/prospective qualifications remain false.
