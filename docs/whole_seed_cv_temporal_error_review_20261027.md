# Independent review of twelve temporal errors in whole-seed training CV

This is retrospective training development analysis, not independent predictor confirmation. It reads the audited old-loss whole-seed CV report, its declared two-seed training sources and cross-fitted work diagnostics only. No v9 data, new GPU collection, threshold change or predictor edit is involved. The report source is `artifacts/predictor-gpu-20261027/background-statistical/continuous-v11/seed-crossfit-audited/audited-report.json`.

All 1,448 offered requests have predictions; each seed's four-candidate block has correct ΔU signs and all pair rankings. The twelve failures among 56 temporal checks comprise seven task P95 checks, one service MARE check, two E2E MARE checks and two queue MAE checks. Every run contains text=167, image=5, ASR=5 and OCR=4 requests. The 167 text requests dominate an all-request average; passing aggregate service MARE does not imply that image/OCR service is accurate.

## Exact failures and absolute scales

| Held-out seed | Candidate | Failed check | Predicted / observed or error | Absolute interpretation |
| --- | --- | --- | --- | --- |
| 627401 | skewed | queue MAE | 1.238 requests | Text wait bias +930.8 ms/request |
| 627402 | skewed | queue MAE | 2.277 requests | Text wait bias +1660.3 ms/request |
| 627402 | skewed | text P95 | 18807.6 / 15060.3 ms, 24.88% | +3747.3 ms |
| 627401 | skewed | image P95, n=5 | 441.6 / 691.9 ms, 36.17% | −250.3 ms |
| 627401 | spill_high | service MARE | 22.46% | Service MAE 168.4 ms; bias +32.2 ms |
| 627401 | spill_high | E2E MARE | 20.73% | E2E MAE 274.9 ms; bias +145.1 ms |
| 627401 | spill_high | image P95, n=5 | 3467.2 / 2754.8 ms, 25.86% | +712.4 ms |
| 627401 | spill_high | OCR P95, n=4 | 840.4 / 628.5 ms, 33.73% | +212.0 ms |
| 627401 | spill_low | E2E MARE | 21.01% | E2E MAE 335.9 ms; bias +189.5 ms |
| 627401 | spill_low | image P95, n=5 | 2459.9 / 1890.0 ms, 30.15% | +569.8 ms |
| 627401 | spill_low | ASR P95, n=5 | 1746.3 / 3723.8 ms, 53.10% | −1977.5 ms |
| 627402 | balanced | OCR P95, n=4 | 2165.4 / 1366.4 ms, 58.47% | +798.9 ms |

The unchanged limits are service/E2E MARE ≤.2, every-task P95 ARE ≤.2, queue MAE ≤1 request. The purpose of the absolute values is to explain the failures, not replace these gates.

## Reproducible text queue bias in skewed routing

Skewed routing shows the strongest directional evidence. Cross-fitted text worker-service mean bias is +18.8 ms for seed 627401 and +28.7 ms for seed 627402, while text service MARE is only 8.36%/9.19%. Mean wait bias nevertheless reaches +930.8/+1660.3 ms, text E2E bias +948.7/+1688.1 ms, and text P95 bias +1839.8/+3747.3 ms. Both queue gates fail. A small persistent service excess is being accumulated by the heavily used alias; this is consistent with the numerical mean bias of the current work objective and nonlinear FIFO amplification. It is not proof that all of the service excess has that single cause.

Independent work reconstruction using the actual held-out phase contexts also has mean decode work below one in both seeds (.9761/.9625) and finalize work .5762/.5920, with prefill near one (1.0024/.9976). Work below one means the model integrates less than one unit over actual execution and would need more time along that recorded context path. This supports a conditional service mismatch independently of the simulator's predicted phase alignment. These are mixed-wall residuals; they do not identify a physical GPU speed or a causal attribution.

The corrected duration objective is a justified next training-only comparison. Evaluate whether it reduces both held-out seeds' directional text bias and cumulative skewed queue error, without harming the predeclared ΔU sign/rank objective. An across-the-board offset tuned from these errors would confound the estimator with condition-specific calibration and should not be introduced silently.

## Small-task tails expose particular requests and queue paths

For the implemented linear-interpolated P95, n=5 gives `.2*fourth + .8*maximum`; n=4 gives `.15*third + .85*maximum`. Thus each low-rate task's reported P95 is mostly one request. These gates still apply, but four/five requests do not estimate a population 95th percentile precisely.

**Skewed image, seed 627401.** The request at 3.8 s has worker service 431.5 ms predicted versus 730.1 ms observed, with admission/FIFO wait approximately 0/.6 ms. Its E2E is 431.5/731.3 ms. This is chiefly conditional service/phase prediction, not queue transport. Actual-context reconstruction has prepare work 1.7285, prefill 1.2752 and decode 2.2003; approximately 410.5 ms of the 730.1 ms actual service uses the model's explicitly labeled phase-context interpolation. This gives a concrete coverage/conditional-model direction: inspect the handler fallback for the supported full task mixture and whether the exact phase combination has sufficient training support. Missing exact contexts must not become falsely measured support. Seed 627402 image P95 error is only +8.9 ms, so one universal image correction is not supported.

**Spill_low ASR, seed 627401.** At 69.4 s, ASR request `measurement:139` has predicted/actual worker service 1498.5/1411.6 ms, but predicted/actual wait 313.7/2907.6 ms, and E2E 1812.2/4320.4 ms. Its own service is reasonably close while waiting is missed by 2593.8 ms. The actual blocking G2 text request `measurement:132`, offered at 65.9459 s, starts at 65.9466 s and finishes at 72.3069 s. Therefore increasing ASR service to fit its P95 would repair the wrong component. Trace the preceding G2 text duration and phase alignment through the same shared alias. The ASR actual-context work is prepare 1.7948, prefill .9973 and decode .8331, which also rules out a simple uniform ASR slowdown interpretation.

**Spill image tails, seed 627401.** In spill_high, image at 63.8 s has worker service 471.2/571.4 ms, wait 3558.4/2459.2 ms, and E2E 4029.6/3031.2 ms. The +1099.2 ms wait excess dominates the top-tail error. In spill_low, images at 63.8/83.8 s have wait errors +645.1/+834.9 ms while their worker services are *underpredicted* by 300.4/208.0 ms. Opposite service and queue errors can partly cancel. A scalar image-service correction cannot simultaneously explain these components.

**Balanced OCR, seed 627402.** The largest contribution is `measurement:110` at 54.6 s: worker service 2340.3/1407.0 ms and E2E 2340.3/1408.2 ms. Wait is 0/.54 ms. The other three OCR service errors are +128.0, +45.1 and +37.4 ms; all four are positive, but one error of +933.3 ms accounts for most of the tail failure. Actual-context OCR work is .5612, entirely direct phase-context coverage, so this cannot be dismissed as unseen-phase fallback or parent response delay. Candidate/time-specific phase/context fit and the size-biased objective are plausible contributors requiring training-CV comparison.

**Spill_high OCR, seed 627401.** All four worker predictions exceed observations, with service mean bias +169.6 ms and P95 excess +212.0 ms. But the same route in seed 627402 has OCR service mean bias −226.2 ms, its individual final request underpredicted by 767.0 ms, and a P95 error of −203.9 ms (14.30%, which passes). OCR therefore lacks a stable global direction across these two seed/route conditions. Passing P95 can coexist with substantial per-request error because predicted and actual maximum identities differ.

The all-request service gate is particularly permissive about these low-rate tasks by construction. For example seed 627402 spill_high aggregate service MARE is 10.49%, while OCR-only service MARE is 37.37% over four requests. These are both correct statistics on different populations; neither should be relabeled as the other.

## What can and cannot be inferred about randomness

There is evidence of recurring conditional bias and queue amplification, especially skewed text, and evidence of errors dominated by a particular shared-alias queue history or a small-task maximum. There is not enough evidence to partition the remainder into inherent run randomness versus omitted context, model misspecification, phase alignment, initialization/clock effects or statistical fitting variance. The two seeds have different input sequences and different cross-fitted models; there is no repeated identical arrival/input trace under the same frozen model. Seed variation is therefore not a clean repeated-run noise estimate. The fact that a failure disappears in the other seed does not prove it is irreducible noise.

The next supported action is the already declared old-loss versus duration-WLS whole-seed training-CV comparison, holding all contexts/bounds/priors/gates and decision-selection rule fixed. Report absolute duration mean bias/MSE plus existing full-cohort service/E2E/queue/P95 and ΔU endpoints. If that comparison does not resolve the path-specific errors, CPU forensics can separately quantify: actual-context conditional work mismatch; fraction of direct versus interpolated service; and fresh-simulation start/finish drift of each shared-alias predecessor. These diagnose where a new model term could be justified without feeding observed future phase cuts into prediction.

A future data protocol can distinguish run effects only by predeclaring repeated identical traces and additional training seeds/offset patterns, then testing a source-bound run/block residual model or richer supported phase contexts on independent data. Neither broadening support by assumption nor injecting arbitrary IID noise is established by these twelve errors. H90 and these small cohorts do not qualify a final H600 capacity or P95 claim.
