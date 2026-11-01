# OCR variant: independent review of training readiness

Recommendation: finish and freeze the selected full fitter, but collect targeted training evidence before immediately launching another independent four-route GPU suite if the intended result is to pass the unchanged predictor gate. An independent diagnostic run would be legitimate after freezing; the present training evidence does not make a successful gate result likely enough to treat it as the next highest-value experiment. This review uses only the predeclared duration-v13 selection and its whole-seed training-CV reports, not any v11 runtime outcome or error.

The fixed lexicographic rule correctly selects `ocr_handler`: all three candidates have zero ΔU sign/rank errors; its temporal failures are 7 versus baseline 8 and phase_ridge1 9. Its normalized temporal error mean is .61870 versus baseline .66667, a 7.2% decrease. Phase_ridge1 has a slightly smaller mean .61685 but loses on the prospectively higher-priority failure count. The selected model is therefore the declared development winner, while `original_training_cv_thresholds_passed=false` remains correct.

Every flow has text=167, image=5, ASR=5, OCR=4; all 1,448 request predictions are available. All eight aggregate service and E2E checks now pass. The remaining seven failures are six task P95 checks and one queue check:

| Held-out seed | Route | Check | Predicted / observed | Error |
| --- | --- | --- | --- | --- |
| 627401 | balanced | OCR P95, n=4 | 654.5 / 860.1 ms | −205.6 ms, 23.90% |
| 627401 | skewed | image P95, n=5 | 536.9 / 691.9 ms | −155.0 ms, 22.40% |
| 627401 | spill_high | image P95, n=5 | 1998.1 / 2754.8 ms | −756.7 ms, 27.47% |
| 627401 | spill_low | ASR P95, n=5 | 1540.5 / 3723.8 ms | −2183.3 ms, 58.63% |
| 627401 | spill_low | OCR P95, n=4 | 683.4 / 1026.0 ms | −342.6 ms, 33.39% |
| 627402 | skewed | queue MAE | 1.594 requests | Limit 1 request |
| 627402 | spill_low | image P95, n=5 | 530.4 / 664.9 ms | −134.5 ms, 20.22% |

The OCR simplification removes three baseline failures (627401 high OCR, 627402 balanced OCR and 627402 skewed OCR) but creates two (627401 balanced and low OCR). Thus it improves complexity and average error without uniformly resolving OCR tails. Six P95 failures all underpredict, but that is not sufficient evidence for a global slowdown adjustment: low ASR previously showed a preceding G2 text queue-path error, and service/wait errors can cancel. The low-image 20.22% case is close to the fixed threshold; the ASR 58.63% and queue 1.594 cases are not. No route passes every temporal check in both training seeds.

These small-task P95 values put .8 weight on the maximum of five observations or .85 on the maximum of four under the fixed interpolation convention. They remain valid gate values, but do not identify a stable population tail from four/five samples. There are only two held-out seed blocks, and the same blocks have now supported several development comparisons. Perfect training ΔU ranking is encouraging and remains the primary declared selection criterion; it is not independent confirmation or a calibrated assurance that the selected model's ranking will transfer.

The next training protocol should target the unresolved mechanisms, not enlarge a parameter grid against these same eight flows: predeclare additional independent training seeds/arrival offsets, include repeated identical input/arrival traces to distinguish run variation from input/context changes, and obtain enough low-rate image/ASR/OCR occurrences to cover the failing exact phase mixtures and G2 predecessor queue paths. Keep H90 work distinct from H600 qualification. CPU forensics can first separate actual-context conditional residual, phase fallback, and fresh-simulation predecessor finish drift. None requires modifying the fixed thresholds or reading an independent run to choose a correction.

After that evidence, freeze one small mechanism-driven alternative set, repeat grouped training selection, fit all declared training sources once and spend the next independent GPU suite on that frozen candidate. If the project instead prioritizes obtaining the first independent measurement of this simpler OCR model now, label the run diagnostic and retain all seven training failures in its report; do not present selection as a gate PASS or final diversion gain.
