# Module 9A: co-path scheduling pressure

This bounded experiment supersedes the proposed Roomie spatial predictor. It
does not change the preserved offline profiling checkpoint, native workers,
runtime, MPS configuration, v2, RC-SG or earlier experiment results.

The only prediction inputs are unprofiled standalone latency `T_p`, the mean
target kernel count `n_p` from three preserved offline traces, and calibrated
co-path pressure `delta_q`:

`T_hat[p <- q] = T_p + n_p * delta_q`.

For each calibration identity, aggregate all three independent-process pair
repetitions and the matched standalone repetitions first. Then take the median
of `(T_pair - T_p)/n_p` over calibration target paths for the co-path. Negative
estimates remain visible. No coefficient belongs to a target path or a pair.

| Co-path | Calibration targets | Validation targets |
|---|---|---|
| Whisper-small ASR | Qwen ASR, Qwen VQA | Qwen LID, Phi4mm VQA |
| Whisper-small LID | Qwen ASR, Qwen VQA | Qwen LID, Phi4mm VQA |
| Qwen VQA | Whisper-small ASR, Whisper-small LID | Whisper-turbo ASR, Whisper-turbo LID |

The validation target sets are disjoint from the calibration target sets for
each co-path. All repetitions of an identity belong to one split. Validation
asks whether calibrated co-path pressure transfers to previously unused target
paths for that coefficient; it does not claim that those paths were never used
anywhere in earlier development experiments.

Compare `I0` (zero pressure), `I1` (calibration residuals pooled by co-model) and
`I2` (pooled by co-path). The frozen primary score is mean absolute error of
slowdown over six equally weighted validation identities. The lowest-error
eligible model is chosen; a more complex model must have a strictly negative
upper bound of the 95% paired error-difference interval against every simpler
model. No percentage improvement threshold or feature expansion is allowed.

The separate uncertainty audit resamples independent process blocks and jointly
matched request indices, refits from calibration only, and reports co-path and
target-path cluster sensitivity. These checks cannot change the frozen primary
selection. Six identities share targets and three co-paths, so statistical
support is conditional and finite-domain, not population-wide evidence.

The unprofiled measurement harness uses unchanged native worker processes and
handlers, fixed 30 request IDs at 7-second intervals, a 210-second window, six
warmups per worker, one pure co-task and continuous backlog eight. This is a
separate frozen protocol; older 9A2 mixed co-task runs are not reused. There are
24 standalone runs, 18 calibration runs and 18 validation runs, with no retries
or outcome-based additions.

The pure evaluator computes co-model baseline demand, empirical-g utilization,
and baseline-demand-weighted co-path mixture. It scales expected co-path
pressure by the co-model busy proxy and adds contributions of different models.
Own replicas are represented only by empirical `g_m(x_m)`. Busy probability,
multiple co-model additivity and multi-replica co-pressure are extrapolations,
not validated GPU mechanisms. Unknown coefficients return zero pressure with
explicit low-confidence metadata, without excluding candidates.

Artifacts: `results/module9a_scheduling_tax/`. Do not rerun preparation or
preflight after measurement begins. The campaign can resume completed runs but
will stop at an existing failed run; it does not overwrite or retry it.

```bash
/root/miniconda3/envs/gsdeploy/bin/python -m module9a_scheduling_tax.campaign
```

Source boundary: iGniter Sec. III-A Eq. (5)-(6),
https://arxiv.org/html/2211.01713v1. The original method uses MPS. This project
borrows kernel-count scheduling-delay decomposition only and fits an empirical
temporal-sharing proxy. Neither its spatial resource model nor its speculative
round-robin scheduler description is adopted.
