# Causal lifecycle mark and full-risk SAVE predictor

The new CPU candidates distinguish **a resident process, a handler with owned work, an issued operation, and a ready GPU node**. This fixes a missing conditioning variable; it does not establish physical GPU scheduling accuracy or final spillover benefit. The frozen v15/v16 implementation remains unchanged.

## Evidence that determined the change

The original 1 s CPU no-spill trajectory restores G2 993 times even though its complete 57-request cohort contains no G2 request. Under the existing binary reference-work interpretation, immediately following HOLD-query intervals certify 429.167 ms with ready text work and zero modeled GPU progress. This is model behavior, not measured physical waste.

Conversely, all 715 training next-owner labels outside the GPU-ready frontier have an open native phase at SAVE; 579 also have a returned launch API without an observed kernel END yet. Deleting those labels or forcing the sole ready owner would destroy legitimate supervision. The new mark retains all 35,737 transitions and all 715 labels.

## Implemented state and engine boundary

`OwnerLifecycle(handler_active, issued_outstanding, gpu_ready)` represents each fact as true/false/unknown. Training and inference use the same type, feature encoder, and active-work definition. Any certified owned work can make an owner eligible; a temporarily empty GPU frontier does not exclude a host-active handler. All three facts false identify an idle reference-service target. Unknown facts remain unknown.

The independent `owner_lifecycle_engine_v21.py` exposes the settled event state before scheduling:

- `owner_lifecycle`: all resident owners and their immutable lifecycle facts.
- `handler_requests`: current owner/request/start identities.
- `pending_requests`: arrived owner FIFO request identities, with no future arrivals exposed.

The numerical DAG loop follows the v10 source; CPU small-graph comparisons preserve starts, finishes, energy, memory, queues and work consumption. Copies/memsets appear as issued activity. A handler with no currently released activity has unknown issued state: the light DAG does not identify physical issued-but-dependency-blocked CUDA operations. A running handler still proves owned work. Unreleased DAG nodes are never relabeled as issued.

Source handler events use their fixed median clock projection. A future handler END does not alter the prefix before that END occurs in the projected clock. Exact common-clock native phase ranges add handler-active evidence. The original mapping residuals remain an uncalibrated source-domain uncertainty; these projected state facts are not exact physical common-clock observations. Older attempts that formed `END - residual` boundaries are retained as historical diagnostics and are not the recommended causal models.

The engine hashes supplied source receipts directly before and after a forecast, including exceptions. A caller must supply the full provider/source closure. Same-size file changes with restored timestamps are detected. The last immutable lifecycle view can be exported as an observation state; it is explicitly **not** an executable checkpoint. Old v10 checkpoints and engine reuse are rejected.

## Mark model and its limits

Recommended model: `artifacts/predictor-owner-lifecycle-mark-v21-20261028-training03/model.json`.

The model adds handler/issued/ready true/unknown indicators to the existing holder, age, current frontier and previous-holder features. No future arrival count, scheduled offset, future kernel START, or completion time is a feature. Two whole-capture training holdouts produce physical next-context accuracies 99.949% and 99.939%; the nonfrontier subset is correct 357/364 and 340/351. A separate active-reference-service output is correct 357/364 and 342/351 on those subsets. These are correlated training-capture diagnostics from only two captures sharing job/input templates; they are not independent request validation or calibrated probabilities.

Every physical class probability is preserved. The separate active-reference-service distribution excludes only owners whose three facts are all false, after checking ready-state/frontier consistency. This is not a ready-only mask. Using that separate distribution as a physical next-context target requires an explicit, unvalidated hypothesis. Initial residual contexts and background contexts are independent missing models; they are not invented by the active-service mark.

`LifecycleActiveServiceMark` and `LifecycleSchedulerBridge` provide the experimental controller connection. A SAVE with only the current holder eligible can still predict same-holder reacquisition, which the existing controller rejects. The bridge does not fabricate a hold, erase the SAVE, waive handoff cost, or renew the context. A lifecycle-conditioned hazard and a justified physical transition model must address that remaining failure.

## Reconstructed full-risk SAVE supervision

The hazard target is **physical `SAVE_END`**, with attained age measured from the original **`RESTORE_START`**. It is neither SAVE_START nor SM execution time. Source context envelopes are clipped to actual job windows while preserving original restore age and right censoring. The extractor splits risk at actual lifecycle and ready-frontier events; only the left-end event prefix supplies features. Future state-change deadlines are not prediction features or scheduler deadlines.

Recommended model and receipts: `artifacts/predictor-owner-lifecycle-hazard-v24-20261028-training03/`.

- 559,130 risk intervals cover 22.383203487 s of qualified context exposure.
- All 35,737 original marked SAVE identities remain, and 62 additional qualified physical SAVE endpoints are preserved. Those 62 lack a qualified next-owner label because an unpaired context event occurs before the next RESTORE; they train the physical hazard, never a fabricated mark.
- 225,559 empty-frontier intervals cover 4.939996014 s and contain 133 physical SAVE endpoints. The old hazard omitted those 133; its 35,666 events plus 133 equal 35,799 new physical endpoints. The apparent 71-event discrepancy between old hazard and mark counts combined these two distinct populations.
- 8.849674422 s have no certified active alternative and contain zero observed physical SAVE endpoints. Zero observations do not establish zero population hazard.

The compressed `risk-chronology.jsonl.gz` retains each interval and feature vector; `physical-SAVE-event-ledger.json` identifies every endpoint and why either older target population lacked it. `job_chronology` separately records paired exposure, qualified risk and context gaps. Unpaired envelopes are never filled in. Source receipts before/after, exact interval/event conservation and original event identity coverage are checked.

## Hazard fitting and honest diagnostics

The first full-risk fit retained the old capture/envelope-weighted likelihood with L2=1. Its global event total looked plausible, but its fit predicted approximately **7,891 SAVE events in a zero-event no-alternative stratum**. That fit is retained as a failed conditional predictor under `predictor-owner-lifecycle-hazard-v21-20261028-training02`; exact implementation and mark-source snapshots with matching historical SHA are saved there. It must not be integrated.

The final v24 fit explicitly uses the full exposure Poisson likelihood, normalized by each capture's event count, with nonintercept L2/N and predeclared holder/competition indicators. This makes regularization scaling consistent with the mark's normalized objective. It does not set no-competition rate to zero. Trust-exact optimization with an analytic Hessian converges for both capture folds and the full fit in 12 iterations; the largest final gradient is below 1e-7. Earlier nonconverged or source-mutated attempts remain separate and are not recommended.

| Held-out training capture | Total predicted / observed SAVE | Empty-frontier predicted / observed | No-alternative predicted / observed |
|---|---:|---:|---:|
| Capture 2 | 17,419.38 / 17,694 | 57.24 / 71 | 1.905 / 0 |
| Capture 1 | 19,300.08 / 18,105 | 84.81 / 62 | 2.211 / 0 |

The full training fit predicts 2.272 events over the 8.850 s no-alternative exposure. This removes the earlier extreme idle-state mismatch while retaining a finite rate and an explicit zero-observation warning. The table also shows remaining errors: empty-frontier event counts are off by roughly −19% and +37%, and total events in one capture are overpredicted by about 6.6%. Global totals alone do not establish conditional calibration, event-time accuracy, queue accuracy or spillover ranking.

## Stable inference interface and next integration

Use the validated `owner_lifecycle_hazard_query_v24.py::LifecycleSaveHazard`, not the unchecked fit implementation boundary:

```python
candidate.query(
    held_owner, pending_frontier, owner_lifecycle,
    context_age_ns, horizon_ns=0,
)
```

`owner_lifecycle` is a complete dictionary of resident `OwnerLifecycle` values. The query rejects inconsistent ready-frontier state. It returns `hazard_per_ms`, `OOD_reasons`, restricted CDF and support information. The rate is constant inside each fixed `EDGES_NS` age bin while lifecycle/frontier remain unchanged. Empty-frontier rates are modeled. Pooled age support is not category-conditional support; unseen categories and no-alternative zero-event support are explicit warnings.

A cumulative controller must integrate the old lifecycle and old frontier up to each current observation, then apply the new state. Lifecycle changes must trigger updates even if the GPU frontier is unchanged. The original single restore budget, integer crossing and compensated integration are reusable. Empty-frontier pause semantics are not appropriate for this fitted model. Physical SAVE timing remains separate from active-service replacement, residual context, background activity, handoff readiness and useful work rate. No kernel permissions or GPU experiments are required for these CPU changes.

CPU tests cover numerical DAG preservation, delayed requests and FIFO state, host-active empty frontiers, no future owner activation, source mutation guards, future-END prefix invariance, inconsistent ready-state rejection, nonzero empty-frontier hazard and distinct physical/service mark probabilities. GPU profiling, independent request prediction, ΔU sign/rank validation, deployment search and final six-term utility gains remain unclaimed.
