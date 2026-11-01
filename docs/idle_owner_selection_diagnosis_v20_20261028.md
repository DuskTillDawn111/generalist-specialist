# Idle owner selection: source evidence and a trainable next step

This is a read-only diagnosis of existing CPU prediction trajectories and audited training captures. It changes neither v15/v16 nor any GPU execution policy. It does not measure prediction accuracy or physical GPU waste.

## What the existing forecast actually does

The retained 57-request **no-spill** cohort offers requests to `text#0`, `text#1` and `ocr#0`. It offers **no request at all** to resident `generalist_2#0`. All four owners nevertheless remain mark classes. In the first second only one text handler starts.

| Existing CPU horizon | Switches | Target absent from GPU frontier | Target has no active handler | G2 target despite no offered G2 request |
|---|---:|---:|---:|---:|
| 1 s | 1,985 | 993 | 993 | 993 |
| 3 s | 4,854 | 1,880 | 1,880 | 1,079 |

In these trajectories every target absent from the ready frontier is also absent from the handler-active set. This is stronger evidence than an empty frontier by itself. It does **not** mean all not-frontier targets in training or other workloads are idle.

The modeled `SERVICE_READY` envelopes of an owner without an active handler total 429.168 ms in the 1 s prefix. In the 3 s prefix they total 466.563 ms for G2, 0.578 ms for text#0 and 1,165.304 ms for text#1. These envelopes can also cover times with no other ready GPU work; they must not all be called wasted service.

For a stricter witness, the audit considers only HOLD queries with an already service-ready holder that has no handler, a nonempty ready GPU frontier, and the immediately following resource interval. The engine fixes rates and active nodes until that next event. The binary scheduler sets all those GPU rates to zero because no ready node belongs to the holder. This certifies **429.167 ms** in the 1 s prefix and **842.585 ms** in the 3 s prefix of modeled zero GPU progress despite ready work. It is a conservative subset, not the full cause of end-to-end slowdown, and not a physical measurement.

## Why training not-frontier labels must be preserved

The v13 training supervision has 35,737 observed adjacent context transitions and 715 `next_owner_is_pending=False` labels. The new audit checks those 715 at their SAVE cutoff using existing common-clock NVTX and issued-kernel observations:

- **715/715** lie inside an already open native phase of the target owner. The phase START is before cutoff and no phase END has occurred by cutoff.
- **579/715** also have a CUDA launch API that has returned by cutoff, without an observed kernel END by cutoff. This witnesses outstanding issued work, not necessarily DAG-ready work.
- **0/715** lack both witnesses.
- **28/715** preserve a context that crosses the current request's starting window.

The remaining 136 have an open native phase without that measured-kernel issue witness. They are not proven CPU-only: API calls still in progress, copies/memsets, synchronization, native host work and unclassified gaps remain possible.

There is therefore **no source evidence here for deleting the 715 labels as idle targets**. A context from the preceding job can legitimately persist until the next handler starts; adjacency across a request boundary is not automatically label contamination. The causal state must include actual handler/phase/issued-work events available at SAVE, rather than equating membership in the current job or ready frontier with all work owned by that process. However, task/pool attribution from a later request must not be retroactively supplied before that request becomes visible.

The current v14 mark uses holder, context age, previous holder and the ready frontier, while the scheduler query does not supply handler activity or host/issued-work lifecycle. Consequently an observed active target with temporarily absent ready kernels and a forecast owner with no request at all can present the same reduced feature state. Training-target retention is correct; the reduced conditioning and subsequent binary progress interpretation are the gap.

## Concrete model work supported by this diagnosis

1. **Expose and reconstruct the same causal owner state on both sides.** At every at-risk interval and SAVE cutoff record resident identity; arrived-but-unstarted owner queue; handler start/end and current visible request; open native phase; launch API in progress/returned; outstanding kernel/copy/memset/synchronization operations; dependency-ready GPU frontier. Keep `UNKNOWN` distinct from zero. Use request IDs for joins, task/pool only after request visibility, and prefix events only. Source NVTX uses the common trace clock; worker lifecycle timestamps need their mapping uncertainty carried through, with ambiguous boundary membership left unknown.
2. **Fit hazard and marks with that state.** Use the full risk intervals for SAVE timing and the retained adjacent transitions for conditional next-owner marks; do not fit timing from transition rows alone. Add handler/issued/ready state and causal ages to both models. Use an owner activity signature in support diagnostics so a trained owner pair does not hide an unseen empty-owner state. Keep all observed next-owner classes, including not-frontier targets. If an idle-owner state has no training evidence, report it as unsupported; do not conceal it with a ready-only mask, a sole-ready takeover, or fixed one/two-request admission.
3. **Separate restoration from useful progress.** Condition handoff/readiness on the chosen target's issued-work state. An open native phase does not prove a runnable GPU kernel; `RESTORE_START + 0` remains an unvalidated lower-bound hypothesis. Context envelopes do not themselves identify exclusive SM service. Preserve the binary reference-progress approximation as an explicitly testable candidate; do not promote it to a hardware fact. An improved service-rate/availability provider requires source-qualified work/progress evidence or bounded/unknown outputs.
4. **Test identifiability before another end-to-end claim.** Reconstruct source prefixes, then compare feature signatures for handler-active/not-ready versus no-handler/no-issued targets; report missing/uncertain support. Use grouped independent captures for model evaluation. On held-out requests evaluate mark probability and timing together with completion/queue predictions and ΔU sign/ranking. Include cases with no G2 request, G2 preparing on the host, issued-but-dependency-blocked G2 work, and genuinely runnable G2 work. Empty-frontier labels remain in the target population. Model and state additions must be chosen from training only.

Known full future arrivals belong to the queue forecast. They establish that G2 has no offered work in this no-spill cohort, but they do not grant the physical CUDA context scheduler knowledge of future request arrivals. State about an owner that has never arrived is distinguishable causally without exposing future timing to the mark model.

## UNKNOWN results need trajectory retention

An UNKNOWN full forecast without its unscored partial trajectory cannot establish whether idle-holder cycling caused the failure. The v19 sidecar preserves queue summary/diagnostics and the original UNKNOWN status, but does not promise full event retention on failure. If the full result is UNKNOWN and no forecast is persisted, a separate wrapper should retain unscored scheduler queries/transitions, current time, request starts/finishes, handler/issued/ready states, original exception and source receipts. Keep score eligibility false. Diagnostic preservation must not replace the failure with synthetic completions, an admission cap, or an alternative score.

## Reproduction and limits

Run `python -m tools.predictor_validation_20261027.idle_owner_selection_audit_v20 --output <new-report-path>`. The output path must be new. The report receipts both existing trajectory inputs before/after and checks the receipts of the training raw/audit/SQLite/kernel observations. It imports no predictor, worker or CUDA runtime and does not initialize a new full forecast.

Evidence: `artifacts/predictor-idle-owner-selection-v20-20261028-cpu01/report.json`. Checks passed for contiguous half-open resource intervals, exact query/event settlement, ordered modeled service envelopes, known owner identity, unchanged trajectory inputs and all 715 source labels. These checks establish audit consistency only. No validation labels, GPU execution, newly fitted model, physical accuracy result or final gain claim is included.
