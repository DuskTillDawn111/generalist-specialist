# New-schema observed sender/worker FIFO diagnostic

`twoFIFO_busy_transport_observed_v1.extract(training_review)` is a CPU-only
downstream entry. It first calls the existing new-schema macro adapter, which
requires a complete native attempt, owned execution receipt and exact review
rederivation. It then reads the authenticated raw/plan and rechecks all source
receipts. It accepts neither the historical 150-donor audit schema nor partial
successful subsets. No collector imports this module.

`summarize(raw, plan, capture_sha256=...)` is the pure CPU test interface. It
checks the complete offered cohort but explicitly does not authenticate native
evidence. Public dispatch order, with arrival/index ties, determines per-owner
predecessors across the whole attempt. The eighteen priming rows establish
history only; case boundaries do not reset history. No predecessor means unknown
initial owner state. Each case also reports its complete prior public history
and the identity/time of its actual captured snapshot when available.

Each measured row reports scheduled→enqueue, enqueue→submit/write record,
submit→write record/START, prior actual READY occurrence→START and the signed
prior READY→submit interval. Busy-at-submit means submit precedes the previous
same-owner handler's READY occurrence. Sender pressure means the previous
parent write record is still pending at enqueue. This is an observed sender
record condition, not a measurement of pipe backpressure. The three parent
stdout axes are checked independently from worker occurrence timestamps.
Parent write records may follow START or RESULT; neither parent write completion
nor READY processing is imposed as a worker START gate.

Busy rows keep latent stdin line availability, worker pre-START demand and
component decomposition unknown. Submit→START is only an observed composite.
For previous READY≤submit, the existing FIFO bounds and partial-identification
joint endpoint API are reused. These are idle-handler feasible components;
they do not establish that the worker was blocked in stdin read. No idle pair
is transferred to a busy schedule and no transport law is fitted.

The output retains the full offered denominator, per-case and per-owner counts
of busy/known/unknown predecessors and sender pressure, late-write counts,
per-request clocks and whole-attempt history. Native evidence authentication
only authenticates these observations: all predictor/provider/final-gain
qualifications remain false.

CPU checks cover busy versus sender pressure, unknown initial state, priming
history, cross-case predecessors, joint idle bounds, legal late writes, rejected
incomplete/old/validation inputs, stdout causality, mandatory native entry
rederivation and absence from the collector's source closure. The first artifact
contains synthetic fixtures only; it is not a report of native busy counts.
