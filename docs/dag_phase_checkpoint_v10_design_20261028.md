# v10 phase scheduler checkpoint integration

The independent v10 engine now implements this contract in
`dag_phase_engine_v10.py`, `dag_phase_checkpoint_v10.py`,
`dag_scheduler_state_v10.py` and `dag_frozen_model_v10.py`. It retains the runtime
v8 phase/availability event processing and adds explicit compact continuation.
The original runtime v8 engine still rejects checkpoints; the v9 busy engine
still uses its separate diagnostic rates. Neither engine is silently relabeled
as v10.

## Frozen contract

Keep the complete original public cohort, resident aliases, plans, original
measurement/deadline/drain, event budget, provider sources and six equal utility
weights. Reuse v9 source, host START/FIFO, explicit worker READY, lifecycle and
resource-history validation. Bind the semantic phase provider and the scheduler
class implementation plus immutable policy as additional provider identities.
Dynamic scheduler state must be excluded from immutable provider identity.

Large immutable plans and numerical provider tables should be verified and
content-addressed once at initialization, then recursively frozen. A checkpoint
binds those model/source receipts and their static digests, rather than encoding
the complete same plans for every scenario. The engine must still reference the
same frozen model, and changing a provider or its source must remain detectable.
Do not cache a digest while leaving the underlying table mutable.

Running checkpoint state can store original public row, START and completed
node IDs. Original nodes, successors and remaining incoming dependencies are
rehydrated from the immutable original DAG and then undergo structural checks.
Keep active remaining work, arrival/ready/lifecycle heaps, resource history,
accumulators and consumed event count. No checkpoint can remove future requests
or reset the event budget. Initialization and cutoff/resume CPU costs must be
reported separately; the current v9 multi-minute prefix is not online-ready.

## Dynamic scheduler state

Preserve current_owner, mode, absolute grant_end_ns, absolute handoff_end_ns,
last_clock_ns and the complete transition history. A restored handoff or grant
uses its original absolute deadline. A host event, copy completion, pause or
resume never restarts it. Same-clock closure must match the runtime v8 engine,
including zero-work nodes and back-to-back owner requests.

Structural field checks alone cannot establish a state under a policy. The
implementation must additionally rederive decisions against the same frozen
policy, with enough original clock/frontier history to reproduce transitions.
A deterministic complete prefix replay to the checkpoint event count is a
stronger alternative: compare the entire DAG and scheduler state, preserving
the original absolute event budget. This verification is predictor CPU work,
not simulated GPU progress or a second measurement run.

## Required counterexamples

- An uninterrupted complete forecast equals pause/resume at every atomic
  boundary, including an incomplete handoff and partially consumed grant.
- Pause during an uncontended interval followed by peer return retains the
  previous finite grant remainder.
- Back-to-back same-owner requests settle at one clock without an artificial
  peer disappearance or new full grant.
- Changing policy, semantic phase mapping, solo/copy work, host supply,
  lifecycle, public arrivals/routes, event budget or original window is rejected.
- Changing current owner, mode, absolute grant, handoff, last clock or history,
  even while re-signing a checkpoint JSON, is rejected by state rederivation.
- Unsupported policy frontier remains UNKNOWN; no equal-context fallback.
- Exact supported visible START can anchor host supply, but cannot create a
  CUDA cursor or bypass preceding predicted worker READY.
- Reader PHASE beyond predicted handler END rejects that joint scenario; no
  clipping or whole-request END stretching.

## Current evidence and limits

`predictor-phase-checkpoint-v10-20261028-final03/report.json` records 95 passing
CPU regression cases and unchanged source receipts. Complete numeric fixtures
match runtime v8, and every actual pause boundary resumes the uninterrupted
trajectory. Re-signed scheduler-state changes fail deterministic prefix
verification. Recursive freezing includes deletion guards, following an
independent counterexample against the earlier final01 source snapshot. The old
artifact remains unchanged. Original graph compilation retains every node ID,
kind and dependency, while a content digest binds unused original metadata.

`dag_visible_busy_v10.py` explicitly selects the independent v10 engine and
retains conditional exact START, reader visibility, the full original cohort and
original deadlines. Unsupported states remain UNKNOWN. Source-derived
`native_queue_phase_busy_v10.py` initializes true two-solo work, preparation-aware
host supply and original semantic phases once, then shares immutable numerical
tables between scenarios. Missing modality templates are retained in the cohort
and listed explicitly. No resource provider is yet bound, so missing E/M axes
cannot produce full utility.

The actual training-source attempt
`predictor-native-queue-phase-busy-v10-20261028-training01/report.json` completed
availability-provider initialization in 245.31 seconds, then failed a software
context-identity check before executing a DAG forecast or risk diagnostic. It
does not measure physical prediction error. The factory now uses the same
canonical native-context identity as the collector and checks it before the
expensive initialization. Its focused regression verifies both acceptance of the
canonical identity and early rejection of a genuinely different context.

The fresh additive
`predictor-native-queue-phase-busy-v10-20261028-training02/report.json` completed
the actual full-cohort path and attached the real risk API at the same first
predeclared 50 ms cutoff. Cold initialization took 266.407 s (including the
245.117 s availability-provider reconstruction), original graph compilation
7.445 s, actual DAG conditioning 18.617 s wall / 18.352 s CPU, separate risk work
reconstruction 47.179 s, and risk inference 0.040 s. Total wall time was
332.248 s; maximum RSS was 6,248,684 KiB. The 23 source receipts remained
unchanged. These costs do not establish online readiness.

The main result is UNKNOWN with zero surviving candidates and all 24 original
requests retained. Two whole-run scenarios encounter an unsupported
`text#1` contention entry with G2 and `text#1` both in
`outside_language_forwards`; two others violate supported visible START lag for
the corresponding input occurrence. Four future G2 image/ASR templates are also
explicitly missing. No equal-context fallback or partial-cohort result is used.
The attached risk score is an uncalibrated training diagnostic (0.4771563423),
not a probability or a replacement for the main result. The runner asserts that
all main-result fields remain exactly unchanged after attaching it. Old failures
and interrupted v9 runs remain immutable.

The small synthetic checkpoint benchmark in final01 concerns CPU fixtures only.
Initialization, actual prefix/forecast and risk costs must be measured separately
on the real source-bound model. None of the continuation tests or training
replays establishes independent slowdown/P95/Delta-U accuracy, calibrated risk,
online readiness, or deployment benefit.
