# Conditional native settled origin, v5.1

The independent v5.1 engine adds `settled_initial_state=` to the v5 owner/phase
reference. Frozen v5 is unchanged. This is a restricted conditional forecast:
all preceding request handlers and their reader/collector messages must have
finished before the captured cutoff. Active native response-state inference
is still unavailable and returns UNKNOWN.

`twoFIFO_settled_history_initial_v1.from_snapshot(snapshot, cutoff_ns=...,
forecast_origin_ns=..., owner_producers=...)` inspects the captured prefix only.
It requires completed successful requests, sender writes and every START,
phase/query, RESULT and actual handler READY occurrence/reader/dequeue/processing
clock at or before cutoff; no queued runtime events/errors; complete owned PID
lifetimes; owner FIFO chronology; and no future scheduled prior requests.
`pending_request_ids` is verified as the record registry, not interpreted as
unfinished work. All roster workers need a captured processed handler READY;
missing startup station clocks are never invented.

The seed retains original past clocks, producer reader endpoints, the shared
collector endpoint, actual READY release, owner watermarks and tracked message
counts across the whole worker lifetime. Startup READY has no captured native
sequence, so counters explicitly count only tracked request messages since
capture start with a declared relative offset. They are not absolute wire
sequence numbers. All prior request labels remain outside the current cohort.

`extract(training_review, case_id=..., forecast_origin_ns=...,
owner_producers=...)` accepts only the new clean v3 training review, authenticates
all receipts by exact independent review rederivation, then reads that case's
captured prior snapshot. Historical v2 data is diagnostic only and cannot pass
this entry point. No final raw response can fill a missing cutoff field.

The engine verifies the seed's origin/producer binding, excludes prior IDs from
the current cohort, seeds its recurrence and binds initial history into the
input hash and continuation. Continuation checks rebuild new issuance relative
to those known offsets; JSON pause/resume remains exactly equal to continuous
execution. The normal owner/phase response mathematics and original measurement
and drain windows remain in effect. Explicit resource memory history must still
be supplied separately to resource prediction; no fresh-memory reset follows
from a settled protocol boundary.

CPU tests cover counter/clock continuity, extra inactive roster owner, pending
registry meaning, partial/future/missing state rejection, worker-lifetime
binding, current-cohort isolation, counter tampering, JSON continuation, source
role gates and independence from future final raw. Qualification flags remain
false: no fitted transport law, no prediction validation, no production queue
provider, no resource or final-gain qualification. Existing v2 GPU capture is
subject to the separate checkpoint-perturbation HOLD.
