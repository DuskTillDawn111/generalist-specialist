# Owner/phase two-FIFO generated continuation v5

`actual_worker_twoFIFO_reference_v5.simulate` preserves the v4 world fields and
actual worker READY release, independent parent sender FIFO, parent-processed
visibility, reader E2E completion and fixed original measurement/drain cutoff.
It uses `(owner, semantic_phase, peer_count, same_semantic_phase_count)` response
rates. The original semantic phase remains unchanged; owner differences do not
change same-phase counting. It neither infers physical service nor learns a
transport model. Frozen v4 is unchanged.

A fresh world requires the original explicit initial sender/worker/station
arguments, complete public plans, transport draws and protocol slacks. Missing
components raise `UnknownEvidence`. Its fresh protocol boundary must be declared
empty; this is a supplied-world condition, not an inferred native condition.

Every output contains `continuation_state`, schema
`gsdeploy.actual_worker_twoFIFO_explicit_continuation.v5`, with:

- exact checkpoint time, complete immutable world-input SHA and explicit CPU
  latent-world qualification;
- remaining public arrivals, sender submissions, worker starts, worker FIFO
  heads, occupied handlers and actual-release clocks;
- active request/macro index, virtual age and next mark cursor;
- processed visible phases, occurrence/issuance watermarks and persistent
  per-owner stdout producer issuance sequences;
- pending protocol reader/collector/RESULT messages and their original event
  sequence, station release clocks and next global sequence;
- cumulative starts/READY/RESULT clocks, source trace, response exposure,
  consumed public-key draws and parallelism statistics;
- original explicit initial boundary, separate from the current latent state.

Resume with the same `requests/plans/rates/protocol_slacks/transport_draws`, the
same original window/drain and `continuation_state=world['continuation_state']`.
Choose a later cutoff. Original `initial_*` arguments may be omitted; if supplied,
they must match the stored original boundary. Checkpoints round-trip through JSON
with exact rational records. A future checkpoint, partial state, changed law or
calendar, future-completed fact, unissued producer sequence, invalid active cursor
or duplicate worker reservation is rejected. Pending future *latent* notifications
issued before the checkpoint remain queue state, not observed future facts.

Arbitrary pause cuts inside response intervals are coalesced in the cumulative
exposure representation. Continuous and resumed worlds therefore compare exactly,
including the final checkpoint, trace and exact work—not merely final latency.
Queries keep their original evidence and do not end a response macro.

`resource_timeline_view` is a separate v5 adapter. It retains the v5 origin and
all qualification limits, deep-copies the world, stores each four-part rate key
in `owner_phase_rate_key`, and exposes only `(phase,n,same)` as the old interval
consumer's `evidence_key`. The actual world keeps its four-part rate key.

This implements **same-calendar explicit generated continuation**. It does not
append a new public cohort, authenticate external checkpoint payloads, infer a
native snapshot, fit latent stdin line availability/worker pre-START demand or
condition unknown residual work on native observations. A future native-history
adapter must first examine only fields available at its cutoff, retain unknown
remaining handlers/notices and return UNKNOWN where state cannot be supplied.
Final raw READY/START and future peer paths cannot fill those missing components.
An input SHA binds the supplied world; it is not evidence that a manually edited
latent payload is a true native state. The explicit checkpoint is a CPU-world
contract, not an integrity or native-authentication certificate.

Ten CPU tests cover exact continuous/split equality over active, pending sender,
reserved START, reader, collector, READY and multiple-macro/tail cuts; JSON and
multiple resumes; coupled owners with different rates but identical semantic
phase; preserved producer lifetime and pending messages; missing/future/changed
state rejection; v4 source-equivalent FIFO behavior; resource key adaptation;
and exclusion from the collector's reverse source closure. No GPU or validation
request is used. Native history assimilation and predictor qualification remain
false.
