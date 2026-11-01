# Runtime admission event contract v2

This change records the actual host admission lifecycle. It does not change strict FIFO, per-alias `max_inflight`, group capacities, request-wide nonpreemption, routing, or the shared execution mode. No global capacity of one or two is assumed or introduced. Historical experiment artifacts remain unchanged.

## Authoritative gate events

`RequestAdmission.snapshot(since_sequence=0)` atomically captures gate state and the transition stream under the admission condition lock. The result retains `active`, `waiting`, `closed`, and `peak_active`, and adds:

| Field | Meaning |
|---|---|
| `event_contract` | `gsdeploy.admission_events.v2` |
| `configuration_sha256` | Identity of the validated resource configuration |
| `clock` | `host_perf_counter_ns` |
| `observed_time_ns` | Host observation timestamp sampled while holding the gate lock |
| `last_sequence` | Last global transition sequence, initially zero |
| `events_after_sequence` | Consumer cursor supplied to this snapshot |
| `events` | Events strictly after that cursor |

Each event contains `sequence`, `kind`, `time_ns`, `alias`, `request_id`, `reason`, `active_count_after`, and `waiting_count_after`. Sequences start at one and define a total order even when timestamps tie. The `perf_counter_ns` clock is monotonic; these values must not be compared to Unix timestamps or unaligned device timestamps. Resource identity and event contract must accompany a trace used for training or evaluation.

The event kinds are:

- `enqueue`: sender appended the ticket to the admission FIFO. This is distinct from the earlier per-model host send-queue enqueue.
- `grant`: the head ticket was removed and its permit entered `active`.
- `release`: an owned permit was removed from `active`. Duplicate release attempts produce no fictitious transition.
- `cancel`: a waiting ticket was removed without grant, with `gate_closed` or `acquire_exception` reason.
- `reject`: acquire was attempted after gate closure, without entering the FIFO.
- `close`: gate was closed to new grants; existing active permits are retained until their actual release.

`acquire()` still returns a boolean. `release(request_id, reason="caller_release")` returns the recorded release event or `None`; `release_alias(alias, reason="alias_release")` returns actual release events. `request_events(request_id)` returns copies of that request's event history. Events are recorded inside the condition critical section, without invoking callbacks or acquiring the runtime record lock. State mutation and its observation therefore cannot be interleaved with another gate transition.

The event history is retained for the lifetime of the runtime. Rolling consumers should use `snapshot(since_sequence=last_sequence)` to avoid repeatedly copying the full history. This is an observation interface, not a bounded-memory archival service; long-running deployments will need an explicit retention/consumer-acknowledgement design before pruning history.

## Runtime request observations

`DeploymentRuntime` copies exact gate transitions into each record's `admission_events`. The existing `admission_wait_start_time_ns` and `admission_granted_time_ns` now reference the authoritative gate enqueue/grant timestamps; previously these were sampled outside the gate's critical section. New fields are `admission_released_time_ns`, `admission_release_reason`, and, when relevant, `admission_cancelled_time_ns`.

Release reasons distinguish `worker_result`, `worker_error`, `late_worker_result`, `late_worker_error`, `worker_exit`, `send_failure`, and `submission_cancelled`. A worker RESULT reporting failed inference still releases on the RESULT protocol event and subsequently receives its existing error classification. Drain timeout does not itself release an active permit. Shutdown closes the gate but does not pretend that active requests have finished.

Response observations separate:

1. `response_reader_event_enqueue_time_ns`: transport has parsed a protocol line and is about to enqueue the host event. This equals the existing `completion_time_ns` for an on-time response; its original score semantics are preserved. It is not raw first-byte receipt time.
2. `response_collector_dequeued_time_ns`: collector returned from the event queue, before obtaining the runtime record lock.
3. `response_collector_processing_time_ns`: collector acquired the runtime lock and began processing the event.
4. `admission_released_time_ns`: actual permit removal in the admission critical section.

START/PHASE `execution_events` also receive collector dequeue/processing observations. Their existing worker timestamps and reader `received_time_ns` retain their meanings. Worker handler END, kernel completion, worker next-request readiness, JSON serialization, and raw pipe transit are distinct events; this instrumentation does not claim to observe each boundary.

`run_worker` and the active native `observed_worker.py` / `semantic_worker.py` now emit `WORKER_READY` after RESULT/ERROR JSON serialization and stdout flush have returned, before the next input read. It includes the previous `request_id`, actual worker `pid`, `perf_ns`, `clock=host_perf_counter_ns`, `boundary=previous_handler_next_request_ready`, and `scope=after_response_flush_before_next_read`. This is an explicit readiness announcement, not the next request's START and not evidence that a future request has already arrived. The announcement's own serialization/flush adds instrumentation overhead after the sampled boundary; the observation is not an exact timestamp of the subsequent read syscall.

The transport recognizes this protocol kind. The collector validates PID, request identity, clock and chronology, then stores `previous_handler_next_request_ready` separately from `execution_events`; this avoids treating readiness as a handler PHASE or kernel anchor. Trace export deliberately ignores it for phase pairing. There is no readiness event for shutdown or for malformed input without a request id. A flushed ERROR for an identified request can have a readiness event even though that request is not eligible for the completed-request lifecycle fit. Existing frozen worker copies and historical artifacts are not rewritten; a newly collected native execution context must bind the changed worker source identities.

## Predictor integration

Capture a coupled runtime-record and gate snapshot while holding `runtime.lock`, with the gate snapshot obtained inside that lock. Gate methods never call back into the runtime lock. The snapshot's actual FIFO order replaces alias-sorted simultaneous arrivals for already queued tickets. For future arrivals, the runtime sender/queue policy must be modeled explicitly rather than claiming a known global enqueue order from scheduled arrivals alone.

The simulator must distinguish handler end, parent response observation, permit release, and worker readiness. A request can finish the parent-receipt latency cohort while still owning a gate permit. Preserve that permit in the initial handoff state until its release transition; do not pop it merely because the handler has ended or the response has been observed. Score deadline/T and full-cohort latency from their original boundaries. Handoff delay models require training-bound observed data and explicit missing-evidence treatment; these events do not establish a universal fixed wait constant.

An exact historical gate state can be replayed using enqueue/grant/release/cancel/reject/close transitions and sequence order. The event stream explains gate ordering but does not by itself provide true GPU SM/DRAM/L2 demand or prove that host admission improves GPU execution.

## Lifecycle training interface

`gsdeploy.forecasting.lifecycle.LifecycleProvider(document, execution_context_sha256, allow_cpu_fixture=False)` exposes `delays(row, aliases, scenario)`. The returned `parent_receipt_ms`, `permit_release_ms`, and `worker_next_ready_ms` are separate delays relative to handler END. It also returns `known`, `permit_scope`, `evidence`, and `scope`. The provider exposes `physical_evidence` and `execution_context_sha256` for simulator identity checks. Exact family/task/pool identity and explicit scenario are required; missing inputs/scenarios raise `UnknownEvidence`. Missing boundaries remain `None`. For `permit_scope=not_applicable` there is no gate release value to infer, and `known` requires only receipt and readiness; request-admission profiles additionally require gate release. For an owned permit, release cannot precede parent receipt. Worker readiness can precede parent receipt because pipe/collector latency is separate.

`export_training_observations(raw, aliases, execution_context_sha256=..., clock_contract=..., source=..., worker_ready_events=None)` requires an explicitly training-only wrapper, completed training records, kind `measured` or `cpu_fixture`, the matching execution context, and a boolean `admission_enabled`. The clock contract explicitly verifies common host perf-counter origins for handler/runtime/readiness; it carries its verification source. The exporter rejects device/Unix clock substitution. It reads handler END from `worker_response.end_perf_ns`, receipt from `response_reader_event_enqueue_time_ns` (cross-checking `completion_time_ns`), releases from a fully replayed v2 gate event stream, and readiness from the explicit record field above. Optional externally sourced readiness events must declare the same boundary and clock and agree with record observations. It does not treat legacy record timestamps as authoritative gate transitions or infer worker readiness from next START.

`fit_lifecycle(observations, execution_context_sha256, minimum_observations=2)` fits explicitly input-conditioned **mean** vectors and preserves each original training joint delay vector as `joint_observations=[{id, delays}]`. It validates training split, identity, source kind, duplicate ids, and consistency of delay values with observed timestamps. If any observation lacks a boundary, that fitted mean boundary remains unknown; missing events are not imputed as zero or excluded to obtain a seemingly complete fit. This is a mean handoff model under the observed execution context. It does not claim a joint P95, load-independent delay, deterministic bound, or delayed-overload recovery model. Additional contexts/scenarios require actual training evidence and an explicitly extended conditioning contract. CPU fixtures are accepted only with the opt-in and never become physical evidence.

Initial state needs a separate contract: a request may already have passed handler END while parent receipt, gate release, or worker readiness remains unresolved. A future forecast must preserve all four observed facts independently, along with the original handler-END time and applicable sequence cursor. Applying an unconditional full handoff duration again from `now`, or subtracting elapsed time from a mean as though it were a conditional residual model, is unsupported. Legacy simulations that collapse all handoff boundaries must identify that assumption explicitly instead of being presented as the new contract.

`LifecycleProvider.residual_handoff(row, aliases, scenario, now_ns, observed, boundary_observation_status=...)` now supports an empirical joint **mean** residual for explicitly confirmed boundary states. `observed` contains exactly `handler_end_ns`, `parent_receipt_ns`, `permit_release_ns`, and `worker_next_ready_ns`. The status dictionary uses the three boundary timestamp keys, each `observed`, `confirmed_pending`, or `unknown`; shared execution uses `permit_release_ns=not_applicable`. Every known timestamp must be between handler END and `now`, and an observed gate release must follow an observed parent receipt. Missing or unknown status returns UNKNOWN. In particular, a WORKER_READY event not yet received does not prove that the worker is not ready; its status must remain unknown unless an independent coherent state observation confirms pending.

Joint samples are selected by the confirmed event pattern at handler age: observed boundaries have training delay `<= age`; confirmed-pending boundaries have delay `> age`. At least two complete applicable training vectors must support the condition. Outputs are `parent_receipt_remaining_ms`, `permit_release_remaining_ms`, and `worker_next_ready_remaining_ms`; observed boundaries return zero, pending boundaries average `sample_delay-age` over the **same jointly selected samples**, and an inapplicable permit returns `None`. Result evidence records selected sample ids, current age, original observed times, boundary statuses and support count. Exact known boundary times validate chronology; the empirical condition is coarsened to event pattern and does not claim exact continuous-time conditional regression. Old profiles without joint observations still support unconditional `delays()` but return UNKNOWN for `residual_handoff()`. Non-mean scenarios, insufficient support and unknown readiness remain UNKNOWN. This interface supplies an explicit distributional assumption, not a calibrated confidence interval, and does not turn scarce training support into proven physical accuracy.

## CPU validation

`tests/test_runtime_event_contract_v2.py` exercises FIFO event replay, post-transition counts, incremental cursors, shutdown cancellation, idempotent release, alias exit release, actual collector blockage behind the runtime lock, and send failure. The lock-blockage case verifies that a parent reader event can already exist while the permit is still active, and that release occurs only after collector processing. `tests/test_forecast_lifecycle_v2.py` covers training-only lifecycle export/provider validation, clock rejection, missing evidence and distinct boundaries. `tests/test_worker_readiness_v2.py` executes real CPU subprocesses through success/error/shutdown and actual collector-to-training-export-to-fit integration. Existing subprocess admission and forecast tests are run alongside them. No GPU workload is started by these tests.

## Visible-state rolling correction

`gsdeploy.forecasting.observed_state.new_observed_state(aliases, execution_context_sha256, mode="online", configuration_sha256=None, frozen_cutoff_ns=None)` creates an empty observation checkpoint. `advance_observed_state(previous, now_ns, admission_snapshot=None, feedback=(), arrivals=(), mode=None)` creates the next checkpoint without mutating the previous one. It replaces stale host assumptions with newly observed facts rather than treating the previous forecast's future timestamps as observations.

State carries its schema, host clock, execution/resource hashes, observation time, mode/cutoff, SHA-256 digest, separate admission/feedback/arrival sequence cursors, `admission_snapshot`, `requests`, and per-task `last_arrivals`. Admission replay validates strict FIFO transitions, active/waiting/closed state, peak counts, resource identity, contiguous events, monotonic clocks and the current runtime snapshot. The first gate snapshot must contain the full history from sequence zero. Later snapshots use the last consumed gate sequence; duplicates and gaps are rejected. Each feedback envelope contains schema `gsdeploy.visible_feedback.v2`, a consumer-feed sequence, `kind`, `payload`, reader `received_time_ns`, the host clock and execution identity. The feedback sequence is explicitly an application ingestion order, not a native global kernel sequence.

`feedback_from_runtime_records(records, execution_context_sha256, now_ns, since_sequence=0)` adapts a consistent, retained runtime request registry to that ordered feed. It exports only visible START/PHASE/RESULT-handler-END/WORKER_READY boundaries. The registry must retain its consumed cohort; a sliding subset is not a valid feed source. It extracts no predictions or reference labels. The cutoff uses the reader receipt, so an event timestamped earlier on the worker but received after `now` is not available yet. Public arrivals have an independent contiguous sequence and strictly allow only request/alias/task/pool/arrival/visibility fields; no labels or inference outputs are accepted.

PHASE observations establish only observed phase starts/ends. The module keeps `kernel_cursor=None` and `kernel_cursor_status=UNKNOWN`; it does not convert elapsed age, host phase labels, a predicted segment duration, or theoretical occupancy into measured kernel progress. No ready boundary is inferred from the next START. A request already after handler END retains its separate parent receipt, permit release and worker-ready facts. `INITIAL_HANDOFF_RESIDUAL_UNMODELLED` remains explicit in this observation-only module; a caller must separately obtain a supported conditional lifecycle result before using residual delays.

In `online` mode the observer may consume feedback visible at the current decision time, including feedback from the live deployment. In `frozen` mode a mandatory `frozen_cutoff_ns` prevents the checkpoint from advancing past its prospective observation cutoff, and the mode cannot later be changed to online. Both modes reject future observations. This is a state correction interface; it does not refit the frozen training model, reuse validation outcomes as training, or prove a prediction accurate. The module does not alter the simulator's scheduling or six-term score. `tests/test_observed_state_v2.py` checks incremental/full replay equivalence, prior-state immutability, delayed visibility, stream gaps/duplicates, clock/hash/state mismatch, frozen-cutoff enforcement, and persistence of unknown kernel progress.
