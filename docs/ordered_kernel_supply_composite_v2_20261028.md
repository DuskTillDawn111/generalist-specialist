# Ordered kernels, launch supply and composite execution

The CPU-only tools now connect exact-source training kernel sequences to
`FullHorizonModel`, while preserving the difference between observation and a
model hypothesis. They do not change any frozen execution/package sources.

## What is implemented

* `kernel_sequence_audit_v2` reads each canonical kernel once (NVTX regions plus
  unattributed, never the duplicated phase view), verifies raw/plan/context and
  SQLite receipts, and reports exact duration sum, interval union, stream FIFO,
  overlap and unclassified gaps. Runtime correlation lookup is linear, not a
  full trace map copy per kernel. Auxiliary copy counts are trace-wide only.
* `ordered_kernel_profiles_v2.convert_collection` accepts isolated
  `split=training, domain=light-kernel` only. Smoke/validation cannot be fitted.
  Each accepted serial-stream request keeps every kernel and every positive
  unclassified interval. Rejected branches remain in the denominator.
* `ordered_occupancy_resolver_v2.AuditedOccupancyResolver` joins the exact SQLite
  SHA, unique GPU UUID, launch row ID and occupancy-relevant group signature.
  It restores each member's original completed-launch evidence rather than
  substituting the representative launch. Old smoke patches cannot transfer.
* `OrderedKernelProfileProvider` makes the timeline executable. Isolated
  profiler-domain replay requires explicit opt-in. A general unclassified gap
  is neither CPU work nor evidence of an idle device; cross-request service
  remains unknown without a conditional model.
* `launch_supply_residual_v2.fit` checks the exact-work three-domain bridge.
  For each supported exact training input, it retains all light-domain kernel
  durations and computes `natural_handler_ns - sum(light_kernel_ns)`. Negative
  residuals are UNKNOWN, never clipped. The nonnegative residual is partitioned
  before kernels using positive measured CUDA launch API elapsed-time weights,
  with exact integer conservation. These are supply-placement hypotheses; API
  elapsed time is not CPU execution and its weights are not natural timing.
  Profiled gaps and GPU durations are never scaled.
* `LaunchSupplyResidualProvider` requires explicit independent-host-supply and
  absent-copy-interference hypotheses. It offers static packing or equal-context
  time slicing, including a separately specified per-kernel context-switch cost
  scenario. No MPS or same-SM capability is inferred. Zero switch cost is an
  explicit optimistic scenario, not an observed fact.
* `composite_execution_v2` compares source-bound physical projections across
  metadata, light and natural domains and both lifecycle captures. Every
  nonmeasurement/nonpacket field remains: worker settings, residency, batch,
  backend, quantization, device, threads and original source hashes. Four
  instrumentation fields and five packet bookkeeping fields have an explicit
  exclusion list. Source components retain their original context IDs. Wrappers
  expose a new composite ID and record the `profiled -> mean` lifecycle scenario
  adapter. The target collector's original natural ID is separate and explicit.
  `verify_target_runtime` additionally checks current actual source bytes against
  the frozen target declaration; it does not start a worker or GPU.
* `launch_supply_replay_v2` runs known-input CPU diagnostics, optionally with
  strict lifecycle, FIFO and future arrivals. Outputs explicitly deny independent
  validation, unseen-input accuracy or utility-gain claims.

## Real training evidence and present limits

The first light capture contains 129,545 request-owned kernels. Six language
requests use one serial stream. Two OCR requests have two/three streams with
0.528/0.844 ms overlap excess; flattening by duration sum is rejected.

For the six serial requests, natural handler minus light GPU work is positive:
178.452, 162.933, 65.298, 208.978, 184.478 and 243.930 ms in request order.
Matching the natural training endpoints after residual fitting is an in-sample
identity, not prediction accuracy. One capture's residual fit does not satisfy
the independent-capture minimum; the separate lifecycle fit has two captures.

`cuda_dependency_dag_v2` reconstructed the entire OCR light trace, including
initialization: 20,197 kernels, 1,390 copies, 78 memsets, 2,534 event records and
4,411 stream waits. Exact `(process, context, eventId, eventSyncId)` matches yield
zero unresolved event waits and no cycle. Timestamp-zero event records never
become completion times. `request_cuda_dag_v2` now attributes all eight requests, preserves all 129,545
kernels, adds request-owned copies/memsets and blocking host synchronization,
and reconstructs an acyclic dependency graph with no unresolved event waits.
`cuda_dag_simulator_v2.DagFullHorizonModel` is an executable multi-node event
engine: ready nodes on different owner streams can run concurrently, while
owner request FIFO and distinct worker-ready/parent-receipt delays remain.
The real known-input replay completed all eight requests; OCR reached three
simultaneously active nodes. This is an explicitly labeled profiler replay,
not fitted natural timing. Default-stream implicit semantics and copy/DRAM
interference remain unmeasured. Production v3 explicitly rejects observed
API-issue releases and the observed handler-end anchor: a separately trained
host-release model is required before claiming fitted DAG prediction.

Clock mapping retains its boundary-placement uncertainty. An earlier
`valid=true` median offset is not a microsecond cross-process calibration
certificate. Inter-kernel profiled gaps are not natural launch delays, and
NVTX-minus-kernel time is not labeled CPU time.

## Next implementation order

1. Add the completed second light/natural capture as repeated training evidence;
   compare kernel work/signatures and residual variability before aggregating.
   Keep whole capture and input grouping separate in all resampling.
2. Train the remaining DAG activity duration and host-release models. Request
   attribution, exact event/wait/blocking-sync edges and concurrent intra-owner
   ready-node execution are implemented; they must not be mistaken for trained
   timing or default-stream/DRAM validation.
3. Replace exact-input residual lookup with a training-only nonnegative model
   using public structure features: input token count, configured generation
   budget, image dimensions, backend/batch/owner, and predicted forward/launch
   counts. Generated token count or OCR detected-box counts from validation
   responses must never become prospective input features. Use train-derived
   conditional scenarios for unknown output work.
4. Independently generalize GPU work/templates by input shape and semantic
   phase. A residual regressor alone cannot predict an unseen kernel sequence.
   Two lengths per owner do not establish a reliable shape model. Collect a
   sufficient training-only shape grid and repetitions, fit phase/work models,
   preserve static launch/occupancy constraints and reject unsupported shape or
   backend domains. Avoid a whole-context slowdown lookup table.
5. Preseal predictions on disjoint unseen-input validation, comparing packing
   and time-slicing/switch-cost hypotheses, start offsets and actual shared
   execution. Evaluate co-run slowdown, full queue/P95, delta-U sign and ranking.
   Known-input repeats only assess repeatability; they cannot pass unseen-input
   gate requirements. Fit uncertainty from training captures, not validation.
6. Connect separately trained board-power and owned-memory models before six
   term utility evaluation. The prototype deliberately leaves both UNKNOWN;
   matching handler endpoints cannot justify E/M or spillover benefit.

The current prototype uses no arbitrary global cap of one/two admitted requests.
Request admission remains an explicit runtime configuration, not a paper-derived
default. Original stability rules and the six-term utility definition are intact.


## Actual fitted kernel coverage and new v3 gate

`trained_supply_provider_v3.TrainedKernelSupplyProvider` now calls the actual
frozen regression resolver, replaces supported kernel durations in the running
service provider and preserves measured training durations only as separate
metadata. Unsupported kernels make the complete request UNKNOWN; there is no
target-duration fallback and the natural residual is not retuned to recover the
measured endpoint. The real two-capture model query covered 29,676 of 109,348
kernels across six serial language requests (27.14%); 79,672 kernels remained
UNKNOWN. None of the six complete requests is yet a supported prediction. OCR's
20,197 kernels stay in the separate DAG denominator. Per-profile unknown causes
are retained in `trained-provider-coverage-v3-20261028-diagnostic01/report.json`.

`forecast_gate_v3` constructs the new providers, actual FullHorizon/DAG engine
and original `score_forecast` before the v2 protocol sealer. It never accepts
prefilled timing/utility cells, keeps the original natural execution identity
separate from the composite model identity, and retains the complete public
cohort and original QTLPME/five normalizers. M/E remain UNKNOWN without trained
resource models. DAG production also rejects profiled release fallback.

The prospective draft in
`artifacts/predictor-gpu-20261027/prospective-disjoint-validation-v3-20261028-prepared01/`
reserves 16 real public inputs disjoint in both pool ID and exact content from
the bound six-capture training lineage, which is indexed with all 48 requests.
The four original shared candidates receive the same 181-request, four-task
schedule: 0.5x / 2x / 0.5x rates over three 30s intervals. This retains the
original average task rates, six weights, normalizers and thresholds, including
no newly invented queue-ms threshold. Each 0.9s queue sample is explicit.

The CPU wiring diagnostic retained all 724 candidate/request rows and produced
four UNKNOWN cells because the reserved unseen-input DAG templates are not
trained. It is not a prediction seal or an independent accuracy evaluation.
A recursive closure precheck also found historical `trace_export.py` receipts
whose bytes differ from the current tool. No old artifact/hash was rewritten.
Source/model closure must be resolved and frozen before any validation launch.
