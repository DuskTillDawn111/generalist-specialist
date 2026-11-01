# DAG host supply and resource event axis

The new CPU path separates trained GPU durations, predicted host submission and
the post-completion-bound host tail. It does not reuse measured API launch
offsets or a measured handler-end anchor in a future forecast.

`dag_host_release_v4.py` first required an explicit natural final-sync boundary.
Its preserved training01 fit supports text and G2 text while keeping four OCR
rows UNKNOWN. The separate `dag_host_release_v5.py` verifies the unchanged OCR
adapter source containing an explicit CUDA synchronization before return.
OCR's measured native-call-return boundary is therefore a device-completion
**upper bound**, not an exact final-sync timestamp. Its post-call tail stays
distinct and runs only after all predicted DAG predecessors complete.

The actual v5 training01 model uses 16 natural request observations, eight
light-domain templates, two independent natural captures and two inputs per
family/task group. All three groups are supported. OCR remains multistream:
training feasibility uses a dependency critical path, never a sum of kernels.
Unlimited ready work inside an owner is an explicit lower-bound hypothesis.

Natural elapsed time does not identify CPU supply uniquely. Two explicit
scenarios are fitted: front-loaded submission and submission proportional to
actual host issue depth. Overlapping host issue intervals remain a partial
order. Pooled parameters assume exchangeability within family/task. Matching
isolated training endpoints is a calibration identity, not prediction accuracy.
Observed parameter ranges are not confidence or prediction bounds.

`forecast_gate_v3` accepts `provider_kind=cuda_dag_host_supply_v4`, a receipt-bound
`dag_host_release` artifact, `host_supply_scenario`, and both explicit flags
`allow_host_supply_identification_hypothesis` and
`allow_host_supply_pooling_hypothesis`. The host model must bind the same DAG and
light context and retain the exact natural target context. v4 and v5 host-model
schemas are separate; old replay and fixture paths retain their behavior.

`cuda_dag_simulator_v4` supports release dictionaries with
`not_before_offset_ns` and `after_predecessors_ns`. GPU prediction changes
propagate to completion, FIFO and the future queue. The terminal delay is
relative to dependency completion, so a target endpoint cannot mask bad GPU
service predictions. Copy/memset without a trained model remain UNKNOWN.

The optional `natural_resources` artifact and explicit
`allow_conditional_resource_hypothesis` connect the direct-state v5/v6 provider.
Its original target context and native source projection are checked. The DAG
event engine labels the full handler-active state, including host delays,
integrates board power only over the original measurement window, and takes
the simultaneous owned-memory maximum through request drain. Idle portions of
the measurement window are included. Missing states propagate UNKNOWN, never
zero. The original scorer takes max(startup peak, forecast peak), then applies
the original memory safety factor once. Original QTLPME weights and formulas
are unchanged; startup evidence is still independently required.

The optional evaluation-only `kernel_duration_mode=numeric_semantic_hypothesis_v5`
requires `allow_numeric_semantic_hypothesis=true` and a receipt-bound
`semantic_fit` artifact. It consumes the dedicated semantic resolver's numeric
regression output only when `numeric_hypothesis_available` is true. It preserves
the resolver's `known`, `strict_known` and original training residual/support
rejections. Results count strictly qualified nodes separately from numerical
hypotheses. This mode enables independently predeclared request-level accuracy
evaluation; it does not change the original 25% kernel fit gate, qualify a
deployment, or use a measured query duration as fallback. Missing numerical
features or trained copy/memset service still stop the whole request as UNKNOWN.

`generation_dag_coverage_v5.py` separately joins actual training work descriptors
to exact known-input DAG templates. Its selector accepts only public request
features and frozen empirical work scenarios; it rejects future completion
fields and preserves every unsupported positive-probability outcome. With the
v4 population distribution, the four old ChatQA owner/input cells each cover
2/3 mass (five forwards) and lack the same-input six-forward template; both G2
text cells cover all work mass. OCR is explicitly non-autoregressive. This
population support review is distinct from a future exact-content conditional
work model and does not establish output-length accuracy or extrapolate a DAG.

CPU tests cover parallel critical paths, distinct feasible release shapes,
poisoned target timings, changed GPU predictions, relative completion tails,
overlapping issue order, FIFO/lifecycle, missing evidence, numerical resource
integration and original scorer wiring. No new GPU workload or independent
accuracy validation is performed by these tools.

The host-supply gate now uses the separately versioned
`cuda_dag_simulator_v6.py`. The original v4 source and failed `sealed01` and
`sealed02` directories remain unchanged. The first failed seal rejected a
future-arrival schema mismatch before simulation. The second reached
`DAG_EVENT_BUDGET_EXCEEDED_NO_TRUNCATION`; it has no completed prediction seal
and cannot authorize collection.

The second failure was traced on its actual first candidate, using the exact
`sealed02` bundle and public arrivals with the unchanged v4 event engine. At
clock 17,179,970,365 ns, event 827,078, a positive residual of
1.3822377695760224e-6 ns exceeded the former completion epsilon but was smaller
than the absolute floating-point clock ULP of 3.814697265625e-6 ns. Adding its
completion interval left the absolute clock unchanged. Thirty requests had
completed; the positive residual then caused a zero-time fixed point. The
read-only diagnostic is preserved at
`artifacts/predictor-dag-actual-campaign-clock-review-20261028-final01/report.json`.
This is a numerical progression failure, not evidence of genuine five-million
event workload complexity.

v6 keeps every event boundary an integer nanosecond. It rounds a positive
relative completion interval upward before adding it to the absolute clock,
consumes service using the actual integer elapsed interval, and uses that same
completion interval to decide completion. Predicted provider durations remain
unchanged. Each upward rounding adds less than one nanosecond, while serial
and interacting activities can accumulate error; no request-level global
one-nanosecond bound is claimed. Event budgets and UNKNOWN handling are
unchanged. An independent review covered 24 fixtures across clock offsets 0,
30 seconds, 10^12 ns and 10^18 ns, including fractional service, near-completion
arrivals, concurrent owners, serial dependencies and drain boundaries.

`dag_queue_diagnostics_v6.py` is now wired into this actual host-supply event
path. It uses the same cached numerical duration per `(alias, task, pool_id,
node)` as execution. It reports all declared arrival/work envelopes, remaining
fitted GPU/copy/memset activity work, oldest waiting request age, owner FIFO
head blocking, the integral of all queued request waits, and backlog-clear
episodes. Handler activity and worker-handoff blocking remain separate causes.
These diagnostics observe the existing rates and routes; they do not impose
a global admission cap or change QTLPME scoring or stability rules. Fitted
activity work excludes unidentified host work and is not request wall time.
SM/DRAM/L2-normalized work remains null without an explicit normalized-demand
provider; occupancy is not converted into resource demand.

Coverage counts on the cached gate path have explicit
`query_count_scope=unique_used_template_nodes_cached_for_all_requests`; they
count distinct queried template activity nodes, not all repeated execution
instances. Profiles belonging to different resident aliases remain separate.
An integrated CPU event-path counterexample with 100 ms of fitted service and
a 50 ms request arriving at 25 ms reports 125 ms of remaining activity work,
75 ms of FIFO wait, and a queue-clear episode from 25 to 100 ms. This validates
the event accounting contract; it does not establish prediction accuracy on
new GPU observations or certify routing gain. Four-task validation and final
routing-gain comparisons remain separate from the 57-request known-input
text/OCR pilot.
