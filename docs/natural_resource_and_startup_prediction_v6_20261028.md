# Resource and premeasurement prediction, 2026-10-28

The original four-owner shared execution now has directly observed training
power/memory support for all 24 handler-activity states. This is resource-term
training, not a service-slowdown model or independent accuracy result.

The full state is the count of active specialists (0/1/2), the generalist task
(idle/text/image/ASR), and OCR activity (idle/busy). The new bounded packet ran
two independent natural captures, each with 30 jobs and 84 measured requests;
the original four residents and 12 priming requests were retained. Both strict
reviews passed, and every state had an actual interior NVML sample in each new
raw capture. No per-model power or memory maxima were added together.

The nine-capture resource fit is
`artifacts/predictor-gpu-20261027/natural-resource-training-v5-20261028-nine-captures-final01/model.json`.
It has 229 measured requests, all 24 states supported by at least two actual
raw captures, and keeps unsampled intervals out of state fitting. The smallest
observed state support is 0.333557113 seconds across its captures; sample counts
do not represent independent experiments.

The source-sealed provider model is
`artifacts/predictor-gpu-20261027/natural-resource-training-v6-20261028-nine-captures-final01/model.json`.
Its 1,357-source recursive closure includes nested strict reviewers and the
exact full training index. `natural_resources_v6.py` preserves every v5 estimate
and original receipt, checks index coverage against every actual raw request,
and recomputes the closure at provider construction. Historical substitutions
require byte-identical retained snapshots and an explicit manifest. The
seven-capture v5/v6 artifacts remain intact. Independent CPU source review
found no blocking issue in the v6 sealing changes.

Provider construction with the actual natural target context passed in
`artifacts/predictor-gpu-20261027/natural-resource-training-v6-20261028-nine-captures-provider-review01/report.json`.
Five new v6 tests cover nested reviewer changes, index changes, altered state
estimates, missing indexed requests and historical snapshot corruption.

These are conditional training means and empirical sampled peaks. Thermal
history, input/phase mixture and pooling the two specialist aliases remain
assumptions. The observed idle state includes previous activity and is not a
causal idle-power baseline. The memory prediction is not a capacity bound.
The scoring layer must take the maximum over startup, active time and drain,
then apply the original 1.110854 factor once.

## Startup energy correction

`startup_prediction_fit_v4.py` explicitly estimated loading-only energy from
`startup_begin` to `startup_end`, excluding priming. The original six-term
scorer in `flow_metrics.py` integrates `startup_begin` through the measurement
start, including priming. The old target therefore cannot be inserted directly
as the original E term's premeasurement prediction.

The separate `startup_prediction_v5.py` retains the old source/results and
integrates `startup_begin` through `active_begin`. It reuses the strict telemetry
and ownership checks, requires at least two independent captures, matching
natural native execution projections and exactly matching actual priming.
Its future collector adapter checks the same planned priming rule against
the public input plan. Collector invariance is explicit; full context identities
remain distinct. Future warmup beyond that priming needs separate evidence.

The seven actual startup-aware natural captures give mean premeasurement energy
5822.193265376832 J and mean premeasurement owned sampled peak 29766 MiB in
`artifacts/predictor-gpu-20261027/startup-including-priming-training-v5-20261028-seven-captures-final01/model.json`.
The startup-aware model excludes active and drain energy and applies no memory
safety factor. Energy is amortized over the unchanged 3600 seconds.

Four new focused tests check the energy window, ownership/bracketing refusal,
actual collector priming order, and source/native/priming transfer checks.
None of these training or CPU checks establishes independent prediction
accuracy or an end-to-end spillover gain.
