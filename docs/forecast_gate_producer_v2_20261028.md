# Actual forecast production for independent validation

`tools/predictor_validation_20261027/forecast_gate_v2.py` connects the frozen
gate to actual `forecast_request → FullHorizonModel → score_forecast` execution.
It derives timing cells from the simulator's handler start/end and parent
receipt. It never accepts caller-supplied predicted request timings or utility
terms. This is CPU implementation evidence, not a physical accuracy result.

The public API is:

```python
run_and_seal(protocol_path, jobs, output_dir,
             allow_cpu_fixture=False, allow_analytical_hypothesis=False)
```

Each job has `run_id`, `candidate_id`, and `public_inputs_path`. The jobs must
cover every prospective run/candidate combination. The protocol freezes each
public file, the producer and the complete current Python package sources,
including predictor, scorer, batch planner and routing implementation. Changed
source/model/data receipts fail before prediction and are checked again when
sealing. A new exclusive output directory contains the actual forecast for each
cell, `predictions.json` and `production-receipt.json`.

Each protocol-bound model is a JSON bundle with:

- `schema: gsdeploy.forecast_gate_model_bundle.v2`;
- `kind: trained_model` (or explicitly opted-in `cpu_fixture`);
- `execution_context_sha256`, exactly matching its declared candidate;
- `forecast_defaults`, containing the frozen provider, lifecycle, resources,
  aliases, continuation routing, quality predictions and objective references;
- `startup_prediction`, with `basis: frozen_training_prediction`, an explicit
  `evidence` identity, `energy_joules` and `peak_memory_mib`;
- `startup_amortization_seconds`, inherited from the experiment contract.

Real model bundles must use `provider_kind: kernel_resource_v2` and
`execution_contract: strict_lifecycle_v2`. The CPU-fixture option permits the
calibrated test provider for hand-checkable integration checks. Enabling an
analytical model does not establish that its physical assumptions are accurate.
Kernel resource fields, timing evidence, lifecycle, board power, memory and
quality still need actual training support. This producer does not manufacture
those models from missing data.

`allow_profiled_baselines=True` (CLI `--allow-profiled-baselines`) separately
permits declared profiled-isolated kernel baselines. It propagates to the actual
provider and is recorded in the production receipt. Such forecasts remain in
their explicit profiling domain; natural timing is not automatically validated.

Public input JSON uses `schema: gsdeploy.forecast_gate_public_inputs.v2` and
`forecast_inputs` with exactly `phase_start_ns`, `now_ns`, `deadline_ns`,
`drain_deadline_ns`, `future_workload`, and `queues`. This prospective whole-run
gate starts at phase start with a declared public schedule. Initial queued
requests at offset zero are explicit; strictly later requests use
`declared_arrivals`. Public queues contain only request/task/input/arrival/route
fields. Validation outputs, labels and execution feedback are excluded. Online
Gamma/CV scenario prediction remains supported in the ordinary forecasting API;
it is not passed off as an exact heldout trace in this paired experiment.

Paired candidates must have identical arrival/input cohorts, measurement and
drain windows, task weights and objective normalizers. Only routes and supported
execution configurations differ. Every request contributes to the seal;
unsupported predictions retain null timings/terms and explicit UNKNOWN
forecasts. No candidate or failed request is removed from the denominator.

Service is handler end minus handler start; queue wait is handler start minus
scheduled arrival; E2E is parent receipt minus scheduled arrival. Depth samples
are waiting-only: `arrival <= sample_time < handler_start`, consistent with
`prediction_observed_v2.py`'s `waiting_until_handler_start` boundary. Preparation
and batch waits therefore remain waiting until the recorded dispatch boundary.

M uses the maximum of training-predicted startup peak and simulated future owned
peak, applying the inherited factor once. E combines measurement board energy
with `startup_J/(reference_W × amortization_seconds)`; simulated drain energy is
excluded. Missing startup energy is UNKNOWN, not zero. The initial peak is a
training prediction, not a validation startup observation.

The quality scorer now sums predicted error/reference units by task before
division, then applies offered-rate task weights. OCR/ASR scalar CER/WER values
alone lack the reference-length denominators and remain UNKNOWN. The frozen
`quality_loss[family][task]` entry accepts `basis: expected_per_request_totals`,
`numerator`, `denominator`, and nonempty `evidence`; an optional `by_input` map
with an explicit `fallback` permits training-derived input conditioning. These
numbers are expected units per request, not unnormalised totals over differently
sized training sets. Text/image scalar losses retain denominator one; edit-error
ratios remain unbounded above one.

Five producer checks run actual simulator forecasts, including nonzero receipt
and next-ready delays, queue depths, startup energy, incomplete evidence,
source/candidate coverage, and output preservation. Four additional quality
tests verify unequal-reference ratio-of-sums and missing-denominator handling.
Neither test suite qualifies deployment or opens the final benefit experiment.

The next collection step must call `verify_before_collection` from
`prediction_observed_v2.py` and bind its receipt plus the actual collection start
time into raw data. See `prediction_observed_adapter_v2_20261028.md` for the
complementary endpoint and six-term observation contract.
