# Learned campaign seal v14

`tools/predictor_validation_20261027/learned_campaign_seal_v14.py` is a CPU
producer for the prepared four-candidate, 57-request learned availability
campaign. It preserves the prepared run ID, request IDs, routes, original
normalizers, startup accounting, six weights, queue grid and validation
thresholds. It emits the original `known_input_gate_v5` protocol and prediction
schemas consumed by `known_input_native_flow_v11` and
`known_input_observed_v5.verify_before_collection`.

The required numeric preflight must bind the exact plan/configuration, have four
complete numeric forecasts, retain all 57 rows in each forecast, and have
unchanged current source receipts. Its results are checked as prerequisite
evidence. The producer accepts no supplied prediction cells and does not use
preflight cells for its seal.

The complete v12 training index must have its adjacent report and audit
artifacts. The producer rechecks each original strict training capture and
compares the exact measured-plus-priming record union with the supplied index.
Actual learned provider construction retains its existing training-source
checks. The model/plan receipt closure and all strict capture review closures
are resolved with `historical_evidence_v4.resolve_closure`; exact original-byte
historical substitutions require an explicit manifest. The index audit's old
gate metadata is bound by receipt without following that separate historical
gate graph. This does not claim an exhaustive audit of all project history.

After saving the protocol, the producer constructs
`LearnedAvailabilityForecast`, executes all four full cohorts again, projects
each with `gate_cell`, validates numeric fields with the original gate, and
runs `check_complete`. It verifies sources before and after execution and only
then calls `seal_predictions`. UNKNOWN or incomplete forecasts preserve a
failure report and cannot produce a prediction seal. Outputs are exclusive.
No workers, collection or GPU execution are started.

Example for a future authorized production run, after its numeric preflight is
complete:

```bash
python -m tools.predictor_validation_20261027.learned_campaign_seal_v14 \
  --plan /absolute/prepared/plan.json \
  --training-index /absolute/complete-index/training-index.json \
  --configuration /absolute/preflight/configuration.json \
  --numeric-preflight-report /absolute/preflight/report.json \
  --historical-manifest /absolute/exact-byte-resolution/manifest.json \
  --output /absolute/new-seal-directory
```

The collector remains a separate invocation and performs its original
precollection verification against these sealed artifacts. Seven small CPU
fake tests verify ordering, completeness, source immutability, original
threshold preservation, priming coverage and the real v5 precollection adapter.
These tests do not establish a complete real forecast, prediction accuracy or
physical gain. No production protocol or predictions were sealed while
implementing this entry point.
