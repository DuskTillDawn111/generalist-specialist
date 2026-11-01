# Learned campaign seal v15

`tools/predictor_validation_20261027/learned_campaign_seal_v15.py` is a CPU
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
are resolved with `scoped_training_evidence_v15.resolve_closure`; exact original-byte
historical substitutions retain the unchanged historical resolver and require
an explicit manifest. Exact strict-reviewed training contexts' top-level
`prior_failed_evidence` targets are byte-bound without parsing their nested
historical metadata. One separately reviewed preparation-inventory edge is
qualified by its exact parent SHA, JSON path and target SHA, with a source-bound
policy receipt and digest. Every boundary retains the parent and target bytes;
any normal edge to the same target upgrades it to recursive traversal and
invokes the outcome-read guard. The index audit's old
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
python -m tools.predictor_validation_20261027.learned_campaign_seal_v15 \
  --plan /absolute/prepared/plan.json \
  --training-index /absolute/complete-index/training-index.json \
  --configuration /absolute/preflight/configuration.json \
  --numeric-preflight-report /absolute/preflight/report.json \
  --historical-manifest /absolute/exact-byte-resolution/manifest.json \
  --output /absolute/new-seal-directory
```

The collector remains a separate invocation and performs its original
precollection verification against these sealed artifacts. Ten small CPU
fake tests verify ordering, completeness, source immutability, original
threshold preservation, priming coverage and the real v5 precollection adapter.
Together with 31 resolver tests, the 41 tests also cover byte drift, both
ordinary-dependency upgrade orders, exact snapshot aliases, validation guards,
and freezing opaque target bytes and their parent. A pre-parse guard checks the
original identity, resolved snapshot and all manifest-declared historical aliases
before deduplicating a normal dependency. Read spies verify that sensitive JSON
snapshots remain unparsed even when the manifest names the snapshot first.
These tests do not establish a complete real forecast, prediction accuracy or
physical gain. No production protocol or predictions were sealed while
implementing this entry point.

The v15 configuration selects the joint HELD/IN_TRANSITION initial prior and
the compact v15 cumulative controller while retaining the same fitted v13 SAVE
model. These implementations and their dependencies enter the same source
freeze; a v14 configuration/preflight cannot qualify this v15 entry point.
