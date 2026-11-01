# v48 prospective artifact binding

`projected_handler_campaign_seal_v48.py` is a CPU-only producer for the original
v5 prediction gate. It accepts already computed full-cohort forecasts. It
explicitly records `preflight_cells_reused=true` and
`provider_constructed_after_freeze=false`; it does not assert the historical
v16 fresh-provider-after-protocol production contract.

Preparation requires four numeric, complete 57-request forecasts, unchanged
producer sources, the original v11 plan/cohort/context/routes/thresholds/six
equal weights, and exact factory/preflight receipts. Any UNKNOWN stops before
creating preparation output. An explicit `--forecast-contract` JSON declares
the forecast schema, preflight schema, scheduler version and implementation
receipts, so a new v50 producer is bound without relabeling v44 evidence:

```json
{
  "schema": "gsdeploy.projected_handler_artifact_contract.v48",
  "forecast_schema": "gsdeploy.full_cohort_projected_worker_handler_context_forecast.v50",
  "preflight_schema": "gsdeploy.projected_worker_handler_context_full_cohort_CPU_preflight.v50",
  "lifecycle_scheduler_version": 50,
  "factory_source": {"path": "ABSOLUTE_SOURCE", "sha256": "EXACT_SHA"},
  "preflight_source": {"path": "ABSOLUTE_SOURCE", "sha256": "EXACT_SHA"}
}
```

Each gzip is independently checked over the complete uncompressed byte stream
(CRC, SHA256 and length). A frozen `ijson` runtime then streams the complete
JSON, rejects duplicate keys, and projects the fields needed to independently
recompute the original six terms and gate cell. Both must exactly equal the
producer's metrics and unsealed cell. Scheduler diagnostics are retained in
the original lossless artifact; the preparation process does not materialize
their large object graphs.

The complete strict v12 training index is reaudited. Original availability DAG
captures must be members of that index and the component bundle must match its
lineage roots. Four reviewed v23 light context training captures add all 57
measurement and 12 priming identities each. Independent risk validation raws
are not training inputs. The source union freezes model/training artifacts,
producer source-before/after, public/routes, collection/runtime sources,
streaming parser sources and exact predictions. Retired parsed training
derivatives remain represented through their existing archive manifests and
gzip artifacts; original missing paths are not reconstructed as fake receipts.

`--additional-execution-sources` accepts an exact JSON manifest with schema
`gsdeploy.additional_execution_sources.v48` and a nonempty `sources` list of
Python entry receipts. Each entry's complete static local import closure and
the manifest itself join the preparation and eventual protocol. This binds a
new external watchdog supervisor without changing the original collector or
prepared context identities.

Preparation writes no protocol or seal. Sealing requires an independent PASS
review bound to the exact preparation SHA, with
`complete_four_forecasts_reprojected=true` and no issues. It freezes a new
original-v5 protocol, seals the four exact verified cells, and runs the original
v5 precollection bindings against the preserved v11 plan. Source drift rejects
production and quarantines a written prediction seal. No GPU operation is
implemented in this tool. Numeric completion establishes neither physical
accuracy nor a positive utility gain; those require later independent data.

CPU checks on 2026-10-28: seven author tests pass. A real v44 balanced artifact
of 671,869,955 uncompressed bytes passed complete roundtrip verification and
independent exact metric/gate reconstruction in 14.45 seconds with 105,244 KiB
maximum RSS. The final actual v44 report is rejected because two candidates are
UNKNOWN. Strict training reauditing produced 1,005 original plus 276 supplemental
identities, and all 200 supplemental source receipts verified. No actual
production preparation or seal has yet been issued.
