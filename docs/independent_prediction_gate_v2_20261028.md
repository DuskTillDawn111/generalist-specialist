# Independent prediction gate v2

`gsdeploy.forecasting.prediction_gate` evaluates the new predictor under separately frozen evidence. It does not import, rewrite, relax or replace the old V4 thresholds/results. Its CPU fixtures validate the gate implementation; they are not GPU accuracy evidence or a reason to rerun the completed historical campaign.

## Freeze before independent collection

`freeze_gate_protocol(...)` requires model paths, predictor/backend source paths, training-index paths, candidate/model/execution-context bindings, the baseline candidate, a prospective validation run/input domain, explicit original six-term weights, user-declared thresholds, and a Unix metadata creation time. It binds all file SHA-256 receipts plus its own evaluator source. Training indices use schema `gsdeploy.independent_prediction_gate_training_index.v2`, split `training`, and records containing `run_id`, `request_id`, `pool_id`, and `task`. Validation domains list `run_id` and requests containing `request_id`, `pool_id`, and `task`. Both validation runs and input pool identities must be disjoint from training. Candidate comparison uses the same declared request domain within a run.

The index must cover all training inputs/runs, including service, interference, lifecycle, power, memory and quality training sources used by the predictor. An index cannot establish that a falsely labeled or incomplete external training lineage is honest. Model/source receipts are likewise necessary evidence identities; actual model fitting provenance and a repeatable prediction command must accompany a real campaign.

`seal_predictions(protocol, cells, created_unix_ns=...)` binds predicted request timings and six-term outputs to the frozen protocol. Each cell contains `run_id`, `candidate_id`, `execution_context_sha256`, `requests`, and `terms`; requests contain public request/input/task identity plus `service_ms`, `queue_ms`, and `e2e_ms`. `terms` contains Q/T/L/P/M/E in the original normalization, or explicit missing evidence. An optional `queue_depth_samples` field is discussed below. No future actual output belongs in this prediction seal.

Observed validation files use schema `gsdeploy.independent_prediction_gate_observed_run.v2`, `split=validation`, `kind=measured` or `cpu_fixture`, a declared `run_id`, `collection_started_unix_ns`, and cells of the same shape. Collection must start strictly after the prediction seal; the prediction seal must follow the protocol freeze. These Unix timestamps describe artifact/collection chronology. Request duration fields stay in milliseconds; this contract never compares Unix metadata directly to CUDA or host perf-counter intervals. Metadata chronology is an explicit audit prerequisite, not a cryptographic external timestamp service.

`evaluate_prediction_gate(protocol, predictions, observed_paths)` rechecks all frozen file identities and validates exact candidate/request/input/task/context bindings. It rejects duplicate or foreign observations and physical/CPU mixing. Missing candidates, requests, timings, utilities and queue samples remain explicitly incomplete rather than disappearing from the requested denominator. Model fitting and threshold tuning have no evaluation API.

## Error and decision metrics

For every run/candidate, the report includes service, queue-wait and end-to-end MAE/MARE and the full task-cohort empirical P95 error. P95 uses the same linear empirical quantile as the predictor metrics; one missing member makes that task P95 unknown. Zero observed denominators are explicit: equal zero contributes zero relative error, while positive prediction versus observed zero makes the relative aggregate unknown instead of dropping the example.

Queue **depth** error is distinct from queue-wait duration error. A protocol can freeze `queue_sample_offsets_ns` and each cell can supply `{offset_ns, depth}` samples on that exact grid. Missing points keep requested coverage incomplete. `queue_depth_mae_max` uses depth units; `queue_mae_ms_max` uses milliseconds. The module does not reinterpret the old V4 depth-MAE threshold as a duration threshold.

Utilities are reconstructed as `U=-sum(weight_i*term_i)` from the explicit six terms. The report compares every nonbaseline candidate with the predeclared baseline and checks ΔU sign with the frozen tie tolerance. It also checks every unordered candidate pair within every run, including predicted/observed ties. All-pair coverage is reported as requested/known; the module cannot select a favorable subset after observation. It includes ΔU MAE, sign accuracy and pair-ranking accuracy, each with the small-sample/descriptive limits of the supplied collection.

Supported threshold names are `service_mare_max`, `queue_mae_ms_max`, `queue_depth_mae_max`, `e2e_mare_max`, `p95_mare_max`, `delta_sign_accuracy_min`, and `pair_rank_accuracy_min`. Values are supplied before evaluation; there are no built-in replacement values for the historical protocol. Per-request timing/P95 thresholds are checked per run/candidate, avoiding a global average hiding a failed cell. Known violations yield FAIL; missing required evidence yields UNKNOWN; complete checks yield PASS. With no thresholds the output is `ASSESSED_NOT_GATED`. Even a passing CPU fixture keeps `physical_gain_certified=false` and `final_four_groups_enabled=false`.

## CPU evaluation command

```sh
python -m gsdeploy.forecasting.prediction_gate \
  --protocol /absolute/path/protocol.json \
  --predictions /absolute/path/predictions.json \
  --observed /absolute/path/observed-run.json \
  --output /absolute/path/new-report.json
```

The command only evaluates previously sealed files, binds its input artifact receipts, records `GPU_started=false`, and refuses to overwrite an existing report. It never launches inference, accesses GPU counters, changes a driver, refits a predictor, or updates the old campaign.

`tests/test_prediction_gate_v2.py` exercises exact and failed errors, full-cohort P95, reversed ΔU/ranking, missing request/candidate coverage, queue-depth sampling, frozen source/model/data/threshold identity changes, training/validation input/run leakage, preseal observations, undefined relative denominators and the command's output preservation. `tests/test_prediction_integrations_v2.py` independently verifies that optional risk CLI output leaves forecast/metrics unchanged, roofline complementarity preserves all pairs and matching precision, serial preparation lacks physical power/memory evidence by default, and an unseen exact batch configuration cannot borrow a solo profile.
