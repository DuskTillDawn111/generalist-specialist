# Original public-flow forecast adapter

`tools/predictor_validation_20261027/twoFIFO_prospective_forecast_v1.py`
reuses the original world engine, ensemble, queue bounds and six-term ranker.
It adds no event engine, admission policy, fit, GPU execution or prediction seal.

The historical public plan and original equal-six-term objective are validated
again. An explicit complete old-to-fresh request-ID bijection and integer epoch
are the only calendar changes. Exact input IDs, tasks, arrival order/offsets,
candidate routes, all residents, offered rates, normalization and queue-sample
offsets are retained. Original181 has 167 text, 5 image, 5 ASR and 4 OCR arrivals;
its 90-second measurement and 120-second drain become [epoch, epoch+90s] and
epoch+210s. The training helper's 10+60 constants are not imported or modified.

## Interfaces

`public_calendars(plan, objective, request_id_map=..., forecast_origin_ns=...)`
returns the complete original contract and every shifted candidate calendar.
The fresh ID map is an explicit draft identity binding, not a fresh-run seal.

`forecast_compiled(..., draws_by_candidate=..., quality=..., settled_seed=...,
resources_by_candidate=...)` is a **model-level propagation interface**. It
accepts the same exact paired joint draw catalog for all original candidates.
Whole-request template/rate/protocol/transport inputs feed the existing ensemble;
the same requests, draws and prefix feed the separate conservative envelope.
Every candidate and original Q/T/L/P/M/E normalization feed the existing ranker.
Only the original resource shared fields are accepted; boundary or cap overrides
are rejected. Missing resources leave M/E/fullU unsupported. A missing, wrong-origin
or wrong-roster settled seed makes numeric worlds UNKNOWN rather than inventing
empty queues. Already UNKNOWN draws keep their original reason and weight.

`forecast_donor_law(..., joint_law=..., donor_blocks=...)` is the conservative
native-source readiness path. Existing native laws explicitly have
`prospective_template_transfer_qualified=False`, so every original draw remains
UNKNOWN. Missing law evidence also remains UNKNOWN. The catalog is four slots
per donor block: two blocks give eight worlds; all three give twelve. Neither
path deletes or renormalizes unknown worlds. A caller setting the qualification
Boolean true cannot bypass the absent native bridge verifier.

`prepare_cpu(public_plan_path, objective_path, donor_blocks=..., output=...)`
saves replayable public-only readiness plus the local import/source closure.
`audit_cpu(report_path)` rederives it without native outcomes. Its CPU draft IDs,
empty quality document and absent law/prefix explicitly carry no execution or
prediction seal. This preparation is not a numerical forecast-accuracy result.

## Evidence and independent validation are distinct

A prospective prediction may be an uncalibrated, predeclared hypothesis with
clear training sources. Independent prediction accuracy is evaluated afterward;
it is not required to manufacture a forecast or authenticate its training input.
The compiled entry supports this model-level propagation and makes no native
source or deployment eligibility claim.

The missing prospective bridge is a different problem: it must check exact
input content, backend/context and worker-protocol identity, causal whole-request
template support, donor membership, rates, transport transfer and initial-state
source evidence. A future bridge must use public/training/past-prefix evidence,
never the target run's future START/READY/response labels. Existing recovery
resource providers restricted to old heldout block/attempt identities cannot be
relabeled for a fresh181 execution. This adapter adds no fake bridge or numeric
support for those missing physical domains.

After source binding and a predeclared ensemble-to-gate projection are available,
the known-input v5 gate can check fresh co-run accuracy on the exact trained
inputs. World-average P95 is not a mixture latency P95. Independent errors,
six-term delta-U sign/rank, actual native quality, full drain and unchanged
stability/rho/g(n) remain prerequisites for production routing and gain claims.
All current wrapper outputs explicitly stay HOLD and unsealed.

## CPU verification

`python -m unittest tests.test_twoFIFO_prospective_forecast_v1 -v` covers the
entire real public181 calendar, unchanged rates/routes and 90+120 boundary,
real original-engine generation of an arrival at 80 seconds, eight/twelve worlds,
partial UNKNOWN mass, invalid map/candidate/catalog/objective/cap rejection,
wrong-prefix UNKNOWN, propagated programming errors and replayable CPU receipts.
These synthetic/model tests are not GPU or independent predictive validation.
