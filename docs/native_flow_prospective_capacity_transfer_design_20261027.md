# Prospective training-condition capacity transfer adapter

This documents the independent `flow_capacity_transfer.py` implementation. It is not a target-seed measurement or a modification to `gsdeploy.performance` or `flow_eligibility`. The module performs CPU-only sealing, verification and descriptive reporting.

Usage after all declared training flows and capacity catalogs are complete:

```
python -m tools.predictor_validation_20261027.flow_capacity_transfer seal \
  --plan TARGET_PLAN --training-manifest TRAINING_CAPACITY_MANIFEST --output NEW_TRANSFER_JSON
python -m tools.predictor_validation_20261027.flow_capacity_transfer verify --sealed NEW_TRANSFER_JSON
python -m tools.predictor_validation_20261027.flow_capacity_transfer report \
  --sealed NEW_TRANSFER_JSON --raw INDEPENDENT_TARGET_RAW --output NEW_REPORT_JSON
```

The seal recognizes the source-bound native runner's periodic/task-offset/seeded-uniform input generator and reconstructs the complete input trace on CPU. Other sources require an explicit supported `gsdeploy.periodic_seeded_uniform_input_contract.v1` contract shared by the training and target plans. A matching domain without a matching generation contract and trace is insufficient. Unsupported or mismatched conditions reject sealing; they are not imputed.

The implemented v1 report deliberately has `acceptance_policy=null`, so service errors and original native stability remain descriptive, with target applicability and physical admission unknown. It does not add a post-hoc threshold or final eligibility policy. The provisional planning filter is explicitly named `provisional_transfer_planning_filter_passed` and always remains separate from original admission.

## Why a separate entry point is required

The original `profile_key` contains seed and horizon; `valid_profile` accepts only `measured` and `synthetic_fixture`. A training-condition mean does not become a measurement of a new seed because its numerical estimate is reused. Adding a new target key with `kind=measured` or labeling a transfer `synthetic_fixture` would misstate evidence. Keep the verified training catalogs unchanged and leave the original target-seed resolver's missing-profile result intact.

Introduce `flow_capacity_transfer.py` with explicit schema `gsdeploy.prospective_training_condition_capacity_transfer.v1`. Its result is a prospective statistical planning estimate, separate from the original resolver's measurement evidence. Do not monkey-patch a resolver or temporarily fabricate accepted profiles.

## Proposed CPU-only interfaces

`seal(target_plan_path, training_manifest_path, output_path)` writes a new exclusive artifact before target startup. `verify_transfer(path)` rechecks all provenance and arithmetic. `report(sealed_path, target_raw_paths, output_path)` is an independent validation addendum; it never updates the seal, catalog, model or estimates.

The sealing operation should:

1. Verify plan self hash, complete declared routes via the unchanged `_routing`, all source receipts and prospective role; reject any target raw/outcome as an input. Verify every selected training catalog using `flow_capacity_catalog.verify_catalog`, including its original raw provenance and service arithmetic.
2. Match a target candidate to exactly one full training condition. Require identical execution context, aliases, resource configuration, alias routing weights, offered rates, arrival process, H, warmup, drain and known input domain. Bind and compare the declared input-generation algorithm/distribution as well: equal domain alone does not establish equal sampling. A new seed and its prospectively generated request sequence are the only permitted changes in this first version. Never match just the family routing aggregate: balanced/skewed aliases can collide there.
3. Select an actual declared training plan/candidate and invoke unchanged `original_rho(training_plan, training_candidate, training_catalog)` to recover and cross-check its service vector, g(n), routing arithmetic and binding. Also read the independently verified audit estimates. Require exact agreement. Training service remains `mean(worker END−START) * original_g(n)/n`; it includes the declared bounded-drain observations and excludes all queue/admission wait.
4. Compute target offered routing with `_routing(target_plan, target_candidate)`. Reproduce the original formula directly using the frozen verified service estimates: `rho_m = sum_t offered_rps[t,m] * service_ms[m,t] / 1000 / original_g(n_m)`. Preserve `.8 + 1e-8`, original capacity factors, original family-average behavior and original safe factor. Bind original source hashes. Do not replace offered rates by finite counts/H, forecast busy time, measured target service, or a new alias threshold.
5. Store source training seeds, exact condition digest, target plan/trace/seed digest, numeric service/count/run means, normalized factors, sampled owned-memory peak/safety arithmetic, resulting rho and threshold decision, and all code/source receipts. Mark `kind=prospective_training_condition_transfer`, `target_outcomes_used=false`, `target_measured=false`, `physical_capacity_bound=false`, `uncertainty_calibrated=false`, `final_gain_eligible=false`. The created/source-freeze times are checked later against actual target startup.

Suggested per-candidate output has separate fields:

```
original_target_resolver: {status: UNKNOWN, reason: MISSING_COLOCATION_PROFILE}
transfer: {status: PROSPECTIVE_TRANSFER_ESTIMATE, ...numeric evidence...}
transferred_rho_threshold_satisfied: true | false | null
training_condition_match: true | false
target_applicability_validated: null
original_rho_admission_passed: null
original_capacity_and_stability_passed: null
final_gain_eligible: false
```

This preserves the frozen original API's honest result. A caller may explicitly use the predeclared transfer threshold as a *provisional planning filter*, with its evidence kind exposed. It must not silently feed that value into a field claiming original measured-profile qualification. Final eligibility needs a separately declared validation policy accepting this transfer evidence after an independent report; this adapter alone cannot manufacture that policy or retroactively pass a failed original gate.

## Independent validation and horizon boundary

The report checks the immutable target plan/source identities, seal before actual startup, actual full input/routes/context/resources, clean complete native cohort and telemetry. It compares frozen transferred service/rho predictions against independent target worker service and reports per-task/family relative error, per-run differences and sample counts. It calls the unchanged `native_stability` for the actual original stability classification, with unchanged thresholds. Actual busy fractions are diagnostics, not replacement rho estimates. Target measurements remain validation-only and cannot update this transfer artifact.

Set the service-transfer acceptance rule and number of independent target operating runs before measurement. With two training seeds, request-level errors do not justify a calibrated run-level capacity bound. An initial report can establish matched-seed empirical predictive error while retaining `physical_capacity_bound=false`; memory is an observed training sampled peak with one safety factor, not a guaranteed bound.

H=90 to H=600 must be rejected by the first-version exact-condition matcher. Same offered rps does not establish stationarity, steady clocks, equivalent thermal behavior, queue recovery, peak memory coverage or startup amortization over a different duration. A later cross-horizon version needs its own prospectively declared assumption and independent long-horizon calibration/validation; it must not borrow a 90-second seal as final 600-second qualification. Admission modes and resource-share changes similarly require their own matched training evidence.

## Meaningful implementation checks

Use CPU fixtures to verify new-seed/same-condition transfer without adding any catalog key; exact original rho arithmetic for n=2 and nonidentity g(n); distinct full alias-routing conditions; rejection of horizon/load/resource/input-generator mismatch; recursive raw/source tamper rejection; prospective timestamp enforcement in reporting; UNKNOWN on missing route service/support; no target outcome inputs; unchanged original resolver still missing the target key. No GPU is needed to implement these interfaces.
