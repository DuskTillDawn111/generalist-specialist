# Admission2 resource and capacity training adapters

These new CPU entrypoints consume only the eight prospective primary jobs declared by `admission_training_v1`: seeds `628101/628102`, four `admission2_*` routes, H600, drain300, warmup0 and global max2 whole-request admission. They do not modify the frozen shared resource/capacity/validation modules or use shared fitted values, mechanism pilot outcomes or independent validation outcomes as numerical training sources.

## Data boundary and chronology

[admission_capacity_v1.py](../tools/predictor_validation_20261027/admission_capacity_v1.py) provides the common `load_campaign(campaign_path)` loader. It calls the new training mapper on each exact campaign job and checks its worker context, separately hashed controller, admission2 resource configuration, training split and complete native service-observation cohort. The mapper remains responsible for actual native START/PHASE/END, source provenance, full owned telemetry, shutdown, route/input and collection checks. The common loader does not call a validation mapper or relabel its outputs.

Both adapters expose `prepare(campaign_path, protocol_file)`, which freezes sources/policy without reading outcomes. On fitting, every original `startup_begin_ns` must be strictly later than the adapter protocol's `created_ns`. All eight raw files must exist before fitting; a missing job or late-frozen protocol fails before a fitted output directory is created. Complete-seed CV excludes the heldout seed from every numerical estimate.

## Resource outputs and applicability

[admission_resources_v1.py](../tools/predictor_validation_20261027/admission_resources_v1.py) exposes `fit(protocol_file, output_dir)` and `verify_manifest(manifest_file, recompute=True)`. It reuses the existing pure resource integration helper with explicitly supplied new admission evidence; that helper loads no shared result. Conditional power/memory statistics remain separate for each route and controller. Full training plus two whole-seed folds produce twelve resource documents.

The resource manifest has `full` and `folds` entries. Each entry contains:

- `route_resources[candidate_id]`: receipt of that route's resource document.
- `candidate_overrides[candidate_id]`: same-route retained startup joules mean and full startup-through-drain owned sampled peak maximum, with controller, raw sources and retained seeds.
- `training_seeds` and `omitted_seed`.

Power uses directly observed full family/task contexts and duration weighting within a route. Each context's memory statistic includes the full-history owned sampled trial peak once per retained run. These sampled statistics are empirical inputs, not physical memory bounds; the original safe factor remains the scoring caller's responsibility and is not applied by fitting. Unseen task contexts remain UNKNOWN. Resource CV uses actual heldout contexts solely for retrospective energy replay; it is not a prospective queue/energy validation.

Call `provider(document, public_plan, candidate)` before prediction. It checks the exact route, separate controller, complete condition, input-generation law and source receipts, then returns a task-conditional resource provider. A new predictor wrapper must select `entry['route_resources'][candidate['id']]` for every candidate and use this checked constructor. The generic frozen `_resources` constructor does not enforce the new route/controller contract; do not use it as a substitute.

## Capacity outputs and later independent qualification

The capacity module exposes `load_training(protocol_file)`, `estimate(public_plan, candidate, runs, original_catalog)`, `fit(protocol_file, output_dir)` and `verify_manifest(manifest_file, recompute=True)`. It filters exact controller and route before using the frozen input-weighted numerical estimator. Each positive-weight public target input must have matched training evidence; no input renormalization, other-route, shared or solo fallback is allowed.

Normalization stays `worker_mean * original_g(n) / resident_n`; `n` is the original resident count, never global max2. The original offered-rate family rho formula and threshold `0.8 + 1e-8` remain unchanged. The original catalog supplies capacity factors only; shared measured service is not used. `capacity-training-fit.json` contains eight full-training public reference predictions and eight whole-seed CV cases. Training error checks use the existing 20% limit and predicted-mean denominator, and are labelled diagnostics. Original admission and independent applicability remain UNKNOWN even when a training numerical check passes.

`seal(target_plan_file, training_protocol_file, output_file)` and `verify_seal(sealed_file)` reserve the later public target without reading target outcomes. The target must use the fixed disjoint seed `628201`, the four admission2 routes and `split='validation'`, `phase='predictor_validation'`, `kind='prospective_capacity_validation'`, H600/drain300/warmup0. Its controller must match the training identity. The seal freezes predictions, sources and the 20% predicted-denominator acceptance contract. This is the single independent role reserved by the main thread; it does not authorize replacement seeds or predictor service-law refitting.

`assess_independent_service(prediction, observations)` provides only the numerical mean-error adapter for a future strictly audited independent mapper. It deliberately reports that it has not audited raw provenance/chronology and grants no applicability or admission. A future controller-aware independent qualification entry must additionally check the pre-startup seal, disjoint target identity, full cohort, clean owned telemetry, original native stability and applicable capacity evidence. Family rho does not establish coupled global-permit capacity: full-horizon bounded drain and independent queue/P95 validation remain separate. This training implementation makes no max2 benefit or final gain claim.

## CPU validation

[test_admission_resource_capacity_v1.py](../tests/test_admission_resource_capacity_v1.py) uses synthetic mapped acquisition packets while running the real public native plan generator, condition/input-law validation, capacity arithmetic, resource integration, fitting and recomputation. It covers full/whole-seed outputs, route/controller isolation, duration weighting, full-history peak retention, UNKNOWN input/resource contexts, missing or foreign-role training evidence, late protocol rejection, independent public sealing and rehashed numerical tampering. These fixtures are not physical GPU measurements.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -m unittest tests.test_admission_resource_capacity_v1
```

Result: **14 tests passed**. No GPU workload or real-data fit was run by this task. Prepare real protocols after the acquisition mapper/campaign source is stable and before starting any training job.
