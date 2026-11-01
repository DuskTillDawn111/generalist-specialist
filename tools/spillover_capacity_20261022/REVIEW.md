# Closed-loop capacity diagnostic API review

Read-only source review, 2026-10-22. No GPU launched and no production or frozen code changed. These observations apply to `artifacts/static-study-gpu-20261022/execution-source-v5`.

## Identity and startup

1. Use the original execution-packet-v2 base catalog (`task_replica_proxy`), not the failed calibrated replay configuration. The diagnostic low-rate text-only route is transport/preflight scaffolding, not a measured open-loop workload or capacity claim. Idle deployed aliases are supported.
2. `gsdeploy.static_study.bind_native_job(packet, job, new_output_dir)` binds every deployed checkpoint, inventories, historical receipts, original quality report files, and unchanged quality document. Idle G2/OCR must remain in `job['plan']['deployment']`. Preserve the resulting native config, quality-evidence manifest, and worker registry.
3. `runtime.preflight.preflight(system, plan, original_runtime_inputs, workers, admission_manifest, require_gpu=True)` checks native inputs, plan constraints, single device, empty compute processes, checkpoint/report fingerprints and quality input-pool hash. It does not assert the intended GPU UUID: compare returned `gpu.uuid` explicitly to the frozen expected UUID.
4. The quality identity checker only covers positive routed model/task paths. Before priming, independently invoke `evidence.verify_quality_evidence` with an explicitly labelled *identity coverage* object whose deployment contains all resident families and whose routing names every requested pair: text->text/G2; image->G2; ASR->G2; OCR->G2/OCR. This checker does not require route conservation; this object must never be confused with the actual diagnostic route.
5. `runtime.runner.registry_for(system, aliases, 'native', workers)` produces alias launch entries. `DeploymentRuntime.start()` only checks READY task coverage. Explicitly compare each READY payload's worker slot, model name, exact task set, spec SHA256 and PID to the corresponding frozen spec/owned process. Snapshot and verify all source, spec, protocol, classmap, checkpoint/adapter, input and reference bindings again before execution.

## Priming and resident state

Frozen `runtime.runner.run(..., priming_policy='largest_three_per_task')` only primes supported tasks with positive workload. A text-only diagnostic would omit G2 image/OCR/ASR and idle OCR priming. It therefore does not reproduce the previously frozen four-task resident state automatically.

Construct `SamplePool(original_runtime_inputs, ['text','image','ocr','asr'], seed)` and use `largest_inputs(pool, task, 3)` without consuming measurement cursors. Explicitly prime every resident alias on its supported/executable tasks in a fixed, packet-bound alias/task order. G2 receives 12 requests; each of two text aliases receives 3; OCR receives 3: **21 priming requests**. Prefer one outstanding priming request at a time and preserve each native response and outcome; this ordering is a declared diagnostic subprotocol, not a claim of byte-identical former concurrent priming. Preserve any subsequent warmup separately. Do not add/reset model components or reclaim caches beyond unchanged worker behavior.

The G2 frozen execution enables retired audio output, SDPA and existing cache reclamation. Do not undo or alter these settings. Full task priming retains the executable input components and avoids a text-only initialization context being silently substituted for the intended mixed-task resident context.

## Request and quality validation

The measured diagnostic gives every active alias the same 32 text inputs x 3 repetitions using independent cursors, at most one outstanding request per alias. That differs from the formal global SamplePool/SWRR sequence. `quality.measure_quality` assumes the formal global task sequence and should **not** receive this interleaved diagnostic cohort.

For each successful completed request, verify request/pool/task/alias/family against the frozen packet and response, require `success is True`, then normalize `result['task'] = response.get('task', response.get('task_type'))` and call:

```python
score = frozen_tasks.score_result(quality['references'][task][pool_id], result)
error_numerator, reference_denominator = frozen_quality.sample_totals(task, score)
```

References stay evaluator-side and never enter worker payloads. For OCR priming quality only, apply the same frozen `page_ocr_text` normalization as `measure_quality`. Preserve raw outputs, EM/F1, per-alias/input/repetition variability and missing outcomes. The diagnostic does not produce six-term utility, a route ranking, or formal stability qualification: `U=None`.

## Interpretation limits

Report handler service, full host timestamps, completed cohorts, errors/cleanup, actual memory and energy separately. Completed does not imply resource eligible. C runs must report observed overlap and each input's coverage before/after other aliases finish. Handler overlap is wallclock exposure, not GPU utilization or an allowed predictive feature. Two repeated blocks diagnose repeatability; their means do not establish a maximum sustainable open-loop arrival rate.
