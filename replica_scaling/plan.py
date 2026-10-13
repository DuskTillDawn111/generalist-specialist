"""Freeze tasks, loads, budget, source bytes and environments before any new worker."""
import statistics
import time
from candidate_pool_audit.provenance import environment_snapshot
from candidate_pool_audit.registry import registry
from candidate_pool_service_demand.prepare import model_files, targets, protected_files
from .common import ROOT, OUT, PLAN, read, write, reference, digest
from .telemetry import snapshot

MODELS = [('qwen', 'generalist', 'asr'), ('whisper-small', 'heavy_specialist', 'asr'), ('resnet50', 'light_specialist', 'image')]

def prepare():
    if PLAN.exists() or (OUT / 'plan_freeze.json').exists():
        raise RuntimeError('Plan already frozen; refuse to overwrite')
    catalog_path = ROOT / 'results/candidate_pool_v2_service_demand/updated_service_demand_catalog.json'
    catalog = read(catalog_path)
    if not catalog['frozen'] or catalog['completeness']['still_pending_paths']:
        raise RuntimeError('Complete frozen real service-demand catalog required')
    candidate_records = {m['id']: m for m in read(ROOT / 'results/candidate_pool_v2/candidate_models.json')['models']}
    environments = environment_snapshot()
    if environments != read(ROOT / 'results/candidate_pool_v2/environment_manifest.json')['environments']:
        raise RuntimeError('Frozen environment drift')
    launchers = registry()
    samples = targets('asr') + targets('image')
    pool = OUT / 'stage1_task_subset.jsonl'
    pool.parent.mkdir(parents=True, exist_ok=True)
    pool.write_text(''.join(__import__('json').dumps(row, ensure_ascii=False) + '\n' for row in samples))
    inputs = {str(pool): reference(pool)}
    for row in samples:
        for value in row['sample'].values():
            if isinstance(value, str) and value.startswith('/') and __import__('pathlib').Path(value).is_file():
                inputs[value] = reference(value)
    protected = {str(p.relative_to(ROOT)): reference(p) for p in protected_files()}
    for directory in ('results/candidate_pool_v2_service_demand', 'results/integer_replica_search_foundation'):
        protected.update({str(p.relative_to(ROOT)): reference(p) for p in (ROOT / directory).rglob('*') if p.is_file()})
    protected.update({str(p.relative_to(ROOT)): reference(p) for p in (ROOT / 'performance_model').rglob('*.py')})
    for name in ('common.py', 'plan.py', 'telemetry.py', 'runtime.py', 'campaign.py'):
        p = ROOT / 'replica_scaling' / name
        protected[str(p.relative_to(ROOT))] = reference(p)
    models = []
    for name, role, task in MODELS:
        entry = catalog['models'][name]
        if task not in entry['capabilities_verified'] or candidate_records[name]['task_compatibility'][task] != 'verified':
            raise RuntimeError('Unverified capability')
        s0 = entry['service_time_ms'][task]
        historical = entry['service_time_sources'][task]['source']['summaries']
        latencies = []
        for ref in historical:
            summary_path = __import__('pathlib').Path(ref['path'])
            raw_path = __import__('pathlib').Path(str(summary_path).removesuffix('.summary.json'))
            rows = [__import__('json').loads(line) for line in raw_path.read_text().splitlines()]
            assert [r['pool_id'] for r in rows] == [r['pool_id'] for r in targets(task)]
            latencies.extend(r['worker_latency_ms'] for r in rows)
        files = {str(p.relative_to(ROOT)): reference(p) for p in model_files(name)}
        frozen_files = read(ROOT / 'results/candidate_pool_v2/source_hashes.json')['files']
        for rel, ref in files.items():
            if rel in frozen_files and ref['sha256'] != frozen_files[rel]['sha256']:
                raise RuntimeError('Frozen model drift: ' + rel)
        adapter = reference(launchers[name]['script'])
        rel = str(launchers[name]['script'].relative_to(ROOT))
        if adapter['sha256'] != frozen_files[rel]['sha256']:
            raise RuntimeError('Frozen adapter drift')
        environment = environments[str(launchers[name]['python'])]
        models.append(dict(model_id=name, role=role, task=task, task_mix={task: 1.0},
            reason_selected={'qwen': 'Required Generalist; fixed ASR task is verified, autoregressive and has historical standalone saturation measurements.',
                             'whisper-small': 'User-preferred ASR specialist; much heavier service demand and residency than ResNet50, frozen native FP32 adapter.',
                             'resnet50': 'User-preferred light specialist; existing Imagenette closed-set image adapter.'}[name],
            service_demand_ms=s0, stage1_source=entry['service_time_sources'][task],
            historical_latency_ms=dict(count=len(latencies), mean=statistics.mean(latencies), sample_std=statistics.stdev(latencies), maximum=max(latencies)),
            loads={level: factor * 1000 / s0 for level, factor in [('low', .45), ('medium', .95), ('high', 2.2)]},
            nominal_capacity_rps=1000 / s0, measured_idle_vram_mib=entry['measured_idle_vram_mib'],
            precision=candidate_records[name]['precision'], source=candidate_records[name]['source'],
            source_revision=candidate_records[name]['source_revision'], model_files=files, model_manifest_sha256=digest(files),
            launcher={key: str(value) for key, value in launchers[name].items()}, adapter=adapter,
            environment_hash=environment['package_manifest_sha256'], executable_hash=environment['executable_sha256'],
            pool_ids=[r['pool_id'] for r in targets(task)], sample_subset_sha256=digest(targets(task))))
    gpu = snapshot()
    if gpu['processes'] or gpu['used_mib'] > 128 or gpu['total_mib'] != 49140:
        raise RuntimeError('GPU not idle or unexpected capacity')
    schedule = []
    for model in models:
        for level in ('low', 'medium', 'high'):
            for repetition, seed in enumerate((7101, 7202, 7303), 1):
                for count in ((1, 2) if repetition != 2 else (2, 1)):
                    schedule.append(dict(run_id=f'{model["model_id"]}_r{count}_{level}_rep{repetition}',
                        model_id=model['model_id'], replica_count=count, load_level=level,
                        repetition=repetition, random_seed=seed, offered_load_rps=model['loads'][level]))
    plan = dict(schema_version=1, module='7C-1', frozen=True, created_unix_ns=time.time_ns(),
        scope='Single-family 1-vs-2 independent CUDA subprocess scaling characterization; no deployment search.',
        gpu=gpu, models=models, replica_counts=[1, 2], repetitions=3, max_physical_runs=54, schedule=schedule,
        load_selection=dict(factors_of_standalone_capacity={'low': .45, 'medium': .95, 'high': 2.2},
            rationale='Low comfortably below historical standalone capacity; medium near queueing boundary; high exceeds both 1x and ideal 2x nominal capacity to reduce throughput censoring. Same loads for both counts; no post-result adjustment.',
            historical_subset='First 30 task rows from original mixed_formal_1200_6task; same as all three frozen Stage-1 repetitions. Seeded shuffle then cycle. ASR durations are not replaced by shorter smoke samples.'),
        input_pool=reference(pool), original_pool=reference(ROOT / 'benchmark/task_pools/mixed_formal_1200_6task.jsonl'),
        input_semantics='Frozen native adapters and full original sample payloads; identical prompts, preprocessing, generation limits and output settings.',
        service_catalog=reference(catalog_path),
        warmup=dict(priming='One task request per independent replica; excluded.', seconds=5,
                    offered_load='Fixed model low load for every condition; drain completely before measurement.'),
        measurement_seconds=30, drain_timeout_seconds=60, shutdown_timeout_seconds=15, startup_timeout_seconds=300,
        max_pending_per_replica=10000, routing='Equal deterministic smooth weighted round-robin over sorted model#i IDs; no routing optimizer.',
        arrival_process='deterministic_periodic', environment_variables=dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'),
        telemetry_interval_seconds=.5,
        protective_stop=dict(max_outstanding_requests=10000, max_outstanding_baseline_service_seconds=60,
            max_oldest_outstanding_age_seconds=20, action='Stop arrivals, bounded drain, preserve raw unstable outcome; skip remaining repetitions of that exact condition. No higher budgets, retries or loads.'),
        external_interruption='Separate execution outcome; retain evidence and no automatic retries. Never relabel unproven worker/harness failure as external interruption.',
        stability_thresholds=__import__('performance_model.local_calibration.stability_v2', fromlist=['DEFAULT_THRESHOLDS']).DEFAULT_THRESHOLDS,
        metrics_definitions=dict(primary_throughput='Successful cohort completions within measurement window / actual measurement seconds; cohort drain completion rate saved separately.',
            latency='Mean and P95 host scheduled-arrival-to-received-response E2E for successful cohort completions; completion ratio always alongside.',
            memory='NVML sum of owned CUDA PID residency post-warmup idle; device-wide and post-load idle also saved. Peak is observed 0.5 s sampled peak over workload plus drain.',
            throughput_gain='T2/T1 at same offered load and seed; efficiency=(T2/T1)/2; low-load gain is demand-censored.',
            memory_ratio='Post-warmup owned-PID idle MiB(2) / MiB(1); peak ratio separately.',
            sustainable='Observed tested loads only. Strict: all three valid repeats steady, full recovery and completion ratio >=.98. Recoverable congestion listed separately, not precise maximum capacity.'),
        interpretation=dict(clear_usable_gain_min=1.10, approximate_linear_gain_interval=[1.8, 2.2],
            rule='Capacity claims need comparable full windows, repeated gain beyond 10% and observed queueing/saturation in the 1-replica baseline. Low-load ratios alone are not capacity estimates. Incomplete/stopped pairs retain descriptive ratios with no sustainable-capacity claim.',
            gate='SUPPORTED: Generalist and at least one specialist have usable capacity gain; PARTIAL: gains differ or evidence incomplete; NOT SUPPORTED: no usable observed gains or pervasive same-GPU interference.'),
        module7b_formulation_unchanged=True)
    write(OUT / 'environment_manifest.json', dict(environments=environments, matches_frozen_v2=True))
    write(OUT / 'source_hashes_before.json', dict(protected_files=protected, inputs=inputs,
          models={m['model_id']: m['model_files'] for m in models}))
    write(PLAN, plan)
    write(OUT / 'plan_freeze.json', dict(plan_sha256=reference(PLAN)['sha256'], frozen_unix_ns=time.time_ns(),
          source_hashes_sha256=reference(OUT / 'source_hashes_before.json')['sha256']))
    print(__import__('json').dumps(dict(plan_sha256=reference(PLAN)['sha256'], runs=len(schedule),
          models=[dict(model=m['model_id'], task=m['task'], loads=m['loads']) for m in models])), flush=True)

if __name__ == '__main__':
    prepare()
