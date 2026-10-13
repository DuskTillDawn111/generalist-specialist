"""Validate real repetitions and publish an additive, immutable demand catalog."""
import json
import math
from pathlib import Path
from statistics import mean, stdev

from candidate_pool_audit.provenance import environment_snapshot, sha
from deployment_core.candidate_pool_v2 import load_candidate_pool_v2
from .prepare import ROOT, OUT, POOL, read, write, reference, targets


def validate_repetition(raw_path, model_id, task, expected_targets):
    raw_path = Path(raw_path)
    audit = read(str(raw_path) + '.validation.json')
    if (audit['status'] != 'measured' or audit['workers_started'] != 1 or
        audit['replica_count'] != 1 or not audit['standalone_only'] or
        not audit['load_succeeded'] or not audit['GPU_memory_released'] or
        audit['foreign_GPU_processes'] or audit.get('monitor_error') or
        audit['before_load']['processes'] or audit['shutdown'] != {'exit_code': 0, 'forced': False}):
        raise ValueError('Standalone load/isolation/unload validation failed')
    if any(s['capacity_mib'] != 49140 or any(p['pid'] != audit['pid'] for p in s['processes'])
           for s in audit['GPU_process_samples']):
        raise ValueError('Unexpected hardware capacity or concurrent GPU worker')
    if not any(s['target_process_mib'] is not None for s in audit['GPU_process_samples']):
        raise ValueError('No real resident CUDA process observation')
    if any(e['kind'] == 'ERROR' for e in audit['events']):
        raise ValueError('Worker inference error; capability revalidation required')
    ready = [e for e in audit['events'] if e['kind'] == 'READY']
    if len(ready) != 1:
        raise ValueError('Exactly one worker READY required')
    responses = [e['payload'] for e in audit['events'] if e['kind'] == 'RESULT']
    if len(responses) != 31 or len(audit['requests']) != 31:
        raise ValueError('Exactly one warmup plus 30 measured calls required')
    warm_request, warm_response = audit['requests'][0], responses[0]
    if (warm_request['request_id'] != -1 or warm_response['request_id'] != -1 or
        warm_request['sample'] != expected_targets[0]['sample'] or
        warm_request['pool_id'] != 'warmup:' + expected_targets[0]['pool_id']):
        raise ValueError('Historical first-target single warmup policy mismatch')
    for index, (request, expected) in enumerate(zip(audit['requests'][1:], expected_targets)):
        if (request['request_id'] != index or request['sample'] != expected['sample'] or
            request['pool_id'] != expected['pool_id'] or request['task_type'] != task or
            request['arrival_index'] != expected['arrival_index']):
            raise ValueError('Historical exact sample/preprocessing input mismatch')
    raw = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    summary = read(str(raw_path) + '.summary.json')
    if (len(raw) != 30 or summary['protocol'] != 'active_single_profile_v1' or
        summary['target_worker'] != model_id or summary['target_task'] != task or summary['target_count'] != 30 or
        [r['pool_id'] for r in raw] != [r['pool_id'] for r in expected_targets] or
        [r['request_id'] for r in raw] != list(range(30))):
        raise ValueError('Original Stage-1 summary/identity/sample count mismatch')
    latencies = []
    for row, response in zip(raw, responses[1:]):
        if row['task_type'] != task or row['profile_role'] != 'single_target':
            raise ValueError('Not a standalone target result')
        if any(row[key] != response[key] for key in response):
            raise ValueError('Raw result does not match captured worker response')
        value = row['worker_latency_ms']
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError('Nonpositive/nonfinite measured worker latency')
        if abs(value - (row['end_ns'] - row['start_ns']) / 1e6) > 1e-9:
            raise ValueError('Worker handler timer mismatch; E2E/queue time cannot substitute')
        if not isinstance(row.get('prediction'), str):
            raise ValueError('Task output interface invalid; capability revalidation required')
        if task in ('text', 'image', 'ocr', 'lid') and not isinstance(row.get('correct'), bool):
            raise ValueError('Task scoring schema invalid; capability revalidation required')
        if task == 'asr' and not all(key in row for key in ('word_errors', 'reference_words', 'wer')):
            raise ValueError('ASR scoring schema invalid; capability revalidation required')
        latencies.append(value)
    if not math.isclose(summary['mean_worker_latency_ms'], mean(latencies), rel_tol=0, abs_tol=1e-9):
        raise ValueError('Raw arithmetic mean disagrees with Stage-1 summary')
    return {'status': 'measured', 'mean_worker_latency_ms': summary['mean_worker_latency_ms'],
            'sample_std_ms': stdev(latencies), 'measurement_count': len(raw), 'worker_latency_values_ms': latencies,
            'raw': reference(raw_path), 'summary': reference(str(raw_path) + '.summary.json'),
            'validation': reference(str(raw_path) + '.validation.json'), 'warmup': warm_response,
            'pid': audit['pid'], 'GPU_uuid': audit['before_load']['gpu_uuid'],
            'GPU_name': audit['before_load']['gpu_name'], 'precision': ready[0]['payload'].get('precision')}


def aggregate_repetitions(repetitions):
    if len(repetitions) != 3 or any(r['status'] != 'measured' or r['measurement_count'] != 30 for r in repetitions):
        raise ValueError('Three valid independent 30-request repetitions required')
    weighted, count, values = 0., 0, []
    for rep in repetitions:
        weighted += rep['mean_worker_latency_ms'] * rep['measurement_count']
        count += rep['measurement_count']
        values.extend(rep['worker_latency_values_ms'])
    aggregate = weighted / count
    if not math.isfinite(aggregate) or aggregate <= 0:
        raise ValueError('Invalid aggregate; never fill zero, infinity or synthetic fallback')
    return {'aggregate_value_ms': aggregate, 'measurement_count': count, 'sample_std_ms': stdev(values),
            'repetition_means_ms': [r['mean_worker_latency_ms'] for r in repetitions],
            'repetition_mean_sample_std_ms': stdev([r['mean_worker_latency_ms'] for r in repetitions]),
            'aggregation': 'sample-count-weighted mean', 'unit': 'milliseconds'}


def completeness(records, services):
    required, available, pending = [], [], []
    workloads = read(ROOT / 'results/deployment_search_multiworkload/benchmark_workloads.json')['workloads']
    active = {k for w in workloads for k, value in w['arrival_rates_rps'].items() if value > 0}
    for name, record in sorted(records.items()):
        for task in sorted(record.supported_tasks & active):
            pair = {'model_id': name, 'task': task}
            required.append(pair)
            value = services.get(name, {}).get(task)
            (available if isinstance(value, (int, float)) and not isinstance(value, bool) and
             math.isfinite(value) and value > 0 else pending).append(pair)
    return {'required_paths': len(required), 'available_paths': len(available),
            'still_pending_paths': len(pending), 'pending_paths': pending,
            'completeness_ratio': len(available) / len(required), 'required_path_list': required}


def finalize():
    before = read(OUT / 'before_hashes.json')
    changed = [name for name, ref in before['files'].items() if sha(ROOT / name) != ref['sha256']]
    if changed:
        raise RuntimeError('Protected frozen input or Module 7B source changed: ' + str(changed))
    environments = environment_snapshot()
    if environments != read(OUT / 'environment_manifest.json')['environments']:
        raise RuntimeError('Frozen runtime environment changed during campaign')
    model_changes = [name for model in before['models'].values() for name, ref in model['files'].items()
                     if sha(ROOT / name) != ref['sha256']]
    if model_changes:
        raise RuntimeError('Frozen model checkpoint/tokenizer files changed')
    records, hardware = load_candidate_pool_v2()
    services = {m: dict(p.service_time_ms) for m, p in records.items()}
    sources = {m: {k: {'kind': 'existing_frozen_stage1', 'protocol': 'active_single_profile_v1',
                      'value_ms': value, 'source': records[m].sources['service_time'][k]}
                   for k, value in services[m].items() if value is not None} for m in services}
    result_paths = []
    for pending in read(OUT / 'pending_service_demand_paths.json')['paths']:
        name, task = pending['model_id'], pending['task']
        result = {'model_id': name, 'task': task, 'capability_status': 'verified',
                  'status': 'MEASUREMENT_FAILED', 'aggregate_value_ms': None, 'repetitions': [],
                  'capability_revalidation_required': False, 'provenance': pending}
        for rep in range(1, 4):
            path = OUT / f'raw_measurements/r{rep}/active_single_{name}_{task}_n30.jsonl'
            try:
                measured = validate_repetition(path, name, task, targets(task))
                measured['repetition'] = rep
                result['repetitions'].append(measured)
            except Exception as error:
                audit_path = Path(str(path) + '.validation.json')
                audit = read(audit_path) if audit_path.exists() else {}
                result['repetitions'].append({'repetition': rep, 'status': 'MEASUREMENT_FAILED',
                    'error': f'{type(error).__name__}: {error}', 'raw': str(path)})
                result['capability_revalidation_required'] |= bool(audit.get('capability_revalidation_required')) or 'schema' in str(error) or 'interface' in str(error)
        if all(r['status'] == 'measured' for r in result['repetitions']):
            result.update(aggregate_repetitions(result['repetitions']), status='measured')
            if services[name][task] is not None:
                raise RuntimeError('Attempt to replace an existing service demand')
            services[name][task] = result['aggregate_value_ms']
            sources[name][task] = {'kind': 'new_real_standalone_measurement', 'protocol': 'active_single_profile_v1',
                'value_ms': result['aggregate_value_ms'], 'measurement_count': result['measurement_count'],
                'aggregation': result['aggregation'], 'sample_std_ms': result['sample_std_ms'],
                'repetitions': [{k: r[k] for k in ('repetition', 'raw', 'summary', 'validation', 'mean_worker_latency_ms', 'sample_std_ms', 'measurement_count')}
                                for r in result['repetitions']], 'environment': pending['runtime_environment'],
                'source_revision': pending['model']['source_revision'], 'model_manifest_sha256': pending['model']['model_manifest_sha256'],
                'adapter': pending['adapter']}
        result_paths.append(result)
    old_values = before['existing_service_values_ms']
    modified_existing = [(m, k) for m, values in old_values.items() for k, v in values.items()
                         if v is not None and services[m][k] != v]
    if modified_existing:
        raise RuntimeError('Existing demands changed')
    gate = completeness(records, services)
    measured_paths = sum(p['status'] == 'measured' for p in result_paths)
    status = 'complete' if gate['still_pending_paths'] == 0 and measured_paths == 14 else 'pending'
    catalog = {'schema_version': 1, 'frozen': status == 'complete', 'status': status, 'unit': 'milliseconds',
        'protocol': 'active_single_profile_v1', 'aggregation': 'sample-count-weighted mean',
        'completeness': dict(gate, measured_paths=gate['available_paths'], new_measured_paths=measured_paths, existing_paths=11),
        'gpu_capacity_mib': hardware.gpu_memory_mib,
        'candidate_pool': reference(ROOT / 'results/candidate_pool_v2/candidate_models.json'),
        'memory_source': reference(ROOT / 'results/candidate_pool_v2/measured_vram.json'),
        'models': {m: {'role': p.role, 'capabilities_verified': sorted(p.supported_tasks),
                       'measured_idle_vram_mib': p.memory_mib, 'service_time_ms': services[m],
                       'service_time_sources': sources[m]} for m, p in records.items()},
        'existing_service_demands_modified': 0, 'new_paths_measured': measured_paths,
        'no_synthetic_fallback': True, 'real_multi_replica_workers_started': 0}
    catalog_path = OUT / 'updated_service_demand_catalog.json'
    if catalog_path.exists() and read(catalog_path).get('frozen') is True:
        if read(catalog_path) != catalog:
            raise RuntimeError('Do not replace a frozen completed service-demand catalog')
    else:
        write(catalog_path, catalog)
    write(OUT / 'standalone_service_demand_results.json', {'status': status, 'unit': 'milliseconds',
          'protocol': 'active_single_profile_v1', 'paths': result_paths})
    validation = {'status': 'passed' if status == 'complete' else 'pending', **gate,
                  'measured_paths': gate['available_paths'], 'new_measured_paths': measured_paths, 'existing_paths_preserved': 11,
                  'modified_existing_service_demands': 0, 'protected_files_unchanged': len(before['files']),
                  'models_unchanged': True, 'environments_unchanged': True, 'standalone_only': True,
                  'planned_independent_repetitions': 42,
                  'valid_independent_repetitions': sum(r['status'] == 'measured' for p in result_paths for r in p['repetitions']),
                  'expected_new_measured_requests': 1260,
                  'valid_new_measured_requests': sum(p.get('measurement_count', 0) for p in result_paths),
                  'real_multi_replica_workers_started': 0, 'catalog': reference(catalog_path),
                  'capability_revalidation_required': [p['model_id'] + '/' + p['task'] for p in result_paths if p['capability_revalidation_required']]}
    write(OUT / 'measurement_validation.json', validation)
    files = set((ROOT / 'candidate_pool_service_demand').glob('*.py'))
    files.update(OUT.rglob('*'))
    files = {p for p in files if p.is_file() and p.name != 'source_hashes.json'}
    write(OUT / 'source_hashes.json', {'before_hashes': reference(OUT / 'before_hashes.json'),
         'protected_files_unchanged': True, 'files': {str(p.relative_to(ROOT)): reference(p) for p in sorted(files)},
         'model_files': before['models']})
    print(json.dumps({'status': status, **gate, 'measured_paths': measured_paths,
                      'existing_demands_modified': 0}, ensure_ascii=False), flush=True)
    return catalog


if __name__ == '__main__':
    finalize()
