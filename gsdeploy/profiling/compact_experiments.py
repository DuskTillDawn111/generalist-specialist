"""Prepare immutable compact-study jobs on CPU; execution requires --execute.

Measurement-only workers can collect admission evidence without fabricating an
admitted runtime catalog. The study catalog remains null until evidence passes.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from .evaluation import read_rows
from .prepare import digest, write
from .runner import save


TASKS = ('text', 'image', 'ocr', 'asr')
MODULE = 'gsdeploy.profiling.compact_experiments'


def runtime_samples(rows):
    return [{'pool_id': row['id'], 'task_type': row['task'], 'sample': row} for row in rows]


def checked_manifest(path):
    data = json.loads(Path(path).read_text())
    for task in TASKS:
        entry = data['tasks'][task]
        if digest(entry['path']) != entry['sha256']:
            raise ValueError('Changed frozen task manifest: ' + task)
    return data


def prepare(study_path, inventory_path, environments_path, execution_path, classmap_path, output):
    from ..backends.compact import CompactAdapter
    study = json.loads(Path(study_path).read_text())
    envs = json.loads(Path(environments_path).read_text())
    execution = json.loads(Path(execution_path).read_text())
    if study.get('execution_configuration') and Path(study['execution_configuration']).resolve() != Path(execution_path).resolve():
        raise ValueError('Study and execution configuration variants do not match')
    memory_required = study.get('memory_validation', {}).get('required', False)
    inventory = json.loads(Path(inventory_path).read_text())
    protocol = Path(study['task_adapter_protocol']['path']).resolve()
    if digest(protocol) != study['task_adapter_protocol']['sha256']:
        raise ValueError('Changed study task protocol')
    adapter = CompactAdapter(json.loads(protocol.read_text()), json.loads(Path(classmap_path).read_text()))
    quality = checked_manifest(study['quality_preparation']['manifest'])
    calibration = checked_manifest(study['calibration_preparation']['manifest'])
    # Validate native call blueprints before creating a packet or starting a GPU.
    # This catches unsupported prompts/decoding instead of discovering them in warmup.
    calibration_rows = {task: read_rows(calibration['tasks'][task]['path']) for task in TASKS}
    for candidate in study['generalists'] + study['specialists']:
        for task in adapter.supported_tasks(candidate):
            for sample in calibration_rows[task]:
                adapter.build(candidate, sample)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    jobs, workers = [], {}
    for candidate in study['generalists'] + study['specialists']:
        slot = candidate['slot']
        tasks = adapter.supported_tasks(candidate)
        env = envs['models'][candidate['model']]['python']
        if not Path(env).is_file():
            raise ValueError('Missing isolated interpreter: ' + env)
        samples = [r for task in tasks for r in calibration_rows[task]]
        prime = [r for task in tasks for r in calibration_rows[task][:3]]
        quality_rows = [r for task in tasks for r in read_rows(quality['tasks'][task]['path'])]
        resources = {'checkpoint': inventory['models'][slot], 'protocol': str(protocol),
                     'classmap': str(Path(classmap_path).resolve()),
                     'request_output': str(output / 'request-temporary' / slot)}
        spec = {'id': slot, 'kind': 'compact', 'adapter': 'compact', 'candidate': candidate,
                'execution': execution['models'][slot], 'resources': resources}
        spec_path = output / 'workers' / (slot + '.json')
        save(spec_path, spec)
        workers[slot] = {'python': env, 'spec': str(spec_path), 'spec_sha256': digest(spec_path), 'tasks': list(tasks)}
        phases = [('smoke', prime, 1, 1)]
        if memory_required:
            phases.append(('memory', samples, 1, 1))
        phases.append(('quality', quality_rows, 1, 1))
        for phase, rows, rounds, replicas in phases + [
                ('profile', samples, 3, n) for n in (1, 2, 3)]:
            job = {'id': f'{phase}-{slot}-n{replicas}', 'phase': phase, 'slot': slot,
                   'python': env, 'spec': str(spec_path), 'samples': rows, 'warmup_samples': prime,
                   'replicas': replicas, 'rounds': rounds, 'exact_cohort': phase != 'profile',
                   'minimum_seconds': 30, 'minimum_requests': 64,
                   'requires': [] if phase == 'smoke' else [f'smoke-{slot}-n1']}
            if replicas > 1:
                job['requires'].append(f'profile-{slot}-n1')
            if memory_required and phase not in {'smoke', 'memory'}:
                job['requires'].append(f'memory-{slot}-n1')
            jobs.append(job)
    rates = {'text': .1, 'image': .2, 'ocr': .02, 'asr': .05}
    # Seeded matched traces, identical task sample pool/order for every candidate.
    for slot, worker in workers.items():
        for task in worker['tasks']:
            pool = output / 'pools' / (task + '.jsonl')
            pool.parent.mkdir(exist_ok=True)
            if not pool.exists():
                write(pool, runtime_samples(read_rows(calibration['tasks'][task]['path'])))
            for seed in (42, 43, 44):
                jobs.append({'id': f'cost-{slot}-{task}-seed{seed}', 'phase': 'matched_cost',
                             'deployment': {slot: 1}, 'routing': {task: {slot: rates[task]}},
                             'rates': {task: rates[task]}, 'duration_seconds': 600,
                             'startup_amortization_seconds': study['utility']['startup_amortization_seconds'],
                             'seed': seed, 'sample_pool': str(pool), 'sample_pool_sha256': digest(pool),
                             'input_manifest_sha256': calibration['tasks'][task]['sha256'],
                             'requires': [f'smoke-{slot}-n1', f'quality-{slot}-n1', f'profile-{slot}-n1']})
    for job in jobs:
        save(output / 'jobs' / (job['id'] + '.json'), job)
    manifest = {'schema': 'gsdeploy.compact_jobs.v1', 'status': 'PREPARED_GPU_DISABLED',
                'gpu_started': False, 'utility': None, 'runtime_catalog': None,
                'workers': workers, 'jobs': [{'id': j['id'], 'phase': j['phase'],
                                             'path': str(output / 'jobs' / (j['id'] + '.json')),
                                             'sha256': digest(output / 'jobs' / (j['id'] + '.json'))} for j in jobs],
                'sources': {str(Path(p).resolve()): digest(p) for p in [study_path, inventory_path, environments_path,
                                                                      execution_path, classmap_path, protocol]},
                'code_hashes': {str(p.resolve()): digest(p) for p in [*Path('gsdeploy').rglob('*.py'),
                                                                    Path('scripts/run_compact_gpu_queue.py')]},
                'study': str(Path(study_path).resolve()),
                'memory_validation': memory_required,
                'notes': ['No automatic queue or GPU launch.',
                          'n2/n3 require successful n1 and measured memory feasibility; failures are retained.',
                          'Cost jobs must all be complete/stable/matched; closed-loop profiles cannot substitute.',
                          'CPU preparation does not assert that any candidate satisfies admission.']}
    save(output / 'experiment-manifest.json', manifest)
    return manifest


class BoardGuard:
    def __init__(self, board, runtime, budget, safe_factor):
        self.board, self.runtime, self.budget, self.factor = board, runtime, budget, safe_factor
    def check(self):
        value = self.board.sample()
        if value is None or self.board.errors:
            raise RuntimeError('Missing board telemetry')
        owned = {w.process.pid for w in self.runtime.workers.values()}
        if any(p['pid'] not in owned for p in value['processes']):
            raise RuntimeError('Foreign GPU process invalidates board energy')
        if value['used_mib'] * self.factor > self.budget:
            raise RuntimeError('Safety-adjusted board memory exceeds study budget')


def prepare_plan(manifest_path, plan_path, output):
    """CPU-validate an explicit mixed/replicated plan for measurement, without admission."""
    from ..runtime.schema import MeasurementConfig
    from ..trace import expand_routes
    manifest = json.loads(Path(manifest_path).read_text())
    plan = json.loads(Path(plan_path).read_text())
    aliases, routes = expand_routes(plan['deployment'], plan['routing'])
    config = MeasurementConfig({a: 1 for a in aliases}, plan['rates'], routes,
                               measurement_seconds=plan['duration_seconds'], random_seed=plan['seed'],
                               idle_residency=plan.get('idle_residency',False))
    config.validate({a: manifest['workers'][s]['tasks'] for a, s in aliases.items()})
    study = json.loads(Path(manifest['study']).read_text())
    calibration = checked_manifest(study['calibration_preparation']['manifest'])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = [r for t in plan['rates'] for r in read_rows(calibration['tasks'][t]['path'])]
    pool = write(output / 'samples.jsonl', runtime_samples(rows))
    job = {'id': 'colocation', 'phase': 'colocation', **plan, 'sample_pool': pool['path'],
           'startup_amortization_seconds': study['utility']['startup_amortization_seconds'],
           'input_manifest_sha256': pool['sha256'], 'sample_pool_sha256': pool['sha256'], 'requires': []}
    save(output / 'colocation.json', job)
    manifest['jobs'] = [{'id': 'colocation', 'phase': 'colocation', 'path': str(output / 'colocation.json'),
                         'sha256': digest(output / 'colocation.json')}]
    manifest['sources'].update({str(Path(plan_path).resolve()): digest(plan_path)})
    manifest['admission_evidence'] = {'manifest':str(Path(manifest_path).resolve()),
        'manifest_sha256':digest(manifest_path),
        'results':str(Path(plan['pool_results']).resolve()) if plan.get('pool_results') else None}
    manifest['sources'][str(Path(manifest_path).resolve())] = digest(manifest_path)
    manifest['notes'].append('System execution revalidates the original complete pool evidence; missing pool_results cannot run on GPU.')
    save(output / 'experiment-manifest.json', manifest)
    return manifest


def measure_plan(job, manifest, output):
    """Real open-loop cohort with queues, bounded drain, whole-board power and startup.

    This path deliberately does not call the admitted deployment API: it collects
    the evidence that API requires. It returns no utility and grants no admission.
    """
    from ..runtime.deployment import DeploymentRuntime
    from ..runtime.schema import MeasurementConfig
    from ..runtime.runner import window, record
    from ..runtime.workload import SamplePool
    from ..trace import expand_routes
    from ..stability import reconstruct, classify, DEFAULT_THRESHOLDS
    from .telemetry import BoardMonitor
    study = json.loads(Path(manifest['study']).read_text())
    if digest(job['sample_pool']) != job['sample_pool_sha256']:
        raise ValueError('Changed runtime input pool')
    aliases, routes = expand_routes(job['deployment'], job['routing'])
    registry = {}
    for alias, slot in aliases.items():
        w = manifest['workers'][slot]
        from ..backends.compact_memory import allocator_environment
        worker_execution = json.loads(Path(w['spec']).read_text())['execution']
        registry[alias] = {'command': [w['python'], '-m', 'gsdeploy.backends.compact_worker',
                                       '--spec', w['spec'], '--execute'],
                           'cwd': str(Path(__file__).resolve().parents[2]),
                           'env': {'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '1',
                                   'PYTHONPATH': str(Path(__file__).resolve().parents[2]),
                                   **allocator_environment(worker_execution)}}
    config = MeasurementConfig({a: 1 for a in aliases}, job['rates'], routes, warmup_seconds=0,
                               measurement_seconds=job['duration_seconds'], random_seed=job['seed'],
                               startup_timeout_seconds=900, drain_timeout_seconds=300,
                               idle_residency=job.get('idle_residency',False))
    config.validate({a: manifest['workers'][s]['tasks'] for a, s in aliases.items()})
    runtime = DeploymentRuntime(config, registry)
    board = BoardMonitor()
    guard = BoardGuard(board, runtime, study['gpu_memory_mib'], study['safe_factor'])
    raw = {'kind': 'measured', 'scope': 'whole_gpu_board', 'workload': 'idle_residency' if config.idle_residency else 'matched_open_loop',
           'status': 'starting', 'errors': [], 'request_records': [], 'priming_request_records': [],
           'warmup_request_records': [], 'utility': None, 'admission_granted': False,
           'transport_config': config.as_dict(), 'job': job,
           'gpu_uuid': board.nv.nvmlDeviceGetUUID(board.handle),
           'power_limit_watts': board.nv.nvmlDeviceGetPowerManagementLimit(board.handle) / 1000,
           'duration_seconds': job['duration_seconds'],
           'startup_amortization_seconds': job['startup_amortization_seconds'],
           'energy_method': 'nvml_board_power_integral',
           'protocol_sha256': study['task_adapter_protocol']['sha256'],
           'input_manifest_sha256': job['input_manifest_sha256']}
    begin = before = after = None
    board.start()
    try:
        guard.check()
        begin = board.sample()
        runtime.start()
        pool = SamplePool(job['sample_pool'], list(job['rates']), job['seed'])
        for task, row in routes.items():
            for alias, rate in row.items():
                if rate <= 0:
                    continue
                item = pool.next(task)
                record0 = record('priming', len(raw['priming_request_records']), task, alias, item, time.perf_counter_ns())
                raw['priming_request_records'].append(record0)
                runtime.enqueue(record0, item['sample'])
        runtime.drain(raw['priming_request_records'], 300)
        if any(r['status'] != 'completed' for r in raw['priming_request_records']):
            raise RuntimeError('Priming failed')
        guard.check()
        before = board.sample()
        raw['resident_mib'] = before['used_mib']
        raw['startup_energy'] = board.window(begin, before)
        raw['measurement_window'] = {}
        window(runtime, config, job['sample_pool'], 'measurement', job['duration_seconds'],
               raw['request_records'], guard, raw['measurement_window'])
        after = board.sample()
        guard.check()
        raw['status'] = 'completed' if all(r['status'] == 'completed' for r in raw['request_records']) else 'completed_with_failures'
    except Exception as error:
        import traceback
        raw.update(status='failed', errors=[str(error)], traceback=traceback.format_exc())
    finally:
        runtime.close()
        raw['errors'].extend(runtime.errors)
        raw['shutdown'] = runtime.shutdown_report
        board.close()
        raw['board_power_samples'] = board.samples
        raw['board_errors'] = board.errors
    if before is not None and after is not None and raw.get('measurement_window'):
        meta = raw['measurement_window']
        run_energy = board.window({'time_ns': meta['start_time_ns'], 'counter_joules': None},
                                  {'time_ns': meta['end_time_ns'], 'counter_joules': None})
        # Fixed H counts board energy over arrivals, not variable cohort drain.
        raw['run_energy'] = run_energy
        raw['drain_energy'] = board.window({'time_ns': meta['end_time_ns'], 'counter_joules': None}, after)
        raw['trace_sha256'] = meta['trace_sha256']
        raw['coverage_valid'] = run_energy['coverage_valid'] and raw['startup_energy']['coverage_valid']
        raw['effective_board_watts'] = (run_energy['joules'] / job['duration_seconds'] +
                                         raw['startup_energy']['joules'] / job['startup_amortization_seconds']) if raw['coverage_valid'] else None
        rows = raw['request_records']
        raw['complete_cohort'] = (config.idle_residency or bool(rows)) and all(r['status'] == 'completed' for r in rows)
        from ..metrics import summarize, percentile
        raw['task_metrics'],raw['system_metrics'],raw['model_metrics'] = summarize(rows,config,meta)
        latency = [(r['completion_time_ns'] - r['scheduled_time_ns']) / 1e6 for r in rows if r['status'] == 'completed']
        raw['mean_latency_ms'] = sum(latency) / len(latency) if latency else None
        raw['p95_latency_ms'] = percentile(latency,95)
        raw['p99_latency_ms'] = percentile(latency,99)
        # Observed service is descriptive; no fake catalog or capacity extrapolation.
        service = {alias: {} for alias in aliases}
        for r in rows:
            if r.get('worker_latency_ms') is not None:
                service[r['model']].setdefault(r['task'], []).append(r['worker_latency_ms'])
        measured = {a: SimpleNamespace(service_time_ms={t: sum(v) / len(v) for t, v in tasks.items()}) for a, tasks in service.items()}
        raw['stability_evidence'] = reconstruct(raw, measured, DEFAULT_THRESHOLDS)
        raw['stability'] = classify(raw['stability_evidence'], DEFAULT_THRESHOLDS)
        if config.idle_residency:
            if rows:
                raise ValueError('Idle residency unexpectedly dispatched requests')
            raw['idle_definition'] = 'Native model instances remain loaded; no priming, warmup or measurement requests.'
            raw['stable'] = (not raw['errors'] and not raw['board_errors'] and raw['status'] == 'completed'
                             and not raw['stability_evidence']['unclean_shutdown'])
            raw['stability'] = {'stability_state_v2':'idle_residency', 'request_metrics_applicable':False}
        else:
            raw['stable'] = raw['complete_cohort'] and raw['stability']['stability_state_v2'] != 'unstable'
        raw['peak_board_used_mib'] = max(s['used_mib'] for s in board.samples)
        if len(job['rates']) == 1:
            raw['task'] = next(iter(job['rates']))
    save(output / 'measurement.json', raw)
    return raw


def verify_system_pool_admission(manifest, output):
    """Revalidate the complete parent pool; a reduced plan has no memory jobs."""
    binding = manifest.get('admission_evidence') or {}
    if not binding.get('manifest') or not binding.get('results'):
        raise ValueError('System experiment requires explicit complete pool evidence')
    parent_path = Path(binding['manifest'])
    if digest(parent_path) != binding.get('manifest_sha256'):
        raise ValueError('Changed system pool evidence manifest')
    parent = json.loads(parent_path.read_text())
    if parent.get('diagnostic_only') or parent.get('workers') != manifest['workers']:
        raise ValueError('System workers differ from the independently evaluated pool')
    for path, expected in {**parent['sources'], **parent['code_hashes']}.items():
        if digest(path) != expected:
            raise ValueError('Frozen pool source changed: ' + path)
    results = Path(binding['results'])
    if not parent.get('memory_validation'):
        raise ValueError('System pool requires complete calibration memory validation')
    verify_memory_screen(parent_path, parent, results)
    calibration = calibration_quality_screen(parent_path, parent, results)
    if not calibration['passed']:
        raise ValueError('System pool complete calibration quality rejected')
    report = assess(parent_path, results, Path(output).parent / (Path(output).name + '-pool-revalidation.json'))
    if not report.get('all_requirements_passed'):
        raise ValueError('Independent quality or paired quality/cost rejected the system pool')
    return report


def execute_job(manifest_path, job_id, output):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    manifest = json.loads(manifest_path.read_text())
    for path, expected in {**manifest['sources'], **manifest['code_hashes']}.items():
        if digest(path) != expected:
            raise ValueError('Preparation changed; regenerate the experiment manifest: ' + path)
    for worker in manifest['workers'].values():
        if digest(worker['spec']) != worker['spec_sha256']:
            raise ValueError('Prepared worker specification changed')
    entry = next(j for j in manifest['jobs'] if j['id'] == job_id)
    if digest(entry['path']) != entry['sha256']:
        raise ValueError('Job fingerprint changed')
    job = json.loads(Path(entry['path']).read_text())
    if job['phase'] != 'diagnostic':
        for worker in manifest['workers'].values():
            worker_spec = json.loads(Path(worker['spec']).read_text())
            worker_policy = json.loads(Path(worker_spec['resources']['protocol']).read_text())
            if worker_policy.get('diagnostic_only'):
                raise ValueError('Diagnostic protocols can execute only diagnostic jobs; no admission/performance evidence')
    for required in job['requires']:
        receipt = output.parent / required / 'receipt.json'
        evidence = json.loads(receipt.read_text()) if receipt.is_file() else {}
        if evidence.get('status') != 'COMPLETE' or evidence.get('manifest_sha256') != digest(manifest_path):
            raise ValueError('Missing successful prerequisite: ' + required)
    if job['phase'] == 'colocation':
        verify_system_pool_admission(manifest, output)
    if manifest.get('memory_validation') and job['phase'] not in {'smoke', 'memory', 'diagnostic', 'colocation'}:
        verify_memory_screen(manifest_path, manifest, output.parent)
        calibration = calibration_quality_screen(manifest_path, manifest, output.parent)
        if not calibration['passed']:
            raise ValueError('Full calibration quality must pass before independent quality or performance jobs')
    output.mkdir(parents=True, exist_ok=False)
    if job['phase'] in {'matched_cost', 'colocation'}:
        result = measure_plan(job, manifest, output)
        success = result['status'] == 'completed' and result.get('stable') and result.get('coverage_valid')
    else:
        # The controller launches the recorded interpreter, never its own arbitrary environment.
        command = [job['python'], '-m', MODULE, 'worker-profile', '--job', entry['path'], '--output', str(output)]
        command.append('--execute')
        from ..backends.compact_memory import allocator_environment
        execution = json.loads(Path(job['spec']).read_text())['execution']
        success = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                                 env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '1',
                                      **allocator_environment(execution)}).returncode == 0
    receipt = {'status': 'COMPLETE' if success else 'FAILED', 'job': job_id,
               'manifest_sha256': digest(manifest_path), 'job_sha256': entry['sha256'], 'utility': None}
    save(output / 'receipt.json', receipt)
    return receipt


def profile_job(job_path, output):
    from .runner import run_profile
    job = json.loads(Path(job_path).read_text())
    spec = json.loads(Path(job['spec']).read_text())
    policy = json.loads(Path(spec['resources']['protocol']).read_text())
    if policy.get('diagnostic_only') and job['phase'] != 'diagnostic':
        raise ValueError('Diagnostic protocol cannot produce formal profiling or quality evidence')
    from ..backends.compact_memory import allocator_environment
    if any(os.environ.get(k) != v for k, v in allocator_environment(spec['execution']).items()):
        raise ValueError('Frozen allocator environment was not applied before the worker started')
    result = run_profile(spec, spec['resources'], job['samples'], job['replicas'], output,
                         rounds=job['rounds'], minimum_seconds=job['minimum_seconds'],
                         minimum_requests=job['minimum_requests'], warmup_samples=job['warmup_samples'],
                         exact_cohort=job['exact_cohort'])
    if job['phase'] in {'smoke', 'memory'} and result['status'] == 'COMPLETE' and any(
            row['failed_requests'] for row in result['rounds']):
        result.update(status='FAILED', error='Invalid completion in calibration memory validation',
                      failure_stage='output_validation', failure_kind='INVALID_COMPLETION')
        save(Path(output) / 'profile.json', result)
    if job['phase'] == 'quality' and result['status'] == 'COMPLETE':
        from ..tasks import summarize_quality
        from .admission import quality_gate
        policy = json.loads(Path(spec['resources']['protocol']).read_text())
        quality = {}
        for row in result['rounds']:
            records = read_rows(row['raw_records'])
            cohort = [s for s in job['samples'] if s['task'] == row['task']]
            scores = summarize_quality(cohort, records)
            quality[row['task']] = {'summary': scores, 'gate': quality_gate(policy, row['task'], scores)}
        save(Path(output) / 'quality.json', quality)
    return result


def verify_memory_screen(manifest_path, manifest, results):
    """A direct CLI run must obey the same full-calibration memory barrier."""
    from .memory_gate import screen_pool_memory
    results = Path(results)
    path = results / 'memory-screen.json'
    report = json.loads(path.read_text()) if path.is_file() else {}
    if report.get('manifest_sha256') != digest(manifest_path) or not report.get('passed'):
        raise ValueError('Missing successful full-calibration pool memory screen')
    profiles = {}
    for slot in manifest['workers']:
        entry = next(j for j in manifest['jobs'] if j['id'] == f'memory-{slot}-n1')
        if digest(entry['path']) != entry['sha256']:
            raise ValueError('Changed full-calibration job: ' + slot)
        job = json.loads(Path(entry['path']).read_text())
        profile_path = results / f'memory-{slot}-n1' / 'profile.json'
        receipt_path = profile_path.parent / 'receipt.json'
        receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
        if (receipt.get('status') != 'COMPLETE' or receipt.get('manifest_sha256') != digest(manifest_path)
                or receipt.get('job_sha256') != entry['sha256']
                or not profile_path.is_file()
                or report.get('profile_sha256', {}).get(slot) != digest(profile_path)):
            raise ValueError('Missing or changed calibration memory evidence: ' + slot)
        profile = json.loads(profile_path.read_text())
        spec = json.loads(Path(manifest['workers'][slot]['spec']).read_text())
        expected_counts = Counter(s['task'] for s in job['samples'])
        rows = profile.get('rounds', [])
        expected_hash = hashlib.sha256(json.dumps(job['samples'], sort_keys=True).encode()).hexdigest()
        if (not job.get('exact_cohort') or job['rounds'] != 1 or job['replicas'] != 1
                or profile.get('model') != spec or profile.get('manifest_sha256') != expected_hash
                or len(rows) != len(expected_counts) or {r['task'] for r in rows} != set(expected_counts)
                or any(r.get('repetition') != 0 or r.get('completed_requests') != expected_counts[r['task']]
                       or r.get('successful_requests') != expected_counts[r['task']]
                       or r.get('failed_requests') != 0 for r in rows)):
            raise ValueError('Incomplete or mismatched full-calibration cohort: ' + slot)
        profiles[slot] = profile
    study = json.loads(Path(manifest['study']).read_text())
    if not screen_pool_memory(study, profiles)['passed']:
        raise ValueError('Full-calibration memory constraints are not satisfied')


def calibration_quality_screen(manifest_path, manifest, results):
    """Rebuild full calibration scores from current-identity raw outputs.

    This is early rejection only. It never substitutes for independent quality
    or the twelve paired specialist comparisons.
    """
    from ..backends.compact import CompactAdapter, fingerprint
    from .admission import quality_gate
    from ..tasks import summarize_quality
    results = Path(results)
    manifest_hash = digest(manifest_path)
    gates, missing, hashes = {}, [], {}
    for slot, worker in manifest['workers'].items():
        entries = [j for j in manifest['jobs'] if j['phase'] == 'memory'
                   and json.loads(Path(j['path']).read_text())['slot'] == slot]
        if len(entries) != 1:
            missing.append(slot + ':missing_unique_memory_job')
            continue
        entry = entries[0]
        if digest(entry['path']) != entry['sha256'] or digest(worker['spec']) != worker['spec_sha256']:
            raise ValueError('Calibration job/spec identity changed')
        job = json.loads(Path(entry['path']).read_text())
        path = results / entry['id'] / 'profile.json'
        receipt_path = path.with_name('receipt.json')
        if not path.exists() or not receipt_path.exists():
            missing.append(slot + ':missing_calibration_collection')
            continue
        profile, receipt = json.loads(path.read_text()), json.loads(receipt_path.read_text())
        if (profile.get('status') != 'COMPLETE' or receipt.get('status') != 'COMPLETE'
                or receipt.get('manifest_sha256') != manifest_hash
                or receipt.get('job_sha256') != entry['sha256']):
            missing.append(slot + ':failed_or_mismatched_calibration_collection')
            continue
        hashes[slot] = digest(path)
        spec = json.loads(Path(worker['spec']).read_text())
        if profile['model']['candidate'] != spec['candidate'] or profile['model']['execution'] != spec['execution']:
            raise ValueError('Calibration model/execution identity changed')
        policy = json.loads(Path(spec['resources']['protocol']).read_text())
        if policy.get('diagnostic_only'):
            raise ValueError('Diagnostic protocols cannot pass formal calibration screening')
        adapter = CompactAdapter(policy, json.loads(Path(spec['resources']['classmap']).read_text()))
        for task in worker['tasks']:
            samples = [s for s in job['samples'] if s['task'] == task]
            rounds = [r for r in profile['rounds'] if r['task'] == task]
            if len(rounds) != 1 or not samples:
                missing.append(slot + ':' + task + ':missing_task_cohort')
                continue
            records = read_rows(rounds[0]['raw_records'])
            by_id, verified = {s['id']: s for s in samples}, []
            for record in records:
                if record['sample_id'] not in by_id:
                    raise ValueError('Unexpected calibration sample')
                sample = by_id[record['sample_id']]
                request = adapter.build(spec['candidate'], sample)
                if record['request_sha256'] != fingerprint(request) or record['protocol_sha256'] != adapter.policy_sha256:
                    raise ValueError('Calibration request/protocol identity changed')
                observation = {k: record[k] for k in ['finish_reason', 'generated_tokens', 'terminal_token_id', 'eos_token_ids']}
                if 'class_scores' in record:
                    observation['class_scores'] = record['class_scores']
                verified.append(adapter.finish(request, sample, record['raw_output'], **observation))
            summary = summarize_quality(samples, verified)
            gates[slot + ':' + task] = {'summary': summary, **quality_gate(policy, task, summary)}
    expected = sum(len(w['tasks']) for w in manifest['workers'].values())
    passed = bool(expected) and not missing and len(gates) == expected and all(g['passed'] for g in gates.values())
    return {'status': 'CALIBRATION_QUALITY_SCREEN_PASSED' if passed else 'POOL_REJECTED_BY_CALIBRATION_QUALITY_GATES',
            'passed': bool(passed), 'manifest_sha256': manifest_hash, 'profile_sha256': hashes,
            'gates': gates, 'missing': missing, 'pool_admitted': False,
            'scope': 'calibration_early_rejection_only_not_independent_admission'}


def assess(manifest_path, results, output):
    """Assemble all 12 G/S comparisons; missing evidence can never pass."""
    from .admission import quality_gate, compare_quality, compare_costs
    from ..tasks import summarize_quality
    from ..backends.compact import CompactAdapter
    manifest = json.loads(Path(manifest_path).read_text())
    study = json.loads(Path(manifest['study']).read_text())
    policy = json.loads(Path(study['task_adapter_protocol']['path']).read_text())
    if policy.get('diagnostic_only'):
        raise ValueError('Diagnostic protocols cannot produce model-pool admission')
    inputs = checked_manifest(study['quality_preparation']['manifest'])
    classmap = next(Path(p) for p in manifest['sources'] if Path(p).name == 'imagenet-class-index.json')
    adapter = CompactAdapter(policy, json.loads(classmap.read_text()))
    results = Path(results)
    observations, absolute, missing = {}, {}, []
    for slot, w in manifest['workers'].items():
        run = results / f'quality-{slot}-n1' / 'profile.json'
        if not run.is_file() or json.loads(run.read_text()).get('status') != 'COMPLETE':
            missing.append(str(run))
            continue
        profile = json.loads(run.read_text())
        spec = json.loads(Path(w['spec']).read_text())
        if profile['model']['candidate'] != spec['candidate'] or profile['model']['execution'] != spec['execution']:
            raise ValueError('Quality evidence has a different model/configuration identity')
        for task in w['tasks']:
            rows = [r for r in profile['rounds'] if r['task'] == task]
            if len(rows) != 1:
                raise ValueError('Quality must have exactly one independent fixed cohort per task')
            records = read_rows(rows[0]['raw_records'])
            samples = read_rows(inputs['tasks'][task]['path'])
            verified = []
            by_id = {sample['id']: sample for sample in samples}
            for record in records:
                request = adapter.build(spec['candidate'], by_id[record['sample_id']])
                from ..backends.compact import fingerprint
                if record['request_sha256'] != fingerprint(request) or record['protocol_sha256'] != adapter.policy_sha256:
                    raise ValueError('Quality request or protocol identity changed')
                observed = {k: record[k] for k in ['finish_reason', 'generated_tokens', 'terminal_token_id', 'eos_token_ids']}
                if 'class_scores' in record:observed['class_scores'] = record['class_scores']
                verified.append(adapter.finish(request, by_id[record['sample_id']], record['raw_output'], **observed))
            observations[slot, task] = verified
            absolute[slot + ':' + task] = quality_gate(policy, task, summarize_quality(samples, verified))
    comparisons = []
    for task in TASKS:
        samples = read_rows(inputs['tasks'][task]['path'])
        for g in [c['slot'] for c in study['generalists']]:
            quality = None
            if (g, task) in observations and (task, task) in observations:
                quality = compare_quality(policy, samples, observations[g, task], observations[task, task])
            costs = []
            for seed in (42, 43, 44):
                pair = []
                for slot in (g, task):
                    path = results / f'cost-{slot}-{task}-seed{seed}' / 'measurement.json'
                    if not path.is_file():missing.append(str(path))
                    else:
                        data = json.loads(path.read_text())
                        if data['job']['deployment'] != {slot: 1} or data['job']['seed'] != seed or data['task'] != task:
                            raise ValueError('Matched-cost evidence identity changed')
                        pair.append(data)
                costs.append(compare_costs(*pair) if len(pair) == 2 else {'passed': False, 'status': 'MISSING_EVIDENCE'})
            passed = (quality is not None and quality['passed'] and all(c['passed'] for c in costs)
                      and absolute.get(g + ':' + task, {}).get('passed', False)
                      and absolute.get(task + ':' + task, {}).get('passed', False))
            comparisons.append({'generalist': g, 'specialist': task, 'task': task,
                                'quality': quality, 'matched_cost_repetitions': costs, 'passed': bool(passed)})
    result = {'status': 'COMPLETE_POOL_SCREEN' if not missing else 'INCOMPLETE_EVIDENCE',
              'absolute_quality_gates': absolute, 'comparisons': comparisons, 'missing': sorted(set(missing)),
              'all_requirements_passed': not missing and all(c['passed'] for c in comparisons),
              'runtime_catalog': None, 'utility': None, 'admission_scope': 'standalone_pool_screen_only',
              'remaining': 'Measured colocation/replica domains and complete deployment evidence are required for runtime catalog.'}
    save(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--study', default='configs/study.json')
    prep.add_argument('--inventory', required=True)
    prep.add_argument('--environments', default='configs/compact_environments.json')
    prep.add_argument('--execution', default='configs/compact_execution.json')
    prep.add_argument('--classmap', required=True)
    prep.add_argument('--output', required=True)
    plan = sub.add_parser('prepare-plan')
    plan.add_argument('--manifest', required=True)
    plan.add_argument('--plan', required=True)
    plan.add_argument('--output', required=True)
    assessment = sub.add_parser('assess')
    assessment.add_argument('--manifest', required=True)
    assessment.add_argument('--results', required=True)
    assessment.add_argument('--output', required=True)
    run = sub.add_parser('run')
    run.add_argument('--manifest', required=True)
    run.add_argument('--job', required=True)
    run.add_argument('--output', required=True)
    run.add_argument('--execute', action='store_true')
    worker = sub.add_parser('worker-profile')
    worker.add_argument('--job', required=True)
    worker.add_argument('--output', required=True)
    worker.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.study, args.inventory, args.environments, args.execution, args.classmap, args.output)
        print(json.dumps({'status': result['status'], 'jobs': len(result['jobs']), 'gpu_started': False}))
    elif args.command == 'prepare-plan':
        result = prepare_plan(args.manifest, args.plan, args.output)
        print(json.dumps({'status': result['status'], 'gpu_started': False}))
    elif args.command == 'assess':
        result = assess(args.manifest, args.results, args.output)
        print(json.dumps({k: result[k] for k in ['status', 'all_requirements_passed', 'runtime_catalog']}))
    elif args.command == 'run':
        if not args.execute:
            parser.error('GPU execution disabled: explicit --execute is required')
        result = execute_job(args.manifest, args.job, args.output)
        print(json.dumps(result))
        if result['status'] != 'COMPLETE':
            raise SystemExit(1)
    else:
        if not args.execute or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
            parser.error('worker-profile is an internal GPU subprocess; use run --execute')
        result = profile_job(args.job, args.output)
        if result['status'] != 'COMPLETE':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
