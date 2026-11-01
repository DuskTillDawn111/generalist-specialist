"""Native fixed five-term shared-GPU experiment with original metric primitives."""
from __future__ import annotations

import argparse
import copy
import importlib
import json
from pathlib import Path
import time

from tools.queue_refined_20261023.native_runner import frozen_runtime
from tools.queue_refined_20261023 import admission as original_admission
from tools.queue_coupled_utility_20261023.runtime_window import window
from tools.queue_five_term_20261023.evaluation import evaluate_run
from tools.queue_five_term_20261023.policy import FIXED_WEIGHTS, OBJECTIVE_SCHEMA, validate_weights


def run(system, winner, samples, policy_contract, *, backend='mock', workers=None,
        duration=600., warmup=30., seed=303, arrival_process='poisson',
        startup_timeout=300., drain_timeout=300., admission=None, capacity_evidence=None):
    from tools.queue_five_term_20261023.protocol import validate_contract, make_dispatcher
    validate_contract(policy_contract)
    validate_weights(system.weights)
    if backend not in {'native', 'mock'}:
        raise ValueError('Unknown backend')
    if winner['deployment'] != policy_contract['deployment']:
        raise ValueError('Resident plan/contract mismatch')
    if system.weights != policy_contract['weights'] or system.latency_reference_ms != policy_contract['latency_refs_ms']:
        raise ValueError('System objective/contract mismatch')
    expected_workload = {'text': policy_contract['text_rate'], **{t: sum(v.values()) for t, v in policy_contract['background_routing'].items()}}
    if system.workload != expected_workload:
        raise ValueError('Workload/contract mismatch')
    workers = copy.deepcopy(workers or {})
    for worker in workers.values():
        worker.setdefault('env', {}).update({key: '1' for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
    original = frozen_runtime()
    aliases, routes = original.expand_routes(winner['deployment'], winner['routing'])
    config = original.MeasurementConfig(deployment={a: 1 for a in aliases}, workload_rps=dict(system.workload), routing_rps=routes,
                                        warmup_seconds=warmup, measurement_seconds=duration, arrival_process=arrival_process,
                                        random_seed=seed, startup_timeout_seconds=startup_timeout,
                                        drain_timeout_seconds=drain_timeout, routing_source='coupled_feedback_contract_v1')
    config.validate({a: system.catalog['models'][m]['supported_tasks'] for a, m in aliases.items()})
    registry = original.registry_for(system, aliases, backend, workers or {})
    for worker in registry.values():
        worker.setdefault('env', {}).update({key: '1' for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
    raw = dict(schema_version=1, backend=backend, config=system.as_dict(), transport_config=config.as_dict(),
               replica_to_family=aliases, plan=copy.deepcopy(winner), status='starting', errors=[],
               request_records=[], warmup_request_records=[], priming_request_records=[],
               priming_policy='largest_three_per_task', priming_size_metric='media_file_bytes_or_text_utf8_bytes_v1',
               measurement_window=None, preflight=None, worker_registry=registry,
               admission_evidence=dict(verified=False, reason='MOCK_BACKEND', pairs=[]),
               memory_evidence={}, energy_evidence={}, per_task_metrics={}, system_metrics=None,
               policy_contract=copy.deepcopy(policy_contract), policy_identity=original_admission.policy_identity(policy_contract),
               strategy=policy_contract['strategy'], sample_pool_path=str(Path(samples).resolve()), samples_path=str(Path(samples).resolve()),
               dynamic_capacity_evidence=copy.deepcopy(capacity_evidence),
               objective_schema=OBJECTIVE_SCHEMA, objective_weights=dict(FIXED_WEIGHTS), P_evaluation_role='diagnostic_only',
               measurement_claim='PROSPECTIVE_FIVE_TERM_ROUTING_NATIVE_OBSERVATION', static_SWRR_certificate_applicable=False)
    runtime = original.DeploymentRuntime(config, registry)
    monitor = power = before = after = startup_begin = None
    try:
        if backend == 'native':
            checks = original_admission.preflight(system, winner, samples, workers or {}, policy_contract,
                                                  admission=admission, require_gpu=True, capacity_evidence=capacity_evidence)
            raw['preflight'] = checks
            raw['admission_evidence'] = checks['admission_evidence']
            if not checks['ready']:
                raise ValueError('Native preflight failed: ' + repr(checks['issues']))
        else:
            checks = None
        device = checks['device'] if checks else '0'
        monitor = original.MemoryMonitor(runtime, system.gpu_memory_mib, enabled=backend == 'native', device=device)
        power = original.PowerMonitor(enabled=backend == 'native', device=device)
        power.sample()
        startup_begin = time.perf_counter_ns()
        power.start()
        runtime.start()
        monitor.sample()
        monitor.start()
        monitor.check()
        pool = original.SamplePool(samples, list(system.workload), seed)
        for alias, family in aliases.items():
            for task in system.catalog['models'][family]['supported_tasks']:
                if system.workload.get(task, 0) <= 0:
                    continue
                for item in original.largest_inputs(pool, task):
                    row = original.record('priming', len(raw['priming_request_records']), task, alias, item, time.perf_counter_ns())
                    raw['priming_request_records'].append(row)
                    runtime.enqueue(row, item['sample'])
        runtime.drain(raw['priming_request_records'], drain_timeout)
        if any(r['status'] != 'completed' for r in raw['priming_request_records']):
            raise RuntimeError('Priming failed')
        if warmup > 0:
            raw['warmup_window'] = window(runtime, config, samples, 'warmup', warmup, raw['warmup_request_records'],
                                          monitor, make_dispatcher(policy_contract))
            if any(r['status'] != 'completed' for r in raw['warmup_request_records']):
                raise RuntimeError('Warmup failed')
        before = monitor.sample()
        monitor.check()
        raw['measurement_window'] = {}
        window(runtime, config, samples, 'measurement', duration, raw['request_records'], monitor,
               make_dispatcher(policy_contract), raw['measurement_window'])
        after = monitor.sample()
        monitor.check()
        raw['status'] = 'completed' if all(r['status'] == 'completed' for r in raw['request_records']) else 'completed_with_failures'
    except (Exception, KeyboardInterrupt) as exc:
        raw['status'] = 'failed'
        raw['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        if raw['measurement_window'] and 'drain_end_time_ns' not in raw['measurement_window']:
            raw['measurement_window'].update(aborted=True, drain_end_time_ns=time.perf_counter_ns())
        if monitor:
            monitor.close()
        if power:
            power.close()
        try:
            raw['shutdown'] = runtime.close()
        except Exception as exc:
            raw['shutdown'] = {}
            raw['errors'].append('shutdown: ' + str(exc))
            raw['status'] = 'failed'
        raw['errors'].extend(runtime.errors)
        if runtime.errors or any(v['forced'] or v['exit_code'] != 0 or not v['bye_received'] for v in raw['shutdown'].values()):
            raw['status'] = 'failed'
        for key in ('request_records', 'warmup_request_records', 'priming_request_records'):
            for row in raw[key]:
                if row['status'] in {'planned', 'queued', 'submitted'}:
                    row.update(status='incomplete', error='Run ended before response')
            original.annotate_records(raw[key])
    measured = before is not None and after is not None and monitor is not None and not monitor.failed
    samples_memory = monitor.samples if monitor else []
    nominal = system.catalog['safe_factor'] * sum(system.catalog['memory_table_mib'][m][str(n)] for m, n in winner['deployment'].items())
    safe = system.catalog['safe_factor'] * max(s['owned_used_mib'] for s in [before, after, *samples_memory]) if measured else nominal
    raw['memory_evidence'] = dict(measured=measured, safe_memory_mib=safe,
                                  resource_valid=measured and safe <= system.gpu_memory_mib and all(not s['foreign_pids'] for s in samples_memory),
                                  scope='owned-process idle memory and sampled peak', samples=samples_memory, before=before, after=after)
    w = raw['measurement_window']
    powers = power.samples if power else []
    run_j = original.integrate_power(powers, w['start_time_ns'], w['end_time_ns']) if w else None
    startup_j = original.integrate_power(powers, startup_begin, w['start_time_ns']) if w and startup_begin is not None else None
    full_j = original.integrate_power(powers, w['start_time_ns'], w['drain_end_time_ns']) if w else None
    raw['energy_evidence'] = dict(kind='measured' if backend == 'native' else 'unmeasured', scope='whole_gpu_board',
                                  coverage_valid=backend == 'native' and power is not None and not power.failed and run_j is not None and startup_j is not None,
                                  run_joules=run_j, startup_joules=startup_j, samples=powers,
                                  sampling_errors=power.errors if power else [], startup_begin_time_ns=startup_begin,
                                  scope_note='Original measurement-window plus startup amortization; drain excluded from formal E')
    raw['cohort_energy_diagnostic'] = dict(joules=full_j, includes_drain=True, excludes_startup=True,
                                          drain_joules=full_j - run_j if full_j is not None and run_j is not None else None,
                                          changes_formal_E=False)
    if w:
        raw['per_task_metrics'], raw['system_metrics'], raw['model_metrics'] = original.summarize(raw['request_records'], config, w)
    try:
        identity = original_admission.measurement_identity(system, raw)
        raw['measurement_identity'] = identity
        raw['memory_evidence']['measurement_identity'] = identity
        raw['energy_evidence'].update(measurement_identity=identity, profile_key=identity)
        raw['evaluation'] = evaluate_run(system, raw)
    except Exception as exc:
        raw['errors'].append('evaluation: ' + str(exc))
        raw['status'] = 'failed'
        raw['evaluation'] = dict(utility=dict(U=None, eligible=False, reason='COUPLED_EVALUATION_FAILED'))
    return raw


def execute_job(packet, job_id, output, *, backend='native', execute=False):
    from tools.queue_five_term_20261023.protocol import validate_packet, digest, write
    check = validate_packet(packet)
    if not check['valid']:
        raise ValueError('Packet/source changed: ' + repr(check['changed']))
    if not execute:
        raise ValueError('Explicit execute=True required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    job = next(j for j in packet['jobs'] if j['id'] == job_id)
    base_receipt = job.get('base_packet', packet['base_packet'])
    base = json.loads(Path(base_receipt['path']).read_text())
    original = frozen_runtime()
    package = original.__package__.split('.runtime')[0]
    static = importlib.import_module(package + '.static_study')
    configuration = importlib.import_module(package + '.config')
    try:
        if backend == 'native':
            binding_base = copy.deepcopy(base)
            binding_base['config']['weights'] = copy.deepcopy(job['policy_contract']['weights'])
            validate_weights(binding_base['config']['weights'])
            binding = copy.deepcopy(base['jobs'][0])
            binding['plan'] = job['nominal_plan']
            binding['seed'] = job['seed']
            binding['identity'].update(seed=job['seed'], horizon_seconds=job['identity']['horizon_seconds'], warmup_seconds=job['identity']['warmup_seconds'])
            config, admission_path, workers = static.bind_native_job(binding_base, binding, output / 'native-binding')
            config['catalog']['performance_mode'] = 'task_replica_proxy'
            config['catalog']['memory_mode'] = 'standalone_proxy'
        else:
            config = copy.deepcopy(base['config'])
            admission_path = None
            workers = {}
        # Native binding starts from the historical six-equal-weight base.
        # Bind the prospective objective before constructing the measured system.
        config['weights'] = copy.deepcopy(job['policy_contract']['weights'])
        validate_weights(config['weights'])
        config['random_seed'] = job['seed']
        config['horizon_seconds'] = job['identity']['horizon_seconds']
        raw = run(configuration.SystemConfig(**config), job['nominal_plan'], Path(base['inputs']) / 'runtime-inputs.jsonl',
                  job['policy_contract'], backend=backend, workers=workers, duration=job['identity']['horizon_seconds'],
                  warmup=job['identity']['warmup_seconds'], seed=job['seed'], arrival_process='poisson',
                  drain_timeout=config['drain_seconds'], admission=str(admission_path) if admission_path else None,
                  capacity_evidence=job.get('capacity_evidence'))
        raw['objective_binding'] = dict(objective_schema=OBJECTIVE_SCHEMA,
                                       base_weights=copy.deepcopy(base['config']['weights']),
                                       contract_weights=copy.deepcopy(job['policy_contract']['weights']),
                                       runtime_weights=copy.deepcopy(config['weights']),
                                       original_base_unchanged=True)
    except (Exception, KeyboardInterrupt) as exc:
        raw = dict(schema_version=1, backend=backend, status='failed', errors=[f'{type(exc).__name__}: {exc}'],
                   failure_stage='NATIVE_BINDING_OR_RUN_INITIALIZATION', plan=job['nominal_plan'],
                   policy_contract=job['policy_contract'], strategy=job['policy_contract']['strategy'],
                   request_records=[], warmup_request_records=[], priming_request_records=[], measurement_window=None,
                   memory_evidence={}, energy_evidence={}, shutdown={},
                   evaluation=dict(utility=dict(U=None, eligible=False, reason='FAILED_BEFORE_COMPLETE_RUN')))
    raw.update(experiment_job_id=job_id, experiment_phase=job['phase'], executed_measurement_identity=job['identity'],
               executed_measurement_id=digest(job['identity']), source_packet=base_receipt)
    write(output / 'run.json', raw)
    return raw


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet', type=Path, required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    result = execute_job(json.loads(args.packet.read_text()), args.job, args.output, execute=args.execute)
    print(json.dumps(dict(status=result['status'], utility=result['evaluation']['utility']), ensure_ascii=False))
