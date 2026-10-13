"""8F-2 serial exact-context bank; copies the existing measurement-sharing adapter.
Changes are per-cell safe-memory admission, peak auditing and the frozen 8F budget caps.
Serving harness, objective, seeds and metric aggregation are unchanged.
"""
import json
from pathlib import Path
from deployment_search.runtime_cache import save_json,digest,file_hash
from integer_replica_pilot.evaluator import runner,summarize,PilotExecutionBlocked
from .common import OUT,read,utc

class MeasurementBank:
    def __init__(self, physical_runner=runner, persist=True, directory=OUT):
        self.runner, self.persist, self.directory = physical_runner, persist, Path(directory)
        self.runs, self.references, self.failures, self.evaluations, self.schedule_log = [], [], [], [], []
        self.donors = {}
        self.states = set()
        self.active = None
        self.first_failure = None

    def write(self):
        if not self.persist: return
        save_json(self.directory / 'physical_measurements.json', {'runs': self.runs})
        save_json(self.directory / 'measurement_sharing_log.json', {'logical_run_references': self.references,
            'logical_evaluations': self.evaluations, 'shared_reference_count': sum(r['cache_hit'] for r in self.references),
            'sharing_count_unit': 'logical repetition references reusing one physical raw run',
            'shared_raw_is_not_independent': True, 'logical_budget_charged_on_cache_hits': True})
        save_json(self.directory / 'runtime_failure_log.json', {'events': self.failures, 'retries': 0,
            'policy': 'Frozen zero automatic retries; genuine unstable valid repetitions retained.',
            'campaign_stopped_on_invalid_execution': self.first_failure is not None})
        save_json(self.directory / 'execution_log.json', {'slots': self.schedule_log})
        save_json(self.directory / 'run_accounting.json', self.accounting())

    def accounting(self):
        logical = sum(e['phase'] == 'search' for e in self.evaluations)
        physical = len(self.states)
        timed = sum(r.get('measurement_started', False) for r in self.runs)
        return {'logical_search_evaluations': logical,
                'logical_confirmation_repetition_references': sum(r['phase'] == 'confirmation' for r in self.references),
                'unique_physical_states': physical, 'new_physical_attempts': len(self.runs), 'reused_measurements': sum(r['cache_hit'] for r in self.references), 'planning_safe_peak_budget_violations': sum(r.get('physical_budget_violation',False) for r in self.runs),
                'new_timed_runs': timed, 'completed_valid_timed_runs': sum(r.get('valid', False) for r in self.runs),
                'confirmation_runs': sum(r['phase'] == 'confirmation' and r.get('measurement_started', False) for r in self.runs),
                'confirmation_attempts': sum(r['phase'] == 'confirmation' for r in self.runs),
                'measurement_sharing_count': sum(r['cache_hit'] for r in self.references),
                'retries': 0, 'frozen_caps': {'logical': 96, 'physical_states': 96, 'timed_runs': 384}, 'structural_adjusted_caps': {'logical':84,'physical_states':84,'timed_runs':324},
                'within_budget': logical <= 84 and physical <= 84 and len(self.runs) <= 324,
                'accounting_rule': 'Invalid premeasurement launches count as physical attempts; only an observed measurement window counts as a timed run. Both attempt and timed caps enforced.'}

    def sharing_identity(self, config, context_path):
        a = self.active
        contract = 'search' if a['phase'] == 'search' else 'confirmation_' + a['role']
        return {'config': config, 'runtime_context_sha256': file_hash(context_path),
                'workload_sha256': a['workload_sha256'], 'phase_contract': contract,
                'protocol_sha256': a['protocol_sha256'],
                'sharing_policy_sha256': a['sharing_policy_sha256']}

    def raw_run(self, config, raw_path, log_path, config_path, context_path):
        assert self.active is not None
        from .campaign import verify_execution
        verify_execution()
        assert self.first_failure is None, 'Frozen no-retry campaign has already stopped'
        context = read(context_path)
        plan = sum(context['memory_table_mib'][m][str(n)] for m,n in config['deployment'].items())
        safe = (1 + context['resource_budget_semantics']['epsilon_mem']) * plan
        assert safe <= context['gpu_capacity_mib'], 'RESOURCE_BUDGET_INFEASIBLE'
        a = dict(self.active)
        identity = self.sharing_identity(config, context_path)
        key = digest(identity)
        base_reference = {k: a[k] for k in ['method', 'workload', 'workload_sha256', 'logical_slot', 'phase', 'role']}
        base_reference.update(repetition_seed=config['random_seed'], deployment=config['deployment'],
                              routing_hash=digest(config['routing_rps']), measurement_identity_hash=key,
                              budget_charge=0, reference_at_utc=utc())
        if key in self.donors:
            donor = self.donors[key]
            assert donor['identity'] == identity and donor['valid'], 'Incompatible/invalid donor'
            assert file_hash(donor['raw_path']) == donor['raw_sha256']
            # Symlink preserves one physical measurement; each evaluator still records its logical use.
            raw_path = Path(raw_path)
            if not raw_path.exists(): raw_path.symlink_to(Path(donor['raw_path']).resolve())
            reference = dict(base_reference, physical_raw_id=donor['physical_raw_id'],
                             raw_sha256=donor['raw_sha256'], cache_hit=True,
                             shared_with=donor['owner'], raw_path=str(raw_path))
            self.references.append(reference); self.write()
            print(json.dumps({'event': 'raw_shared', 'method': a['method'], 'workload': a['workload'],
                              'seed': config['random_seed'], 'physical_raw_id': donor['physical_raw_id']}), flush=True)
            return dict(donor['outcome'])
        assert len(self.runs) < 324, 'Frozen physical attempt budget reached'
        state_key = digest({'deployment': config['deployment'], 'workload': config['workload_rps'],
                            'routing': config['routing_rps'], 'context': file_hash(context_path),
                            'workload_hash': a['workload_sha256']})
        assert state_key in self.states or len(self.states) < 84, 'Frozen physical state cap reached'
        self.states.add(state_key)
        record = dict(base_reference, physical_raw_id=f'raw_{len(self.runs)+1:04d}',
                      raw_path=str(raw_path), identity=identity, owner=f"{a['workload']}/{a['method']}/{a['phase']}/{a['role']}",
                      started_at_utc=utc(), state_identity_sha256=state_key, status='started')
        self.runs.append(record); self.write()
        print(json.dumps({'event': 'physical_start', 'raw_id': record['physical_raw_id'],
                          'method': a['method'], 'workload': a['workload'], 'phase': a['phase'],
                          'role': a['role'], 'seed': config['random_seed'], 'deployment': config['deployment']}), flush=True)
        try:
            outcome = self.runner(config, raw_path, log_path, config_path, context_path)
        except Exception as exc:
            outcome = {'returncode': -1, 'adapter_exception': repr(exc), 'timed_out': False}
        record.update(outcome=outcome, finished_at_utc=utc())
        if not Path(raw_path).exists():
            self.fail('external_interruption', 'Physical runner produced no raw result', record)
            raise PilotExecutionBlocked('Missing raw runtime output; zero retries')
        raw = read(raw_path)
        record['raw_sha256'] = file_hash(raw_path)
        record['measurement_started'] = bool(raw.get('measurement_window'))
        record['valid'] = bool(raw.get('valid_execution') and raw.get('measurement_window') and not outcome.get('timed_out'))
        record['status'] = raw.get('execution_status', 'unknown')
        obs = raw.get('memory_observations', {})
        record.update(M_plan_mib=plan,M_safe_mib=safe,budget_mib=context['gpu_capacity_mib'],actual_idle_mib=obs.get('actual_idle_mib'),actual_peak_mib=obs.get('peak_device_mib'),memory_observations=obs,planning_safe=True)
        record['physical_budget_violation'] = obs.get('peak_device_mib') is not None and obs['peak_device_mib'] > context['gpu_capacity_mib']
        if record['physical_budget_violation']:
            record['valid'] = False
            record['resource_status'] = 'PHYSICAL_BUDGET_VIOLATION'
            self.fail('PHYSICAL_BUDGET_VIOLATION', 'Planning-safe deployment exceeded the frozen per-cell physical peak limit; no budget growth', record)
            raise PilotExecutionBlocked('PHYSICAL_BUDGET_VIOLATION; raw evidence retained, zero retries')
        assert all(raw.get(k) == v for k, v in config.items() if k != 'arrival_process'), 'Raw config identity differs'
        rep = summarize(raw, self.workload, self.catalog, outcome, raw_path)
        record['stability'] = rep.get('stability_state_v2')
        record['valid'] = bool(rep['valid_runtime_repetition'])
        self.references.append(dict(base_reference, physical_raw_id=record['physical_raw_id'],
                                    raw_sha256=record['raw_sha256'], cache_hit=False,
                                    shared_with=None, raw_path=str(raw_path)))
        if not record['valid']:
            resource = raw.get('runtime_resource_status') == 'RUNTIME_RESOURCE_INFEASIBLE'
            category = 'runtime_resource_infeasible' if resource else 'external_interruption' if raw.get('execution_status') == 'external_interruption' or outcome.get('timed_out') else 'model_or_harness_failure'
            self.fail(category, '; '.join(raw.get('errors', [])), record)
        elif record['stability'] != 'steady':
            self.failures.append({'category': record['stability'], 'physical_raw_id': record['physical_raw_id'],
                                  'workload': a['workload'], 'method': a['method'],
                                  'valid_timed_execution': True, 'retry': False})
        if record['valid']: self.donors[key] = record
        self.write()
        print(json.dumps({'event': 'physical_complete', 'raw_id': record['physical_raw_id'],
                          'valid': record['valid'], 'stability': record['stability'],
                          'mean_ms': rep.get('weighted_mean_e2e'), 'timed_runs': self.accounting()['new_timed_runs']}), flush=True)
        return outcome

    def fail(self, category, reason, record):
        self.first_failure = {'category': category, 'reason': reason, 'physical_raw_id': record['physical_raw_id'],
                              'workload': record['workload'], 'method': record['method'],
                              'raw_path': record['raw_path'], 'retry': False}
        self.failures.append(self.first_failure); self.write()

