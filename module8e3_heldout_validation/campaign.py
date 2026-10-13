"""Serial GPU schedule around unchanged independent method engines."""
import argparse
from dataclasses import dataclass, field
import fcntl
import json
import os
from pathlib import Path
import threading
import time
import traceback

from deployment_core import Workload
from deployment_search.runtime import RuntimeEvaluator
from deployment_search.runtime_cache import save_json, digest, file_hash
from deployment_search.real_objective import NoiseAwareRuntimeObjective
from deployment_search.result import jsonable
from deployment_search.tests.fixtures import forbid_enumeration
from integer_replica_pilot.evaluator import ReplicaRuntimeEvaluator, PilotExecutionBlocked, runner, summarize, aggregate
from joint_deploy.measurement.schema import MeasurementConfig
from .common import ROOT, F, OUT, read, ref, immutable, utc, inputs, prepare, verify_manifest


@dataclass
class Request:
    deployment: dict
    workload: object
    routing: dict
    index: int
    iteration: int
    rank: int
    response: object = None
    error: object = None
    event: object = field(default_factory=threading.Event)


class ScheduledEvaluator(RuntimeEvaluator):
    def __init__(self, actor):
        self.actor = actor
        self.count = 0

    def evaluate(self, deployment, workload, routing):
        actor = self.actor
        with actor.condition:
            index = self.count
            self.count += 1
            assert self.count <= 6
            iteration, rank = 0, 0
            if index:
                trace = actor.trace['iterations'][-1]
                iteration = trace['iteration'] + 1
                candidate = next(c for c in trace['candidates'] if c['selected_top_b']
                                 and c['mutation']['deployment'] == deployment)
                rank = candidate['rank']
                assert iteration in actor.committed
            request = Request(dict(deployment), workload, dict(routing), index, iteration, rank)
            assert actor.pending is None
            actor.pending = request
            actor.condition.notify_all()
        request.event.wait()
        if request.error: raise request.error
        return request.response


class Actor:
    def __init__(self, method, workload, catalog, hardware, roles, cal, directory, persist=True):
        self.method, self.workload = method, workload
        self.catalog, self.hardware, self.roles, self.cal = catalog, hardware, roles, cal
        self.directory, self.persist = Path(directory), persist
        self.condition = threading.Condition()
        self.pending = None
        self.trace = None
        self.committed = {}
        self.done, self.error = False, None
        self.evaluator = ScheduledEvaluator(self)
        self.thread = threading.Thread(target=self.work, name=method, daemon=True)

    def checkpoint(self, result):
        snapshot = result.to_dict()
        with self.condition:
            self.trace = snapshot
            for tr in snapshot['iterations']:
                it = tr['iteration'] + 1
                if tr['selected_top_b'] and it not in self.committed:
                    commitment = {'method': self.method, 'iteration': it,
                                  'current_deployment': tr['deployment'],
                                  'ranking': tr['ranking'], 'selected_top_b': tr['selected_top_b'],
                                  'candidate_analytic_records': [{k: c[k] for k in ['mutation', 'delta_z', 'rank', 'selected_top_b', 'memory_mib']}
                                                               for c in tr['candidates']],
                                  'committed_at_utc': utc()}
                    commitment['commitment_sha256'] = digest(commitment)
                    self.committed[it] = commitment
                    if self.persist: immutable(self.directory / f'round_{it}_commitment.json', commitment)
            if self.persist:
                save_json(self.directory / 'search_trace.json', snapshot)
                for tr in snapshot['iterations']:
                    save_json(self.directory / f'iteration_{tr["iteration"]+1}.json', tr)
            self.condition.notify_all()

    def work(self):
        try:
            from method_v1 import create, config
            from method_v2 import MethodV2, frozen_config
            objective = NoiseAwareRuntimeObjective(1., .05)
            engine = create(self.evaluator, runtime_objective=objective, trace_callback=self.checkpoint) if self.method == 'M1' else MethodV2(
                self.evaluator, predictor=None, runtime_objective=objective, trace_callback=self.checkpoint)
            result = engine.search(self.workload, self.catalog, self.hardware,
                                   config() if self.method == 'M1' else frozen_config(),
                                   roles=self.roles, capacity_mode='empirical', memory_mode='empirical', calibration=self.cal)
            self.checkpoint(result)
            if self.persist: save_json(self.directory / 'search_complete.json', result.to_dict())
        except BaseException as exc:
            self.error = exc
            if self.persist: save_json(self.directory / 'actor_failure.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        finally:
            with self.condition:
                self.done = True
                self.condition.notify_all()

    def wait_pending(self, timeout=120):
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.pending is None and not self.done:
                remaining = deadline - time.monotonic()
                if remaining <= 0: raise RuntimeError('CPU engine failed to reach scheduled request: ' + self.method)
                self.condition.wait(min(remaining, 1))
            if self.error: raise self.error
            return self.pending

    def return_result(self, request, response):
        with self.condition:
            assert self.pending is request
            self.pending = None
            request.response = response
            request.event.set()
            self.condition.notify_all()

    def abort(self, error):
        with self.condition:
            if self.pending:
                self.pending.error = error
                self.pending.event.set()
                self.pending = None


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
                'unique_physical_states': physical, 'new_physical_attempts': len(self.runs),
                'new_timed_runs': timed, 'completed_valid_timed_runs': sum(r.get('valid', False) for r in self.runs),
                'confirmation_runs': sum(r['phase'] == 'confirmation' and r.get('measurement_started', False) for r in self.runs),
                'confirmation_attempts': sum(r['phase'] == 'confirmation' for r in self.runs),
                'measurement_sharing_count': sum(r['cache_hit'] for r in self.references),
                'retries': 0, 'frozen_caps': {'logical': 78, 'physical_states': 72, 'timed_runs': 264},
                'within_budget': logical <= 78 and physical <= 72 and len(self.runs) <= 264,
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
        assert len(self.runs) < 264, 'Frozen physical attempt budget reached'
        state_key = digest({'deployment': config['deployment'], 'workload': config['workload_rps'],
                            'routing': config['routing_rps'], 'context': file_hash(context_path),
                            'workload_hash': a['workload_sha256']})
        assert state_key in self.states or len(self.states) < 72, 'Frozen physical state cap reached'
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


def request_for_slot(actor, slot):
    pending = actor.wait_pending()
    if pending is None: return None
    if pending.index != slot['logical_state_index']:
        assert pending.index > slot['logical_state_index'], 'Engine request behind frozen schedule'
        return None
    if slot['role'] == 'candidate':
        assert (pending.iteration, pending.rank) == (slot['iteration'], slot['Top_B_rank']), 'Frozen iteration/rank slot mismatch'
    return pending


def round_barrier(actors, iteration):
    for actor in actors.values():
        pending = actor.wait_pending()
        if pending is None: continue
        # The current round commitment must be written before a round value is delivered.
        assert pending.iteration == iteration and iteration in actor.committed
    return {method: actor.committed.get(iteration, {}).get('commitment_sha256') for method, actor in actors.items()}


def run_workload(block, bank, catalog, hardware, roles, cal):
    name = block['workload']
    w = next(w for w in read(F / 'heldout_workloads.json')['workloads'] if w['id'] == name)
    workload = Workload(w['lambda_rps'])
    bank.catalog, bank.workload = catalog, workload
    actors = {m: Actor(m, workload, catalog, hardware, roles, cal,
                      OUT / ('m1_search_traces' if m == 'M1' else 'm2_search_traces') / name) for m in ['M1', 'M2']}
    directory = OUT / name
    directory.mkdir(parents=True, exist_ok=True)
    protocol = read(F / 'method_comparison_freeze.json')
    evaluators = {m: ReplicaRuntimeEvaluator(directory / m / 'search_cache', catalog,
        OUT / 'runtime_context.json', protocol['measurement_settings'],
        seeds=protocol['search_seeds'], max_unique=1 if m == 'B2' else 6, run=bank.raw_run) for m in ['M1', 'M2', 'B2']}
    plan = next(r for r in read(F / 'planning_sanity_check.json')['workloads'] if r['workload'] == name)
    seed_requests = {}
    # Starting threads in sequence keeps their initial native MILP calls separate.
    for m, actor in actors.items():
        actor.thread.start()
        seed_requests[m] = actor.wait_pending()
        assert seed_requests[m] is not None and seed_requests[m].index == 0
        assert seed_requests[m].deployment == plan['seed']['deployment'], 'Frozen seed deployment differs'
        assert jsonable(seed_requests[m].routing) == plan['canonical']['routing'], 'Frozen seed routing differs'
    identities = [evaluators[m].identity(seed_requests[m].deployment, workload, seed_requests[m].routing) for m in ['M1', 'M2']]
    assert identities[0] == identities[1], 'Mandatory common seed exact identity differs'
    immutable(directory / 'seed_reproducibility.json', {'identical': True, 'identities': identities,
        'frozen_planning_seed': plan['seed']['deployment'], 'verified_before_first_workload_GPU_run': True})
    confirmations = {m: {'seed': [], 'final': []} if m != 'B2' else {'fixed': []} for m in ['M1', 'M2', 'B2']}
    b2_runtime = None
    barriers = {}
    search_frozen = False
    try:
        for slot_index, slot in enumerate(block['slots']):
            verify_manifest()
            bank.active = dict(slot, workload=name, workload_sha256=block['workload_sha256'],
                logical_slot=slot_index, protocol_sha256=file_hash(F / 'method_comparison_freeze.json'),
                sharing_policy_sha256=file_hash(F / 'measurement_sharing_policy.json'))
            log = {'workload': name, 'slot_index': slot_index, 'slot': slot, 'started_at_utc': utc(), 'status': 'STARTED'}
            bank.schedule_log.append(log); bank.write()
            save_json(OUT / 'campaign_status.json', {'status': 'running', 'workload': name,
                'slot_index': slot_index, 'active_slot': slot, 'accounting': bank.accounting(), 'ROUTING_REVISION_PENDING': True})
            method = slot['method']
            if slot['phase'] == 'search':
                request = None
                if method == 'B2':
                    x = {m: plan['B2']['deployment'][m] for m in catalog}
                    routing = {(t, m): value for t, row in plan['B2']['routing_rps'].items() for m, value in row.items()}
                else:
                    if slot['role'] == 'candidate' and slot['iteration'] not in barriers:
                        barriers[slot['iteration']] = round_barrier(actors, slot['iteration'])
                        immutable(directory / f'round_{slot["iteration"]}_barrier.json', {
                            'workload': name, 'iteration': slot['iteration'], 'commitments': barriers[slot['iteration']],
                            'before_first_round_measurement': True, 'at_utc': utc()})
                    request = request_for_slot(actors[method], slot)
                    if request is None:
                        log.update(status='SKIPPED', reason='Method stopped or requested state absent; no slot reassignment', completed_at_utc=utc())
                        bank.write(); continue
                    x, routing = request.deployment, request.routing
                identity = evaluators[method].identity(x, workload, routing)
                logical = {'method': method, 'workload': name, 'workload_sha256': block['workload_sha256'],
                           'phase': 'search', 'role': slot['role'], 'logical_slot': slot_index,
                           'method_state_index': slot['logical_state_index'], 'budget_charge': 1,
                           'deployment': identity['deployment'], 'routing_hash': digest(identity['routing']),
                           'identity_sha256': digest(identity), 'state_identity': identity,
                           'round_barrier': barriers.get(slot.get('iteration')), 'status': 'selected'}
                bank.evaluations.append(logical)
                assert bank.accounting()['logical_search_evaluations'] <= 78
                bank.write()
                runtime = evaluators[method].evaluate(x, workload, routing)
                logical.update(status='completed', raw_result_paths=runtime.raw_result_paths,
                               repetitions=runtime.valid_repetition_count, stability=runtime.stability_state_v2)
                if method == 'B2':
                    b2_runtime = runtime
                    save_json(directory / 'B2/search_result.json', {'deployment': x, 'routing': jsonable(routing), 'runtime': runtime})
                else:
                    # No other method's measurements are present in this response.
                    actors[method].return_result(request, runtime)
                    actors[method].wait_pending()
            else:
                if not search_frozen:
                    for actor in actors.values():
                        assert actor.wait_pending() is None and actor.done and not actor.error
                    decisions = {m: {'trace': ref(a.directory / 'search_complete.json'),
                        'seed': a.trace['seed']['deployment'], 'final': a.trace['deployment'],
                        'final_routing': a.trace['canonical']['routing'], 'termination_reason': a.trace['status']}
                        for m, a in actors.items()}
                    immutable(directory / 'post_search_freeze.json', {'workload': name, 'methods': decisions,
                        'confirmation_does_not_influence_search': True, 'decision_frozen_at_utc': utc()})
                    search_frozen = True
                role = slot['role']
                if method == 'B2':
                    x = {m: plan['B2']['deployment'][m] for m in catalog}
                    routing = {(t, m): value for t, row in plan['B2']['routing_rps'].items() for m, value in row.items()}
                else:
                    trace = actors[method].trace
                    x = trace['seed']['deployment'] if role == 'seed' else trace['deployment']
                    route = trace['iterations'][0]['canonical']['routing'] if role == 'seed' else trace['canonical']['routing']
                    routing = {tuple(k.split('::')): v for k, v in route.items()}
                identity = evaluators[method].identity(x, workload, routing)
                config = MeasurementConfig(identity['deployment'], identity['workload'], identity['routing'],
                    random_seed=slot['seed'], **protocol['measurement_settings']).as_dict()
                folder = directory / method / 'confirmation'
                folder.mkdir(parents=True, exist_ok=True)
                prefix = folder / (role + '_rep_' + str(slot['repetition']))
                raw_path, log_path, config_path = Path(str(prefix)+'.json'), Path(str(prefix)+'.log'), Path(str(prefix)+'.config.json')
                assert not raw_path.exists(), 'Do not rerun an existing confirmation'
                save_json(config_path, config)
                outcome = bank.raw_run(config, raw_path, log_path, config_path, OUT / 'runtime_context.json')
                rep = summarize(read(raw_path), workload, catalog, outcome, raw_path)
                rep.update(repetition=slot['repetition'], random_seed=slot['seed'], post_search_confirmation=True)
                save_json(Path(str(prefix)+'.summary.json'), rep)
                if not rep['valid_runtime_repetition']: raise PilotExecutionBlocked('Invalid confirmation; zero retries')
                confirmations[method][role].append(rep)
                save_json(directory / 'confirmation_progress.json', confirmations)
            log.update(status='COMPLETED', completed_at_utc=utc())
            bank.write()
        results = {}
        for m in ['M1', 'M2', 'B2']:
            aggs = {}
            for role, reps in confirmations[m].items():
                assert len(reps) == 2 and len({str(Path(r['raw_result_path']).resolve()) for r in reps}) == 2
                counts = {k: plan['B2']['deployment'][k] for k in catalog} if m == 'B2' else actors[m].trace['seed']['deployment'] if role == 'seed' else actors[m].trace['deployment']
                path = directory / m / 'confirmation' / (role + '_aggregate.json')
                aggs[role] = aggregate(sorted(reps, key=lambda r: r['repetition']), 'independent_confirmation', path, counts)
                save_json(path, aggs[role])
            if m != 'B2':
                seed_paths = {str(Path(r['raw_result_path']).resolve()) for r in confirmations[m]['seed']}
                final_paths = {str(Path(r['raw_result_path']).resolve()) for r in confirmations[m]['final']}
                assert not seed_paths & final_paths, 'Seed/final confirmations cannot share within method'
            results[m] = {'method': m, 'workload': name, 'status': 'complete', 'confirmations': aggs,
                'search_runtime': b2_runtime if m == 'B2' else actors[m].trace['runtime'],
                'deployment': {k: plan['B2']['deployment'][k] for k in catalog} if m == 'B2' else actors[m].trace['deployment'],
                'routing': jsonable(routing) if m == 'B2' else actors[m].trace['canonical']['routing'],
                'logical_search_evaluations': 1 if m == 'B2' else actors[m].trace['runtime_evaluation_count']}
            # B2 routing always comes from the frozen deterministic planning state.
            if m == 'B2': results[m]['routing'] = {f'{t}::{model}': value for t, row in plan['B2']['routing_rps'].items() for model, value in row.items()}
            save_json(directory / m / 'result.json', results[m])
        save_json(directory / 'status.json', {'status': 'complete', 'workload': name,
            'logical_counts': {m: r['logical_search_evaluations'] for m, r in results.items()}})
        print(json.dumps({'event': 'workload_complete', 'workload': name,
                          'counts': {m: r['logical_search_evaluations'] for m, r in results.items()},
                          'accounting': bank.accounting()}), flush=True)
    except BaseException as exc:
        for actor in actors.values(): actor.abort(exc)
        save_json(directory / 'status.json', {'status': 'incomplete', 'reason': repr(exc), 'no_retry': True})
        bank.write()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    for key, value in read(F / 'method_comparison_freeze.json')['execution_environment_overrides'].items():
        os.environ[key] = value
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / '.campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = prepare()
        if args.prepare_only: return
        if (OUT / 'campaign_status.json').exists():
            status = read(OUT / 'campaign_status.json')
            if status['status'] == 'complete':
                from .report import report
                report(); return
            raise PilotExecutionBlocked('Campaign already started; preserve attempts, no automatic restart/retry')
        catalog, hardware, roles, cal = inputs()
        bank = MeasurementBank()
        bank.write()
        try:
            with forbid_enumeration() as guards:
                for block in read(F / 'execution_order.json')['workloads']:
                    run_workload(block, bank, catalog, hardware, roles, cal)
                assert sum(g.call_count for g in guards) == 0
            verify_manifest()
            assert bank.accounting()['within_budget']
            save_json(OUT / 'campaign_status.json', {'status': 'complete', 'accounting': bank.accounting(),
                'completed_at_utc': utc(), 'ROUTING_REVISION_PENDING': True, 'enumeration_calls': 0})
        except BaseException as exc:
            save_json(OUT / 'campaign_status.json', {'status': 'incomplete', 'error': repr(exc),
                'traceback': traceback.format_exc(), 'accounting': bank.accounting(),
                'no_automatic_retry_or_budget_growth': True, 'ROUTING_REVISION_PENDING': True})
            bank.write()
            from .report import report
            report()
            raise
        from .report import report
        report()


if __name__ == '__main__': main()
