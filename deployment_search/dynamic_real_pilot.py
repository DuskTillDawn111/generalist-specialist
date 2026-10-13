"""Module 5B: frozen real-feedback dynamic pilot; no forecasting or enumeration.

Run with the existing gsdeploy interpreter: -m deployment_search.dynamic_real_pilot.
Search probes are isolated harness runs. Only committed paths are replayed with
live workers for switch measurements. Confirmation never enters search.
"""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import queue
import shutil
import statistics
import subprocess
import time

from deployment_core.models import load_hardware, load_model_catalog
from deployment_core.workload import Workload
from joint_deploy.measurement.deployment_runtime import (
    ObservedWorker, TimestampedEvents, default_registry)
from joint_deploy.measurement.schema import MeasurementConfig
from .campaign_runtime import CampaignRuntimeEvaluator, execution_status
from .dynamic_adaptation import (AdaptationConfig, AdaptationHistory,
    DynamicAdaptation, StructuralReconfigurationCost, committed_switches)
from .measurement_runtime import (MeasurementRuntimeEvaluation, aggregate_repetitions,
    measurement_subprocess, summarize_repetition)
from .neighborhood import deployment_key, memory_mib
from .pilot_provenance import source_hashes, verify_assets_unchanged
from .real_objective import NoiseAwareRuntimeObjective
from .real_pilot import SETTINGS, SEEDS, CONFIG, freeze
from .result import jsonable
from .routing import solve_canonical_routing
from .runtime import STABILITY
from .runtime_cache import digest, file_hash, save_json
from .search import GuidedDeploymentSearch
from .tests.fixtures import forbid_enumeration

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/dynamic_adaptation_real_pilot'
HIST = ROOT / 'results/deployment_search_multiworkload'
NAMES = ('W1_balanced', 'W2_generalist_heavy', 'W3_specialist_friendly', 'W1_balanced')
PHASES = ('transition_w1_w2', 'transition_w2_w3', 'transition_w3_w1')
CONFIRM_SEEDS = (8101, 8202)
WARM_CONFIG = AdaptationConfig(top_b=2, max_adaptation_iterations=3,
    max_runtime_evaluations=6)
OBJECTIVE = NoiseAwareRuntimeObjective(1., .05)


def read(path):
    return json.loads(Path(path).read_text())


def append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(jsonable(value), ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def stamp():
    return dict(timestamp=datetime.now(timezone.utc).isoformat(),
                unix_ns=time.time_ns(), monotonic_ns=time.perf_counter_ns())


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    source = HIST / 'benchmark_workloads.json'
    if file_hash(source) != '4b2cc94257870199d36fca03cea0b39028e05f20a5c37eabe55c5a7df9a6fbed':
        raise RuntimeError('Module 3C frozen workload source changed')
    records = {r['name']: r for r in read(source)['workloads']}
    sequence = dict(sequence=list(NAMES), source=dict(path=str(source), sha256=file_hash(source)),
        phases=[dict(phase=i, workload=n, arrival_rates_rps=records[n]['arrival_rates_rps'])
                for i, n in enumerate(NAMES)], state_driver='D2_actual_final_only')
    freeze(OUT / 'dynamic_sequence.json', sequence)
    (OUT / 'dynamic_sequence.sha256').write_text(file_hash(OUT / 'dynamic_sequence.json') + '\n')
    assets = read(ROOT / 'results/deployment_search_real_pilot/asset_hashes.json')
    verify_assets_unchanged(assets)
    sources = source_hashes(assets)
    sources['campaign_runtime_sha256'] = file_hash(ROOT / 'deployment_search/campaign_runtime.py')
    freeze(OUT / 'source_model_hashes.json', sources)
    initial_source = HIST / 'algorithm_runs/W1_balanced__B2/run_result.json'
    initial = read(initial_source)['final_deployment']
    assert deployment_key(initial) == ('paddleocr', 'qwen', 'resnet50', 'whisper')
    config = dict(module='5B', predictor=None, search=jsonable(CONFIG), warm=jsonable(WARM_CONFIG),
        max_valid_unique_runtime_evaluations=6, repetitions_per_state=3,
        search_repetition_seeds=SEEDS, confirmation_repetitions=2,
        confirmation_seeds=CONFIRM_SEEDS, confirmation_influences_search=False,
        measurement_settings=SETTINGS, objective='Unchanged Module 3 NoiseAwareRuntimeObjective',
        min_relative_improvement=.05, routing='canonical_only; no routing search',
        sequence_sha256=file_hash(OUT / 'dynamic_sequence.json'),
        initial_deployment=initial, initial_source=dict(path=str(initial_source), sha256=file_hash(initial_source)),
        strategy_order=['D0', 'D1', 'D2'],
        close_to_cold_relative_gap=.05, cost_equal_relative_tolerance=.05,
        decision_rule='PASS: >=2 close+cheaper transitions; PARTIAL: mixed; NEUTRAL: all costs/performance similar; FAIL: consistently >5% worse or audit fails',
        runtime_cache='strict exact identity; lazy import only upon selected state access',
        reconfiguration='accepted path live-worker execution after search; cold seed reset starts from the same actual old deployment',
        physical_caps=dict(search=150, confirmation=40),
        continuous_live_arrivals_during_switch=False,
        adaptation_wall_clock_time_ms=None,
        adaptation_time_note='Probe search, counterfactual runs and later accepted-path activation are isolated; elapsed orchestration time is not online time-to-adapt.',
        execution_status_policy='Data Quality Audit: only completed measurements receive serving labels',
        scope='pilot only; no forecasting/change detection/prewarming/multi-replica/routing search/final benchmark')
    freeze(OUT / 'pilot_config.json', config)
    freeze(OUT / 'implementation_hashes.json', {str(p.relative_to(ROOT)): file_hash(p)
        for folder in ('deployment_search', 'deployment_core')
        for p in sorted((ROOT / folder).glob('*.py'))})
    return records, sources, initial


class SelectedStateEvaluator:
    """Do not inspect an aggregate until the algorithm requests its exact state."""
    def __init__(self, catalog, sources, algorithm, directory):
        self.base = CampaignRuntimeEvaluator(OUT / 'search_cache', catalog, sources, SETTINGS,
            algorithm, SEEDS, max_unique=45, physical_attempt_cap=150)
        self.catalog, self.algorithm, self.directory = catalog, algorithm, directory
        self.records = []

    def evaluate(self, deployment, workload, routing):
        # Reject alternate routing before any cache lookup.
        canonical = solve_canonical_routing(deployment, workload, self.catalog, CONFIG.rho_max)
        if canonical is None or canonical.routing != routing:
            raise RuntimeError('Runtime access requires exact freshly solved canonical routing')
        identity = self.base.identity(deployment, workload, routing)
        key = digest(identity)
        audit = dict(algorithm_id=self.algorithm, access_reason='algorithm_selected_state',
                     cache_key=key, identity=identity, timestamp=stamp(), imported_from=None)
        with self.base.cache.locked() as cache:
            if key not in cache.data['configurations']:
                manifest_path = HIST / 'shared_cache/runtime_cache_manifest.json'
                # Exact key lookup only; never scan candidate runtime objectives.
                historical = read(manifest_path)['configurations'].get(key)
                if historical and historical['status'] == 'complete' and historical['identity'] == jsonable(identity):
                    validate_entry(historical, identity)
                    _, target = cache.reserve(identity)
                    target.update(historical)
                    target['imported_from'] = str(manifest_path)
                    target['attempts'] = []
                    cache.flush()
                    audit['imported_from'] = str(manifest_path)
            cache.flush()
        manifest_before = read(self.base.cache.path)
        audit['cache_hit'] = manifest_before['configurations'].get(key, {}).get('status') == 'complete'
        append(OUT / 'dynamic_cache_access_audit.jsonl', audit)
        before = len(manifest_before['run_attempts'])
        result = self.base.evaluate(deployment, workload, routing)
        after = len(read(self.base.cache.path)['run_attempts'])
        self.records.append(dict(index=len(self.records)+1, deployment=deployment, cache_key=key,
            cache_hit=audit['cache_hit'], imported_from=audit['imported_from'],
            physical_attempts=after-before, result=jsonable(result), completed_timestamp=stamp()))
        save_json(self.directory / 'runtime_accesses.json', self.records)
        return result


def validate_entry(entry, identity):
    if entry['identity'] != jsonable(identity) or digest(identity) != digest(entry['identity']):
        raise RuntimeError('Cache identity mismatch')
    if file_hash(entry['aggregate']) != entry['aggregate_sha256']:
        raise RuntimeError('Cache aggregate hash mismatch')
    result = MeasurementRuntimeEvaluation(**read(entry['aggregate']))
    if len(result.repetitions) != 3:
        raise RuntimeError('Search requires three valid repetitions')
    for index, rep in enumerate(result.repetitions):
        raw = read(rep['raw_result_path'])
        cfg = MeasurementConfig(identity['deployment'], identity['workload'], identity['routing'],
            random_seed=identity['measurement_settings']['repetition_seeds'][index], **SETTINGS).as_dict()
        if (rep['run_status'] != 'completed' or file_hash(rep['raw_result_path']) != rep['raw_sha256']
                or any(raw.get(k) != v for k, v in cfg.items() if k != 'arrival_process')):
            raise RuntimeError('Exact cached repetition configuration or provenance mismatch')
    return result


class LiveAcceptedPath:
    """Measure only applied accepted changes, retaining unchanged worker processes."""
    def __init__(self, catalog, directory):
        self.catalog, self.directory = catalog, directory
        self.events = TimestampedEvents()
        self.workers, self.ready = {}, {}
        self.registry = default_registry()

    def load(self, name):
        worker = ObservedWorker(name, self.registry[name], self.events)
        self.workers[name] = worker
        worker.start()
        deadline = time.monotonic() + SETTINGS['startup_timeout_seconds']
        while time.monotonic() < deadline:
            if worker.process.poll() is not None:
                raise RuntimeError(f'{name} exited before READY')
            try:
                m, kind, payload, mono_ns, unix_ns = self.events.get(timeout=.1)
            except queue.Empty:
                continue
            if kind == 'READY' and m == name:
                if not self.catalog[name].supported_tasks <= set(payload.get('tasks', [])):
                    raise RuntimeError('READY capabilities mismatch')
                self.ready[name] = dict(unix_ns=unix_ns, monotonic_ns=mono_ns,
                    timestamp=datetime.fromtimestamp(unix_ns/1e9, timezone.utc).isoformat(),
                    payload=payload, pid=worker.process.pid)
                return
            if kind == 'ERROR':
                raise RuntimeError(f'{m} worker startup error: {payload}')
        raise TimeoutError(f'{name} READY timeout')

    def unload(self, name):
        worker = self.workers[name]
        worker.shutdown()
        forced = False
        try:
            worker.process.wait(timeout=SETTINGS['shutdown_timeout_seconds'])
        except subprocess.TimeoutExpired:
            forced = True
            worker.process.terminate()
            try:
                worker.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.process.kill()
                worker.process.wait(timeout=5)
        worker.reader_done.wait(timeout=2)
        worker.process.stdin.close()
        worker.process.stdout.close()
        self.workers.pop(name)
        self.ready.pop(name, None)
        return dict(forced=forced, exit_code=worker.process.returncode, exited_timestamp=stamp())

    def execute(self, start, path):
        result_path = self.directory / 'reconfiguration_execution.json'
        completed = read(result_path) if result_path.exists() else []
        # Recover only already completed actual events; setup residency is excluded.
        resident = completed[-1]['new_deployment'] if completed else start
        log = self.directory / 'accepted_path_workers.log'
        with log.open('a') as stream, redirect_stdout(stream):
            try:
                for name in deployment_key(resident):
                    self.load(name)
                setup = dict(purpose='unmetered initial residency; excluded from switch totals',
                             deployment=resident, workers_ready=self.ready, timestamp=stamp())
                append(self.directory / 'residency_setups.jsonl', setup)
                for i, item in enumerate(path):
                    if i < len(completed):
                        assert completed[i]['new_deployment'] == item['new_deployment']
                        continue
                    old, new = item['old_deployment'], item['new_deployment']
                    assert set(self.workers) == set(deployment_key(old))
                    added = sorted(set(deployment_key(new)) - set(self.workers))
                    removed = sorted(set(self.workers) - set(deployment_key(new)))
                    event = dict(event_index=i, mode=item['mode'], iteration=item.get('iteration'),
                        old_deployment=old, new_deployment=new, models_to_unload=removed,
                        models_to_load=added, switch_start_timestamp=stamp(),
                        workers_ready_timestamp=None, switch_ready_timestamp=None,
                        measured_switch_time_ms=None, cost=jsonable(item['cost']),
                        measurement_context='post-search accepted-path execution; no concurrent request arrivals',
                        retained_worker_pids={m:self.workers[m].process.pid for m in self.workers if m not in removed})
                    save_json(self.directory / 'active_switch.json', event)
                    event['unload_evidence'] = {m:self.unload(m) for m in removed}
                    for m in added:
                        self.load(m)
                    if any(w.process.poll() is not None for w in self.workers.values()):
                        raise RuntimeError('Retained worker exited during accepted transition')
                    event['workers_ready_timestamp'] = jsonable(self.ready)
                    event['switch_ready_timestamp'] = stamp()
                    event['measured_switch_time_ms'] = (
                        event['switch_ready_timestamp']['monotonic_ns'] -
                        event['switch_start_timestamp']['monotonic_ns']) / 1e6
                    event['cost']['measured_switch_time_ms'] = event['measured_switch_time_ms']
                    event['status'] = 'applied_all_workers_ready'
                    completed.append(event)
                    save_json(result_path, completed)
                    print(json.dumps(dict(event='accepted_switch_ready', time_ms=event['measured_switch_time_ms'],
                                          models_to_load=added, models_to_unload=removed)), flush=True)
                save_json(self.directory / 'final_residency_verified.json',
                    dict(deployment=resident if not path else path[-1]['new_deployment'],
                         workers_ready=self.ready, timestamp=stamp(), verification='all actual worker processes alive and READY'))
                if not result_path.exists():
                    save_json(result_path, [])
            finally:
                for name in list(self.workers):
                    self.unload(name)
        return completed


def run_strategy(phase, strategy, current, old_workload, workload, catalog, hardware, sources, history):
    directory = OUT / phase / strategy
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / 'result.json'
    if result_path.exists():
        return read(result_path)
    frozen_search = directory / 'search_result_frozen.json'
    evaluator = SelectedStateEvaluator(catalog, sources, phase+'__'+strategy, directory)
    if frozen_search.exists():
        record = read(frozen_search)
    else:
        start = stamp()
        callback = lambda r: save_json(directory / 'search_checkpoint.json', r.to_dict())
        cost_model = StructuralReconfigurationCost(catalog)
        if strategy == 'D0':
            canonical = solve_canonical_routing(current, workload, catalog, CONFIG.rho_max)
            runtime = evaluator.evaluate(current, workload, canonical.routing) if canonical else None
            record = dict(strategy=strategy, initial_deployment=current, final_deployment=current,
                feasible=canonical is not None, final_objective=jsonable(runtime),
                status='NO_ADAPT' if canonical else 'INFEASIBLE', switches=[],
                trace=dict(lp_solve_count=1, runtime_evaluation_count=int(canonical is not None),
                           milp_solve_count=0, search_iterations=0, iterations=[],
                           visited_deployments=[list(deployment_key(current))]),
                fallback_to_milp=False)
        elif strategy == 'D1':
            search = GuidedDeploymentSearch(evaluator, predictor=None,
                runtime_objective=OBJECTIVE, trace_callback=callback).search(
                    workload, catalog, hardware, CONFIG, initial_deployment=None)
            record = dict(strategy=strategy, initial_deployment=current,
                final_deployment=search.deployment, final_objective=jsonable(search.runtime),
                status=search.status, trace=search.to_dict(), fallback_to_milp=False,
                switches=jsonable(committed_switches(search, current, cost_model, include_cold_seed=True)))
        else:
            dynamic = DynamicAdaptation(evaluator, predictor=None, runtime_objective=OBJECTIVE,
                history=history, trace_callback=callback).adapt(current, old_workload, workload,
                    catalog, hardware, WARM_CONFIG, phase_id=phase)
            record = dict(strategy=strategy, initial_deployment=current,
                final_deployment=dynamic.final_deployment, final_objective=jsonable(dynamic.final_objective),
                status=dynamic.termination_reason, action=dynamic.action,
                trace=dynamic.adaptation_trace, dynamic=jsonable(dynamic),
                fallback_to_milp=dynamic.fallback_to_milp, switches=jsonable(dynamic.accepted_switches))
        record.update(search_start=start, search_end=stamp(), runtime_accesses=evaluator.records,
                      confirmation_influences_search=False, frozen_before_confirmation=True)
        save_json(frozen_search, record)
    if strategy != 'D0':
        record['executed_reconfiguration_events'] = LiveAcceptedPath(catalog, directory).execute(
            current, record['switches'])
    else:
        record['executed_reconfiguration_events'] = []
    save_json(result_path, record)
    return record


def confirmation(phase, strategy, record, workload, catalog, sources):
    directory = OUT / phase / strategy / 'confirmation'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'result.json'
    state = record['final_deployment']
    canonical = solve_canonical_routing(state, workload, catalog, CONFIG.rho_max)
    assert canonical is not None
    base = CampaignRuntimeEvaluator(OUT / 'search_cache', catalog, sources, SETTINGS,
        'identity_only', SEEDS, 45, 150)
    search_identity = base.identity(state, workload, canonical.routing)
    reps = []
    for i, seed in enumerate(CONFIRM_SEEDS, 1):
        identity = dict(deployment=state, workload=dict(workload.arrival_rate), routing=search_identity['routing'],
            measurement_settings=dict(SETTINGS, random_seed=seed), source_model_hashes=sources)
        key = digest(identity)
        rep_target = directory / f'rep_{i}.json'
        conf_cache = OUT / 'confirmation_cache' / (key+'.json')
        audit = dict(algorithm_id=phase+'__'+strategy, access_reason='post_search_selected_final_confirmation',
                     identity=identity, cache_key=key, timestamp=stamp(), cache_hit=False)
        rep = None
        if conf_cache.exists():
            saved = read(conf_cache)
            if saved['identity'] == identity:
                rep = saved['repetition']
                audit.update(cache_hit=True, imported_from=str(conf_cache))
        if rep is None:
            # Read saved confirmation metadata only after this final state is chosen;
            # runtime values for unvisited deployments never guide the search.
            historical_ledger = HIST / 'confirmation_outcomes.jsonl'
            for line in historical_ledger.read_text().splitlines():
                metadata = json.loads(line)
                if metadata.get('identity_sha256') == key and metadata.get('run_status') == 'completed':
                    p = (HIST / 'algorithm_runs' / metadata['run_id'] / 'confirmation' /
                         metadata['which'] / f"rep_{metadata['repetition']}.json")
                    rep = read(p)
                    audit.update(cache_hit=True, imported_from=str(p))
                    break
        if rep is not None:
            raw = read(rep['raw_result_path'])
            cfg = MeasurementConfig(state, dict(workload.arrival_rate), search_identity['routing'],
                random_seed=seed, **SETTINGS).as_dict()
            if (rep['run_status'] != 'completed' or file_hash(rep['raw_result_path']) != rep['raw_sha256']
                    or any(raw.get(k) != v for k, v in cfg.items() if k != 'arrival_process')):
                raise RuntimeError('Confirmation exact identity validation failed')
        append(OUT / 'dynamic_cache_access_audit.jsonl', audit)
        if rep is None:
            for retry in range(1, 4):
                attempts_path = OUT / 'confirmation_attempts.jsonl'
                n = len(attempts_path.read_text().splitlines()) if attempts_path.exists() else 0
                if n >= 40:
                    raise RuntimeError('Confirmation physical cap reached')
                raw_path = directory / f'rep_{i}_attempt_{n+1}.raw.json'
                log_path = raw_path.with_suffix('.log')
                cfg_path = raw_path.with_suffix('.config.json')
                cfg = MeasurementConfig(state, dict(workload.arrival_rate), search_identity['routing'],
                    random_seed=seed, **SETTINGS).as_dict()
                save_json(cfg_path, cfg)
                attempt = dict(attempt_id=n+1, algorithm_id=phase+'__'+strategy,
                    cache_key=key, identity=identity, started_timestamp=stamp(),
                    raw_result_path=str(raw_path), log_path=str(log_path), config_path=str(cfg_path))
                append(attempts_path, attempt)
                print(json.dumps(dict(event='confirmation_start', phase=phase, strategy=strategy, repetition=i)), flush=True)
                outcome = measurement_subprocess(cfg, raw_path, log_path, cfg_path)
                raw = read(raw_path) if raw_path.exists() else None
                if raw is not None and any(raw.get(k) != v for k, v in cfg.items() if k != 'arrival_process'):
                    outcome['configuration_mismatch'] = True
                status, reason = execution_status(raw, outcome, log_path)
                append(OUT / 'confirmation_outcomes.jsonl', dict(attempt, run_status=status,
                    reason=reason, outcome=outcome, ended_timestamp=stamp()))
                if status != 'completed':
                    archive = OUT / 'interruption_archive' / f'confirmation_{n+1}'
                    archive.mkdir(parents=True, exist_ok=True)
                    for p in (raw_path, log_path, cfg_path, raw_path.with_suffix('.process.json')):
                        if p.exists():
                            shutil.copy2(p, archive / p.name)
                    continue
                rep = summarize_repetition(raw, workload, catalog, dict(returncode=0), raw_path)
                rep.update(run_status='completed', post_search_confirmation=True, random_seed=seed,
                    repetition=i, measurement_identity_sha256=key)
                break
            if rep is None:
                raise RuntimeError('Three invalid confirmation attempts')
        save_json(conf_cache, dict(identity=identity, repetition=rep))
        save_json(rep_target, rep)
        reps.append(rep)
    agg = aggregate_repetitions(reps, 'confirmation', directory / 'aggregate.json')
    save_json(directory / 'aggregate.json', agg)
    result = dict(repetitions=reps, aggregate=jsonable(agg), final_deployment=state,
                  independent_of_search=True, decisions_unchanged=True)
    save_json(target, result)
    return result


def restore_history(history, phase, record, workload):
    history.last_deployment = dict(record['final_deployment'])
    history.phase_workloads[phase] = dict(workload.arrival_rate)
    history.phase_visited_deployments[phase] = {tuple(s) for s in record['trace']['visited_deployments']}


def performance(value):
    if value is None:
        return None
    result = {k:value[k] for k in ('stability_state_v2', 'weighted_mean_e2e', 'weighted_p95_e2e')}
    result['weighted_task_p95'] = result['weighted_p95_e2e']
    result['statistics'] = {}
    for metric in ('weighted_mean_e2e', 'weighted_p95_e2e'):
        values = [r[metric] for r in value['repetitions']]
        valid = [v for v in values if v is not None]
        result['statistics'][metric] = dict(mean=statistics.mean(valid) if valid else None,
            median=statistics.median(valid) if valid else None,
            std=statistics.stdev(valid) if len(valid)>1 else None, individual_repetitions=values)
    return result


def metrics(record):
    trace, accesses = record['trace'], record['runtime_accesses']
    events = record['executed_reconfiguration_events']
    first_steady = next((r['index'] for r in accesses if r['result']['stability_state_v2']=='steady'), None)
    final_index = next((r['index'] for r in accesses if r['deployment']==record['final_deployment']), None)
    endpoint = StructuralReconfigurationCost(load_model_catalog()).estimate(
        record['initial_deployment'], record['final_deployment'])
    return dict(performance=performance(record['final_objective']), status=record['status'],
        action=record.get('action'), initial_deployment=record['initial_deployment'],
        final_deployment=record['final_deployment'], fallback_to_milp=record['fallback_to_milp'],
        endpoint_structural_sensitivity=dict(models_loaded=endpoint.models_added,
            models_unloaded=endpoint.models_removed,
            VRAM_moved_mib=endpoint.added_vram_mib+endpoint.removed_vram_mib,
            switches=int(record['initial_deployment']!=record['final_deployment']),
            measured_switch_time_ms=None,
            note='Alternative direct old->final path structural cost only; not executed or timed'),
        search=dict(valid_runtime_evaluations=trace['runtime_evaluation_count'],
            valid_unique_runtime_evaluations=len({r['cache_key'] for r in accesses}),
            physical_attempts=sum(r['physical_attempts'] for r in accesses),
            cache_hits=sum(r['cache_hit'] for r in accesses),
            fresh_runtime_states=sum(not r['cache_hit'] for r in accesses),
            LP_solves=trace['lp_solve_count'], MILP_solves=trace['milp_solve_count'],
            canonical_cache_validation_LP_solves=len(accesses),
            iterations=trace['search_iterations']+trace.get('repair_iterations',0),
            visited_states=trace['visited_deployments'],
            evaluations_to_first_steady_state=first_steady,
            initial_evaluated_state_steady=bool(accesses and accesses[0]['result']['stability_state_v2']=='steady'),
            evaluations_to_final_accepted_state=final_index,
            time_to_first_steady_note='Counts state accesses including exact hits; no online wall-clock claim'),
        reconfiguration=dict(accepted_switches=len(events),
            local_search_switches=sum(e['mode']=='normal' for e in events),
            cold_seed_resets=sum(e['mode']=='cold_milp_initialization' for e in events),
            models_loaded=sum(len(e['models_to_load']) for e in events),
            models_unloaded=sum(len(e['models_to_unload']) for e in events),
            VRAM_added_mib=sum(e['cost']['added_vram_mib'] for e in events),
            VRAM_removed_mib=sum(e['cost']['removed_vram_mib'] for e in events),
            VRAM_moved_mib=sum(e['cost']['added_vram_mib']+e['cost']['removed_vram_mib'] for e in events),
            measured_switch_time_ms=sum(e['measured_switch_time_ms'] for e in events) if events else None,
            individual_switch_times_ms=[e['measured_switch_time_ms'] for e in events]))


def gap(a, b):
    return (a-b)/b if a is not None and b else None


def saving(a, b):
    return 1-a/b if b else None


def audit_search(record, catalog):
    errors = []
    trace = record['trace']
    accesses = record['runtime_accesses']
    if trace['runtime_evaluation_count'] != len(accesses):
        errors.append('runtime access count diverges from search trace')
    if len({r['cache_key'] for r in accesses}) > 6:
        errors.append('runtime unique budget exceeded')
    for t in trace['iterations']:
        ranked = sorted((c for c in t['candidates'] if c['rank'] is not None), key=lambda c:c['rank'])
        if [c['delta_z'] for c in ranked] != sorted((c['delta_z'] for c in ranked), reverse=True):
            errors.append('LP ranking')
        selected = [c for c in ranked if c['selected_top_b']]
        if selected != ranked[:2]:
            errors.append('Top-B selection')
        best = None
        for c in selected:
            if c['runtime'] is None:
                continue
            runtime = MeasurementRuntimeEvaluation(**c['runtime'])
            current = MeasurementRuntimeEvaluation(**t['current_runtime'])
            improves, _ = OBJECTIVE.compare(runtime, current, c['memory_mib'], memory_mib(t['deployment'],catalog))
            if improves and (best is None or OBJECTIVE.compare(runtime,
                    MeasurementRuntimeEvaluation(**best['runtime']), c['memory_mib'], best['memory_mib'])[0]):
                best = c
        expected = list(deployment_key(best['mutation']['deployment'])) if best else None
        if t['accepted'] != expected:
            errors.append('accept/rollback differs from unchanged runtime objective')
        accepted = [c for c in selected if c['status']=='ACCEPTED']
        if len(accepted) != int(best is not None):
            errors.append('accepted candidate label')
    current = record['initial_deployment']
    for switch, applied in zip(record['switches'], record['executed_reconfiguration_events']):
        if switch['old_deployment'] != current or applied['old_deployment'] != current:
            errors.append('accepted-path state discontinuity')
        if switch['new_deployment'] != applied['new_deployment']:
            errors.append('executed path differs from selected path')
        current = switch['new_deployment']
    if current != record['final_deployment'] or len(record['switches']) != len(record['executed_reconfiguration_events']):
        errors.append('final deployment did not follow applied path')
    return errors


def report(transitions, enumeration_calls):
    errors, rows, events = [], [], []
    current = read(OUT / 'pilot_config.json')['initial_deployment']
    for phase, strategies, confirmations in transitions:
        row = dict(transition=phase, strategies={s:metrics(r) for s,r in strategies.items()})
        for s, r in strategies.items():
            errors.extend(phase+'__'+s+': '+e for e in audit_search(r, load_model_catalog()))
            if r['initial_deployment'] != current:
                errors.append('Dynamic actual old deployment mismatch')
            events.extend(dict(e, transition=phase, strategy=s, drives_primary_sequence=s=='D2')
                          for e in r['executed_reconfiguration_events'])
        cold, warm = row['strategies']['D1'], row['strategies']['D2']
        cg, wg = cold['performance']['weighted_mean_e2e'], warm['performance']['weighted_mean_e2e']
        relative = gap(wg, cg)
        stable_close = STABILITY[warm['performance']['stability_state_v2']] >= STABILITY[cold['performance']['stability_state_v2']]
        close = stable_close and relative is not None and relative <= .05
        cheaper = (warm['search']['valid_runtime_evaluations'] < cold['search']['valid_runtime_evaluations'] or
                   warm['reconfiguration']['accepted_switches'] < cold['reconfiguration']['accepted_switches'])
        row.update(Gap_warm_cold=relative, latency_difference_ms=wg-cg if None not in (wg,cg) else None,
            Saving_eval=saving(warm['search']['valid_runtime_evaluations'],cold['search']['valid_runtime_evaluations']),
            Saving_switch=saving(warm['reconfiguration']['accepted_switches'],cold['reconfiguration']['accepted_switches']),
            close_to_cold=close, cheaper=cheaper, supports_warm=close and cheaper)
        conf = {s:performance(c['aggregate']) for s,c in confirmations.items()}
        cgap = gap(conf['D2']['weighted_mean_e2e'],conf['D1']['weighted_mean_e2e'])
        row['confirmation'] = dict(strategies=conf, Gap_warm_cold=cgap,
            mean_direction_maintained=(relative*cgap>=0 if relative is not None and cgap is not None else None),
            close_to_cold=STABILITY[conf['D2']['stability_state_v2']]>=STABILITY[conf['D1']['stability_state_v2']] and cgap is not None and cgap<=.05,
            search_decisions_unchanged=True)
        current = dict(strategies['D2']['final_deployment'])
        rows.append(row)
    if enumeration_calls:
        errors.append('enumeration calls nonzero')
    n_support = sum(r['supports_warm'] for r in rows)
    consistent_worse = all(not r['close_to_cold'] for r in rows)
    neutral = all(r['close_to_cold'] and abs(r['Gap_warm_cold'])<=.05 and not r['cheaper'] for r in rows)
    decision = 'FAIL' if errors or consistent_worse else 'PASS' if n_support>=2 else 'NEUTRAL' if neutral else 'PARTIAL'
    totals = {}
    for s in ('D1','D2'):
        totals[s] = dict(runtime_evaluations=sum(r['strategies'][s]['search']['valid_runtime_evaluations'] for r in rows),
            physical_attempts=sum(r['strategies'][s]['search']['physical_attempts'] for r in rows),
            **{k:sum(r['strategies'][s]['reconfiguration'][k] for r in rows) for k in
               ('accepted_switches','models_loaded','models_unloaded','VRAM_added_mib','VRAM_removed_mib','VRAM_moved_mib')},
            measured_switch_time_ms=sum(e['measured_switch_time_ms'] for e in events if e['strategy']==s))
    totals['runtime_evaluations_saved'] = totals['D1']['runtime_evaluations']-totals['D2']['runtime_evaluations']
    totals['Saving_eval'] = saving(totals['D2']['runtime_evaluations'],totals['D1']['runtime_evaluations'])
    totals['Saving_switch'] = saving(totals['D2']['accepted_switches'],totals['D1']['accepted_switches'])
    totals['endpoint_structural_sensitivity'] = {s:{k:sum(r['strategies'][s]['endpoint_structural_sensitivity'][k]
        for r in rows) for k in ('models_loaded','models_unloaded','VRAM_moved_mib','switches')} for s in ('D1','D2')}
    manifest = read(OUT / 'search_cache/runtime_cache_manifest.json')
    evals = [dict(cache_key=k, identity=e['identity'], imported_from=e.get('imported_from'),
                 result=read(e['aggregate'])) for k,e in manifest['configurations'].items()]
    conf_out = OUT / 'confirmation_outcomes.jsonl'
    conf_attempts = [json.loads(line) for line in conf_out.read_text().splitlines()] if conf_out.exists() else []
    all_attempts = manifest['run_attempts'] + conf_attempts
    instability = [dict(cache_key=e['cache_key'], repetition=r['repetition'], raw_result_path=r['raw_result_path'])
        for e in evals for r in e['result']['repetitions'] if r['stability_state_v2']=='unstable']
    interruptions = [a for a in all_attempts if a.get('run_status',a.get('status')) != 'completed']
    save_json(OUT / 'cold_vs_warm_metrics.json', dict(transitions=rows, totals=totals, decision=decision,
        confirmation_close_count=sum(r['confirmation']['close_to_cold'] for r in rows),
        enumeration_calls=enumeration_calls, audit_errors=errors))
    save_json(OUT / 'dynamic_search_trace.json', dict(transitions=[dict(transition=p, strategies=s) for p,s,c in transitions],
        state_driver='D2 only', enumeration_calls=enumeration_calls))
    save_json(OUT / 'dynamic_runtime_evaluations.json', dict(evaluations=evals, unique_exact_states=len(evals),
        physical_search_attempts=len(manifest['run_attempts']), physical_confirmation_attempts=len(conf_attempts),
        genuine_unstable_search_repetitions=instability, invalid_attempts=interruptions))
    save_json(OUT / 'reconfiguration_events.json', dict(events=events,
        setup_teardown_and_candidate_profiling_excluded=True))
    save_json(OUT / 'reconfiguration_costs.json', dict(totals=totals,
        transitions=[dict(transition=r['transition'],strategies={s:m['reconfiguration'] for s,m in r['strategies'].items()}) for r in rows],
        primary_sequence_strategy='D2', cold_comparison='same actual old state; includes explicit reset to MILP seed',
        vram_basis='frozen catalog model peak VRAM; structural values, not measured bytes transferred'))
    accesses = [json.loads(line) for line in (OUT / 'dynamic_cache_access_audit.jsonl').read_text().splitlines()]
    save_json(OUT / 'dynamic_cache_access_audit.json', dict(accesses=accesses,
        policy='each access after algorithm chooses that state; exact routing, source/model/hardware/config; confirmation separate',
        speculative_candidate_result_queries=0))
    save_json(OUT / 'decision_audit.json', dict(passed=not errors, errors=errors, enumeration_calls=enumeration_calls,
        state_propagation_verified=True, runtime_objective_replayed=True))
    def active(x): return '+'.join(deployment_key(x))
    def fmt(v): return 'null' if v is None else f'{v:.3f}'
    lines = ['# Module 5B：Real Dynamic Transition Pilot', '', f'当前证据：**{decision}**；decision audit={"PASS" if not errors else "FAIL"}。', '',
        '冻结 W1→W2→W3→W1，arrival rates 未改。Predictor=None、canonical routing、Module 3 的三态目标和 5% 接受门槛保持原样。D0/D1 不驱动主序列。', '',
        '| Transition | Actual warm 起点 → 终点 | Action | D0 / cold / warm mean ms | Cold / warm P95 ms | Cold / warm evals | Gap | Saving eval |',
        '|---|---|---|---|---|---|---|---|']
    for r in rows:
        d0,c,w=(r['strategies'][s] for s in ('D0','D1','D2'))
        lines.append(f"| {r['transition']} | {active(w['initial_deployment'])} → {active(w['final_deployment'])} | {w['action']} | "
            f"{fmt(d0['performance']['weighted_mean_e2e']) if d0['performance'] else 'INFEASIBLE'} / {fmt(c['performance']['weighted_mean_e2e'])} / {fmt(w['performance']['weighted_mean_e2e'])} | "
            f"{fmt(c['performance']['weighted_p95_e2e'])} / {fmt(w['performance']['weighted_p95_e2e'])} | {c['search']['valid_runtime_evaluations']} / {w['search']['valid_runtime_evaluations']} | {r['Gap_warm_cold']:.2%} | {r['Saving_eval']:.2%} |")
    lines += ['', '## Confirmation 与状态', '']
    for r in rows:
        c,w=r['confirmation']['strategies']['D1'],r['confirmation']['strategies']['D2']
        lines.append(f"- {r['transition']}：cold/warm independent confirmation mean={fmt(c['weighted_mean_e2e'])}/{fmt(w['weighted_mean_e2e'])} ms；gap={r['confirmation']['Gap_warm_cold']:.2%}；mean direction maintained={r['confirmation']['mean_direction_maintained']}；warm close={r['confirmation']['close_to_cold']}。")
    lines += ['', '## Reconfiguration 与搜索成本', '',
        '| Transition / strategy | Accepted switches | Load / unload | VRAM moved MiB | Actual measured switches ms | Search physical attempts | LP solves | Iterations | First steady / final accepted eval |',
        '|---|---:|---|---:|---|---:|---:|---:|---|']
    for r in rows:
        for s in ('D1','D2'):
            m=r['strategies'][s];rc=m['reconfiguration'];sc=m['search']
            lines.append(f"| {r['transition']} / {s} | {rc['accepted_switches']} | {rc['models_loaded']} / {rc['models_unloaded']} | {rc['VRAM_moved_mib']:.1f} | {', '.join(fmt(v) for v in rc['individual_switch_times_ms']) or 'null; no switch'} | {sc['physical_attempts']} | {sc['LP_solves']} | {sc['iterations']} | {sc['evaluations_to_first_steady_state']} / {sc['evaluations_to_final_accepted_state']} |")
    lines += ['', f"累计 cold/warm state evaluations={totals['D1']['runtime_evaluations']}/{totals['D2']['runtime_evaluations']}；节省 {totals['runtime_evaluations_saved']} 次（{totals['Saving_eval']:.2%}）。",
        f"累计 cold/warm models loaded={totals['D1']['models_loaded']}/{totals['D2']['models_loaded']}；unloaded={totals['D1']['models_unloaded']}/{totals['D2']['models_unloaded']}；accepted switches={totals['D1']['accepted_switches']}/{totals['D2']['accepted_switches']}；Saving_switch={totals['Saving_switch']:.2%}。",
        f"实际新执行 search physical attempts={len(manifest['run_attempts'])}；confirmation physical attempts={len(conf_attempts)}。Exact cache hits 与实际新 GPU state measurements 分开记录；evaluation 节省是算法访问 state 的计数，不冒充独立 GPU run 节省。", '',
        '## 14 项核对', '']
    for i,r in enumerate(rows[:2],1):
        w=r['strategies']['D2'];lines.append(f"{i}. {r['transition']} actual warm：{active(w['initial_deployment'])} → {active(w['final_deployment'])}。")
    lines += [f"3. W3→W1 action={rows[2]['strategies']['D2']['action']}；未强制 mutation。",
        '4. 每个 transition 的 cold/warm evaluation 数见首表；均 <=6，含起点，3 completed repetitions/state。',
        '5. Warm/cold final latency 的绝对差与 Gap 见 cold_vs_warm_metrics.json；mean/P95/stability 及 individual repetitions 全部保留。',
        '6. Confirmation 见上表；2 independent seeds，与搜索隔离，无任何 post-confirmation decision 更新。',
        '7. 装载/卸载累计与逐 transition 数见上表；D1 从同一 actual old 显式执行 seed reset，另列 local search switches 和 cold seed reset。',
        f"8. 真实 READY-based switch time 成功测量 {len(events)} 次；所有无 switch 阶段时间=null。没有将 benchmark repetitions 计作 switch latency。",
        f"9. Warm MILP fallback：{[r['strategies']['D2']['fallback_to_milp'] for r in rows]}。",
        f"10. Genuine unstable search repetitions={len(instability)}；confirmation unstable={sum(rep['stability_state_v2']=='unstable' for p,s,c in transitions for v in c.values() for rep in v['repetitions'])}。Completed instability 保留，不重标为外部中断。",
        f"11. External interruptions={sum(a.get('run_status',a.get('status'))=='external_interruption' for a in interruptions)}；全部 invalid attempts={len(interruptions)}。无证据的缺失 raw 为 uncertain，不猜测原因。",
        f"12. Enumeration calls={enumeration_calls}（runtime tripwire + audit）。",
        f"13. 累计节省 {totals['runtime_evaluations_saved']} runtime evaluations（{totals['Saving_eval']:.2%}）。",
        f"14. 当前 pilot={decision}；close+cheaper transitions={n_support}/3；confirmation close={sum(r['confirmation']['close_to_cold'] for r in rows)}/3。", '',
        '## 解释边界', '',
        'Accepted-path worker activation 在 search 决策冻结后真实执行，保留未变化 workers。为避免 GPU 干扰，candidate profiling、D1 counterfactual 和 D2 activation 的进程 residency 分开；这是显式 phase 的动态 deployment pilot，没有持续到达请求下的零停机证明。Setup residency 和 teardown 另列且不计 accepted switch；全流程实际物理 load/unload 不能由 structural accepted-path totals 代表。',
        'VRAM moved 来自冻结 catalog 的模型峰值 MiB，仅表示结构重配置量。Switch time 来自同一进程 monotonic clock，包含 accepted removals 和 loads 到所有 workers READY；不含 warmup、candidate profiling 或 confirmation。',
        '冷搜索最终部署也给出从 actual old 到 final 的 endpoint structural sensitivity（见 README），避免只依赖 seed reset 的成本优势。',
        'Close 门槛预先固定为 warm 同等或更好 stability 且 mean <= cold×1.05。所有统计仅是 pilot 描述，不声称统计显著性。Confirmation 数字不覆盖 search objective；排序接近时 direction 可以翻转。',
        '已停止于 Module 5B；未进入 forecasting、change detection、prewarming、multi-replica、routing reoptimization 或 final paper benchmark。', '']
    (OUT / 'module5b_summary.md').write_text('\n'.join(lines))
    return decision


def run():
    records, sources, current = prepare()
    catalog, hardware = load_model_catalog(), load_hardware()
    baseline_dir = OUT / 'phase0_baseline'
    baseline_dir.mkdir(exist_ok=True)
    workload0 = Workload(records[NAMES[0]]['arrival_rates_rps'])
    canonical = solve_canonical_routing(current, workload0, catalog, CONFIG.rho_max)
    assert canonical is not None
    if not (baseline_dir / 'result.json').exists():
        evaluator = SelectedStateEvaluator(catalog, sources, 'phase0_baseline', baseline_dir)
        runtime = evaluator.evaluate(current, workload0, canonical.routing)
        save_json(baseline_dir / 'result.json', dict(deployment=current, canonical=jsonable(canonical),
            runtime=jsonable(runtime), runtime_accesses=evaluator.records,
            note='Module 3C W1 B2 only supplies initial system state; subsequent state comes exclusively from D2'))
    history = AdaptationHistory()
    transitions = []
    with forbid_enumeration() as guards:
        for i, phase in enumerate(PHASES):
            old = Workload(records[NAMES[i]]['arrival_rates_rps'])
            new = Workload(records[NAMES[i+1]]['arrival_rates_rps'])
            freeze(OUT / phase / 'transition_input.json', dict(actual_current_deployment=current,
                old_workload=dict(old.arrival_rate), new_workload=dict(new.arrival_rate), primary_driver='D2'))
            strategies = {}
            for strategy in ('D0','D1','D2'):
                print(json.dumps(dict(event='strategy_start', phase=phase, strategy=strategy, actual_current=current)), flush=True)
                strategies[strategy] = run_strategy(phase, strategy, current, old, new, catalog, hardware, sources, history)
                print(json.dumps(dict(event='strategy_complete', phase=phase, strategy=strategy,
                    final=strategies[strategy]['final_deployment'], status=strategies[strategy]['status'])), flush=True)
            warm = strategies['D2']
            if warm['action'] == 'INFEASIBLE':
                raise RuntimeError('D2 could not restore feasible actual system state')
            restore_history(history, phase, warm, new)
            current = dict(warm['final_deployment'])
            save_json(OUT / 'actual_dynamic_state.json', dict(completed_phase=phase,
                actual_current_deployment=current, history=jsonable(history)))
            confirmations = {s:confirmation(phase,s,strategies[s],new,catalog,sources) for s in ('D1','D2')}
            transitions.append((phase, strategies, confirmations))
        calls = sum(g.call_count for g in guards)
    verify_assets_unchanged(read(ROOT / 'results/deployment_search_real_pilot/asset_hashes.json'))
    print(json.dumps(dict(event='pilot_complete', decision=report(transitions,calls))), flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT / '.pilot.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.prepare_only:
            prepare()
        else:
            run()


if __name__=='__main__':
    main()
