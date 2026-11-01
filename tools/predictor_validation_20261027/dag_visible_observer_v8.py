"""Reader-visible native feedback for a complete public DAG cohort.

Observed host facts never restore a simulated checkpoint or expose a CUDA
cursor. Only an empty, visibly ready cutoff can restart the future suffix;
busy residual service and dispatch delays require separate evidence.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from gsdeploy.forecasting.observed_state import (
    advance_observed_state, feedback_from_runtime_records, new_observed_state,
)
from gsdeploy.forecasting.metrics import score_forecast
from gsdeploy.forecasting.workload import FutureWorkload
from gsdeploy.resources import identity
from gsdeploy.routing import Router
from .dag_checkpoint_v7 import receipt
from .forecast_gate_v2 import INPUT_SCHEMA, _public_cohort

SCHEMA = 'gsdeploy.reader_visible_dag_cutoff.v8'
PUBLIC_FIELDS = {'request_id', 'alias', 'task', 'pool_id', 'arrival_ns'}


def _ns(value, name):
    if type(value) is not int or value < 0:
        raise ValueError('Nonnegative integer ' + name + ' required')
    return value


def _hash(document):
    return identity({k: v for k, v in document.items() if k != 'snapshot_sha256'})


def public_dag_cohort(public, aliases, routing_rps):
    """Materialize the gate's entire declared schedule with its original routes."""
    _public_cohort(dict(schema=INPUT_SCHEMA, forecast_inputs=public))
    router = Router(routing_rps)
    initial = [copy.deepcopy(r) for rows in public['queues'].values() for r in rows]
    for row in sorted(initial, key=lambda r: (r['arrival_ns'], r['task'], r['request_id'])):
        if router.choose(row['task']) != row['alias']:
            raise ValueError('Initial route differs from whole-cohort continuation policy')
    workload = FutureWorkload(**public['future_workload'])
    future = workload.suffix(public['phase_start_ns'], public['now_ns'], public['deadline_ns'])
    rows = initial + [dict(r, alias=router.choose(r['task'])) for r in future['arrivals']]
    return _cohort(rows, aliases, public['phase_start_ns'], public['deadline_ns'])


def _cohort(rows, aliases, start, deadline):
    result = {}
    if isinstance(rows, dict):
        rows = list(rows.values())
    for row in rows:
        if (set(row) != PUBLIC_FIELDS or not isinstance(row['request_id'], str)
                or not row['request_id'] or row['request_id'] in result
                or row['alias'] not in aliases
                or row['task'] not in aliases[row['alias']]['supported_tasks']
                or not isinstance(row['pool_id'], str) or not row['pool_id']
                or not start <= _ns(row['arrival_ns'], 'public arrival') < deadline):
            raise ValueError('Complete nonduplicated public DAG identity/arrival/route required')
        result[row['request_id']] = copy.deepcopy(row)
    return result


def _visible_records(records, cutoff, *, expected=None, aliases=None, ready=None):
    """Select fields after checking visibility, before touching response bodies."""
    clean, submitted, seen = [], {}, set()
    for record in records:
        rid, alias = record['request_id'], record['model']
        if rid in seen or alias not in aliases:
            raise ValueError('Duplicate or foreign native request/owner')
        seen.add(rid)
        if expected is not None:
            if rid not in expected or any(record[key] != expected[rid][public] for key, public in
                    (('model', 'alias'), ('task', 'task'), ('pool_id', 'pool_id'), ('scheduled_time_ns', 'arrival_ns'))):
                raise ValueError('Native record changed public cohort/route/task/input/arrival')
        row = dict(request_id=rid, model=alias, execution_events=[])
        sent = record.get('actual_submit_time_ns')
        if sent is not None and _ns(sent, 'actual submit') <= cutoff:
            submitted[rid] = sent
        for event in record.get('execution_events', ()):
            if _ns(event.get('received_time_ns'), 'reader event receipt') > cutoff:
                continue
            fields = ('request_id', 'perf_ns', 'received_time_ns')
            if 'event' in event:
                fields += ('event', 'segment_id', 'kind')
            selected = {key: event[key] for key in fields}
            if selected['request_id'] != rid:
                raise ValueError('Native event changed request identity')
            if 'event' not in event and event.get('pid') != ready[alias]['pid']:
                raise ValueError('Native START does not bind the READY owner PID')
            row['execution_events'].append(selected)
        received = record.get('response_reader_event_enqueue_time_ns')
        if received is not None and _ns(received, 'RESULT reader receipt') <= cutoff:
            # No token count, quality, prediction, terminal token, or future
            # response body is accessed. END is selected only after receipt.
            response = record.get('worker_response')
            if response is not None and response.get('end_perf_ns') is not None:
                row['response_reader_event_enqueue_time_ns'] = received
                row['worker_response'] = dict(end_perf_ns=response['end_perf_ns'])
        next_ready = record.get('previous_handler_next_request_ready')
        if next_ready is not None and _ns(next_ready.get('received_time_ns'), 'READY reader receipt') <= cutoff:
            if next_ready.get('pid') != ready[alias]['pid']:
                raise ValueError('Native readiness does not bind the READY owner PID')
            row['previous_handler_next_request_ready'] = {key: next_ready[key] for key in
                ('request_id', 'perf_ns', 'clock', 'boundary', 'scope', 'received_time_ns')}
        clean.append(row)
    return clean, submitted


def _observe(clean, aliases, context, cutoff):
    state = new_observed_state(aliases, context, mode='frozen', frozen_cutoff_ns=cutoff)
    feedback = feedback_from_runtime_records(clean, context, cutoff)
    return advance_observed_state(state, cutoff, feedback=feedback)


def _bound_plan(raw, cohort, aliases, start, deadline, *, fixture=False,
                allow_retrospective_training_arrivals=False):
    """Rebuild preceding work inventory from the executed public plan."""
    if raw['schema'] == 'gsdeploy.cpu_fixture.native.v8' and fixture:
        expected_initial = raw.get('initial_request_inventory', [])
        return 'CPU_FIXTURE_DECLARED_SCHEDULE', expected_initial, []
    ref = raw.get('plan')
    if not isinstance(ref, dict) or receipt(ref['path']) != ref:
        raise ValueError('Executed native public plan receipt required')
    plan = json.loads(Path(ref['path']).read_text())
    if plan.get('execution_context_sha256') != raw['execution_context_sha256']:
        raise ValueError('Executed plan context differs from native raw')
    sources = [copy.deepcopy(ref)]
    candidate = None
    if raw['schema'] == 'gsdeploy.native_flow_collection.v1':
        candidates = [c for c in plan['candidates'] if c['id'] == raw['candidate_id']]
        if len(candidates) != 1 or plan['aliases'] != aliases:
            raise ValueError('Native candidate/owner plan binding differs')
        candidate = candidates[0]
        expected = {r['request_id']: dict(request_id=r['request_id'], task=r['task'],
            pool_id=r['pool_id'], alias=candidate['routes'][r['request_id']],
            arrival_ns=start+r['offset_ns']) for r in plan['trace']}
        if (expected != cohort or deadline-start != round(plan['window_seconds']*1e9)
                or raw.get('deadline_ns') != deadline):
            raise ValueError('Full declared trace/routes/window differ from native plan')
        ref = plan['profile_plan']
        if receipt(ref['path']) != ref:
            raise ValueError('Priming input profile plan changed')
        profile = json.loads(Path(ref['path']).read_text()); sources.append(copy.deepcopy(ref))
        scope = 'ORIGINAL_DECLARED_SCHEDULE_FIXED_ROUTES'
    else:
        if (not allow_retrospective_training_arrivals or plan.get('training_only') is not True
                or any(r.get('split') != 'training' for r in raw['request_records'])):
            raise ValueError('Adaptive profile arrivals require explicit retrospective training diagnostic opt-in')
        profile = plan
        jobs = {j['id']: j for j in plan['jobs']}
        for record in raw['request_records']:
            job = jobs[record['profile_job_id']]
            if job['inputs'].get(record['model']) != record['pool_id']:
                raise ValueError('Retrospective public input differs from planned training job')
        scope = 'RETROSPECTIVE_TRAINING_ARRIVAL_DIAGNOSTIC_NOT_PROSPECTIVE'
    expected_initial = []
    from .profiling import RESIDENTS
    if set(aliases) != set(RESIDENTS):
        raise ValueError('Original fixed-resident priming recipe required')
    for alias, family in RESIDENTS.items():
        tasks = ['text','image','asr'] if family == 'generalist_2' else ['ocr' if family == 'ocr' else 'text']
        for task in tasks:
            selected = sorted((x for x in profile['inputs'] if x['task'] == task),
                              key=lambda x:x['input_size'], reverse=True)[:3]
            for item in selected:
                expected_initial.append(dict(request_id='priming:'+str(len(expected_initial)),
                    model=alias, task=task, pool_id=item['pool_id']))
    if candidate is not None:
        for index,item in enumerate(plan['trace']):
            if item['offset_ns'] >= plan['warmup_seconds']*1e9:
                break
            expected_initial.append(dict(request_id='warmup:'+str(index),
                model=candidate['routes'][item['request_id']], task=item['task'], pool_id=item['pool_id']))
    return scope, expected_initial, sources


def observe_cutoff(raw, public_rows, aliases, *, execution_context_sha256,
                   phase_start_ns, deadline_ns, drain_deadline_ns, cutoff_ns,
                   raw_source=None, allow_cpu_fixture=False,
                   allow_retrospective_training_arrivals=False):
    """Adapt an immutable native registry to an actual DAG forecast boundary.

    ``public_rows`` comes from the declared planning schedule, never future
    results. A persisted replay binds the raw byte receipt. Live callers must
    persist a consistent registry snapshot before using this entry point.
    """
    start, cutoff, deadline = map(lambda pair: _ns(*pair), (
        (phase_start_ns, 'phase start'), (cutoff_ns, 'cutoff'), (deadline_ns, 'deadline')))
    if not start <= cutoff <= deadline or start >= deadline:
        raise ValueError('Original measurement window and visible cutoff required')
    if type(drain_deadline_ns) is not int or drain_deadline_ns < deadline:
        raise ValueError('Original complete drain window required')
    if isinstance(raw, (str, Path)):
        source = receipt(raw)
        if raw_source is not None and source != raw_source:
            raise ValueError('Native raw receipt changed')
        raw_source = source
        raw = json.loads(Path(source['path']).read_text())
    elif not allow_cpu_fixture:
        raise ValueError('Actual replay requires a native raw path, not an unbound document')
    if raw_source is not None and receipt(raw_source['path']) != raw_source:
        raise ValueError('Native raw receipt changed')
    if (raw.get('execution_context_sha256') != execution_context_sha256
            or raw.get('phase_start_ns', raw.get('observation_start_ns')) != start
            or raw.get('schema') not in ('gsdeploy.native_flow_collection.v1',
                'gsdeploy.startup_native_phase_collection.v4', 'gsdeploy.cpu_fixture.native.v8')):
        raise ValueError('Native execution/measurement identity differs from public DAG')
    ready = raw.get('ready', {})
    if set(ready) != set(aliases) or any(type(ready[a].get('pid')) is not int for a in aliases):
        raise ValueError('Actual READY context for every resident owner required')
    cohort = _cohort(public_rows, aliases, start, deadline)
    if len(raw['request_records']) != len(cohort) or {r['request_id'] for r in raw['request_records']} != set(cohort):
        raise ValueError('Native registry must retain the complete original public cohort')
    arrival_scope, expected_initial, plan_sources = _bound_plan(raw, cohort, aliases, start, deadline,
        fixture=allow_cpu_fixture, allow_retrospective_training_arrivals=allow_retrospective_training_arrivals)
    clean, submitted = _visible_records(raw['request_records'], cutoff, expected=cohort,
                                       aliases=aliases, ready=ready)
    state = _observe(clean, aliases, execution_context_sha256, cutoff)
    if set(state['requests']) - set(submitted):
        raise ValueError('Visible execution lacks its causal actual submission')
    if any(at < cohort[rid]['arrival_ns'] for rid, at in submitted.items()):
        raise ValueError('Actual submission precedes public arrival')
    for rid, observed in state['requests'].items():
        if observed['handler_start_ns'] is None or observed['handler_start_ns'] < submitted[rid]:
            raise ValueError('Native execution precedes public arrival/submission')
    # Startup READY alone does not establish idle after priming/warmup. Every
    # resident must have its complete preceding work explicitly observed ready.
    initial_records = list(raw.get('priming_request_records', ())) + list(raw.get('warmup_request_records', ()))
    actual_initial = [{k:r[k] for k in ('request_id','model','task','pool_id')} for r in initial_records]
    if actual_initial != expected_initial:
        raise ValueError('Complete planned priming/warmup request inventory required')
    if any(r['request_id'] in cohort for r in initial_records):
        raise ValueError('Initial priming and measurement cohort overlap')
    initial_clean, initial_submitted = _visible_records(initial_records, start, aliases=aliases, ready=ready)
    initial_state = _observe(initial_clean, aliases, execution_context_sha256, start)
    initial_idle = {}
    for alias in aliases:
        rows = [r for r in initial_clean if r['model'] == alias]
        known = bool(rows) and all(r['request_id'] in initial_submitted and
            initial_state['requests'].get(r['request_id'], {}).get('worker_next_ready_ns') is not None for r in rows)
        initial_idle[alias] = dict(known=known, worker_pid=ready[alias]['pid'],
            basis='bound_native_priming_warmup_reader_READY_before_phase_start',
            request_ids=[r['request_id'] for r in rows])
    statuses, owners, completed, reasons = {}, {}, [], []
    for rid, row in cohort.items():
        observed = state['requests'].get(rid, {})
        if row['arrival_ns'] > cutoff:
            if rid in submitted or observed:
                raise ValueError('Native feedback precedes its public arrival')
            status = 'FUTURE_DECLARED'
        elif rid not in submitted:
            status = 'ARRIVED_DISPATCH_UNOBSERVED'
        elif observed.get('handler_start_ns') is None:
            status = 'SUBMITTED_START_NOT_VISIBLE'
        elif observed.get('handler_end_ns') is None:
            status = 'STARTED_RESIDUAL_UNKNOWN'
        elif observed.get('worker_next_ready_ns') is None:
            status = 'HANDLER_END_READY_NOT_VISIBLE'
        else:
            status = 'COMPLETED_READY_VISIBLE'
        statuses[rid] = status
        if observed.get('handler_end_ns') is not None:
            completed.append(dict(row, completion_ns=observed['parent_receipt_ns']))
        if status not in ('FUTURE_DECLARED', 'COMPLETED_READY_VISIBLE'):
            reasons.append(status + ':' + rid)
    for alias in aliases:
        owned = [rid for rid, row in cohort.items() if row['alias'] == alias]
        unresolved = [rid for rid in owned if statuses[rid] not in ('FUTURE_DECLARED', 'COMPLETED_READY_VISIBLE')]
        if not initial_idle[alias]['known']:
            reasons.append('INITIAL_OWNER_IDLE_NOT_VERIFIED:' + alias)
        owners[alias] = dict(status='IDLE_READY_OBSERVED' if not unresolved and initial_idle[alias]['known'] else 'UNRESOLVED',
            unresolved_request_ids=unresolved, kernel_cursor=None, kernel_cursor_status='UNKNOWN')
    # No observed prefix is replaced by zero. Board sampling is not a complete
    # resource integral or an instantaneous owner-memory peak.
    resources = dict(energy_known=False, memory_known=False, board_energy_joules=None,
        peak_memory_mib=None, window_ns=[start, cutoff],
        reason='COMPLETE_VISIBLE_PREFIX_RESOURCE_EVIDENCE_NOT_PROVIDED')
    snapshot = dict(schema=SCHEMA, clock='host_perf_counter_ns', execution_context_sha256=execution_context_sha256,
        raw_source=copy.deepcopy(raw_source), plan_sources=plan_sources, arrival_scope=arrival_scope,
        aliases=copy.deepcopy(aliases), phase_start_ns=start,
        cutoff_ns=cutoff, deadline_ns=deadline, drain_deadline_ns=drain_deadline_ns,
        full_public_cohort=cohort, observed_host_state=state, visible_submissions=submitted,
        visible_feedback=feedback_from_runtime_records(clean, execution_context_sha256, cutoff),
        request_status=statuses, owners=owners, initial_owner_idle=initial_idle,
        completed_prefix=completed, future_suffix=[copy.deepcopy(row) for rid, row in cohort.items() if statuses[rid] == 'FUTURE_DECLARED'],
        prefix_resources=resources, unknown_reasons=sorted(reasons),
        simulated_checkpoint=None, observed_GPU_cursor=False, busy_residual_model_available=False,
        rolling_risk_correction_complete=False, physical_accuracy_validated=False)
    if any(receipt(ref['path']) != ref for ref in plan_sources):
        raise ValueError('Executed plan changed during cutoff adaptation')
    if raw_source is not None and receipt(raw_source['path']) != raw_source:
        raise ValueError('Native raw changed during cutoff adaptation')
    snapshot['snapshot_sha256'] = _hash(snapshot)
    return snapshot


def forecast_visible_suffix(snapshot, engine_factory, *, score_kwargs=None):
    """Call the actual DAG engine at an empty idle cutoff, then score all rows.

    ``engine_factory(now_ns=...)`` must construct the same selected providers
    and aliases, with fresh queue diagnostics at the visible cutoff. Busy
    states return UNKNOWN without calling the engine. This is a suffix restart,
    never a restoration or correction of simulated active GPU work.
    """
    if snapshot.get('schema') != SCHEMA or snapshot.get('snapshot_sha256') != _hash(snapshot):
        raise ValueError('Mutated or foreign visible DAG snapshot')
    common = dict(schema='gsdeploy.visible_dag_suffix_result.v8',
        full_public_cohort=copy.deepcopy(snapshot['full_public_cohort']),
        completed_prefix=copy.deepcopy(snapshot['completed_prefix']),
        visible_snapshot_sha256=snapshot['snapshot_sha256'],
        prefix_resources=copy.deepcopy(snapshot['prefix_resources']),
        simulated_checkpoint_used=False, observed_GPU_cursor=False,
        busy_residual_model_available=False, rolling_risk_correction_complete=False,
        deployment_qualified=False, GPU_started=False)
    if snapshot['unknown_reasons']:
        return dict(common, status='UNKNOWN', unknown_reasons=snapshot['unknown_reasons'], forecast=None, metrics=None)
    cutoff = snapshot['cutoff_ns']
    model = engine_factory(now_ns=cutoff)
    if model.aliases != snapshot['aliases']:
        raise ValueError('Suffix engine changed resident owner bindings')
    forecast = model.forecast(snapshot['future_suffix'], cutoff, snapshot['deadline_ns'],
                             drain_deadline_ns=snapshot['drain_deadline_ns'])
    expected = {r['request_id']: r for r in snapshot['future_suffix']}
    if forecast['cohort'] != expected:
        raise ValueError('Actual DAG engine dropped or changed complete future schedule')
    completed = snapshot['completed_prefix']
    if set(expected) | {r['request_id'] for r in completed} != set(snapshot['full_public_cohort']):
        raise ValueError('Visible prefix and future suffix lost the original scoring cohort')
    metrics = None
    if score_kwargs is not None:
        forbidden = {'phase_start_ns', 'now_ns', 'deadline_ns', 'observed_peak_memory_mib',
                     'observed_board_energy_joules', 'aliases', 'completed', 'forecast'}
        if forbidden & set(score_kwargs):
            raise ValueError('Scoring cannot replace observed window/prefix/owner bindings')
        if 'weights' in score_kwargs and score_kwargs['weights'] != {k: 1/6 for k in 'QTLPME'}:
            raise ValueError('Original six equal objective weights required')
        resources = snapshot['prefix_resources']
        metrics = score_forecast(forecast, completed, snapshot['aliases'],
            phase_start_ns=snapshot['phase_start_ns'], now_ns=cutoff, deadline_ns=snapshot['deadline_ns'],
            observed_peak_memory_mib=resources['peak_memory_mib'],
            observed_board_energy_joules=resources['board_energy_joules'], **score_kwargs)
        if not resources['energy_known']:
            metrics['terms']['E'] = None
            metrics['unknown_reasons'].append('UNKNOWN_VISIBLE_PREFIX_BOARD_ENERGY')
            metrics.update(cost=None, full_U=None, six_term_comparison_complete=False, absolute_utility_complete=False)
            metrics['E_scope'] = 'Whole original window requires observed prefix plus predicted suffix; prefix missing'
    return dict(common, status='IDLE_SUFFIX_FORECAST', unknown_reasons=[], forecast=forecast, metrics=metrics,
                suffix_restart_only=True,
                suffix_energy_diagnostic_joules=forecast.get('board_energy_joules'))
