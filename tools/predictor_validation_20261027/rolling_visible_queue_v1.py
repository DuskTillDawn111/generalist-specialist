"""Causal visible-prefix rolling interface and real FullHorizon continuation.

This does not resume v54's lifecycle engine: that engine has no compatible
checkpoint contract. Its main-path adapter returns explicit UNKNOWN. The
separate segment backend can continue only with directly observed current
progress or waiting evidence; host age never becomes a GPU work cursor.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.metrics import score_forecast
from gsdeploy.forecasting.observed_state import (advance_observed_state,
    feedback_from_runtime_records, forecast_snapshot_from_observed, new_observed_state)
from gsdeploy.forecasting.queue_diagnostics import remaining_work, workload_envelopes
from gsdeploy.forecasting.simulator import FullHorizonModel
from gsdeploy.resources import identity
from .native_collect import receipt

SCHEMA = 'gsdeploy.rolling_visible_queue_session.v1'
PUBLIC = {'request_id', 'alias', 'task', 'pool_id', 'arrival_ns'}


def require(condition, reason):
    if not condition: raise ValueError(reason)


def hashed(document):
    return identity({k:v for k,v in document.items() if k != 'session_sha256'})


def seal(document):
    document['session_sha256'] = hashed(document)
    return document


def validate(session):
    require(session.get('schema') == SCHEMA and session.get('session_sha256') == hashed(session),
            'Exact immutable rolling observer session required')


def new_session(rows, aliases, execution_context_sha256, phase_start_ns, deadline_ns, drain_deadline_ns,
                *, mode='online', frozen_cutoff_ns=None):
    require(all(type(t) is int and t >= 0 for t in (phase_start_ns, deadline_ns, drain_deadline_ns))
            and phase_start_ns < deadline_ns <= drain_deadline_ns, 'Original full measurement/drain windows required')
    cohort = {}
    for row in rows:
        require(set(row) == PUBLIC and row['request_id'] not in cohort and row['alias'] in aliases
                and row['task'] in aliases[row['alias']]['supported_tasks'] and type(row['arrival_ns']) is int
                and phase_start_ns <= row['arrival_ns'] < deadline_ns, 'Exact public identity/route/complete arrival schedule required')
        cohort[row['request_id']] = copy.deepcopy(row)
    require(cohort and list(cohort.values()) == sorted(cohort.values(), key=lambda r:r['arrival_ns']),
            'Full original public cohort in chronological FIFO order required')
    observed = new_observed_state(aliases, execution_context_sha256, mode=mode, frozen_cutoff_ns=frozen_cutoff_ns)
    return seal(dict(schema=SCHEMA, full_public_cohort=cohort, aliases=copy.deepcopy(aliases),
        execution_context_sha256=execution_context_sha256, phase_start_ns=phase_start_ns,
        deadline_ns=deadline_ns, drain_deadline_ns=drain_deadline_ns, observer=observed,
        feedback_prefix=[], completed_registry_retained=True, simulated_checkpoint=None,
        observed_GPU_cursor=False, main_v54_continuation_supported=False))


def advance(session, now_ns, feedback=()):
    validate(session)
    require(type(now_ns) is int and session['phase_start_ns'] <= now_ns <= session['deadline_ns'],
            'Cutoff inside original measurement window required')
    state = session['observer']; seen = {rid for rid,r in state['requests'].items() if 'arrival_ns' in r}
    arrivals = []
    for rid,row in session['full_public_cohort'].items():
        if row['arrival_ns'] <= now_ns and rid not in seen:
            arrivals.append(dict(row, sequence=state['last_arrival_sequence']+len(arrivals)+1,
                                 observed_time_ns=now_ns))
    for event in feedback:
        payload = event['payload']; rid = payload['request_id']
        require(rid in session['full_public_cohort'] and payload['alias'] == session['full_public_cohort'][rid]['alias']
                and session['full_public_cohort'][rid]['arrival_ns'] <= event['received_time_ns'],
                'Visible feedback must match public request and causal arrival')
        require(payload['perf_ns'] >= session['full_public_cohort'][rid]['arrival_ns'],
                'Observed worker boundary precedes public arrival')
    observed = advance_observed_state(state, now_ns, feedback=feedback, arrivals=arrivals)
    result = copy.deepcopy(session); result['observer'] = observed
    result['feedback_prefix'].extend(copy.deepcopy(list(feedback)))
    return seal(result)


def advance_runtime_records(session, now_ns, records):
    """Reader visibility is tested before response bodies are selected."""
    validate(session)
    complete = feedback_from_runtime_records(records, session['execution_context_sha256'], now_ns)
    previous = session['feedback_prefix']
    require(complete[:len(previous)] == previous, 'Runtime registry changed an already consumed event prefix')
    return advance(session, now_ns, complete[len(previous):])


def evidence(reference, session, rid, kind, allow_cpu_fixture):
    require(set(reference) == {'path', 'sha256'} and receipt(reference['path']) == reference, 'Exact current progress/waiting receipt required')
    value = json.loads(Path(reference['path']).read_text())
    require(value.get('schema') == 'gsdeploy.visible_remaining_work_evidence.v1'
            and value.get('kind') == kind and value.get('request_id') == rid
            and value.get('execution_context_sha256') == session['execution_context_sha256']
            and value.get('observed_at_ns') == session['observer']['observed_at_ns']
            and type(value.get('received_time_ns')) is int
            and value['observed_at_ns'] <= value['received_time_ns'] <= session['observer']['observed_at_ns']
            and value.get('provenance') in ('runtime_direct_observation', 'cpu_fixture')
            and (value['provenance'] != 'cpu_fixture' or allow_cpu_fixture),
            'Evidence must be current, reader-visible, directly observed and execution-bound')
    require(receipt(reference['path']) == reference, 'Progress evidence changed during read')
    return value


def remaining_state(session, provider, *, cursor_evidence=None, waiting_evidence=None, allow_cpu_fixture=False):
    validate(session); observer = session['observer']; now = observer['observed_at_ns']
    require(type(now) is int, 'Advance a visible prefix before constructing remaining state')
    forecast_snapshot_from_observed(observer, session['aliases'], session['execution_context_sha256'], now,
                                    {r['task'] for r in session['full_public_cohort'].values()})
    cursors, waiting = dict(cursor_evidence or {}), dict(waiting_evidence or {})
    require(set(cursors) <= set(session['full_public_cohort']) and set(waiting) <= set(session['full_public_cohort']),
            'Residual evidence cannot introduce foreign requests')
    queues = {o:[] for o in session['aliases']}; completed, future, unknown, costs, statuses = [], [], [], {}, {}
    busy_starts, proof_refs = {}, []
    for rid, public in session['full_public_cohort'].items():
        observed = observer['requests'].get(rid, {})
        if public['arrival_ns'] > now:
            future.append(copy.deepcopy(public)); statuses[rid] = 'FUTURE_PUBLIC'
        elif observed.get('handler_end_ns') is not None:
            if observed.get('worker_next_ready_ns') is None:
                unknown.append('VISIBLE_HANDLER_END_WITHOUT_READY_RESIDUAL:'+rid); statuses[rid] = 'HANDOFF_UNKNOWN'
            else:
                completed.append(dict(public, completion_ns=observed['parent_receipt_ns'],
                    start_ns=observed['handler_start_ns'], handler_end_ns=observed['handler_end_ns']))
                statuses[rid] = 'COMPLETED_READY_VISIBLE'
            continue
        else:
            row = copy.deepcopy(public)
            if observed.get('handler_start_ns') is not None:
                statuses[rid] = 'BUSY_PROGRESS_UNKNOWN'
                if rid not in cursors:
                    unknown.append('DIRECT_OBSERVED_REMAINING_SEGMENT_CURSOR_REQUIRED:'+rid); continue
                proof = evidence(cursors[rid], session, rid, 'DIRECT_SEGMENT_PROGRESS', allow_cpu_fixture)
                require(set(proof) == {'schema','kind','request_id','execution_context_sha256','observed_at_ns',
                    'received_time_ns','provenance','profile_id','segment_index','remaining_fraction'},
                    'Cursor evidence accepts direct progress only, no future labels or elapsed estimate')
                row['forecast_cursor'] = {k:proof[k] for k in ('profile_id','segment_index','remaining_fraction','observed_at_ns')}
                row['forecast_cursor']['evidence'] = cursors[rid]['sha256']
                row['head_age_ms'] = (now-observed['handler_start_ns'])/1e6
                busy_starts[rid] = observed['handler_start_ns']; proof_refs.append(cursors[rid])
                statuses[rid] = 'BUSY_DIRECT_PROGRESS_VISIBLE'
            else:
                statuses[rid] = 'ARRIVED_START_VISIBILITY_UNKNOWN'
                if rid not in waiting:
                    unknown.append('VISIBLE_WAITING_NOT_STARTED_PROOF_REQUIRED:'+rid); continue
                proof = evidence(waiting[rid], session, rid, 'WAITING_NOT_STARTED', allow_cpu_fixture)
                require(set(proof) == {'schema','kind','request_id','execution_context_sha256','observed_at_ns',
                    'received_time_ns','provenance'}, 'Waiting evidence cannot include labels or predicted progress')
                proof_refs.append(waiting[rid]); statuses[rid] = 'WAITING_NOT_STARTED_VISIBLE'
            queues[public['alias']].append(row)
        try:
            plan = provider.plan(public, session['aliases'], 'mean')
            cursor = next((r.get('forecast_cursor') for r in queues[public['alias']] if r['request_id'] == rid), None)
            if cursor is not None: require(cursor['profile_id'] == plan['profile_id'], 'Direct cursor must bind exact frozen profile')
            costs[rid] = remaining_work(plan, cursor)
        except UnknownEvidence as exc:
            unknown.append('UNSUPPORTED_FROZEN_WORK:'+rid+':'+str(exc))
    for owner, rows in queues.items():
        active = [i for i,row in enumerate(rows) if 'forecast_cursor' in row]
        require(not active or active == [0], 'Only one directly observed active FIFO head per owner')
    return dict(schema='gsdeploy.visible_remaining_queue_work.v1', session_sha256=session['session_sha256'],
        observed_at_ns=now, queues=queues, completed_prefix=completed, future_public_arrivals=future,
        request_status=statuses, remaining_profile_work=costs, observed_busy_START_ns=busy_starts,
        proof_sources=proof_refs, unknown_reasons=sorted(unknown),
        future_work_envelopes=workload_envelopes(future, {rid:v['solo_work_ms'] for rid,v in costs.items()}),
        simulated_checkpoint=None, elapsed_fraction_used=False, observed_segment_progress=bool(busy_starts),
        observed_GPU_cursor=False,
        main_v54_continuation_supported=False)


def forecast_suffix(session, model, *, cursor_evidence=None, waiting_evidence=None,
                    score_kwargs=None, allow_cpu_fixture=False):
    """Execute remaining public work in the actual generic segment simulator."""
    validate(session)
    common = dict(schema='gsdeploy.rolling_visible_queue_forecast.v1', GPU_started=False,
        full_public_cohort=copy.deepcopy(session['full_public_cohort']),
        session_sha256=session['session_sha256'], simulated_checkpoint_used=False,
        main_v54_rolling_complete=False, physical_prediction_accuracy_validated=False)
    if type(model) is not FullHorizonModel:
        return dict(common, status='UNKNOWN', reason='MAIN_LIFECYCLE_CONTINUATION_NOT_IMPLEMENTED', forecast=None, metrics=None)
    require(model.aliases == session['aliases'] and not model.gated and not model.batch_configurations,
            'This backend supports the exact shared unbatched segment execution contract only')
    require(getattr(model.service_provider, 'execution_context_sha256', None) == session['execution_context_sha256'],
            'Remaining service profiles must bind the same execution context')
    state = remaining_state(session, model.service_provider, cursor_evidence=cursor_evidence,
                            waiting_evidence=waiting_evidence, allow_cpu_fixture=allow_cpu_fixture)
    if state['unknown_reasons']:
        return dict(common, status='UNKNOWN', reason=state['unknown_reasons'], remaining_state=state, forecast=None, metrics=None)
    original = session['full_public_cohort']; now = state['observed_at_ns']
    forecast = model.forecast(state['queues'], now, session['deadline_ns'],
        future_arrivals=state['future_public_arrivals'],
        route_future=lambda row, *_:original[row['request_id']]['alias'],
        drain_deadline_ns=session['drain_deadline_ns'], require_energy=True, require_memory=True)
    expected = set(original)-{r['request_id'] for r in state['completed_prefix']}
    require(set(forecast['cohort']) == expected, 'Continuation dropped or changed complete future cohort')
    if set(forecast['finishes']) != expected:
        return dict(common, status='UNKNOWN', reason='INCOMPLETE_REMAINING_COHORT', remaining_state=state, forecast=forecast, metrics=None)
    # Generic simulator reports continuation START=now. Preserve actual visible
    # START for queue/service accounting; this does not change predicted END.
    forecast['cohort'] = {rid:copy.deepcopy(original[rid]) for rid in forecast['cohort']}
    for rid,row in forecast['finishes'].items():
        for key,value in original[rid].items(): require(row[key] == value, 'Continuation changed original request route/arrival')
        for key in ('forecast_cursor', 'head_age_ms'): row.pop(key, None)
        if rid in state['observed_busy_START_ns']: row['start_ns'] = state['observed_busy_START_ns'][rid]
    metrics = None
    if score_kwargs is not None:
        forbidden = {'forecast','completed','aliases','phase_start_ns','now_ns','deadline_ns',
                     'observed_peak_memory_mib','observed_board_energy_joules'}
        require(not forbidden & set(score_kwargs) and score_kwargs.get('weights', {k:1/6 for k in 'QTLPME'}) == {k:1/6 for k in 'QTLPME'},
                'Original prefix/window/cohort and six equal weights cannot be replaced')
        metrics = score_forecast(forecast, state['completed_prefix'], session['aliases'],
            phase_start_ns=session['phase_start_ns'], now_ns=now, deadline_ns=session['deadline_ns'], **score_kwargs)
        metrics['terms']['E'] = None; metrics.update(full_U=None, cost=None, absolute_utility_complete=False,
            six_term_comparison_complete=False)
        metrics['unknown_reasons'].append('VISIBLE_PREFIX_ENERGY_NOT_PROVIDED')
    for ref in state['proof_sources']: require(receipt(ref['path']) == ref, 'Remaining-work evidence changed during continuation')
    all_finishes = {r['request_id']:r['completion_ns'] for r in state['completed_prefix']}
    all_finishes.update({rid:r['finish_ns'] for rid,r in forecast['finishes'].items()})
    starts = {r['request_id']:r['start_ns'] for r in state['completed_prefix']}
    starts.update({rid:r['start_ns'] for rid,r in forecast['finishes'].items()})
    queue_wait_ms = {rid:(starts[rid]-row['arrival_ns'])/1e6 for rid,row in original.items()}
    return dict(common, status='SEGMENT_BACKEND_ROLLING_FORECAST', remaining_state=state, forecast=forecast,
        metrics=metrics, all_original_request_ids_retained=set(all_finishes) == set(original),
        original_boundary_queue_wait_ms=queue_wait_ms, future_arrivals_repredicted=len(state['future_public_arrivals']),
        current_waiting_depth=sum(r['arrival_ns'] <= now and (r['request_id'] not in starts or starts[r['request_id']] > now)
                                  for r in original.values()), observed_START_restored_for_latency_accounting=True)
