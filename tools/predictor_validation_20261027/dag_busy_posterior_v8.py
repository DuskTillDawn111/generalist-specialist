"""Training-supported joint DAG trajectory conditioning at a visible cutoff.

Each candidate is an actual complete DAG rollout under one joint training
scenario. Absence of RESULT conditions its reader receipt, not physical END.
Visible PHASE is an observation emission, never a CUDA instruction pointer.
No measured target suffix, endpoint multiplier, or elapsed work fraction enters
this interface. Empirical conditional masses are not calibrated probabilities.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from collections import Counter

from gsdeploy.forecasting.metrics import score_forecast
from gsdeploy.resources import identity
from .cuda_dag_simulator_v9 import DagFullHorizonModel
from .dag_checkpoint_v9 import contract, decode, receipt
from .dag_visible_observer_v8 import SCHEMA as VISIBLE_SCHEMA, _hash

SCHEMA = 'gsdeploy.joint_dag_reader_scenario_support.v8'
RESULT_SCHEMA = 'gsdeploy.visible_busy_dag_empirical_conditioning.v8'

COLLECTION_PLAN = dict(
    status='ADDITIONAL_TRAINING_REQUIRED_NOT_VALIDATION_FIT',
    workload='Prospectively declared full trace/routes/window with all four owners resident and verified initial READY inventory.',
    repeats='At least two independent complete training runs per supported joint scenario; distinct run IDs and immutable raw receipts.',
    boundaries='Actual submit, worker START, every PHASE start/end, handler END, RESULT reader enqueue, worker READY and READY reader enqueue in verified host clock.',
    coupling='Preserve joint within-request and across-owner/run vectors; include END-before-RESULT and RESULT-before-READY-reader examples.',
    dag='Source-bound ordered CUDA DAG and independently trained node duration/host supply/lifecycle models; relate each PHASE emission to the same public family/task/input.',
    fitting='Cross-fit training runs for emission timing support radii and scenario construction; no heldout labels, future response body, token count or quality input.',
    validation='Freeze model first; evaluate unseen runs at predeclared busy cutoffs, joint residual coverage and full-cohort P95/Delta-U sign/ranking.',
    resources='Capture original prefix board-energy integral and owning-process peak memory; missing axes remain UNKNOWN.')


def key(alias, task, pool_id):
    return json.dumps([alias, task, pool_id], separators=(',', ':'))


def occurrence_indices(cohort):
    seen=Counter();result={}
    rows=sorted(enumerate(cohort.values()),key=lambda item:(item[1]['arrival_ns'],item[0]))
    for _,row in rows:
        k=key(row['alias'],row['task'],row['pool_id'])
        result[row['request_id']]=seen[k];seen[k]+=1
    return result


def input_draw(mapping,row,indices):
    value=mapping[key(row['alias'],row['task'],row['pool_id'])]
    if isinstance(value,list):
        return value[indices[row['request_id']]]
    return value


def _ns(value):
    if type(value) is not int or value < 0:
        raise ValueError('Nonnegative integer scenario timing required')
    return value


class VisibleStartSupply:
    """Training public-input supply, restricted by explicit observed host facts.

    A visible START is a fact about host execution, never an estimate of CUDA
    progress. Actual submit is a lower bound. Future requests use the declared
    training supply; their arrivals and FIFO ordering are preserved.
    """
    def __init__(self,snapshot,scenario):
        self.cohort=copy.deepcopy(snapshot['full_public_cohort'])
        self.visible_snapshot_sha256=snapshot['snapshot_sha256']
        self.training_run_ids=copy.deepcopy(scenario['training_run_ids'])
        self.scenario_id=scenario['id']
        self.lags={};self.visible_lower_bounds={}
        delays=scenario['request_start_delay_ns_by_input']
        indices=occurrence_indices(self.cohort)
        for rid,row in self.cohort.items():
            delay=_ns(input_draw(delays,row,indices))
            observed=snapshot['observed_host_state']['requests'].get(rid,{})
            self.lags[rid]=delay
            self.visible_lower_bounds[rid]=max(snapshot['visible_submissions'].get(rid,0),observed.get('handler_start_ns') or 0)

    def __call__(self,row):
        return self.from_owner_ready(row,owner_ready_ns=row['arrival_ns'])

    def from_owner_ready(self,row,*,owner_ready_ns):
        if self.cohort.get(row['request_id'])!=row:
            raise ValueError('Request-start supply changed public arrival/route/input')
        rid=row['request_id']
        return max(max(row['arrival_ns'],_ns(owner_ready_ns))+self.lags[rid],self.visible_lower_bounds[rid])


def _sources(document, *, allow_cpu_fixture):
    if document.get('schema') != SCHEMA or document.get('split') != 'training':
        raise ValueError('Explicit joint training-only DAG/reader support required')
    if document.get('not_for_training', False) or not document.get('sources'):
        raise ValueError('Independent immutable training source receipts required')
    sources = {}
    for ref in document['sources']:
        if receipt(ref['path']) != ref:
            raise ValueError('Changed joint scenario training source')
        raw = json.loads(Path(ref['path']).read_text())
        if (raw.get('split') != 'training' or raw.get('not_for_training', False)
                or raw.get('kind') not in ('measured','cpu_fixture')
                or raw.get('kind') == 'cpu_fixture' and not allow_cpu_fixture
                or raw.get('execution_context_sha256') != document['execution_context_sha256']):
            raise ValueError('Validation or foreign source cannot support busy conditioning')
        run_id = raw.get('run_id')
        if not isinstance(run_id, str) or not run_id or run_id in sources:
            raise ValueError('Distinct source-bound independent training run identities required')
        if any(r.get('split') != 'training' for r in raw.get('request_records', ())):
            raise ValueError('Validation requests cannot support busy conditioning')
        sources[run_id] = ref
    return sources


def _profile(profile):
    if set(profile) != {'start_reader_delay_ns', 'ready_reader_delay_ns', 'phases'}:
        raise ValueError('Explicit joint reader delays and phase emissions required')
    _ns(profile['start_reader_delay_ns']); _ns(profile['ready_reader_delay_ns'])
    seen = set()
    for p in profile['phases']:
        if (set(p) != {'segment_id', 'kind', 'event', 'perf_offset_ns', 'reader_delay_ns'}
                or not isinstance(p['segment_id'], str) or not p['segment_id']
                or not isinstance(p['kind'], str) or not p['kind']
                or p['event'] not in ('start', 'end') or (p['segment_id'], p['event']) in seen):
            raise ValueError('Distinct source-supported PHASE emissions required')
        seen.add((p['segment_id'], p['event']))
        _ns(p['perf_offset_ns']); _ns(p['reader_delay_ns'])
    starts = {p['segment_id']:p for p in profile['phases'] if p['event'] == 'start'}
    ends = {p['segment_id']:p for p in profile['phases'] if p['event'] == 'end'}
    if set(starts) != set(ends) or any(starts[s]['kind'] != ends[s]['kind'] or
            starts[s]['perf_offset_ns'] > ends[s]['perf_offset_ns'] for s in starts):
        raise ValueError('Joint PHASE start/end order required')
    if any(a['perf_offset_ns']>b['perf_offset_ns'] or
            a['perf_offset_ns']+a['reader_delay_ns']>b['perf_offset_ns']+b['reader_delay_ns']
            for a,b in zip(profile['phases'],profile['phases'][1:])):
        raise ValueError('Joint PHASE emission worker/reader stream order required')


def _event(kind, rid, alias, worker, reader, phase=None):
    row = dict(kind=kind, request_id=rid, alias=alias, perf_ns=worker, received_time_ns=reader)
    if phase:
        row.update({k:phase[k] for k in ('segment_id', 'kind', 'event')})
        row['kind'] = 'PHASE'; row['phase_kind'] = phase['kind']
    return row


def _emissions(forecast, scenario, *, partial=False):
    """Project a *single joint draw*; never draw END and READY independently."""
    events = []
    indices=occurrence_indices(forecast['cohort'])
    for rid, row in forecast['cohort'].items():
        if key(row['alias'],row['task'],row['pool_id']) not in scenario['emissions']:
            raise ValueError('Complete public-input reader-emission support required')
        p=input_draw(scenario['emissions'],row,indices)
        _profile(p)
        start = forecast['starts'].get(rid)
        if start is None:
            continue
        start_read = start+p['start_reader_delay_ns']
        events.append(_event('START', rid, row['alias'], start, start_read))
        phase_events = [_event('PHASE', rid, row['alias'], start+x['perf_offset_ns'],
            start+x['perf_offset_ns']+x['reader_delay_ns'], x) for x in p['phases']]
        # Reader transport is ordered, while worker READY can precede RESULT
        # reader processing. This does not infer a READY from a later START.
        if any(x['received_time_ns'] < start_read for x in phase_events):
            raise ValueError('Joint reader PHASE order precedes START')
        events.extend(phase_events)
        done = forecast['finishes'].get(rid)
        if done is None:
            if partial:
                continue
            raise ValueError('Complete training scenario drain required for joint residual state')
        end, result, ready = (done[k] for k in ('handler_end_ns','finish_ns','worker_next_ready_ns'))
        ready_read = ready+p['ready_reader_delay_ns']
        if (start_read > result or result > ready_read or ready < end or
                any(x['perf_ns'] > end or x['received_time_ns'] > result for x in phase_events)):
            raise ValueError('DAG/reader joint scenario violates observed boundary chronology')
        events.extend([_event('HANDLER_END', rid, row['alias'], end, result),
                       _event('WORKER_READY', rid, row['alias'], ready, ready_read)])
    return sorted(events, key=lambda x:(x['received_time_ns'], x['request_id'], x['kind']))


def _event_key(event):
    return (event['request_id'], event['kind'], event.get('segment_id'), event.get('event'))


def _observed(snapshot):
    result = {}
    for event in snapshot['visible_feedback']:
        p = event['payload']
        row = dict(kind=event['kind'], request_id=p['request_id'], alias=p['alias'],
                   perf_ns=p['perf_ns'], received_time_ns=event['received_time_ns'])
        if event['kind'] == 'PHASE':
            row.update(segment_id=p['segment_id'], event=p['event'], phase_kind=p['kind'])
        k = _event_key(row)
        if k in result:
            raise ValueError('Duplicate visible event evidence')
        result[k] = row
    return result


def _consistent(snapshot, events, radius):
    observed = _observed(snapshot)
    predicted = {_event_key(e):e for e in events if e['received_time_ns'] <= snapshot['cutoff_ns']}
    # Absence of RESULT is a survival condition on *receipt*, not on END.
    if set(observed) != set(predicted):
        return False, 'VISIBLE_EVENT_PATTERN_OR_READER_SURVIVAL_MISMATCH'
    for k, a in observed.items():
        b = predicted[k]
        if a['kind'] in ('START','HANDLER_END','WORKER_READY') and a['perf_ns']!=b['perf_ns']:
            return False,'EXACT_VISIBLE_HOST_LIFECYCLE_BOUNDARY_MISMATCH'
        if (a['alias'] != b['alias'] or a.get('phase_kind') != b.get('phase_kind') or
                any(abs(a[f]-b[f]) > radius for f in ('perf_ns','received_time_ns'))):
            return False, 'VISIBLE_BOUNDARY_OUTSIDE_TRAINING_SUPPORT_RADIUS'
    return True, None


def _bracket(factory, scenario_id, snapshot, final, expected_contract, start_supply):
    """Locate simulated atomic states around cutoff without fabricating an edge.

    A non-atomic cutoff is an interval bracket, never a restorable observation
    checkpoint. The complete predicted suffix comes from the uninterrupted
    actual DAG rollout, which consumes service under its declared scheduler.
    """
    rows = list(snapshot['full_public_cohort'].values())
    start, deadline, drain = (snapshot[k] for k in ('phase_start_ns','deadline_ns','drain_deadline_ns'))
    cutoff = snapshot['cutoff_ns']; cache = {}
    def run(count):
        if count not in cache:
            engine = factory(scenario_id=scenario_id, now_ns=start,request_start_provider=start_supply)
            result = engine.forecast(rows, start, deadline, drain_deadline_ns=drain,
                                    pause_after_event_count=count)
            cache[count] = result.get('checkpoint')
            if cache[count] is not None and cache[count]['contract'] != expected_contract:
                raise ValueError('Scenario factory changed DAG providers between latent state proposals')
        return cache[count]
    low, high = 0, final['events']
    while high-low > 1:
        mid = (low+high)//2; cp = run(mid)
        if cp is None or decode(cp['state'])['clock'] > cutoff:
            high = mid
        else:
            low = mid
    left = run(low) if low else None
    right = run(high)
    return dict(kind='LATENT_SIMULATED_DAG_ATOMIC_BRACKET_NOT_OBSERVED_CURSOR',
        cutoff_ns=cutoff, left_checkpoint=left, right_checkpoint=right,
        left_clock_ns=decode(left['state'])['clock'] if left else start,
        right_clock_ns=decode(right['state'])['clock'] if right else None,
        exact_atomic_cutoff=left is not None and decode(left['state'])['clock'] == cutoff,
        observed_GPU_cursor=False, elapsed_fraction_cursor_used=False)


def _scoring_forecast(snapshot, forecast):
    """Replace completed prefix and integrate only predicted suffix resources."""
    value = copy.deepcopy(forecast); cutoff = snapshot['cutoff_ns']; deadline = snapshot['deadline_ns']
    completed = {r['request_id'] for r in snapshot['completed_prefix']}
    for field in ('cohort','finishes','starts','queue_wait_ns'):
        value[field] = {k:v for k,v in value[field].items() if k not in completed}
    intervals = []
    for row in value.get('resource_transitions', []):
        begin, end = max(cutoff, row['time_ns']), row['until_ns']
        if begin < end:
            intervals.append(dict(row, time_ns=begin))
    value['resource_transitions'] = intervals
    energy = [r for r in intervals if r['time_ns'] < deadline]
    memory = [r['peak_memory_mib'] for r in intervals]
    value['board_energy_joules'] = (sum(r['board_watts']*(min(deadline,r['until_ns'])-r['time_ns'])/1e9
        for r in energy) if energy and all(r['board_watts'] is not None for r in energy) else None)
    value['peak_memory_mib'] = max(memory) if memory and all(x is not None for x in memory) else None
    value['energy_window_ns'] = [cutoff, deadline]
    value['memory_window_ns'] = [cutoff, forecast['memory_window_ns'][1]]
    return value


def condition_busy_scenarios(snapshot, support, engine_factory, *, score_kwargs=None,
                             capture_latent_checkpoints=True, allow_cpu_fixture=False,
                             prefix_screen=True, prefix_screen_only=False):
    """Execute source-supported full DAG scenarios; preserve conditional draws.

    ``engine_factory(scenario_id=..., now_ns=original_phase_start,
    request_start_provider=...)`` binds each
    joint scenario's trained node/host/lifecycle providers. No caller-supplied
    target finish/cursor is accepted. At least two surviving independent runs
    are required. A support declaration alone is not scientific calibration.
    """
    if snapshot.get('schema') != VISIBLE_SCHEMA or snapshot.get('snapshot_sha256') != _hash(snapshot):
        raise ValueError('Mutated or foreign visible cutoff snapshot')
    if prefix_screen_only and not prefix_screen:
        raise ValueError('Prefix-only diagnostics require actual DAG prefix screening')
    common = dict(schema=RESULT_SCHEMA, full_public_cohort=copy.deepcopy(snapshot['full_public_cohort']),
        visible_snapshot_sha256=snapshot['snapshot_sha256'], original_window_ns=[snapshot['phase_start_ns'],snapshot['deadline_ns']],
        observed_GPU_cursor=False, elapsed_fraction_cursor_used=False, whole_request_multiplier_used=False,
        posterior_calibrated=False, risk_bound_calibrated=False, rolling_risk_correction_complete=False,
        physical_accuracy_validated=False, deployment_qualified=False, GPU_started=False,
        additional_collection_plan=copy.deepcopy(COLLECTION_PLAN))
    reasons = []
    if support is None:
        reasons.append('JOINT_DAG_AND_READER_TRAINING_SUPPORT_UNAVAILABLE')
    if 'visible_feedback' not in snapshot:
        reasons.append('COMPLETE_VISIBLE_READER_EVENT_FEED_REQUIRED')
    if any(not r['known'] for r in snapshot['initial_owner_idle'].values()):
        reasons.append('INITIAL_OWNER_IDLE_NOT_VERIFIED')
    if any(s == 'ARRIVED_DISPATCH_UNOBSERVED' for s in snapshot['request_status'].values()):
        reasons.append('UNSENT_ARRIVAL_DISPATCH_MODEL_REQUIRED')
    if reasons:
        return dict(common, status='UNKNOWN', unknown_reasons=reasons, candidates=[])
    sources = _sources(support, allow_cpu_fixture=allow_cpu_fixture)
    measured=any(json.loads(Path(ref['path']).read_text()).get('kind')=='measured' for ref in support['sources'])
    fitted=measured or support.get('fit_contract') is not None
    if fitted:
        from .dag_busy_support_v9 import verify_support
        verify_support(support,allow_cpu_fixture=allow_cpu_fixture)
        if support['fit_status']!='FITTED_EMPIRICAL_HOST_READER_JOINT_SCENARIOS':
            return dict(common,status='UNKNOWN',unknown_reasons=['INDEPENDENT_RAW_DERIVED_JOINT_TRAINING_SUPPORT_INSUFFICIENT'],
                candidates=[],conditional_mass_available=False)
        shape=[dict(offset_ns=row['arrival_ns']-snapshot['phase_start_ns'],alias=row['alias'],task=row['task'],pool_id=row['pool_id'])
               for _,row in sorted(enumerate(snapshot['full_public_cohort'].values()),key=lambda item:(item[1]['arrival_ns'],item[0]))]
        if identity(shape) not in support['supported_trace_signatures']:
            return dict(common,status='UNKNOWN',unknown_reasons=['UNTRAINED_PUBLIC_TRACE_OR_ROUTE_OCCURRENCE_TRANSFER'],
                candidates=[],conditional_mass_available=False)
    if support['execution_context_sha256'] != snapshot['execution_context_sha256']:
        raise ValueError('Busy support belongs to another execution context')
    if (support.get('generation_work_scope') != 'training_only_public_input_conditioned'
            or support.get('joint_scope') != 'whole_run_node_host_lifecycle_reader_emission'
            or support.get('timing_radius_basis') != 'cross_fitted_training_run_support'):
        raise ValueError('Explicit joint training model and cross-fitted observation timing support required')
    radius = _ns(support['observation_tolerance_ns'])
    from .dag_visible_anchor_v10 import (ConditionalVisibleStartSupply,
        UnsupportedStartObservation, start_lag_radii)
    lag_radii=start_lag_radii(support) if fitted else None
    minimum = support.get('minimum_independent_survivor_runs', 2)
    if type(minimum) is not int or minimum < 2:
        raise ValueError('At least two independent surviving training runs required')
    if len(sources) < minimum:
        return dict(common,status='UNKNOWN',unknown_reasons=['INSUFFICIENT_INDEPENDENT_JOINT_TRAINING_RUNS'],
            candidates=[],surviving_independent_runs=None,conditional_mass_available=False)
    scenario_ids = set(); candidates = []; rejected = []; all_units = set();prefix_screened=[]
    indices=occurrence_indices(snapshot['full_public_cohort'])
    for scenario in support['scenarios']:
        sid = scenario['id']; units = set(scenario['training_run_ids'])
        if (not isinstance(sid,str) or not sid or sid in scenario_ids or not units or not units <= sources.keys()
                or all_units & units or type(scenario['prior_mass']) not in (int,float)
                or not math.isfinite(scenario['prior_mass']) or scenario['prior_mass'] <= 0):
            raise ValueError('Distinct joint scenarios with disjoint independent training runs and finite prior masses required')
        scenario_ids.add(sid); all_units.update(units)
        public_keys={key(row['alias'],row['task'],row['pool_id']) for row in snapshot['full_public_cohort'].values()}
        if not public_keys<=scenario.get('request_start_delay_ns_by_input',{}).keys():
            rejected.append(dict(id=sid,reason='TRAINED_FUTURE_PUBLIC_INPUT_REQUEST_START_SUPPLY_REQUIRED'));continue
        if not public_keys<=scenario.get('emissions',{}).keys():
            rejected.append(dict(id=sid,reason='UNKNOWN_PUBLIC_INPUT_READER_EMISSION_SUPPORT'));continue
        if any(isinstance(mapping.get(key(row['alias'],row['task'],row['pool_id'])),list) and
                len(mapping[key(row['alias'],row['task'],row['pool_id'])])<=indices[row['request_id']]
                for mapping in (scenario['emissions'],scenario['request_start_delay_ns_by_input'])
                for row in snapshot['full_public_cohort'].values()):
            rejected.append(dict(id=sid,reason='UNKNOWN_PUBLIC_INPUT_OCCURRENCE_SUPPORT'));continue
        try:
            start_supply=(ConditionalVisibleStartSupply(snapshot,scenario,lag_radii) if fitted
                          else VisibleStartSupply(snapshot,scenario))
        except UnsupportedStartObservation as exc:
            rejected.append(dict(id=sid,reason=str(exc),stage='EXACT_START_CONDITIONING'));continue
        engine = engine_factory(scenario_id=sid, now_ns=snapshot['phase_start_ns'],request_start_provider=start_supply)
        if (type(engine) is not DagFullHorizonModel or engine.aliases != snapshot['aliases']
                or engine.request_start_provider is not start_supply):
            raise ValueError('Actual v9 DAG engine, bound request-start supply and unchanged resident owners required')
        if fitted:
            from .dag_busy_support_v9 import JointLifecycleFromSupport
            life=engine.lifecycle_provider
            if (type(life) is not JointLifecycleFromSupport or life.support_sha256!=support['model_sha256']
                    or life.scenario!=scenario or life.cohort!=snapshot['full_public_cohort']):
                raise ValueError('DAG lifecycle must use the same raw-rederived whole-run joint scenario')
        binding = engine.checkpoint_binding
        if (not binding or binding.get('execution_context_sha256') != snapshot['execution_context_sha256']
                or not all(ref in binding.get('evidence',[]) for ref in sources.values())):
            raise ValueError('DAG scenario providers must bind declared training source evidence')
        engine_contract=contract(engine,list(snapshot['full_public_cohort'].values()),snapshot['phase_start_ns'],
            snapshot['deadline_ns'],snapshot['drain_deadline_ns'])
        rows=list(snapshot['full_public_cohort'].values())
        checkpoint=None
        try:
            if prefix_screen:
                prefix=engine.forecast(rows,snapshot['phase_start_ns'],snapshot['deadline_ns'],
                    drain_deadline_ns=snapshot['drain_deadline_ns'],pause_at_or_after_ns=snapshot['cutoff_ns']+1)
                checkpoint=prefix.get('checkpoint')
                if checkpoint is not None:
                    state=decode(checkpoint['state'])
                    projected=dict(cohort=state['cohort'],starts=state['starts'],
                        finishes={**state['finishes'],**state['receipt_pending']})
                else:
                    projected=prefix
                try:
                    prefix_events=_emissions(projected,scenario,partial=True)
                except ValueError as exc:
                    rejected.append(dict(id=sid,reason='DAG_READER_JOINT_CHRONOLOGY_INCONSISTENT',
                        diagnostic=str(exc),stage='ACTUAL_DAG_PREFIX'));continue
                consistent,reason=_consistent(snapshot,prefix_events,radius)
                if not consistent:
                    rejected.append(dict(id=sid,reason=reason));continue
                if prefix_screen_only:
                    prefix_screened.append(dict(id=sid,status='ACTUAL_DAG_PREFIX_COMPATIBLE_NOT_FULL_FORECAST',
                        atomic_clock_ns=state['clock'] if checkpoint is not None else None,
                        complete_original_cohort=len(snapshot['full_public_cohort']),
                        simulated_prefix_events=state['event_count'] if checkpoint is not None else prefix['events'],
                        observed_GPU_cursor=False))
                    continue
            forecast=(prefix if prefix_screen and checkpoint is None else engine.forecast(
                rows,snapshot['phase_start_ns'],snapshot['deadline_ns'],
                drain_deadline_ns=snapshot['drain_deadline_ns'],checkpoint=checkpoint))
        except UnsupportedStartObservation as exc:
            rejected.append(dict(id=sid,reason=str(exc),stage='EXACT_START_CONDITIONING'));continue
        if forecast['cohort'] != snapshot['full_public_cohort']:
            raise ValueError('Scenario engine changed or dropped original full future schedule')
        if set(forecast['finishes']) != set(forecast['cohort']):
            rejected.append(dict(id=sid,reason='DRAIN_CENSORED_JOINT_RESIDUAL_SUPPORT'));continue
        if any(forecast['starts'].get(rid, math.inf) < at for rid,at in snapshot['visible_submissions'].items()):
            rejected.append(dict(id=sid,reason='DAG_START_PRECEDES_OBSERVED_SUBMISSION')); continue
        try:
            events = _emissions(forecast, scenario)
        except ValueError as exc:
            rejected.append(dict(id=sid,reason='DAG_READER_JOINT_CHRONOLOGY_INCONSISTENT',
                diagnostic=str(exc),stage='FULL_DAG_ROLLOUT'));continue
        consistent, reason = _consistent(snapshot, events, radius)
        if not consistent:
            rejected.append(dict(id=sid,reason=reason)); continue
        latent = {}
        for rid, done in forecast['finishes'].items():
            ready_receipt = next(e['received_time_ns'] for e in events if e['request_id']==rid and e['kind']=='WORKER_READY')
            cutoff = snapshot['cutoff_ns']
            latent[rid] = dict(handler_end_ns=done['handler_end_ns'], result_reader_ns=done['finish_ns'],
                worker_ready_ns=done['worker_next_ready_ns'], ready_reader_ns=ready_receipt,
                handler_active_at_cutoff=done['start_ns'] <= cutoff < done['handler_end_ns'],
                end_happened_result_in_transit=done['handler_end_ns'] <= cutoff < done['finish_ns'],
                ready_happened_reader_in_transit=done['worker_next_ready_ns'] <= cutoff < ready_receipt,
                handler_end_remaining_ns=max(0,done['handler_end_ns']-cutoff),
                result_reader_remaining_ns=max(0,done['finish_ns']-cutoff),
                ready_reader_remaining_ns=max(0,ready_receipt-cutoff))
        candidate = dict(id=sid, prior_mass=scenario['prior_mass'], training_run_ids=sorted(units),
            latent_request_states=latent, full_dag_forecast=forecast,
            conditional_rule='Joint visible reader pattern, timing support and absent-reader-event survival; no END survival from absent RESULT',
            dag_state=_bracket(engine_factory,sid,snapshot,forecast,engine_contract,start_supply) if capture_latent_checkpoints else None,
            scoring_forecast=_scoring_forecast(snapshot,forecast), metrics=None)
        if score_kwargs is not None:
            forbidden = {'phase_start_ns','now_ns','deadline_ns','observed_peak_memory_mib','observed_board_energy_joules','aliases','completed','forecast'}
            if forbidden & set(score_kwargs) or 'weights' in score_kwargs and score_kwargs['weights'] != {k:1/6 for k in 'QTLPME'}:
                raise ValueError('Original scoring window, prefix and six equal weights required')
            resources=snapshot['prefix_resources']
            metric=score_forecast(candidate['scoring_forecast'], snapshot['completed_prefix'], snapshot['aliases'],
                phase_start_ns=snapshot['phase_start_ns'],now_ns=snapshot['cutoff_ns'],deadline_ns=snapshot['deadline_ns'],
                observed_peak_memory_mib=resources['peak_memory_mib'],
                observed_board_energy_joules=resources['board_energy_joules'], **score_kwargs)
            if not resources['energy_known']:
                metric['terms']['E']=None;metric['unknown_reasons'].append('UNKNOWN_VISIBLE_PREFIX_BOARD_ENERGY')
                metric.update(cost=None,full_U=None,six_term_comparison_complete=False,absolute_utility_complete=False)
            candidate['metrics']=metric
        candidates.append(candidate)
    if any(receipt(ref['path']) != ref for ref in support['sources']):
        raise ValueError('Training scenario sources changed during conditioning')
    if prefix_screen_only:
        return dict(common,status='ACTUAL_DAG_PREFIX_SCREEN_ONLY_NOT_FORECAST',unknown_reasons=[],
            candidates=[],rejected=rejected,prefix_screened=prefix_screened,conditional_mass_available=False,
            full_horizon_forecast_complete=False,full_U=None)
    survivors={run for c in candidates for run in c['training_run_ids']}
    if len(survivors) < minimum:
        return dict(common,status='UNKNOWN',unknown_reasons=['INSUFFICIENT_SURVIVING_INDEPENDENT_JOINT_TRAINING_RUNS'],
            candidates=candidates,rejected=rejected,surviving_independent_runs=len(survivors),conditional_mass_available=False)
    mass=sum(c['prior_mass'] for c in candidates)
    for candidate in candidates:
        candidate['conditional_empirical_mass']=candidate['prior_mass']/mass
    return dict(common,status='EMPIRICAL_CONDITIONAL_DAG_SCENARIOS_NOT_CALIBRATED',unknown_reasons=[],
        candidates=candidates,rejected=rejected,surviving_independent_runs=len(survivors),conditional_mass_available=True,
        support_sha256=identity(support),selection='Original complete scenario draws retained; no averaging before task P95 or six-term scoring')
