"""Visible busy conditioning through the independent trained phase-DAG engine.

This shares reader/observation/scoring checks with v9, while requiring the v10
scheduler-state checkpoint contract and solo reference-work scheduling. It has
no equal-context fallback. Unsupported full original scenarios stay UNKNOWN.
"""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.metrics import score_forecast
from gsdeploy.resources import identity
from .dag_phase_engine_v10 import DagPhaseCheckpointModel
from .dag_phase_checkpoint_v10 import contract,decode,receipt
from .dag_frozen_model_v10 import plain
from .dag_visible_observer_v8 import SCHEMA as VISIBLE_SCHEMA,_hash
from .dag_busy_posterior_v8 import (
    COLLECTION_PLAN,SCHEMA,VisibleStartSupply,key,occurrence_indices,_ns,_sources,
    _emissions,_consistent,_bracket,_scoring_forecast)

RESULT_SCHEMA='gsdeploy.visible_busy_trained_phase_DAG_empirical_conditioning.v10'

def forecast_with_optional_risk(snapshot,support,engine_factory,*,risk_model_ref=None,
                               risk_cost_provider=None,**kwargs):
    """Execute the actual phase DAG path, then attach same-cutoff frozen risk.

    Persisted receipts bind the original plan and raw timing feed. The risk
    adapter projects reader-visible START/READY only, preserving main status,
    six-term U and stability.
    """
    result=condition_visible_phase_scenarios(snapshot,support,engine_factory,**kwargs)
    if risk_model_ref is None:
        if risk_cost_provider is not None:raise ValueError('Explicit frozen risk model receipt required')
        return result
    if risk_cost_provider is None or snapshot.get('raw_source') is None or not snapshot.get('plan_sources'):
        raise ValueError('Risk sidecar requires the same source-bound original raw and plan')
    from .native_queue_risk_inference_v9 import predict_cutoff,attach_diagnostic
    raw_ref=snapshot['raw_source'];plan_ref=snapshot['plan_sources'][0]
    refs=[raw_ref,plan_ref,risk_model_ref]
    if any(receipt(ref['path'])!=ref for ref in refs):raise ValueError('Changed source-bound risk cutoff inputs')
    raw=json.loads(Path(raw_ref['path']).read_text());plan=json.loads(Path(plan_ref['path']).read_text())
    model=json.loads(Path(risk_model_ref['path']).read_text())
    if (raw.get('plan')!=plan_ref or plan.get('execution_context_sha256')!=snapshot['execution_context_sha256']
            or raw.get('phase_start_ns')!=snapshot['phase_start_ns']):
        raise ValueError('Risk cutoff must use the exact observed DAG plan, context and clock')
    risk=predict_cutoff(plan,raw['request_records'],phase_start_ns=snapshot['phase_start_ns'],
        now_ns=snapshot['cutoff_ns'],model=model,cost_provider=risk_cost_provider)
    risk['evidence']=copy.deepcopy(refs)
    risk['visible_snapshot_sha256']=snapshot['snapshot_sha256']
    if any(receipt(ref['path'])!=ref for ref in refs):raise ValueError('Risk inputs changed during same-cutoff inference')
    return attach_diagnostic(result,risk)


def condition_visible_phase_scenarios(snapshot, support, engine_factory, *, score_kwargs=None,
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
    common = dict(schema=RESULT_SCHEMA, engine_kind='independent_v10_trained_availability_DAG',
        equal_context_fallback_used=False, reference_work_domain='SOLO_ONLY_REFERENCE_NS', full_public_cohort=copy.deepcopy(snapshot['full_public_cohort']),
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
        if (type(engine) is not DagPhaseCheckpointModel or plain(engine.aliases) != plain(snapshot['aliases'])
                or engine.request_start_provider is not start_supply):
            raise ValueError('Actual v10 trained-phase DAG engine, bound host START supply and unchanged resident owners required')
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
        engine_contract=(contract(engine,list(snapshot['full_public_cohort'].values()),snapshot['phase_start_ns'],
            snapshot['deadline_ns'],snapshot['drain_deadline_ns']) if capture_latent_checkpoints else None)
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
        except UnknownEvidence as exc:
            rejected.append(dict(id=sid,reason=str(exc),stage='TRAINED_PHASE_DAG_UNSUPPORTED'));continue
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
