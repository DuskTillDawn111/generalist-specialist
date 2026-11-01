"""Detached validation/atomic commit for v59 simulated lifecycle continuation.

This is not an observed GPU cursor or general native assimilation protocol.
Deterministic providers must declare no mutable RNG/execution state. The only
supported random budget is v24's saved keyed SEEDED_UNIFORM/MEDIAN_LOG2 state.
"""
from __future__ import annotations
import copy
import math
import inspect
from gsdeploy.forecast_errors import UnknownEvidence
from . import dag_checkpoint_v7 as structural
from . import lifecycle_scheduler_checkpoint_v1 as scheduler_cp
from .dag_checkpoint_v9 import request_start_bound
from .dag_frozen_model_v10 import plain
from collections.abc import Mapping
from collections import Counter,deque
from .dag_queue_diagnostics_v6 import DagQueueDiagnostics
from .implementation_source_closure_v27 import implementation_sources
from .owner_lifecycle_mark_v21 import engine_states, receipt
from .profiling import digest

SCHEMA='gsdeploy.simulated_lifecycle_engine_checkpoint.v59'
STAGE='EXACT_CUTOFF_WITH_UNFINISHED_CONSTANT_RATE_INTERVAL'
STATE_FIELDS=tuple(structural.STATE_FIELDS)+('owner_last_ready','node_completion_ns','advance')
ADVANCE_FIELDS=set(('origin_ns boundary_ns rates finish_ticks origin_active origin_energy '
    'origin_peak origin_energy_known origin_memory_known origin_resource_count '
    'diagnostics_origin resource').split())


def checkpoint_value(value):
    if isinstance(value,Mapping):
        cls=Counter if type(value) is Counter else dict
        return cls({checkpoint_value(k):checkpoint_value(v) for k,v in value.items()})
    if type(value) in (list,tuple,set,frozenset,deque):
        return type(value)(checkpoint_value(v) for v in value)
    return value


def diagnostic_state(diagnostic):
    if diagnostic is None:return None
    if type(diagnostic) is not DagQueueDiagnostics:
        raise ValueError('v59 supports exact DagQueueDiagnostics only; other sidecars cannot resume')
    return copy.deepcopy({k:v for k,v in vars(diagnostic).items() if k not in structural.DIAGNOSTIC_EXCLUDED})


def install_diagnostic_state(diagnostic,state):
    if diagnostic is not None:
        diagnostic.__dict__={**vars(diagnostic),**copy.deepcopy(state)}


def checkpoint_contract(engine,requests,now_ns,deadline_ns,drain_deadline_ns):
    if (type(requests) is not list or type(now_ns) is not int or now_ns<0
            or type(deadline_ns) is not int or deadline_ns<now_ns
            or drain_deadline_ns is not None and
                (type(drain_deadline_ns) is not int or drain_deadline_ns<deadline_ns)):
        raise ValueError('Full public request list and original integer windows required')
    expected=set(('_forecast_invoked _frozen_model _source_receipts aliases plans '
        'allow_context_time_slice_hypothesis allow_independent_copy_hypothesis '
        'allow_no_implicit_default_stream_hypothesis allow_profiled_replay checkpoint_binding '
        'duration_provider lifecycle_provider max_events memory_provider phase_provider '
        'power_provider queue_diagnostics release_provider request_start_provider require_energy '
        'require_memory scenario scheduler').split())
    if set(vars(engine))!=expected:
        raise ValueError('Untracked lifecycle engine instance fields cannot resume')
    binding=engine.checkpoint_binding
    if (type(binding) is not dict or set(binding)!={'execution_context_sha256','evidence','provider_state_policy'}
            or binding['provider_state_policy']!='DETERMINISTIC_STATELESS_NO_MUTABLE_RNG'
            or type(binding['execution_context_sha256']) is not str
            or len(binding['execution_context_sha256'])!=64
            or any(c not in '0123456789abcdef' for c in binding['execution_context_sha256'])
            or not binding['evidence']):
        raise ValueError('Explicit deterministic stateless provider/RNG contract and source evidence required')
    for ref in binding['evidence']:
        if type(ref) is not dict or receipt(ref['path'])!=ref:
            raise ValueError('Changed checkpoint provider evidence')
    scheduler_cp._components(engine.scheduler) # reject unknown classes/fields before any work
    diagnostic_state(engine.queue_diagnostics)
    identities={}
    for field,method in (('duration_provider',None),('release_provider',None),('phase_provider',None),
            ('request_start_provider',None),('lifecycle_provider','delays'),
            ('power_provider','power'),('memory_provider','memory')):
        value=getattr(engine,field)
        identities[field]=structural.callable_identity(getattr(value,method) if value is not None and method else value)
    diagnostic=engine.queue_diagnostics
    identities['queue_diagnostics']=(None if diagnostic is None else dict(
        initial_ns=diagnostic.initial_ns,
        duration=structural.callable_identity(diagnostic.duration_provider),
        demand=structural.callable_identity(diagnostic.resource_demand_provider)))
    options={k:getattr(engine,k) for k in ('allow_profiled_replay','allow_independent_copy_hypothesis',
        'allow_context_time_slice_hypothesis','allow_no_implicit_default_stream_hypothesis',
        'scenario','max_events','require_energy','require_memory')}
    return dict(binding=copy.deepcopy(binding),providers=identities,options=options,
        aliases_sha256=digest(scheduler_cp.encode(plain(engine.aliases))),
        graph_sha256=digest(scheduler_cp.encode(plain(engine.plans))),
        cohort_sha256=structural.digest(structural.encode(requests)),
        original_now_ns=now_ns,deadline_ns=deadline_ns,drain_deadline_ns=drain_deadline_ns,
        implementation_sources=implementation_sources(__file__,inspect.getfile(type(engine))))


def scheduler_binding(contract,clock):
    return dict(execution_context_sha256=contract['binding']['execution_context_sha256'],
        cohort_sha256=contract['cohort_sha256'],graph_sha256=contract['graph_sha256'],
        original_now_ns=contract['original_now_ns'],deadline_ns=contract['deadline_ns'],
        drain_deadline_ns=contract['drain_deadline_ns'],clock_ns=clock,atomic_stage=scheduler_cp.STAGE)


def prefix_accounting(state,contract):
    """Lossless scorer inputs; P95 and weighted Q are intentionally not pre-averaged."""
    by_task={}
    for rid,row in state['cohort'].items():
        item=by_task.setdefault(row['task'],dict(offered_ids=[],received_latency_ns=[],completed_by_deadline=[]))
        item['offered_ids'].append(rid)
        if rid in state['finishes']:
            end=state['finishes'][rid]['finish_ns']
            item['received_latency_ns'].append((rid,end-row['arrival_ns']))
            if end<=contract['deadline_ns']:item['completed_by_deadline'].append(rid)
    return dict(kind='LOSSLESS_ORIGINAL_SIX_TERM_INPUTS_NOT_PARTIAL_UTILITY',
        cohort=copy.deepcopy(state['cohort']),tasks=by_task,
        received=copy.deepcopy(state['finishes']),
        measurement_energy_joules=state['energy'] if state['energy_known'] else None,
        memory_peak_through_cutoff_mib=state['peak'] if state['memory_known'] else None,
        original_measurement_window_ns=[contract['original_now_ns'],contract['deadline_ns']],
        cutoff_ns=state['clock'],safe_factor_applied=False,startup_amortization_applied=False,
        quality_training_units_not_reestimated=True,weights_and_thresholds_changed=False)


def integrate_resources(engine,advance,transitions,running,target,deadline):
    origin=advance['origin_ns'];delta=target-origin
    energy=advance['origin_energy'];peak=advance['origin_peak']
    eknown=advance['origin_energy_known'];mknown=advance['origin_memory_known']
    if delta<=0:return energy,peak,eknown,mknown
    if advance['resource'] is None:
        handlers={owner:state['row'] for owner,state in running.items()}
        def read(provider,method,field):
            value=getattr(provider,method)(handlers,engine.aliases,engine.scenario) if provider is not None else {}
            number=value.get(field)
            return number if value.get('known') is True and type(number) in (int,float) and math.isfinite(number) and number>=0 else None
        watts=read(engine.power_provider,'power','watts') if origin<deadline else None
        memory=read(engine.memory_provider,'memory','peak_memory_mib')
        advance['resource']=dict(time_ns=origin,handler_active_aliases=sorted(handlers),board_watts=watts,peak_memory_mib=memory)
    resource=advance['resource'];watts=resource['board_watts'];memory=resource['peak_memory_mib']
    if origin<deadline:
        if watts is None:
            eknown=False
            if engine.require_energy:raise UnknownEvidence('UNKNOWN_DAG_WHOLE_BOARD_POWER_CONTEXT')
        else:energy+=watts*min(delta,deadline-origin)/1e9
    if memory is None:
        mknown=False
        if engine.require_memory:raise UnknownEvidence('UNKNOWN_DAG_WHOLE_BOARD_PEAK_MEMORY_CONTEXT')
    else:peak=max(peak,memory)
    transitions[advance['origin_resource_count']:]=[dict(resource,until_ns=target)]
    return energy,peak,eknown,mknown


def _validate_diagnostic(saved,state,engine):
    diagnostic=engine.queue_diagnostics
    if diagnostic is None:
        if saved is not None:raise ValueError('Missing connected queue diagnostics')
        return
    expected=set(vars(diagnostic))-structural.DIAGNOSTIC_EXCLUDED
    if (type(saved) is not dict or set(saved)!=expected or saved['now_ns']!=state['clock']
            or saved['waiting']!={row['request_id'] for rows in state['pending'].values() for row in rows}):
        raise ValueError('Queue diagnostic fields/clock/waiting state disagree')
    structural.validate_diagnostic(saved,state,diagnostic)


def _validate_lifecycle(state,engine,contract):
    now=contract['original_now_ns'];clock=state['clock']
    ended=list(state['finishes'].values())+list(state['receipt_pending'].values())
    expected={owner:now for owner in engine.aliases}
    for row in ended:
        delays=(engine.lifecycle_provider.delays(row,engine.aliases,engine.scenario)
                if engine.lifecycle_provider else dict(parent_receipt_ms=0.,worker_next_ready_ms=0.,known=True))
        if (delays.get('known') is not True
                or row['finish_ns']!=row['handler_end_ns']+round(delays['parent_receipt_ms']*1e6)
                or row['worker_next_ready_ns']!=row['handler_end_ns']+round(delays['worker_next_ready_ms']*1e6)):
            raise ValueError('Saved parent receipt/worker READY differs from fixed provider')
        expected[row['alias']]=max(expected[row['alias']],row['worker_next_ready_ns'])
    if state['owner_last_ready']!=expected:
        raise ValueError('owner_last_ready loses absolute previous worker READY')
    for owner in engine.aliases:
        rows=sorted((state['cohort'][rid] for rid in state['starts'] if state['cohort'][rid]['alias']==owner),
                    key=lambda row:(row['arrival_ns'],list(state['cohort']).index(row['request_id'])))
        last=now
        for row in rows:
            rid=row['request_id']
            if state['starts'][rid]<request_start_bound(engine,row,owner_ready_ns=last):
                raise ValueError('Saved owner FIFO START precedes worker READY/input supply')
            record=next((r for r in ended if r['request_id']==rid),None)
            if record:last=record['worker_next_ready_ns']
    done=state['node_completion_ns']
    if type(done) is not dict or len(done)!=state['all_completed_nodes']:
        raise ValueError('Complete original absolute node completion clocks required')
    for (owner,rid,node),at in done.items():
        row=state['cohort'].get(rid)
        if (row is None or row['alias']!=owner or type(at) is not int
                or not state['starts'][rid]<=at<=clock
                or node not in {n['id'] for n in engine.plans[owner,row['task'],row['pool_id']]['nodes']}):
            raise ValueError('Invalid original node completion clock')
    for owner,running in state['running'].items():
        rid=running['row']['request_id']
        if {node for o,r,node in done if (o,r)==(owner,rid)}!=running['done']:
            raise ValueError('Completed node clocks disagree with current DAG')
    for release,_,owner,node_id in state['ready']:
        running=state['running'][owner];row=running['row'];node=running['nodes'][node_id]
        supplied=(engine.release_provider(node,row) if engine.release_provider else node.get('release_ns',0))
        if type(supplied) is dict:
            predecessors=[edge['before'] for edge in engine.plans[owner,row['task'],row['pool_id']]['edges'] if edge['after']==node_id]
            completed=max((done[owner,row['request_id'],n] for n in predecessors),default=running['start_ns'])
            expected_release=max(running['start_ns']+supplied['not_before_offset_ns'],completed+supplied['after_predecessors_ns'])
        else:expected_release=running['start_ns']+supplied
        if release!=expected_release or release<=clock:
            raise ValueError('Saved absolute DAG host release shifted across cutoff')


def _validate_advance(state,engine,contract,scheduler_state,diagnostic):
    a=state['advance'];clock=state['clock'];origin=a['origin_ns']
    if (type(a) is not dict or set(a)!=ADVANCE_FIELDS or type(origin) is not int
            or not contract['original_now_ns']<=origin<=clock<a['boundary_ns']
            or type(a['boundary_ns']) is not int):
        raise ValueError('Complete unfinished original interval required')
    s=scheduler_state['scheduler']
    if (s['last_clock_ns']!=origin or scheduler_state['owner_last_ready']!=state['owner_last_ready']
            or scheduler_state['oracle']['lifecycle']!=engine_states(engine.aliases,state['running'],state['active'])):
        raise ValueError('Engine and scheduler origin/lifecycle/owner READY disagree')
    if set(a['origin_active'])!=set(state['active']) or set(a['rates'])!=set(state['active']):
        raise ValueError('Unfinished interval active-rate domain changed')
    ticks={}
    for key,node in state['active'].items():
        rate=a['rates'][key]
        expected_rate=float(key[0]==s['held_owner'] and s['service_available']) if node['kind']=='gpu' else 1.
        if type(rate) not in (int,float) or rate!=expected_rate:
            raise ValueError('Saved interval rate differs from supported held-context scheduler')
        old=a['origin_active'][key]
        if (old['kind']!=node['kind'] or old['remaining_ns']<=0
                or node['remaining_ns']!=old['remaining_ns']-(clock-origin)*rate):
            raise ValueError('Saved remaining work disagrees with original absolute interval')
        row=state['running'][key[0]]['row'];plan_node=state['running'][key[0]]['nodes'][key[1]]
        duration=engine.duration_provider(plan_node,row) if engine.duration_provider else plan_node['duration_ns']
        if old['remaining_ns']>duration:
            raise ValueError('Saved node work exceeds its bound duration')
        if rate:ticks[key]=max(1,math.ceil(old['remaining_ns']/rate))
    if ticks!=a['finish_ticks']:
        raise ValueError('Original integer completion ticks changed')
    candidates=[origin+min(ticks.values()) if ticks else math.inf]
    candidates.extend(stamp for stamp,_,_ in s['context_events'] if stamp>origin)
    if s['review_deadline_ns'] is not None and s['review_deadline_ns']>origin:candidates.append(s['review_deadline_ns'])
    for field in ('arrivals','ready','finish_events','wake_events'):
        if state[field]:candidates.append(state[field][0][0])
    for owner,queue in state['pending'].items():
        if queue and owner not in state['running'] and owner not in state['unready']:
            at=request_start_bound(engine,queue[0],owner_ready_ns=state['owner_last_ready'][owner])
            if at>origin:candidates.append(at)
    if origin<contract['deadline_ns']:candidates.append(contract['deadline_ns'])
    if contract['drain_deadline_ns'] is not None:candidates.append(contract['drain_deadline_ns'])
    if a['boundary_ns']!=min(candidates):
        raise ValueError('Saved original event boundary differs from all absolute event clocks')
    count=a['origin_resource_count']
    if type(count) is not int or count<0 or len(state['resource_transitions'])!=count+int(clock>origin):
        raise ValueError('Unfinished resource prefix length changed')
    previous=dict(state,clock=origin,active=copy.deepcopy(a['origin_active']),
        energy=a['origin_energy'],peak=a['origin_peak'],energy_known=a['origin_energy_known'],
        memory_known=a['origin_memory_known'],resource_transitions=state['resource_transitions'][:count])
    structural.validate_resources(previous,engine,contract)
    _validate_diagnostic(a['diagnostics_origin'],previous,engine)
    if engine.queue_diagnostics is not None and clock>origin:
        fresh=copy.copy(engine.queue_diagnostics)
        install_diagnostic_state(fresh,a['diagnostics_origin'])
        fresh.observe(origin,clock-origin,state['pending'],state['running'],a['origin_active'],a['rates'])
        if diagnostic_state(fresh)!=diagnostic:
            raise ValueError('Queue prefix differs from unfinished interval integration')
    if clock==origin:
        if a['resource'] is not None:raise ValueError('Unconsumed interval has future resource labels')
    else:
        expected=dict(state['resource_transitions'][-1]);expected.pop('until_ns')
        if expected!=a['resource']:raise ValueError('Saved resource interval differs from prefix')


def _validate_state(state,engine,contract,scheduler_state,diagnostic):
    if type(state) is not dict or set(state)!=set(STATE_FIELDS):raise ValueError('Complete lifecycle engine state required')
    structural.validate_state({k:state[k] for k in structural.STATE_FIELDS},engine,contract)
    _validate_lifecycle(state,engine,contract)
    _validate_diagnostic(diagnostic,state,engine)
    _validate_advance(state,engine,contract,scheduler_state,diagnostic)


def pause_result(engine,state,contract):
    binding=scheduler_binding(contract,state['advance']['origin_ns'])
    scheduler=scheduler_cp.export_scheduler_snapshot(engine.scheduler,binding=binding,owner_last_ready=state['owner_last_ready'])
    diagnostic=diagnostic_state(engine.queue_diagnostics)
    detached=scheduler_cp.import_scheduler_snapshot(engine.scheduler,scheduler,binding=binding)
    _validate_state(state,engine,contract,detached,diagnostic)
    accounting=prefix_accounting(state,contract)
    result=dict(schema=SCHEMA,kind='SIMULATED_CONTINUATION_NOT_NATIVE_OBSERVATION',atomic_stage=STAGE,
        contract=copy.deepcopy(contract),state=scheduler_cp.encode(checkpoint_value(state)),scheduler=scheduler,
        diagnostics=scheduler_cp.encode(diagnostic),six_term_prefix=scheduler_cp.encode(accounting),
        physical_progress_observed=False,native_observation_assimilation_supported=False)
    result['checkpoint_sha256']=digest(result)
    return dict(status='PAUSED',cutoff_ns=state['clock'],checkpoint=result,six_term_prefix=accounting,
        full_future_cohort_retained=True,original_scheduler_budget_retained=True,
        physical_progress_observed=False,native_observation_assimilation_supported=False)


def decode_checkpoint(engine,checkpoint,contract):
    keys={'schema','kind','atomic_stage','contract','state','scheduler','diagnostics','six_term_prefix',
        'physical_progress_observed','native_observation_assimilation_supported','checkpoint_sha256'}
    if (type(checkpoint) is not dict or set(checkpoint)!=keys or checkpoint['schema']!=SCHEMA
            or checkpoint['kind']!='SIMULATED_CONTINUATION_NOT_NATIVE_OBSERVATION'
            or checkpoint['atomic_stage']!=STAGE or checkpoint['contract']!=contract
            or checkpoint['physical_progress_observed'] is not False
            or checkpoint['native_observation_assimilation_supported'] is not False
            or checkpoint['checkpoint_sha256']!=digest({k:v for k,v in checkpoint.items() if k!='checkpoint_sha256'})):
        raise ValueError('Changed/foreign lifecycle engine checkpoint or execution contract')
    state=scheduler_cp.decode(checkpoint['state']);diagnostic=scheduler_cp.decode(checkpoint['diagnostics'])
    scheduler=scheduler_cp.import_scheduler_snapshot(engine.scheduler,checkpoint['scheduler'],
        binding=scheduler_binding(contract,state['advance']['origin_ns']))
    try:_validate_state(state,engine,contract,scheduler,diagnostic)
    except (KeyError,TypeError,OverflowError) as exc:raise ValueError('Malformed lifecycle engine checkpoint') from exc
    if scheduler_cp.decode(checkpoint['six_term_prefix'])!=prefix_accounting(state,contract):
        raise ValueError('Changed original six-term prefix accounting')
    state['_scheduler']=scheduler;state['_diagnostic']=diagnostic
    return state


def commit_checkpoint(engine,state):
    scheduler=engine.scheduler;oracle=scheduler.oracle;mark=oracle.next_owner
    objects=[(scheduler,state['_scheduler']['scheduler']),(oracle,state['_scheduler']['oracle']),
        (mark,state['_scheduler']['mark'])]
    if engine.queue_diagnostics is not None:objects.append((engine.queue_diagnostics,state['_diagnostic']))
    staged=[(obj,{**vars(obj),**copy.deepcopy(update)}) for obj,update in objects]
    previous=[(obj,vars(obj).copy()) for obj,_ in objects]
    try:
        for obj,values in staged:obj.__dict__=values
    except BaseException:
        for obj,values in previous:obj.__dict__=values
        raise
