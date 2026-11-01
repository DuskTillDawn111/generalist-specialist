"""Compact, source-bound phase-DAG checkpoints with complete-prefix validation.

Static DAG content stays in an immutable model. A checkpoint carries dynamic
progress and absolute scheduler state; deterministic prefix re-execution proves
their joint consistency before continuation. No observed GPU cursor is claimed.
"""
from __future__ import annotations

import inspect
import math

from . import dag_checkpoint_v7 as base
from .dag_checkpoint_v9 import request_start_bound
from .dag_frozen_model_v10 import FrozenDagModel,SOURCE_CACHE
from .dag_scheduler_state_v10 import CheckpointTrainedScheduler,STATE_FIELDS as SCHEDULER_FIELDS

STATE_FIELDS=base.STATE_FIELDS
encode=base.encode
decode=base.decode
receipt=base.receipt
SCHEMA='gsdeploy.simulated_phase_DAG_compact_checkpoint.v10'


def contract(engine,requests,now_ns,deadline_ns,drain_deadline_ns):
    model=engine._frozen_model
    if (type(model) is not FrozenDagModel or engine.aliases is not model.aliases
            or engine.plans is not model.plans or type(engine.scheduler) is not CheckpointTrainedScheduler):
        raise ValueError('Original immutable DAG model and trained phase scheduler required')
    binding=engine.checkpoint_binding
    if (type(binding) is not dict or set(binding)!={'execution_context_sha256','evidence'}
            or not isinstance(binding['execution_context_sha256'],str)
            or len(binding['execution_context_sha256'])!=64 or not binding['evidence']):
        raise ValueError('Complete immutable model evidence and execution context required')
    for ref in binding['evidence']:
        SOURCE_CACHE.verify(ref)
    sources={__file__,inspect.getsourcefile(type(engine)),inspect.getsourcefile(FrozenDagModel),
        inspect.getsourcefile(CheckpointTrainedScheduler),inspect.getsourcefile(base),
        inspect.getsourcefile(CheckpointTrainedScheduler.__mro__[1])}
    providers={}
    choices=dict(duration=engine.duration_provider,release=engine.release_provider,
        lifecycle=engine.lifecycle_provider.delays if engine.lifecycle_provider else None,
        power=engine.power_provider.power if engine.power_provider else None,
        memory=engine.memory_provider.memory if engine.memory_provider else None,
        diagnostics=engine.queue_diagnostics.duration if engine.queue_diagnostics else None,
        phase=engine.phase_provider,scheduler=engine.scheduler.decide,
        request_start=engine.request_start_provider,
        request_start_from_owner_ready=engine.request_start_provider.from_owner_ready
            if engine.request_start_provider is not None and hasattr(engine.request_start_provider,'from_owner_ready') else None)
    for name,provider in choices.items():
        providers[name]=base.callable_identity(provider)
        if provider is not None:
            target=provider if inspect.isfunction(provider) or inspect.ismethod(provider) else type(provider)
            source=inspect.getsourcefile(target)
            if source is None:raise ValueError('Explicit provider implementation source required')
            sources.add(source)
    options={name:getattr(engine,name) for name in (
        'allow_profiled_replay','allow_independent_copy_hypothesis','allow_context_time_slice_hypothesis',
        'allow_no_implicit_default_stream_hypothesis','scenario','max_events','require_energy','require_memory')}
    return dict(binding=binding,sources=[receipt(p) for p in sorted(sources)],providers=providers,
        aliases_sha256=model.aliases_sha256,plans_sha256=model.plans_sha256,
        cohort_sha256=base.digest(encode(requests)),options=options,
        original_now_ns=now_ns,deadline_ns=deadline_ns,drain_deadline_ns=drain_deadline_ns,
        provider_state_contract='Immutable DAG/provider identities; separate absolute scheduler state; complete deterministic prefix replay verifies continuation')


def _compact(state):
    if set(state)!=set(STATE_FIELDS):raise ValueError('Complete dynamic DAG state required')
    result=dict(state)
    result['running']={owner:dict(row=value['row'],start_ns=value['start_ns'],done=value['done'])
        for owner,value in state['running'].items()}
    return result


def _hydrate(state,engine):
    if type(state) is not dict or set(state)!=set(STATE_FIELDS):
        raise ValueError('Complete compact dynamic DAG state required')
    for owner,value in state['running'].items():
        if type(value) is not dict or set(value)!={'row','start_ns','done'} or type(value['done']) is not set:
            raise ValueError('Explicit compact running progress required')
        row=value['row'];plan=engine.plans[owner,row['task'],row['pool_id']]
        nodes={n['id']:n for n in plan['nodes']};incoming={n:0 for n in nodes};successors={n:[] for n in nodes}
        for edge in plan['edges']:
            successors[edge['before']].append(edge['after'])
            if edge['before'] not in value['done']:incoming[edge['after']]+=1
        value.update(nodes=nodes,incoming=incoming,successors=successors)
    return state


def export_checkpoint(state,engine,binding):
    diagnostic=engine.queue_diagnostics
    diag=None if diagnostic is None else {k:v for k,v in vars(diagnostic).items() if k not in base.DIAGNOSTIC_EXCLUDED}
    result=dict(schema=SCHEMA,kind='SIMULATED_PREDICTOR_STATE_NOT_OBSERVED_GPU_CURSOR',
        atomic_boundary='After interval consumption/node completion; before next same-clock closure',
        contract=binding,state=encode(_compact(state)),scheduler_state=encode(engine.scheduler.saved_state()),
        diagnostics=encode(diag),visible_feedback_applied=False,physical_progress_observed=False)
    result['checkpoint_sha256']=base.digest(result)
    return result


def _lifecycle(state,engine,binding):
    for field in ('finishes','receipt_pending'):
        for rid,record in state[field].items():
            row=state['cohort'][rid]
            delays=(dict(parent_receipt_ms=0.,worker_next_ready_ms=0.) if engine.lifecycle_provider is None
                else engine.lifecycle_provider.delays(dict(row),engine.aliases,engine.scenario))
            if engine.lifecycle_provider is not None and delays.get('known') is not True:
                raise ValueError('Known source-bound joint lifecycle required')
            for source,target in (('parent_receipt_ms','finish_ns'),('worker_next_ready_ms','worker_next_ready_ns')):
                value=delays.get(source)
                if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                    raise ValueError('Finite nonnegative joint lifecycle delays required')
                if record[target]!=record['handler_end_ns']+round(value*1e6):
                    raise ValueError('Checkpoint RESULT/READY differs from bound lifecycle')
    ready={rid:r['worker_next_ready_ns'] for name in ('finishes','receipt_pending') for rid,r in state[name].items()}
    order={rid:i for i,rid in enumerate(state['cohort'])}
    for owner in engine.aliases:
        preceding=binding['original_now_ns']
        actual=sorted((at,order[rid],rid) for rid,at in state['starts'].items() if state['cohort'][rid]['alias']==owner)
        expected=sorted((r['arrival_ns'],order[rid],rid) for rid,r in state['cohort'].items() if r['alias']==owner)
        if [r[2] for r in actual]!=[r[2] for r in expected[:len(actual)]]:
            raise ValueError('Historical START is not the original FIFO prefix')
        for index,(at,_,rid) in enumerate(actual):
            if at<request_start_bound(engine,state['cohort'][rid],owner_ready_ns=preceding):
                raise ValueError('Historical START bypasses original supply or preceding READY')
            if index+1<len(actual) and rid not in ready:
                raise ValueError('Later START lacks explicit preceding worker READY')
            preceding=ready.get(rid,preceding)


def _fresh(engine,binding):
    from .dag_phase_engine_v10 import DagPhaseCheckpointModel
    if type(engine) is not DagPhaseCheckpointModel:
        raise ValueError('Actual independent v10 phase DAG engine required')
    diagnostic=None
    if engine.queue_diagnostics is not None:
        from .dag_queue_diagnostics_v6 import DagQueueDiagnostics
        old=engine.queue_diagnostics
        if type(old) is not DagQueueDiagnostics:raise ValueError('Explicit supported queue diagnostics required')
        diagnostic=DagQueueDiagnostics(old.aliases,old.plans,old.duration_provider,
            now_ns=old.initial_ns,resource_demand_provider=old.resource_demand_provider)
    return DagPhaseCheckpointModel(engine.aliases,engine.plans,frozen_model=engine._frozen_model,
        **{name:getattr(engine,name) for name in ('allow_profiled_replay','allow_independent_copy_hypothesis',
            'allow_context_time_slice_hypothesis','allow_no_implicit_default_stream_hypothesis','lifecycle_provider',
            'scenario','duration_provider','release_provider','power_provider','memory_provider','require_energy',
            'require_memory','max_events','phase_provider','checkpoint_binding','request_start_provider')},
        queue_diagnostics=diagnostic,scheduler=CheckpointTrainedScheduler(engine.scheduler.policy))


def import_checkpoint(checkpoint,engine,binding):
    fields={'schema','kind','atomic_boundary','contract','state','scheduler_state','diagnostics',
        'visible_feedback_applied','physical_progress_observed','checkpoint_sha256'}
    if (type(checkpoint) is not dict or set(checkpoint)!=fields or checkpoint['schema']!=SCHEMA
            or checkpoint['checkpoint_sha256']!=base.digest({k:v for k,v in checkpoint.items() if k!='checkpoint_sha256'})
            or checkpoint['contract']!=binding or checkpoint['kind']!='SIMULATED_PREDICTOR_STATE_NOT_OBSERVED_GPU_CURSOR'
            or checkpoint['visible_feedback_applied'] is not False or checkpoint['physical_progress_observed'] is not False):
        raise ValueError('Changed/foreign compact phase checkpoint or original execution contract')
    state=_hydrate(decode(checkpoint['state']),engine)
    scheduler=decode(checkpoint['scheduler_state'])
    if type(scheduler) is not dict or set(scheduler)!=set(SCHEDULER_FIELDS):
        raise ValueError('Complete separate absolute scheduler state required')
    base.validate_state(state,engine,binding)
    _lifecycle(state,engine,binding)
    if state['event_count']<=0:raise ValueError('Checkpoint must consume a real event prefix')
    replay=_fresh(engine,binding).forecast(list(state['cohort'].values()),binding['original_now_ns'],
        binding['deadline_ns'],drain_deadline_ns=binding['drain_deadline_ns'],
        pause_after_event_count=state['event_count'])
    expected=replay.get('checkpoint')
    if expected is None or any(checkpoint[k]!=expected[k] for k in ('state','scheduler_state','diagnostics')):
        raise ValueError('Compact DAG/scheduler state differs from complete deterministic original-prefix replay')
    diagnostic=decode(checkpoint['diagnostics'])
    if engine.queue_diagnostics is None:
        if diagnostic is not None:raise ValueError('Queue diagnostic connection changed')
    else:
        base.validate_diagnostic(diagnostic,state,engine.queue_diagnostics)
        for name,value in diagnostic.items():setattr(engine.queue_diagnostics,name,value)
    engine.scheduler.restore_state(scheduler)
    return state
