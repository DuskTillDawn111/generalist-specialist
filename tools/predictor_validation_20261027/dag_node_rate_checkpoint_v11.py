"""v11 node-rate checkpoints; reuse original DAG structural/prefix checks."""
from __future__ import annotations

import copy
import inspect
import math

from . import dag_phase_checkpoint_v10 as previous
from . import dag_checkpoint_v7 as base
from .dag_frozen_model_v10 import FrozenDagModel,SOURCE_CACHE
from .dag_node_rate_scheduler_v11 import CheckpointNodeScheduler
from .implementation_source_closure_v27 import implementation_sources

SCHEMA='gsdeploy.simulated_node_rate_DAG_checkpoint.v11'
STATE_FIELDS=[*previous.STATE_FIELDS,'node_work']


def contract(engine,requests,now_ns,deadline_ns,drain_deadline_ns):
    from .dag_node_rate_engine_v11 import DagNodeRateCheckpointModel
    model=engine._frozen_model
    if (type(engine) is not DagNodeRateCheckpointModel or type(model) is not FrozenDagModel
            or engine.aliases is not model.aliases or engine.plans is not model.plans
            or type(engine.scheduler) is not CheckpointNodeScheduler):
        raise ValueError('Explicit v11 node-rate consumer and frozen DAG required')
    if (type(now_ns) is not int or now_ns<0 or type(deadline_ns) is not int or now_ns>=deadline_ns
            or drain_deadline_ns is not None and
                (type(drain_deadline_ns) is not int or drain_deadline_ns<deadline_ns)):
        raise ValueError('Original integer measurement and fixed drain required')
    binding=engine.checkpoint_binding
    if (type(binding) is not dict or set(binding)!={'execution_context_sha256','evidence'}
            or type(binding['execution_context_sha256']) is not str
            or len(binding['execution_context_sha256'])!=64 or not binding['evidence']):
        raise ValueError('Explicit immutable node-rate source evidence/context required')
    for ref in binding['evidence']:SOURCE_CACHE.verify(ref)
    choices=dict(duration=engine.duration_provider,release=engine.release_provider,
        lifecycle=engine.lifecycle_provider.delays,power=engine.power_provider.power if engine.power_provider else None,
        memory=engine.memory_provider.memory if engine.memory_provider else None,
        diagnostics=engine.queue_diagnostics.duration if engine.queue_diagnostics else None,
        phase=engine.phase_provider,request_start=engine.request_start_provider,
        request_start_from_owner_ready=engine.request_start_provider.from_owner_ready
            if hasattr(engine.request_start_provider,'from_owner_ready') else None,
        non_gpu_rate=engine.non_gpu_rate_provider)
    # Scheduler dynamic state is separate; initial provider identity/config is
    # bound before consumption and re-created for complete prefix replay.
    provider=engine.scheduler.provider
    choices.update(node_decide=provider.decide,node_fresh=provider.fresh,
        node_configuration=provider.checkpoint_configuration,node_state=provider.saved_state,
        node_restore=provider.restore_state,node_provenance=provider.provenance)
    providers={name:base.callable_identity(value) for name,value in choices.items()}
    sources=implementation_sources(__file__,inspect.getfile(type(engine)))
    extra=set()
    for value in choices.values():
        if value is None:continue
        target=value if inspect.isfunction(value) or inspect.ismethod(value) else type(value)
        path=inspect.getsourcefile(target)
        if path is None:raise ValueError('Explicit node/provider implementation source required')
        extra.add(path)
    for path in sorted(extra):
        ref=base.receipt(path)
        if ref not in sources:sources.append(ref)
    options={name:getattr(engine,name) for name in ('allow_profiled_replay','allow_independent_copy_hypothesis',
        'allow_context_time_slice_hypothesis','allow_no_implicit_default_stream_hypothesis',
        'scenario','max_events','require_energy','require_memory')}
    return dict(binding=copy.deepcopy(binding),sources=sources,providers=providers,
        scheduler_configuration=engine.scheduler.checkpoint_configuration(),
        aliases_sha256=model.aliases_sha256,plans_sha256=model.plans_sha256,
        cohort_sha256=base.digest(base.encode(requests)),options=options,
        original_now_ns=now_ns,deadline_ns=deadline_ns,drain_deadline_ns=drain_deadline_ns,
        provider_state_contract='Explicit deterministic provider fresh/config/state/restore; complete prefix replay; not an observed GPU cursor')


def _compact(state):
    result=previous._compact({k:state[k] for k in previous.STATE_FIELDS})
    result['node_work']=state['node_work']
    return result


def _hydrate(state,engine):
    if type(state) is not dict or set(state)!=set(STATE_FIELDS):
        raise ValueError('Complete v11 dynamic node state required')
    work=state['node_work']
    hydrated=previous._hydrate({k:state[k] for k in previous.STATE_FIELDS},engine)
    hydrated['node_work']=work
    return hydrated


def _fresh(engine):
    from .dag_node_rate_engine_v11 import DagNodeRateCheckpointModel
    diagnostic=None
    if engine.queue_diagnostics is not None:
        from .dag_queue_diagnostics_v6 import DagQueueDiagnostics
        old=engine.queue_diagnostics
        if type(old) is not DagQueueDiagnostics:raise ValueError('Supported original queue diagnostics required')
        diagnostic=DagQueueDiagnostics(old.aliases,old.plans,old.duration_provider,
            now_ns=old.initial_ns,resource_demand_provider=old.resource_demand_provider)
    return DagNodeRateCheckpointModel(engine.aliases,engine.plans,frozen_model=engine._frozen_model,
        **{name:getattr(engine,name) for name in ('allow_profiled_replay','allow_independent_copy_hypothesis',
            'allow_context_time_slice_hypothesis','allow_no_implicit_default_stream_hypothesis',
            'lifecycle_provider','scenario','duration_provider','release_provider','power_provider',
            'memory_provider','require_energy','require_memory','max_events','phase_provider',
            'checkpoint_binding','request_start_provider','non_gpu_rate_provider')},
        queue_diagnostics=diagnostic,scheduler=CheckpointNodeScheduler(engine.scheduler.provider))


def export_checkpoint(state,engine,binding):
    diagnostic=engine.queue_diagnostics
    diag=None if diagnostic is None else {k:v for k,v in vars(diagnostic).items() if k not in base.DIAGNOSTIC_EXCLUDED}
    result=dict(schema=SCHEMA,kind='SIMULATED_NODE_CLOCK_STATE_NOT_OBSERVED_GPU_CURSOR',
        contract=binding,state=base.encode(_compact(state)),
        scheduler_state=base.encode(engine.scheduler.saved_state()),diagnostics=base.encode(diag),
        visible_feedback_applied=False,physical_progress_observed=False)
    result['checkpoint_sha256']=base.digest(result)
    return result


def import_checkpoint(checkpoint,engine,binding):
    fields={'schema','kind','contract','state','scheduler_state','diagnostics',
        'visible_feedback_applied','physical_progress_observed','checkpoint_sha256'}
    if (type(checkpoint) is not dict or set(checkpoint)!=fields or checkpoint['schema']!=SCHEMA
            or checkpoint['checkpoint_sha256']!=base.digest({k:v for k,v in checkpoint.items() if k!='checkpoint_sha256'})
            or checkpoint['contract']!=binding or checkpoint['kind']!='SIMULATED_NODE_CLOCK_STATE_NOT_OBSERVED_GPU_CURSOR'
            or checkpoint['visible_feedback_applied'] is not False or checkpoint['physical_progress_observed'] is not False):
        raise ValueError('Changed/foreign node checkpoint or immutable provider contract')
    state=_hydrate(base.decode(checkpoint['state']),engine)
    base.validate_state(state,engine,binding)
    previous._lifecycle(state,engine,binding)
    if type(state['node_work']) is not dict:raise ValueError('Explicit reference-work ledger required')
    for key,work in state['node_work'].items():
        if (type(key) is not tuple or len(key)!=2 or key[0] not in state['starts']
                or type(work) is not dict or set(work)!={'reference_ns','consumed_ns','remaining_ns'}
                or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in work.values())
                or work['reference_ns']<=0
                or not math.isclose(work['reference_ns'],work['consumed_ns']+work['remaining_ns'],rel_tol=1e-14)):
            raise ValueError('Reference-work ledger violates conservation')
    if state['event_count']<=0:raise ValueError('Checkpoint must consume a real event prefix')
    replay=_fresh(engine).forecast(list(state['cohort'].values()),binding['original_now_ns'],
        binding['deadline_ns'],drain_deadline_ns=binding['drain_deadline_ns'],
        pause_after_event_count=state['event_count'])
    expected=replay.get('checkpoint')
    if expected is None or any(checkpoint[k]!=expected[k] for k in ('state','scheduler_state','diagnostics')):
        raise ValueError('Node work/provider state differs from complete deterministic prefix replay')
    diagnostic=base.decode(checkpoint['diagnostics'])
    if engine.queue_diagnostics is None:
        if diagnostic is not None:raise ValueError('Queue diagnostics connection changed')
    else:
        base.validate_diagnostic(diagnostic,state,engine.queue_diagnostics)
    # Stage all validation before committing scheduler/diagnostic state.
    staged=CheckpointNodeScheduler(engine.scheduler.provider)
    staged.restore_state(base.decode(checkpoint['scheduler_state']))
    engine.scheduler=staged
    if engine.queue_diagnostics is not None:
        for name,value in diagnostic.items():setattr(engine.queue_diagnostics,name,value)
    return state
