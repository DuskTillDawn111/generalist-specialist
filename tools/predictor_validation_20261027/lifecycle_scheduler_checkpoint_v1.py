"""Pure simulated v49/v24 scheduler state, not a main-engine checkpoint.

Caller must exclusively own paused objects at POST_DECISION_PRE_ADVANCE.
Export/import never call snapshot/decide/observe/update/start_restore. Restore
validates and stages everything before replacing any object's dynamic fields.
The v21 main engine has not been connected to this component.
"""
from __future__ import annotations

from collections import Counter, deque
import copy
from dataclasses import fields
import json
import math

from .context_initial_prior_v14 import InitialContextPrior
from .context_initial_prior_v15 import JointInitialContextPrior
from .cumulative_lifecycle_hazard_v24 import CumulativeLifecycleHazard, SCHEMA as HAZARD_SCHEMA
from .handoff_candidate_v14 import HandoffCandidate
from .held_context_scheduler_v12_1 import ContextAvailabilityDecision
from .implementation_source_closure_v27 import implementation_sources
from .learned_lifecycle_scheduler_v49 import LearnedLifecycleOracle, LearnedLifecycleScheduler
from .native_collect import receipt
from .owner_lifecycle_mark_v21 import OwnerLifecycle
from .physical_context_mark_bridge_v56 import PhysicalContextMark
from .profiling import digest
from .projected_handler_bridge_v44 import ObservableReferenceMark, ObservableSaveHazard
from .projected_handler_context_v42 import ObservableContextCandidate
from .projected_state_save_hazard_v57 import ProjectedStateSaveHazard

SCHEMA = 'gsdeploy.simulated_lifecycle_scheduler_component.v1'
STAGE = 'POST_DECISION_PRE_ADVANCE'
SCHEDULER_FIELDS = tuple(('held_owner held_knowledge restore_start_ns service_available '
    'review_deadline_ns pending_switch context_events pending_since last_clock_ns last_frontier '
    'transitions queries initial_decision initial_service_at_ns initial_transition_decision '
    'expired_review_republication_count').split())
ORACLE_FIELDS = tuple(('initial_prediction selected_initial restore_key last_restored_owner '
    'previous_held_owner hazard last_frontier uncertainty decisions lifecycle lifecycle_clock_ns '
    'last_lifecycle').split())
ORACLE_STATIC = tuple(('initial_prior controller next_owner handoff public_setup public_workload '
    'identity quantile initial_rule handoff_rule allow_pooling allow_transfer allow_extrapolation').split())
HAZARD_FIELDS = set(('schema controller_identity held_owner restore_ns observation_ns last_update_ns '
    'restore_key frontier lifecycle hazard_budget consumed_hazard summation_compensation '
    'terminal_save_candidate_ns next_review_ns uncertainty_reasons visible_update_count state_sha256').split())
BINDING_FIELDS = set(('execution_context_sha256 cohort_sha256 graph_sha256 original_now_ns '
    'deadline_ns drain_deadline_ns clock_ns atomic_stage').split())
DATACLASSES = {cls.__name__: cls for cls in (OwnerLifecycle, ContextAvailabilityDecision)}


def encode(value):
    """Exact JSON round trip for these supported state types; no object pickle."""
    if value is None or type(value) in (bool, int, str):
        return ['scalar', value]
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('Nonfinite scheduler checkpoint value')
        return ['scalar', value]
    if type(value) in DATACLASSES.values():
        return [type(value).__name__, [[f.name, encode(getattr(value, f.name))] for f in fields(value)]]
    if type(value) in (dict, Counter):
        return ['Counter' if type(value) is Counter else 'dict',
                [[encode(k), encode(v)] for k, v in value.items()]]
    if type(value) in (list, tuple, set, frozenset, deque):
        items = [encode(item) for item in value]
        if type(value) in (set, frozenset):
            items.sort(key=lambda item: json.dumps(item, sort_keys=True))
        return [type(value).__name__, items]
    raise ValueError('Unsupported checkpoint value: ' + type(value).__name__)


def decode(value):
    if type(value) is not list or len(value) != 2:
        raise ValueError('Typed checkpoint value required')
    kind, body = value
    if kind == 'scalar':
        if body is not None and type(body) not in (bool, int, float, str):
            raise ValueError('Invalid checkpoint scalar')
        if type(body) is float and not math.isfinite(body):
            raise ValueError('Nonfinite checkpoint scalar')
        return body
    if type(body) is not list:
        raise ValueError('Invalid checkpoint container')
    if kind in DATACLASSES:
        names = [name for name, _ in body]
        cls = DATACLASSES[kind]
        if names != [field.name for field in fields(cls)]:
            raise ValueError('Changed checkpoint dataclass fields')
        return cls(**{name: decode(item) for name, item in body})
    if kind in ('dict', 'Counter'):
        rows = [(decode(k), decode(v)) for k, v in body]
        result = dict(rows)
        if len(result) != len(rows):
            raise ValueError('Duplicate checkpoint key')
        return Counter(result) if kind == 'Counter' else result
    constructors = dict(list=list, tuple=tuple, set=set, frozenset=frozenset, deque=deque)
    if kind not in constructors:
        raise ValueError('Unsupported checkpoint container')
    items = [decode(item) for item in body]
    result = constructors[kind](items)
    if kind in ('set', 'frozenset') and len(result) != len(items):
        raise ValueError('Duplicate checkpoint set member')
    return result


def _shape(obj, cls, keys):
    if type(obj) is not cls or set(vars(obj)) != set(keys):
        raise ValueError('Unsupported class or untracked fields: ' + type(obj).__name__)


def _plain_component(obj):
    """Only explicitly audited stateless components are static bindings."""
    shapes = {InitialContextPrior: ('rows', 'model_sha256', 'excluded'),
        JointInitialContextPrior: ('rows', 'model_sha256'), HandoffCandidate: ('model',),
        ObservableContextCandidate: ('model', 'coefficients'), ProjectedStateSaveHazard: ('model',)}
    if obj is None:
        return None
    if type(obj) not in shapes:
        raise ValueError('Unsupported static component: ' + type(obj).__name__)
    _shape(obj, type(obj), shapes[type(obj)])
    return dict(type=type(obj).__module__ + '.' + type(obj).__qualname__, state=encode(vars(obj)))


def _check_receipts(value):
    if type(value) is dict:
        if set(value) == {'path', 'sha256'} and receipt(value['path']) != value:
            raise ValueError('Checkpoint component source changed')
        for item in value.values():
            _check_receipts(item)
    elif type(value) in (tuple, list):
        for item in value:
            _check_receipts(item)


def _components(scheduler):
    _shape(scheduler, LearnedLifecycleScheduler, (*SCHEDULER_FIELDS, 'owners', 'oracle', 'allow_source_replay'))
    oracle = scheduler.oracle
    _shape(oracle, LearnedLifecycleOracle, (*ORACLE_FIELDS, *ORACLE_STATIC))
    controller, mark = oracle.controller, oracle.next_owner
    _shape(controller, CumulativeLifecycleHazard,
        ('_candidate', 'edges', '_sources', '_policy', '_seed', '_configuration', 'identity'))
    if type(mark) not in (ObservableReferenceMark, PhysicalContextMark):
        raise ValueError('Unsupported mark checkpoint class')
    _shape(mark, type(mark), ('candidate', 'states', 'clock'))
    if scheduler.allow_source_replay is not False:
        raise ValueError('Observed source replay is not simulated checkpoint state')
    return oracle, controller, mark


def _contract(scheduler, binding):
    oracle, controller, mark = _components(scheduler)
    if (type(binding) is not dict or set(binding) != BINDING_FIELDS or binding['atomic_stage'] != STAGE
            or any(type(binding[k]) is not int for k in ('original_now_ns', 'deadline_ns', 'clock_ns'))
            or not binding['original_now_ns'] <= binding['clock_ns']
            or not binding['original_now_ns'] < binding['deadline_ns']
            or (binding['drain_deadline_ns'] is not None and
                (type(binding['drain_deadline_ns']) is not int
                 or binding['drain_deadline_ns'] < max(binding['clock_ns'], binding['deadline_ns'])))):
        raise ValueError('Original windows and explicit atomic binding required')
    for key in ('execution_context_sha256', 'cohort_sha256', 'graph_sha256'):
        if (type(binding[key]) is not str or len(binding[key]) != 64
                or any(c not in '0123456789abcdef' for c in binding[key])):
            raise ValueError('Complete source-bound execution/cohort/graph identities required')
    configuration = controller._configuration
    if (controller.identity != digest(configuration)
            or controller._policy != configuration['budget_policy'] or controller._seed != configuration['seed']
            or controller.edges != tuple(configuration['age_edges_ns'])
            or list(controller._sources) != configuration['sources']):
        raise ValueError('Controller immutable configuration changed')
    candidate = controller._candidate
    if type(candidate) is ObservableSaveHazard:
        _shape(candidate, ObservableSaveHazard, ('candidate',))
        candidate_binding = dict(type='ObservableSaveHazard', candidate=_plain_component(candidate.candidate))
        _check_receipts(vars(candidate.candidate))
    else:
        candidate_binding = _plain_component(candidate)
        _check_receipts(vars(candidate))
    static_oracle = {key: getattr(oracle, key) for key in ORACLE_STATIC
                     if key not in ('controller', 'next_owner', 'initial_prior', 'handoff')}
    static = dict(owners=sorted(scheduler.owners), oracle=static_oracle,
        controller=dict(configuration=configuration, identity=controller.identity, candidate=candidate_binding),
        mark=dict(type=type(mark).__module__ + '.' + type(mark).__qualname__, candidate=_plain_component(mark.candidate)),
        initial_prior=_plain_component(oracle.initial_prior), handoff=_plain_component(oracle.handoff))
    _check_receipts(configuration)
    _check_receipts(vars(mark.candidate))
    for component in (oracle.initial_prior, oracle.handoff):
        if component is not None:
            _check_receipts(vars(component))
    # Other frozen data are also byte-bound in their exact encoded static state.
    sources = implementation_sources(__file__)
    return dict(binding=copy.deepcopy(binding), static_state=encode(static), implementation_sources=sources)


def _state(scheduler, owner_last_ready):
    oracle, _, mark = _components(scheduler)
    return dict(scheduler={key: copy.deepcopy(getattr(scheduler, key)) for key in SCHEDULER_FIELDS},
        oracle={key: copy.deepcopy(getattr(oracle, key)) for key in ORACLE_FIELDS},
        mark=copy.deepcopy(dict(states=mark.states, clock=mark.clock)),
        owner_last_ready=copy.deepcopy(owner_last_ready))


def _validate(state, scheduler, binding):
    if type(state) is not dict or set(state) != {'scheduler', 'oracle', 'mark', 'owner_last_ready'}:
        raise ValueError('Complete independent scheduler state required')
    s, o, m = state['scheduler'], state['oracle'], state['mark']
    if set(s) != set(SCHEDULER_FIELDS) or set(o) != set(ORACLE_FIELDS) or set(m) != {'states', 'clock'}:
        raise ValueError('Missing or foreign checkpoint state fields')
    owners, clock = set(scheduler.owners), binding['clock_ns']
    ready = state['owner_last_ready']
    if (type(ready) is not dict or set(ready) != owners
            or any(type(t) is not int or t < binding['original_now_ns'] for t in ready.values())):
        raise ValueError('Complete explicit owner_last_ready required')
    if s['last_clock_ns'] != clock or o['lifecycle_clock_ns'] != clock or m['clock'] != clock:
        raise ValueError('Same-clock settled scheduler/oracle/mark required')
    if (type(o['lifecycle']) is not dict or set(o['lifecycle']) != owners
            or any(type(item) is not OwnerLifecycle for item in o['lifecycle'].values())
            or m['states'] != o['lifecycle']):
        raise ValueError('Complete consistent typed lifecycle required')
    if (s['held_owner'] not in owners | {None} or type(s['service_available']) is not bool
            or (s['restore_start_ns'] is not None and
                (type(s['restore_start_ns']) is not int or s['restore_start_ns'] > clock))
            or s['service_available'] and (s['held_owner'] is None or s['restore_start_ns'] is None)):
        raise ValueError('Invalid simulated held context')
    if (type(s['context_events']) is not list or any(type(t) is not int or t <= clock
            or kind not in ('SAVE_END', 'RESTORE_START', 'SERVICE_READY') or owner not in owners
            for t, kind, owner in s['context_events'])
            or s['context_events'] != sorted(s['context_events'], key=lambda row: row[0])):
        raise ValueError('Absolute unconsumed context events required')
    for key, at in s['pending_since'].items():
        if type(key) is not tuple or len(key) != 3 or key[0] not in owners or type(at) is not int or at > clock:
            raise ValueError('Invalid pending GPU node age')
    action = s['pending_switch']
    if action is None and s['context_events']:
        raise ValueError('Initial pre-service context events are not yet supported by this component')
    if action is not None:
        if (type(action) is not ContextAvailabilityDecision or action.kind != 'switch'
                or action.target_owner not in owners or not s['queries']
                or s['queries'][-1]['decision'] != action.__dict__ or o['restore_key'] is None):
            raise ValueError('Pending handoff and accepted decision disagree')
        ticks = [action.save_end_ns, action.restore_start_ns, action.service_ready_ns, action.next_review_ns]
        if any(type(t) is not int for t in ticks) or ticks != sorted(ticks) or ticks[-2] == ticks[-1]:
            raise ValueError('Ordered absolute pending handoff required')
        all_events = [(ticks[0], 'SAVE_END', o['restore_key'][0]),
                      (ticks[1], 'RESTORE_START', action.target_owner),
                      (ticks[2], 'SERVICE_READY', action.target_owner)]
        if s['context_events'] != [row for row in all_events if row[0] > clock]:
            raise ValueError('Pending absolute handoff events disagree')
    hazard = o['hazard']
    if hazard is not None:
        controller = scheduler.oracle.controller
        sealed = hazard['state']
        body = {key: value for key, value in sealed.items() if key != 'state_sha256'}
        if (set(sealed) != HAZARD_FIELDS or sealed['schema'] != HAZARD_SCHEMA
                or sealed['controller_identity'] != controller.identity
                or sealed['state_sha256'] != digest(body)
                or sealed['held_owner'] not in owners
                or any(type(sealed[k]) is not int for k in ('restore_ns', 'observation_ns', 'last_update_ns'))
                or not sealed['restore_ns'] <= sealed['observation_ns'] <= sealed['last_update_ns'] <= clock):
            raise ValueError('Invalid sealed cumulative SAVE state')
        expected_budget = (math.log(2.) if controller._policy == 'MEDIAN_LOG2' else
            -math.log((int(digest(dict(seed=controller._seed, restore_key=sealed['restore_key'],
                controller_identity=controller.identity))[:13], 16) + .5) / 2**52))
        if (sealed['hazard_budget'] != expected_budget or not 0 <= sealed['consumed_hazard'] <= expected_budget
                or not math.isfinite(sealed['summation_compensation'])
                or o['restore_key'] != (sealed['held_owner'], sealed['restore_ns'])
                or sealed['restore_key'] != str(o['restore_key'])):
            raise ValueError('SAVE origin, budget or accumulated history changed')
        if s['service_available'] and o['restore_key'] != (s['held_owner'], s['restore_start_ns']):
            raise ValueError('Active scheduler restore and SAVE origin disagree')
        for key in ('hazard_budget', 'consumed_hazard', 'next_review_ns', 'observation_ns'):
            if hazard[key] != sealed[key]:
                raise ValueError('Hazard wrapper and sealed state disagree')
        if (hazard['origin_restore_ns'] != sealed['restore_ns']
                or hazard['restore_age_ns'] != sealed['last_update_ns'] - sealed['restore_ns']
                or hazard['budget_redrawn'] is not False):
            raise ValueError('SAVE age cannot restart at cutoff')


def export_scheduler_snapshot(scheduler, *, binding, owner_last_ready):
    contract = _contract(scheduler, binding)
    state = _state(scheduler, owner_last_ready)
    _validate(state, scheduler, binding)
    result = dict(schema=SCHEMA, atomic_stage=STAGE, contract=contract, state=encode(state),
        main_engine_resume_supported=False, physical_progress_observed=False,
        native_observation_assimilation_supported=False)
    result['checkpoint_sha256'] = digest(result)
    return result


def import_scheduler_snapshot(scheduler, checkpoint, *, binding):
    """Pure validation/decode to detached state; never mutate destination."""
    expected_keys = {'schema', 'atomic_stage', 'contract', 'state', 'main_engine_resume_supported',
        'physical_progress_observed', 'native_observation_assimilation_supported', 'checkpoint_sha256'}
    if (type(checkpoint) is not dict or set(checkpoint) != expected_keys
            or checkpoint['schema'] != SCHEMA or checkpoint['atomic_stage'] != STAGE
            or any(checkpoint[key] is not False for key in ('main_engine_resume_supported',
                'physical_progress_observed', 'native_observation_assimilation_supported'))
            or checkpoint['checkpoint_sha256'] != digest({key: value for key, value in checkpoint.items()
                                                         if key != 'checkpoint_sha256'})):
        raise ValueError('Changed or foreign lifecycle component checkpoint')
    if checkpoint['contract'] != _contract(scheduler, binding):
        raise ValueError('Lifecycle checkpoint execution/model/source contract differs')
    state = decode(checkpoint['state'])
    try:
        _validate(state, scheduler, binding)
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('Malformed lifecycle component checkpoint state') from exc
    return state


def restore_scheduler_snapshot(scheduler, checkpoint, *, binding):
    """Transactionally replace supported dynamic state under caller ownership.

    Return the detached owner_last_ready map for a FUTURE main-engine adapter.
    No engine local state/queues/diagnostics/resources are restored here.
    """
    state = import_scheduler_snapshot(scheduler, checkpoint, binding=binding)
    oracle, _, mark = _components(scheduler)
    objects = [(scheduler, state['scheduler']), (oracle, state['oracle']), (mark, state['mark'])]
    staged = [(obj, {**vars(obj), **copy.deepcopy(update)}) for obj, update in objects]
    previous = [(obj, vars(obj).copy()) for obj, _ in staged]
    try:
        for obj, values in staged:
            obj.__dict__ = values
    except BaseException:
        for obj, values in previous:
            obj.__dict__ = values
        raise
    return copy.deepcopy(state['owner_last_ready'])
