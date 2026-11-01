"""JSON-only simulated DAG checkpoints; never observed CUDA progress.

Providers must be deterministic, with their complete immutable model evidence
declared by the caller. Receipts verify that declaration, not its completeness.
Only the engine's natural atomic event boundaries are checkpointable.
"""
from __future__ import annotations

from collections import Counter, deque
import functools
import hashlib
import inspect
import json
import math
from pathlib import Path
import types


SCHEMA = 'gsdeploy.simulated_dag_atomic_checkpoint.v7'
STATE_FIELDS = (
    'arrivals cohort pending running active ready finish_events wake_events '
    'unready finishes starts queue_wait sequence clock event_count maximum_parallel '
    'maximum_owner_parallel all_completed_nodes max_waiting receipt_pending '
    'release_hypotheses energy peak energy_known memory_known resource_transitions'
).split()
DIAGNOSTIC_EXCLUDED = {'aliases', 'plans', 'duration_provider', 'resource_demand_provider'}


def encode(value):
    if value is None or type(value) in (bool, int, str):
        return ['scalar', value]
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('Checkpoint cannot contain nonfinite numbers')
        return ['scalar', value]
    if type(value) in (dict, Counter):
        return ['counter' if type(value) is Counter else 'dict',
                [[encode(k), encode(v)] for k, v in value.items()]]
    if type(value) in (list, tuple, set, deque):
        name = {list: 'list', tuple: 'tuple', set: 'set', deque: 'deque'}[type(value)]
        rows = [encode(v) for v in value]
        if type(value) is set:
            rows.sort(key=lambda v: json.dumps(v, sort_keys=True))
        return [name, rows]
    raise ValueError('Unsupported checkpoint value: ' + type(value).__name__)


def decode(value):
    if type(value) is not list or len(value) != 2:
        raise ValueError('Explicit typed checkpoint value required')
    kind, body = value
    if kind == 'scalar':
        if body is not None and type(body) not in (bool, int, float, str):
            raise ValueError('Invalid checkpoint scalar')
        if type(body) is float and not math.isfinite(body):
            raise ValueError('Nonfinite checkpoint scalar')
        return body
    if type(body) is not list:
        raise ValueError('Checkpoint container must contain a list')
    if kind in ('dict', 'counter'):
        rows = [(decode(k), decode(v)) for k, v in body]
        result = dict(rows)
        if len(result) != len(rows):
            raise ValueError('Duplicate checkpoint mapping key')
        return Counter(result) if kind == 'counter' else result
    constructors = dict(list=list, tuple=tuple, set=set, deque=deque)
    if kind not in constructors:
        raise ValueError('Unknown checkpoint container kind')
    rows = [decode(v) for v in body]
    result = constructors[kind](rows)
    if kind == 'set' and len(result) != len(rows):
        raise ValueError('Duplicate checkpoint set value')
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def receipt(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return dict(path=str(path), sha256=h.hexdigest())


def code_identity(code):
    """Serialize code values, independently of marshal's object-reference tags.

    CPython may intern constant strings during execution. Marshal records that
    representation and reference sharing, so its byte stream is not a stable
    semantic fingerprint even while the code and source remain unchanged.
    ``co_code`` exposes the original instructions, not adaptive interpreter
    caches. Bind all execution metadata and recursively type-tag constants.
    """
    def constant(value):
        if value is None or type(value) in (bool, int, str):
            return [type(value).__name__, value]
        if type(value) is float:
            return ['float', value.hex()]
        if type(value) is complex:
            return ['complex', value.real.hex(), value.imag.hex()]
        if type(value) is bytes:
            return ['bytes', value.hex()]
        if value is Ellipsis:
            return ['ellipsis']
        if type(value) is types.CodeType:
            return ['code', code_identity(value)]
        if type(value) in (tuple, frozenset):
            values = [constant(v) for v in value]
            if type(value) is frozenset:
                values.sort(key=lambda v: json.dumps(v, sort_keys=True))
            return [type(value).__name__, values]
        raise ValueError('Unsupported checkpoint code constant: ' + type(value).__name__)
    names = ('co_argcount', 'co_posonlyargcount', 'co_kwonlyargcount', 'co_nlocals',
             'co_stacksize', 'co_flags', 'co_code', 'co_consts', 'co_names',
             'co_varnames', 'co_freevars', 'co_cellvars', 'co_filename', 'co_name',
             'co_qualname', 'co_firstlineno', 'co_linetable', 'co_lnotab',
             'co_exceptiontable')
    return {name: constant(getattr(code, name, None)) for name in names}


def callable_identity(provider):
    """Bind selected implementation and configuration, including nested calls.

    Opaque mutable objects must expose a JSON-encodable static
    ``checkpoint_configuration()``. This describes static model selection;
    stateful/stochastic service providers are outside the continuation contract.
    """
    if provider is None:
        return None
    if isinstance(provider, functools.partial):
        return dict(kind='partial', function=callable_identity(provider.func),
                    args=encode(provider.args), keywords=encode(provider.keywords))
    if inspect.ismethod(provider):
        obj = provider.__self__
        implementation = callable_identity(provider.__func__)
        if type(obj).__module__ == 'tools.predictor_validation_20261027.dag_queue_diagnostics_v6' and type(obj).__name__ == 'DagQueueDiagnostics':
            configuration = dict(kind='queue_diagnostic_bound_method',
                duration_provider=callable_identity(obj.duration_provider),
                resource_demand_provider=callable_identity(obj.resource_demand_provider))
        else:
            configuration = object_configuration(obj)
        return dict(kind='bound_method', implementation=implementation, configuration=configuration)
    if inspect.isfunction(provider):
        source = inspect.getsourcefile(provider)
        if source is None:
            raise ValueError('Checkpoint function source must be identifiable')
        return dict(kind='function', module=provider.__module__, qualname=provider.__qualname__,
                    source=receipt(source), first_line=provider.__code__.co_firstlineno,
                    code_sha256=digest(code_identity(provider.__code__)),
                    defaults=encode(provider.__defaults__), keyword_defaults=encode(provider.__kwdefaults__),
                    closure=encode(tuple(c.cell_contents for c in provider.__closure__)) if provider.__closure__ else None)
    if not callable(provider):
        raise ValueError('Explicit callable provider required')
    return callable_identity(provider.__call__)


def object_configuration(obj):
    cls = type(obj)
    if hasattr(obj, 'checkpoint_configuration'):
        values = obj.checkpoint_configuration()
    elif hasattr(obj, '__dict__') and not any('__slots__' in c.__dict__ for c in cls.__mro__):
        values = vars(obj)
    else:
        raise ValueError('Opaque checkpoint provider needs explicit checkpoint_configuration')
    return dict(module=cls.__module__, qualname=cls.__qualname__, configuration=encode(values))


def contract(engine, requests, now_ns, deadline_ns, drain_deadline_ns):
    binding = engine.checkpoint_binding
    if (not isinstance(binding, dict) or set(binding) != {'execution_context_sha256', 'evidence'}
            or not isinstance(binding['execution_context_sha256'], str)
            or len(binding['execution_context_sha256']) != 64 or not binding['evidence']):
        raise ValueError('Explicit execution identity and complete provider evidence required for checkpoint')
    for ref in binding['evidence']:
        if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'} or receipt(ref['path']) != ref:
            raise ValueError('Checkpoint provider evidence changed')
    sources = {str(Path(__file__).resolve()), inspect.getsourcefile(type(engine))}
    for obj in (engine.duration_provider, engine.release_provider, engine.lifecycle_provider,
                engine.power_provider, engine.memory_provider, engine.queue_diagnostics):
        if obj is None:
            continue
        target = obj if inspect.isfunction(obj) or inspect.ismethod(obj) else type(obj)
        source = inspect.getsourcefile(target)
        if source is None:
            raise ValueError('Checkpoint provider implementation source required')
        sources.add(source)
    options = {k: getattr(engine, k) for k in (
        'allow_profiled_replay', 'allow_independent_copy_hypothesis',
        'allow_context_time_slice_hypothesis', 'allow_no_implicit_default_stream_hypothesis',
        'scenario', 'max_events', 'require_energy', 'require_memory')}
    providers = dict(duration=callable_identity(engine.duration_provider),
        release=callable_identity(engine.release_provider),
        lifecycle=callable_identity(engine.lifecycle_provider.delays) if engine.lifecycle_provider else None,
        power=callable_identity(engine.power_provider.power) if engine.power_provider else None,
        memory=callable_identity(engine.memory_provider.memory) if engine.memory_provider else None,
        diagnostics=callable_identity(engine.queue_diagnostics.duration) if engine.queue_diagnostics else None)
    return dict(binding=binding, sources=[receipt(p) for p in sorted(sources)], providers=providers,
                aliases_sha256=digest(encode(engine.aliases)), plans_sha256=digest(encode(engine.plans)),
                cohort_sha256=digest(encode(requests)), options=options,
                original_now_ns=now_ns, deadline_ns=deadline_ns, drain_deadline_ns=drain_deadline_ns,
                provider_state_contract='Deterministic providers with caller-declared complete immutable evidence; no hidden mutable execution state')


def export_checkpoint(state, diagnostic, binding):
    if set(state) != set(STATE_FIELDS):
        raise ValueError('Complete engine state required')
    diag = None if diagnostic is None else {
        k: v for k, v in diagnostic.__dict__.items() if k not in DIAGNOSTIC_EXCLUDED}
    result = dict(schema=SCHEMA, kind='SIMULATED_PREDICTOR_STATE_NOT_OBSERVED_GPU_CURSOR',
                  atomic_boundary='After interval consumption/node completion; before next event dispatch',
                  contract=binding, state=encode(state), diagnostics=encode(diag),
                  visible_feedback_applied=False, physical_progress_observed=False)
    result['checkpoint_sha256'] = digest(result)
    return result


def import_checkpoint(checkpoint, engine, binding):
    fields = {'schema', 'kind', 'atomic_boundary', 'contract', 'state', 'diagnostics',
              'visible_feedback_applied', 'physical_progress_observed', 'checkpoint_sha256'}
    if (set(checkpoint) != fields or checkpoint['schema'] != SCHEMA
            or checkpoint['checkpoint_sha256'] != digest({k:v for k,v in checkpoint.items() if k != 'checkpoint_sha256'})
            or checkpoint['contract'] != binding
            or checkpoint['kind'] != 'SIMULATED_PREDICTOR_STATE_NOT_OBSERVED_GPU_CURSOR'
            or checkpoint['visible_feedback_applied'] is not False
            or checkpoint['physical_progress_observed'] is not False):
        raise ValueError('Changed/foreign simulated checkpoint or original execution window')
    state = decode(checkpoint['state'])
    if type(state) is not dict or set(state) != set(STATE_FIELDS):
        raise ValueError('Complete checkpoint engine state required')
    validate_state(state, engine, binding)
    diagnostic = decode(checkpoint['diagnostics'])
    if engine.queue_diagnostics is None:
        if diagnostic is not None:
            raise ValueError('Checkpoint diagnostics must remain connected')
    else:
        expected = set(engine.queue_diagnostics.__dict__) - DIAGNOSTIC_EXCLUDED
        if (type(diagnostic) is not dict or set(diagnostic) != expected
                or diagnostic['now_ns'] != state['clock']
                or diagnostic['waiting'] != {r['request_id'] for rows in state['pending'].values() for r in rows}):
            raise ValueError('Checkpoint queue diagnostic state differs from engine')
        validate_diagnostic(diagnostic, state, engine.queue_diagnostics)
        for k,v in diagnostic.items():
            setattr(engine.queue_diagnostics, k, v)
    return state


def validate_diagnostic(saved, state, diagnostic):
    """Rederive cached service and reconcile remaining work with DAG progress."""
    from .dag_queue_diagnostics_v6 import DagQueueDiagnostics
    if type(diagnostic) is not DagQueueDiagnostics:
        raise ValueError('Checkpoint requires explicit supported DAG queue diagnostics')
    cohort = state['cohort']
    arrived = set(cohort) - {r['request_id'] for _,_,r in state['arrivals']}
    if saved['registered'] != arrived or saved['initial_ns'] != diagnostic.initial_ns:
        raise ValueError('Checkpoint registered diagnostic cohort differs from arrivals')
    fresh = DagQueueDiagnostics(diagnostic.aliases, diagnostic.plans, diagnostic.duration_provider,
        now_ns=diagnostic.initial_ns, resource_demand_provider=diagnostic.resource_demand_provider)
    for key,profile in saved['cache'].items():
        row = next((r for r in cohort.values() if fresh._key(r) == key), None)
        if row is None or fresh._profile(row) != profile:
            raise ValueError('Checkpoint cached activity duration/resource profile differs from provider')
    expected_costs = {}
    remaining = {kind:0. for kind in ('gpu','copy','memset')}
    for rid in arrived:
        row = cohort[rid]
        profile = fresh._profile(row)
        expected_costs[rid] = sum(profile['full_work'].values()) / 1e6
        if rid in state['starts']:
            running = state['running'].get(row['alias'])
            if running is None or running['row']['request_id'] != rid:
                continue
            for node,full in profile['durations'].items():
                if node in running['done']:
                    continue
                value = state['active'].get((row['alias'],node),{}).get('remaining_ns',full)
                remaining[profile['kinds'][node]] += value
        else:
            for kind,value in profile['full_work'].items():
                remaining[kind] += value
    if saved['costs'] != expected_costs or any(not math.isclose(saved['remaining'][k],v,rel_tol=1e-9,abs_tol=1e-3) for k,v in remaining.items()):
        raise ValueError('Checkpoint diagnostic costs/remaining work disagree with executable DAG state')
    areas = dict.fromkeys(diagnostic.aliases,0)
    for rid in arrived:
        row = cohort[rid]
        areas[row['alias']] += state['starts'].get(rid,state['clock'])-row['arrival_ns']
    if saved['fifo_request_wait_ns'] != areas:
        raise ValueError('Checkpoint diagnostic FIFO wait integral disagrees with request boundaries')


def validate_state(s, engine, binding):
    clock = s['clock']
    if (type(clock) is not int or clock < binding['original_now_ns']
            or binding['drain_deadline_ns'] is not None and clock > binding['drain_deadline_ns']
            or type(s['event_count']) is not int or not 0 <= s['event_count'] <= engine.max_events
            or set(s['pending']) != set(engine.aliases)):
        raise ValueError('Invalid checkpoint time, owner domain or consumed event budget')
    cohort = s['cohort']
    if (digest(encode(list(cohort.values()))) != binding['cohort_sha256']
            or any(rid != row.get('request_id') for rid,row in cohort.items())):
        raise ValueError('Checkpoint cohort key differs from original request identity')
    locations = ([r for _,_,r in s['arrivals']]
                 + [r for rows in s['pending'].values() for r in rows]
                 + [v['row'] for v in s['running'].values()]
                 + list(s['receipt_pending'].values()) + list(s['finishes'].values()))
    ids = [r['request_id'] for r in locations]
    if len(ids) != len(set(ids)) or set(ids) != set(cohort):
        raise ValueError('Checkpoint loses, duplicates or revives a request')
    for row in locations:
        original = cohort[row['request_id']]
        if any(row[k] != original[k] for k in ('request_id','alias','task','pool_id','arrival_ns')):
            raise ValueError('Checkpoint changed request route/input/arrival')
    for owner, rows in s['pending'].items():
        if any(r['alias'] != owner or r['arrival_ns'] > clock for r in rows):
            raise ValueError('Checkpoint pending request is routed to a foreign/future owner')
        order = {rid:i for i,rid in enumerate(cohort)}
        if any((a['arrival_ns'],order[a['request_id']]) > (b['arrival_ns'],order[b['request_id']]) for a,b in zip(rows,list(rows)[1:])):
            raise ValueError('Checkpoint changed owner FIFO arrival order')
    input_order = {rid:i for i,rid in enumerate(cohort)}
    for at,index,row in s['arrivals']:
        if (at != row['arrival_ns'] or index != input_order[row['request_id']] or at < clock):
            raise ValueError('Checkpoint arrival heap differs from public arrival/order')
    for owner, running in s['running'].items():
        row = running['row']
        plan = engine.plans[owner,row['task'],row['pool_id']]
        nodes = {n['id']:n for n in plan['nodes']}
        if owner != row['alias'] or running['nodes'] != nodes or not running['done'] <= set(nodes):
            raise ValueError('Checkpoint running DAG differs from original plan')
        expected = {n:0 for n in nodes}
        successors = {n:[] for n in nodes}
        for edge in plan['edges']:
            successors[edge['before']].append(edge['after'])
            if edge['before'] not in running['done']:
                expected[edge['after']] += 1
            if edge['after'] in running['done'] and edge['before'] not in running['done']:
                raise ValueError('Checkpoint completed nodes violate dependency closure')
        if running['incoming'] != expected or running['successors'] != successors:
            raise ValueError('Checkpoint changed DAG dependency progress')
    ready_keys = [(owner,node) for _,_,owner,node in s['ready']]
    if len(set(ready_keys)) != len(ready_keys) or set(ready_keys) & set(s['active']):
        raise ValueError('Checkpoint duplicates ready/active work')
    for owner,node in ready_keys + list(s['active']):
        running = s['running'].get(owner)
        if (running is None or node not in running['nodes'] or node in running['done']
                or running['incoming'][node] != 0):
            raise ValueError('Checkpoint revives completed or blocked DAG work')
    for key, node in s['active'].items():
        if (node['kind'] not in ('gpu','copy','memset') or type(node['remaining_ns']) not in (int,float)
                or not math.isfinite(node['remaining_ns']) or node['remaining_ns'] <= 0):
            raise ValueError('Checkpoint active work must have positive remaining service')
        if node['kind'] != s['running'][key[0]]['nodes'][key[1]]['kind']:
            raise ValueError('Checkpoint changed activity resource kind')
    ready_set = set(ready_keys)
    for owner,running in s['running'].items():
        for node in running['nodes']:
            executable = node not in running['done'] and running['incoming'][node] == 0
            present = (owner,node) in ready_set or (owner,node) in s['active']
            if executable != present:
                raise ValueError('Checkpoint loses or invents executable DAG work')
        if running['row']['request_id'] not in s['starts'] or running['start_ns'] != s['starts'][running['row']['request_id']]:
            raise ValueError('Checkpoint running START differs from original START accounting')
    for field in ('arrivals','ready','finish_events','wake_events'):
        heap = s[field]
        if any(heap[(i-1)//2] > heap[i] for i in range(1,len(heap))):
            raise ValueError('Checkpoint changed event heap ordering')
    if set(s['running']) & set(s['unready']):
        raise ValueError('Checkpoint owner cannot be running and awaiting readiness')
    if sorted((rid,at) for at,_,rid in s['finish_events']) != sorted(
            (rid,row['finish_ns']) for rid,row in s['receipt_pending'].items()):
        raise ValueError('Checkpoint receipt events differ from pending receipts')
    if sorted((owner,at) for at,owner in s['wake_events']) != sorted(s['unready'].items()):
        raise ValueError('Checkpoint worker-ready events differ from unavailable owners')
    if any(at < clock for field in ('finish_events','wake_events') for at,*_ in s[field]):
        raise ValueError('Checkpoint has an overdue undispatched receipt/ready event')
    if any(row['finish_ns'] > clock for row in s['finishes'].values()):
        raise ValueError('Checkpoint invents a future received result')
    started_ids = ({v['row']['request_id'] for v in s['running'].values()}
                   | set(s['receipt_pending']) | set(s['finishes']))
    if set(s['starts']) != started_ids:
        raise ValueError('Checkpoint START registry differs from lifecycle state')
    expected_completed = sum(len(v['done']) for v in s['running'].values())
    for row in list(s['receipt_pending'].values()) + list(s['finishes'].values()):
        expected_completed += len(engine.plans[row['alias'],row['task'],row['pool_id']]['nodes'])
        if (row['request_id'] not in s['starts'] or row['start_ns'] != s['starts'][row['request_id']]
                or not row['start_ns'] <= row['handler_end_ns'] <= row['finish_ns']
                or row['worker_next_ready_ns'] < row['handler_end_ns']):
            raise ValueError('Checkpoint completed lifecycle accounting changed')
    if s['all_completed_nodes'] != expected_completed:
        raise ValueError('Checkpoint completed node count differs from DAG progress')
    if (set(s['starts']) != set(s['queue_wait']) or any(
        type(at) is not int or not cohort[rid]['arrival_ns'] <= at <= clock
        or s['queue_wait'][rid] != at-cohort[rid]['arrival_ns'] for rid,at in s['starts'].items())):
        raise ValueError('Checkpoint START/wait accounting changed')
    for field in ('energy','peak'):
        if type(s[field]) not in (int,float) or not math.isfinite(s[field]) or s[field] < 0:
            raise ValueError('Invalid checkpoint accumulated resource accounting')
    validate_resources(s, engine, binding)


def validate_resources(s, engine, binding):
    """Reconcile saved resource history with handler facts and fresh providers.

    Checksums detect byte changes; they cannot by themselves establish that a
    saved cumulative value agrees with the state being resumed. Drain energy
    remains excluded and memory remains an unapplied sampled-peak hypothesis.
    """
    ended = {r['request_id']: r['handler_end_ns']
             for r in list(s['finishes'].values()) + list(s['receipt_pending'].values())}
    edges = []
    for rid, start in s['starts'].items():
        end = ended.get(rid)
        if end == start:
            continue
        row = s['cohort'][rid]
        edges.append((start, 1, rid, row))
        if end is not None:
            edges.append((end, 0, rid, row))
    edges.sort(key=lambda edge: edge[:3])
    cursor = 0
    handlers = {}
    energy = peak = 0.
    energy_known = engine.power_provider is not None
    memory_known = engine.memory_provider is not None
    clock = binding['original_now_ns']
    deadline = binding['deadline_ns']
    cache = {}
    for transition in s['resource_transitions']:
        if (type(transition) is not dict or set(transition) != {
                'time_ns','until_ns','handler_active_aliases','board_watts','peak_memory_mib'}
                or type(transition['time_ns']) is not int
                or type(transition['until_ns']) is not int
                or transition['time_ns'] != clock
                or not clock < transition['until_ns'] <= s['clock']
                or clock < deadline < transition['until_ns']):
            raise ValueError('Checkpoint resource history has a gap or invalid measurement boundary')
        while cursor < len(edges) and edges[cursor][0] <= clock:
            _, kind, rid, row = edges[cursor]
            owner = row['alias']
            if kind == 0:
                if handlers.get(owner, {}).get('request_id') != rid:
                    raise ValueError('Checkpoint handler resource history overlaps or loses ownership')
                del handlers[owner]
            else:
                if owner in handlers:
                    raise ValueError('Checkpoint handler resource history overlaps ownership')
                handlers[owner] = row
            cursor += 1
        if cursor < len(edges) and edges[cursor][0] < transition['until_ns']:
            raise ValueError('Checkpoint resource interval crosses a handler boundary')
        if transition['handler_active_aliases'] != sorted(handlers):
            raise ValueError('Checkpoint resource state differs from START/handler END history')
        key = (clock < deadline, tuple(sorted((a, r['request_id']) for a, r in handlers.items())))
        if key not in cache:
            power = (engine.power_provider.power(handlers, engine.aliases, engine.scenario)
                     if clock < deadline and engine.power_provider is not None else {})
            memory = (engine.memory_provider.memory(handlers, engine.aliases, engine.scenario)
                      if engine.memory_provider is not None else {})
            def value(result, name):
                number = result.get(name)
                return (number if result.get('known') is True
                        and type(number) in (int, float) and math.isfinite(number) and number >= 0 else None)
            cache[key] = value(power, 'watts'), value(memory, 'peak_memory_mib')
        watts, memory = cache[key]
        watts = watts if clock < deadline else None
        if (transition['board_watts'] != watts or transition['peak_memory_mib'] != memory):
            raise ValueError('Checkpoint resource values differ from bound providers')
        if clock < deadline:
            if watts is None:
                energy_known = False
            else:
                energy += watts * (min(transition['until_ns'], deadline) - clock) / 1e9
        if memory is None:
            memory_known = False
        else:
            peak = max(peak, memory)
        clock = transition['until_ns']
    if (clock != s['clock'] or s['energy'] != energy or s['peak'] != peak
            or type(s['energy_known']) is not bool or type(s['memory_known']) is not bool
            or s['energy_known'] != energy_known or s['memory_known'] != memory_known):
        raise ValueError('Checkpoint accumulated resource accounting disagrees with complete history')
