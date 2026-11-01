"""Full declared-cohort binding and original-score gate projection.

This module does not fit, choose routes, or seal predictions. The supplied
engine must execute all declared requests, including the entire future suffix.
Startup energy and the memory safety factor enter the scorer exactly once.
"""
from __future__ import annotations

import copy
import math

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.metrics import score_forecast
from .forecast_gate_v2 import INPUT_SCHEMA, _public_cohort
from .known_input_native_flow_v5 import validate_plan


def original_flow_defaults(plan, trained_defaults):
    """A task subset may narrow references; it cannot choose new normalizers."""
    result = {k:copy.deepcopy(trained_defaults[k]) for k in ('aliases','memory_budget_mib',
        'energy_reference_watts','memory_safe_factor')}
    tasks = plan['workload_rps']
    if not set(tasks) <= set(trained_defaults['latency_reference_ms']):
        raise ValueError('Task references must exist in original trained objective')
    result['latency_reference_ms'] = {t:trained_defaults['latency_reference_ms'][t] for t in tasks}
    for key,value in result.items():
        if plan[key] != value:
            raise ValueError('Original trained flow normalizer differs: '+key)
    result['route_initial_with_continuation_policy'] = False
    return result


def bind_public(plan, candidate_id, public, routes, defaults, weights):
    """Return the exact executable FIFO cohort, never re-route by weights."""
    validate_plan(plan)
    _, cohort = _public_cohort(dict(schema=INPUT_SCHEMA, forecast_inputs=public))
    candidates = [c for c in plan['candidates'] if c['id'] == candidate_id]
    if len(candidates) != 1:
        raise ValueError('Exactly one declared flow candidate required')
    candidate = candidates[0]
    if routes != candidate['routes'] or set(routes) != set(cohort):
        raise ValueError('Exact complete frozen candidate routes required')
    start = public['phase_start_ns']
    rows = [dict(request_id=r['request_id'], task=r['task'], pool_id=r['pool_id'],
                 arrival_ns=start+r['offset_ns'], alias=routes[r['request_id']])
            for r in plan['trace']]
    if [r['arrival_ns'] for r in rows] != sorted(r['arrival_ns'] for r in rows):
        raise ValueError('Declared trace must preserve chronological FIFO order')
    for row in rows:
        actual = cohort[row['request_id']]
        expected = row if 'alias' in actual else {k:v for k,v in row.items() if k != 'alias'}
        if actual != expected:
            raise ValueError('Public input, arrival or initial route differs from declared flow')
    if set(public['queues']) != set(plan['aliases']):
        raise ValueError('All resident initial queues must be explicit')
    for alias, queue in public['queues'].items():
        expected = [r for r in rows if r['alias'] == alias and r['arrival_ns'] == start]
        if queue != expected:
            raise ValueError('Phase-start per-owner FIFO differs from declared trace')
    future = [{k:r[k] for k in ('request_id','task','pool_id','offset_ns')}
              for r in plan['trace'] if r['offset_ns'] > 0]
    workload = public['future_workload']
    pools = {t: sorted({r['pool_id'] for r in rows if r['task'] == t}) for t in plan['workload_rps']}
    if (set(workload) != {'rates','input_pools','process','declared_arrivals'}
            or workload['process'] != 'declared_schedule'
            or workload['declared_arrivals'] != future
            or workload['rates'] != plan['workload_rps'] or workload['input_pools'] != pools
            or public['now_ns'] != start
            or public['deadline_ns'] != start + round(plan['window_seconds']*1e9)
            or type(public['drain_deadline_ns']) is not int
            or public['drain_deadline_ns'] != public['deadline_ns'] + round(plan['drain_seconds']*1e9)):
        raise ValueError('Complete declared suffix, rates and original windows required')
    for field in ('aliases','latency_reference_ms','memory_budget_mib','energy_reference_watts','memory_safe_factor'):
        if defaults[field] != plan[field]:
            raise ValueError('Original flow objective/defaults differ: '+field)
    if (weights != {k:1/6 for k in 'QTLPME'} or weights != plan['weights']
            or plan['startup_amortization_seconds'] != 3600.
            or plan['warmup_seconds'] != 0.
            or defaults.get('route_initial_with_continuation_policy') is not False):
        raise ValueError('Original equal weights, fixed routes and startup contract required')
    return copy.deepcopy(rows)


def score_complete_forecast(forecast, rows, plan, quality, startup, *, phase_start_ns):
    """Score actual whole-horizon engine output with full startup accounting."""
    expected = {r['request_id']:r for r in rows}
    if forecast.get('cohort') != expected or set(forecast.get('finishes',{})) != set(expected):
        raise UnknownEvidence('COMPLETE_DECLARED_FLOW_DAG_AND_PARENT_RECEIPTS_REQUIRED')
    for rid, finish in forecast['finishes'].items():
        row = expected[rid]
        if any(finish.get(k) != v for k,v in row.items()):
            raise ValueError('Completion differs from exact declared public identity/route')
        times = [row['arrival_ns'],finish.get('start_ns'),finish.get('handler_end_ns'),
                 finish.get('finish_ns')]
        if any(type(t) is not int for t in times) or times != sorted(times):
            raise ValueError('Exact arrival/START/handler END/parent receipt chronology required')
        drain_end = phase_start_ns + round(plan['window_seconds']*1e9) + round(plan['drain_seconds']*1e9)
        if finish['finish_ns'] > drain_end:
            raise UnknownEvidence('PARENT_RECEIPT_OUTSIDE_FROZEN_DRAIN_BOUNDARY')
    for field in ('energy_joules','peak_memory_mib'):
        value = startup.get(field)
        if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
            raise UnknownEvidence('FINITE_VERIFIED_STARTUP_RESOURCE_REQUIRED:'+field)
    if (startup.get('energy_includes_priming') is not True
            or startup.get('safety_factor_applied') is not False
            or startup.get('startup_amortization_seconds') != 3600.
            or startup.get('warmup_seconds') != 0.):
        raise ValueError('Full unscaled startup/priming hypothesis required')
    horizon = plan['window_seconds']
    amortized = startup['energy_joules'] * horizon / 3600.
    metrics = score_forecast(forecast, [], plan['aliases'], plan['workload_rps'],
        plan['latency_reference_ms'], quality, phase_start_ns=phase_start_ns,
        now_ns=phase_start_ns, deadline_ns=phase_start_ns+round(horizon*1e9),
        memory_budget_mib=plan['memory_budget_mib'], energy_reference_watts=plan['energy_reference_watts'],
        memory_safe_factor=plan['memory_safe_factor'], weights=plan['weights'],
        observed_peak_memory_mib=startup['peak_memory_mib'], observed_board_energy_joules=amortized)
    if metrics['full_U'] is None or not math.isfinite(metrics['full_U']):
        raise UnknownEvidence('FINITE_COMPLETE_ORIGINAL_SIX_TERM_FORECAST_REQUIRED')
    metrics.update(startup_accounting='included_exactly_once',
        startup_energy_amortized_to_window_joules=amortized,
        E_scope='measurement board energy plus startup/priming amortized over 3600 seconds; drain excluded',
        M_scope='one safety factor times startup-through-drain owned sampled peak',
        physical_six_term_improvement_certified=False)
    return metrics


def gate_cell(result, rows, candidate, run_id, queue_offsets_ns, *, phase_start_ns):
    """Convert scored results without the old adapter's second startup charge."""
    forecast = result.get('forecast',{})
    finishes = forecast.get('finishes',{})
    expected = {r['request_id']:r for r in rows}
    if finishes and (forecast.get('cohort') != expected or set(finishes)-set(expected)):
        raise ValueError('Gate projection cannot change the complete declared cohort')
    requests = []
    for row in rows:
        finish = finishes.get(row['request_id'],{})
        start, end, receipt = (finish.get(k) for k in ('start_ns','handler_end_ns','finish_ns'))
        if all(type(t) is int for t in (start,end,receipt)):
            if not row['arrival_ns'] <= start <= end <= receipt:
                raise ValueError('Invalid predicted handler/receipt chronology')
            service, queue, e2e = (end-start)/1e6,(start-row['arrival_ns'])/1e6,(receipt-row['arrival_ns'])/1e6
        else:
            service = queue = e2e = None
        requests.append({k:row[k] for k in ('request_id','task','pool_id')} |
                        dict(service_ms=service,queue_ms=queue,e2e_ms=e2e))
    metrics = result.get('metrics')
    terms = None
    if metrics is not None:
        if metrics.get('startup_accounting') != 'included_exactly_once':
            raise ValueError('Explicit single startup accounting required for gate projection')
        terms = copy.deepcopy(metrics['terms'])
        if set(terms) != set('QTLPME'):
            raise ValueError('Exactly original six utility terms required')
    samples = []
    for offset in queue_offsets_ns:
        if type(offset) is not int or offset < 0:
            raise ValueError('Nonnegative integer queue offsets required')
        clock = phase_start_ns+offset; depth = 0; known = True
        for row in rows:
            if row['arrival_ns'] > clock:
                continue
            start = finishes.get(row['request_id'],{}).get('start_ns')
            if type(start) is not int:
                known = False
            else:
                depth += clock < start
        samples.append(dict(offset_ns=offset,depth=depth if known else None))
    return dict(run_id=run_id,candidate_id=candidate['id'],
        execution_context_sha256=candidate['execution_context_sha256'],
        requests=requests,terms=terms,queue_depth_samples=samples)
