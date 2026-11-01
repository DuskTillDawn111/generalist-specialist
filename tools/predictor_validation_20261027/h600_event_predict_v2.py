"""CPU prediction helper with explicit selected-completion-event v2 numerics.

Same prediction schema and frozen metric/provider behavior as core.predict;
new numerical source receipts are explicit and no module is monkeypatched.
"""
from __future__ import annotations
from gsdeploy.forecast_errors import UnknownEvidence
from .h600_event_simulator_v2 import FullHorizonModel
from .flow_validation import _provider,_resources,_score,_queue_samples
from .native_flow import receipt


def predict(plan, candidate, model, resource_document, quality, *, provider=None):
    """One full-cohort point rollout. Missing evidence remains explicit."""
    end = round(plan['window_seconds'] * 1e9)
    queues = {a: [] for a in plan['aliases']}
    future = []
    for item in plan['trace']:
        row = {k: item[k] for k in ('request_id', 'task', 'pool_id')}
        row['arrival_ns'] = item['offset_ns']
        (queues[candidate['routes'][row['request_id']]] if not row['arrival_ns'] else future).append(row)
    reason, forecast, metric = None, None, None
    try:
        provider = provider or _provider(model, candidate['execution_context_sha256'])
        resource = _resources(resource_document, candidate['execution_context_sha256'])
        forecast = FullHorizonModel(plan['aliases'], provider, resource, memory_provider=resource,
            resources=candidate.get('resource_configuration'),
            max_requests=plan.get('max_forecast_requests', 20000)).forecast(
                queues, 0, end, future_arrivals=future,
                route_future=lambda row, pending, now: candidate['routes'][row['request_id']],
                require_energy=False, require_memory=False,
                drain_deadline_ns=end + round(plan['drain_seconds'] * 1e9))
        metric = _score(plan, forecast, quality,
            startup_joules=candidate.get('predicted_startup_joules'),
            historical_peak=candidate.get('predicted_prephase_peak_mib'))
    except (UnknownEvidence, ValueError) as exc:
        reason = type(exc).__name__ + ':' + str(exc)
    requests = []
    for item in plan['trace']:
        finish = forecast['finishes'].get(item['request_id']) if forecast else None
        requests.append({'request_id': item['request_id'], 'task': item['task'], 'pool_id': item['pool_id'],
            'alias': candidate['routes'][item['request_id']],
            'service_ms': (finish['finish_ns'] - finish['start_ns']) / 1e6 if finish else None,
            'wait_ms': (finish['start_ns'] - item['offset_ns']) / 1e6 if finish else None,
            'e2e_ms': (finish['finish_ns'] - item['offset_ns']) / 1e6 if finish else None,
            'finish_ns': finish['finish_ns'] if finish else None})
    comparison = (-metric['cost'] if metric and metric['cost'] is not None and
        (metric['absolute_utility_complete'] or plan.get('startup_comparison_policy')) else None)
    return {'candidate_id': candidate['id'], 'requests': requests, 'metrics': metric,
            'comparison_U': comparison, 'queue_samples': _queue_samples(plan, forecast) if forecast else [],
            'unknown_reason': reason, 'numerics_source': receipt(FullHorizonModel.forecast.__code__.co_filename),
            'forecast_numerics': 'selected_completion_event_v2', 'original_event_budget': 100000}

