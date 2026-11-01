"""Retrospective training cross-validation, never independent GPU validation.

Each fold predicts the complete held-out continuous flow with a model fitted
without that flow. All future arrivals/routes are declared planning inputs;
actual future execution phases are used only on the observed side of errors.
"""
from __future__ import annotations
import argparse
import collections
import itertools
import json
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting import FullHorizonModel
from .flow_validation import _provider, _resources, _score, _queue_samples, observed
from .validation import errors
from .probe_validation import read, receipt, save
from .native_flow import checked


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
            'unknown_reason': reason}


def compare(plan, candidate, prediction, raw):
    actual = observed(plan, candidate, raw)
    by_id = {r['request_id']: r for r in actual['requests']}
    timing = {key: errors([(r[key + '_ms'], by_id[r['request_id']][key + '_ms'])
        for r in prediction['requests'] if r[key + '_ms'] is not None and
        by_id[r['request_id']][key + '_ms'] is not None]) for key in ('service', 'wait', 'e2e')}
    by_time = {r['offset_ns']: r['depth'] for r in actual['queue_samples']}
    queue_error = errors([(r['depth'], by_time[r['offset_ns']]) for r in prediction['queue_samples']])
    pm, om = prediction['metrics'], actual['metrics']
    p95 = {}
    for task in plan['workload_rps']:
        p = (pm or {}).get('per_task', {}).get(task, {}).get('p95_ms')
        a = om['per_task'].get(task, {}).get('p95_ms')
        p95[task] = {'predicted_ms': p, 'observed_ms': a,
                    'absolute_relative_error': abs(p - a) / a if p is not None and a is not None and a > 0 else None}
    return {'candidate_id': candidate['id'], 'requested': len(plan['trace']), 'known': timing['e2e']['count'],
        'timing_errors': timing, 'queue_error': queue_error, 'per_task_p95': p95,
        'prediction_metrics': pm, 'observed_metrics': om,
        'predicted_comparison_U': prediction['comparison_U'], 'observed_U': om['full_U'],
        'unknown_reason': prediction['unknown_reason'], 'eligible_for_gain_claim': False}


def ranking(rows, baseline_id, *, tie_tolerance=0.):
    """Per-block descriptive decision checks; incomplete pairs stay missing."""
    if len({r['candidate_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate candidates in a crossfit ranking block')
    sign = lambda v: 0 if abs(v) <= tie_tolerance else (1 if v > 0 else -1)
    values = {r['candidate_id']: (r['predicted_comparison_U'], r['observed_U']) for r in rows}
    base = values.get(baseline_id)
    paired, pairs = [], []
    for cid, val in values.items():
        if cid != baseline_id and base and all(x is not None for x in (*val, *base)):
            p, a = val[0] - base[0], val[1] - base[1]
            paired.append({'candidate_id': cid, 'predicted_delta_U': p, 'observed_delta_U': a,
                           'sign_correct': sign(p) == sign(a)})
    for a, b in itertools.combinations(sorted(values), 2):
        if all(x is not None for x in (*values[a], *values[b])):
            pairs.append({'a': a, 'b': b,
                'correct': sign(values[a][0] - values[b][0]) == sign(values[a][1] - values[b][1])})
    return {'paired_deltas': paired, 'candidate_pairs': pairs,
        'delta_sign_accuracy': sum(r['sign_correct'] for r in paired) / len(paired) if paired else None,
        'rank_accuracy': sum(r['correct'] for r in pairs) / len(pairs) if pairs else None,
        'known_delta_pairs': len(paired), 'known_rank_pairs': len(pairs),
        'independent_validation': False, 'final_gain_eligible': False}


def scored_heldout(raw_ref):
    """Read actual Q without accepting any modified native observation."""
    path = checked(raw_ref)
    raw = read(path)
    if 'quality_evidence' in raw:
        native_ref = raw.get('native_raw')
        if not native_ref:
            raise ValueError('Scored flow must bind its immutable native raw')
        native = read(checked(native_ref))
    else:
        scored_path = path.with_name('scored-raw.json')
        scored = read(scored_path)
        if scored.get('native_raw') != raw_ref:
            raise ValueError('Scored heldout flow binds a different native raw')
        native, raw = raw, scored
        path = scored_path
    if any(raw.get(k) != value for k, value in native.items()):
        raise ValueError('Scoring changed immutable native observations')
    if set(raw) != set(native) | {'quality_evidence', 'native_raw'}:
        raise ValueError('Only native raw binding and measured Q annotation allowed')
    for ref in raw['quality_evidence']['source_receipts']:
        checked(ref)
    return raw, receipt(path)


def run(manifest_path, resources_path, quality_path, output):
    from .flow_fit import verify_training_model
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = read(manifest_path)
    seed_cv = manifest.get('schema') == 'gsdeploy.continuous_seed_crossfit_manifest.v1'
    if (manifest.get('schema') not in ('gsdeploy.continuous_flow_crossfit_manifest.v1',
                                    'gsdeploy.continuous_seed_crossfit_manifest.v1') or
            manifest.get('retrospective_training_cross_validation') is not True or
            manifest.get('no_independent_validation') is not True):
        raise ValueError('Explicit retrospective training crossfit manifest required')
    resources, quality_doc = read(resources_path), read(quality_path)
    quality = quality_doc.get('loss_by_family_task', quality_doc)
    refs = [receipt(p) for p in (manifest_path, resources_path, quality_path, __file__)]
    refs.extend(manifest.get('source_receipts', []))
    refs.extend(quality_doc.get('source_receipts', []))
    refs.extend(resources.get('source_receipts', resources.get('provenance', {}).get('sources', [])))
    rows, by_seed = [], collections.defaultdict(list)
    cases = []
    for fold in manifest['folds']:
        if seed_cv:
            if fold.get('unit') != 'seed':
                raise ValueError('Complete held-out seed unit required')
            for heldout in fold['heldout_flows']:
                cases.append({**fold, **heldout, 'fold_id': fold['fold_id'] + '__' + heldout['candidate_id']})
        else:
            cases.append(fold)
    verified_models = {}
    for fold in cases:
        for name in ('model', 'heldout_raw', 'heldout_plan'):
            checked(fold[name]); refs.append(fold[name])
        model_key = fold['model']['sha256']
        if model_key not in verified_models:
            verified_models[model_key] = verify_training_model(fold['model']['path'])
        verification = verified_models[model_key]
        flow_id = fold.get('flow_id', fold['fold_id'])
        omitted = (verification.get('omitted_seed') == fold['heldout_seed'] and
                   flow_id in verification.get('omitted_flow_ids', [])) if seed_cv else (
                   verification.get('omitted_flow_id') == flow_id)
        if (verification.get('verified') is not True or
                not omitted or
                verification.get('training_flow_count') != len(fold['training_flow_ids']) or
                set(verification.get('training_flow_ids', [])) != set(fold['training_flow_ids']) or
                fold['heldout_trace_id'] in verification.get('training_trace_ids', [])):
            raise ValueError('Crossfit model does not exclude the declared heldout flow')
        plan = read(fold['heldout_plan']['path'])
        if plan.get('phase') != 'predictor_training' or plan.get('split') != 'training':
            raise ValueError('Crossfit heldout must remain a training flow')
        raw, scored_ref = scored_heldout(fold['heldout_raw']); refs.append(scored_ref)
        if raw['plan'] != fold['heldout_plan']:
            raise ValueError('Crossfit heldout raw/plan mismatch')
        candidates = [c for c in plan['candidates'] if c['id'] == fold['candidate_id']]
        if len(candidates) != 1:
            raise ValueError('One exact heldout candidate required')
        candidate = candidates[0]
        prediction = predict(plan, candidate, read(fold['model']['path']), resources, quality)
        row = compare(plan, candidate, prediction, raw)
        row.update(fold_id=fold['fold_id'], seed=plan['seed'], model=fold['model'],
                   heldout_raw=fold['heldout_raw'], heldout_scored=scored_ref,
                   independent_validation=False)
        rows.append(row); by_seed[plan['seed']].append(row)
    for ref in refs:
        checked(ref)
    value = {'schema': 'gsdeploy.training_flow_crossfit_error_report.v1',
        'scope': ('Retrospective leave-one-seed-out development; not independent validation' if seed_cv else
                  'Retrospective leave-one-continuous-flow-out development; not independent validation'),
        'candidates': rows,
        'by_seed': {str(seed): ranking(items, 'shared_balanced') for seed, items in sorted(by_seed.items())},
        'complete_prediction_coverage': all(r['known'] == r['requested'] for r in rows),
        'requested': sum(r['requested'] for r in rows), 'known': sum(r['known'] for r in rows),
        'source_receipts': refs, 'independent_validation': False, 'GPU_started': False,
        'final_gain_eligible': False, 'thresholds_tuned': False}
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'report.json', value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'resources', 'quality', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    result = run(args.manifest, args.resources, args.quality, args.output)
    print(json.dumps({'requested': result['requested'], 'known': result['known'],
                      'independent_validation': False, 'GPU_started': False}))
