"""Autonomous planned-batch replay and aggregate accuracy diagnostics.

Prediction accepts only the frozen fit and public plan. Heldout timestamps,
observed priming, controller gaps, output tokens and realized stage boundaries
are used only by scoring. This is an experimental adapter to the shared FIFO.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.fit_activation_native import load_bundle
from experiments.review_activation_aggregate_accuracy import metrics
from gsdeploy.energy import integrate_power
from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.lifecycle import LifecycleProvider
from gsdeploy.forecasting.native_context import NativeContextProvider
from gsdeploy.forecasting.native_training import validate_capture
from gsdeploy.forecasting.simulator import FullHorizonModel


def predict_plan(fit, plan):
    template = fit['profiles']
    context = template['execution_context_sha256']
    if plan['execution_context_sha256'] != context:
        raise ValueError('Forecast plan/runtime mismatch')
    if plan['aliases'] != fit['resident_aliases']:
        raise ValueError('Forecast resident roster differs from fitted runtime')
    provider = NativeContextProvider(template, context, allow_cpu_fixture=True)
    lifecycle = LifecycleProvider(fit['lifecycle'], context, allow_cpu_fixture=True)
    aliases = {a: dict(family=f, supported_tasks=sorted({p['task'] for p in template['profiles'] if p['family']==f}))
               for a, f in plan['aliases'].items()}
    model = FullHorizonModel(aliases, provider, lifecycle_provider=lifecycle)
    envelope = fit['episode_envelope']['means']
    now = round(envelope['startup_ms'] * 1e6)
    last_native = now - round(envelope['initial_idle_age_ms'] * 1e6)
    energy = envelope['startup_board_joules']
    energy_known = provider.requires_energy_history
    reports, counter, failure = [], 0, None

    def advance_idle(duration_ms):
        nonlocal now, last_native, energy, energy_known
        end = now + round(duration_ms * 1e6)
        if end > now and energy_known:
            try:
                joules, last_native = provider.energy_interval({}, now, end, last_native)
                energy += joules
            except UnknownEvidence:
                energy_known = False
        now = end

    advance_idle(envelope['formal_lead_ms'])
    for job in plan['jobs']:
        if reports:
            advance_idle(envelope['between_batch_ms'])
        start = now
        offered = []
        for request in sorted(job['requests'], key=lambda r: r['offset_ms']):
            offered.append(dict(request_id=f'training:{counter}', task=request['task'], pool_id=request['pool_id'],
                alias=request['alias'], arrival_ns=start+round(request['offset_ms']*1e6)))
            counter += 1
        queues, future = defaultdict(list), []
        for row in offered:
            if row['arrival_ns'] == start:
                queues[row['alias']].append(row)
            else:
                future.append(row)
        declared_end = start + round(job.get('duration_ms', 0.) * 1e6)
        horizon = max([declared_end, start, *[r['arrival_ns'] for r in offered]]) + 1
        try:
            forecast = model.forecast(dict(queues), start, horizon, future_arrivals=future,
                route_future=lambda row, pending, at_ns: row['alias'],
                drain_deadline_ns=horizon+120_000_000_000,
                require_energy=False, require_memory=False, require_episode_energy=True,
                energy_state=last_native if energy_known else None)
            if forecast['incomplete_ids']:
                raise UnknownEvidence('INCOMPLETE_AGGREGATE_REPLAY')
            results = []
            for row in offered:
                finish = forecast['finishes'][row['request_id']]['finish_ns']
                results.append(dict(row, predicted_e2e_ms=(finish-row['arrival_ns'])/1e6,
                                    predicted_finish_ns=finish))
            reports.append(dict(id=job['id'], status='predicted', requests=results,
                design=job.get('design', {}), predicted_start_ns=start,
                predicted_arrival_end_ns=declared_end if job.get('duration_ms') else None,
                predicted_drain_end_ns=forecast['drain_finish_ns']))
            if energy_known:
                joules = forecast['through_drain_board_energy_joules']
                if joules is None:
                    energy_known = False
                else:
                    energy += joules
                    last_native = forecast['final_energy_state']
            now = forecast['drain_finish_ns']
            advance_idle(envelope['batch_closure_ms'])
        except UnknownEvidence as exc:
            failure = str(exc)
            reports.append(dict(id=job['id'], status='UNKNOWN', reason=failure,
                                requested_count=len(offered)))
            break
    if failure is None:
        advance_idle(envelope['formal_tail_ms'])
    return dict(schema='gsdeploy.aggregate_plan_prediction.v1', jobs=reports,
                requested_jobs=len(plan['jobs']), requested_requests=sum(len(j['requests']) for j in plan['jobs']),
                complete=failure is None, failure=failure,
                predicted_episode_ms=now/1e6 if failure is None else None,
                predicted_episode_board_joules=energy if failure is None and energy_known else None,
                boundary=fit['episode_envelope']['scope'],
                observed_timestamps_used=False, physical_predictor_qualified=False)


def score_prediction(prediction, bundle, policy):
    validate_capture(bundle, allowed_splits=('training', 'development_validation'))
    raw, plan = bundle['raw'], bundle['plan']
    rows = {r['request_id']: r for r in raw['request_records']}
    actual_batches = {b['id']: b for b in raw['batches']}
    cohorts = defaultdict(list)
    throughput = []
    seen = set()
    for job in prediction['jobs']:
        if job['status'] != 'predicted':
            continue
        actual_batch = actual_batches[job['id']]
        if {r['request_id'] for r in job['requests']} != set(actual_batch['request_ids']):
            raise ValueError('Predicted and measured batch identities differ')
        for predicted in job['requests']:
            rid = predicted['request_id']
            if rid in seen:
                raise ValueError('Duplicate prediction')
            seen.add(rid)
            actual = rows[rid]
            if (actual['model'], actual['task'], actual['pool_id']) != (predicted['alias'], predicted['task'], predicted['pool_id']):
                raise ValueError('Prediction request identity changed')
            row = dict(observed_e2e_ms=(actual['completion_time_ns']-actual['scheduled_time_ns'])/1e6,
                       predicted_e2e_ms=predicted['predicted_e2e_ms'])
            path = plan['aliases'][actual['model']] + '/' + actual['task']
            cohorts['all'].append(row)
            cohorts['path:' + path].append(row)
            kind = job['design'].get('kind', 'solo')
            cohorts['kind:' + kind + '/path:' + path].append(row)
            if kind == 'continuous_arrivals':
                cohorts[job['id'] + '/task:' + actual['task']].append(row)
        if job['design'].get('kind') == 'continuous_arrivals':
            predicted_n = sum(r['predicted_finish_ns'] <= job['predicted_arrival_end_ns'] for r in job['requests'])
            actual_n = sum(rows[r]['completion_time_ns'] <= actual_batch['arrival_end_ns'] for r in actual_batch['request_ids'])
            duration = (actual_batch['arrival_end_ns']-actual_batch['start_ns'])/1e9
            error = abs(predicted_n-actual_n)/actual_n if actual_n else (0. if predicted_n == 0 else None)
            throughput.append(dict(job_id=job['id'], offered=len(job['requests']), predicted_completed=predicted_n,
                observed_completed=actual_n, window_seconds=duration,
                predicted_rps=predicted_n/duration, observed_rps=actual_n/duration, relative_error=error))
    complete = prediction['complete'] and seen == set(rows)
    thresholds = policy['thresholds']
    summaries = {}
    for key, cohort in cohorts.items():
        value = metrics(cohort)
        value['descriptive_within_latency_tolerances'] = bool(
            abs(value['mean_signed_relative_bias']) <= thresholds['per_candidate_load_task_aggregate_mean_relative_error_max']
            and abs(value['p95_signed_relative_bias']) <= thresholds['per_candidate_load_task_aggregate_p95_relative_error_max'])
        value['small_cohort_below_20'] = value['n'] < 20
        summaries[key] = value
    observed_energy = integrate_power(raw['board_samples'], raw['startup_begin_time_ns'],
                                      raw['observation_end_ns'], max_gap_seconds=.25)
    predicted_energy = prediction['predicted_episode_board_joules']
    observed_ms = (raw['observation_end_ns']-raw['startup_begin_time_ns'])/1e6
    return dict(run_id=raw['run_id'], source=bundle['source'], complete_cohort=complete,
        requested=len(rows), predicted=len(seen), aggregate_metrics=summaries if complete else {},
        partial_prefix_metrics={} if complete else summaries, throughput=throughput,
        episode=dict(observed_ms=observed_ms, predicted_ms=prediction['predicted_episode_ms'],
                     observed_board_joules=observed_energy, predicted_board_joules=predicted_energy,
                     relative_energy_error=abs(predicted_energy/observed_energy-1)
                     if predicted_energy is not None and observed_energy else None),
        status='TRAINING_REPLAY_ONLY' if raw['split']=='training' else 'DEVELOPMENT_REPLAY_ONLY',
        physical_predictor_qualified=False, admission_enabled=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--capture', type=Path, action='append', required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fit, policy = json.loads(args.fit.read_text()), json.loads(args.policy.read_text())
    reports = []
    for path in args.capture:
        bundle = load_bundle(path.resolve())
        if bundle['raw']['split']=='training' and bundle['source'] not in fit['source_receipts']:
            raise ValueError('Training replay must match fitted source receipts')
        if bundle['raw']['split']=='development_validation' and bundle['source'] in fit['source_receipts']:
            raise ValueError('Validation overlaps fitted sources')
        prediction = predict_plan(fit, bundle['plan'])
        reports.append(dict(prediction=prediction, assessment=score_prediction(prediction, bundle, policy)))
    result = dict(schema='gsdeploy.aggregate_autonomous_replay.v1',
        fit_sha256=hashlib.sha256(args.fit.read_bytes()).hexdigest(),
        policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(), reports=reports,
        physical_predictor_qualified=False, formal_utility_qualified=False,
        limitation='Existing fixed input profiles; independent input generalization and candidate ranking not qualified')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), complete_runs=sum(r['assessment']['complete_cohort'] for r in reports),
                          requested_runs=len(reports), physical_predictor_qualified=False)))
