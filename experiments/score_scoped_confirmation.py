"""Score sealed candidate confirmations with unchanged six-term references."""
from collections import defaultdict
import argparse
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.analyze_activation_semantic import output_identity
from experiments.analyze_activation_trim_episode_energy import describe as describe_energy
from experiments.audit_activation_aggregate_capture import score_output
from experiments.fit_activation_native import load_bundle
from experiments.review_activation_aggregate_accuracy import metrics
from experiments.scoped_routing_objective import scoped_reference, file_receipt
from gsdeploy.forecasting.native_training import validate_capture
from gsdeploy import normalization as norm
from gsdeploy.prediction import components


def score(bundle, prediction, system, evidence):
    validate_capture(bundle, allowed_splits=('confirmation',))
    raw, plan, context = (bundle[k] for k in ('raw', 'plan', 'context'))
    if (plan.get('evaluation_role') != 'locked_candidate_confirmation' or plan.get('not_for_training') is not True
            or plan['execution_context_sha256'] != evidence['execution_context_sha256']):
        raise ValueError('Locked confirmation/context identity required')
    for alias, family in context['resident_aliases'].items():
        actual = raw['ready'][alias].get('runtime_scope', {})
        strict = context['runtime_policy']['scope'] == 'all_strict' or family == 'generalist_2'
        if (actual.get('scope') != context['runtime_policy']['scope'] or actual.get('slot') != family
                or actual.get('strict_requested') is not strict or actual.get('deterministic_algorithms') is not strict
                or actual.get('cublas_workspace_config') != (':4096:8' if strict else None)
                or strict and (actual.get('deterministic_warn_only') is not False
                    or actual.get('cudnn_benchmark') is not False or actual.get('cudnn_deterministic') is not True)):
            raise ValueError('Observed runtime differs from frozen scope')
    for row in raw['priming_request_records']+raw['request_records']:
        response = row['worker_response']
        if response.get('success') is not True:
            raise ValueError('Complete successful cohort required; never drop failed outputs')
        if context['resident_aliases'][row['model']] == 'generalist_2':
            modifier = response.get('memory_optimizations', {}).get('exllama_reconstruct_runtime', {})
            if (modifier.get('matmul_recons_thd') != 1 or modifier.get('configured_after_model_load') is not True
                    or modifier.get('quantized_weights_changed') is not False or modifier.get('quantized_layers', 0) <= 0):
                raise ValueError('G2 runtime path differs')
    data, rates = scoped_reference(system, plan)
    energy = describe_energy(raw)
    if not energy['coverage_valid']:
        raise ValueError('Complete GPU episode energy coverage required')
    joules = next(r['joules'] for r in energy['totals'] if r['window'] == 'startup_through_observation_end')
    episode = dict(kind='measured', scope='whole_gpu_board', episode_coverage_valid=True,
                   episode_protocol=norm.contract(data)['episode_protocol'], episode_joules=joules)
    samples = [s for s in raw['board_samples'] if raw['startup_begin_time_ns'] <= s['time_ns'] <= raw['observation_end_ns']]
    if not samples or any(type(s.get('owned_used_mib')) not in (int, float)
                          or not math.isfinite(s['owned_used_mib']) or s['owned_used_mib'] < 0 for s in samples):
        raise ValueError('Complete owned memory samples required')
    peak = max(s['owned_used_mib'] for s in samples)
    predicted = {r['request_id']: r for r in prediction['prediction']['jobs'][0]['requests']}
    if set(predicted) != {r['request_id'] for r in raw['request_records']}:
        raise ValueError('Prediction and observed cohorts differ')
    baselines = {(r['family'], r['task'], r['pool_id']): r for r in evidence['quality']}
    quality, latencies, errors, changed = defaultdict(list), defaultdict(list), defaultdict(list), []
    for row in raw['request_records']:
        p = predicted[row['request_id']]
        if (p['alias'], p['task'], p['pool_id']) != (row['model'], row['task'], row['pool_id']):
            raise ValueError('Sealed routing/input assignment changed')
        task = row['task']
        latency = (row['completion_time_ns']-row['scheduled_time_ns'])/1e6
        latencies[task].append(latency)
        errors[task].append(dict(observed_e2e_ms=latency, predicted_e2e_ms=p['predicted_e2e_ms']))
        quality[task].append(score_output(row['worker_response'], task, row['pool_id'], data))
        baseline = baselines[context['resident_aliases'][row['model']], task, row['pool_id']]
        if output_identity(row['worker_response']) != baseline['output_sha256']:
            changed.append(row['request_id'])
    native_quality = {t: sum(r[0] for r in values)/sum(r[1] for r in values) for t, values in quality.items()}
    Q = sum(rates[t]*norm.quality_loss(data, t, native_quality[t]) for t in rates)/sum(rates.values())
    stats = {t: dict(mean_ms=float(np.mean(values)), p95_ms=float(np.percentile(values, 95)), request_count=len(values))
             for t, values in latencies.items()}
    T = norm.deadline_loss([(t, latency, True) for t, values in latencies.items() for latency in values], system['latency_reference_ms'])
    terms = components(stats, rates, system['latency_reference_ms'], T, peak/system['gpu_memory_mib'],
                       system['weights'], norm.episode_cost(data, episode), Q, data=data)
    routing = plan['jobs'][0]['design']['routing_rps']
    rho = {m: sum(routing[t].get(m, 0)*ms/1000 for t, ms in services.items())
           for m, services in evidence['capacity_service_ms'].items()}
    safe_memory = peak*data['safe_factor']
    eligible = safe_memory <= system['gpu_memory_mib']+1e-6 and max(rho.values()) <= .8+1e-8
    end = raw['batches'][0]['arrival_end_ns']
    observed_n = sum(r['completion_time_ns'] <= end for r in raw['request_records'])
    predicted_job = prediction['prediction']['jobs'][0]
    predicted_n = sum(r['predicted_finish_ns'] <= predicted_job['predicted_arrival_end_ns'] for r in predicted.values())
    return dict(run_id=raw['run_id'], candidate_id=plan['candidate_id'], round=plan['round'],
        source=bundle['source'], **terms, constraints_eligible=eligible, rho=rho, max_rho=max(rho.values()),
        owned_peak_mib=peak, safe_memory_mib=safe_memory, energy=energy,
        per_task=stats, native_quality=native_quality, changed_output_ids=changed,
        predicted_U=prediction['U'], prediction_error={t: metrics(rows) for t, rows in errors.items()},
        energy_relative_error=abs(prediction['episode_energy_evidence']['episode_joules']/joules-1),
        throughput=dict(observed_completed=observed_n, predicted_completed=predicted_n,
                        relative_error=abs(predicted_n-observed_n)/observed_n if observed_n else (0. if predicted_n == 0 else None)),
        memory_estimate_error_mib=prediction['raw_peak_estimate_mib']-peak,
        original_rho_contract=evidence['capacity_contract'], physical_generalization_qualified=False)


def assess_panel(reports, tolerance=.02):
    candidates = sorted({r['candidate_id'] for r in reports})
    if (not candidates or len(candidates) > 3 or len(reports) != 2*len(candidates)
            or {(r['candidate_id'], r['round']) for r in reports} != {(c, i) for c in candidates for i in (1, 2)}):
        raise ValueError('Complete unique paired candidate panel required')
    predicted_mean = {c: sum(r['predicted_U'] for r in reports if r['candidate_id'] == c)/2 for c in candidates}
    chosen = max(candidates, key=lambda c: (predicted_mean[c], c))
    observed_mean = {c: sum(r['U'] for r in reports if r['candidate_id'] == c)/2
                     for c in candidates if all(r['constraints_eligible'] for r in reports if r['candidate_id'] == c)}
    best = max(observed_mean, key=lambda c: (observed_mean[c], c)) if observed_mean else None
    per_round = []
    for i in (1, 2):
        feasible = {r['candidate_id']: r['U'] for r in reports if r['round'] == i and r['constraints_eligible']}
        regret = max(feasible.values())-feasible[chosen] if chosen in feasible else None
        per_round.append(dict(round=i, selected_candidate_feasible=chosen in feasible, selection_regret=regret))
    regret = observed_mean[best]-observed_mean[chosen] if chosen in observed_mean else None
    accepted = regret is not None and regret <= tolerance and all(r['selection_regret'] is not None and r['selection_regret'] <= tolerance for r in per_round)
    pairs = []
    for a, b in itertools.combinations(candidates, 2):
        left = {r['round']: r for r in reports if r['candidate_id'] == a}
        right = {r['round']: r for r in reports if r['candidate_id'] == b}
        delta = [right[i]['U']-left[i]['U'] for i in (1, 2)]
        pairs.append(dict(candidate_a=a, candidate_b=b, delta_U_b_minus_a=delta, mean_delta_U=sum(delta)/2,
            per_term_mean_delta={t: sum(right[i][t+'_contribution']-left[i][t+'_contribution'] for i in (1, 2))/2
                                 for t in ('Q', 'T', 'L', 'P', 'M', 'E')},
            near_tie=abs(sum(delta)/2) <= tolerance,
            both_candidates_feasible_in_both_rounds=a in observed_mean and b in observed_mean))
    return dict(predicted_selected_candidate=chosen, observed_best_candidate=best,
        predicted_mean_U=predicted_mean, observed_feasible_mean_U=observed_mean,
        mean_selection_regret=regret, per_round=per_round,
        descriptive_panel_selection_accepted=accepted, selection_regret_tolerance=tolerance,
        paired_comparisons=pairs, independent_runs_per_candidate=2,
        population_confidence_claim=False, generalization_to_unseen_inputs=False,
        next_action='Report scoped measured comparison; no more predictor experiments' if accepted else
                    'Identify decision-changing errors or restrict selection to measured reliable candidates; no automatic new campaign')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'evidence', 'reference', 'policy'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    protocol = json.loads((a.base/'protocol.json').read_text())
    for receipt in protocol['input_receipts']:
        if file_receipt(receipt['path']) != receipt:
            raise ValueError('Frozen confirmation input changed')
    for receipt in protocol['source_receipts']:
        if any(file_receipt(receipt[k])['sha256'] != receipt['sha256'] for k in ('path', 'snapshot')):
            raise ValueError('Frozen confirmation source changed')
    system, evidence, policy = (json.loads(p.read_text()) for p in (a.reference, a.evidence, a.policy))
    reports = []
    for receipt in protocol['predictions']:
        for key in ('plan', 'prediction'):
            if file_receipt(receipt[key]['path']) != receipt[key]:
                raise ValueError('Sealed candidate plan/prediction changed')
        directory = Path(receipt['directory'])
        hardware = json.loads((directory/'hardware.json').read_text())
        if hardware['started_unix']*1e9 <= protocol['sealed_unix_ns']:
            raise ValueError('Collection preceded full-panel prediction seal')
        reports.append(score(load_bundle(directory), json.loads((directory/'prediction-sealed.json').read_text()), system, evidence))
    assessment = assess_panel(reports, policy['thresholds']['panel_selection_regret_max'])
    thresholds = policy['thresholds']
    failures = []
    for r in reports:
        for task, cell in r['prediction_error'].items():
            if abs(cell['mean_signed_relative_bias']) > thresholds['per_candidate_load_task_aggregate_mean_relative_error_max'] or abs(cell['p95_signed_relative_bias']) > thresholds['per_candidate_load_task_aggregate_p95_relative_error_max']:
                failures.append(r['run_id']+':'+task+':LATENCY_TOLERANCE')
        if r['energy_relative_error'] > thresholds['full_gpu_board_episode_energy_relative_error_max']:
            failures.append(r['run_id']+':EPISODE_ENERGY_TOLERANCE')
        error = r['throughput']['relative_error']
        if error is None or error > thresholds['fixed_window_throughput_relative_error_max']:
            failures.append(r['run_id']+':THROUGHPUT_TOLERANCE')
    result = dict(schema='gsdeploy.scoped_measured_comparison.v1', reports=reports, assessment=assessment,
        prediction_tolerance_failures=failures, local_accuracy_does_not_replace_selection_regret=True,
        protocol_receipt=file_receipt(a.base/'protocol.json'),
        interpretation='Known-input fixed-residency two-history panel comparison; not a general physical or continuous-global-optimality claim')
    with (a.base/'comparison.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(assessment, ensure_ascii=False))


if __name__ == '__main__':
    main()
