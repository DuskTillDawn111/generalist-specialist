"""Offline v3 utility review of the existing sealed six-capture panel. No GPU calls."""
import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.audit_activation_aggregate_capture import score_output
from experiments.fit_activation_native import load_bundle
from experiments.scoped_routing_objective import file_receipt, scoped_reference
from experiments.score_scoped_confirmation import score as original_score
from gsdeploy.forecasting.native_training import digest
from gsdeploy.utility_tradeoffs import (TERMS, WEIGHT_PROFILES, episode_metrics,
                                       score_terms, compare_panel)


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def review(campaign, config):
    # This CLI deliberately has one finite analysis scope, not an adaptive sweep.
    require(config['evaluation_role'] == 'retrospective_exploratory'
            and config['weight_profiles'] == list(WEIGHT_PROFILES)
            and config['deadline_multipliers'] == [1, 2, 4, 8]
            and config['energy_scale_factors'] == [.5, 1, 2]
            and config['quality_span_factors'] == [.5, 1, 2]
            and config['maximum_scenarios'] == 108
            and config['default_profile'] == 'four_groups_v3'
            and config['default_deadline_multiplier'] == 2
            and config['latency_scale'] == 'frozen_v2_deadline_ms_independent_of_slo'
            and all(config[k] == 0 for k in ('gpu_episodes', 'predictor_calls', 'new_search_runs')),
            'Expected fixed bounded offline review configuration')
    base = campaign / 'scoped-confirmation-decision-v2'
    captures = base / 'captures'
    receipts = {}

    def verify(receipt):
        for key in ('path', 'snapshot'):
            if key in receipt:
                actual = file_receipt(receipt[key])
                require(actual['sha256'] == receipt['sha256'], 'Changed frozen receipt: ' + receipt[key])
                receipts[actual['path']] = actual

    def recorded(path):
        r = file_receipt(path)
        receipts[r['path']] = r
        return read(path)

    final_review = recorded(base / 'final-review.json')
    for receipt in final_review['document_receipts'] + [final_review['final_decision'], final_review['completion_audit']]:
        verify(receipt)
    audit = recorded(base / 'final-audit.json')
    require(audit['status'] == 'PASS_SCOPED_COMPLETION_AUDIT', 'Prior completed audit required')
    verify(audit['comparison'])
    verify(audit['protocol'])
    protocol, comparison = read(captures / 'protocol.json'), read(captures / 'comparison.json')
    for receipt in protocol['input_receipts'] + protocol['source_receipts']:
        verify(receipt)
    require(len(protocol['predictions']) == len(comparison['reports']) == len(audit['captures']) == 6,
            'Exactly the existing six captures required')
    for capture in audit['captures']:
        verify(capture['raw'])
        verify(capture['hardware'])
    system = recorded(campaign / 'references-v2/system-frozen.json')
    evidence = recorded(campaign / 'scoped-optimization-preparation-v1/training-evidence.json')
    normalization = system['catalog']['utility_normalization']
    baseline_quality = {(r['family'], r['task'], r['pool_id']): r['quality_totals'] for r in evidence['quality']}
    require(len(baseline_quality) == len(evidence['quality']), 'Unique baseline quality identities required')
    archived = {r['run_id']: r for r in comparison['reports']}
    require(len(archived) == 6, 'Unique archived reports required')
    cohorts, episodes, metric_cache = {}, [], {}
    for sealed in protocol['predictions']:
        for key in ('plan', 'prediction'):
            verify(sealed[key])
        directory = Path(sealed['directory'])
        bundle = load_bundle(directory)
        raw, plan, context = (bundle[k] for k in ('raw', 'plan', 'context'))
        prediction = read(directory / 'prediction-sealed.json')
        observed = original_score(bundle, prediction, system, evidence)
        require(observed == archived[raw['run_id']], 'Original v2 score failed exact reproduction')
        require(not observed['changed_output_ids'], 'Unexpected quality replay mismatch')
        cohort = [{k: r[k] for k in ('task', 'pool_id', 'offset_ms')} for r in plan['jobs'][0]['requests']]
        require(digest(cohort) == sealed['offered_cohort_sha256'], 'Offered cohort hash changed')
        round_id = plan['round']
        require(round_id not in cohorts or cohorts[round_id] == cohort, 'Unpaired offered inputs/arrivals')
        cohorts[round_id] = cohort
        predicted_requests = prediction['prediction']['jobs'][0]['requests']
        # validate_capture inside original_score has matched ordered plan requests to batch IDs.
        ordered_plan = sorted(plan['jobs'][0]['requests'], key=lambda r: r.get('offset_ms', 0))
        expected = dict(zip(raw['batches'][0]['request_ids'], [r['task'] for r in ordered_plan]))
        require(len(expected) == len(cohort) == 96, 'Complete 96-request panel cohort required')
        data, rates = scoped_reference(system, plan)
        task_weights = {t: value / sum(rates.values()) for t, value in rates.items()}
        rows = dict(observed=[], predicted=[])
        for r in raw['request_records']:
            errors, units = score_output(r['worker_response'], r['task'], r['pool_id'], data)
            rows['observed'].append(dict(request_id=r['request_id'], task=r['task'],
                success=r['worker_response']['success'], errors=errors, units=units,
                latency_ms=(r['completion_time_ns'] - r['scheduled_time_ns']) / 1e6))
        for r in predicted_requests:
            errors, units = baseline_quality[context['resident_aliases'][r['alias']], r['task'], r['pool_id']]
            rows['predicted'].append(dict(request_id=r['request_id'], task=r['task'], success=True,
                errors=errors, units=units, latency_ms=r['predicted_e2e_ms']))
        joules = next(r['joules'] for r in observed['energy']['totals'] if r['window'] == 'startup_through_observation_end')
        require(math.isclose(sum(r['joules'] for r in observed['energy']['partitions']), joules,
                             abs_tol=1e-6, rel_tol=1e-12), 'Energy partitions do not sum to episode')
        for k in config['deadline_multipliers']:
            refs = {t: dict(best_loss=r['best_loss'], worst_loss=r['worst_loss'],
                           latency_scale_ms=system['latency_reference_ms'][t],
                           deadline_ms=k * r['reference_mean_e2e_ms'])
                    for t, r in normalization['tasks'].items()}
            pair = {}
            for kind in ('observed', 'predicted'):
                pair[kind] = episode_metrics(rows[kind], expected_tasks_by_id=expected,
                    task_references=refs, task_weights=task_weights,
                    episode_joules=joules if kind == 'observed' else prediction['episode_energy_evidence']['episode_joules'],
                    energy_scale_joules=normalization['episode_reference_joules'],
                    owned_peak_mib=observed['owned_peak_mib'] if kind == 'observed' else prediction['raw_peak_estimate_mib'],
                    memory_budget_mib=system['gpu_memory_mib'])
                if k == 2:
                    old = observed if kind == 'observed' else prediction
                    for term in TERMS:
                        require(math.isclose(pair[kind]['terms'][term], old[term + '_norm'], rel_tol=1e-11, abs_tol=1e-11),
                                'Per-request v2 term reconstruction failed: ' + kind + ':' + term)
                    require(math.isclose(score_terms(pair[kind]['terms'], WEIGHT_PROFILES['six_equal_v2'])['U'],
                                         old['U'], rel_tol=1e-11, abs_tol=1e-11), 'v2 U reconstruction failed')
            metric_cache[raw['run_id'], k] = pair
        episodes.append(dict(run_id=raw['run_id'], candidate_id=plan['candidate_id'], round=round_id,
            observed_feasible=observed['constraints_eligible'], predicted_feasible=prediction['constraints_eligible'],
            old_U=observed['U'], old_predicted_U=prediction['U'],
            metrics_by_deadline_multiplier={str(k): metric_cache[raw['run_id'], k] for k in config['deadline_multipliers']},
            energy_partitions=observed['energy']['partitions'], safe_memory_mib=observed['safe_memory_mib'],
            max_rho=observed['max_rho']))
    require({(r['candidate_id'], r['round']) for r in episodes} ==
            {(c, i) for c in ('candidate-1', 'candidate-2', 'candidate-3') for i in (1, 2)},
            'Original paired panel identities required')
    scenarios = []
    for profile, k, ef, qf in itertools.product(config['weight_profiles'], config['deadline_multipliers'],
                                               config['energy_scale_factors'], config['quality_span_factors']):
        panel = []
        for episode in episodes:
            row = {key: episode[key] for key in ('candidate_id', 'round', 'observed_feasible', 'predicted_feasible')}
            for kind in ('observed', 'predicted'):
                row[kind] = score_terms(metric_cache[episode['run_id'], k][kind]['terms'], WEIGHT_PROFILES[profile],
                                       quality_span_factor=qf, energy_scale_factor=ef)
            panel.append(row)
        scenarios.append(dict(profile=profile, deadline_multiplier=k, energy_scale_factor=ef,
                              quality_span_factor=qf, assessment=compare_panel(panel)))
    require(len(scenarios) == config['maximum_scenarios'], 'Scenario budget mismatch')
    default = next(r for r in scenarios if r['profile'] == config['default_profile'] and
                   r['deadline_multiplier'] == 2 and r['energy_scale_factor'] == r['quality_span_factor'] == 1)
    energy_pairs = []
    for episode in episodes:
        episode['default_scores'] = {kind: score_terms(
            metric_cache[episode['run_id'], 2][kind]['terms'], WEIGHT_PROFILES[config['default_profile']])
            for kind in ('observed', 'predicted')}
        control = next(r for r in episodes if r['candidate_id'] == 'candidate-1' and r['round'] == episode['round'])
        value = episode['metrics_by_deadline_multiplier']['2']['observed']['episode_joules']
        baseline = control['metrics_by_deadline_multiplier']['2']['observed']['episode_joules']
        energy_pairs.append(dict(run_id=episode['run_id'], episode_joules=value, joules_per_offered=value / 96,
            control_run_id=control['run_id'], measured_energy_saving_fraction_vs_pure_specialist=1 - value / baseline,
            comparison_only_not_utility_denominator=True))
    dominance = []
    for k in config['deadline_multipliers']:
        means = {c: {t: sum(metric_cache[e['run_id'], k]['observed']['terms'][t] for e in episodes
                            if e['candidate_id'] == c) / 2 for t in TERMS}
                 for c in ('candidate-1', 'candidate-2', 'candidate-3')}
        pairs = [dict(better=a, worse=b, strictly_better_every_term=all(means[a][t] < means[b][t] for t in TERMS))
                 for a, b in itertools.permutations(means, 2)
                 if all(means[a][t] <= means[b][t] for t in TERMS) and any(means[a][t] < means[b][t] for t in TERMS)]
        dominance.append(dict(deadline_multiplier=k, mean_terms=means, dominance=pairs))
    # Fixed task weights and spans expose preference gradients in raw loss units.
    require(all(e['metrics_by_deadline_multiplier']['2']['observed']['task_weights'] == task_weights
                for e in episodes), 'Shared task weights required for the reported gradients')
    gradients = {profile: {t: -weights['Q'] * task_weights[t] / (r['worst_loss'] - r['best_loss'])
                          for t, r in normalization['tasks'].items()}
                 for profile, weights in WEIGHT_PROFILES.items()}
    return dict(schema='gsdeploy.scoped_utility_review.v3', status='OFFLINE_REVIEW_COMPLETED',
        evaluation_role='retrospective_exploratory', config=config, weight_profiles=WEIGHT_PROFILES,
        default=default, episodes=episodes, scenarios=scenarios, energy_pairs=energy_pairs,
        pareto_by_deadline=dominance, dU_d_native_loss=gradients,
        winner_counts=dict(Counter(','.join(s['assessment']['observed_best']) for s in scenarios)),
        predicted_winner_counts=dict(Counter(s['assessment']['predicted_selected'] for s in scenarios)),
        maximum_mean_regret=max((s['assessment']['mean_regret'] for s in scenarios
                                if s['assessment']['mean_regret'] is not None), default=None),
        undefined_mean_regret_count=sum(s['assessment']['mean_regret'] is None for s in scenarios),
        maximum_per_round_regret=max((value for s in scenarios for value in s['assessment']['per_round_regret']
                                     if value is not None), default=None),
        undefined_per_round_regret_count=sum(value is None for s in scenarios for value in s['assessment']['per_round_regret']),
        original_six_observed_scores_exactly_reproduced=True,
        observed_and_predicted_v2_terms_reconstructed=True, shared_offered_cohorts_verified=True,
        source_receipts=list(receipts.values()), new_gpu_episodes=0, predictor_calls=0, new_search_runs=0,
        optimizer_default_changed=False, independent_v3_confirmation=False,
        limits=['Known inputs, fixed five-model residency, one replica each, two historical arrival traces',
                '108 scenarios are preference sensitivity, not a confidence interval',
                'Four equal groups are an explicit preference, not a scientifically optimal weighting',
                'Text/image joint delivery is diagnostic only; OCR/ASR joint value remains undefined',
                'Historical GPU energy scale is not a matched workload or whole-system baseline',
                'No new search, global optimality, unseen-input or fine near-tie ranking claim'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=ROOT / 'artifacts/activation-gpu-20261104')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/utility_tradeoffs_v3_20261105.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite existing review: ' + str(args.output))
    result = review(args.campaign.resolve(), read(args.config))
    result['implementation_receipts'] = [file_receipt(p) for p in
        (Path(__file__), ROOT / 'gsdeploy/utility_tradeoffs.py', args.config)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'winner_counts', 'maximum_mean_regret',
                                             'undefined_mean_regret_count', 'new_gpu_episodes')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
