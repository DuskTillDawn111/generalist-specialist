"""CPU quality gates and paired specialist comparisons for frozen cohorts."""

import math

from ..tasks import score_result, summarize_quality


def quality_gate(policy, task, summary):
    rules = policy['quality_gate']['minimum_quality_thresholds'][task]
    issues = []
    if summary['failed_samples']:
        issues.append('FAILED_OR_UNVERIFIED_COMPLETION')
    for metric, threshold in rules.items():
        if metric in {'answerable_EM', 'unanswerable_EM'}:
            value = summary[metric.removesuffix('_EM')]['EM']
        else:
            value = summary.get(metric.removesuffix('_max'))
        if type(value) not in (float, int) or not math.isfinite(value):
            issues.append('MISSING_' + metric)
        elif (value > threshold) if metric.endswith('_max') else (value < threshold):
            issues.append('BELOW_POLICY_' + metric)
    return {'passed': not issues, 'issues': issues, 'rules': rules}


def compare_quality(policy, samples, generalist_results, specialist_results):
    """Paired cluster bootstrap; positive improvement means specialist better."""
    import numpy as np

    task = samples[0]['task']
    g_summary = summarize_quality(samples, generalist_results)
    s_summary = summarize_quality(samples, specialist_results)
    rule = policy['quality_gate']['specialist_comparison']['paired_cluster_bootstrap']
    by_g = {r['sample_id']: r for r in generalist_results}
    by_s = {r['sample_id']: r for r in specialist_results}
    clusters = {}
    for sample in samples:
        cluster = str(sample[rule['clusters'][task]])
        a, b = score_result(sample, by_g[sample['id']]), score_result(sample, by_s[sample['id']])
        if task == 'text':
            gain, count = b['EM'] - a['EM'], 1
        elif task == 'image':
            gain, count = int(b['correct']) - int(a['correct']), 1
        else:
            if a['reference_units'] != b['reference_units']:
                raise ValueError('Reference denominators differ between models')
            gain, count = a['errors'] - b['errors'], a['reference_units']
        values = clusters.setdefault(cluster, [0.0, 0.0])
        values[0] += gain
        values[1] += count
    groups = np.asarray(list(clusters.values()), dtype=float)
    if len(groups) < 2:
        raise ValueError('At least two independent source clusters are required')
    point = float(groups[:, 0].sum() / groups[:, 1].sum())
    rng = np.random.default_rng(rule['seed'])
    gains = []
    for start in range(0, rule['resamples'], 256):
        count = min(256, rule['resamples'] - start)
        picked = rng.integers(0, len(groups), size=(count, len(groups)))
        totals = groups[picked].sum(axis=1)
        gains.extend((totals[:, 0] / totals[:, 1]).tolist())
    alpha = (1 - rule['one_sided_confidence']) / rule['familywise_comparisons']
    lower = float(np.quantile(gains, alpha))
    valid = not (g_summary['failed_samples'] or s_summary['failed_samples'])
    return {'task': task, 'primary_metric': policy['quality_gate']['specialist_comparison']['primary_metrics'][task],
            'paired_samples': len(samples), 'source_clusters': len(groups),
            'specialist_improvement': point, 'one_sided_lower_bound': lower,
            'confidence': rule['one_sided_confidence'], 'familywise_comparisons': rule['familywise_comparisons'],
            'passed': bool(valid and point > 0 and lower > 0),
            'failure_reason': None if valid else 'FAILED_OR_UNVERIFIED_COMPLETION',
            'generalist_summary': g_summary, 'specialist_summary': s_summary,
            'scope': 'paired_quality_only_not_complete_pool_admission'}


def compare_costs(generalist, specialist, memory_ratio=(0.5, 2), latency_ratio=(0.5, 2)):
    """Only matched open-loop observations can establish the energy constraint."""
    required = ['task', 'input_manifest_sha256', 'trace_sha256', 'gpu_uuid', 'duration_seconds',
                'startup_amortization_seconds', 'energy_method', 'protocol_sha256', 'power_limit_watts']
    issues = []
    for key in required:
        if (key not in generalist or key not in specialist or generalist.get(key) is None
                or generalist.get(key) != specialist.get(key)):
            issues.append('UNMATCHED_' + key)
    for value in [generalist, specialist]:
        if (value.get('kind') != 'measured' or value.get('scope') != 'whole_gpu_board'
                or value.get('workload') != 'matched_open_loop' or not value.get('coverage_valid')
                or not value.get('stable') or not value.get('complete_cohort')
                or value.get('energy_method') != 'nvml_board_power_integral'):
            issues.append('INVALID_MEASURED_COST_EVIDENCE')
        for key in ['resident_mib', 'mean_latency_ms', 'effective_board_watts',
                    'duration_seconds', 'startup_amortization_seconds', 'power_limit_watts']:
            v = value.get(key)
            if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
                issues.append('MISSING_' + key)
    if issues:
        return {'passed': False, 'status': 'INSUFFICIENT_OR_UNMATCHED_EVIDENCE', 'issues': sorted(set(issues))}
    m = specialist['resident_mib'] / generalist['resident_mib']
    l = specialist['mean_latency_ms'] / generalist['mean_latency_ms']
    e = specialist['effective_board_watts'] - generalist['effective_board_watts']
    return {'passed': memory_ratio[0] <= m <= memory_ratio[1] and latency_ratio[0] <= l <= latency_ratio[1] and e > 0,
            'memory_ratio': m, 'latency_ratio': l, 'specialist_energy_increase_watts': e,
            'scope': 'matched_single_path_cost_gate_not_whole_plan_utility'}
