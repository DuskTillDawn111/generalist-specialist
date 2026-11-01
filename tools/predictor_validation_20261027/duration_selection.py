"""Predeclared training-only model selection; no automatic independent PASS."""
import argparse
import json
import math
from pathlib import Path

from .flow_crossfit_report import ranking
from .native_flow import checked, receipt, save
from .probe_validation import read
from .profiling import digest


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def evaluate(report, units, thresholds, contracts):
    """Lexicographic score with fixed denominators and unknown penalties."""
    expected = {(seed, cid) for seed in units['seeds'] for cid in units['routes']}
    rows = {(r['seed'], r['candidate_id']): r for r in report['candidates']}
    if len(rows) != len(report['candidates']) or set(rows) != expected:
        raise ValueError('Both algorithms must report every declared seed and candidate exactly once')
    decision_errors, decisions, metrics = 0, {}, []
    for seed in units['seeds']:
        selected = [rows[(seed, cid)] for cid in units['routes']]
        for row in selected:
            for field in ('predicted_comparison_U', 'observed_U'):
                if row[field] is not None and not finite(row[field]):
                    raise ValueError('Nonfinite utility is not a known decision outcome')
        result = ranking(selected, contracts[seed]['baseline'],
                         tie_tolerance=contracts[seed]['tie_tolerance'])
        known_sign = sum(r['sign_correct'] for r in result['paired_deltas'])
        known_rank = sum(r['correct'] for r in result['candidate_pairs'])
        expected_sign = len(units['routes']) - 1
        expected_rank = len(units['routes']) * (len(units['routes']) - 1) // 2
        result['errors_including_missing'] = expected_sign + expected_rank - known_sign - known_rank
        decision_errors += result['errors_including_missing']; decisions[str(seed)] = result
    def add(seed, cid, name, value, threshold):
        if not finite(threshold) or threshold <= 0:
            raise ValueError('Positive original temporal threshold required')
        known = finite(value) and value >= 0
        metrics.append({'seed': seed, 'candidate_id': cid, 'metric': name,
                        'value': value if known else None, 'threshold': threshold,
                        'normalized_error': value / threshold if known else None,
                        'failed_or_unknown': not known or value > threshold})
    for (seed, cid), row in sorted(rows.items()):
        for name, key in (('service', 'service_mare_max'), ('e2e', 'e2e_mare_max')):
            value = row.get('timing_errors', {}).get(name, {}).get('mean_absolute_relative_error')
            add(seed, cid, name + '_mare', value, thresholds[key])
        add(seed, cid, 'queue_mae', row.get('queue_error', {}).get('mae'),
            thresholds['queue_depth_mean_absolute_error_max_requests'])
        for task in contracts[seed]['tasks']:
            value = row.get('per_task_p95', {}).get(task, {}).get('absolute_relative_error')
            add(seed, cid, task + '_p95_are', value,
                thresholds['each_task_p95_absolute_relative_error_max'])
    unknown = any(r['normalized_error'] is None for r in metrics)
    normalized = None if unknown else sum(r['normalized_error'] for r in metrics) / len(metrics)
    failures = sum(r['failed_or_unknown'] for r in metrics)
    return {'decision_errors': decision_errors, 'temporal_failed_or_unknown': failures,
            'normalized_temporal_error_mean': normalized,
            'normalized_temporal_error_reason': 'UNKNOWN_COUNTS_AS_POSITIVE_INFINITY' if unknown else None,
            'metrics': metrics, 'by_seed': decisions,
            'original_training_cv_thresholds_passed': decision_errors == 0 and failures == 0}


def select(protocol_path, addendum_path, old_report_path, new_report_path, output):
    protocol, addendum = read(protocol_path), read(addendum_path)
    if (protocol.get('schema') != 'gsdeploy.duration_loss_training_cv_selection_protocol.v1' or
            protocol.get('protocol_sha256') != digest({k: v for k, v in protocol.items() if k != 'protocol_sha256'}) or
            addendum.get('schema') != 'gsdeploy.training_cv_selection_threshold_source_addendum.v1' or
            addendum['selection_protocol'] != receipt(protocol_path)):
        raise ValueError('Frozen predeclared model selection and threshold addendum required')
    plan = read(checked(addendum['threshold_plan']))
    if addendum['thresholds'] != plan['validation_thresholds']:
        raise ValueError('Selection thresholds differ from original frozen flow gate')
    for ref in protocol['source_receipts']:
        checked(ref)
    fit_protocol = read(checked(protocol['frozen_training_protocol']))
    contracts = {}
    for job in fit_protocol['flow_jobs']:
        training_plan = read(checked(job['plan']))
        contract = {'baseline': training_plan['baseline_candidate_id'],
                    'tie_tolerance': training_plan.get('utility_tie_tolerance', 0.),
                    'tasks': sorted(training_plan['workload_rps'])}
        if contracts.setdefault(training_plan['seed'], contract) != contract:
            raise ValueError('Inconsistent comparison contract within a training seed')
    paths = {'relative_work_v11': old_report_path, 'duration_weighted_relative_work_v12': new_report_path}
    reports, scores, refs = {}, {}, [receipt(p) for p in (protocol_path, addendum_path, __file__)]
    for name, path in paths.items():
        report = read(path)
        if report.get('independent_validation') is not False or report.get('final_gain_eligible') is not False:
            raise ValueError('Training development reports only')
        sources = report.get('source_receipts', []) + report.get('predictor_source_receipts', [])
        if not sources or not report.get('heldout_identity_audit', {}).get('omitted_raw_identity_verified'):
            raise ValueError('Complete heldout-identity and source audit required')
        for ref in sources:
            checked(ref)
        reports[name] = report; refs.extend([receipt(path), *sources])
        scores[name] = evaluate(report, protocol['training_units'], addendum['thresholds'], contracts)
    new_report = reports['duration_weighted_relative_work_v12']
    if (new_report.get('selection_protocol') != receipt(protocol_path) or
            new_report.get('threshold_addendum') != receipt(addendum_path)):
        raise ValueError('New model report does not bind the prospectively frozen selection rule')
    duration_protocol = read(checked(new_report['duration_protocol']))
    if not (protocol['created_ns'] < duration_protocol['created_ns'] < new_report['created_ns'] and
            addendum['created_ns'] < duration_protocol['created_ns']):
        raise ValueError('Selection rule and thresholds must precede new model development report')
    old_actual = {(r['seed'], r['candidate_id']): (r['observed_U'], r['observed_metrics'])
                  for r in reports['relative_work_v11']['candidates']}
    new_actual = {(r['seed'], r['candidate_id']): (r['observed_U'], r['observed_metrics'])
                  for r in reports['duration_weighted_relative_work_v12']['candidates']}
    if old_actual != new_actual:
        raise ValueError('Algorithms are not assessed against identical heldout measurements')
    key = lambda name: (scores[name]['decision_errors'], scores[name]['temporal_failed_or_unknown'],
                        scores[name]['normalized_temporal_error_mean'] if
                        scores[name]['normalized_temporal_error_mean'] is not None else float('inf'))
    old, new = protocol['candidate_algorithms']
    winner = new if key(new) < key(old) else old
    value = {'schema': 'gsdeploy.duration_loss_training_cv_selection_result.v1',
        'selected_algorithm': winner, 'scores': scores,
        'selection_rule': protocol['selection'], 'source_receipts': refs,
        'independent_validation': False, 'GPU_started': False, 'final_gain_eligible': False,
        'original_training_cv_thresholds_passed': scores[winner]['original_training_cv_thresholds_passed'],
        'next_scope': 'Selected development model still requires fresh independent validation; winning is not gate PASS'}
    save(output, value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'addendum', 'old-report', 'new-report', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    value = select(args.protocol, args.addendum, args.old_report, args.new_report, args.output)
    print(json.dumps({'selected': value['selected_algorithm'],
                      'training_cv_passed': value['original_training_cv_thresholds_passed']}))
