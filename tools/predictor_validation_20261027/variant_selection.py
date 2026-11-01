"""Choose among prospectively frozen training-only shrinkage variants."""
import argparse
import json
from pathlib import Path

from .duration_selection import evaluate
from .native_flow import checked, receipt, save
from .probe_validation import read
from .profiling import digest


def choose(scores, preference):
    def key(name):
        row = scores[name]
        normalized = row['normalized_temporal_error_mean']
        return (row['decision_errors'], row['temporal_failed_or_unknown'],
                float('inf') if normalized is None else normalized, preference.index(name))
    return min(preference, key=key)


def select(protocol_path, report_paths, output):
    protocol = read(protocol_path)
    if (protocol.get('schema') != 'gsdeploy.duration_variants_training_cv_selection_protocol.v1' or
            protocol.get('protocol_sha256') != digest({k: v for k, v in protocol.items() if k != 'protocol_sha256'})):
        raise ValueError('Prospectively frozen three-candidate training protocol required')
    preference = protocol['selection']['exact_tie_preference']
    ids = [c['id'] for c in protocol['candidates']]
    if preference != ids or set(report_paths) != set(ids):
        raise ValueError('Every predeclared candidate must be compared in fixed tie order')
    direct = [protocol[k] for k in ('original_training_protocol', 'duration_fit_protocol',
        'baseline_seed_manifest', 'prior_selection_protocol', 'original_threshold_addendum')]
    for ref in direct:
        checked(ref)
    threshold_addendum = read(protocol['original_threshold_addendum']['path'])
    threshold_plan = read(checked(threshold_addendum['threshold_plan']))
    if threshold_addendum['thresholds'] != threshold_plan['validation_thresholds']:
        raise ValueError('Original thresholds changed')
    fit = read(protocol['original_training_protocol']['path'])
    contracts = {}
    for job in fit['flow_jobs']:
        plan = read(checked(job['plan']))
        contract = {'baseline': plan['baseline_candidate_id'],
                    'tie_tolerance': plan.get('utility_tie_tolerance', 0.),
                    'tasks': sorted(plan['workload_rps'])}
        if contracts.setdefault(plan['seed'], contract) != contract:
            raise ValueError('Inconsistent training comparison plan')
    refs = [receipt(protocol_path), receipt(__file__), receipt(evaluate.__code__.co_filename), *direct]
    scores, measurements = {}, None
    units = protocol['CV_unit']
    for name in ids:
        path = report_paths[name]; report = read(path)
        if (report.get('independent_validation') is not False or report.get('final_gain_eligible') is not False or
                not report.get('heldout_identity_audit', {}).get('omitted_raw_identity_verified')):
            raise ValueError('Audited whole-seed training development reports required')
        if name != 'duration_v12':
            if (report.get('variant') != name or report.get('selection_protocol') != receipt(protocol_path) or
                    report.get('threshold_addendum') != protocol['original_threshold_addendum']):
                raise ValueError('Variant report differs from frozen candidate/threshold identity')
            numerical = read(checked(report['variant_protocol']))
            if not protocol['created_ns'] < numerical['created_ns'] < report['created_ns']:
                raise ValueError('Selection protocol must precede variant model development')
        sources = report.get('source_receipts', []) + report.get('predictor_source_receipts', [])
        if not sources:
            raise ValueError('Actual model/report execution sources required')
        for ref in sources:
            checked(ref)
        refs.extend([receipt(path), *sources])
        observed = {(r['seed'], r['candidate_id']): (r['observed_U'], r['observed_metrics'])
                    for r in report['candidates']}
        if measurements is None:
            measurements = observed
        elif measurements != observed:
            raise ValueError('Variants must use identical heldout measurements')
        scores[name] = evaluate(report, units, threshold_addendum['thresholds'], contracts)
    selected = choose(scores, preference)
    result = {'schema': 'gsdeploy.duration_variants_training_cv_selection_result.v1',
        'selected_variant': selected, 'selection_protocol': receipt(protocol_path),
        'scores': scores, 'source_receipts': refs,
        'original_training_cv_thresholds_passed': scores[selected]['original_training_cv_thresholds_passed'],
        'selection_rule': protocol['selection'], 'independent_validation': False, 'final_gain_eligible': False,
        'GPU_started': False, 'scope': 'Training-only development selection; winning does not imply gate PASS'}
    save(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'baseline-report', 'ocr-report', 'ridge-report', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    value = select(args.protocol, {'duration_v12': args.baseline_report,
        'ocr_handler': args.ocr_report, 'phase_ridge1': args.ridge_report}, args.output)
    print(json.dumps({'selected': value['selected_variant'],
                      'training_cv_passed': value['original_training_cv_thresholds_passed']}))
