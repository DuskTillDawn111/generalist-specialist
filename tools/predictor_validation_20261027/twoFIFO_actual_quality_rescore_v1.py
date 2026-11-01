"""Append actual-output training labels after persisted all-case predictions.

No forecast, folded fit, original measured label or UNKNOWN draw is rewritten.
This sidecar evaluates quality from exact native outputs and frozen references.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import twoFIFO_native_quality_labels_v1 as quality
from . import twoFIFO_source_rebound_resource_rescore_v1 as previous
from . import twoFIFO_recovery_generated_v1 as generated
from . import twoFIFO_recovery_resource_v1 as resource
from .implementation_source_closure_v27 import implementation_sources, strict_references

SCHEMA = 'gsdeploy.twoFIFO_actual_quality_allcase_training_rescore.v1'
require, receipt, checked, save = previous.require, previous.receipt, previous.checked, previous.save


def evaluate_case(raw, plan, planned, epoch, scoring_plan, catalog_ref, observed, prediction):
    """Pure label computation after the caller authenticates persisted evidence."""
    measured = quality.measure_case(raw, plan, planned, epoch, scoring_plan, catalog_ref)
    reader_ids = [r['request_id'] for r in observed['reader_labels']['requests']]
    require(measured['offered_requests'] == observed['offered_requests']
        and len(reader_ids) == len(set(reader_ids)) == observed['offered_requests']
        and {r['request_id'] for r in measured['requests']} ==
            set(reader_ids),
        'Actual quality and observed timing must cover the identical offered cohort')
    score = quality.with_measured_quality(observed, measured, scoring_plan)
    errors = previous.score_errors(prediction['ensemble'], score)
    errors.update(frozen_quality_used_on_both_sides=False,
        actual_native_outputs_used_for_measured_quality=True)
    return measured, score, errors


def run(completed_rescore_path, catalog_path, output):
    implementation = implementation_sources(__file__)
    report_ref, catalog_ref = receipt(completed_rescore_path), receipt(catalog_path)
    report = checked(report_ref)
    require(report.get('schema') == previous.SCHEMA
        and report.get('status') == 'COMPLETE_REBOUND_ALLCASE_RESOURCE_TRAINING_DIAGNOSTICS'
        and report.get('generated_producer_schema') == generated.SCHEMA
        and report.get('labels_generated') is True
        and report.get('validation_labels_read') is False
        and report.get('cases') == 150 and report.get('offered_requests') == 546,
        'Complete recovery all-case training prediction/label report required')
    for ref in report['sources']:
        require(receipt(ref['path']) == ref, 'All-case training source changed')
    parent = checked(report['completed_generated_report'])
    require(parent.get('schema') == generated.SCHEMA and parent.get('response_fit_reused') is True,
            'Exact completed recovery prediction producer required')
    require(receipt(parent['design']['path']) == parent['design'], 'Generated design receipt changed')
    design = generated.audit(parent['design']['path'])
    fitted = generated.response.audit_fit(parent['response_fit'])['report']
    require(fitted['manifest'] == design['fit_manifest'], 'Parent fit and generated design differ')
    manifest = generated.response.audit(design['fit_manifest']['path'])
    executor_ref = manifest['original_executor_manifest']
    executor = checked(executor_ref)
    plan_refs = [item['plan'] for item in executor['attempts']]
    plans = [checked(ref) for ref in plan_refs]
    previous._layout(parent, plans)
    canonical_plans = {p['attempt_id']: ref for p,ref in zip(plans,plan_refs)}
    expected_keys = {(p['capture_id'],p['attempt_id'],c['case_id']) for p in plans for c in p['cases']}
    archived = {(f['heldout_block_id'],c['attempt_id'],c['case_id']):c['prediction']
        for f in parent['whole_block_leaveout'] for c in f['cases']}
    model = checked(report['resource_model'])
    # The producer APIs authenticate their explicit recovery schemas.
    require(model.get('schema') == resource.SCHEMA and model.get('status') == resource.COMPLETE_STATUS
        and model.get('intake_evidence') == fitted['intake_evidence']
        and model.get('continuation') == fitted['continuation'], 'Resource/response native intake differs')
    require(receipt(model['design']['path']) == model['design'], 'Resource design receipt changed')
    require(resource.audit(model['design']['path'])['fit_manifest'] == design['fit_manifest'],
            'Resource and response designs differ')
    canonical_reviews = {item['attempt_id']:item['training_review'] for item in model['intake_evidence']['attempts']}
    require(set(canonical_reviews) == set(canonical_plans), 'All twelve canonical native reviews required')
    scoring_plan = checked(design['scoring_plan'])
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    results, keys, sources = [], set(), [report_ref, catalog_ref, report['sources'], implementation,
        report['completed_generated_report'], parent['sources'], report['resource_model'], model['sources'],
        parent['design'], model['design'], parent['response_fit'], fitted['sources'], executor_ref, plan_refs]
    for ref in strict_references(sources):
        require(receipt(ref['path']) == ref, 'Upstream prediction/model source changed')
    raw_cache = {}
    for index, row in enumerate(report['output_cases']):
        key = row['block_id'], row['attempt_id'], row['case_id']
        require(key in expected_keys and key not in keys, 'Missing, foreign or duplicate actual-quality case identity')
        keys.add(key)
        prediction, observed = checked(row['prediction']), checked(row['measured_label'])
        require(prediction['archived_prediction'] == archived[key]
            and prediction['resource_model'] == report['resource_model']
            and prediction['resource_fold_binding'] == report['fold_provider_cache'][row['block_id']],
            'Prediction archive/resource/fold lineage differs from the complete upstream report')
        plan_ref = prediction['target_plan']
        require(plan_ref == canonical_plans[row['attempt_id']], 'Prediction target is not the canonical training plan')
        plan = checked(plan_ref)
        require((plan['capture_id'], plan['attempt_id'], observed['case_id']) == key
            and observed['attempt_id'] == plan['attempt_id'], 'Prediction/label plan identity changed')
        evidence = observed['measured_resources']['source_evidence']
        require(evidence['persisted_resource_prediction'] == row['prediction']
            and evidence['training_review'] == canonical_reviews[row['attempt_id']],
                'Measured native label must follow this exact persisted prediction')
        review = checked(evidence['training_review'])
        raw_ref = evidence['raw']
        require(review['raw'] == raw_ref and review['issues'] == []
            and review['schema'] == generated.response.intake.native_review.SCHEMA
            and review['source_rebound_manifest'] == executor_ref
            and review['inputs']['plan.json'] == plan_ref
            and review['training_feature_extraction_eligible'] is True,
            'Exact accepted native training evidence required')
        if raw_ref['path'] not in raw_cache:
            raw_cache.clear()
            raw_cache[raw_ref['path']] = checked(raw_ref)
        raw = raw_cache[raw_ref['path']]
        require(raw['plan'] == plan_ref and raw['context'] == plan['context'],
                'Raw execution and canonical plan/context differ')
        planned = [c for c in plan['cases'] if c['case_id'] == row['case_id']]
        require(len(planned) == 1, 'One exact public case required')
        epoch = observed['measured_resources']['original_window_ns'][0]
        measured, score, errors = evaluate_case(raw, plan, planned[0], epoch,
            scoring_plan, catalog_ref, observed, prediction)
        folder = out / ('case-' + str(index + 1))
        folder.mkdir()
        results.append(dict(block_id=key[0], attempt_id=key[1], case_id=key[2],
            offered_requests=measured['offered_requests'], original_prediction=row['prediction'],
            original_measured_label=row['measured_label'],
            actual_quality_label=save(folder/'actual-quality.json', measured),
            actual_six_score=save(folder/'actual-score.json', score), score_errors=errors))
        sources.extend([row['prediction'], row['measured_label'], plan_ref, raw_ref,
                        evidence['training_review']])
    require(keys == expected_keys and len(results) == 150 and sum(r['offered_requests'] for r in results) == 546,
            'Every offered training case and request must remain')
    sources = strict_references(sources)
    require(implementation_sources(__file__) == implementation, 'Actual-quality implementation changed')
    for ref in sources:
        require(receipt(ref['path']) == ref, 'Actual-quality evaluation source changed')
    return save(out/'report.json', dict(schema=SCHEMA,
        status='COMPLETE_ACTUAL_NATIVE_QUALITY_TRAINING_LABELS_NOT_QUALIFICATION',
        original_resource_rescore=report_ref, catalog=catalog_ref, cases=150, offered_requests=546,
        output_cases=results, sources=sources,
        known_actual_quality_cases=sum(checked(r['actual_quality_label'])['measured_Q'] is not None for r in results),
        full_six_term_label_cases=sum(checked(r['actual_six_score'])['full_U'] is not None for r in results),
        original_predictions_and_labels_rewritten=False,
        actual_outputs_used_as_predictor_inputs=False, UNKNOWN_world_mass_dropped=False,
        upstream_predictions_numerically_recomputed=False,
        upstream_timing_and_resource_labels_numerically_recomputed=False,
        trusted_upstream_scope='Persisted all-case report; this sidecar checks lineage and actual output quality only',
        quality_from_frozen_training_expectations_not_native_outcome=False,
        training_diagnostics_only=True, independent_prediction_accuracy_validated=False,
        original_stability_evaluated=False, final_gain_eligible=False,
        numerical_fit_run=False, GPU_started=False, validation_labels_read=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completed-rescore', required=True)
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(run(args.completed_rescore, args.catalog, args.output))
