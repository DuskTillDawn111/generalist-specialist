"""Predeclared matching training leaveout, with complete-cohort queue bounds.

This is training diagnostics, not prospective independent validation. It reads
all training labels during intake/fit; each folded forecast uses only donors,
the public calendar, and the immutable pre-case history. Predictions are saved
before extracting that case's evaluation labels. No GPU work is started here.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

from . import twoFIFO_matching_phase12_v1 as response
from . import twoFIFO_matching_joint_law_v1 as laws
from . import twoFIFO_generated_wholeblock_evaluation_v1 as generated
from . import twoFIFO_queue_envelope_companion_v1 as envelope
from .implementation_source_closure_v27 import implementation_sources, strict_references

SCHEMA = 'gsdeploy.twoFIFO_matching_generated_training.v1'
require = response.require
receipt = response.receipt
checked = response.checked
save = generated.save


def prepare(fit_manifest_path, scoring_plan_path, quality_path, output):
    manifest = response.audit(fit_manifest_path)
    require(not (Path(fit_manifest_path).resolve().parent / 'phase12-fit-claim.json').exists(),
            'Generated draw and score design must be frozen before the one permitted fit')
    score_ref, quality_ref = receipt(scoring_plan_path), receipt(quality_path)
    score, quality = checked(score_ref), checked(quality_ref)
    # Reuse exactly the established six-term contract and unchanged draw scheme.
    generated.whole_score(dict(original_window=generated.encode([Fraction(0), Fraction(generated.MEASUREMENT_NS)]),
        original_drain_end_ns=generated.encode(Fraction(generated.MEASUREMENT_NS + generated.DRAIN_NS)),
        cutoff_ns=generated.encode(Fraction(generated.MEASUREMENT_NS + generated.DRAIN_NS)),
        RESULT_receipts={}, unfinished_request_ids=[], offered_requests=0), [], score, quality)
    sources = strict_references([receipt(fit_manifest_path), manifest['sources'], score_ref, quality_ref,
                                 implementation_sources(__file__)])
    for source in sources:
        response.verify(source)
    design = dict(schema=SCHEMA + '.design', status='CPU_PREDECLARED_BEFORE_MATCHING_FIT',
        fit_manifest=receipt(fit_manifest_path), scoring_plan=score_ref, quality=quality_ref,
        scheme=generated.SCHEME, scheme_sha256=generated.digest(generated.SCHEME),
        heldout_unit='whole_capture_block_all_five_shards', expected_cases=171,
        expected_measured_requests=390, expected_priming_history_requests=270,
        queue_envelope_attached=True, queue_bounds_used_for_utility=False,
        six_term_weights='original_equal_QTLPME', validation_labels_read=False,
        numerical_fit_run=False, GPU_started=False, sources=sources)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    return save(out / 'design.json', design)


def audit(path):
    design = checked(receipt(path))
    require(design.get('schema') == SCHEMA + '.design'
        and design.get('status') == 'CPU_PREDECLARED_BEFORE_MATCHING_FIT'
        and design.get('scheme') == generated.SCHEME
        and design.get('scheme_sha256') == generated.digest(generated.SCHEME)
        and design.get('heldout_unit') == 'whole_capture_block_all_five_shards'
        and design.get('expected_cases') == 171 and design.get('expected_measured_requests') == 390
        and design.get('expected_priming_history_requests') == 270
        and design.get('queue_envelope_attached') is True
        and design.get('queue_bounds_used_for_utility') is False
        and design.get('six_term_weights') == 'original_equal_QTLPME'
        and design.get('validation_labels_read') is False
        and design.get('numerical_fit_run') is False and design.get('GPU_started') is False,
        'Exact matching predeclared draw, cohort and score contract required')
    manifest = response.audit(design['fit_manifest']['path'])
    expected = strict_references([design['fit_manifest'], manifest['sources'], design['scoring_plan'],
                                  design['quality'], implementation_sources(__file__)])
    require(design['sources'] == expected, 'Generated design implementation closure changed')
    for source in expected:
        response.verify(source)
    return design


def evaluate_case(raw, case, epoch, snapshot, law, scoring_plan, quality, folder, *, donor_blocks,
                  unqualified_reason=None):
    """Forecast and bound the same draws; persist before evaluation extraction."""
    paired = envelope.forecast_case(case, epoch, snapshot, law, scoring_plan, quality,
        donor_blocks=donor_blocks, unqualified_reason=unqualified_reason)
    compact = generated.compact_prediction(paired['prediction'])
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    prediction_ref = save(folder / 'prediction.json', compact)
    bounds_ref = save(folder / 'queue-envelope.json', paired['queue_envelope'])
    labels = generated.native_labels(raw, case, epoch, scoring_plan, quality)
    labels_ref = save(folder / 'labels.json', labels)
    value = compact['ensemble']
    return dict(case_id=case['case_id'], prediction=prediction_ref, queue_envelope=bounds_ref,
        labels=labels_ref, errors=generated.errors(compact, labels),
        ensemble_summary=dict(worlds=len(value['draws']),
            generated_worlds=sum(d['status'] == 'GENERATED_WORLD' for d in value['draws']),
            unknown_worlds=sum(d['status'] == 'UNKNOWN_WORLD' for d in value['draws']),
            unknown_probability_mass=value['world_unknown_probability_mass']),
        durable_before_case_evaluation_label_extraction=True,
        conditional_training_leaveout_not_independent_validation=True)


def evaluate_bundle(bundle, fitted, scoring_plan, quality, output):
    """Pure orchestration boundary; caller authenticates bundle and fit first."""
    response._validate_bundle(bundle)
    require(fitted.get('denominator') == bundle['denominator']
            and fitted.get('features') == list(response.FEATURES), 'Fit/bundle mismatch')
    plans, docs = bundle['plans'], bundle['documents']
    reviews = {}
    for ref in bundle['review_receipts']:
        review = checked(ref)
        plan = checked(review['inputs']['plan.json'])
        require(plan['attempt_id'] not in reviews, 'Duplicate reviewed attempt')
        reviews[plan['attempt_id']] = review
    require(set(reviews) == {p['attempt_id'] for p in plans}, 'Exact fifteen reviewed raw inputs required')
    raws = [checked(reviews[p['attempt_id']]['raw']) for p in plans]
    blocks = sorted({p['capture_id'] for p in plans})
    eligible = fitted.get('response_law_compilation_eligible') is True
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    folds = []
    full_ref = None
    if eligible:
        full_ref = save(output / 'full-joint-law.json', laws.build_joint_law(docs, plans, raws, fitted))
    for heldout in blocks:
        donors = [b for b in blocks if b != heldout]
        law = laws.build_joint_law(docs, plans, raws, fitted, heldout_block_id=heldout) if eligible else None
        law_ref = save(output / (heldout + '-folded-law.json'), law) if law is not None else None
        cases = []
        for plan, raw in zip(plans, raws):
            if plan['capture_id'] != heldout:
                continue
            by_case = {c['case_id']: c for c in raw['cases']}
            snapshots = {h['case_id']: h['prior_history_snapshot'] for h in raw['case_histories']}
            for index, case in enumerate(plan['cases']):
                epoch = by_case[case['case_id']]['observation_start_ns']
                row = evaluate_case(raw, case, epoch, snapshots.get(case['case_id']), law, scoring_plan,
                    quality, output / plan['attempt_id'] / ('case-' + str(index + 1)), donor_blocks=donors,
                    unqualified_reason=None if eligible else 'UNKNOWN_COMPLETE_RESPONSE_FIT_OR_OCR_SUPPORT_UNQUALIFIED')
                row['attempt_id'] = plan['attempt_id']
                cases.append(row)
        require(len(cases) == 57 and sum(c['errors']['offered_requests'] for c in cases) == 130,
                'Whole heldout block must retain 57 cases and 130 requests')
        folds.append(dict(heldout_block_id=heldout, training_block_ids=donors,
            heldout_attempt_ids=[p['attempt_id'] for p in plans if p['capture_id'] == heldout],
            donor_attempt_ids=[p['attempt_id'] for p in plans if p['capture_id'] != heldout],
            folded_law=law_ref, cases=cases, summary=generated.fold_summary(cases)))
    rows = [c for fold in folds for c in fold['cases']]
    require(len(rows) == 171 and sum(c['errors']['offered_requests'] for c in rows) == 390,
            'Full 171-case/390-request denominator required')
    return dict(schema=SCHEMA,
        status='COMPLETE_MATCHING_CONDITIONAL_TRAINING_LEAVEOUT' if eligible else
               'COMPLETE_MATCHING_UNQUALIFIED_FIT_ALL_WORLDS_UNKNOWN',
        fit=fitted, full_law=full_ref, whole_block_leaveout=folds, forecasts_generated=171,
        ensemble_worlds_generated=sum(c['ensemble_summary']['generated_worlds'] for c in rows),
        ensemble_worlds_unknown=sum(c['ensemble_summary']['unknown_worlds'] for c in rows),
        summary=generated.fold_summary(rows), numerical_fit_run=False, response_fit_reused=True,
        validation_labels_read=False,
        heldout_future_service_labels_used_for_generation=False,
        intake_and_fit_have_read_training_holdout_labels=True,
        startup_and_intercase_orchestration_predicted=False,
        all_offered_requests_and_original_probability_mass_retained=True,
        queue_bounds_used_for_utility=False, calibrated_risk=False, full_U_resource_qualification=False,
        actual_native_quality_validated=False, independent_prediction_accuracy_validated=False,
        main_queue_provider_eligible=False, final_gain_eligible=False, GPU_started=False)


def run(design_path, fit_report_path, output):
    """Reuse one saved fit after rechecking the exact source-bound intake."""
    design_ref = receipt(design_path)
    design = audit(design_path)
    fit_audit = response.audit_fit(fit_report_path)
    fit_report = fit_audit['report']
    require(fit_report['manifest'] == design['fit_manifest'],
            'Saved fit belongs to a different predeclared design')
    score, quality = checked(design['scoring_plan']), checked(design['quality'])

    def continue_bundle(bundle, intake_document):
        require(intake_document == fit_report['intake_evidence'],
                'Saved fit intake did not rederive from this exact continuation')
        result = evaluate_bundle(bundle, fit_report['fit'], score, quality, output)
        result.update(design=design_ref, response_fit=fit_audit['receipt'],
            sources=strict_references([design_ref, design['sources'], fit_audit['receipt'],
                fit_report['sources'], intake_document['sources'], implementation_sources(__file__)]))
        return result

    result = response.with_verified_bundle(design['fit_manifest']['path'],
        fit_report['continuation']['path'], continue_bundle)
    audit(design_path)
    response.verify(design_ref)
    response.verify(fit_audit['receipt'])
    for source in result['sources']:
        response.verify(source)
    return save(Path(output) / 'report.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--fit-manifest', required=True)
    prep.add_argument('--scoring-plan', required=True)
    prep.add_argument('--quality', required=True)
    prep.add_argument('--output', required=True)
    execute = sub.add_parser('run')
    execute.add_argument('--design', required=True)
    execute.add_argument('--fit-report', required=True)
    execute.add_argument('--output', required=True)
    args = parser.parse_args()
    print(prepare(args.fit_manifest, args.scoring_plan, args.quality, args.output)
        if args.command == 'prepare' else run(args.design, args.fit_report, args.output))
