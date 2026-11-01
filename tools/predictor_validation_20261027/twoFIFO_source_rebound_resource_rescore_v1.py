"""All-case rebound resource scoring and heldout measured score labels.

The response forecasts and their hashes stay fixed. Only the appropriate
two-block resource provider supplies predictions. Measured resources enter a
separate label artifact after the resource prediction has been persisted.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import twoFIFO_source_rebound_generated_v1 as source
from . import twoFIFO_source_rebound_resource_v1 as resource
from . import twoFIFO_source_rebound_observed_score_v1 as gold
from . import twoFIFO_fold_resource_batch_rescore_v1 as batch
from .implementation_source_closure_v27 import implementation_sources, strict_references

SCHEMA = 'gsdeploy.twoFIFO_source_rebound_allcase_resource_rescore.v1'
require, receipt, checked, save = source.require, source.receipt, source.checked, source.save


def _layout(report, plans):
    expected = {(p['capture_id'], p['attempt_id'], c['case_id'])
                for p in plans for c in p['cases']}
    require(len(expected) == 150 and sum(len(c['arrivals']) for p in plans for c in p['cases']) == 546,
            'Exact public 150-case/546-request design required')
    blocks = {p['capture_id'] for p in plans}
    folds = report['whole_block_leaveout']
    require(len(folds) == 3 and {f['heldout_block_id'] for f in folds} == blocks,
            'Every whole heldout block exactly once required')
    seen = set()
    for fold in folds:
        block = fold['heldout_block_id']
        require(set(fold['donor_attempt_ids']) == {p['attempt_id'] for p in plans if p['capture_id'] != block}
                and len(fold['donor_attempt_ids']) == 8
                and set(fold['heldout_attempt_ids']) == {p['attempt_id'] for p in plans if p['capture_id'] == block}
                and len(fold['heldout_attempt_ids']) == 4
                and set(fold['training_block_ids']) == blocks - {block} and len(fold['cases']) == 50,
                'Exact 8/4 whole-block donor separation required')
        for case in fold['cases']:
            key = (block, case['attempt_id'], case['case_id'])
            require(key in expected and key not in seen, 'Missing, duplicate or foreign full case key')
            seen.add(key)
    require(seen == expected, 'Every offered case must remain')


def score_errors(ensemble, native_score):
    value = batch.single.decode(ensemble)
    terms = {}
    for term in 'QTLPME':
        predicted = value['six_terms'][term]['expected_value']
        actual = native_score['terms'][term]
        terms[term] = dict(predicted_unconditional_expected_value=predicted,
            measured_value=actual, error=None if predicted is None or actual is None else predicted-actual,
            original_distribution=value['six_terms'][term])
    predicted = value['utility']['expected_value']
    actual = native_score['full_U']
    return batch.single.encode(dict(terms=terms,
        predicted_unconditional_expected_U=predicted, measured_U=actual,
        U_error=None if predicted is None or actual is None else predicted-actual,
        measured_unknown_reasons=native_score['unknown_reasons'],
        conditional_U_used_as_complete_U=False, UNKNOWN_mass_dropped=False,
        frozen_quality_used_on_both_sides=True, original_stability_evaluated=False))


def run_all(completed_report_path, resource_model_path, output, *, past_memory_state_paths=None,
            scenario='mean', include_measured_labels=True, generated_api=None, resource_api=None):
    # Explicit producer APIs permit the new recovery chain to use exactly the
    # same scoring mathematics. Their own audits enforce their distinct input
    # schemas; no native document is relabeled and no module global is patched.
    source = generated_api if generated_api is not None else globals()['source']
    resource = resource_api if resource_api is not None else globals()['resource']
    implementation_entries = (__file__, source.__file__, resource.__file__)
    implementation_start = implementation_sources(*implementation_entries)
    require(type(include_measured_labels) is bool, 'Explicit measured-label mode required')
    report_ref = receipt(completed_report_path)
    report = checked(report_ref)
    require(report.get('schema') == source.SCHEMA and report.get('status') in (
        'COMPLETE_REBOUND_CONDITIONAL_TRAINING_LEAVEOUT', 'COMPLETE_REBOUND_UNQUALIFIED_FIT_ALL_WORLDS_UNKNOWN')
        and report.get('response_fit_reused') is True and report.get('validation_labels_read') is False,
        'Completed new-schema generated training report required')
    for ref in report['sources']:
        source.response.verify(ref)
    design = source.audit(report['design']['path'])
    source.response.verify(report['design'])
    fit = source.response.audit_fit(report['response_fit'])['report']
    require(fit['manifest'] == design['fit_manifest'], 'Generated and response fit manifests differ')
    model_ref = receipt(resource_model_path)
    model = checked(model_ref)
    require(model.get('schema') == resource.SCHEMA
            and model.get('status') == getattr(resource, 'COMPLETE_STATUS',
                'COMPLETE_SOURCE_REBOUND_RESOURCE_TRAINING_NOT_QUALIFICATION')
            and model.get('resource_fit_run') is True and model.get('validation_labels_read') is False,
            'Complete source-rebound resource model required')
    resource_design = resource.audit(model['design']['path'])
    source.response.verify(model['design'])
    require(resource_design['fit_manifest'] == design['fit_manifest']
            and model['continuation'] == fit['continuation']
            and model['intake_evidence'] == fit['intake_evidence'],
            'Exact same response/resource continuation and authenticated intake required')
    for ref in model['sources']:
        source.response.verify(ref)
    fit_manifest = source.response.audit(design['fit_manifest']['path'])
    executor_refs = [fit_manifest[k] for k in ('executor_manifest', 'original_executor_manifest')
                     if k in fit_manifest]
    require(executor_refs and all(ref == executor_refs[0] for ref in executor_refs),
            'One unambiguous original executor manifest required')
    executor_manifest = checked(executor_refs[0])
    plan_refs = [item['plan'] for item in executor_manifest['attempts']]
    plans = [checked(ref) for ref in plan_refs]
    source.response.intake._attempt_layout(plans)
    _layout(report, plans)
    by_attempt = {p['attempt_id']: (ref, p) for ref, p in zip(plan_refs, plans)}
    state_paths = {} if past_memory_state_paths is None else past_memory_state_paths
    require(type(state_paths) is dict, 'Explicit past memory mapping required')
    counts = {}
    for plan in plans:
        for case in plan['cases']:
            counts[case['case_id']] = counts.get(case['case_id'], 0) + 1
    require(not any(n > 1 and key in state_paths for key, n in counts.items()),
            'Repeated case IDs require block|attempt|case memory key')
    score, quality = checked(design['scoring_plan']), checked(design['quality'])
    reviews = {item['attempt_id']: item['training_review'] for item in model['intake_evidence']['attempts']}
    require(set(reviews) == set(by_attempt), 'Exact twelve resource review bindings required')
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    case_reports, provider_bindings, raw_cache = [], {}, {}
    for fold in report['whole_block_leaveout']:
        block = fold['heldout_block_id']
        target = next(p for p in plans if p['capture_id'] == block)
        provider = resource.FoldResourceTrainingProvider(resource_model_path, target, checked(target['context']),
            heldout_block_id=block, allow_training_resource_hypothesis=True, scenario=scenario)
        provider_bindings[block] = provider.fold_binding
        law = checked(fold['folded_law']) if fold['folded_law'] is not None else None
        if law is not None:
            require(law['heldout_block_id'] == block and law['blocks'] == fold['training_block_ids'],
                    'Archived law uses foreign donor blocks')
        for case in fold['cases']:
            plan_ref, plan = by_attempt[case['attempt_id']]
            require(plan['capture_id'] == block and plan['attempt_id'] in provider.fold_binding['heldout_attempt_ids'],
                    'Per-case plan must belong to selected heldout fold')
            context = checked(plan['context'])
            compact = checked(case['prediction'])
            restored, requests = batch.single.restore_full_ensemble(compact, law, score, quality,
                donor_blocks=fold['training_block_ids'])
            start, end = map(batch.clock, restored['original_window'])
            drain = batch.clock(restored['original_drain_end_ns'])
            planned = next(c for c in plan['cases'] if c['case_id'] == case['case_id'])
            require(start.denominator == 1 and end-start == source.generated.MEASUREMENT_NS
                and drain-end == source.generated.DRAIN_NS
                and batch.canonical_requests(requests) == batch.canonical_requests(source.generated.public_case(planned, int(start))),
                'Archived case cohort/window changed')
            past_path = batch.resolve_past_memory_ref(state_paths, block_id=block, attempt_id=plan['attempt_id'],
                case_id=case['case_id'], case_id_count=counts[case['case_id']])
            prediction = batch._score_with_provider(restored, requests, provider, plan, context,
                heldout_block_id=block, scoring_plan=score, quality=quality,
                window_start_ns=int(start), window_end_ns=int(end), drain_end_ns=int(drain),
                past_memory_state=receipt(past_path) if past_path else None, scenario=scenario)
            folder = out / block / plan['attempt_id'] / ('case-' + str(len(case_reports) + 1))
            folder.mkdir(parents=True, exist_ok=False)
            prediction.update(target_plan=plan_ref, archived_prediction=case['prediction'])
            prediction_ref = save(folder / 'prediction.json', prediction)
            row = dict(block_id=block, attempt_id=plan['attempt_id'], case_id=case['case_id'],
                offered_requests=len(requests), prediction=prediction_ref, measured_label=None, score_errors=None)
            if include_measured_labels:
                # This path is deliberately after prediction persistence.
                if plan['attempt_id'] not in raw_cache:
                    review = checked(reviews[plan['attempt_id']])
                    raw_cache.clear()  # At most one large capture retained.
                    raw_cache[plan['attempt_id']] = (review['raw'], checked(review['raw']))
                raw_ref, raw = raw_cache[plan['attempt_id']]
                label = gold.score_case(raw, plan, planned, int(start), score, quality,
                    source_evidence=dict(raw=raw_ref, training_review=reviews[plan['attempt_id']],
                                         persisted_resource_prediction=prediction_ref))
                row['measured_label'] = save(folder / 'measured-label.json', label)
                row['score_errors'] = score_errors(prediction['ensemble'], label['measured_original_score'])
            case_reports.append(row)
    require(len(case_reports) == 150 and sum(r['offered_requests'] for r in case_reports) == 546,
            'All 150 cases and 546 requests must remain in final report')
    sources = strict_references([report_ref, report['sources'], model_ref, model['sources'], plan_refs,
        list(provider_bindings.values()), implementation_start])
    result = dict(schema=SCHEMA, status='COMPLETE_REBOUND_ALLCASE_RESOURCE_TRAINING_DIAGNOSTICS',
        generated_producer_schema=source.SCHEMA, resource_producer_schema=resource.SCHEMA,
        completed_generated_report=report_ref, resource_model=model_ref, cases=150, offered_requests=546,
        output_cases=case_reports, fold_provider_cache=provider_bindings, provider_loads=3,
        labels_generated=include_measured_labels,
        all_original_worlds_weights_and_offered_requests_retained=True,
        missing_memory_state_assumed_fresh=False, native_label_values_used_as_predictions=False,
        quality_from_frozen_training_expectations_not_native_outcome=True,
        original_six_terms_reused=True, original_stability_evaluated=False,
        numerical_fit_run=False, resource_fit_run=False, GPU_started=False, validation_labels_read=False,
        independent_prediction_accuracy_validated=False, final_gain_eligible=False, sources=sources)
    source.audit(report['design']['path'])
    resource.audit(model['design']['path'])
    require(implementation_sources(*implementation_entries) == implementation_start,
            'Resource rescore or explicit producer implementation changed during run')
    for ref in sources:
        source.response.verify(ref)
    return save(out / 'report.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completed-report', required=True)
    parser.add_argument('--resource-model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scenario', choices=['mean', 'max'], default='mean')
    parser.add_argument('--skip-measured-labels', action='store_true')
    args = parser.parse_args()
    print(run_all(args.completed_report, args.resource_model, args.output,
        scenario=args.scenario, include_measured_labels=not args.skip_measured_labels))
