"""All171case matching resource scoring and separate measured score labels.

The response forecasts and their hashes stay fixed. Only the appropriate
two-block resource provider supplies predictions. Measured resources enter a
separate label artifact after the resource prediction has been persisted.
Derived explicitly from the retained allcase and world-score companions. Only
their generic contracts, replay and original scoring/distribution math are
reused; matching physical identity and predict_timeline are called directly.
"""
from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import math
from .actual_worker_twoFIFO_query_adapter_v5_2 import query_bound_view,clock as public_clock
from .actual_worker_twoFIFO_reference_v5_2 import resource_timeline_view
from .macro_resource_timeline_v1 import timeline
from .macro_response_queue_reference_v1 import encode
from .macro_six_term_bridge_v1 import whole_score
from pathlib import Path

from . import twoFIFO_matching_generated_v1 as source
from . import twoFIFO_matching_resource_v1 as resource
from . import twoFIFO_fold_resource_batch_rescore_v1 as batch
from .implementation_source_closure_v27 import implementation_sources, strict_references

SCHEMA = 'gsdeploy.twoFIFO_matching_allcase_resource_rescore.v1'
require, receipt, checked, save = source.require, source.receipt, source.checked, source.save
clock=public_clock


def _layout(report, plans):
    expected = {(p['capture_id'], p['attempt_id'], c['case_id'])
                for p in plans for c in p['cases']}
    require(len(expected) == 171 and sum(len(c['arrivals']) for p in plans for c in p['cases']) == 390,
            'Exact public 171-case/390-request design required')
    blocks = {p['capture_id'] for p in plans}
    folds = report['whole_block_leaveout']
    require(len(folds) == 3 and {f['heldout_block_id'] for f in folds} == blocks,
            'Every whole heldout block exactly once required')
    seen = set()
    for fold in folds:
        block = fold['heldout_block_id']
        require(set(fold['donor_attempt_ids']) == {p['attempt_id'] for p in plans if p['capture_id'] != block}
                and len(fold['donor_attempt_ids']) == 10
                and set(fold['heldout_attempt_ids']) == {p['attempt_id'] for p in plans if p['capture_id'] == block}
                and len(fold['heldout_attempt_ids']) == 5
                and set(fold['training_block_ids']) == blocks - {block} and len(fold['cases']) == 57,
                'Exact 10/5 whole-block donor separation required')
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



def score_world(reference, public_requests, scoring_plan, quality, *,
                window_start_ns, window_end_ns, drain_end_ns,
                resource_provider=None, target_context=None, resource_source_evidence=None,
                resource_memory_initial_state=None):
    """Score the complete original drain world, retaining unsupported terms.

    The provider must authenticate its own training sources; supplied evidence
    is a caller declaration and does not independently certify native identity.
    Only public requests, explicit generated state and training providers enter.
    """
    bound=query_bound_view(reference,public_requests,window_start_ns=window_start_ns,
        window_end_ns=window_end_ns,drain_end_ns=drain_end_ns)
    start,end=map(clock,bound['original_window']);drain=clock(bound['original_drain_end_ns'])
    if clock(bound['cutoff_ns'])!=drain:
        raise ValueError('Whole-world integration requires the original complete drain cutoff')
    if any(v.denominator!=1 for v in (start,end,drain)):
        raise ValueError('Original integer scoring boundaries required')
    board=timeline(resource_timeline_view(bound),public_requests,
        resident_owners=tuple(scoring_plan['aliases']))
    estimates=resources=None
    if resource_provider is not None:
        if target_context is None or not resource_source_evidence:
            raise ValueError('Explicit resource execution context and source evidence required')
        binding=dict(basis='caller_declared_explicit_reference_execution_binding',
            physical_execution_identity=resource_provider.physical_execution_identity,
            resource_timeline_sha256=resource.digest(board))
        if resource_memory_initial_state is not None:
            binding['memory_initial_state']=copy.deepcopy(resource_memory_initial_state)
        estimates=resource.predict_timeline(board,resource_provider,target_context=target_context,
            execution_binding=binding)
        # Keep startup and the phase-start-to-drain peak separate. The unchanged
        # scorer alone takes their maximum and applies its safety factor once.
        future_peak=(estimates['partial_known_peak_memory_mib']
            if public_clock(estimates['memory_unknown_observation_duration_ns'])==0 else None)
        resources=dict(original_window_ns=[int(start),int(end)],original_drain_end_ns=int(drain),
            resident_owners=list(scoring_plan['aliases']),
            startup_energy_joules=estimates['startup_energy_joules'],
            startup_through_phase_start_owned_peak_mib=estimates['premeasurement_owned_sampled_peak_mib'],
            phase_start_through_drain_owned_peak_mib=future_peak,
            measurement_board_energy_joules=estimates['board_energy_joules'],
            measurement_energy_includes_drain=False,memory_safety_factor_already_applied=False,
            startup_energy_already_amortized=False,source_evidence=copy.deepcopy(resource_source_evidence))
    elif (target_context is not None or resource_source_evidence is not None
          or resource_memory_initial_state is not None):
        raise ValueError('Resource context/evidence cannot imply an absent provider')
    scored=whole_score(bound,public_requests,scoring_plan,quality,resources)
    return dict(schema=SCHEMA+'.world_score',
        query_fingerprint_sha256=bound['query_fingerprint_sha256'],
        public_query_identity_verified=True,resource_timeline=board,
        resource_prediction=estimates,original_score_bridge=scored,
        original_score=copy.deepcopy(scored['score']),
        resource_execution_binding_is_caller_declared=True,
        no_global_request_admission_cap=bound['global_admission_cap'] is None,
        actual_READY_releases_worker=True,RESULT_reader_is_E2E_endpoint=True,
        original_stability_evaluated=False,main_queue_provider_eligible=False,
        independent_prediction_accuracy_validated=False,final_gain_eligible=False,
        validation_labels_read=False,GPU_started=False)


def score_with_provider(document, requests, provider, target_plan, target_context, *, heldout_block_id,
                         scoring_plan, quality, window_start_ns, window_end_ns, drain_end_ns,
                         past_memory_state=None, scenario='mean'):
    """Same six-term bridge as v1, with an already authenticated provider."""
    requests = [dict(row) for row in requests]
    document = copy.deepcopy(document)
    start, end, drain = map(clock, (window_start_ns, window_end_ns, drain_end_ns))
    require(start < end <= drain and all(v.denominator == 1 for v in (start, end, drain)),
            'Original integer scoring boundaries required')
    batch.single._score_contract(document, requests, scoring_plan, quality, start, end, drain)
    binding = provider.fold_binding
    require(isinstance(provider,resource.FoldResourceTrainingProvider)
            and provider.target_plan['capture_id'] == heldout_block_id
            and target_plan['capture_id']==heldout_block_id
            and target_plan['attempt_id'] in binding['heldout_attempt_ids']
            and binding['heldout_block_id']==heldout_block_id
            and len(binding['training_block_ids'])==2
            and heldout_block_id not in binding['training_block_ids']
            and len(binding['training_attempt_ids'])==10 and len(set(binding['training_attempt_ids']))==10
            and len(binding['heldout_attempt_ids'])==5 and len(set(binding['heldout_attempt_ids']))==5
            and not set(binding['training_attempt_ids'])&set(binding['heldout_attempt_ids'])
            and target_plan['resident_owners'] == provider.target_plan['resident_owners']
            and target_context['residents'] == provider.target_context['residents']
            and resource.physical_execution_identity(target_plan, target_context)
                == provider.physical_execution_identity,
            'Cached fold provider physical target differs')
    require(set(scoring_plan['aliases'])==set(target_plan['resident_owners'])==set(target_context['residents'])
            and all(scoring_plan['aliases'][owner]['family']==target_context['residents'][owner]
                for owner in scoring_plan['aliases']), 'Original scoring resident/family roster differs')
    require(all(row['block_id'] in binding['training_block_ids'] for row in document['draws']),
            'Every generated donor block must belong to the resource training fold')
    state, state_sources = batch.single.past_memory(past_memory_state, start)
    model_ref = binding['full_native_model']
    evidence = dict(schema=SCHEMA + '.explicit_source_evidence', resource_model=model_ref,
        selected_fold_model_sha256=binding['selected_fold_model_sha256'], heldout_block_id=heldout_block_id,
        training_attempt_ids=binding['training_attempt_ids'], scenario=scenario,
        memory_state=past_memory_state, memory_state_derivation_is_caller_declared=True)
    rows = []
    for original in document['draws']:
        row = copy.deepcopy(original)
        if row['status'] == 'GENERATED_WORLD':
            world = row['forecast']['world']; world_sha = resource.digest(world)
            scored = score_world(world, requests, scoring_plan, quality,
                window_start_ns=int(start), window_end_ns=int(end), drain_end_ns=int(drain),
                resource_provider=provider, target_context=target_context,
                resource_source_evidence=evidence, resource_memory_initial_state=state)
            require(scored['query_fingerprint_sha256'] == row['forecast']['query_fingerprint_sha256'],
                    'Resource batch changed generated query identity')
            scored['resource_fold_binding'] = copy.deepcopy(binding)
            row['forecast']['scoring'] = scored; row['source_world_sha256'] = world_sha
            require(resource.digest(row['forecast']['world']) == world_sha, 'Resource batch changed generated world')
        rows.append(row)
    values = batch.single.decode(rows)
    result = copy.deepcopy(document)
    result.update(draws=rows,
        six_terms={term: batch.ensemble.distribution(values,
            lambda f, t=term: f['scoring']['original_score']['terms'][t]) for term in 'QTLPME'},
        utility=batch.ensemble.distribution(values, lambda f: f['scoring']['original_score']['full_U']))
    for source in state_sources + binding['sources']:
        require(receipt(source['path']) == source, 'Resource batch source drift')
    return dict(schema=SCHEMA, status='COMPLETE_FOLD_RESOURCE_RESCORE_TRAINING_HYPOTHESIS_ONLY',
        ensemble=result, resource_model=dict(path=model_ref['path'], sha256=model_ref['sha256']),
        resource_fold_binding=binding, resource_scenario=scenario,
        past_memory_state=past_memory_state, past_memory_sources=state_sources,
        missing_memory_state_assumed_fresh=False, resource_memory_provider_replaced=False,
        generated_worlds_and_public_cohort_unchanged=True, all_original_draws_and_probability_mass_retained=True,
        original_six_score_reused=True, partial_U_reported=False,
        energy_scope='original_measurement_only_startup_once_at_3600_seconds',
        memory_scope='startup_and_explicit_past_through_original_drain_owned_sampled_peak',
        memory_state_derivation_is_caller_declared=True, case_memory_candidate_merge_enabled=False,
        actual_native_quality_validated=False,
        quality_from_frozen_training_expectations_not_native_outcome=True,
        resource_prediction_qualified=False, main_queue_provider_eligible=False,
        final_gain_eligible=False, independent_prediction_accuracy_validated=False,
        full_U_resource_qualification=False, validation_labels_read=False, GPU_started=False)



def measured_score_case(raw, plan, case, epoch, scoring_plan, quality, *, source_evidence):
    """Measure fixed-window E and startup-through-fixed-drain sampled M.

    These labels may use the heldout future. They must never be passed to the
    folded provider, native seed, donor law or candidate-selection procedure.
    """
    require(raw.get('schema') == source.response.executor.RAW_SCHEMA and raw.get('backend') == 'native'
            and raw.get('status') == 'completed' and raw.get('split')=='training',
            'Complete matching native training raw required')
    require(all(raw.get(k) == plan[k] for k in ('capture_id', 'attempt_id', 'execution_id')),
            'Exact raw/plan identity required')
    require(type(source_evidence) is dict and source_evidence, 'Explicit native label provenance required')
    planned = [c for c in plan['cases'] if c['case_id'] == case['case_id']]
    require(len(planned) == 1 and planned[0] == case, 'Exact public case required')
    native_case = [c for c in raw['cases'] if c['case_id'] == case['case_id']]
    require(len(native_case) == 1 and native_case[0]['observation_start_ns'] == epoch,
            'Original native case epoch required')
    measured = resource.observed.observed(raw)
    rows = [r for r in measured['cases'] if r['case_id'] == case['case_id']]
    require(len(rows) == 1, 'Exact measured case resource row required')
    row = rows[0]
    start, end, drain = epoch, epoch + source.generated.MEASUREMENT_NS, epoch + source.generated.MEASUREMENT_NS + source.generated.DRAIN_NS
    require([row[k] for k in ('measurement_start_ns', 'measurement_end_ns', 'fixed_drain_deadline_ns')]
            == [start, end, drain], 'Original fixed measurement and drain boundaries required')
    require(set(scoring_plan['aliases']) == set(plan['resident_owners']), 'All resident owners must be scored')
    startup = measured['startup']['start_ns']
    before = [s['owned_used_mib'] for s in raw['board_samples'] if startup <= s['time_ns'] <= start]
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in before),
            'Finite nonnegative sampled owned memory required')
    # Existing measured adapter uses all inactive residents and excludes worker
    # close/shutdown. Keep prior history separate for the original scorer.
    resources = dict(original_window_ns=[start, end], original_drain_end_ns=drain,
        resident_owners=list(plan['resident_owners']),
        startup_energy_joules=measured['startup']['energy_joules'],
        startup_through_phase_start_owned_peak_mib=max(before) if before else None,
        phase_start_through_drain_owned_peak_mib=row['local_measurement_through_drain_owned_sampled_peak_mib'],
        measurement_board_energy_joules=row['measurement_energy_joules'],
        measurement_energy_includes_drain=False, memory_safety_factor_already_applied=False,
        startup_energy_already_amortized=False, source_evidence=copy.deepcopy(source_evidence))
    labels = source.generated.native_labels(raw, case, epoch, scoring_plan, quality)
    requests = source.generated.public_case(case, epoch)
    arrivals = {r['request_id']: r['arrival_ns'] for r in requests}
    receipts = {r['request_id']: Fraction(arrivals[r['request_id']] + r['reader_E2E_ns'])
                for r in labels['requests'] if r['completed_within_fixed_drain']}
    frame = dict(original_window=encode([Fraction(start), Fraction(end)]),
        original_drain_end_ns=encode(Fraction(drain)), cutoff_ns=encode(Fraction(drain)),
        RESULT_receipts=encode(receipts), offered_requests=len(requests),
        unfinished_request_ids=[r['request_id'] for r in requests if r['request_id'] not in receipts])
    scored = whole_score(frame, requests, scoring_plan, quality, resources)
    require(all(scored['score']['terms'][k] == labels['original_score']['terms'][k] for k in 'QTLP'),
            'Observed resources must not change original Q/T/L/P')
    return dict(schema=SCHEMA, status='MEASURED_TRAINING_SCORE_LABEL_NOT_PREDICTION',
        case_id=case['case_id'], attempt_id=plan['attempt_id'], block_id=plan['capture_id'],
        reader_labels=labels, measured_resources=resources, measured_original_score=scored['score'],
        original_score_bridge=scored, observed_case_resources=row, offered_requests=len(requests),
        complete_offered_cohort_preserved=True, heldout_future_used_for_evaluation_only=True,
        quality_from_frozen_training_expectations_not_native_outcome=True,
        actual_native_quality_validated=False, sampled_memory_is_continuous_capacity_bound=False,
        original_stability_evaluated=False, predictor_accuracy_qualified=False,
        native_evidence_authenticated_by_this_pure_helper=False,
        final_gain_eligible=False, validation_labels_read=False, GPU_started=False)


def run_all(completed_report_path, resource_model_path, output, *, past_memory_state_paths=None,
            scenario='mean', include_measured_labels=True):
    implementation_entries = (__file__, source.__file__, resource.__file__)
    implementation_start = implementation_sources(*implementation_entries)
    require(type(include_measured_labels) is bool, 'Explicit measured-label mode required')
    report_ref = receipt(completed_report_path)
    report = checked(report_ref)
    require(report.get('schema') == source.SCHEMA and report.get('status') in (
        'COMPLETE_MATCHING_CONDITIONAL_TRAINING_LEAVEOUT', 'COMPLETE_MATCHING_UNQUALIFIED_FIT_ALL_WORLDS_UNKNOWN')
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
                'COMPLETE_MATCHING_RESOURCE_TRAINING_NOT_QUALIFICATION')
            and model.get('resource_fit_run') is True and model.get('validation_labels_read') is False,
            'Complete matching resource model required')
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
    require(set(reviews) == set(by_attempt), 'Exact fifteen resource review bindings required')
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
            prediction = score_with_provider(restored, requests, provider, plan, context,
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
                label = measured_score_case(raw, plan, planned, int(start), score, quality,
                    source_evidence=dict(raw=raw_ref, training_review=reviews[plan['attempt_id']],
                                         persisted_resource_prediction=prediction_ref))
                row['measured_label'] = save(folder / 'measured-label.json', label)
                row['score_errors'] = score_errors(prediction['ensemble'], label['measured_original_score'])
            case_reports.append(row)
    require(len(case_reports) == 171 and sum(r['offered_requests'] for r in case_reports) == 390,
            'All171 cases and390 requests must remain in final report')
    sources = strict_references([report_ref, report['sources'], model_ref, model['sources'], plan_refs,
        list(provider_bindings.values()), implementation_start])
    result = dict(schema=SCHEMA, status='COMPLETE_MATCHING_ALLCASE_RESOURCE_TRAINING_DIAGNOSTICS',
        generated_producer_schema=source.SCHEMA, resource_producer_schema=resource.SCHEMA,
        completed_generated_report=report_ref, resource_model=model_ref, cases=171, offered_requests=390,
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
