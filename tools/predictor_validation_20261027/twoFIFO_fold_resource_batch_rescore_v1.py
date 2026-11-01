"""Batch continuation for one completed generated 150-case report.

This companion keeps the single-case v1 contract while caching one verified
eight-donor FoldResourceTrainingProvider per held-out block. It never opens
case labels, fits, collects, starts a GPU, or mutates the frozen runner.
"""
from __future__ import annotations

import copy
from pathlib import Path

from . import twoFIFO_fold_resource_rescore_v1 as single
from . import twoFIFO_resource_fold_provider_v1 as folds
from . import twoFIFO_forecast_ensemble_v1 as ensemble
from . import twoFIFO_generated_wholeblock_evaluation_v1 as generated
from .actual_worker_world_score_v3 import score_world
from .actual_worker_twoFIFO_reference_v5_2 import decode
from .actual_worker_twoFIFO_query_adapter_v5_2 import canonical_requests, clock
from .macro_response_queue_reference_v1 import encode
from .implementation_source_closure_v27 import implementation_sources, strict_references

SCHEMA = 'gsdeploy.twoFIFO_fold_resource_batch_rescore.v1'
require = single.require
receipt = single.receipt
checked = single.checked
save = single.save
digest = single.digest


def _score_with_provider(document, requests, provider, target_plan, target_context, *, heldout_block_id,
                         scoring_plan, quality, window_start_ns, window_end_ns, drain_end_ns,
                         past_memory_state=None, scenario='mean'):
    """Same six-term bridge as v1, with an already authenticated provider."""
    requests = [dict(row) for row in requests]
    document = copy.deepcopy(document)
    start, end, drain = map(clock, (window_start_ns, window_end_ns, drain_end_ns))
    require(start < end <= drain and all(v.denominator == 1 for v in (start, end, drain)),
            'Original integer scoring boundaries required')
    single._score_contract(document, requests, scoring_plan, quality, start, end, drain)
    binding = provider.fold_binding
    require(provider.target_plan['capture_id'] == heldout_block_id
            and target_plan['resident_owners'] == provider.target_plan['resident_owners']
            and target_context['residents'] == provider.target_context['residents']
            and folds.resource.physical_execution_identity(target_plan, target_context)
                == provider.physical_execution_identity,
            'Cached fold provider physical target differs')
    require(all(row['block_id'] in binding['training_block_ids'] for row in document['draws']),
            'Every generated donor block must belong to the resource training fold')
    state, state_sources = single.past_memory(past_memory_state, start)
    model_ref = binding['full_native_model']
    evidence = dict(schema=single.SCHEMA + '.explicit_source_evidence', resource_model=model_ref,
        selected_fold_model_sha256=binding['selected_fold_model_sha256'], heldout_block_id=heldout_block_id,
        training_attempt_ids=binding['training_attempt_ids'], scenario=scenario,
        memory_state=past_memory_state, memory_state_derivation_is_caller_declared=True)
    rows = []
    for original in document['draws']:
        row = copy.deepcopy(original)
        if row['status'] == 'GENERATED_WORLD':
            world = row['forecast']['world']; world_sha = digest(world)
            scored = score_world(world, requests, scoring_plan, quality,
                window_start_ns=int(start), window_end_ns=int(end), drain_end_ns=int(drain),
                resource_provider=provider, target_context=target_context,
                resource_source_evidence=evidence, resource_memory_initial_state=state)
            require(scored['query_fingerprint_sha256'] == row['forecast']['query_fingerprint_sha256'],
                    'Resource batch changed generated query identity')
            scored['resource_fold_binding'] = copy.deepcopy(binding)
            row['forecast']['scoring'] = scored; row['source_world_sha256'] = world_sha
            require(digest(row['forecast']['world']) == world_sha, 'Resource batch changed generated world')
        rows.append(row)
    values = decode(rows)
    result = copy.deepcopy(document)
    result.update(draws=rows,
        six_terms={term: ensemble.distribution(values,
            lambda f, t=term: f['scoring']['original_score']['terms'][t]) for term in 'QTLPME'},
        utility=ensemble.distribution(values, lambda f: f['scoring']['original_score']['full_U']))
    for source in state_sources + binding['sources']:
        require(receipt(source['path']) == source, 'Resource batch source drift')
    return dict(schema=single.SCHEMA, status='COMPLETE_FOLD_RESOURCE_RESCORE_TRAINING_HYPOTHESIS_ONLY',
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


def build_fold_provider_cache(resource_model_path, plans, fold_rows, *, scenario='mean'):
    """Authenticate exactly one cached provider for each held-out block."""
    by_block = {}
    by_attempt = {p['attempt_id']: p for p in plans}
    for fold in fold_rows:
        block = fold['heldout_block_id']
        heldout = [p for p in plans if p['capture_id'] == block]
        require(len(heldout) == 4, 'Each heldout block must have four attempts')
        donor_ids = fold.get('donor_attempt_ids')
        require(type(donor_ids) is list and len(donor_ids) == 8
            and len(fold['heldout_attempt_ids']) == 4
            and set(donor_ids) == {p['attempt_id'] for p in plans if p['capture_id'] != block}
            and set(fold['heldout_attempt_ids']) == {p['attempt_id'] for p in heldout},
            'Fold donor/heldout attempts differ from campaign')
        target = heldout[0]; context = checked(target['context'])
        provider = folds.FoldResourceTrainingProvider(resource_model_path, target, context,
            heldout_block_id=block, allow_training_resource_hypothesis=True, scenario=scenario)
        by_block[block] = (provider, target, context, by_attempt[target['attempt_id']])
    require(len(by_block) == 3, 'Complete three-fold provider cache required')
    return by_block


def resolve_past_memory_ref(state_paths, *, block_id, attempt_id, case_id, case_id_count):
    """Resolve only an unambiguous full case key, never cross-fold reuse."""
    require(type(state_paths) is dict, 'Past memory states must be an explicit case mapping')
    require(case_id_count == 1 or case_id not in state_paths,
            'Ambiguous bare case_id memory mapping; use block/attempt/case key')
    return state_paths.get(block_id + '|' + attempt_id + '|' + case_id,
        state_paths.get(case_id) if case_id_count == 1 else None)


def run_all_completed_cases(completed_report_path, resource_model_path, *, output,
                            scenario='mean', past_memory_state_paths=None):
    """Rescore every archived case, loading each held-out fold exactly once."""
    report_ref = receipt(completed_report_path); report = checked(report_ref)
    require(report.get('schema') == generated.SCHEMA and report.get('numerical_fit_run') is True
        and report.get('status') in ('COMPLETE_CONDITIONAL_GENERATED_WHOLEBLOCK_TRAINING_EVALUATION',
            'COMPLETE_NATIVE_TRAINING_FIT_UNQUALIFIED_ALL_GENERATED_WORLDS_UNKNOWN')
        and report.get('validation_labels_read') is False, 'Complete generated training report required')
    for source in report['sources']: require(receipt(source['path']) == source, 'Completed source drift')
    _, design = generated.audit_design(report['design']['path'])
    require(receipt(report['design']['path']) == report['design'], 'Completed design drift')
    model_ref = receipt(resource_model_path); model_doc = checked(model_ref)
    require(model_doc.get('schema') == folds.resource.SCHEMA and model_doc.get('campaign') == design['campaign'],
            'Resource and generated campaign receipts must match exactly')
    reviews = model_doc.get('training_reviews')
    require(type(reviews) is list and len(reviews) == 12 and len({r['path'] for r in reviews}) == 12
        and all(r in report['sources'] and receipt(r['path']) == r for r in reviews),
        'All twelve resource reviews must belong to completed sources')
    score, quality = checked(design['scoring_plan']), checked(design['quality'])
    campaign = checked(design['campaign']); plans = [checked(ref) for ref in campaign['attempts']]
    by_attempt = {p['attempt_id']: (ref, p) for ref, p in zip(campaign['attempts'], plans)}
    require(len(report['whole_block_leaveout']) == 3, 'Complete three-fold report required')
    state_paths = past_memory_state_paths or {}
    require(type(state_paths) is dict, 'Past memory states must be an explicit case mapping')
    case_id_counts = {}
    expected_keys = {(p['capture_id'], p['attempt_id'], c['case_id']) for p in plans for c in p['cases']}
    recorded_keys = [(f['heldout_block_id'], c['attempt_id'], c['case_id'])
        for f in report['whole_block_leaveout'] for c in f['cases']]
    require(len(recorded_keys) == len(set(recorded_keys)) == 150
        and set(recorded_keys) == expected_keys
        and sum(len(c['arrivals']) for p in plans for c in p['cases']) == 546,
        'Exact unique planned 150-case/546-request cohort required')
    for fold in report['whole_block_leaveout']:
        for c in fold['cases']: case_id_counts[c['case_id']] = case_id_counts.get(c['case_id'], 0) + 1
    require(all(count == 1 or case_id not in state_paths for case_id, count in case_id_counts.items()),
            'Ambiguous bare case_id memory mapping; use block/attempt/case key')
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    fold_cache = {}; case_reports = []; total_offered = 0
    seen_case_keys = set()
    providers = build_fold_provider_cache(resource_model_path, plans, report['whole_block_leaveout'], scenario=scenario)
    for fold in report['whole_block_leaveout']:
        block = fold['heldout_block_id']; provider, target_plan, context, _ = providers[block]
        require(len(fold['cases']) == 50, 'Each heldout fold must contain exactly fifty cases')
        fold_cache[block] = provider.fold_binding
        law = checked(fold['folded_law']) if fold['folded_law'] is not None else None
        if law is not None:
            require(law.get('schema') == generated.joint.SCHEMA and law.get('heldout_block_id') == block
                and law['blocks'] == fold['training_block_ids'] and fold.get('folded_law_sha256') == digest(law),
                'Folded law binding mismatch')
        for case in fold['cases']:
            attempt = case['attempt_id']; require(attempt in by_attempt, 'Case attempt outside campaign')
            plan_ref, plan = by_attempt[attempt]; require(plan['capture_id'] == block, 'Case block mismatch')
            case_key = (block, attempt, case['case_id'])
            require(case_key not in seen_case_keys, 'Duplicate full heldout case identity')
            seen_case_keys.add(case_key)
            case_context = checked(plan['context'])
            compact = checked(case['prediction'])
            restored, requests = single.restore_full_ensemble(compact, law, score, quality,
                donor_blocks=fold['training_block_ids'])
            start, end = map(clock, restored['original_window']); drain = clock(restored['original_drain_end_ns'])
            planned = [c for c in plan['cases'] if c['case_id'] == case['case_id']]
            require(len(planned) == 1 and start.denominator == 1
                and canonical_requests(requests) == canonical_requests(generated.public_case(planned[0], int(start)))
                and end - start == generated.MEASUREMENT_NS and drain - end == generated.DRAIN_NS,
                'Case cohort or fixed boundaries differ from plan')
            past = resolve_past_memory_ref(state_paths, block_id=block, attempt_id=attempt,
                case_id=case['case_id'], case_id_count=case_id_counts[case['case_id']])
            result = _score_with_provider(restored, requests, provider, plan, case_context,
                heldout_block_id=block, scoring_plan=score, quality=quality,
                window_start_ns=int(start), window_end_ns=int(end), drain_end_ns=int(drain),
                past_memory_state=(receipt(past) if past else None), scenario=scenario)
            result.update(completed_training_report=report_ref, archived_prediction=case['prediction'],
                target_plan=plan_ref, original_labels_file_opened=False, numerical_fit_run=False,
                full_worlds_restored_and_verified_against_archived_hashes=True,
                batch_provider_cache_key=block, batch_provider_loaded_once=True)
            result['sources'] = strict_references([report_ref, report['sources'], case['prediction'],
                fold['folded_law'] or [], plan_ref, plan['context'], result['past_memory_sources'],
                result['resource_fold_binding']['sources'], model_ref, implementation_sources(__file__)])
            require(all(receipt(s['path']) == s for s in result['sources']), 'Batch result source drift')
            case_out = output / block / attempt / ('case-' + str(len(case_reports) + 1))
            case_out.mkdir(parents=True, exist_ok=True)
            case_ref = save(case_out / 'report.json', result)
            case_reports.append(dict(block_id=block, attempt_id=attempt, case_id=case['case_id'], report=case_ref,
                offered_requests=result['ensemble']['offered_requests']))
            total_offered += result['ensemble']['offered_requests']
    require(len(case_reports) == 150 and total_offered == 546, 'Complete 150-case/546-request denominator required')
    final = dict(schema=SCHEMA, status='COMPLETE_BATCH_FOLD_RESOURCE_RESCORE_TRAINING_HYPOTHESIS_ONLY',
        completed_training_report=report_ref, resource_model=model_ref, output_cases=case_reports,
        cases=len(case_reports), offered_requests=total_offered, folds=len(fold_cache),
        fold_provider_cache={b: v for b, v in fold_cache.items()}, each_fold_provider_loaded_once=True,
        labels_opened=False, numerical_fit_run=False, native_fit_run=False, GPU_started=False,
        collector_started=False, controller_started=False, all_original_draws_and_probability_mass_retained=True,
        missing_memory_state_assumed_fresh=False, case_memory_candidate_merge_enabled=False,
        resource_prediction_qualified=False, full_U_resource_qualification=False,
        final_gain_eligible=False, validation_labels_read=False,
        sources=strict_references([report_ref, report['sources'], model_ref, implementation_sources(__file__)]))
    require(all(receipt(s['path']) == s for s in final['sources']), 'Batch source drift')
    return save(output / 'report.json', final)

