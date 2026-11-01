"""Isolated resource companion for complete generated v5_2 ensembles.

The frozen runner/coordinator are unchanged. Actual source-authenticated v3
whole-block resources rescore a full generated world through the original
six-term bridge. Missing past memory is UNKNOWN, never an assumed fresh state.
"""
from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import json
from pathlib import Path

from . import twoFIFO_resource_fold_provider_v1 as folds
from . import twoFIFO_forecast_ensemble_v1 as ensemble
from . import twoFIFO_generated_wholeblock_evaluation_v1 as generated
from .actual_worker_world_score_v3 import score_world
from .actual_worker_twoFIFO_reference_v5_2 import decode
from .actual_worker_twoFIFO_query_adapter_v5_2 import canonical_requests,clock
from .macro_response_queue_reference_v1 import encode
from .macro_six_term_bridge_v1 import whole_score
from .implementation_source_closure_v27 import implementation_sources,strict_references

SCHEMA='gsdeploy.twoFIFO_fold_resource_rescore.v1'
require=generated.require
receipt=generated.receipt
checked=generated.checked
save=generated.save
digest=generated.digest


def past_memory(ref,origin):
    """Read an explicit past-only state contract, never derive it from labels.

    This binds evidence/cutoff bytes; physical correctness of the caller's
    historical state derivation remains an explicit additional assumption.
    No fresh-after-priming branch is admitted by this companion.
    """
    if ref is None:return None,[]
    doc=checked(ref)
    require(type(doc) is dict and set(doc)=={'schema','cutoff_ns','forecast_origin_ns','state','sources',
        'future_labels_used','basis'},'Closed causal past memory document required')
    require(doc['schema']==SCHEMA+'.causal_past_memory_state'
        and doc['basis']=='immutable_pre_case_history_only' and doc['future_labels_used'] is False
        and clock(doc['cutoff_ns'])<=origin==clock(doc['forecast_origin_ns']),
        'Memory state must be explicitly available no later than forecast origin')
    state=doc['state']
    require(type(state) is dict and set(state)=={'basis','ordered_state_history_identity','last_state_key',
        'prior_owned_sampled_peak_mib'} and state['basis']=='explicit_prior_attempt_history',
        'Only explicit past attempt history is accepted; missing is not fresh')
    require(all(type(state[key]) is str and len(state[key])==64 for key in
        ('ordered_state_history_identity','last_state_key'))
        and folds.resource.legacy.finite(state['prior_owned_sampled_peak_mib']),
        'Explicit prior history hashes and finite nonnegative owned peak required')
    require(type(doc['sources']) is list and doc['sources'],'Explicit past-only memory source evidence required')
    for source in doc['sources']:require(receipt(source['path'])==source,'Past memory source drift')
    return copy.deepcopy(state),strict_references([ref,doc['sources']])


def _score_contract(document,requests,scoring_plan,quality,start,end,drain):
    require(document.get('schema')==ensemble.SCHEMA,'Full generated finite ensemble required')
    require(document.get('public_requests')==canonical_requests(requests)
        and document.get('offered_requests')==len(requests),'Exact complete public cohort/order required')
    require(list(map(clock,document['original_window']))==[start,end]
        and clock(document['original_drain_end_ns'])==drain,'Original fixed measurement/drain required')
    require(document.get('score_comparison_contract')==dict(
        scoring_plan={key:scoring_plan[key] for key in ensemble.SCORE_FIELDS},quality=quality),
        'Original score normalization and frozen quality must match generated ensemble')
    whole_score(dict(original_window=encode([start,end]),original_drain_end_ns=encode(drain),cutoff_ns=encode(drain),
        RESULT_receipts={},unfinished_request_ids=[r['request_id'] for r in requests],offered_requests=len(requests)),
        requests,scoring_plan,quality)
    draws=document.get('draws');require(type(draws) is list and draws,'Every original draw required')
    require(len({row['draw_id'] for row in draws})==len(draws)
        and all(type(row['draw_id']) is str and row['draw_id'] for row in draws),'Distinct original draw IDs required')
    require(all(clock(row['weight'])>0 and row['offered_requests']==len(requests) for row in draws)
        and sum((clock(row['weight']) for row in draws),Fraction(0))==1,'All draw weights/cohort denominators required')
    ids=[r['request_id'] for r in requests]
    for row in draws:
        require((row['status']=='GENERATED_WORLD' and type(row.get('forecast')) is dict and 'world' in row['forecast'])
            or (row['status']=='UNKNOWN_WORLD' and row.get('forecast') is None
                and row.get('all_public_request_ids_retained')==ids),
            'Full world or explicit full-cohort UNKNOWN required; compact traces cannot be rescored directly')


def rescore_ensemble(document,requests,resource_model_path,target_plan,target_context,*,heldout_block_id,
        scoring_plan,quality,window_start_ns,window_end_ns,drain_end_ns,past_memory_state=None,scenario='mean'):
    """Use one true eight-donor fold provider for every full-cohort draw.

    No raw heldout M/E or future request outcomes are accepted as an argument.
    ``past_memory_state`` is a receipt for the exact past-only contract above.
    Full-model audit internally authenticates all twelve native attempts; only
    its selected two-block/eight-attempt document supplies resource values.
    """
    requests=[dict(row) for row in requests];document=copy.deepcopy(document)
    start,end,drain=map(clock,(window_start_ns,window_end_ns,drain_end_ns))
    require(start<end<=drain and all(v.denominator==1 for v in (start,end,drain)),
        'Original integer scoring boundaries required')
    _score_contract(document,requests,scoring_plan,quality,start,end,drain)
    require(target_plan['capture_id']==heldout_block_id,'Resource fold must leave out the target attempt block')
    require(set(scoring_plan['aliases'])==set(target_plan['resident_owners'])==set(target_context['residents'])
        and all(scoring_plan['aliases'][owner]['family']==target_context['residents'][owner]
            for owner in scoring_plan['aliases']),'Original scoring resident/family physical roster differs')
    model_ref=receipt(resource_model_path);state,state_sources=past_memory(past_memory_state,start)
    provider=folds.FoldResourceTrainingProvider(resource_model_path,target_plan,target_context,
        heldout_block_id=heldout_block_id,allow_training_resource_hypothesis=True,scenario=scenario)
    binding=provider.fold_binding
    require(target_plan['attempt_id'] in binding['heldout_attempt_ids']
        and len(binding['training_attempt_ids'])==8 and len(binding['training_block_ids'])==2
        and heldout_block_id not in binding['training_block_ids'],'Exact heldout target and eight-donor fold required')
    require(all(row['block_id'] in binding['training_block_ids'] for row in document['draws']),
        'Every generated donor block must belong to the resource training fold')
    resource_evidence=dict(schema=SCHEMA+'.explicit_source_evidence',resource_model=model_ref,
        selected_fold_model_sha256=binding['selected_fold_model_sha256'],heldout_block_id=heldout_block_id,
        training_attempt_ids=binding['training_attempt_ids'],scenario=scenario,
        memory_state=past_memory_state,memory_state_derivation_is_caller_declared=True)
    rows=[]
    for original in document['draws']:
        row=copy.deepcopy(original)
        if row['status']=='GENERATED_WORLD':
            world=row['forecast']['world'];source_world_sha=digest(world)
            # This calls the existing v5_2 query-bound timeline and exact v3
            # predict_timeline through the original six-score bridge. Provider
            # polymorphism uses only its selected eight-donor document.
            scored=score_world(world,requests,scoring_plan,quality,
                window_start_ns=int(start),window_end_ns=int(end),drain_end_ns=int(drain),
                resource_provider=provider,target_context=target_context,
                resource_source_evidence=resource_evidence,resource_memory_initial_state=state)
            require(scored['query_fingerprint_sha256']==row['forecast']['query_fingerprint_sha256'],
                'Resource rescore changed generated query identity')
            scored['resource_fold_binding']=copy.deepcopy(binding)
            row['forecast']['scoring']=scored
            row['source_world_sha256']=source_world_sha
            require(digest(row['forecast']['world'])==source_world_sha,'Resource rescore changed generated world')
        # Unknown transport worlds remain unchanged, with all cohort IDs and
        # original weights. Unsupported resources remain None in known worlds.
        rows.append(row)
    values=decode(rows)
    result=copy.deepcopy(document)
    result.update(draws=rows,six_terms={term:ensemble.distribution(values,
        lambda f,t=term:f['scoring']['original_score']['terms'][t]) for term in 'QTLPME'},
        utility=ensemble.distribution(values,lambda f:f['scoring']['original_score']['full_U']))
    require(receipt(resource_model_path)==model_ref,'Resource model changed during rescore')
    for source in state_sources+binding['sources']:require(receipt(source['path'])==source,'Resource companion source drift')
    return dict(schema=SCHEMA,status='COMPLETE_FOLD_RESOURCE_RESCORE_TRAINING_HYPOTHESIS_ONLY',ensemble=result,
        resource_model=model_ref,resource_fold_binding=binding,resource_scenario=scenario,
        past_memory_state=past_memory_state,past_memory_sources=state_sources,
        missing_memory_state_assumed_fresh=False,resource_memory_provider_replaced=False,
        generated_worlds_and_public_cohort_unchanged=True,all_original_draws_and_probability_mass_retained=True,
        original_six_score_reused=True,partial_U_reported=False,
        energy_scope='original_measurement_only_startup_once_at_3600_seconds',
        memory_scope='startup_and_explicit_past_through_original_drain_owned_sampled_peak',
        memory_state_derivation_is_caller_declared=True,
        case_memory_candidate_merge_enabled=False,
        actual_native_quality_validated=False,quality_from_frozen_training_expectations_not_native_outcome=True,
        resource_prediction_qualified=False,main_queue_provider_eligible=False,final_gain_eligible=False,
        independent_prediction_accuracy_validated=False,full_U_resource_qualification=False,
        validation_labels_read=False,GPU_started=False)


def restore_full_ensemble(compact,law,scoring_plan,quality,*,donor_blocks):
    """Replay fixed training-law forecasts and verify their archived trace hashes.

    The original runner omits full world traces. Replay consumes only public
    arrivals, its saved past seed and folded law; no native raw/label is opened.
    This is generation, not numerical fitting or retuning.
    """
    archived=compact['ensemble'];requests=[]
    for row in archived['public_requests']:
        at=Fraction(*row['arrival_ns']);require(at.denominator==1,'Original integer public arrival required')
        requests.append(dict(row,arrival_ns=int(at)))
    a,b=map(clock,archived['original_window']);d=clock(archived['original_drain_end_ns'])
    seed=compact.get('settled_seed');reason=compact.get('settled_seed_unknown_reason')
    if seed is None:
        require(type(reason) is str and reason,'Archived missing past seed must remain explicit UNKNOWN')
        draws=generated.unknown_draws(donor_blocks,reason)
    elif law is None:
        draws=generated.unknown_draws(donor_blocks,'UNKNOWN_COMPLETE_RESPONSE_FIT_OR_OCR_SUPPORT_UNQUALIFIED')
    else:
        draws,_=generated.compile_case_draws(requests,law)
    restored=ensemble.evaluate(requests,draws,scoring_plan=scoring_plan,quality=quality,
        window_start_ns=int(a),window_end_ns=int(b),drain_end_ns=int(d),settled_initial_state=seed)
    require(len(restored['draws'])==len(archived['draws']),'Archived complete draw count differs')
    for old,new in zip(archived['draws'],restored['draws']):
        require(all(old[k]==new[k] for k in ('draw_id','block_id','weight','status','offered_requests')),
            'Replay cannot change predefined draw/status/denominator')
        if new['status']=='GENERATED_WORLD':
            require(digest(new['forecast'])==old['forecast_sha256'],'Full generated forecast differs from archived hash')
        else:require(old.get('reason')==new.get('reason'),'Archived UNKNOWN reason changed during replay')
    require(encode(generated.compact_prediction(dict(ensemble=restored)))['ensemble']==archived,
        'Archived complete ensemble contract, summaries or UNKNOWN cohort changed during replay')
    return restored,requests


def run_completed_case(completed_report_path,resource_model_path,*,heldout_block_id,attempt_id,case_id,
                       output,past_memory_state_path=None,scenario='mean'):
    """Explicit CPU continuation from completed twelve-attempt training results.

    No native fit, labels, GPU, automatic loop, controller or new validation is
    invoked. It restores only one already recorded case's complete draw worlds.
    """
    report_ref=receipt(completed_report_path);report=checked(report_ref)
    require(report.get('schema')==generated.SCHEMA and report.get('numerical_fit_run') is True
        and report.get('status') in ('COMPLETE_CONDITIONAL_GENERATED_WHOLEBLOCK_TRAINING_EVALUATION',
            'COMPLETE_NATIVE_TRAINING_FIT_UNQUALIFIED_ALL_GENERATED_WORLDS_UNKNOWN')
        and report.get('validation_labels_read') is False,'Complete twelve-attempt generated training report required')
    for source in report['sources']:require(receipt(source['path'])==source,'Completed training source drift')
    _,design=generated.audit_design(report['design']['path'])
    require(receipt(report['design']['path'])==report['design'],'Completed report design receipt changed')
    resource_ref=receipt(resource_model_path);resource_document=checked(resource_ref)
    require(resource_document.get('schema')==folds.resource.SCHEMA
        and resource_document.get('campaign')==design['campaign'],
        'Resource model and completed generated design must bind the exact same campaign receipt')
    reviews=resource_document.get('training_reviews')
    require(type(reviews) is list and len(reviews)==12
        and len({r['path'] for r in reviews})==12
        and all(r in report['sources'] and receipt(r['path'])==r for r in reviews),
        'All twelve exact resource review receipts must belong to completed training sources without drift')
    score,quality=checked(design['scoring_plan']),checked(design['quality'])
    campaign=checked(design['campaign']);plans=[checked(ref) for ref in campaign['attempts']]
    matches=[(ref,plan) for ref,plan in zip(campaign['attempts'],plans) if plan['attempt_id']==attempt_id]
    require(len(matches)==1,'Exact original target attempt required');plan_ref,plan=matches[0]
    require(plan['capture_id']==heldout_block_id,'Completed case target block/fold mismatch')
    foldrows=[fold for fold in report['whole_block_leaveout'] if fold['heldout_block_id']==heldout_block_id]
    require(len(foldrows)==1,'Exact recorded whole-block fold required');fold=foldrows[0]
    require(set(fold['training_block_ids'])==set(design['training_block_ids'])-{heldout_block_id}
        and len(fold['donor_attempt_ids'])==8 and len(fold['heldout_attempt_ids'])==4
        and set(fold['donor_attempt_ids'])=={p['attempt_id'] for p in plans if p['capture_id']!=heldout_block_id}
        and set(fold['heldout_attempt_ids'])=={p['attempt_id'] for p in plans if p['capture_id']==heldout_block_id},
        'Original complete eight/four whole-block training split required')
    cases=[c for c in fold['cases'] if c['attempt_id']==attempt_id and c['case_id']==case_id]
    require(len(cases)==1,'Exact recorded heldout case required');case=cases[0]
    compact=checked(case['prediction']);law=checked(fold['folded_law']) if fold['folded_law'] is not None else None
    if law is not None:require(law.get('schema')==generated.joint.SCHEMA
        and law.get('heldout_block_id')==heldout_block_id
        and law['blocks']==fold['training_block_ids'] and fold.get('folded_law_sha256')==digest(law),
        'Folded donor law or completed-report digest mismatch')
    restored,requests=restore_full_ensemble(compact,law,score,quality,donor_blocks=fold['training_block_ids'])
    a,b=map(clock,restored['original_window']);d=clock(restored['original_drain_end_ns'])
    planned=[c for c in plan['cases'] if c['case_id']==case_id]
    require(len(planned)==1 and a.denominator==1
        and canonical_requests(requests)==canonical_requests(generated.public_case(planned[0],int(a)))
        and b-a==generated.MEASUREMENT_NS and d-b==generated.DRAIN_NS,
        'Archived cohort and fixed boundaries must exactly match the planned public case')
    past_ref=None if past_memory_state_path is None else receipt(past_memory_state_path)
    context=checked(plan['context'])
    result=rescore_ensemble(restored,requests,resource_model_path,plan,context,heldout_block_id=heldout_block_id,
        scoring_plan=score,quality=quality,window_start_ns=int(a),window_end_ns=int(b),drain_end_ns=int(d),
        past_memory_state=past_ref,scenario=scenario)
    require(result['resource_model']==resource_ref,'Resource model changed after completed-training binding')
    result.update(completed_training_report=report_ref,archived_prediction=case['prediction'],target_plan=plan_ref,
        original_labels_file_opened=False,numerical_fit_run=False,
        resource_and_generated_campaign_receipts_equal=True,
        resource_reviews_bound_to_completed_training_sources=True,
        full_worlds_restored_and_verified_against_archived_hashes=True)
    result['sources']=strict_references([report_ref,report['sources'],case['prediction'],fold['folded_law'] or [],
        plan_ref,plan['context'],result['past_memory_sources'],result['resource_fold_binding']['sources'],
        resource_ref,implementation_sources(__file__)])
    require(all(receipt(source['path'])==source for source in result['sources']),'Completed-case rescore source drift')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    return save(output/'report.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--completed-report',required=True);p.add_argument('--resource-model',required=True)
    p.add_argument('--heldout-block',required=True);p.add_argument('--attempt-id',required=True);p.add_argument('--case-id',required=True)
    p.add_argument('--past-memory-state');p.add_argument('--scenario',choices=('mean','max'),default='mean')
    p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(run_completed_case(a.completed_report,a.resource_model,heldout_block_id=a.heldout_block,
        attempt_id=a.attempt_id,case_id=a.case_id,output=a.output,past_memory_state_path=a.past_memory_state,scenario=a.scenario)))
