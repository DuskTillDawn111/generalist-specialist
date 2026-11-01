"""Conditional generated whole-block training leaveout with predefined draws.

Preparation fixes public draw slots and score/quality receipts before fitting.
Only complete twelve-attempt v3 intake enables numerical work. Each heldout
case uses its public calendar and immutable past snapshot; generated predictions
are durably saved before that case's evaluation labels are extracted. Intake
and full/fold numerical review already read heldout labels; generation excludes
them from its inputs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from . import twoFIFO_phase12_fit_v2 as response
from . import twoFIFO_joint_request_law_v4 as joint
from . import twoFIFO_forecast_ensemble_v1 as ensemble
from . import twoFIFO_settled_history_initial_v1 as history
from . import phase_heterogeneous_protocol_candidate_v1 as rates
from .macro_response_queue_reference_v1 import encode
from .actual_worker_twoFIFO_reference_v5_2 import decode
from .macro_six_term_bridge_v1 import whole_score
from .validation import percentile
from .twoFIFO_training_manifest_v1 import MEASUREMENT_NS,DRAIN_NS
from .implementation_source_closure_v27 import implementation_sources,strict_references

SCHEMA='gsdeploy.twoFIFO_generated_wholeblock_training_evaluation.v1'
DESIGN_SCHEMA=SCHEMA+'.predeclared_design'
SCHEME=dict(schema=SCHEMA+'.finite_draw_scheme',slots_per_training_block=4,
    slot_weights='equal_within_equal_training_block',
    donor_quantiles='public_key_ordinal_sha256_rotation_of_four_midpoint_strata',
    public_salt='twoFIFO-generated-wholeblock-v1-fixed-before-fit',
    seed_selection_from_heldout_error=False,
    joint_request_vector_draw=True,per_macro_draw=False,
    conditional_request_draws_independent=False,
    dependence='common_training_block_and_finite_stratum_slot_with_public_key_ordinal_rotation',
    quadrature_is_fitted_latent_probability_law=False)
require=response.require
receipt=response.receipt
checked=response.checked


def digest(value):
    return hashlib.sha256(json.dumps(encode(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def save(path,value):
    # Reuse the exact reviewed atomic fsync/replace routine; CPU files only.
    from .twoFIFO_owned_collect_v3 import atomic_save
    atomic_save(path,encode(value));return receipt(path)


def prepare(campaign_path,scoring_plan_path,quality_path,output):
    campaign=response.intake.campaign_v2.audit(campaign_path)
    score_ref,quality_ref=receipt(scoring_plan_path),receipt(quality_path)
    score,quality=checked(score_ref),checked(quality_ref)
    require(type(score) is dict and type(quality) is dict,'Explicit public scoring plan and quality documents required')
    # Validate the unchanged scoring arithmetic without reading a native label.
    whole_score(dict(original_window=encode([Fraction(0),Fraction(MEASUREMENT_NS)]),
        original_drain_end_ns=encode(Fraction(MEASUREMENT_NS+DRAIN_NS)),
        cutoff_ns=encode(Fraction(MEASUREMENT_NS+DRAIN_NS)),RESULT_receipts={},unfinished_request_ids=[],offered_requests=0),[],score,quality)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    result=dict(schema=DESIGN_SCHEMA,status='CPU_PREDECLARED_BEFORE_NUMERICAL_FIT',
        campaign=receipt(campaign_path),scoring_plan=score_ref,quality=quality_ref,
        scheme=SCHEME,scheme_sha256=digest(SCHEME),training_block_ids=list(response.intake.campaign_v2.strict.CAPTURES),
        attempts=campaign['attempts'],heldout_unit='all_four_shards_of_one_training_block',
        generation_inputs='public_calendar_and_past_settled_snapshot_and_training_block_donors_only',
        external_validation_labels_read=False,numerical_fit_run=False,GPU_started=False,
        sources=strict_references([receipt(campaign_path),score_ref,quality_ref,implementation_sources(__file__)]))
    return save(output/'design.json',result)


def audit_design(path):
    ref=receipt(path);design=checked(ref)
    require(type(design) is dict and set(design)=={'schema','status','campaign','scoring_plan','quality','scheme','scheme_sha256',
        'training_block_ids','attempts','heldout_unit','generation_inputs','external_validation_labels_read',
        'numerical_fit_run','GPU_started','sources'},'Closed predeclared design required')
    require(design.get('schema')==DESIGN_SCHEMA and design.get('status')=='CPU_PREDECLARED_BEFORE_NUMERICAL_FIT'
        and design.get('scheme')==SCHEME and design.get('scheme_sha256')==digest(SCHEME),
        'Exact predefined draw scheme required; no error-dependent draw selection')
    campaign=checked(design['campaign'])
    require(campaign.get('schema')==response.intake.campaign_v2.SCHEMA
        and design['attempts']==campaign['attempts']
        and design['training_block_ids']==list(response.intake.campaign_v2.strict.CAPTURES)
        and design['heldout_unit']=='all_four_shards_of_one_training_block'
        and design['generation_inputs']=='public_calendar_and_past_settled_snapshot_and_training_block_donors_only'
        and design['external_validation_labels_read'] is False and design['numerical_fit_run'] is False
        and design['GPU_started'] is False,'Exact native v3 design, whole blocks and pre-fit scope required')
    for source in design['sources']:require(receipt(source['path'])==source,'Predeclared design source drift')
    require(design['sources']==strict_references([design['campaign'],design['scoring_plan'],design['quality'],implementation_sources(__file__)]),
        'Complete predeclared implementation and score/quality binding required')
    return ref,design


def catalog(blocks):
    require(blocks and len(set(blocks))==len(blocks),'Distinct training block latents required')
    n=SCHEME['slots_per_training_block']
    return [dict(draw_id=block+':predeclared-slot:'+str(slot),block_id=block,slot=slot,
        weight=Fraction(1,len(blocks)*n)) for block in blocks for slot in range(n)]


def public_quantile(key,ordinal,slot):
    require(type(ordinal) is int and ordinal>=0 and type(slot) is int and 0<=slot<4,'Exact public ordinal and predefined slot required')
    message=json.dumps([SCHEME['public_salt'],list(key),ordinal],separators=(',',':'),ensure_ascii=True).encode()
    rotation=int.from_bytes(hashlib.sha256(message).digest()[:8],'big')%4
    return Fraction(2*((slot+rotation)%4)+1,8)


def unknown_draws(blocks,reason,evidence=None):
    return [dict(draw_id=d['draw_id'],block_id=d['block_id'],weight=d['weight'],
        inputs=ensemble.UnknownForecastInputs(reason,evidence)) for d in catalog(blocks)]


def compile_case_draws(requests,law):
    """No heldout execution row, macro, READY or query body is an input here."""
    require(law.get('schema')==joint.SCHEMA,'New v3-source joint law required')
    blocks=law['blocks'];heldout=law.get('heldout_block_id')
    require(heldout is None or heldout not in blocks,'Heldout block cannot be a donor latent')
    require(all(v['training_block_id'] in blocks for v in law['vectors']),'Vector donor block outside training law')
    table=rates.rate_table(dict(features=law['features'],coefficients=law['coefficients']))
    draws=[];selection=[]
    for draw in catalog(blocks):
        plans=defaultdict(list);timings={};transport={};selected=[];ordinals=defaultdict(int)
        try:
            for _,row in sorted(enumerate(requests),key=lambda pair:(pair[1]['arrival_ns'],pair[0])):
                key=(row['owner'],row['task'],row['pool_id']);ordinal=ordinals[key];ordinals[key]+=1
                selected_vector=joint.choose_vector(law,key,ordinal,
                    block_quantile=float(Fraction(2*blocks.index(draw['block_id'])+1,2*len(blocks))),
                    within_block_quantile=float(public_quantile(key,ordinal,draw['slot'])))
                if selected_vector['status']!='COMPLETE_REQUEST_JOINT_VECTOR_DRAW':
                    raise UnknownEvidence(selected_vector['status']+':'+repr((key,ordinal,draw['block_id'])))
                require(selected_vector['block_id']==draw['block_id'],'Shared block latent drift')
                vector=selected_vector['vector'];compiled=joint.compile_draw_v4(vector,ordinal)
                plan=compiled['plan'];require(plan not in timings,'Distinct public slots must retain their joint protocol vectors')
                plans[key].append(plan);timings[plan]=compiled['protocol_slacks'];transport[(*key,ordinal)]=compiled['transport_draw']
                selected.append(dict(public_key=list(key),public_key_arrival_ordinal=ordinal,vector_id=vector['vector_id'],
                    training_block_id=vector['training_block_id'],within_block_quantile=public_quantile(key,ordinal,draw['slot'])))
            inputs=dict(plans={k:tuple(v) for k,v in plans.items()},rates=table,protocol_slacks=timings,transport_draws=transport)
            status='COMPILED_COMPLETE_PUBLIC_COHORT_JOINT_VECTORS'
        except UnknownEvidence as exc:
            inputs=ensemble.UnknownForecastInputs(str(exc),getattr(exc,'evidence',None));status='UNKNOWN_COMPILE_EVIDENCE'
        draws.append(dict(draw_id=draw['draw_id'],block_id=draw['block_id'],weight=draw['weight'],inputs=inputs))
        selection.append(dict(draw_id=draw['draw_id'],status=status,selections=selected,
            incomplete_compilation_does_not_drop_world_or_requests=True))
    return draws,selection


def public_case(case,epoch):
    require(type(epoch) is int and epoch>=0,'Known integer public case epoch required')
    return [dict(request_id=a['request_id'],owner=a['owner'],task=a['task'],pool_id=a['pool_id'],arrival_ns=epoch+a['arrival_ns'])
        for a in case['arrivals']]


def forecast_case(case,epoch,snapshot,law,scoring_plan,quality,*,donor_blocks=None,unqualified_reason=None):
    """Strict causal interface: public plan, past snapshot and donor law only."""
    requests=public_case(case,epoch)
    blocks=law['blocks'] if law is not None else donor_blocks
    if unqualified_reason is not None:
        draws=unknown_draws(blocks,unqualified_reason);selections=[]
    else:draws,selections=compile_case_draws(requests,law)
    seed=None;seed_reason=None
    try:
        if snapshot is None:raise UnknownEvidence('UNKNOWN_MISSING_IMMUTABLE_PRECASE_HISTORY')
        require(type(snapshot) is dict,'Immutable snapshot mapping required')
        producers={owner:owner+':modeled_stdout_lifetime' for owner in scoring_plan['aliases']}
        seed=history.from_snapshot(snapshot,cutoff_ns=snapshot['snapshot_time_ns'],forecast_origin_ns=epoch,owner_producers=producers)
    except UnknownEvidence as exc:
        seed_reason=str(exc);draws=unknown_draws(blocks,seed_reason,getattr(exc,'evidence',None))
    result=ensemble.evaluate(requests,draws,scoring_plan=scoring_plan,quality=quality,
        window_start_ns=epoch,window_end_ns=epoch+MEASUREMENT_NS,drain_end_ns=epoch+MEASUREMENT_NS+DRAIN_NS,
        settled_initial_state=seed)
    return dict(ensemble=result,selections=selections,settled_seed=seed,settled_seed_unknown_reason=seed_reason,
        generation_inputs_exclude_heldout_future_labels=True,conditional_on_observed_public_case_epoch=True,
        intercase_or_startup_calendar_predicted=False)


def compact_prediction(prediction):
    """Retain every draw, probability and request; avoid duplicating full traces."""
    result=dict(prediction);value=decode(prediction['ensemble']);rows=[]
    for row in value['draws']:
        row=dict(row);forecast=row.pop('forecast')
        if forecast is not None:
            world=decode(forecast['world']);row['forecast_sha256']=digest(forecast)
            row['forecast_summary']=dict(RESULT_receipts=world['RESULT_receipts'],unfinished_request_ids=world['unfinished_request_ids'],
                starts=world['starts'],READY_occurrences=world['READY_occurrences'],
                original_score=forecast['scoring']['original_score'],reader_E2E_P95_ns=forecast['queue_diagnostics']['reader_E2E_P95_ns'],
                query_fingerprint_sha256=forecast['query_fingerprint_sha256'])
        else:row['forecast_summary']=None
        rows.append(row)
    value['draws']=rows;result['ensemble']=value
    result['full_generated_traces_omitted_after_deterministic_hashing']=True
    return result


def native_labels(raw,case,epoch,scoring_plan,quality):
    """Evaluation only: never pass this function's outputs to generation."""
    requests=public_case(case,epoch);wanted={r['request_id'] for r in requests}
    rows={r['request_id']:r for r in raw['request_records'] if r['request_id'] in wanted}
    require(set(rows)==wanted,'Retain exact complete offered native label denominator')
    deadline=epoch+MEASUREMENT_NS+DRAIN_NS;receipts={};labels=[]
    for request in requests:
        row=rows[request['request_id']];at=row.get('completion_time_ns')
        complete=row.get('status')=='completed' and type(at) is int and request['arrival_ns']<=at<=deadline
        if complete:receipts[request['request_id']]=at
        labels.append(dict(request_id=request['request_id'],status=row.get('status'),completed_within_fixed_drain=complete,
            reader_E2E_ns=at-request['arrival_ns'] if complete else None))
    frame=dict(original_window=encode([Fraction(epoch),Fraction(epoch+MEASUREMENT_NS)]),
        original_drain_end_ns=encode(Fraction(deadline)),cutoff_ns=encode(Fraction(deadline)),
        RESULT_receipts={rid:encode(Fraction(at)) for rid,at in receipts.items()},
        unfinished_request_ids=[r['request_id'] for r in requests if r['request_id'] not in receipts],offered_requests=len(requests))
    return dict(requests=labels,original_score=whole_score(frame,requests,scoring_plan,quality)['score'],
        complete_cohort_reader_E2E_P95_ns=percentile([r['reader_E2E_ns'] for r in labels]) if len(receipts)==len(requests) else None,
        quality_from_frozen_training_expectations_not_native_outcome=True,actual_native_quality_validated=False,
        full_U_resource_qualification=False,
        future_macro_READY_query_bodies_read=False,heldout_training_outcomes_used_for_evaluation_only=True)


def errors(prediction,labels):
    value=decode(prediction['ensemble']);observed={r['request_id']:r for r in labels['requests']};requests=value['public_requests'];rows=[]
    for request in requests:
        rid=request['request_id'];truth=observed[rid]['reader_E2E_ns'];known=Fraction(0);unfinished=Fraction(0);total=Fraction(0);draws=[]
        for draw in value['draws']:
            weight=Fraction(draw['weight']);summary=draw.get('forecast_summary');estimate=None
            if summary is not None:
                receipt_at=summary['RESULT_receipts'].get(rid)
                if receipt_at is None:unfinished+=weight
                else:estimate=Fraction(receipt_at)-Fraction(*request['arrival_ns']);known+=weight;total+=weight*estimate
            draws.append(dict(draw_id=draw['draw_id'],weight=weight,predicted_reader_E2E_ns=estimate,
                error_ns=None if estimate is None or truth is None else estimate-truth,
                status='UNKNOWN_WORLD' if summary is None else 'KNOWN_INCOMPLETE' if estimate is None else 'GENERATED_COMPLETE'))
        point=total if known==1 else None
        rows.append(dict(request_id=rid,actual_reader_E2E_ns=truth,draws=draws,
            known_complete_probability_mass=known,known_incomplete_probability_mass=unfinished,unknown_world_probability_mass=1-known-unfinished,
            predicted_mean_reader_E2E_ns=point,error_ns=None if point is None or truth is None else point-truth,
            conditional_on_known_complete_mean_ns=total/known if known else None))
    p=value['expected_world_complete_cohort_P95_ns']['expected_value'];actual=labels['complete_cohort_reader_E2E_P95_ns']
    return dict(requests=rows,offered_requests=len(requests),point_supported_requests=sum(r['error_ns'] is not None for r in rows),
        expected_world_P95_error_ns=None if p is None or actual is None else p-actual,
        expected_world_P95_is_not_mixture_P95=True,unknowns_dropped=False)


def fold_summary(cases):
    rows=[row for case in cases for row in case['errors']['requests']]
    values=[float(r['error_ns']) for r in rows if r['error_ns'] is not None]
    return dict(cases=len(cases),offered_requests=len(rows),point_supported_requests=len(values),unknown_or_incomplete_requests=len(rows)-len(values),
        mean_absolute_reader_E2E_error_ns=sum(abs(v) for v in values)/len(values) if values else None,
        mean_signed_reader_E2E_error_ns=sum(values)/len(values) if values else None,
        metrics_conditioned_on_point_supported_subset=len(values)!=len(rows),all_offered_denominator_retained=True)


def evaluate_case(raw,case,epoch,snapshot,law,scoring_plan,quality,folder,*,donor_blocks,unqualified_reason=None):
    """Persist prediction before this case's evaluation label extraction.

    Native intake and fit review have already accessed heldout labels. Only
    the public plan, past snapshot and folded law enter forecast_case.
    """
    prediction=forecast_case(case,epoch,snapshot,law,scoring_plan,quality,donor_blocks=donor_blocks,
        unqualified_reason=unqualified_reason)
    compact=compact_prediction(prediction);folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    prediction_ref=save(folder/'prediction.json',compact)
    labels=native_labels(raw,case,epoch,scoring_plan,quality);label_ref=save(folder/'labels.json',labels)
    return dict(case_id=case['case_id'],prediction=prediction_ref,labels=label_ref,errors=errors(compact,labels),
        ensemble_summary=dict(worlds=len(compact['ensemble']['draws']),
            generated_worlds=sum(d['status']=='GENERATED_WORLD' for d in compact['ensemble']['draws']),
            unknown_worlds=sum(d['status']=='UNKNOWN_WORLD' for d in compact['ensemble']['draws']),
            unknown_probability_mass=compact['ensemble']['world_unknown_probability_mass']),
        durable_before_case_evaluation_label_extraction=True)


def run(design_path,review_paths,output):
    design_ref,design=audit_design(design_path);score,quality=checked(design['scoring_plan']),checked(design['quality'])
    bundle=response.load_complete(design['campaign']['path'],review_paths)
    result=dict(schema=SCHEMA,status=bundle.get('status','WAITING_FOR_ALL_TWELVE_COMPLETE_NATIVE_ATTEMPTS'),design=design_ref,
        intake=bundle['intake'],numerical_fit_run=False,fit=None,full_law=None,whole_block_leaveout=[],
        forecasts_generated=0,external_validation_labels_read=False,validation_labels_read=False,GPU_started=False,
        main_queue_provider_eligible=False,independent_prediction_accuracy_validated=False,final_gain_eligible=False,
        quality_from_frozen_training_expectations_not_native_outcome=True,actual_native_quality_validated=False,
        calibrated_risk=False,full_U_resource_qualification=False,draws_selected_from_heldout_error=False)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    if bundle['documents'] is not None:
        require(bundle['denominator']['measured_requests']==546 and bundle['denominator']['priming_requests_excluded_as_labels']==216,
            'No partial/native-denominator shortcut to generation')
        fitted=response.fit_new_blocks(bundle['documents'],bundle['plans']);result.update(numerical_fit_run=True,fit=fitted)
        # Full/fold laws use the reviewed builder; only folded laws generate
        # heldout forecasts. Heldout documents never become folded donors.
        refs={checked(checked(receipt(p))['inputs']['plan.json'])['attempt_id']:checked(receipt(p)) for p in review_paths}
        raws=[checked(refs[p['attempt_id']]['raw']) for p in bundle['plans']]
        eligible=fitted['response_law_compilation_eligible']
        if eligible:
            full=joint.build_joint_law_v4(bundle['documents'],bundle['plans'],raws,fitted)
            result['full_law']=dict(schema=full['schema'],sha256=digest(full),vectors=full['measured_request_vectors'],
                blocks=full['blocks'],artifact=save(output/'full-joint-law.json',full),used_for_heldout_generation=False)
        for heldout in design['training_block_ids']:
            law=joint.build_joint_law_v4(bundle['documents'],bundle['plans'],raws,fitted,heldout_block_id=heldout) if eligible else None
            law_ref=save(output/(heldout+'-folded-joint-law.json'),law) if law is not None else None
            blocks=[b for b in design['training_block_ids'] if b!=heldout];cases=[]
            for plan,raw in zip(bundle['plans'],raws):
                if plan['capture_id']!=heldout:continue
                by_case={c['case_id']:c for c in raw['cases']};snapshots={h['case_id']:h['prior_history_snapshot'] for h in raw['case_histories']}
                for index,case in enumerate(plan['cases']):
                    epoch=by_case[case['case_id']]['observation_start_ns']
                    evaluation=evaluate_case(raw,case,epoch,snapshots.get(case['case_id']),law,score,quality,
                        output/plan['attempt_id']/('case-'+str(index+1)),donor_blocks=blocks,
                        unqualified_reason=None if eligible else 'UNKNOWN_COMPLETE_RESPONSE_FIT_OR_OCR_SUPPORT_UNQUALIFIED')
                    evaluation['attempt_id']=plan['attempt_id'];cases.append(evaluation)
            require(len(cases)==50 and sum(c['errors']['offered_requests'] for c in cases)==182,'Whole heldout block denominator changed')
            result['whole_block_leaveout'].append(dict(heldout_block_id=heldout,training_block_ids=blocks,
                heldout_attempt_ids=[p['attempt_id'] for p in bundle['plans'] if p['capture_id']==heldout],
                donor_attempt_ids=[p['attempt_id'] for p in bundle['plans'] if p['capture_id']!=heldout],
                folded_law_sha256=digest(law) if law is not None else None,folded_law=law_ref,cases=cases,summary=fold_summary(cases)))
        all_cases=[c for f in result['whole_block_leaveout'] for c in f['cases']]
        result.update(status=('COMPLETE_CONDITIONAL_GENERATED_WHOLEBLOCK_TRAINING_EVALUATION' if eligible else
            'COMPLETE_NATIVE_TRAINING_FIT_UNQUALIFIED_ALL_GENERATED_WORLDS_UNKNOWN'),forecasts_generated=150,
            ensemble_worlds_generated=sum(c['ensemble_summary']['generated_worlds'] for c in all_cases),
            ensemble_worlds_unknown=sum(c['ensemble_summary']['unknown_worlds'] for c in all_cases),
            heldout_future_service_labels_used_for_generation=False,conditional_on_native_past_and_known_case_epoch=True,
            startup_and_intercase_orchestration_predicted=False,
            equal_whole_block_mean_absolute_error_ns=(sum(f['summary']['mean_absolute_reader_E2E_error_ns'] for f in result['whole_block_leaveout'])/3
                if all(f['summary']['point_supported_requests']==182 for f in result['whole_block_leaveout']) else None))
    result['sources']=strict_references([design_ref,design['sources'],bundle['sources'],implementation_sources(__file__)])
    require(all(receipt(s['path'])==s for s in result['sources']),'Evaluation source drift')
    return save(output/'report.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('prepare');p.add_argument('--campaign',required=True);p.add_argument('--scoring-plan',required=True);p.add_argument('--quality',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('run');p.add_argument('--design',required=True);p.add_argument('--training-reviews',nargs='*',default=[]);p.add_argument('--output',required=True)
    a=parser.parse_args();print(json.dumps(prepare(a.campaign,a.scoring_plan,a.quality,a.output) if a.op=='prepare' else run(a.design,a.training_reviews,a.output)))
