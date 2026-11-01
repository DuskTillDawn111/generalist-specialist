"""New-schema complete-block phase12 conditional response fit; CPU only.

All numerical solving uses the existing analytic trust/root implementation.
No partial campaign, validation labels, column deletion or candidate selection.
"""
import argparse
import json
import math
from pathlib import Path

from . import twoFIFO_training_intake_v4 as intake
from . import macro_projection_contrast_training_v1 as numeric
from .macro_joint_response_fit_v1 import evaluate
from .implementation_source_closure_v27 import implementation_sources,strict_references

SCHEMA='gsdeploy.twoFIFO_phase12_complete_block_fit.v2'
DESIGN='own_family_phase_count_global_same'
FEATURES=intake.phase.DESIGNS[DESIGN]
require=intake.require
receipt=intake.receipt
checked=intake.checked
CONTAMINATION_HOLD='COLLECTOR_MEASUREMENT_CHECKPOINT_CONTAMINATION_HOLD'


def load_complete(campaign_path,review_paths):
    campaign_ref=receipt(campaign_path);metadata=checked(campaign_ref)
    # The v2 collector copied the growing raw under runtime/monitor locks
    # inside each measurement. Preserve its evidence but never fit this domain,
    # even if a caller later supplies twelve apparently complete native reviews.
    # This gate precedes review/raw reads, extraction and numerical work.
    if metadata.get('schema')=='gsdeploy.original_twoFIFO_owned_training_campaign.v2':
        return dict(status=CONTAMINATION_HOLD,intake=dict(complete=False,status=CONTAMINATION_HOLD),
            plans=[],documents=None,sources=[campaign_ref],collector_training_domain_eligible=False)
    campaign=intake.campaign_v2.audit(campaign_path)
    plans=[checked(ref) for ref in campaign['attempts']]
    reviews=[]
    for path in review_paths:
        ref=receipt(path);reviews.append((ref,checked(ref)))
    admission=intake.planned_intake(plans,reviews)
    sources=[campaign_ref,campaign['attempts'],[ref for ref,review in reviews]]
    if not admission['complete']:return dict(intake=admission,plans=plans,documents=None,sources=sources)
    by_attempt={checked(review['inputs']['plan.json'])['attempt_id']:ref for ref,review in reviews}
    documents=[intake.macros_v2.extract(by_attempt[p['attempt_id']]['path']) for p in plans]
    denominator=intake.complete_documents(plans,documents)
    require(denominator['measured_requests']==546 and denominator['priming_requests_excluded_as_labels']==216,
        'Complete fixed546 measured/216 history-only priming required')
    sources.extend(d['sources'] for d in documents)
    return dict(intake=admission,plans=plans,documents=documents,denominator=denominator,sources=sources)


def fitted_rows(rows):
    """Do not turn optimizer exceptions or unsupported likelihoods into a fit."""
    require(len(FEATURES)==12 and rows and all(r['matrix'].shape[1]==12 for r in rows),'Exact complete phase12 rows required')
    try:
        result,beta,mu,variance=numeric.fit(rows,FEATURES)
    except (ValueError,FloatingPointError,OverflowError) as exc:
        return dict(status='UNKNOWN_NUMERICAL_FIT_FAILURE',reason=type(exc).__name__+': '+str(exc),
            features=list(FEATURES),macro_rows=len(rows),numerical_qualification=False,
            whole_blocks=sorted({r['capture'] for r in rows}),all_rows_retained=True),None
    require(result['features']==list(FEATURES) and result['macro_rows']==len(rows),'Numeric core changed design/denominator')
    result.update(status='FITTED_ROOT_DIAGNOSTICS',whole_blocks=sorted({r['capture'] for r in rows}),all_rows_retained=True)
    return result,(beta,mu,variance)


def fit_new_blocks(documents,plans):
    denominator=intake.complete_documents(plans,documents)
    require(denominator['measured_requests']==546,'Complete public offered denominator required')
    rows=intake.prepared_rows(documents,FEATURES)
    require({r['capture'] for r in rows}==set(intake.campaign_v2.strict.CAPTURES),'Three whole blocks required')
    support=intake.observed_support(documents,plans);full,parameters=fitted_rows(rows);folds=[];block_scores=[]
    for heldout in intake.campaign_v2.strict.CAPTURES:
        train=[r for r in rows if r['capture']!=heldout];test=[r for r in rows if r['capture']==heldout]
        require(len({r['attempt_id'] for r in train})==8 and len({r['attempt_id'] for r in test})==4,
            'Leaveout must exclude all four shards, never individual raw files')
        fitted,params=fitted_rows(train)
        if params is None:
            scored=dict(summary=dict(rows=len(test),supported_rows=0,unknown_rows=len(test)),records=[dict(
                block_id=r['capture'],attempt_id=r['attempt_id'],request_id=r['request_id'],macro_id=r['macro_id'],
                support='UNKNOWN_TRAINING_FIT_UNAVAILABLE') for r in test])
        else:scored=evaluate(test,*params)
        # Existing evaluator's historical key capture_sha256 now contains the
        # explicit block unit supplied by new rows; preserve provenance apart.
        by_identity={(r['capture'],r['request_id'],r['macro_id']):r for r in test}
        for record in scored['records']:
            block=record.pop('capture_sha256',record.get('block_id',heldout));record['block_id']=block
            original=by_identity[block,record['request_id'],record['macro_id']]
            record.update(raw_sha256=original['raw_sha256'],attempt_id=original['attempt_id'])
        mean=scored['summary'].get('mean_nll')
        if mean is not None and math.isfinite(mean):block_scores.append(dict(block_id=heldout,mean_conditional_nll=mean))
        folds.append(dict(heldout_block_id=heldout,training_block_ids=sorted({r['capture'] for r in train}),
            training_attempt_ids=sorted({r['attempt_id'] for r in train}),heldout_attempt_ids=sorted({r['attempt_id'] for r in test}),
            training_fit=fitted,evaluation=scored))
    root_qualified=bool(full['numerical_qualification'] and all(f['training_fit']['numerical_qualification'] for f in folds))
    supported=all(f['evaluation']['summary']['unknown_rows']==0 for f in folds)
    conditional_qualified=root_qualified and supported
    return dict(design=DESIGN,features=list(FEATURES),denominator=denominator,macro_rows=len(rows),
        full_fit=full,whole_block_leaveout=folds,block_scores=block_scores,
        equal_block_mean_conditional_nll=sum(r['mean_conditional_nll'] for r in block_scores)/3 if len(block_scores)==3 and supported else None,
        fitted_root_numerical_qualification=root_qualified,all_leaveout_nuisance_groups_supported=supported,
        conditional_response_fit_qualified=conditional_qualified,observed_support=support,
        OCR_entry_support_qualified=support['each_target_OCR_input_each_block_n0_and_npositive'],
        response_law_compilation_eligible=bool(conditional_qualified and support['each_target_OCR_input_each_block_n0_and_npositive']),
        selected_candidate=None,no_candidate_selection=True,no_columns_deleted=True,all_macro_nodes_retained=True,
        training_replication_units=3,startup_attempts=12,raw_SHA_used_as_replication_unit=False,
        conditional_peer_path_not_prospective_queue=True,independent_prediction_accuracy_validated=False,
        main_queue_provider_eligible=False,final_gain_eligible=False,validation_labels_read=False,GPU_started=False)


def run(campaign_path,review_paths,output):
    bundle=load_complete(campaign_path,review_paths)
    result=dict(schema=SCHEMA,status=bundle.get('status','WAITING_FOR_ALL_TWELVE_COMPLETE_NATIVE_ATTEMPTS'),
        intake=bundle['intake'],features=list(FEATURES),design=DESIGN,numerical_fit_run=False,fit=None,
        validation_labels_read=False,GPU_started=False,main_queue_provider_eligible=False,final_gain_eligible=False)
    if bundle['documents'] is not None:
        fitted=fit_new_blocks(bundle['documents'],bundle['plans'])
        result.update(status='COMPLETE_PHASE12_FITTED_ROOT_TRAINING_DIAGNOSTICS',numerical_fit_run=True,fit=fitted)
    result['sources']=strict_references([bundle['sources'],implementation_sources(__file__)])
    require(all(receipt(r['path'])==r for r in result['sources']),'Fit source changed')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with (output/'report.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return receipt(output/'report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',required=True)
    parser.add_argument('--training-reviews',nargs='*',default=[]);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(run(args.campaign,args.training_reviews,args.output)))
