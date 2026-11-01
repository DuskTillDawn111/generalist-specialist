"""Complete three-block intake and fixed-design support diagnostics for v2 FIFO.

No optimizer or validation reader is invoked. Twelve raw SHAs are provenance;
the only leaveout units are the three whole four-shard training blocks.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np

from . import twoFIFO_owned_collect_v3 as campaign_v2
from . import twoFIFO_composable_macros_v3 as macros_v2
from . import phase_heterogeneous_response_training_v1 as phase
from .macro_joint_response_second_order_v1 import differential,local_minimum_certificate
from .macro_joint_response_fit_v1 import DOMAIN
from .twoFIFO_owned_training_review_v3 import SCHEMA as REVIEW_SCHEMA,require
from .stage_response_features_v1 import receipt,checked
from .implementation_source_closure_v27 import implementation_sources,strict_references

SCHEMA='gsdeploy.twoFIFO_complete_block_training_intake.v4'
MACRO_SCHEMA='gsdeploy.twoFIFO_composable_response_macro_training.v3'
COMPLETE='PASS_COMPLETE_NATIVE_TRAINING_NOT_PREDICTOR_QUALIFICATION'


def processed_peers(state,cutoff,owner,own_phase):
    peers=state['peer_macros']
    require(type(state[phase.COUNT]) is int and len(peers)==state[phase.COUNT]
        and len({p['owner'] for p in peers})==len(peers),'Exact distinct processed peer owner/count denominator required')
    for peer in peers:
        require(peer['owner'] in campaign_v2.OWNERS and peer['owner']!=owner
            and isinstance(peer['request_id'],str) and peer['request_id']
            and isinstance(peer['phase_id'],str) and peer['phase_id']
            and type(peer['available_ns']) is int and 0<=peer['available_ns']<=cutoff,
            'Peer BEGIN identity must actually be processed at the exposure cutoff')
        phase.category(peer['owner'],peer['phase'])
    require(state[phase.SAME]==sum(p['phase']==own_phase for p in peers),'Exact same-phase peer count required')
    return peers


def planned_intake(plans,reviews):
    """Review metadata only: no partial attempt raw/response is opened here."""
    require(len(plans)==12 and len({p['attempt_id'] for p in plans})==12
        and len({p['execution_id'] for p in plans})==12,'Exact twelve distinct planned attempts required')
    expected={p['attempt_id']:p for p in plans};blocks=defaultdict(list)
    for plan in plans:
        require(plan['split']=='training' and plan['shards_in_block']==4
            and type(plan['shard_index']) is int and 1<=plan['shard_index']<=4,'Four training shards per block required')
        blocks[plan['capture_id']].append(plan)
    require(set(blocks)==set(campaign_v2.strict.CAPTURES)
        and all(len(ps)==4 and {p['shard_index'] for p in ps}=={1,2,3,4} for ps in blocks.values()),
        'Exactly three whole four-shard blocks required')
    found={};ineligible=[]
    for review_ref,review in reviews:
        require(review.get('schema')==REVIEW_SCHEMA and review.get('validation_labels_read') is False,
            'New training-only review role required before reading labels')
        bound=review['inputs']['plan.json'];plan=checked(bound)
        attempt=plan.get('attempt_id')
        require(attempt in expected and plan==expected[attempt],'Review belongs to another campaign/attempt')
        require(attempt not in found,'Duplicate reviewed attempt is not another replication unit')
        found[attempt]=review_ref
        if (review.get('status')!=COMPLETE or review.get('training_feature_extraction_eligible') is not True
            or review.get('issues')!=[]):ineligible.append(attempt)
    missing=[p['attempt_id'] for p in plans if p['attempt_id'] not in found]
    complete=[p['attempt_id'] for p in plans if p['attempt_id'] in found and p['attempt_id'] not in ineligible]
    return dict(complete=len(complete)==12,expected_attempt_ids=[p['attempt_id'] for p in plans],
        reviewed_attempts=len(found),complete_attempts=len(complete),missing_attempt_ids=missing,
        ineligible_attempt_ids=ineligible,complete_attempt_ids=complete,
        expected_cases=sum(len(p['cases']) for p in plans),
        expected_measured_offered=sum(c['offered'] for p in plans for c in p['cases']),
        expected_priming_offered=sum(len(p['priming']) for p in plans),
        expected_startups=12,training_replication_units=3,leaveout_unit='whole_capture_block_all_four_shards')


def complete_documents(plans,documents):
    """Authenticate extracted document/plan membership before any row design."""
    require(len(plans)==len(documents)==12,'No partial campaign feature design permitted')
    raw_ids=set();all_requests=set();count=0
    for plan,doc in zip(plans,documents):
        require(doc.get('schema')==MACRO_SCHEMA and doc.get('mode')=='natural'
            and doc.get('status')=='COMPLETE_ATTEMPT_DISJOINT_TRAINING_MACROS_NOT_FITTED_PROVIDER'
            and doc.get('validation_labels_read') is False
            and doc.get('whole_attempt_history_used_for_visible_prefix') is True
            and doc.get('case_history_reset') is False,'Exact complete new-schema natural macros required')
        require(all(doc.get(k)==plan[k] for k in ('capture_id','attempt_id','execution_id')),
            'Exact ordered plan/document attempt binding required')
        expected=[a for c in plan['cases'] for a in c['arrivals']]
        requests=doc['requests']
        require([r['request_id'] for r in requests]==[a['request_id'] for a in expected]
            and len(requests)==doc['measured_requests'] and doc['priming_rows_excluded_as_labels']==18,
            'Every planned measured offered row and eighteen excluded priming labels required')
        hashes=set()
        for request,arrival in zip(requests,expected):
            require(request['request_id'] not in all_requests,'Duplicate offered identity across shards')
            all_requests.add(request['request_id'])
            require(request['owner']==arrival['owner'] and all(request.get(k)==plan[k]
                for k in ('capture_id','attempt_id','execution_id')),'Exact request owner/attempt binding required')
            require(request['macros'] and all(m['pool_id']==arrival['pool_id'] and m['task']==arrival['task']
                and m['request_id']==arrival['request_id'] and m['owner']==arrival['owner']
                and m['training_replication_unit']==plan['capture_id'] and m['attempt_id']==plan['attempt_id']
                for m in request['macros']),'No public input/macro identity may change')
            hashes.update(m['capture_sha256'] for m in request['macros'])
        require(len(hashes)==1 and not raw_ids.intersection(hashes),'Twelve distinct raw provenance receipts required')
        raw_ids.update(hashes)
        n=sum(len(r['macros']) for r in requests);require(n==doc['macros'],'Every macro denominator retained')
        count+=n
    return dict(measured_requests=len(all_requests),macro_rows=count,raw_provenance_receipts=len(raw_ids),
        startup_attempts=12,priming_requests_excluded_as_labels=216,training_replication_units=3)


def prepared_rows(documents,features):
    """New-schema adapter with original nuisance groups and fixed phase vector.

    This deliberately does not relabel a v2 macro document as old v1 evidence.
    """
    require(tuple(features) in tuple(phase.DESIGNS.values()),'Only existing fixed phase designs supported')
    rows=[]
    for doc in documents:
        require(doc['schema']==MACRO_SCHEMA and doc['validation_labels_read'] is False,'New training-only macro schema required')
        for request in doc['requests']:
            macros=request['macros']
            require(macros and all(a['right_ns']==b['left_ns'] for a,b in zip(macros,macros[1:]))
                and sum(m['response_wall_ns'] for m in macros)==request['worker_start_to_ready_ns'],
                'Complete request including entry/OCR/final READY tail required')
            for macro in macros:
                require(macro['response_domain']==DOMAIN and macro['additional_stream_elapsed_or_sync_wait_may_be_added'] is False
                    and macro['true_SM_service_identified'] is False,'Original macro response-wall domain required')
                left,right=macro['left_ns'],macro['right_ns'];path=macro['causal_covariate_path']
                require(type(left) is int and type(right) is int and right>left and path
                    and path[0]['left_ns']==left and path[-1]['right_ns']==right
                    and all(a['right_ns']==b['left_ns'] for a,b in zip(path,path[1:])), 'Complete positive macro path required')
                totals=defaultdict(float);ordered=[];durations=[]
                for piece in path:
                    duration=piece['right_ns']-piece['left_ns'];state=piece['state']
                    require(type(duration) is int and duration>0 and duration==piece['duration_ns']
                        and state['cutoff_processing_ns']==piece['left_ns']
                        and piece['each_state_uses_only_its_processed_prefix'] is True,'Exact positive processed-prefix path required')
                    processed_peers(state,piece['left_ns'],macro['owner'],macro['semantic_phase'])
                    vector=tuple(phase.vector(macro['owner'],macro['semantic_phase'],state,features))
                    totals[vector]+=duration/1e6;ordered.append(vector);durations.append(duration/1e6)
                require(sum(p['duration_ns'] for p in path)==right-left==macro['response_wall_ns'],'No response exposure may be removed')
                endpoint=ordered[-1];keys=[k for k in totals if k!=endpoint]+[endpoint]
                shape=macro['forward_shape'] or {}
                group=(macro['owner'],macro['pool_id'],macro['task'],macro['macro_id'],macro['semantic_phase'],macro['kind'],
                    *(shape.get(k) for k in ('input_sequence_length','input_batch_size','forward_index')))
                rows.append(dict(group=group,capture=doc['capture_id'],raw_sha256=macro['capture_sha256'],
                    attempt_id=doc['attempt_id'],request_id=macro['request_id'],macro_id=macro['macro_id'],
                    owner=macro['owner'],phase=macro['semantic_phase'],kind=macro['kind'],
                    category=phase.category(macro['owner'],macro['semantic_phase']),response_ms=(right-left)/1e6,
                    durations=np.asarray([totals[k] for k in keys]),matrix=np.asarray(keys,dtype=float).reshape(len(keys),len(features)),
                    ordered_durations=np.asarray(durations),ordered_matrix=np.asarray(ordered,dtype=float).reshape(len(ordered),len(features))))
    require(rows,'Complete macro design rows required')
    return rows


def zero_point_diagnostic(rows,features):
    """Necessary support diagnostics, explicitly not fitted-root qualification."""
    width=len(features);groups=sorted({r['group'] for r in rows},key=repr);lookup={g:i for i,g in enumerate(groups)}
    nuisance=np.zeros((len(rows),len(groups)));derivative=[]
    for i,row in enumerate(rows):
        nuisance[i,lookup[row['group']]]=-1
        weights=row['durations']/sum(row['durations']);derivative.append(-(weights@row['matrix']))
    derivative=np.asarray(derivative).reshape(len(rows),width);adjusted=derivative.copy()
    for group in groups:
        indices=[i for i,r in enumerate(rows) if r['group']==group]
        adjusted[indices]-=np.mean(derivative[indices],axis=0)
    joint=np.column_stack((nuisance,derivative));singular=np.linalg.svd(joint,compute_uv=False)
    zero=[f for j,f in enumerate(features) if all(np.all(r['matrix'][:,j]==0) for r in rows)]
    identical=[[features[a],features[b]] for a in range(width) for b in range(a+1,width)
        if all(np.array_equal(r['matrix'][:,a],r['matrix'][:,b]) for r in rows)]
    numerical=None
    try:
        value,gradient,hessian=differential(np.zeros(width),rows)
        numerical=dict(mean_nll=value,**local_minimum_certificate(gradient,hessian))
    except ValueError as exc:
        numerical=dict(status='UNKNOWN_DEGENERATE_RESPONSE_LIKELIHOOD',reason=str(exc))
    return dict(evaluation_point='zero_response_coefficients_not_fitted_root',rows=len(rows),
        block_ids=sorted({r['capture'] for r in rows}),attempt_ids=sorted({r['attempt_id'] for r in rows}),
        nuisance_groups=len(groups),features=list(features),joint_rank=int(np.linalg.matrix_rank(joint)),
        joint_parameters=len(groups)+width,joint_full_column_rank=int(np.linalg.matrix_rank(joint))==len(groups)+width,
        nuisance_adjusted_derivative_rank=int(np.linalg.matrix_rank(adjusted)),feature_dimension=width,
        joint_singular_values=singular.tolist(),exact_piecewise_zero_columns=zero,
        exact_piecewise_identical_columns=identical,zero_point_score_curvature=numerical,
        optimizer_run=False,fitted_root_numerical_qualification=False,no_columns_deleted=True)


def observed_support(documents,plans):
    """Actual positive-duration paths, never public offsets as support labels."""
    pools={a['pool_id'] for p in plans for c in p['cases'] if c['case_id'].startswith('OCR-entry-')
        for a in c['arrivals'] if a['owner']=='ocr#0'}
    support={ (block,pool):dict(capture_id=block,pool_id=pool,entry_macros=0,n0_exposure_ns=0,
        npositive_exposure_ns=0,npositive_processed_language_BEGIN_exposure_ns=0,
        desired_single_peer_phase_exposure_ns={owner+':'+p:0 for owner in ('text#0','generalist_2#0') for p in ('prefill','decode')},
        desired_two_language_peer_composition_exposure_ns={},peer_phase_composition_exposure_ns={},peer_identity_exposures=[])
        for block in campaign_v2.strict.CAPTURES for pool in sorted(pools)}
    category_rows=Counter();total_ns=0
    for doc in documents:
        for request in doc['requests']:
            for macro in request['macros']:
                category_rows[phase.category(macro['owner'],macro['semantic_phase'])]+=1
                total_ns+=macro['response_wall_ns']
                if macro['owner']!='ocr#0' or macro['semantic_phase']!='entry' or macro['pool_id'] not in pools:continue
                record=support[doc['capture_id'],macro['pool_id']];record['entry_macros']+=1
                compositions=record['peer_phase_composition_exposure_ns']
                for piece in macro['causal_covariate_path']:
                    state=piece['state'];duration=piece['duration_ns'];n=state[phase.COUNT]
                    peers=processed_peers(state,piece['left_ns'],macro['owner'],macro['semantic_phase'])
                    record['n0_exposure_ns' if n==0 else 'npositive_exposure_ns']+=duration
                    language=[p for p in peers if p['owner'] in ('text#0','text#1','generalist_2#0') and p['phase'] in ('prefill','decode')]
                    if language:record['npositive_processed_language_BEGIN_exposure_ns']+=duration
                    if n==1 and language:
                        desired=language[0]['owner']+':'+language[0]['phase']
                        if desired in record['desired_single_peer_phase_exposure_ns']:
                            record['desired_single_peer_phase_exposure_ns'][desired]+=duration
                    key=json.dumps(sorted([p['owner'],p['phase']] for p in peers),separators=(',',':'))
                    compositions[key]=compositions.get(key,0)+duration
                    if n==2 and len(language)==2 and {p['owner'] for p in language}=={'text#0','generalist_2#0'}:
                        desired=record['desired_two_language_peer_composition_exposure_ns']
                        desired[key]=desired.get(key,0)+duration
                    if peers:record['peer_identity_exposures'].append(dict(request_id=macro['request_id'],
                        left_ns=piece['left_ns'],right_ns=piece['right_ns'],duration_ns=duration,processed_peers=peers))
    records=list(support.values())
    for record in records:
        record['status']='SUPPORTED_N0_AND_NPOSITIVE' if record['n0_exposure_ns']>0 and record['npositive_processed_language_BEGIN_exposure_ns']>0 else 'UNKNOWN_MISSING_POSITIVE_DURATION_SUPPORT'
        record['desired_phase_cross_status']={k:'OBSERVED_POSITIVE_DURATION' if v>0 else 'UNKNOWN_MISSING_POSITIVE_DURATION_SUPPORT'
            for k,v in record['desired_single_peer_phase_exposure_ns'].items()}
    return dict(OCR_exact_input_by_whole_block=records,
        each_target_OCR_input_each_block_n0_and_npositive=all(r['status']=='SUPPORTED_N0_AND_NPOSITIVE' for r in records),
        category_macro_rows=dict(category_rows),complete_response_ns=total_ns,
        all_entry_forward_final_READY_and_OCR_nodes_retained=True,planned_offset_used_as_observed_label=False,
        missing_phase_crosses_remain_UNKNOWN=True)


def run(campaign_path,review_paths,output):
    campaign_ref=receipt(campaign_path);campaign=campaign_v2.audit(campaign_path)
    plans=[checked(ref) for ref in campaign['attempts']]
    reviews=[(receipt(path),checked(receipt(path))) for path in review_paths]
    intake=planned_intake(plans,reviews)
    sources=[campaign_ref,campaign['attempts'],[ref for ref,review in reviews],implementation_sources(__file__)]
    result=dict(schema=SCHEMA,status='WAITING_FOR_ALL_TWELVE_COMPLETE_NATIVE_ATTEMPTS',intake=intake,
        expected_plan_receipts=campaign['attempts'],response_documents_extracted=0,designs=[],observed_support=None,
        numerical_fit_run=False,fitted_root_qualification=False,raw_SHA_used_as_replication_unit=False,
        replication_unit='whole_capture_block_all_four_shards',planned_startup_attempts=12,
        completed_native_startup_attempts=intake['complete_attempts'],training_replication_units=3,
        partial_capture_fitted=False,validation_labels_read=False,GPU_started=False,main_queue_provider_eligible=False,
        final_gain_eligible=False,native_CUDA_API_numeric_authentication='UNQUALIFIED')
    if intake['complete']:
        by_attempt={checked(review['inputs']['plan.json'])['attempt_id']:ref for ref,review in reviews}
        documents=[macros_v2.extract(by_attempt[p['attempt_id']]['path']) for p in plans]
        denominator=complete_documents(plans,documents);designs=[]
        for name,features in phase.DESIGNS.items():
            rows=prepared_rows(documents,features)
            full=zero_point_diagnostic(rows,features);folds=[]
            for heldout in campaign_v2.strict.CAPTURES:
                train=[r for r in rows if r['capture']!=heldout];test=[r for r in rows if r['capture']==heldout]
                nuisance_groups={r['group'] for r in train}
                unknown=[dict(attempt_id=r['attempt_id'],request_id=r['request_id'],macro_id=r['macro_id'],group=list(r['group']))
                    for r in test if r['group'] not in nuisance_groups]
                folds.append(dict(heldout_block_id=heldout,training_block_ids=sorted({r['capture'] for r in train}),
                    training_attempt_ids=sorted({r['attempt_id'] for r in train}),heldout_attempt_ids=sorted({r['attempt_id'] for r in test}),
                    heldout_rows=len(test),heldout_nuisance_supported_rows=len(test)-len(unknown),
                    heldout_nuisance_unknown_rows=len(unknown),heldout_nuisance_unknown_records=unknown,
                    training_diagnostic=zero_point_diagnostic(train,features)))
            designs.append(dict(design=name,features=list(features),full=full,whole_block_leaveout=folds,
                zero_point_support_is_only_necessary_not_fitted_qualification=True))
        result.update(status='COMPLETE_THREE_BLOCK_FIXED_DESIGN_SUPPORT_DIAGNOSTICS_NOT_FITTED',
            response_documents_extracted=12,denominator=denominator,designs=designs,observed_support=observed_support(documents,plans))
        sources.extend(d['sources'] for d in documents)
    result['sources']=strict_references(sources)
    require(all(receipt(ref['path'])==ref for ref in result['sources']),'Intake source changed')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with (output/'report.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return receipt(output/'report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',required=True)
    parser.add_argument('--training-reviews',nargs='*',default=[]);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(run(args.campaign,args.training_reviews,args.output)))
