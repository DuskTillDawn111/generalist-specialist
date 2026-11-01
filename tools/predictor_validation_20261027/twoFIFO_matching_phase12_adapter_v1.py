"""Matching exact15 to phase12 row adapter; never a fitter.

The adapter authenticates the exact15 intake and rederives each review/macro
source, then passes explicitly validated new-schema macro documents to the
frozen row compiler and zero-point diagnostics. It returns full and whole-block
fold rows plus root/support gates. No optimizer, fit, GPU or validation labels
are touched; new rows are never relabeled as old raw captures.
"""
from __future__ import annotations
import argparse, copy, json
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path
from .stage_response_features_v1 import checked, receipt
from .implementation_source_closure_v27 import strict_references
from . import twoFIFO_matching_intake_v1 as source_intake
from . import twoFIFO_matching_review_v1 as native_review
from . import twoFIFO_matching_macros_v1 as macros
from . import twoFIFO_training_intake_v4 as frozen

SCHEMA='gsdeploy.twoFIFO_matching_phase12_row_adapter.v1'
MACRO_SCHEMA='gsdeploy.twoFIFO_matching_composable_response_macro_training.v1'
DESIGN='own_family_phase_count_global_same'
FEATURES=tuple(frozen.phase.DESIGNS[DESIGN])


def require(value,message):
    if not value: raise ValueError(message)


def _load_report(path):
    report=checked(receipt(path))
    require(report.get('schema')==source_intake.SCHEMA
        and report.get('status')=='COMPLETE_MATCHING_EXACT15_TRAINING_INTAKE_NOT_FIT'
        and report.get('attempt_count')==15 and report.get('measured_requests')==390
        and report.get('priming_requests')==270 and report.get('validation_labels_read') is False
        and report.get('numerical_fit_run') is False and report.get('fitted_root_qualification') is False
        and report.get('downstream_old_fit_compatible') is False,
        'Complete new exact15 intake required; old fit/intake schema rejected')
    return report


def _rederive(report, *, workdir):
    docs=[]; plans=[]; review_refs=[]
    manifest=report['manifest']; continuation=report['continuation']
    source_intake.ref(manifest,'intake manifest'); source_intake.ref(continuation,'intake continuation')
    for index,item in enumerate(report['attempts']):
        review_ref=source_intake.ref(item['training_review'],'training review')
        review=checked(review_ref); plan_ref=source_intake.ref(review['inputs']['plan.json'],'review plan') if 'inputs' in review else None
        require(review.get('schema')==native_review.SCHEMA and review.get('validation_labels_read') is False
            and review.get('issues')==[] and review.get('training_feature_extraction_eligible') is True,
            'New native review gate failed')
        review_raw=source_intake.ref(review['raw'],'review raw')
        rebased=native_review.review(review_raw['path'],Path(workdir)/('review-'+str(index)),review['execution']['path'],manifest_path=manifest['path'])
        require(checked(rebased)==review,'Native review source did not rederive exactly')
        macro_ref=source_intake.ref(item['macros'],'macro')
        doc=checked(macro_ref)
        require(doc.get('schema')==MACRO_SCHEMA and doc.get('validation_labels_read') is False
            and doc.get('fitted_model_qualified') is False and doc.get('matching_manifest')==manifest,
            'New matching macro schema/binding required')
        rebuilt=macros.extract(review_ref['path'])
        require(rebuilt==doc,'Macro source did not rederive exactly')
        plans.append(checked(review['inputs']['plan.json'])); docs.append(doc); review_refs.append(review_ref)
    require(len(docs)==15 and sum(d['measured_requests'] for d in docs)==390
        and sum(d['priming_rows_excluded_as_labels'] for d in docs)==270,
        'Exact 390 measured/270 priming denominator required')
    source_intake._attempt_layout(plans)
    return plans,docs,review_refs


def _rows(docs,plans):
    denominator=dict(measured_requests=sum(d['measured_requests'] for d in docs),
        priming_requests_excluded_as_labels=sum(d['priming_rows_excluded_as_labels'] for d in docs),
        training_replication_units=len({d['capture_id'] for d in docs}))
    require(denominator['measured_requests']==390 and denominator['priming_requests_excluded_as_labels']==270
        and denominator['training_replication_units']==3, 'Complete offered denominator and history-only priming required')
    rows=prepared_rows_new(docs,FEATURES)
    require(len(rows)>0 and {r['capture'] for r in rows}=={p['capture_id'] for p in plans},
        'Rows must contain all three whole capture blocks')
    return rows,denominator


def prepared_rows_new(documents, features):
    """Compile new macro schema without changing its identity to the old one."""
    require(tuple(features) in tuple(frozen.phase.DESIGNS.values()), 'Only frozen phase design supported')
    rows=[]
    for doc in documents:
        require(doc.get('schema')==MACRO_SCHEMA and doc.get('mode')=='natural'
            and doc.get('status')=='COMPLETE_ATTEMPT_DISJOINT_TRAINING_MACROS_NOT_FITTED_PROVIDER'
            and doc.get('validation_labels_read') is False and doc.get('fitted_model_qualified') is False
            and doc.get('whole_attempt_history_used_for_visible_prefix') is True
            and doc.get('case_history_reset') is False, 'Complete new macro document required')
        for request in doc['requests']:
            require(request.get('macros') and request.get('capture_id')==doc['capture_id']
                and request.get('attempt_id')==doc['attempt_id'] and request.get('execution_id')==doc['execution_id'],
                'Complete macro request identity required')
            for macro in request['macros']:
                require(all(macro.get(k)==request.get(k) for k in ('request_id','owner','pool_id'))
                    and macro.get('capture_sha256') and macro.get('attempt_id')==doc['attempt_id']
                    and macro.get('training_replication_unit')==doc['capture_id']
                    and macro.get('response_domain')==frozen.DOMAIN
                    and macro.get('additional_stream_elapsed_or_sync_wait_may_be_added') is False
                    and macro.get('true_SM_service_identified') is False,
                    'Macro public identity/domain changed')
                path=macro.get('causal_covariate_path'); require(path, 'Complete causal path required')
                left,right=macro['left_ns'],macro['right_ns']
                require(type(left) is int and type(right) is int and right>left
                    and macro['response_wall_ns']==right-left
                    and path[0]['left_ns']==left and path[-1]['right_ns']==right
                    and all(a['right_ns']==b['left_ns'] for a,b in zip(path,path[1:])),
                    'Positive contiguous macro path required')
                totals=defaultdict(float); ordered=[]; durations=[]
                for piece in path:
                    duration=piece['duration_ns']; state=piece['state']
                    require(type(duration) is int and duration>0 and piece['left_ns']+duration==piece['right_ns']
                        and state['cutoff_processing_ns']==piece['left_ns']
                        and piece['each_state_uses_only_its_processed_prefix'] is True,
                        'Processed-prefix state path required')
                    peers=frozen.processed_peers(state,piece['left_ns'],macro['owner'],macro['semantic_phase'])
                    vector=tuple(frozen.phase.vector(macro['owner'],macro['semantic_phase'],state,features))
                    totals[vector]+=duration/1e6;ordered.append(vector);durations.append(duration/1e6)
                require(sum(p['duration_ns'] for p in path)==right-left, 'Macro exposure removed')
                keys=[k for k in totals if k!=ordered[-1]]+[ordered[-1]]
                shape=macro.get('forward_shape') or {}
                group=(macro['owner'],macro['pool_id'],macro['task'],macro['macro_id'],macro['semantic_phase'],
                    macro['kind'],shape.get('input_sequence_length'),shape.get('input_batch_size'),shape.get('forward_index'))
                rows.append(dict(group=group,
                    capture=doc['capture_id'],raw_sha256=macro['capture_sha256'],attempt_id=doc['attempt_id'],
                    request_id=macro['request_id'],macro_id=macro['macro_id'],owner=macro['owner'],
                    phase=macro['semantic_phase'],kind=macro['kind'],category=frozen.phase.category(macro['owner'],macro['semantic_phase']),
                    response_ms=(right-left)/1e6,durations=np.asarray([totals[k] for k in keys]),
                    matrix=np.asarray(keys,dtype=float).reshape(len(keys),len(features)),
                    ordered_durations=np.asarray(durations),ordered_matrix=np.asarray(ordered,dtype=float).reshape(len(ordered),len(features))))
            for previous, current in zip(request['macros'], request['macros'][1:]):
                require(previous['right_ns'] == current['left_ns'],
                    'Adjacent request macros must form one contiguous worker path')
            require(sum(r['response_wall_ns'] for r in request['macros'])==request.get('worker_start_to_ready_ns'),
                'Entry/forward/OCR/READY tail denominator changed')
    require(rows,'Complete new macro rows required'); return rows


def observed_support(documents,plans):
    """Actual positive-duration paths, never public offsets as support labels."""
    pools={a['pool_id'] for p in plans for c in p['cases'] 
        for a in c['arrivals'] if a['owner']=='ocr#0'}
    require(pools and len({p['capture_id'] for p in plans})==3, 'Explicit nonempty OCR targets in three matching blocks required')
    support={ (block,pool):dict(capture_id=block,pool_id=pool,entry_macros=0,n0_exposure_ns=0,
        npositive_exposure_ns=0,npositive_processed_language_BEGIN_exposure_ns=0,
        desired_single_peer_phase_exposure_ns={owner+':'+p:0 for owner in ('text#0','generalist_2#0') for p in ('prefill','decode')},
        desired_two_language_peer_composition_exposure_ns={},peer_phase_composition_exposure_ns={},peer_identity_exposures=[])
        for block in sorted({p['capture_id'] for p in plans}) for pool in sorted(pools)}
    category_rows=Counter();total_ns=0
    for doc in documents:
        for request in doc['requests']:
            for macro in request['macros']:
                category_rows[frozen.phase.category(macro['owner'],macro['semantic_phase'])]+=1
                total_ns+=macro['response_wall_ns']
                if macro['owner']!='ocr#0' or macro['semantic_phase']!='entry' or macro['pool_id'] not in pools:continue
                record=support[doc['capture_id'],macro['pool_id']];record['entry_macros']+=1
                compositions=record['peer_phase_composition_exposure_ns']
                for piece in macro['causal_covariate_path']:
                    state=piece['state'];duration=piece['duration_ns'];n=state[frozen.phase.COUNT]
                    peers=frozen.processed_peers(state,piece['left_ns'],macro['owner'],macro['semantic_phase'])
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



def build_training_rows(intake_path, *, workdir=None):
    report=_load_report(intake_path)
    root=Path(workdir or Path(intake_path).resolve().parent)/'phase12-adapter-rechecks'; root.mkdir(parents=True,exist_ok=True)
    plans,docs,review_refs=_rederive(report,workdir=root)
    rows,denominator=_rows(docs,plans)
    blocks=sorted({r['capture'] for r in rows})
    folds=[]
    for heldout in blocks:
        train=[r for r in rows if r['capture']!=heldout]; test=[r for r in rows if r['capture']==heldout]
        require(len({r['attempt_id'] for r in train})==10 and len({r['attempt_id'] for r in test})==5,
            'Whole-block leaveout must exclude all five shards')
        # Reuse only frozen zero-point rank/support diagnostics. This is not a
        # fitted root and does not choose/drop columns or use heldout labels.
        diagnostics=frozen.zero_point_diagnostic(train,FEATURES)
        folds.append(dict(heldout_block_id=heldout,training_block_ids=sorted({r['capture'] for r in train}),
            training_attempt_ids=sorted({r['attempt_id'] for r in train}),heldout_attempt_ids=sorted({r['attempt_id'] for r in test}),
            training_rows=train,heldout_rows=test,zero_point_diagnostic=diagnostics,
            heldout_rows_excluded_from_training=True,fit_required=True,fit_performed=False))
    full_diag=frozen.zero_point_diagnostic(rows,FEATURES)
    support=observed_support(docs,plans)
    return dict(schema=SCHEMA,status='ROWS_READY_FOR_SEPARATE_PHASE12_FIT_NOT_FIT',design=DESIGN,
        producer_schema=source_intake.executor.SCHEMA,upstream_intake_schema=source_intake.SCHEMA,
        candidate_context=plans[0]['context'],
        features=list(FEATURES),plans=plans,documents=docs,review_receipts=review_refs,
        denominator=denominator,full=dict(rows=rows,zero_point_diagnostic=full_diag,fit_required=True,fit_performed=False),
        whole_block_folds=folds,observed_support=support,
        exact_three_blocks=True,training_replication_units=3,raw_SHA_used_as_replication_unit=False,
        whole_block_leaveout=True,heldout_rows_never_used_for_fit=True,no_columns_deleted=True,
        root_numerical_qualification=False,all_leaveout_nuisance_groups_supported=None,
        physical_SM_service_identified=False,USHER_physical_qualification=False,
        validation_labels_read=False,numerical_fit_run=False,GPU_started=False,
        main_queue_provider_eligible=False,final_gain_eligible=False,
        downstream_fit_interface='Call a separately source-bound phase12 fitter with this full/fold row bundle; this adapter never fits')


def write_bundle(intake_path,output,*,workdir=None):
    # JSON serialization is intentionally unsupported for numpy row objects;
    # write only a source receipt/index, retaining rows through Python API.
    bundle=build_training_rows(intake_path,workdir=workdir)
    out=Path(output); out.mkdir(parents=True,exist_ok=False)
    index=copy.deepcopy({k:v for k,v in bundle.items() if k not in ('plans','documents','full','whole_block_folds')})
    index['full_row_count']=len(bundle['full']['rows']); index['fold_row_counts']=[len(f['heldout_rows']) for f in bundle['whole_block_folds']]
    (out/'index.json').write_text(__import__('json').dumps(index,indent=2,default=str)+'\n')
    return bundle

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--intake',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    bundle=write_bundle(a.intake,a.output);print({'schema':bundle['schema'],'full_rows':len(bundle['full']['rows']),'folds':len(bundle['whole_block_folds']),'fit_performed':False})
