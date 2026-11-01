"""Source-rebound to phase12 row adapter; schema-only, never a fitter.

The adapter authenticates the exact12 intake and rederives each review/macro
source, then passes explicitly validated new-schema macro documents to the
frozen row compiler and zero-point diagnostics. It returns full and whole-block
fold rows plus root/support gates. No optimizer, fit, GPU or validation labels
are touched; new rows are never relabeled as old raw captures.
"""
from __future__ import annotations
import argparse, copy
from collections import defaultdict
import numpy as np
from pathlib import Path
from .stage_response_features_v1 import checked, receipt
from .implementation_source_closure_v27 import strict_references
from . import twoFIFO_source_rebound_intake_v1 as source_intake
from . import twoFIFO_source_rebound_review_v1 as native_review
from . import twoFIFO_source_rebound_macros_v1 as macros
from . import twoFIFO_training_intake_v4 as frozen

SCHEMA='gsdeploy.twoFIFO_source_rebound_phase12_row_adapter.v1'
MACRO_SCHEMA='gsdeploy.twoFIFO_source_rebound_composable_response_macro_training.v1'
DESIGN='own_family_phase_count_global_same'
FEATURES=tuple(frozen.phase.DESIGNS[DESIGN])


def require(value,message):
    if not value: raise ValueError(message)


def _load_report(path):
    report=checked(receipt(path))
    require(report.get('schema')==source_intake.SCHEMA
        and report.get('status')=='COMPLETE_SOURCE_REBOUND_EXACT12_TRAINING_INTAKE_NOT_FIT'
        and report.get('attempt_count')==12 and report.get('measured_requests')==546
        and report.get('priming_requests')==216 and report.get('validation_labels_read') is False
        and report.get('numerical_fit_run') is False and report.get('fitted_root_qualification') is False
        and report.get('downstream_old_fit_compatible') is False,
        'Complete new exact12 intake required; old fit/intake schema rejected')
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
            and doc.get('fitted_model_qualified') is False and doc.get('source_rebound_manifest')==manifest,
            'New source-rebound macro schema/binding required')
        rebuilt=macros.extract(review_ref['path'])
        require(rebuilt==doc,'Macro source did not rederive exactly')
        plans.append(checked(review['inputs']['plan.json'])); docs.append(doc); review_refs.append(review_ref)
    require(len(docs)==12 and sum(d['measured_requests'] for d in docs)==546
        and sum(d['priming_rows_excluded_as_labels'] for d in docs)==216,
        'Exact 546 measured/216 priming denominator required')
    return plans,docs,review_refs


def _rows(docs,plans):
    denominator=dict(measured_requests=sum(d['measured_requests'] for d in docs),
        priming_requests_excluded_as_labels=sum(d['priming_rows_excluded_as_labels'] for d in docs),
        training_replication_units=len({d['capture_id'] for d in docs}))
    require(denominator['measured_requests']==546 and denominator['priming_requests_excluded_as_labels']==216
        and denominator['training_replication_units']==3, 'Complete offered denominator and history-only priming required')
    rows=prepared_rows_new(docs,FEATURES)
    require(len(rows)>0 and {r['capture'] for r in rows}==set(frozen.campaign_v2.strict.CAPTURES),
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


def build_training_rows(intake_path, *, workdir=None):
    report=_load_report(intake_path)
    root=Path(workdir or Path(intake_path).resolve().parent)/'phase12-adapter-rechecks'; root.mkdir(parents=True,exist_ok=True)
    plans,docs,review_refs=_rederive(report,workdir=root)
    rows,denominator=_rows(docs,plans)
    blocks=sorted({r['capture'] for r in rows})
    folds=[]
    for heldout in blocks:
        train=[r for r in rows if r['capture']!=heldout]; test=[r for r in rows if r['capture']==heldout]
        require(len({r['attempt_id'] for r in train})==8 and len({r['attempt_id'] for r in test})==4,
            'Whole-block leaveout must exclude all four shards')
        # Reuse only frozen zero-point rank/support diagnostics. This is not a
        # fitted root and does not choose/drop columns or use heldout labels.
        diagnostics=frozen.zero_point_diagnostic(train,FEATURES)
        folds.append(dict(heldout_block_id=heldout,training_block_ids=sorted({r['capture'] for r in train}),
            training_attempt_ids=sorted({r['attempt_id'] for r in train}),heldout_attempt_ids=sorted({r['attempt_id'] for r in test}),
            training_rows=train,heldout_rows=test,zero_point_diagnostic=diagnostics,
            heldout_rows_excluded_from_training=True,fit_required=True,fit_performed=False))
    full_diag=frozen.zero_point_diagnostic(rows,FEATURES)
    support=frozen.observed_support(docs,plans)
    return dict(schema=SCHEMA,status='ROWS_READY_FOR_SEPARATE_PHASE12_FIT_NOT_FIT',design=DESIGN,
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
