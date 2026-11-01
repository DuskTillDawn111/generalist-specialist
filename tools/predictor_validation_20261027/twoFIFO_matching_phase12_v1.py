"""Source-bound matching exact15 phase12 coordinator; prepare/audit never fit.

Running with fitting enabled is a separate action after the complete native
continuation. Missing final evidence returns WAITING without loading raw data or
calling numerical code. All numerical work reuses the original twelve-feature
fitted_rows and evaluator; failed roots/support retain an all-UNKNOWN interface.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import tempfile

from .stage_response_features_v1 import checked, receipt
from .implementation_source_closure_v27 import implementation_sources, strict_references
from . import twoFIFO_matching_execute_v1 as executor
from . import twoFIFO_matching_intake_v1 as intake
from . import twoFIFO_matching_phase12_adapter_v1 as adapter
from . import twoFIFO_phase12_fit_v2 as original

SCHEMA = 'gsdeploy.twoFIFO_matching_source_bound_phase12_fit.v1'
MANIFEST_SCHEMA = SCHEMA+'.manifest'
FEATURES = tuple(original.FEATURES)
DESIGN = original.DESIGN


def require(value, message):
    if not value:
        raise ValueError(message)


def verify(ref):
    require(type(ref) is dict and set(ref) == {'path','sha256'}
        and receipt(ref['path']) == ref, 'Changed source/evidence receipt: '+str(ref))
    return ref


def _write(path, document):
    with Path(path).open('x') as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _review_bindings(review_refs, required):
    require(type(review_refs) is list and review_refs, 'Explicit independent review receipts required')
    covered=[]
    for ref in review_refs:
        document=checked(verify(ref))
        require(type(document.get('status')) is str and document['status'].startswith('PASS')
            and 'independent' in document.get('schema','').lower()
            and 'WITHDRAWN' not in document['status'] and 'SUPERSEDED' not in document['status'],
            'Only completed PASS independent reviews may bind the coordinator')
        covered.extend(strict_references(document))
    require(all(ref in covered for ref in required),
        'Independent review receipts must cover current executor, intake and adapter sources')


def _closure():
    return implementation_sources(__file__)


def prepare(executor_manifest_path, continuation_path, independent_review_paths, output):
    """Bind the final executor manifest and future final path; no label reads."""
    execution_ref=receipt(executor_manifest_path)
    producer=executor.audit(execution_ref['path'])
    supplement=checked(producer['matching_supplement'])
    require(not Path(continuation_path).exists(),'Predeclare matching numerics before the native final exists')
    entries=[receipt(module.__file__) for module in (executor,intake,adapter)]+[receipt(__file__)]
    review_refs=[receipt(path) for path in independent_review_paths]
    if review_refs:_review_bindings(review_refs,entries)
    final_path=Path(continuation_path).resolve()
    require(final_path.name == 'report.json' and final_path.parent.parent == Path(execution_ref['path']).parent,
        'Continuation final must be report.json in a run directory of this exact executor manifest')
    output=Path(output).resolve()
    sources=strict_references([_closure(),execution_ref,producer['sources'],review_refs])
    for ref in sources: verify(ref)
    manifest=dict(schema=MANIFEST_SCHEMA,status='CPU_PREPARED_REQUIRES_EXPLICIT_FIT',
        executor_manifest=execution_ref,expected_continuation_report_path=str(final_path),
        independent_reviews=review_refs,reviewed_entry_sources=entries,sources=sources,
        independent_source_review_complete=bool(review_refs),producer_schema=executor.SCHEMA,
        source_context=supplement['artifacts']['candidate_context'],
        source_pool=supplement['artifacts']['pool'],expected_pool_inputs=22,
        matching_supplement=producer['matching_supplement'],
        execution_input_domain='MATCHING_CONTEXT_UNION22_NOT_EXACT12_SOURCE_REBOUND',
        features=list(FEATURES),design=DESIGN,expected_measured_requests=390,
        expected_attempt_ids=[a['attempt_id'] for a in producer['attempts']],
        expected_priming_requests=270,expected_attempts=15,expected_blocks=3,
        macro_row_count='DERIVED_FROM_ALL_RETAINED_MACROS_NOT_FIXED_TO_REQUEST_COUNT',
        whole_block_fold_count=3,training_attempts_per_fold=10,heldout_attempts_per_fold=5,
        no_retries=True,no_columns_deleted=True,thresholds_changed=False,
        automatic_fit=False,numerical_fit_run=False,validation_labels_read=False,GPU_started=False)
    output.mkdir(parents=True,exist_ok=False)
    _write(output/'manifest.json',manifest)
    return receipt(output/'manifest.json')


def audit(manifest_path):
    manifest=checked(receipt(manifest_path))
    require(manifest.get('schema')==MANIFEST_SCHEMA
        and manifest.get('status')=='CPU_PREPARED_REQUIRES_EXPLICIT_FIT'
        and manifest.get('no_retries') is True and manifest.get('automatic_fit') is False
        and manifest.get('numerical_fit_run') is False and manifest.get('validation_labels_read') is False
        and manifest.get('GPU_started') is False and manifest.get('no_columns_deleted') is True
        and manifest.get('thresholds_changed') is False
        and manifest.get('features')==list(FEATURES) and tuple(adapter.FEATURES)==FEATURES
        and len(FEATURES)==12 and manifest.get('design')==DESIGN,
        'Original source-bound phase12 coordinator contract required')
    require(all(manifest.get(key)==value for key,value in dict(expected_measured_requests=390,
        expected_priming_requests=270,expected_attempts=15,expected_blocks=3,whole_block_fold_count=3,
        training_attempts_per_fold=10,heldout_attempts_per_fold=5).items()),
        'Fixed complete campaign/fold denominator changed')
    entries=[receipt(module.__file__) for module in (executor,intake,adapter)]+[receipt(__file__)]
    require(manifest['reviewed_entry_sources']==entries, 'Reviewed entry source drift')
    if manifest['independent_reviews']:_review_bindings(manifest['independent_reviews'],entries)
    producer=executor.audit(manifest['executor_manifest']['path'])
    supplement=checked(producer['matching_supplement'])
    require(manifest.get('producer_schema')==executor.SCHEMA
        and manifest.get('expected_attempt_ids')==[a['attempt_id'] for a in producer['attempts']]
        and manifest.get('matching_supplement')==producer['matching_supplement']
        and manifest.get('source_context')==supplement['artifacts']['candidate_context']
        and manifest.get('source_pool')==supplement['artifacts']['pool']
        and manifest.get('expected_pool_inputs')==22
        and manifest.get('execution_input_domain')=='MATCHING_CONTEXT_UNION22_NOT_EXACT12_SOURCE_REBOUND'
        and manifest.get('independent_source_review_complete') is bool(manifest['independent_reviews']),
        'Exact new matching context/union22/source-review identity required')
    expected=strict_references([_closure(),manifest['executor_manifest'],producer['sources'],manifest['independent_reviews']])
    require(manifest['sources']==expected,'Complete current coordinator implementation closure required')
    for ref in expected: verify(ref)
    final=Path(manifest['expected_continuation_report_path'])
    require(final.is_absolute() and final.name=='report.json'
        and final.parent.parent==Path(manifest['executor_manifest']['path']).parent,
        'Expected final path differs from executor continuation location')
    return manifest


def _complete_final(manifest, final):
    return (final.get('schema')==executor.CONT_SCHEMA+'.run'
        and final.get('manifest')==manifest['executor_manifest']
        and final.get('status')==intake.FINAL_STATUS and final.get('no_retries') is True
        and final.get('numerical_fit_run') is False and final.get('validation_labels_read') is False
        and final.get('fit_eligible') is False and final.get('automatic_fit') is False
        and type(final.get('completed_attempts')) is list and len(final['completed_attempts'])==15
        and len({a.get('attempt_id') for a in final['completed_attempts']})==15
        and {a.get('attempt_id') for a in final['completed_attempts']}==set(manifest['expected_attempt_ids']))


@contextmanager
def _verified_bundle(manifest, continuation_path):
    _review_bindings(manifest['independent_reviews'],manifest['reviewed_entry_sources'])
    final_ref=receipt(continuation_path)
    require(str(Path(continuation_path).resolve())==manifest['expected_continuation_report_path']
        and _complete_final(manifest,checked(final_ref)), 'Exact complete predeclared continuation required')
    with tempfile.TemporaryDirectory(prefix='matching-phase12-') as workdir:
        intake_ref=intake.intake(manifest['executor_manifest']['path'],str(continuation_path),
            output=Path(workdir)/'intake')
        intake_doc=checked(intake_ref)
        for ref in intake_doc['sources']:verify(ref)
        bundle=adapter.build_training_rows(intake_ref['path'],workdir=workdir)
        for ref in strict_references(bundle['documents']):verify(ref)
        _validate_bundle(bundle)
        try:
            yield bundle,intake_doc
        finally:
            for ref in intake_doc['sources']:verify(ref)
            verify(final_ref)


def with_verified_bundle(manifest_path, continuation_path, callback):
    """Run callback(bundle, intake_document) inside one matching exact15 source check.

    No fitting occurs here. The intake document is detached, whereas temporary
    intake paths expire after return. A training runner can use stored fit
    diagnostics and this bundle without fitting or extracting twice.
    """
    manifest_ref=receipt(manifest_path);manifest=audit(manifest_path)
    with _verified_bundle(manifest,continuation_path) as (bundle,intake_doc):
        result=callback(bundle,intake_doc)
    verify(manifest_ref);audit(manifest_path)
    return result


def audit_fit(fit_report):
    """Authenticate saved fit receipts/contract without recomputing numerics.

    Pass an immutable receipt when continuing from an already reviewed report.
    This audit is not an independent reproduction of fitted coefficients.
    """
    fit_ref=verify(fit_report) if type(fit_report) is dict else receipt(fit_report)
    report=checked(fit_ref)
    require(report.get('schema')==SCHEMA
        and report.get('status')=='COMPLETE_MATCHING_SOURCE_BOUND_PHASE12_TRAINING_DIAGNOSTICS'
        and report.get('numerical_fit_run') is True and report.get('validation_labels_read') is False
        and report.get('GPU_started') is False, 'Saved new-schema complete training fit required')
    manifest_ref=verify(report['manifest']);manifest=audit(manifest_ref['path'])
    final_ref=verify(report['continuation'])
    require(final_ref['path']==manifest['expected_continuation_report_path']
        and _complete_final(manifest,checked(final_ref)), 'Saved fit belongs to another/incomplete continuation')
    sources=strict_references([manifest_ref,manifest['sources'],final_ref,report['intake_evidence']['sources']])
    require(report.get('sources')==sources, 'Saved fit source closure differs')
    for ref in sources:verify(ref)
    fitted=report['fit']
    require(fitted.get('features')==list(FEATURES) and fitted.get('design')==DESIGN
        and fitted.get('measured_requests')==390 and fitted.get('priming_history_requests')==270
        and fitted.get('no_columns_deleted') is True and fitted.get('thresholds_changed') is False
        and len(fitted['whole_block_leaveout'])==3, 'Saved fit changed phase12 design/denominator')
    return dict(report=report,receipt=fit_ref,numeric_recomputed=False,
        source_contract_verified=True,predictor_accuracy_qualified=False)


def _validate_bundle(bundle):
    require(bundle.get('schema')==adapter.SCHEMA and bundle.get('features')==list(FEATURES)
        and bundle.get('design')==DESIGN and bundle.get('numerical_fit_run') is False
        and bundle.get('validation_labels_read') is False and bundle.get('GPU_started') is False,
        'Only the audited unfitted matching row adapter is accepted')
    require(bundle['denominator']['measured_requests']==390
        and bundle['denominator']['priming_requests_excluded_as_labels']==270,
        '390 measured requests and separate 270 priming rows required')
    rows=bundle['full']['rows'];plans=bundle['plans'];documents=bundle['documents']
    intake._attempt_layout(plans)
    require(bundle.get('producer_schema')==executor.SCHEMA
        and bundle.get('upstream_intake_schema')==intake.SCHEMA
        and all(p['context']==bundle.get('candidate_context') for p in plans),
        'New matching producer/intake/context identity must reach numerical rows')
    require(len({p['execution_id'] for p in plans})==15 and len(documents)==15,
        'Fifteen independent execution IDs and macro documents required')
    require(len(rows)==sum(len(r['macros']) for d in documents for r in d['requests'])
        and len({(r['attempt_id'],r['request_id']) for r in rows})==390,
        'Request count and retained macro row count must be checked separately')
    require(all(r['matrix'].shape[1]==12 for r in rows), 'Every macro must keep all twelve features')
    blocks=tuple(sorted({p['capture_id'] for p in plans}))
    require({r['capture'] for r in rows}==set(blocks), 'Exact three whole capture blocks required')
    identities=[(r['capture'],r['attempt_id'],r['request_id'],r['macro_id']) for r in rows]
    require(len(set(identities))==len(identities), 'Duplicate macro row identity')
    require({r['attempt_id'] for r in rows}=={p['attempt_id'] for p in plans},
        'Macro rows must retain every matching exact15 attempt')
    offered={p['attempt_id']:{r['request_id'] for c in p['cases'] for r in c['arrivals']} for p in plans}
    require(all({r['request_id'] for r in rows if r['attempt_id']==aid}==ids
                for aid,ids in offered.items()), 'Every macro request must retain its exact offered attempt')
    measured=[r['request_id'] for p in plans for c in p['cases'] for r in c['arrivals']]
    priming=[r['request_id'] for p in plans for r in p['priming']]
    require(len(measured)==390 and len(priming)==270 and len(set(measured+priming))==660,
            'All390 measured plus270 owned-history source identities required')
    return rows,blocks


def fit_bundle(bundle):
    """Explicit numerical boundary; tests inject CPU responses into this helper.

    Do not invoke for native evidence until the coordinator has its final source
    review. Root, nuisance support and OCR support gates match original phase12.
    """
    rows,blocks=_validate_bundle(bundle)
    full,_=original.fitted_rows(rows)
    folds=[];scores=[]
    for heldout in blocks:
        train=[r for r in rows if r['capture']!=heldout]
        test=[r for r in rows if r['capture']==heldout]
        require(len({r['attempt_id'] for r in train})==10 and len({r['attempt_id'] for r in test})==5,
            'Fold must exclude every shard of its heldout block')
        fitted,parameters=original.fitted_rows(train)
        if parameters is None:
            scored=dict(summary=dict(rows=len(test),supported_rows=0,unknown_rows=len(test)),
                records=[dict(block_id=r['capture'],attempt_id=r['attempt_id'],request_id=r['request_id'],
                    macro_id=r['macro_id'],support='UNKNOWN_TRAINING_FIT_UNAVAILABLE') for r in test])
        else:
            scored=original.evaluate(test,*parameters)
        # Keep raw SHA provenance separate from the three block leaveout units.
        lookup={(r['capture'],r['request_id'],r['macro_id']):r for r in test}
        require(len(scored['records'])==len(test) and scored['summary']['rows']==len(test)
            and scored['summary']['supported_rows']+scored['summary']['unknown_rows']==len(test),
            'Evaluator must retain every heldout macro including UNKNOWN support')
        seen=set()
        for record in scored['records']:
            block=record.pop('capture_sha256',record.get('block_id',heldout))
            key=(block,record['request_id'],record['macro_id'])
            require(key in lookup and key not in seen,'Evaluator changed/duplicated heldout identity')
            seen.add(key);row=lookup[key]
            record.update(block_id=block,raw_sha256=row['raw_sha256'],attempt_id=row['attempt_id'])
        mean=scored['summary'].get('mean_nll')
        if mean is not None and math.isfinite(mean):
            scores.append(dict(block_id=heldout,mean_conditional_nll=mean))
        folds.append(dict(heldout_block_id=heldout,training_block_ids=sorted({r['capture'] for r in train}),
            training_attempt_ids=sorted({r['attempt_id'] for r in train}),
            heldout_attempt_ids=sorted({r['attempt_id'] for r in test}),
            training_fit=fitted,evaluation=scored,heldout_rows_excluded_from_training=True))
    roots=bool(full['numerical_qualification'] and all(f['training_fit']['numerical_qualification'] for f in folds))
    support=all(f['evaluation']['summary']['unknown_rows']==0 for f in folds)
    ocr=bundle['observed_support']['each_target_OCR_input_each_block_n0_and_npositive']
    qualified=bool(roots and support and ocr)
    return dict(design=DESIGN,features=list(FEATURES),denominator=bundle['denominator'],
        measured_requests=390,priming_history_requests=270,macro_rows=len(rows),full_fit=full,
        whole_block_leaveout=folds,block_scores=scores,
        equal_block_mean_conditional_nll=sum(s['mean_conditional_nll'] for s in scores)/3
            if len(scores)==3 and support else None,
        fitted_root_numerical_qualification=roots,all_leaveout_nuisance_groups_supported=support,
        conditional_response_fit_qualified=bool(roots and support),
        observed_support=bundle['observed_support'],OCR_entry_support_qualified=ocr,
        response_law_compilation_eligible=qualified,
        unqualified_prediction_interface='ALL_UNKNOWN_RETAIN_COMPLETE_COHORT_AND_PREDECLARED_DRAW_MASS'
            if not qualified else None,
        selected_candidate=None,no_candidate_selection=True,no_columns_deleted=True,
        thresholds_changed=False,all_macro_nodes_retained=True,training_replication_units=3,
        startup_attempts=15,raw_SHA_used_as_replication_unit=False,
        physical_SM_service_identified=False,USHER_physical_qualification=False,
        independent_prediction_accuracy_validated=False,main_queue_provider_eligible=False,
        final_gain_eligible=False,validation_labels_read=False,GPU_started=False)


def run(manifest_path, output, *, fit_enabled=False):
    """WAITING/READY are read-only gates; explicit fitting starts only after final."""
    manifest_ref=receipt(manifest_path);manifest=audit(manifest_path)
    final_path=Path(manifest['expected_continuation_report_path'])
    result=dict(schema=SCHEMA,manifest=manifest_ref,status='WAITING_FOR_MATCHING_EXACT15_CONTINUATION_FINAL',
        numerical_fit_run=False,fit=None,no_retries=True,validation_labels_read=False,GPU_started=False,
        independent_prediction_accuracy_validated=False,main_queue_provider_eligible=False,
        final_gain_eligible=False,unqualified_prediction_interface=
            'ALL_UNKNOWN_RETAIN_COMPLETE_COHORT_AND_PREDECLARED_DRAW_MASS')
    result['sources']=strict_references([manifest_ref,manifest['sources']])
    if final_path.is_file():
        final_ref=receipt(final_path);final=checked(final_ref)
        result['continuation']=final_ref
        # Check role before opening any review/raw or entering numeric helpers.
        complete=_complete_final(manifest,final)
        result['sources']=strict_references([result['sources'],final_ref])
        if not complete:
            result.update(status='HOLD_INCOMPLETE_OR_FOREIGN_CONTINUATION_FINAL',
                retained_continuation_status=final.get('status'))
        elif not fit_enabled:
            result['status']='READY_EXPLICIT_FIT_REQUIRED'
        else:
            # A persisted claim prevents accidental reruns after a fit failure.
            claim=Path(manifest_path).resolve().parent/'phase12-fit-claim.json'
            _write(claim,dict(manifest=manifest_ref,continuation=final_ref,no_retries=True))
            try:
                with _verified_bundle(manifest,final_path) as (bundle,intake_doc):
                    result['sources']=strict_references([result['sources'],intake_doc['sources']])
                    result['numerical_fit_run']=True
                    fitted=fit_bundle(bundle)
                    result.update(status='COMPLETE_MATCHING_SOURCE_BOUND_PHASE12_TRAINING_DIAGNOSTICS',
                        fit=fitted,intake_evidence=intake_doc,
                        unqualified_prediction_interface=fitted['unqualified_prediction_interface'])
            except Exception as exc:
                result.update(status='STOPPED_NO_RETRY',error=type(exc).__name__+': '+str(exc),fit=None,
                    unqualified_prediction_interface='ALL_UNKNOWN_RETAIN_COMPLETE_COHORT_AND_PREDECLARED_DRAW_MASS')
            verify(final_ref)
    verify(manifest_ref);audit(manifest_path)
    for ref in result['sources']:verify(ref)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    _write(output/'report.json',result)
    return receipt(output/'report.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--executor-manifest',required=True)
    p.add_argument('--continuation',required=True);p.add_argument('--independent-reviews',nargs='*',default=[])
    p.add_argument('--output',required=True)
    p=sub.add_parser('audit');p.add_argument('--manifest',required=True)
    p=sub.add_parser('run');p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--fit',action='store_true',default=False)
    args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.executor_manifest,args.continuation,args.independent_reviews,args.output)
    elif args.command=='audit':result=audit(args.manifest)
    else:result=run(args.manifest,args.output,fit_enabled=args.fit)
    print(json.dumps(result))


if __name__=='__main__':main()
