"""Exact source-bound intake for the fifteen matching training attempts.

This module is a CPU provenance gate. It consumes an audited continuation final,
fifteen new native reviews and fifteen matching macro documents, rederives both
review and macro sources, and emits only a complete training eligibility record.
It never fits, reads validation labels, changes the old intake, or starts GPU.
"""
from __future__ import annotations
import argparse, copy, json, tempfile
from collections import defaultdict
from pathlib import Path
from .stage_response_features_v1 import checked, receipt
from .implementation_source_closure_v27 import strict_references, implementation_sources
from . import twoFIFO_matching_execute_v1 as executor
from . import twoFIFO_matching_review_v1 as native_review
from . import twoFIFO_matching_macros_v1 as macros

SCHEMA = 'gsdeploy.twoFIFO_matching_exact15_training_intake.v1'
FINAL_STATUS = 'COMPLETE_ALL_FIFTEEN_MATCHING_NATIVE_ATTEMPTS_REVIEWED_NO_FIT'


def require(value, message):
    if not value:
        raise ValueError(message)


def ref(value, label):
    require(type(value) is dict and set(value) == {'path','sha256'},
            'Explicit '+label+' receipt required')
    require(receipt(value['path']) == value, label+' receipt changed')
    return value


def _refs_from_final(final, key):
    rows = final.get('completed_attempts')
    require(type(rows) is list and len(rows) == 15, 'Continuation final must retain all fifteen attempts')
    result=[]
    for row in rows:
        value=row.get(key)
        result.append(ref(value, key+' receipt'))
    return result


def _attempt_layout(plans):
    require(len(plans) == 15 and len({p.get('attempt_id') for p in plans}) == 15,
            'Exactly fifteen unique attempt plans required')
    blocks=defaultdict(list)
    for plan in plans:
        require(plan.get('schema') == executor.SCHEMA+'.attempt'
                and plan.get('split') == 'training' and plan.get('no_retries') is True,
                'New matching training attempt plan required')
        require(type(plan.get('capture_id')) is str and type(plan.get('shard_index')) is int
                and plan.get('shards_in_block') == 5 and 1 <= plan['shard_index'] <= 5,
                'Five-shard block identity required')
        blocks[plan['capture_id']].append(plan)
    require(len(blocks) == 3 and all(len(v)==5 and {p['shard_index'] for p in v} == {1,2,3,4,5}
                                     for v in blocks.values()),
            'Exact three blocks times five shards required')
    return blocks


def _review_identity(review, plan, manifest_ref, execution_ref=None):
    require(review.get('schema') == native_review.SCHEMA
            and review.get('status') == 'PASS_COMPLETE_MATCHING_NATIVE_TRAINING_NOT_PREDICTOR_QUALIFICATION'
            and review.get('training_feature_extraction_eligible') is True
            and review.get('issues') == [] and review.get('validation_labels_read') is False
            and review.get('matching_manifest') == manifest_ref,
            'Complete new native review required')
    rp=ref(review['raw'], 'review raw')
    raw=checked(rp); pp=ref(raw['plan'], 'review plan'); plan2=checked(pp)
    require(plan2 == plan, 'Review plan differs from audited attempt')
    require(raw.get('backend') == 'native' and raw.get('status') == 'completed'
            and raw.get('attempt_id') == plan['attempt_id']
            and raw.get('execution_id') == plan['execution_id'], 'Native raw binding required')
    if execution_ref is not None:
        require(review.get('execution') == execution_ref, 'Review execution receipt differs from continuation')
    return rp, pp, raw


def _macro_identity(doc, plan, manifest_ref):
    require(doc.get('schema') == macros.__dict__.get('SCHEMA',
        'gsdeploy.twoFIFO_matching_composable_response_macro_training.v1')
            and doc.get('status') == 'COMPLETE_ATTEMPT_DISJOINT_TRAINING_MACROS_NOT_FITTED_PROVIDER'
            and doc.get('mode') == 'natural' and doc.get('validation_labels_read') is False
            and doc.get('matching_manifest') == manifest_ref
            and doc.get('whole_attempt_history_used_for_visible_prefix') is True
            and doc.get('case_history_reset') is False,
            'Complete new matching macro document required')
    require(doc.get('capture_id') == plan['capture_id'] and doc.get('attempt_id') == plan['attempt_id']
            and doc.get('execution_id') == plan['execution_id'], 'Macro attempt identity mismatch')
    require(type(doc.get('requests')) is list and doc.get('measured_requests') == len(doc['requests'])
            and doc.get('priming_rows_excluded_as_labels') == 18,
            'Measured cohort and eighteen priming exclusion required')
    expected=[a for c in plan['cases'] for a in c['arrivals']]
    require([r.get('request_id') for r in doc['requests']] == [a['request_id'] for a in expected],
            'Macro rows must preserve complete offered request order')
    require(doc.get('validation_labels_read') is False and doc.get('fitted_model_qualified') is False,
            'Macro document cannot claim fit or read validation')


def intake(manifest_path, continuation_path, *, review_paths=None, macro_paths=None, output=None):
    implementation_start=implementation_sources(__file__)
    manifest_ref=ref(receipt(manifest_path), 'manifest')
    manifest=executor.audit(manifest_path)
    final_ref=ref(receipt(continuation_path), 'continuation final')
    final=checked(final_ref)
    require(final.get('schema') == executor.CONT_SCHEMA+'.run'
            and final.get('status') == FINAL_STATUS and final.get('no_retries') is True
            and final.get('fit_eligible') is False and final.get('automatic_fit') is False
            and final.get('numerical_fit_run') is False
            and final.get('validation_labels_read') is False,
            'Complete no-fit continuation final required')
    require(final.get('manifest') == manifest_ref, 'Continuation manifest binding changed')
    attempts=manifest['attempts']; plans=[checked(ref(a['plan'],'attempt plan')) for a in attempts]
    blocks=_attempt_layout(plans)
    expected={p['attempt_id']:p for p in plans}
    completed=final['completed_attempts']; ids=[r.get('attempt_id') for r in completed]
    require(set(ids)==set(expected) and len(ids)==15, 'Continuation must contain each attempt exactly once')
    require(len(set(ids))==15, 'Continuation duplicate attempt is not a replication unit')
    review_paths=_refs_from_final(final,'training_review') if review_paths is None else [ref(x,'native review') for x in review_paths]
    macro_paths=_refs_from_final(final,'macros') if macro_paths is None else [ref(x,'macro') for x in macro_paths]
    require(len(review_paths)==len(macro_paths)==15, 'Exact fifteen review/macro receipts required')
    # Snapshot every declared nested source before any independent rederivation.
    # Final source checks use these original receipts, never newly refreshed SHAs.
    resource_paths=_refs_from_final(final,'observed_resources')
    start_sources=strict_references([implementation_start,manifest_ref,final_ref,completed,resource_paths,
        manifest.get('sources',[]),manifest['attempts'],review_paths,macro_paths,
        [checked(r).get('sources',[]) for r in review_paths],
        [checked(r).get('sources',[]) for r in macro_paths]])
    for source in start_sources:ref(source,'intake initial source')
    # Bind continuation entries by attempt, avoiding positional ambiguity.
    review_by_id={};macro_by_id={}
    for row, rr, mr in zip(completed, review_paths, macro_paths):
        aid=row['attempt_id']; require(aid in expected, 'Unknown continuation attempt')
        require(row.get('training_review') == rr and row.get('macros') == mr, 'Final receipt list mismatch')
        require(aid not in review_by_id, 'Duplicate review attempt')
        review_by_id[aid]=rr; macro_by_id[aid]=mr
    review_docs=[]; macro_docs=[]; raw_refs=[]; plan_refs=[]; resource_docs=[]
    for plan in plans:
        aid=plan['attempt_id']; rr=review_by_id[aid]; mr=macro_by_id[aid]
        execution_ref=next(row['execution'] for row in completed if row['attempt_id']==aid)
        execution=checked(ref(execution_ref, 'execution'))
        item=next(item for item in manifest['attempts'] if item['attempt_id']==aid)
        require(item['execution_id'] == plan['execution_id'] and item['plan'] == next(
            a['plan'] for a in attempts if a['attempt_id']==aid), 'Execution plan identity mismatch')
        executor.completed_execution(execution, manifest_ref, item)
        review=checked(rr)
        rp,pp,raw=_review_identity(review,plan,manifest_ref,execution_ref)
        # Independently rederive both artifacts from their source paths.
        require(type(review.get('execution')) is dict and review['execution'].get('path'),
                'Native review must retain owned execution receipt')
        with tempfile.TemporaryDirectory(prefix='matching-review-recheck-') as recheck:
            rebuilt=native_review.review(rp['path'], Path(recheck)/('review-'+aid),
                review['execution']['path'], manifest_path=manifest_path)
            require(checked(rebuilt)==review, 'Native review did not independently rederive exactly')
        doc=checked(mr); _macro_identity(doc,plan,manifest_ref)
        rebuilt_macro=macros.extract(rr['path'])
        require(rebuilt_macro==doc, 'Macro document did not independently rederive exactly')
        entry=next(row for row in completed if row['attempt_id']==aid)
        resource_ref=ref(entry['observed_resources'],'observed resources')
        require(Path(resource_ref['path']).resolve() == Path(continuation_path).resolve().parent/aid/'observed-resources.json',
                'Canonical continuation resource sidecar required')
        resources=checked(resource_ref)
        require(resources == native_review.resource_facts(raw)
                and resources == review.get('observed_resources'),
                'Every observed resource window must independently rederive from reviewed raw')
        resource_docs.append(resources)
        review_docs.append(review); macro_docs.append(doc); raw_refs.append(rp); plan_refs.append(pp)
    measured=sum(d['measured_requests'] for d in macro_docs)
    priming=sum(d['priming_rows_excluded_as_labels'] for d in macro_docs)
    require(measured == 390 and priming == 270, 'Exact 390 measured and 270 priming denominator required')
    semantic_ids=[]
    for plan,review in zip(plans,review_docs):
        planned=executor.planned_priming_semantics(plan,checked(plan['context']))
        proof=review.get('priming_semantics',{})
        require(proof.get('planned')==planned and proof.get('actual_native_history_qualified') is True
                and proof.get('actual_native_identity')==planned['sha256']
                and proof.get('provenance_request_ids')==[r['request_id'] for r in plan['priming']],
                'Native priming content semantics and exact provenance IDs must both authenticate')
        semantic_ids.append(planned['sha256'])
    require(len(set(semantic_ids))==1,'All three blocks require the same ordered physical priming semantics')
    require(sum(len(p['cases']) for p in plans)==171
            and sum(d['startup']['occurrences'] for d in resource_docs)==15
            and sum(d['window_count'] for d in resource_docs)==171,
            'Exactly 171 case windows and fifteen independent startup occurrences required')
    nested_sources=[r.get('sources',[]) for r in review_docs] + [d.get('sources',[]) for d in macro_docs]
    result=dict(schema=SCHEMA,status='COMPLETE_MATCHING_EXACT15_TRAINING_INTAKE_NOT_FIT',
        manifest=manifest_ref,continuation=final_ref,attempt_count=15,
        blocks={k:sorted(p['attempt_id'] for p in v) for k,v in blocks.items()},
        attempts=[dict(attempt_id=p['attempt_id'],capture_id=p['capture_id'],shard_index=p['shard_index'],
            execution_id=p['execution_id'],training_review=review_by_id[p['attempt_id']],macros=macro_by_id[p['attempt_id']]) for p in plans],
        measured_requests=measured,priming_requests=priming,owned_history_requests=priming,
        priming_semantic_identity=semantic_ids[0],priming_request_identity_is_provenance=True,
        training_replication_units=3,case_count=sum(len(p['cases']) for p in plans),
        startup_occurrences=sum(d['startup']['occurrences'] for d in resource_docs),
        observed_resource_windows=sum(d['window_count'] for d in resource_docs),
        resource_forecast_qualified=False,kernel_resource_fit=False,
        leaveout_unit='whole_capture_block_all_five_shards',raw_SHA_used_as_replication_unit=False,
        review_count=15,macro_count=15,validation_labels_read=False,numerical_fit_run=False,
        fitted_root_qualification=False,downstream_old_fit_compatible=False,
        downstream_numeric_interface_gap=['new macro schema must be adapted to phase12 fit row schema',
            'new exact15 receipt must feed a source-bound coordinator before any fit',
            'validation and independent prediction labels remain absent'],
        sources=strict_references([start_sources,plan_refs,raw_refs,nested_sources]),
        GPU_started=False,main_queue_provider_eligible=False,final_gain_eligible=False)
    require(implementation_sources(__file__)==implementation_start,'Intake implementation closure changed')
    for source in result['sources']:ref(source,'intake final source')
    if output is not None:
        out=Path(output); out.mkdir(parents=True,exist_ok=False); (out/'report.json').write_text(json.dumps(result,indent=2)+'\n'); return ref(receipt(out/'report.json'),'intake report')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--continuation',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(); print(json.dumps(intake(a.manifest,a.continuation,output=a.output)))
