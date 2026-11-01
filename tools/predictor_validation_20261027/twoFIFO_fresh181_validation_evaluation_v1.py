"""Independently replay all available validation reviews; keep four/six totals.

Predeclared pilot thresholds remain descriptive. Missing candidates, requests,
world mass and censored endpoints cannot be selected out of the denominator.
No validation row is used to update a predictor or the original stability rule.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile
from . import twoFIFO_fresh181_validation_review_v1 as validation
from . import twoFIFO_public181_expected_score_v2 as expected
from . import twoFIFO_fresh181_native_v4 as executor
from .twoFIFO_source_rebound_execute_v1 import SourceChangeGuard
from .twoFIFO_training_manifest_v1 import require
from .implementation_source_closure_v27 import implementation_sources,strict_references
from .stage_response_features_v1 import receipt,checked
from .macro_response_queue_reference_v1 import encode
from .actual_worker_twoFIFO_reference_v5_2 import decode
SCHEMA='gsdeploy.twoFIFO_fresh181_independent_validation_evaluation.v1'


def tri_state_all(checks):
    """A known failed requirement cannot be hidden by missing evidence."""
    checks=list(checks)
    require(all(value is None or type(value) is bool for value in checks),'Boolean or UNKNOWN diagnostic checks required')
    if any(value is False for value in checks):return False
    if any(value is None for value in checks):return None
    return True


def accuracy_threshold(agreement,known,total,minimum):
    """Preserve the full denominator, including still unknown pair outcomes."""
    if (agreement+total-known)/total<minimum:return False
    return agreement/total>=minimum if known==total else None


def threshold_arithmetic(errors,contract,offered=181):
    errors=decode(errors);limits=contract['validation_thresholds']
    def bound(value,maximum):return None if value is None else value<=maximum
    queue=errors['point_queue_grid_errors']['arrived_RESULT_unfinished']
    p95={t:dict(relative_error=v['absolute_relative_error'],
        passes=bound(v['absolute_relative_error'],limits['each_task_p95_absolute_relative_error_max'])) for t,v in errors['per_task_P95'].items()}
    reader_complete=errors['point_reader_MARE_request_coverage']==1
    service_complete=errors['point_START_to_READY_MARE_request_coverage']==1
    # A survivor-only diagnostic must never satisfy a whole-cohort threshold.
    checks=dict(request_coverage=errors['offered_requests']==offered,
        e2e=bound(errors['point_reader_MARE'],limits['e2e_mare_max']) if reader_complete else None,
        service=bound(errors['point_START_to_READY_MARE'],limits['service_mare_max']) if service_complete else None,
        each_task_P95=tri_state_all(v['passes'] for v in p95.values()),
        queue_grid=bound(queue['mean_absolute_error_upper'],limits['queue_depth_mean_absolute_error_max_requests']))
    return dict(checks=checks,all_local_pilot_thresholds_pass=tri_state_all(checks.values()),
        per_task_P95=p95,full_reader_request_coverage=reader_complete,full_service_request_coverage=service_complete,
        original_thresholds=limits,physical_service_accuracy_qualified=False,
        service_scope='CONDITIONAL_HOST_START_TO_READY_REFERENCE_NOT_PHYSICAL_SM_SERVICE',
        one_block_general_accuracy_qualified=False,post_observation_tuning=False)


def evaluate(manifest_path,review_paths,output):
    manifest_ref=receipt(manifest_path);snapshot=checked(manifest_ref)
    refs=[receipt(p) for p in review_paths];documents=[checked(r) for r in refs]
    require(len(refs)<=4 and len({r['path'] for r in refs})==len(refs),'At most four distinct original candidate reviews required')
    watched=strict_references([manifest_ref,snapshot,refs,documents,implementation_sources(__file__)])
    with SourceChangeGuard(watched) as guard:
        manifest=executor.audit(manifest_path);require(manifest==snapshot,'Canonical native4 target manifest drift')
        candidates=[a['candidate_id'] for a in manifest['attempts']]
        require(len(candidates)==len(set(candidates))==4,'Original four-candidate evaluation domain required')
        contract=checked(executor.verify(checked(manifest['attempts'][0]['plan'])['public_contract']))
        predictions={};observations={};local=[];seen=set();reviews=[]
        for ref,document in zip(refs,documents):
            require(document.get('schema')==validation.SCHEMA and document.get('manifest')==manifest_ref
                and document.get('validation_labels_read') is True and document.get('validation_labels_used_for_fit') is False,
                'Independent validation role/source review required before inspecting errors')
            plan=checked(executor.verify(document['plan']));cid=plan['candidate_id']
            require(cid in candidates and cid not in seen,'Original unique candidate validation evidence required');seen.add(cid)
            with tempfile.TemporaryDirectory(prefix='twoFIFO-independent-validation-') as temp:
                rebuilt=validation.review(document['raw']['path'],Path(temp)/'review',manifest_path=manifest_path,
                    execution_path=document['execution']['path'])
                require(checked(rebuilt)==document,'Whole raw/protocol/resource/quality/prediction review must rederive exactly')
            eligible=document['status']=='PASS_CONDITIONAL_NATIVE_VALIDATION_EVIDENCE_AND_DESCRIPTIVE_ERRORS'
            if eligible:
                prediction=checked(executor.verify(document['source_authenticated_prediction']['report']))
                actual=document['actual_six_term_score']['score'];resources=document['actual_resources']['joint_tuple']
                predictions[cid]=prediction;observations[cid]=dict(terms=actual['terms'],full_U=actual['full_U'],weights=contract['weights'],primitives=resources)
                local.append(dict(candidate_id=cid,**threshold_arithmetic(document['forecast_errors'],contract)))
            else:local.append(dict(candidate_id=cid,all_local_pilot_thresholds_pass=None,issues=document['issues']))
            reviews.append(ref)
        for cid in candidates:
            if cid not in seen:local.append(dict(candidate_id=cid,all_local_pilot_thresholds_pass=None,reason='MISSING_ORIGINAL_CANDIDATE'))
        comparison=decode(expected.evaluate_validation(predictions,observations,candidate_ids=candidates))
        pairs=comparison['pairs'];known=sum(p['exact_sign_agreement'] is not None for p in pairs)
        agreement=sum(p['exact_sign_agreement'] is True for p in pairs)
        rank_pass=accuracy_threshold(agreement,known,6,contract['validation_thresholds']['candidate_pair_rank_accuracy_min'])
        sign_pass=accuracy_threshold(agreement,known,6,contract['validation_thresholds']['paired_delta_sign_accuracy_min'])
        resource_checks=[row['term_errors'][term]['within_engineering_tolerance'] for row in comparison['candidates']
            for term in expected.EVALUATION_PROTOCOL['engineering_tolerance_terms']]
        require(len(resource_checks)==12,'All four candidates and three QME checks must be retained')
        resource_pass=tri_state_all(resource_checks)
        local_pass=[r['all_local_pilot_thresholds_pass'] for r in local]
        checks=local_pass+[rank_pass,sign_pass,resource_pass]
        pilot=tri_state_all(checks)
        value=dict(schema=SCHEMA,status='DESCRIPTIVE_PREDECLARED_PILOT_EVALUATION_NOT_FINAL_GAIN',manifest=manifest_ref,reviews=reviews,
            candidate_denominator=4,pair_denominator=6,offered_per_candidate=181,worlds_per_candidate=12,
            complete_candidate_coverage=len(predictions)/4,known_pair_coverage=known/6,
            original_local_thresholds=local,full_six_term_comparison=comparison,exact_sign_accuracy=None if known!=6 else agreement/6,
            candidate_pair_rank_accuracy=None if known!=6 else agreement/6,all_predeclared_pilot_thresholds_pass=pilot,
            QME_engineering_thresholds_pass=resource_pass,near_zero_pairs_retained=True,indifference_is_not_sign_tolerance=True,
            empirical_transfer_qualified=False,independent_prediction_accuracy_validated=False,one_four_candidate_block_generalizes=False,
            validation_labels_read=True,validation_labels_used_for_fit=False,model_updated_after_validation=False,
            original_stability_evaluated=False,original_stability_rule_unchanged=True,final_gain_eligible=False,GPU_started=False,sources=watched)
        guard.check();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
        result=executor.save(out/'report.json',encode(value));guard.check();return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',required=True)
    parser.add_argument('--reviews',nargs='*',default=[]);parser.add_argument('--output',required=True);args=parser.parse_args()
    print(json.dumps(evaluate(args.manifest,args.reviews,args.output)))
