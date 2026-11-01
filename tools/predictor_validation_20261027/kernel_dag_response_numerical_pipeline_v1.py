"""CPU-only, source-bound complete repeated-kernel numerical fit or HOLD.

Preparation precedes native data. Sealing reauthenticates the complete native
SQL/DAG chain, and fitting reauthenticates again before using the unchanged
compiler and exact-Hessian/root likelihood helper. No native execution occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from . import kernel_dag_response_replication_continuation_v1 as continuation
from . import kernel_dag_repeated_instance_alignment_v1 as repeated
from . import kernel_dag_response_training_v1 as compiler
from . import kernel_dag_response_fit_v1 as likelihood
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .native_collect import receipt
from .kernel_sequence_audit_v2 import verify
from .twoFIFO_source_rebound_execute_v1 import SourceChangeGuard

SCHEMA='gsdeploy.kernel_dag_response_numerical_pipeline.v1'
need=continuation.need
read=continuation.read
write=continuation.write


def protocol():
    return dict(features=list(compiler.features.FEATURES), offered_captures=3,
        offered_requests=60, requests_per_capture=20, exact_work_templates=4,
        instances_per_template=15, rows_dropped=False, columns_dropped=False,
        starts=[[v]*6 for v in (0.,.25,-.25)], optimizer='trust-exact',
        optimizer_gtol=1e-10, optimizer_maxiter=300, root='lm',
        root_ftol=1e-12, root_xtol=1e-12, root_gtol=1e-12, root_maxiter=300,
        root_solution_agreement=1e-7, stationarity_tolerance=1e-10,
        structural_gate='ALL_WITHIN_EXACT_NODE_GROUP_PATH_CONTRASTS_RANK_SIX',
        actual_root_gate='UNCHANGED_NUISANCE_CERTIFICATE_AND_ALL_THREE_STATIONARY_STRICT_LOCAL_MINIMA',
        beta_zero_derivative_rank_is_hold_gate=False,
        baseline_nuisance='owner_exact_DAG_profile_node_context_profiler',
        target='PROFILER_KERNEL_LIFETIME_POSITIVE_TIME_CHANGE_WITH_ENDPOINT_JACOBIAN',
        response_sigma_floor=None, capture_is_replication_unit=True,
        independent_accuracy_test=False, prospective_prediction_qualified=False)


def checked_continuation_review(ref,manifest_ref):
    review=read(ref)
    need(review.get('schema')=='gsdeploy.kernel_dag_response_replication_continuation_independent_review.v1'
         and review.get('status')=='PASS_CPU_PREPARED_WAIT_EXECUTE_INTERFACES_NOT_NATIVE_QUALIFICATION'
         and review.get('manifest')==manifest_ref and review.get('unresolved_blocking_findings')==[]
         and review.get('test_summary',{}).get('status')=='PASS',
         'Exact independently reviewed native continuation required')
    return review


def prepare(continuation_path,independent_review,output):
    ref=receipt(continuation_path);manifest=read(ref)
    need(manifest.get('schema')==continuation.SCHEMA
         and manifest.get('status')=='CPU_PREPARED_EXPLICIT_WAIT_EXECUTE_REQUIRED'
         and manifest.get('validation_labels_read') is False,
         'Frozen training-only native continuation manifest required')
    review_ref=receipt(independent_review);review=checked_continuation_review(review_ref,ref)
    campaign=continuation.producer.audit(manifest['campaign']['path'])
    need(len(campaign['runs'])==3 and len({e['capture_id'] for e in campaign['runs']})==3,
         'Exact complete three-capture campaign required')
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    sources=strict_references([ref,manifest,review_ref,review,campaign,implementation_sources(__file__)])
    for source in sources:verify(source)
    result=dict(schema=SCHEMA,status='CPU_PREPARED_DATA_SEAL_THEN_EXPLICIT_FIT_REQUIRED',
        continuation=ref,independent_review=review_ref,campaign=manifest['campaign'],
        capture_ids=[e['capture_id'] for e in campaign['runs']],
        continuation_final_path=str(Path(manifest['output'])/'report.json'),
        repeated_design_path=str(Path(manifest['output'])/'repeated-design.json'),
        repeated_report_path=str(Path(manifest['output'])/'repeated-alignment/report.json'),
        protocol=protocol(),sources=sources,GPU_started=False,fit_started=False,
        validation_labels_read=False,training_only=True,no_retries=True)
    write(root/'plan.json',result);return result


def audit(path):
    ref=receipt(path);plan=read(ref)
    need(plan.get('schema')==SCHEMA and plan.get('status')=='CPU_PREPARED_DATA_SEAL_THEN_EXPLICIT_FIT_REQUIRED'
         and plan.get('protocol')==protocol() and plan.get('training_only') is True
         and plan.get('validation_labels_read') is False and plan.get('no_retries') is True,
         'Unchanged complete CPU fit/HOLD protocol required')
    manifest=read(plan['continuation']);review=checked_continuation_review(plan['independent_review'],plan['continuation'])
    campaign=continuation.producer.audit(plan['campaign']['path'])
    need(plan['campaign']==manifest['campaign']
         and plan['capture_ids']==[e['capture_id'] for e in campaign['runs']]
         and plan['continuation_final_path']==str(Path(manifest['output'])/'report.json')
         and plan['repeated_design_path']==str(Path(manifest['output'])/'repeated-design.json')
         and plan['repeated_report_path']==str(Path(manifest['output'])/'repeated-alignment/report.json'),
         'Canonical complete native lifecycle paths required')
    expected=strict_references([plan['continuation'],manifest,plan['independent_review'],review,campaign,
                                implementation_sources(__file__)])
    need(plan['sources']==expected,'Exact current numerical and native source closure required')
    for source in plan['sources']:verify(source)
    return plan


def native_publication(plan):
    final_ref=receipt(plan['continuation_final_path']);final=read(final_ref)
    need(final.get('schema')==continuation.SCHEMA+'.run'
         and final.get('status')=='COMPLETE_THREE_NATIVE_CAPTURES_AND_CPU_DIAGNOSTICS_NO_FIT'
         and final.get('manifest')==plan['continuation'] and final.get('fit_started') is False
         and final.get('validation_labels_read') is False
         and [r['capture_id'] for r in final.get('completed_captures',[])]==plan['capture_ids']
         and len(final.get('completed_cpu_stages',[]))==12,
         'Complete exact native continuation final required before numerical labels')
    report_ref=receipt(plan['repeated_report_path']);report=read(report_ref)
    need(final.get('repeated_alignment')==report_ref
         and report.get('schema')==repeated.SCHEMA and report.get('offered_captures')==3
         and report.get('offered_instances')==60 and report.get('training_only') is True
         and report.get('validation_labels_read') is False and report.get('fit_started') is False
         and report.get('design')==receipt(plan['repeated_design_path']),
         'Complete sixty-instance repeated alignment binding required')
    refs=strict_references([final_ref,final,report_ref,report])
    for ref in refs:verify(ref)
    return final_ref,report,refs


def remap_receipts(value,mapping):
    """Only substitute previously compared temporary proof/export receipts."""
    if isinstance(value,dict):
        if set(value)=={'path','sha256'} and value['path'] in mapping:
            old,new=mapping[value['path']]
            need(value==old,'Unexpected rebuilt receipt under known path')
            return new
        for key,child in value.items():value[key]=remap_receipts(child,mapping)
    elif isinstance(value,list):
        for i,child in enumerate(value):value[i]=remap_receipts(child,mapping)
    return value


def compare_rebuild(published,fresh):
    """Compare full SQLite-rebuilt outputs, including every timing/path field."""
    need(len(published['matched'])==len(fresh['matched']),'Rebuilt exact-work denominator differs')
    mapping={}
    for left,right in zip(published['matched'],fresh['matched']):
        old,new=right['equivalence'],left['equivalence']
        need(read(old)==read(new) and old['sha256']==new['sha256'],
             'Fresh full native template equivalence proof differs')
        mapping[old['path']]=(old,new)
    need(set(published['artifacts'])==set(fresh['artifacts']), 'Fresh complete export set differs')
    for name,old in fresh['artifacts'].items():
        new=published['artifacts'][name]
        need(remap_receipts(read(old),mapping)==read(new),'Fresh complete repeated export differs: '+name)
        mapping[old['path']]=(old,new)
    need(remap_receipts(fresh,mapping)==published,'Fresh repeated alignment report differs')


def authenticated_alignment(plan,report):
    # Each original intake is reconstructed from native SQLite by the unchanged
    # adapter. Temp proofs differ in absolute path only after full content match.
    with tempfile.TemporaryDirectory(prefix='kernel-numerical-reauth-',
            dir=Path(plan['repeated_report_path']).parent.parent) as scratch:
        fresh=repeated.align(plan['repeated_design_path'],Path(scratch)/'rebuilt')
        compare_rebuild(report,fresh)


def unknown_seal(plan_ref,final_ref,report_ref,report):
    return dict(schema=SCHEMA+'.seal',status='UNKNOWN_FULL_INSTANCE_COHORT_RETAINED',
        plan=plan_ref,native_final=final_ref,repeated_alignment=report_ref,
        offered_requests=60,rejected=report['rejected'],GPU_started=False,fit_started=False,
        numerical_qualification=False,validation_labels_read=False)


def compile_complete(plan,report):
    need(report['status']=='COMPLETE_REPEATED_INSTANCE_TEMPLATE_PATHS_NO_FIT'
         and len(report['matched'])==4 and not report['rejected']
         and all(m['instances']==15 for m in report['matched']),
         'UNKNOWN template alignment cannot become a numerical subset')
    design=read(report['design'])
    hardware=repeated.AuditedOccupancyResolver(design['captures'][0]['occupancy_audit']['path']).hardware
    catalog=read(report['artifacts']['catalog.json']);captures=read(report['artifacts']['captures.json'])
    need([c['capture_id'] for c in captures]==plan['capture_ids']
         and all(len(c['offered_request_ids'])==20 for c in captures),
         'Every native capture must retain all twenty measured responses')
    prepared=compiler.compile_paths(catalog,hardware,captures)
    need(prepared['offered_requests']==60,'All sixty request responses required')
    counts=Counter((r['capture_id']) for r in prepared['full_offered_request_registry'])
    need(dict(counts)==dict.fromkeys(plan['capture_ids'],20),'Full response registry differs')
    return prepared


def denominator(prepared):
    digest=hashlib.sha256();counts=Counter()
    for row in prepared['rows']:
        key=(row['capture'],row['request_id'],row['macro_id'])
        digest.update(json.dumps(key,separators=(',',':')).encode()+b'\n');counts[key[:2]]+=1
    return dict(offered_requests=prepared['offered_requests'],
        offered_kernel_count=prepared['offered_kernel_count'],accepted_kernel_count=prepared['accepted_kernel_count'],
        rejected_kernel_count=len(prepared['rejected_rows']),
        full_offered_request_registry=prepared['full_offered_request_registry'],
        ordered_complete_kernel_registry_sha256=digest.hexdigest(),
        accepted_kernels_per_request=[dict(capture_id=c,request_id=r,kernels=n) for (c,r),n in sorted(counts.items())])


def structural_summary(prepared):
    # Exactly the frozen compiler's complete contrast matrix and NumPy rank;
    # no pointwise derivative-rank precheck or feature selection is substituted.
    origins={};contrasts=[]
    for row in prepared['rows']:
        origin=origins.setdefault(row['group'],row['ordered_matrix'][0])
        contrasts.extend(row['ordered_matrix']-origin)
    rank=int(np.linalg.matrix_rank(np.asarray(contrasts,dtype=float))) if contrasts else 0
    return dict(nuisance_groups=len(origins),path_contrast_rank=rank,features=6,
        structural_design_rank=len(origins)+rank,parameters=len(origins)+6,
        beta_zero_derivative_rank_used=False)


def data_record(plan_ref,final_ref,report,prepared):
    return dict(schema=SCHEMA+'.data',plan=plan_ref,native_final=final_ref,
        design=report['design'],artifacts=report['artifacts'],denominator=denominator(prepared),
        structural_rank=structural_summary(prepared),
        all_native_sqlite_freshly_reauthenticated=True,training_only=True,
        validation_labels_read=False,GPU_started=False,fit_started=False)


def seal(plan_path,output):
    plan=audit(plan_path);final_ref,report,refs=native_publication(plan)
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    with SourceChangeGuard(strict_references([receipt(plan_path),plan,refs])) as guard:
        authenticated_alignment(plan,report);guard.check()
        if report['status']!='COMPLETE_REPEATED_INSTANCE_TEMPLATE_PATHS_NO_FIT':
            result=unknown_seal(receipt(plan_path),final_ref,receipt(plan['repeated_report_path']),report)
        else:
            prepared=compile_complete(plan,report)
            result=data_record(receipt(plan_path),final_ref,report,prepared)
            result.update(schema=SCHEMA+'.seal',status='SEALED_COMPLETE_CPU_NUMERICAL_DATA_NO_FIT',
                repeated_alignment=receipt(plan['repeated_report_path']))
        guard.check();write(root/'seal.json',result)
    return result


def fit_sealed(plan_path,seal_path,output):
    # A rehashed seal cannot authenticate caller-modified labels: native rebuild
    # is mandatory again, under the full source change-and-restore guard.
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU runner requires CUDA_VISIBLE_DEVICES empty')
    plan=audit(plan_path);seal_ref=receipt(seal_path);sealed=read(seal_ref)
    need(sealed.get('schema')==SCHEMA+'.seal' and sealed.get('plan')==receipt(plan_path)
         and sealed.get('validation_labels_read') is False,'Exact training data seal required')
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    result=dict(schema=SCHEMA+'.fit',plan=receipt(plan_path),seal=seal_ref,
        GPU_started=False,fit_started=False,validation_labels_read=False,
        numerical_qualification=False,independent_accuracy_test=False,
        prospective_prediction_qualified=False,automatic_retry=False)
    write(root/'claim.json',result)
    try:
        final_ref,report,refs=native_publication(plan)
        with SourceChangeGuard(strict_references([receipt(plan_path),plan,seal_ref,sealed,refs])) as guard:
            authenticated_alignment(plan,report);guard.check()
            if sealed.get('status')=='UNKNOWN_FULL_INSTANCE_COHORT_RETAINED':
                need(report['status']=='UNKNOWN_FULL_INSTANCE_COHORT_RETAINED'
                     and sealed==unknown_seal(receipt(plan_path),final_ref,receipt(plan['repeated_report_path']),report),
                     'UNKNOWN seal/native denominator differs')
                result.update(status='UNKNOWN_FULL_INSTANCE_COHORT_RETAINED',offered_requests=60)
            else:
                need(sealed.get('status')=='SEALED_COMPLETE_CPU_NUMERICAL_DATA_NO_FIT', 'Complete sealed numerical data required')
                prepared=compile_complete(plan,report)
                expected=data_record(receipt(plan_path),final_ref,report,prepared)
                expected.update(schema=SCHEMA+'.seal',status='SEALED_COMPLETE_CPU_NUMERICAL_DATA_NO_FIT',
                                repeated_alignment=receipt(plan['repeated_report_path']))
                need(sealed==expected,'Fresh full numerical data/denominator differs from seal')
                complete=not prepared['rejected_rows'] and prepared['accepted_kernel_count']==prepared['offered_kernel_count']
                result['fit_started']=bool(complete and expected['structural_rank']['path_contrast_rank']==6)
                write(root/'fit-start.json',dict(fit_started=result['fit_started'],structural_rank=expected['structural_rank']))
                info=compiler.fit_paths(prepared)
                result.update(status='COMPLETE_NUMERICAL_TRAINING_DIAGNOSTIC' if info['numerical_qualification'] else 'UNKNOWN_NUMERICAL_QUALIFICATION_FAILED',
                    response_fit=info,numerical_qualification=info['numerical_qualification'],
                    denominator=expected['denominator'],structural_rank=expected['structural_rank'],
                    native_sources_authenticated=True,all_native_sqlite_freshly_reauthenticated=True)
            guard.check()
    except Exception as exc:
        if 'response_fit' in result:
            result['unauthenticated_response_fit_diagnostic']=dict(
                result=result.pop('response_fit'),authentication_passed=False,
                usable_as_qualification=False)
        result.update(status='STOPPED_NO_RETRY_EVIDENCE_PRESERVED',error=type(exc).__name__+': '+str(exc),
            numerical_qualification=False,prospective_prediction_qualified=False,
            native_sources_authenticated=False,all_native_sqlite_freshly_reauthenticated=False,
            response_fit_authenticated=False)
    write(root/'report.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('prepare');p.add_argument('--continuation',required=True);p.add_argument('--independent-review',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('audit');p.add_argument('--plan',required=True)
    p=sub.add_parser('seal');p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('fit');p.add_argument('--plan',required=True);p.add_argument('--seal',required=True);p.add_argument('--output',required=True);p.add_argument('--fit',required=True,action='store_true')
    args=parser.parse_args()
    if args.op=='prepare':result=prepare(args.continuation,args.independent_review,args.output)
    elif args.op=='audit':result=audit(args.plan)
    elif args.op=='seal':result=seal(args.plan,args.output)
    else:result=fit_sealed(args.plan,args.seal,args.output)
    print(json.dumps({k:result[k] for k in ('schema','status','fit_started','error') if k in result}))


if __name__=='__main__':main()
