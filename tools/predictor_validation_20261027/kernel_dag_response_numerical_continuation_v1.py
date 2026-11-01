"""CPU-only continuation: exact kernel native final, data seal, one fit/HOLD.

No GPU action, resource-chain dependency, feature selection or retry exists.
An exact independent review of the prepared manifest is required at execution.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import sys
import time

from . import kernel_dag_response_numerical_pipeline_v1 as numerical
from . import kernel_dag_response_replication_continuation_v1 as kernel
from .implementation_source_closure_v27 import implementation_sources,strict_references
from .kernel_sequence_audit_v2 import verify
from .native_collect import receipt
from .twoFIFO_source_rebound_execute_v1 import SourceChangeGuard

SCHEMA='gsdeploy.kernel_dag_response_numerical_continuation.v1'
MODULE='tools.predictor_validation_20261027.kernel_dag_response_numerical_continuation_v1'
need=kernel.need
read=kernel.read
write=kernel.write


def prepare(plan_path,kernel_pid,kernel_claim,output):
    plan_ref=receipt(plan_path);plan=numerical.audit(plan_path)
    identity=kernel.continuation_identity(kernel_pid,plan['continuation']['path'])
    claim_ref=receipt(kernel_claim);claim=read(claim_ref)
    manifest=read(plan['continuation'])
    need(claim.get('manifest')==plan['continuation'] and claim.get('no_retries') is True
         and Path(kernel_claim).resolve()==Path(manifest['output'])/'claim.json',
         'Exact running kernel continuation claim required')
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    sources=strict_references([plan_ref,plan,claim_ref,claim,implementation_sources(__file__)])
    for source in sources:verify(source)
    result=dict(schema=SCHEMA,status='CPU_PREPARED_ROOT_REVIEW_THEN_WAIT_SEAL_FIT_REQUIRED',
        numerical_plan=plan_ref,kernel_identity=identity,kernel_claim=claim_ref,
        kernel_final_path=plan['continuation_final_path'],output=str(root/'run-v1'),
        sources=sources,training_only=True,validation_labels_read=False,GPU_started=False,
        fit_started=False,no_retries=True,resource_chain_dependency=False,
        numerical_HOLD_is_completed_outcome=True)
    write(root/'manifest.json',result);return result


def audit(path):
    manifest=read(receipt(path))
    need(manifest.get('schema')==SCHEMA
         and manifest.get('status')=='CPU_PREPARED_ROOT_REVIEW_THEN_WAIT_SEAL_FIT_REQUIRED'
         and manifest.get('no_retries') is True and manifest.get('training_only') is True
         and manifest.get('validation_labels_read') is False
         and manifest.get('resource_chain_dependency') is False
         and manifest.get('numerical_HOLD_is_completed_outcome') is True,
         'Explicit CPU-only nonretrying numerical continuation required')
    plan=numerical.audit(manifest['numerical_plan']['path']);verify(manifest['numerical_plan'])
    claim=read(manifest['kernel_claim']);native_manifest=read(plan['continuation'])
    need(manifest['kernel_identity']['manifest']==plan['continuation']
         and claim.get('manifest')==plan['continuation'] and claim.get('no_retries') is True
         and manifest['kernel_final_path']==plan['continuation_final_path']
         and manifest['kernel_claim']['path']==str(Path(native_manifest['output'])/'claim.json'),
         'Exact kernel controller/claim/final binding required')
    expected=strict_references([manifest['numerical_plan'],plan,manifest['kernel_claim'],claim,implementation_sources(__file__)])
    need(manifest['sources']==expected,'Complete unchanged CPU numerical continuation source closure required')
    for source in expected:verify(source)
    return manifest,plan


def checked_review(review_ref,manifest_ref,plan_ref):
    review=read(review_ref)
    need(review.get('schema')==SCHEMA+'.independent_review'
         and review.get('status')=='PASS_CPU_WAIT_SEAL_FIT_INTERFACES_NOT_NUMERICAL_QUALIFICATION'
         and review.get('manifest')==manifest_ref and review.get('numerical_plan')==plan_ref
         and review.get('unresolved_blocking_findings')==[]
         and type(review.get('test_summary')) is dict
         and review.get('test_summary',{}).get('status')=='PASS',
         'Exact root independent CPU numerical/continuation review required')
    for source in strict_references([review_ref,review]):verify(source)
    return review


def wait_kernel(manifest,plan,check_sources,*,poll_seconds=30,pause=time.sleep,
                identity_reader=kernel.continuation_identity):
    need(type(poll_seconds) in (int,float) and 0<poll_seconds<=60,'Bounded CPU polling interval required')
    identity=manifest['kernel_identity']
    while True:
        check_sources()
        try:current=identity_reader(identity['pid'],plan['continuation']['path'])
        except (FileNotFoundError,ProcessLookupError):current=None
        if current is not None:need(current==identity,'Kernel continuation PID/start ticks/command changed')
        publication=kernel.load_publication(manifest['kernel_final_path'])
        if publication is not None:
            ref,final=publication
            need(final.get('schema')==kernel.SCHEMA+'.run' and final.get('manifest')==plan['continuation'],
                 'Foreign kernel continuation final rejected')
            need(final.get('status')=='COMPLETE_THREE_NATIVE_CAPTURES_AND_CPU_DIAGNOSTICS_NO_FIT',
                 'Kernel continuation stopped/incomplete; never retry native or numerical stages')
            return numerical.native_publication(plan)
        need(current is not None,'Kernel controller exited before complete canonical final')
        pause(poll_seconds)


def temporary_requirement(plan,report):
    native_manifest=read(plan['continuation']);storage=read(native_manifest['storage_scenario'])
    rebuild_bytes=storage['temporary_single_intake_rebuild_bytes']
    exports=strict_references([report.get('artifacts',{}),[m['equivalence'] for m in report['matched']]])
    exported_bytes=sum(Path(ref['path']).stat().st_size for ref in exports)
    need(type(rebuild_bytes) is int and rebuild_bytes>0,'Original measured single-intake temporary scenario required')
    # The unchanged adapter closes all per-intake temporary exports before
    # writing its own complete repeated exports: these peaks do not overlap.
    return max(rebuild_bytes,exported_bytes)


def command(op,plan_path,output,*,seal_path=None):
    args=[sys.executable,'-m','tools.predictor_validation_20261027.kernel_dag_response_numerical_pipeline_v1',
          op,'--plan',str(plan_path),'--output',str(output)]
    if op=='fit':args.extend(['--seal',str(seal_path),'--fit'])
    return args


def run(manifest_path,independent_review,*,poll_seconds=30,run_command=subprocess.run):
    manifest,plan=audit(manifest_path);manifest_ref=receipt(manifest_path);review_ref=receipt(independent_review)
    review=checked_review(review_ref,manifest_ref,manifest['numerical_plan'])
    root=Path(manifest['output']);root.mkdir(parents=True,exist_ok=False)
    result=dict(schema=SCHEMA+'.run',manifest=manifest_ref,independent_review=review_ref,
        status='WAITING_FOR_EXACT_KERNEL_NATIVE_AND_CPU_FINAL',GPU_started=False,fit_started=False,
        numerical_qualification=False,prospective_prediction_qualified=False,
        validation_labels_read=False,resource_chain_dependency=False,automatic_retry=False)
    write(root/'claim.json',result)
    try:
        refs=strict_references([manifest_ref,manifest,review_ref,review])
        with SourceChangeGuard(refs) as guard:
            final_ref,report,native_refs=wait_kernel(manifest,plan,guard.check,poll_seconds=poll_seconds)
            guard.check();result['kernel_final']=final_ref
            with SourceChangeGuard(strict_references([refs,native_refs])) as native_guard, ExitStack() as outputs:
                published_guards=[]
                capacity=kernel.wait_capacity(manifest,temporary_requirement(plan,report),native_guard.check,
                    phase='BEFORE_CPU_SEAL_AND_REAUTHENTICATED_FIT',poll_seconds=poll_seconds)
                write(root/'capacity.json',capacity)
                for op in ('seal','fit'):
                    native_guard.check()
                    for published_guard in published_guards:published_guard.check()
                    out=root/op;seal_path=root/'seal/seal.json'
                    action=dict(op=op,command=command(op,manifest['numerical_plan']['path'],out,seal_path=seal_path))
                    write(root/(op+'-started.json'),action)
                    if op=='fit':result['fit_command_started']=True
                    with (root/(op+'.log')).open('x') as stream:
                        process=run_command(action['command'],stdout=stream,stderr=subprocess.STDOUT,check=False,
                            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
                    native_guard.check()
                    for published_guard in published_guards:published_guard.check()
                    need(process.returncode==0,'CPU '+op+' subprocess failed; no retry')
                    output_ref=receipt(out/('seal.json' if op=='seal' else 'report.json'));output=read(output_ref)
                    need(output.get('schema')==numerical.SCHEMA+'.'+('seal' if op=='seal' else 'fit')
                         and output.get('plan')==manifest['numerical_plan']
                         and output.get('GPU_started') is False and output.get('validation_labels_read') is False,
                         'Exact CPU '+op+' result/role binding required')
                    if op=='seal':
                        need(output.get('status') in ('SEALED_COMPLETE_CPU_NUMERICAL_DATA_NO_FIT','UNKNOWN_FULL_INSTANCE_COHORT_RETAINED')
                             and output.get('fit_started') is False,'Complete data seal or full-cohort HOLD required')
                        result['data_seal']=output_ref
                    else:
                        result['fit_started']=output.get('fit_started',False)
                        need(output.get('seal')==result['data_seal'] and output.get('status') in
                             ('COMPLETE_NUMERICAL_TRAINING_DIAGNOSTIC','UNKNOWN_NUMERICAL_QUALIFICATION_FAILED','UNKNOWN_FULL_INSTANCE_COHORT_RETAINED'),
                             'Complete fit/HOLD result required; stopped child cannot qualify')
                        qualified=output.get('numerical_qualification') is True
                        need(qualified==(output['status']=='COMPLETE_NUMERICAL_TRAINING_DIAGNOSTIC')
                             and output.get('prospective_prediction_qualified') is False,
                             'Numerical HOLD must remain unqualified')
                        result.update(status='COMPLETE_CPU_NUMERICAL_TRAINING_DIAGNOSTIC' if qualified else 'COMPLETE_CPU_NUMERICAL_HOLD',
                            numerical_result=output_ref,numerical_qualification=qualified,fit_started=output['fit_started'])
                    write(root/(op+'-complete.json'),dict(result=output_ref,status=output['status']))
                    published_guards.append(outputs.enter_context(SourceChangeGuard([output_ref])))
                # Include immutable published CPU outputs before final acceptance.
                with SourceChangeGuard(strict_references([result['data_seal'],result['numerical_result']])) as output_guard:
                    native_guard.check();guard.check();output_guard.check()
                    for published_guard in published_guards:published_guard.check()
    except Exception as exc:
        if 'numerical_result' in result:
            result['unauthenticated_numerical_result_diagnostic']=result.pop('numerical_result')
        result.update(status='STOPPED_NO_RETRY_PREFIX_PRESERVED',error=type(exc).__name__+': '+str(exc),
            numerical_qualification=False,prospective_prediction_qualified=False,
            numerical_result_authenticated=False)
    write(root/'report.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('prepare');p.add_argument('--plan',required=True);p.add_argument('--kernel-pid',required=True,type=int);p.add_argument('--kernel-claim',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('audit');p.add_argument('--manifest',required=True)
    p=sub.add_parser('run');p.add_argument('--manifest',required=True);p.add_argument('--independent-review',required=True);p.add_argument('--execute',required=True,action='store_true')
    args=parser.parse_args()
    if args.op=='prepare':result=prepare(args.plan,args.kernel_pid,args.kernel_claim,args.output)
    elif args.op=='audit':result=audit(args.manifest)[0]
    else:result=run(args.manifest,args.independent_review)
    print({k:result[k] for k in ('schema','status','fit_started','error') if k in result})


if __name__=='__main__':main()
