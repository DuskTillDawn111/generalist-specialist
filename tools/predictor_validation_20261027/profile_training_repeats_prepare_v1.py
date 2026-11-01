"""CPU-only independent capture02/03 plans using frozen v6/v3 execution.

This preparation tool is not imported by the profile execution closure. Each
new target has its own standard supervisor manifest, attempt boundary and future
raw. Original completed captures are immutable origins, never retried.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

from . import paired_profile_campaign_v6 as campaign
from . import profile_training_supervisor_v3 as supervisor
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .native_collect import receipt, save

SCHEMA='gsdeploy.profile_training_independent_repeats.v1'
ORDER=((2,'natural'),(2,'kernel'),(3,'natural'),(3,'kernel'))


def checked(ref):
    if receipt(ref['path'])!=ref:raise ValueError('Changed immutable repeat preparation reference')
    value=json.loads(Path(ref['path']).read_text())
    if receipt(ref['path'])!=ref:raise ValueError('Source changed during read')
    return value


def source_capture(packet,mode,review_path):
    run=packet/'training'/mode;plan=campaign.audit(run)
    ref=receipt(review_path);review=checked(ref)
    if (review.get('schema')!='gsdeploy.profile_training_evidence_review.v1'
            or review.get('status')!='PASS_COMPLETE_TRAINING_COLLECTION_NOT_PREDICTOR_QUALIFICATION'
            or review.get('role',{}).get('mode')!=mode or review['role'].get('split')!='training'
            or review.get('training_feature_extraction_eligible') is not True
            or review.get('cohort',{}).get('jobs')!=26
            or review['cohort'].get('measured_requests')!=50
            or review['cohort'].get('priming_requests')!=18):
        raise ValueError('Completed independently reviewed original 26-job training capture required')
    for name in ('plan.json','context.json','raw.json','executed-jobs.json','attempt.json'):
        if review['inputs'].get(name)!=receipt(run/name):
            raise ValueError('Exact completed origin run bindings required')
    for item in review.get('sources',[]):
        if receipt(item['path'])!=item:raise ValueError('Origin review source changed')
    supervisor.completed_execution(review['execution'],run)
    if plan.get('split')!='training' or plan.get('mode')!=mode or len(plan['jobs'])!=26:
        raise ValueError('Original paired v6 training role required')
    return dict(run=str(run),review=ref,plan=receipt(run/'plan.json'),context=receipt(run/'context.json'),
        raw=receipt(run/'raw.json'),execution=review['execution'],pair=plan['pair'])


def prepare(packet,output,*,natural_review,kernel_review):
    packet=Path(packet).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    targets=[packet/('training-repeat%02d'%number)/mode for number,mode in ORDER]
    if any(path.exists() for path in targets):
        raise FileExistsError('Independent repeat target already exists; no reuse/retry')
    if any((packet/('execution-training-repeat%02d-%s'%(number,mode))).exists() for number,mode in ORDER):
        raise FileExistsError('Independent repeat execution output already exists')
    origins={mode:source_capture(packet,mode,path) for mode,path in
             (('natural',natural_review),('kernel',kernel_review))}
    plans={mode:checked(ref['plan']) for mode,ref in origins.items()}
    if (plans['natural']['jobs']!=plans['kernel']['jobs']
            or plans['natural']['inputs']!=plans['kernel']['inputs']
            or plans['natural']['pair']!=plans['kernel']['pair']):
        raise ValueError('Exact shared offered training pair required')
    smokes={}
    # Probe directory semantics with the actual frozen execution gate after the
    # plans exist below. The gate's run.parents[1]/smoke resolves this packet.
    for mode in campaign.MODES:
        smoke=packet/'smoke'/mode;report_ref=receipt(smoke/'smoke-report.json')
        report=checked(report_ref)
        if report.get('status')!='PASS' or report.get('issues')!=[]:
            raise ValueError('Both original successful smoke modes required')
        supervisor.completed_execution(report['execution'],smoke)
        smokes[mode]=dict(run=str(smoke),report=report_ref,execution=report['execution'])
    free_before=shutil.disk_usage(packet).free
    output.mkdir(parents=True,exist_ok=False)
    (output/'manifests').mkdir()
    schedule=[]
    for sequence,(number,mode) in enumerate(ORDER,start=3):
        target=packet/('training-repeat%02d'%number)/mode
        target.mkdir(parents=True,exist_ok=False)
        for name in ('plan.json','context.json'):
            shutil.copyfile(packet/'training'/mode/name,target/name)
            if receipt(target/name)['sha256']!=receipt(packet/'training'/mode/name)['sha256']:
                raise ValueError('Repeat plan/context must be byte-identical to original mode')
        plan=campaign.execution_gate(target)
        if plan!=plans[mode]:raise ValueError('Frozen repeat gate altered the plan')
        manifest_path=output/'manifests'/('capture%02d-%s.json'%(number,mode))
        supervisor.prepare('paired',target,manifest_path)
        supervisor.audit(manifest_path)
        capture=dict(schema=SCHEMA+'.capture',capture_index=number,mode=mode,
            planned_campaign_execution_ordinal=sequence,target=str(target),
            plan=receipt(target/'plan.json'),context=receipt(target/'context.json'),
            supervisor_manifest=receipt(manifest_path),origin=origins[mode],
            paired_public_workload=plan['pair'],original_smokes=smokes,
            replication='independent fresh collector and four newly launched worker processes; same offered jobs and input content',
            no_retry_meaning='Each new target may be attempted once. Failed/partial capture stays preserved; never restart old target.',
            planned_thermal_state='UNCONTROLLED_MEASURE_WITH_EXISTING_NVML_TELEMETRY',
            actual_execution_ordinal=None,actual_start_time_ns=None,actual_end_time_ns=None,
            actual_GPU_temperature_C=None,actual_cold_GPU_state_certified=False,
            process_startup='fresh worker processes with unchanged original priming',
            natural_instrumentation='same CUDA event/query/stdout/phase probes; not uninstrumented',
            profiler_perturbation_absent=mode=='natural',paired_timing_unperturbed=False,
            independent_validation=False,GPU_started=False)
        metadata_path=target/'capture-metadata.json';save(metadata_path,capture)
        schedule.append(dict(capture_index=number,mode=mode,planned_campaign_execution_ordinal=sequence,
            target=str(target),capture_metadata=receipt(metadata_path),supervisor_manifest=receipt(manifest_path),
            execution_output=str(packet/('execution-training-repeat%02d-%s'%(number,mode))),
            original_pair=plan['pair'],launch_preflight_free_bytes_required=supervisor.PREFLIGHT,
            execute_command_not_run=[sys.executable,'-m',
                'tools.predictor_validation_20261027.profile_training_supervisor_v3','execute',
                '--manifest',str(manifest_path),'--output',str(packet/('execution-training-repeat%02d-%s'%(number,mode))),
                '--execute']))
    sources=strict_references([origins,smokes,implementation_sources(__file__),
        [checked(o['context'])['sources'] for o in origins.values()]])
    result=dict(schema=SCHEMA,status='CPU_PREPARED_NOT_EXECUTED_REQUIRES_REVIEW_AND_GPU_DISK_PREFLIGHT',
        packet=str(packet),origins=origins,original_smokes=smokes,schedule=schedule,
        completed_capture_count_per_mode=1,additional_capture_count_per_mode=2,
        resulting_initial_training_repetition_count_per_mode_if_successful=3,
        jobs_per_new_capture=26,measured_requests_per_new_capture=50,priming_requests_per_new_capture=18,
        no_retries=True,original_runs_restarted=False,execution_closure_changed=False,
        shared_original_immutable_execution_sources=True,
        preparation_sources=sources,supervisor_budget=supervisor.BUDGET,
        GPU_launch_preflight_bytes=supervisor.PREFLIGHT,disk_abort_floor_bytes=13*2**30,
        preparation_disk=dict(free_before_bytes=free_before,free_after_bytes=shutil.disk_usage(packet).free),
        known_order_confounding='natural02/kernel02/natural03/kernel03 is fixed, not randomized/counterbalanced; thermal/time drift remains a covariate/limitation',
        thermal_state_not_observed_at_preparation=True,hardware_temperature_not_in_existing_raw_schema=True,
        fit_performed=False,validation_labels_read=False,GPU_started=False,final_gain_eligible=False)
    save(output/'report.json',result)
    audit(output/'report.json')
    return receipt(output/'report.json')


def audit(path):
    report=checked(receipt(path))
    if (report.get('schema')!=SCHEMA or report.get('GPU_started') is not False
            or report.get('supervisor_budget')!=supervisor.BUDGET
            or report.get('GPU_launch_preflight_bytes')!=supervisor.PREFLIGHT
            or report.get('disk_abort_floor_bytes')!=13*2**30
            or [(r['capture_index'],r['mode']) for r in report['schedule']]!=list(ORDER)):
        raise ValueError('Exact independent-repeat schema/order/budgets required')
    packet=Path(report['packet'])
    for ref in report['preparation_sources']:
        if receipt(ref['path'])!=ref:raise ValueError('Repeat preparation/source drift')
    for row in report['schedule']:
        expected=packet/('training-repeat%02d'%row['capture_index'])/row['mode']
        if row['target']!=str(expected) or expected.parents[1]!=packet:
            raise ValueError('Original packet smoke gate directory semantics required')
        meta=checked(row['capture_metadata']);manifest=supervisor.audit(row['supervisor_manifest']['path'])
        if receipt(row['supervisor_manifest']['path'])!=row['supervisor_manifest']:
            raise ValueError('Repeat supervisor binding changed')
        if (meta['target']!=row['target'] or meta['supervisor_manifest']!=row['supervisor_manifest']
                or meta['planned_campaign_execution_ordinal']!=row['planned_campaign_execution_ordinal']
                or manifest['binding']['target']!=row['target']
                or manifest['binding']['plan']!=meta['plan'] or manifest['binding']['context']!=meta['context']):
            raise ValueError('Exact independent target/metadata/supervisor binding required')
        campaign.execution_gate(expected)
        origin=report['origins'][row['mode']]
        if any(receipt(expected/name)['sha256']!=origin[key]['sha256'] for name,key in
               (('plan.json','plan'),('context.json','context'))):
            raise ValueError('Repeat copied execution identity differs from origin')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('prepare');p.add_argument('--packet',required=True);p.add_argument('--output',required=True)
    p.add_argument('--natural-review',required=True);p.add_argument('--kernel-review',required=True)
    p=sub.add_parser('audit');p.add_argument('--report',required=True)
    args=parser.parse_args()
    if args.op=='prepare':print(json.dumps(prepare(args.packet,args.output,
        natural_review=args.natural_review,kernel_review=args.kernel_review)))
    else:print(json.dumps(dict(status=audit(args.report)['status'])))


if __name__=='__main__':main()
