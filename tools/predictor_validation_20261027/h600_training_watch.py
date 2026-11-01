"""One-attempt CPU continuation after the declared H600 campaign completes.

Polls only the completion marker while collection is active. Never launches GPU,
full16 fitting, model selection, independent validation, or prediction sealing.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .native_flow import checked,receipt,save
from .probe_validation import read
from . import run_aware_report, h600_resources_v1

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/predictor-gpu-20261027'


def emit(event,**data):
    print(json.dumps(dict(event=event,time_ns=time.perf_counter_ns(),**data)),flush=True)


def configuration(output):
    campaign=BASE/'continuous-training-v14-h600/campaign.json'
    suite=BASE/'continuous-training-v14-h600/primary-training-suite.json'
    protocol=BASE/'background-statistical/run-aware-v14/protocol.json'
    resource_protocol=BASE/'resources-h600-v1/protocol.json'
    quality=BASE/'background-statistical/training-v8-quality.json'
    thresholds=BASE/'background-statistical/selection-threshold-source-addendum.json'
    refs=[*run_aware_report.predictor_sources(),*h600_resources_v1.sources(),receipt(__file__),
        *[receipt(p) for p in (campaign,suite,protocol,resource_protocol,quality,thresholds)]]
    for path in (protocol,resource_protocol):refs.extend(read(path)['source_receipts'])
    unique={}
    for ref in refs:
        checked(ref)
        if ref['path'] in unique and unique[ref['path']]!=ref:raise ValueError('Conflicting frozen source receipts')
        unique[ref['path']]=ref
    return dict(schema='gsdeploy.H600_one_attempt_CPU_continuation.v1',created_ns=time.perf_counter_ns(),
        campaign=receipt(campaign),primary_suite=receipt(suite),training_protocol=receipt(protocol),
        resource_protocol=receipt(resource_protocol),quality=receipt(quality),threshold_addendum=receipt(thresholds),
        completion_marker=str(BASE/'continuous-training-v14-h600/completed-main.json'),
        startup_marker=receipt(BASE/'continuous-training-v14-h600/started-main.json'),
        service_output=str(BASE/'background-statistical/run-aware-v14-cv'),
        resource_output=str(BASE/'resources-h600-v1/fitted'),
        report_output=str(BASE/'background-statistical/run-aware-v14-H600-report'),
        source_receipts=list(unique.values()),poll_seconds=30,GPU_started=False,
        full_fit_authorized=False,seal_authorized=False,one_attempt=True,retries=False)


def check_sources(config):
    for ref in config['source_receipts']:checked(ref)


def audit_ready(config):
    """Only called after marker and all eight immutable scoring files exist."""
    from .flow_crossfit_report import scored_heldout
    from .flow_profiles import map_flow_collection
    from .training_run_audit import verify_quality_annotation
    marker=read(config['completion_marker'])
    started=read(checked(config['startup_marker']))
    if (marker.get('campaign')!=config['campaign'] or marker.get('jobs')!=10
            or started.get('campaign')!=config['campaign'] or started.get('phase')!='main'
            or type(marker.get('completed_ns')) is not int or marker['completed_ns']<=started['created_ns']):
        raise ValueError('Completion marker does not bind the frozen main campaign')
    if marker.get('status') in ('failed','FAIL','FAILED'):
        raise ValueError('Campaign completion marker reports failure')
    suite=read(checked(config['primary_suite']))
    if len(suite['jobs'])!=8:raise ValueError('Exactly eight primary runs required')
    audits=[]
    for job in suite['jobs']:
        raw_path=Path(job['output'])/'raw.json';raw,scored=scored_heldout(receipt(raw_path))
        plan=read(checked(job['plan']))
        if (raw.get('status')!='completed' or raw.get('collection_complete') is not True
                or raw.get('plan')!=job['plan'] or raw.get('candidate_id')!=job['candidate_id']
                or len(raw.get('request_records',[]))!=1200
                or any(raw.get(k) for k in ('errors','runtime_errors','telemetry_errors','qualification_issues'))):
            raise ValueError('Completed clean primary raw required: '+str(raw_path))
        mapped=map_flow_collection(raw_path)
        quality=verify_quality_annotation(raw,plan,read(checked(plan['catalog'])))
        audits.append(dict(raw=receipt(raw_path),scored_raw=scored,run_id=job['run_id'],
            quality=quality,mapping_report=mapped['mapping_report']))
    return audits


def run_fits(config):
    commands=[('service_CV',[sys.executable,'-u','-m','tools.predictor_validation_20261027.run_aware_training','fit',
        '--protocol',config['training_protocol']['path'],'--output',config['service_output']]),
        ('resources',[sys.executable,'-u','-m','tools.predictor_validation_20261027.h600_resources_v1','fit',
        '--protocol',config['resource_protocol']['path'],'--output',config['resource_output']])]
    processes=[]
    try:
        for label,command in commands:
            emit('starting_CPU_fit',component=label,command=command)
            processes.append((label,subprocess.Popen(command,cwd=ROOT,env=os.environ.copy())))
        pending=dict(processes)
        while pending:
            for label,process in list(pending.items()):
                code=process.poll()
                if code is not None:
                    emit('CPU_fit_exited',component=label,returncode=code)
                    if code:raise RuntimeError(label+' failed; no retry or report')
                    del pending[label]
            if pending:time.sleep(1)
    finally:
        for label,process in processes:
            if process.poll() is None:process.terminate()
        for label,process in processes:
            if process.poll() is None:
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()


def run(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        if os.environ.get(key)!='1':raise ValueError('Single-thread environment required: '+key)
    config=configuration(output)
    for key in ('service_output','resource_output','report_output'):
        if Path(config[key]).exists():raise FileExistsError(config[key])
    output.mkdir(parents=True,exist_ok=False);save(output/'configuration.json',config)
    emit('waiting_for_completed_main',marker=config['completion_marker'],poll_seconds=30)
    try:
        while True:
            if Path(config['completion_marker']).exists():
                suite=read(checked(config['primary_suite']))
                if all((Path(j['output'])/name).is_file() for j in suite['jobs'] for name in ('raw.json','scored-raw.json')):
                    break
            time.sleep(30)
        check_sources(config);audits=audit_ready(config)
        save(output/'primary-readiness-audit.json',dict(status='PASS',runs=audits,GPU_started=False))
        for key in ('service_output','resource_output','report_output'):
            if Path(config[key]).exists():raise FileExistsError(config[key])
        save(output/'started-CPU.json',dict(created_ns=time.perf_counter_ns(),source=receipt(__file__),GPU_started=False))
        run_fits(config);check_sources(config)
        command=[sys.executable,'-u','-m','tools.predictor_validation_20261027.run_aware_report',
            '--manifest',str(Path(config['service_output'])/'seed-crossfit-manifest.json'),
            '--resources',str(Path(config['resource_output'])/'full-resources.json'),
            '--resource-manifest',str(Path(config['resource_output'])/'manifest.json'),
            '--quality',config['quality']['path'],'--threshold-addendum',config['threshold_addendum']['path'],
            '--output',config['report_output']]
        emit('starting_H600_training_CV_report',command=command)
        subprocess.run(command,cwd=ROOT,env=os.environ.copy(),check=True);check_sources(config)
        report=Path(config['report_output'])/'report.json';value=read(report)
        completion=dict(status='COMPLETED',report=receipt(report),requested=value['requested'],known=value['known'],
            original_threshold_assessment=value['original_threshold_assessment'],created_ns=time.perf_counter_ns(),
            full_fit_performed=False,independent_validation=False,GPU_started=False)
        save(output/'completed-CPU.json',completion);emit('completed_H600_training_CV',**completion)
    except BaseException as exc:
        failure=dict(status='FAILED_STOPPED_NO_RETRY',error=repr(exc),created_ns=time.perf_counter_ns(),GPU_started=False)
        save(output/'failed-CPU.json',failure);emit('CPU_continuation_failed',**failure)
        traceback.print_exc();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    run(p.parse_args().output)
