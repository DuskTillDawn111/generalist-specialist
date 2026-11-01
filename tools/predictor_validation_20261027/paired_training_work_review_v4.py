"""Compare actual light/natural training work without transferring timings."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .native_collect import receipt,save
from .smoke_verifier_v3 import check_receipt


FIELDS=('raw_output','prediction','success','finish_reason','input_tokens','generated_tokens',
        'effective_generation','request_sha256','protocol_sha256')
FORWARD_FIELDS=('id','semantic_phase','input_sequence_length','input_batch_size','past_key_values_present')


def actual_work(response):
    return dict(outputs={k:response.get(k) for k in FIELDS},
        forwards=[{k:r.get(k) for k in FORWARD_FIELDS} for r in response.get('semantic_forward_observations',{}).get('records',[])])


def load(path,domain):
    report=json.loads(Path(path).read_text())
    if report.get('schema')!='gsdeploy.startup_training_collection_strict_review.v4' or report.get('status')!='PASS':raise ValueError('Full strict training review PASS required')
    for field in ('raw','plan','context'):
        if not check_receipt(report[field]):raise ValueError('Changed '+field)
    if not all(check_receipt(r) for r in report['evaluator_sources']):raise ValueError('Changed evaluator')
    plan=json.loads(Path(report['plan']['path']).read_text());raw=json.loads(Path(report['raw']['path']).read_text())
    if plan.get('domain')!=domain or plan.get('split')!='training':raise ValueError('Explicit training domain required')
    return report,plan,raw


def review(light_report,natural_report,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    light,lplan,lraw=load(light_report,'light-kernel');natural,nplan,nraw=load(natural_report,'natural')
    issues=[];pairs=[]
    if lplan['jobs']!=nplan['jobs']:issues.append('EXACT_PAIRED_PLAN_JOB_MISMATCH')
    key=lambda r:(r['profile_job_id'],r['model'],r['pool_id'],r['task'])
    if Counter(map(key,lraw['request_records']))!=Counter(map(key,nraw['request_records'])):issues.append('EXACT_PAIRED_REQUEST_COVERAGE_MISMATCH')
    if len({key(r) for r in lraw['request_records']})!=len(lraw['request_records']):issues.append('DUPLICATE_TRIAL_KEYS')
    lookup={key(r):r for r in nraw['request_records']}
    for lrow in lraw['request_records']:
        nrow=lookup.get(key(lrow))
        if nrow is None:continue
        same=actual_work(lrow['worker_response'])==actual_work(nrow['worker_response'])
        if not same:issues.append('ACTUAL_WORK_OR_OUTPUT_MISMATCH:'+lrow['request_id'])
        pairs.append(dict(profile_job_id=lrow['profile_job_id'],alias=lrow['model'],pool_id=lrow['pool_id'],
            actual_work_matched=same,light_request_id=lrow['request_id'],natural_request_id=nrow['request_id'],
            light_handler_ms=lrow['worker_response']['worker_latency_ms'],natural_handler_ms=nrow['worker_response']['worker_latency_ms'],
            light_parent_e2e_ms=(lrow['completion_time_ns']-lrow['scheduled_time_ns'])/1e6,
            natural_parent_e2e_ms=(nrow['completion_time_ns']-nrow['scheduled_time_ns'])/1e6))
    primer=lambda r:(r['model'],r['task'],r['pool_id'],actual_work(r['worker_response']))
    priming_matched=[primer(r) for r in lraw['priming_request_records']]==[primer(r) for r in nraw['priming_request_records']]
    if not priming_matched:issues.append('ACTUAL_PRIMING_COHORT_OR_WORK_MISMATCH')
    result=dict(schema='gsdeploy.paired_training_actual_work_review.v4',status='PASS' if not issues else 'FAIL',issues=issues,
        light_review=receipt(light_report),natural_review=receipt(natural_report),pairs=pairs,
        priming_actual_work_matched=priming_matched,training_only=True,validation_started=False,
        natural_kernel_timing_transfer_validated=False,duration_scaling_used=False,
        cross_process_GPU_clock_or_same_SM_concurrency_qualified=False,evaluator=receipt(__file__),
        interpretation='Matched actual work leaves observed light and natural endpoint durations separate. This is training collection evidence, not independent prediction validation or deployment benefit.')
    save(output/'report.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--light-report',required=True);parser.add_argument('--natural-report',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();r=review(args.light_report,args.natural_report,args.output)
    print(json.dumps(dict(status=r['status'],issues=r['issues'],request_pairs=len(r['pairs']),report=receipt(Path(args.output)/'report.json'))))


if __name__=='__main__':main()
