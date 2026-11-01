"""Exact observed work/launch-order bridge; mismatches remain UNKNOWN.

Compares metadata and light-kernel captures by request and semantic forward,
using actual CUDA launch order and observed static launch attributes. Natural
endpoints remain a separate observation domain. No durations are rescaled.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3

from .native_collect import receipt, save
from .profiling import digest
from .smoke_verifier_v3 import check_receipt
from .trace_export import linux_pid

ATTRIBUTES=('launchType','cacheConfig','sharedMemoryLimitConfig','sharedMemoryExecuted',
            'sharedMemoryRequestedPercentage','clusterSchedulingPolicy')


def request_kernels(parsed,sql,row):
    pid=next(p['pid'] for p in parsed['phases'] if p['request_id']==row['request_id'])
    connection=sqlite3.connect(f'file:{Path(sql).resolve()}?mode=ro',uri=True);connection.row_factory=sqlite3.Row
    launches={}
    for r in connection.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME'):
        if linux_pid(r['globalTid'])==pid:launches[r['correlationId']]=dict(r)
    observed={}
    for r in connection.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'):
        if linux_pid(r['globalPid'])==pid:
            observed[(r['correlationId'],r['start'],r['end'])]={name:r[name] for name in ATTRIBUTES if name in r.keys()}
    connection.close()
    semantic=row['worker_response'].get('semantic_forward_observations',{}).get('records',[])
    ids={r['id'] for r in semantic}
    forward_ranges=[p for p in parsed['phases'] if p['request_id']==row['request_id'] and p['segment_id'] in ids]
    kernels=[]
    for phase in parsed['phases']:
        if phase['request_id']!=row['request_id']:continue
        for kernel in phase['kernels']:
            launch=launches.get(kernel['correlation_id'])
            if launch is None:raise ValueError('Kernel has no genuine CUDA launch record')
            matching=[p for p in forward_ranges if p['nsys_start_ns']<=launch['start'] and launch['end']<=p['nsys_end_ns']]
            if len(matching)>1:raise ValueError('Ambiguous semantic forward range')
            group=matching[0]['segment_id'] if matching else '__outside_language_forwards__'
            extra=observed[(kernel['correlation_id'],kernel['nsys_start_ns'],kernel['nsys_end_ns'])]
            kernels.append(dict(kernel=kernel,launch_ns=launch['start'],group=group,extra=extra,
                                operator_id=phase['segment_id'] if phase['segment_id'].startswith('affine_') else None))
    kernels.sort(key=lambda r:(r['launch_ns'],r['kernel']['nsys_start_ns'],r['kernel']['nsys_end_ns']))
    roles={}
    for item in kernels:
        k=item['kernel'];stream=k['stream']
        if stream not in roles:roles[stream]=len(roles)
        item['signature']=dict(name=k['name'],block=k['block'],grid=k['grid'],registers_per_thread=k['registers_per_thread'],
            static_shared_bytes=k['static_shared_bytes'],dynamic_shared_bytes=k['dynamic_shared_bytes'],cluster=k['cluster'],
            stream_role=roles[stream],**item['extra'])
    return kernels


def compare_sequences(metadata,light):
    """Transfer operator membership only when entire group signatures match."""
    groups=defaultdict(lambda:[[],[]])
    for side,items in enumerate((metadata,light)):
        for item in items:groups[item['group']][side].append(item)
    results=[]
    for name,(left,right) in groups.items():
        ls,rs=[r['signature'] for r in left],[r['signature'] for r in right]
        same=bool(left) and ls==rs
        first=next((i for i,(a,b) in enumerate(zip(ls,rs)) if a!=b),None)
        if first is None and len(ls)!=len(rs):first=min(len(ls),len(rs))
        pairs=[]
        if same:
            for a,b in zip(left,right):
                if a['operator_id']:
                    ka,kb=a['kernel'],b['kernel']
                    pairs.append(dict(operator_id=a['operator_id'],
                        metadata_kernel=dict(correlation_id=ka['correlation_id'],nsys_start_ns=ka['nsys_start_ns']),
                        light_kernel=dict(correlation_id=kb['correlation_id'],nsys_start_ns=kb['nsys_start_ns']),
                        light_kernel_duration_ns=kb['end_ns']-kb['start_ns']))
        results.append(dict(semantic_group=name,status='EXACT_OBSERVED_SIGNATURE_ORDER_MATCH' if same else 'UNKNOWN_DIFFERENT_OBSERVED_KERNEL_SEQUENCE',
            metadata_kernel_count=len(left),light_kernel_count=len(right),first_mismatch_index=first,
            metadata_sequence_sha256=digest(ls),light_sequence_sha256=digest(rs),mapped_operator_kernels=pairs,
            kernel_duration_scaling_used=False))
    return results


def load_report(path,domain):
    path=Path(path).resolve();report=json.loads(path.read_text())
    if report.get('schema')!='gsdeploy.training_collection_strict_review.v3' or report.get('status')!='PASS':
        raise ValueError('Passing strict training review required for '+domain)
    for name in ('raw','plan','context'):
        if not check_receipt(report[name]):raise ValueError('Changed '+domain+' '+name)
    sources=report.get('evaluator_sources',[])
    if not sources or not all(check_receipt(source) for source in sources):
        raise ValueError('Changed or absent '+domain+' evaluator source receipts')
    plan=json.loads(Path(report['plan']['path']).read_text());raw=json.loads(Path(report['raw']['path']).read_text())
    if plan.get('domain')!=domain or plan.get('split')!='training':raise ValueError('Explicit training/domain mismatch')
    for owner in report['owners'].values():
        for name in ('sqlite','parsed','operator_kernel_index'):
            if name in owner and not check_receipt(owner[name]):raise ValueError('Changed '+domain+' '+name)
    return report,plan,raw


def review(metadata_report,light_report,natural_report,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    issues,requests=[],[]
    try:
        domains={}
        for domain,path in [('metadata',metadata_report),('light-kernel',light_report),('natural',natural_report)]:
            domains[domain]=load_report(path,domain)
        mreport,mplan,mraw=domains['metadata'];lreport,lplan,lraw=domains['light-kernel'];nreport,nplan,nraw=domains['natural']
        if mplan['jobs']!=lplan['jobs'] or mplan['jobs']!=nplan['jobs']:raise ValueError('Exact three-domain planned jobs required')
        key=lambda r:(r['profile_job_id'],r['model'],r['pool_id'])
        lookup={d:{key(r):r for r in raw['request_records']} for d,(_,_,raw) in domains.items()}
        for mrow in mraw['request_records']:
            identity=key(mrow);alias=mrow['model'];lrow=lookup['light-kernel'][identity];nrow=lookup['natural'][identity]
            outputs=[r['worker_response'] for r in (mrow,lrow,nrow)]
            fields=('raw_output','prediction','finish_reason','input_tokens','generated_tokens','effective_generation','request_sha256','protocol_sha256')
            same=all(all(w.get(name)==outputs[0].get(name) for name in fields) for w in outputs[1:])
            forward=lambda w:[{k:r.get(k) for k in ('id','semantic_phase','input_sequence_length','input_batch_size','past_key_values_present')}
                              for r in w.get('semantic_forward_observations',{}).get('records',[])]
            same=same and forward(outputs[0])==forward(outputs[1])==forward(outputs[2])
            if not same:issues.append('ACTUAL_WORK_OR_OUTPUT_MISMATCH:'+alias+':'+mrow['request_id'])
            mp=json.loads(Path(mreport['owners'][alias]['parsed']['path']).read_text())
            lp=json.loads(Path(lreport['owners'][alias]['parsed']['path']).read_text())
            ms=request_kernels(mp,mreport['owners'][alias]['sqlite']['path'],mrow)
            ls=request_kernels(lp,lreport['owners'][alias]['sqlite']['path'],lrow)
            groups=compare_sequences(ms,ls) if same else []
            exact=bool(groups) and all(g['status']=='EXACT_OBSERVED_SIGNATURE_ORDER_MATCH' for g in groups)
            if not exact:issues.append('METADATA_TO_LIGHT_KERNEL_MAPPING_UNKNOWN:'+alias+':'+mrow['request_id'])
            path=output/('mapping-'+mrow['request_id'].replace(':','-')+'.json');save(path,groups)
            requests.append(dict(alias=alias,pool_id=mrow['pool_id'],trial_job_id=mrow['profile_job_id'],
                exact_work_matched=same,all_observed_kernel_groups_matched=exact,mapping=receipt(path),
                metadata_handler_ms=outputs[0]['worker_latency_ms'],light_handler_ms=outputs[1]['worker_latency_ms'],
                natural_handler_ms=outputs[2]['worker_latency_ms'],
                mapped_operator_kernel_count=sum(len(g['mapped_operator_kernels']) for g in groups),
                affine_coverage='UNKNOWN_OUTSIDE_LANGUAGE_OBSERVER' if alias=='ocr#0' else 'recognized_leaf_affine_only'))
    except Exception as exc:issues.append('BRIDGE_REVIEW_INCOMPLETE:'+type(exc).__name__+':'+str(exc))
    result=dict(schema='gsdeploy.kernel_three_domain_bridge_review.v3',status='PASS' if not issues else 'PARTIAL_OR_UNKNOWN',
        issues=issues,requests=requests,metadata_review=receipt(metadata_report),light_review=receipt(light_report),natural_review=receipt(natural_report),
        evaluator=receipt(__file__),training_only=True,validation_started=False,
        independent_captures_per_domain=1,minimum_training_capture_requirement_satisfied=False,
        duration_rescaling_used=False,natural_timing_transfer_validated=False,
        interpretation='Exact observed mapping may provide light profiler-domain kernel targets; natural host gaps and whole-request timings stay separate.')
    save(output/'report.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-report',required=True);parser.add_argument('--light-report',required=True);parser.add_argument('--natural-report',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=review(args.metadata_report,args.light_report,args.natural_report,args.output)
    print(json.dumps(dict(status=result['status'],issues=result['issues'],report=receipt(Path(args.output)/'report.json'))))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
