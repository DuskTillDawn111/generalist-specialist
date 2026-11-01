#!/usr/bin/env python3
"""Summarize four terminal text26 diagnostics; no fitting, utility or raw reopen."""
import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import sys
import numpy as np


def receipt(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def stats(values):
    values=[v for v in values if type(v) in (int,float) and math.isfinite(v)]
    if not values:return dict(count=0,mean=None,p50=None,p95=None,max=None)
    return dict(count=len(values),mean=math.fsum(values)/len(values),
        p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95)),max=max(values))


def summarize(job,expected):
    records=[r['original_record'] for r in job['requests']]
    analyzed=[r for r in job['requests'] if r['diagnostic_status']=='ANALYZED_COMPLETED']
    expected_records=expected['scheduled_records']
    identity=lambda r:(r['arrival_index'],r['task'],r['pool_id'],r.get('model',r.get('alias')))
    planned_match=(len(records)==len(expected_records) and
        sorted(map(identity,records))==sorted(map(identity,expected_records)))
    aliases=sorted({r.get('model') for r in records if r.get('model') is not None})
    window=job['measurement_window'];end=window.get('end_time_ns')
    startunix=window.get('start_unix_ns');duration=window.get('duration_seconds')
    endunix=startunix+round(duration*1e9) if type(startunix) is int and type(duration) in (int,float) else None
    output=[]
    for alias in aliases:
        rawseq=[r for r in records if r.get('model')==alias]
        seq=[r for r in analyzed if r['alias']==alias]
        unknown=sum(r['diagnostic_status']=='UNKNOWN' and r.get('alias')==alias for r in job['requests'])
        timing={}
        if type(end) is int:
            before=lambda r,k:type(r.get(k)) is int and r[k]<=end
            timing=dict(offered_by_window_end=sum(before(r,'scheduled_time_ns') for r in rawseq),
                enqueued_by_window_end=sum(before(r,'enqueued_time_ns') for r in rawseq),
                submitted_by_window_end=sum(before(r,'actual_submit_time_ns') for r in rawseq),
                completed_by_window_end=sum(before(r,'completion_time_ns') and r.get('status')=='completed' for r in rawseq))
            timing['offered_not_completed_at_window_end']=timing['offered_by_window_end']-timing['completed_by_window_end']
            timing['submitted_not_completed_at_window_end']=timing['submitted_by_window_end']-timing['completed_by_window_end']
        clipped_ms=(math.fsum(max(0,min(r['worker_end_unix_ns'],endunix)-max(r['worker_start_unix_ns'],startunix))/1e6 for r in seq)
            if endunix is not None else None)
        tasks=[]
        for task in sorted({r['task'] for r in rawseq}):
            rs=[r for r in rawseq if r['task']==task];ss=[r for r in seq if r['task']==task]
            tasks.append(dict(task=task,offered=len(rs),raw_completed=sum(r.get('status')=='completed' for r in rs),
                analyzed_completed=len(ss),unknown=len(rs)-len(ss),offered_unique_inputs=len({r['pool_id'] for r in rs}),
                analyzed_unique_inputs=len({r['sample_id'] for r in ss}),
                observed_handler_sum_ms=math.fsum(r['worker_handler_ms'] for r in ss),
                observed_service_sum_ms=math.fsum(r['service_ms'] for r in ss),
                service_ms=stats([r['service_ms'] for r in ss]),handler_ms=stats([r['worker_handler_ms'] for r in ss]),
                service_over_baseline=stats([r['service_over_baseline'] for r in ss]),
                submit_to_worker_start_ms=stats([r['ledger']['submit_to_worker_start_ms'] for r in ss]),
                host_schedule_to_submit_ms=stats([r['ledger']['schedule_to_submit_ms'] for r in ss]),
                input_tokens=stats([r['input_tokens'] for r in ss]),generated_tokens=stats([r['generated_tokens'] for r in ss])))
        output.append(dict(alias=alias,offered=len(rawseq),analyzed_completed=len(seq),unknown=unknown,
            timing=timing,completed_handler_overlap_with_offered_window_ms=clipped_ms,
            completed_handler_window_busy_fraction_lower_bound=clipped_ms/(duration*1000) if clipped_ms is not None and duration>0 else None,
            occupancy_scope='Observed completed handler intervals clipped to offered measurement window; unknown handler service omitted, so a lower bound. Wallclock changes/overlaps can invalidate interpretation; inspect diagnostic flags.',
            per_task=tasks,fifo=job['fifo_by_alias'].get(alias)))
    return dict(job_id=job['job_id'],labels=job['labels'],terminal_status=job['terminal_status'],
        original_eligible=job['original_eligible'],eligibility_source=job['eligibility_source'],
        resource_evidence=job['resource_evidence'],original_stability_state=job['original_stability_state'],
        offered_identity_matches_metadata_exposure=planned_match,
        cohort=job['cohort'],aliases=output,ledger_summary=job['ledger_summary'],
        negative_time_flags=job['negative_time_flags'],censoring_limitations=job['censoring_limitations'],U=None)


def paired_comparisons(jobs):
    groups=[]
    for left,right in combinations(jobs,2):
        a=defaultdict(list);b=defaultdict(list)
        for target,job in [(a,left),(b,right)]:
            for r in job['requests']:
                if r['diagnostic_status']=='ANALYZED_COMPLETED':target[(r['family'],r['task'],r['sample_id'])].append(r)
        for family,task in sorted({key[:2] for key in a}&{key[:2] for key in b}):
            common=sorted(k[2] for k in set(a)&set(b) if k[:2]==(family,task))
            per_input=[]
            for sample in common:
                aa=a[(family,task,sample)];bb=b[(family,task,sample)]
                ma=float(np.median([r['service_ms'] for r in aa]));mb=float(np.median([r['service_ms'] for r in bb]))
                per_input.append(dict(sample_id=sample,left_completed=len(aa),right_completed=len(bb),
                    left_service_median_ms=ma,right_service_median_ms=mb,right_over_left=mb/ma,
                    left_generated_tokens=stats([r['generated_tokens'] for r in aa]),
                    right_generated_tokens=stats([r['generated_tokens'] for r in bb]),
                    left_input_tokens=stats([r['input_tokens'] for r in aa]),
                    right_input_tokens=stats([r['input_tokens'] for r in bb])))
            groups.append(dict(left=left['labels'][0],right=right['labels'][0],family=family,task=task,
                left_n=left['plan']['deployment'][family],right_n=right['plan']['deployment'][family],
                common_completed_inputs=len(common),per_input=per_input,
                per_input_median_service_right_over_left=stats([r['right_over_left'] for r in per_input]),
                scope='Equal-input descriptive paired medians, completed-only. Inputs need not have equal counts or overlap context; no causal adjustment, fitting, utility or rank.'))
    return groups


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostics',type=Path,required=True);p.add_argument('--exposure',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    manifests=[]
    for directory in [args.diagnostics,args.exposure]:
        manifest=json.loads((directory/'manifest.json').read_text())
        for filename,digest in manifest['files'].items():
            if Path(filename).name!=filename or receipt(directory/filename)['sha256']!=digest:
                raise ValueError('Diagnostic/exposure file mismatch: '+filename)
        manifests.append(receipt(directory/'manifest.json'))
    report=json.loads((args.diagnostics/'report.json').read_text())
    exposure=json.loads((args.exposure/'report.json').read_text())
    if report.get('waiting') or len(report['jobs'])!=4 or len(exposure['jobs'])!=4:
        raise ValueError('All four predeclared conditions must have terminal diagnostics')
    expected={j['job_id']:json.loads((args.exposure/j['detail_file']).read_text()) for j in exposure['jobs']}
    jobs=[];inputs=[]
    for k,job in enumerate(report['jobs']):
        path=args.diagnostics/f'job-{k+1}-{job["job_id"][:12]}.json'
        jobs.append(json.loads(path.read_text()));inputs.append(receipt(path))
    if set(expected)!={j['job_id'] for j in jobs}:raise ValueError('Exposure and diagnostic rosters differ')
    summarized=[summarize(j,expected[j['job_id']]) for j in jobs]
    comparisons=paired_comparisons(jobs)
    if args.output.exists():raise FileExistsError('Never overwrite analysis evidence')
    args.output.mkdir(parents=True,exist_ok=False)
    limitations=['Completed-only service and tokens can have survival bias; failed/UNKNOWN service remains unknown.',
        'Observed handler window occupancy is a lower bound, not GPU utilization; concurrent aliases share GPU.',
        'Host pre-write submit is not worker line arrival; waiting includes IPC/parse/previous-handler and emit overhead.',
        'Historical per-input baseline protocols are not asserted equivalent; ratios are diagnostic only.',
        'Ledger closure is an algebraic consistency and clock-span diagnostic, not independent validation of a queue/service predictor.',
        'Common-input pairs do not equalize concurrency, generation, thermal state, route mix or residency; no causal interference claim.',
        'No production U, quality rescore, ranking, parameter fit, GPU work or raw file reopen.']
    result=dict(schema='gsdeploy.text26_service_analysis.v1',jobs=summarized,common_input_comparisons=comparisons,
        limitations=limitations,U=None,ranking_performed=False,fitted_parameters=False,raw_files_reopened=0,gpu_started=False)
    (args.output/'report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    lines=['# text26 四条件终态服务诊断','',
        '所有条件均按预声明attempt1保留；本报告不计算U、不排名、不拟合。原生失败仍保留UNKNOWN。','',
        '| 条件 | 终态 | 原资格 | 资源有效 | 可计账完成／UNKNOWN | 计划输入身份吻合 | G2完成handler窗口占用下界 | G2窗口结束积压 |',
        '|---|---|---|---|---:|---|---:|---:|']
    for j in summarized:
        g=next((a for a in j['aliases'] if a['alias']=='generalist_2#0'),None)
        occupancy=g['completed_handler_window_busy_fraction_lower_bound'] if g else None
        lines.append('| '+j['labels'][0]+' | '+j['terminal_status']+' | '+str(j['original_eligible'])+' | '+
            str(j['resource_evidence']['resource_valid'])+' | '+
            str(j['cohort']['analyzed_completed_requests'])+'/'+str(j['cohort']['unknown_requests'])+' | '+
            str(j['offered_identity_matches_metadata_exposure'])+' | '+(f'{occupancy:.6f}' if occupancy is not None else 'UNKNOWN')+' | '+
            str(g['timing'].get('offered_not_completed_at_window_end') if g else None)+' |')
    lines+=['','各alias与任务的服务、handler、历史基线倍率、生成token、host排队及FIFO状态见report.json；共同输入比较保留逐输入重复数和median，不把不同输入的路径均值当配对比较。',
        '原资格只读取raw.evaluation.utility.eligible，资源字段只转录原证据；completed不等于eligible。n3资源无效，仅作服务诊断，不能作为当前资源条件下的正式候选。',
        '窗口占用仅累计有完整时间戳的已完成handler，UNKNOWN没有当作零；若已有下界接近1，可描述该alias已观测handler接近连续占用，不能等同GPU利用率或排除其他瓶颈。',
        '非文本背景样本数量小，且与text共享G2 alias；原生handler时间占比和生成长度可以描述，不能单凭同时发生推断GPU因果干扰。',
        '含UNKNOWN或中止窗口的FIFO被诊断工具标为UNKNOWN_CENSORED；完整单请求时间账本仍能核对，但不能声称已解释未完成队尾或全轨迹延迟。','']
    (args.output/'report.md').write_text('\n'.join(lines))
    (args.output/'manifest.json').write_text(json.dumps(dict(inputs=inputs,source_manifests=manifests,
        tool=receipt(Path(__file__)),files={x.name:receipt(x)['sha256'] for x in args.output.iterdir() if x.is_file()}),indent=2)+'\n')
    print(json.dumps(dict(status='TEXT26_TERMINAL_DIAGNOSTIC_SUMMARY_COMPLETE',jobs=len(jobs),output=str(args.output.resolve()),U=None,gpu_started=False)))


if __name__=='__main__':main()
