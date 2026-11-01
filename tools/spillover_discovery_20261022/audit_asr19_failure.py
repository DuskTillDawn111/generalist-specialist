#!/usr/bin/env python3
"""Read-only diagnosis of the preserved seed52 no-split failed attempt."""
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_interim_data import receipt,read,write,csv_write,percentile

BASE=ROOT/'artifacts/static-spillover-study-20261022/development-asr19-replication-v1'
RAW=BASE/'runs/asr19_n2_specialist_share_1p0_seed52-attempt1/run.json'
LOG=BASE/'runs/asr19_n2_specialist_share_1p0_seed52-attempt1.log'
OUTPUT=BASE.parent/'asr19-seed52-failure-audit-v1'
SOURCE=ROOT/'artifacts/static-study-gpu-20261022/execution-source-v5'


def summary(values):
    return dict(n=len(values),mean=statistics.fmean(values) if values else None,
        median=statistics.median(values) if values else None,p95=percentile(values,95) if values else None,
        maximum=max(values,default=None),sum=math.fsum(values))


def main():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    raw=read(RAW);start=raw['measurement_window']['start_time_ns'];end=raw['measurement_window']['end_time_ns'];drain=raw['measurement_window']['drain_end_time_ns']
    references=raw['config']['catalog']['quality']['references']['asr'];inputs=[receipt(RAW),receipt(LOG),receipt(BASE/'experiment-packet.json')]
    baseline=defaultdict(list);baseline_outputs=defaultdict(set)
    for path in raw['config']['catalog']['task_replica_profiles']['asr']['asr']['2']['raw_service_samples']:
        inputs.append(receipt(path))
        for line in Path(path).read_text().splitlines():
            record=json.loads(line)
            if record.get('success'):
                baseline[record['sample_id']].append(record['service_ms']);baseline_outputs[record['sample_id']].add(record['prediction'])
    means={key:statistics.median(values) for key,values in baseline.items()}
    aliases={};request_rows=[];sample_rows=[]
    for alias in raw['replica_to_family']:
        rows=[r for r in raw['request_records'] if r['model']==alias];completed=[r for r in rows if r['status']=='completed'];pending=[r for r in rows if r['status']!='completed']
        sample_counts=Counter(r['pool_id'] for r in rows)
        first_pending=pending[0] if pending else None
        detail=dict(offered=len(rows),statuses=dict(Counter(r['status'] for r in rows)),
            unique_inputs=len(sample_counts),sample_counts=dict(sample_counts),completed_worker_handler_ms=summary([r['worker_latency_ms'] for r in completed]),
            completed_generated_tokens=summary([r['worker_response']['generated_tokens'] for r in completed if r['worker_response'].get('generated_tokens') is not None]),
            completed_e2e_ms=summary([r['e2e_latency_ms'] for r in completed]),
            completed_queue_wait_ms=summary([r['worker_queue_wait_ms'] for r in completed]),
            last_completed=[dict(request_id=r['request_id'],sample_id=r['pool_id'],completion_offset=(r['completion_time_ns']-start)/1e9,worker_ms=r['worker_latency_ms']) for r in completed[-5:]],
            first_pending={k:first_pending.get(k) for k in ('request_id','pool_id','scheduled_offset_seconds','actual_submit_time_ns','submit_write_completed_time_ns','error')} if first_pending else None,
            incomplete_write_finished=sum(r.get('submit_write_completed_time_ns') is not None for r in pending),
            incomplete_send_started_only=sum(r.get('actual_submit_time_ns') is not None and r.get('submit_write_completed_time_ns') is None for r in pending),
            incomplete_never_send_started=sum(r.get('actual_submit_time_ns') is None for r in pending),
            late_responses_after_drain=sum(r.get('late_response_after_drain') is not None for r in pending))
        if alias.startswith('asr'):
            detail.update(offered_audio_duration_seconds=summary([references[r['pool_id']]['duration_s'] for r in rows]),
                offered_baseline_service_ms=summary([means[r['pool_id']] for r in rows]),
                known_baseline_offered_load=math.fsum(means[r['pool_id']] for r in rows)/600000,
                completed_service_over_matching_baseline=math.fsum(r['worker_latency_ms'] for r in completed)/math.fsum(means[r['pool_id']] for r in completed),
                generated_finish_reasons=dict(Counter(r['worker_response'].get('finish_reason') for r in completed)),
                outputs_differ_from_standalone=sum(r['worker_response']['prediction'] not in baseline_outputs[r['pool_id']] for r in completed))
            for sample,count in sample_counts.items():
                group=[r for r in completed if r['pool_id']==sample]
                sample_rows.append(dict(alias=alias,sample_id=sample,offered=count,completed=len(group),audio_seconds=references[sample]['duration_s'],
                    baseline_median_service_ms=means[sample],actual_service_mean_ms=statistics.fmean(r['worker_latency_ms'] for r in group) if group else None,
                    generated_tokens_mean=statistics.fmean(r['worker_response']['generated_tokens'] for r in group) if group else None,
                    outputs_differ_from_standalone=sum(r['worker_response']['prediction'] not in baseline_outputs[sample] for r in group)))
        aliases[alias]=detail
        for r in rows:
            request_rows.append(dict(alias=alias,request_id=r['request_id'],task=r['task'],sample_id=r['pool_id'],status=r['status'],
                error=r.get('error'),scheduled_offset_seconds=r['scheduled_offset_seconds'],
                submit_offset_seconds=(r['actual_submit_time_ns']-start)/1e9 if r.get('actual_submit_time_ns') else None,
                write_completed_offset_seconds=(r['submit_write_completed_time_ns']-start)/1e9 if r.get('submit_write_completed_time_ns') else None,
                completion_offset_seconds=(r['completion_time_ns']-start)/1e9 if r.get('completion_time_ns') else None,
                worker_ms=r.get('worker_latency_ms'),queue_wait_ms=r.get('worker_queue_wait_ms'),
                generated_tokens=r.get('worker_response',{}).get('generated_tokens'),finish_reason=r.get('worker_response',{}).get('finish_reason'),
                late_response=bool(r.get('late_response_after_drain'))))
    sample_sets=[set(aliases[a]['sample_counts']) for a in ('asr#0','asr#1')]
    all_asr=[r for r in raw['request_records'] if r['task']=='asr']
    cycle=32
    periodic=all(all_asr[i]['pool_id']==all_asr[i%cycle]['pool_id'] and all_asr[i]['model']==all_asr[i%cycle]['model'] for i in range(len(all_asr)))
    registry={alias:{k:record.get(k) for k in ('model','execution','shared_worker_status','protocol_sha256','spec_sha256','env','command')} for alias,record in raw['worker_registry'].items()}
    log=LOG.read_text();suspicious=[line for line in log.splitlines() if any(token.lower() in line.lower() for token in ('error','exception','traceback','out of memory','broken','timeout'))]
    report=dict(schema='gsdeploy.asr19_seed52_failure_diagnosis.v1',status='FAILED_ATTEMPT_PRESERVED_NOT_UTILITY_EVIDENCE',
        raw_status=raw['status'],errors=raw['errors'],shutdown=raw['shutdown'],measurement_window=raw['measurement_window'],aliases=aliases,
        periodic_input_alias_lock=dict(pool_size=32,cycle_verified=periodic,asr_alias_input_sets_disjoint=not(sample_sets[0]&sample_sets[1]),
            explanation='Repeated 32-sample order and strict two-alias alternation permanently assign 16 samples to each alias. Which samples land on each alias depends on seed.'),
        worker_registry=registry,log_suspicious_lines=suspicious,utility=raw['evaluation']['utility'],stability=raw['evaluation']['stability'],
        incomplete_total=sum(r['status']!='completed' for r in raw['request_records']),
        no_single_hang_evidence='Both ASR aliases continued returning successful EOS responses until 897.616/899.609 seconds; no single exceptionally long or non-EOS response is recorded. Inflight request without response cannot be diagnosed beyond censoring.',
        slowdown_caveat='Completed handler times greatly exceed matching standalone n2 input medians. Alias input mix explains a fixed imbalance but does not establish the full cause of wall-time slowdown; concurrent execution/environment cause needs independent observation.',
        shutdown_caveat='Frozen close closes stdin/stdout after joins; BrokenPipe is recorded only as shutdown error and shutdown={} loses per-worker exit status. No traceback or OS signal cause is available in saved log.',
        quality_caveat='Incomplete outputs force Q/U null. diagnostic_partial_U excludes Q and is not a score; never compare it with valid U or fill missing U.',
        gpu_started=False,rerun=False,fitted=False)
    write(OUTPUT/'audit.json',report);csv_write(OUTPUT/'requests.csv',request_rows);csv_write(OUTPUT/'asr-input-alias.csv',sample_rows)
    lines=['# ASR1.9 seed52 不分流失败诊断','', '失败由 ASR 队列持续积压和 300 秒排空仍不完整构成；没有记录支持“某一次生成死循环”。两条 ASR 进程到排空末仍返回正常 EOS；通用与 OCR 全部完成。','',
        '| alias | 请求 | 完成 | 未完成 | 完成handler均值ms | 完成handler总秒 |','|---|---:|---:|---:|---:|---:|']
    for a,d in aliases.items():lines.append(f"| {a} | {d['offered']} | {d['statuses'].get('completed',0)} | {d['offered']-d['statuses'].get('completed',0)} | {d['completed_worker_handler_ms']['mean']:.3f} | {d['completed_worker_handler_ms']['sum']/1000:.3f} |")
    lines.extend(['','32 条 ASR 输入反复循环且两个副本严格轮询，已核验每个副本固定收到互不重叠的 16 个输入；因此 seed52 把较慢的一半持续固定给 asr#1。输入混合不平衡是真实现象；不能仅由此解释全部服务时间变慢。'])
    for a in ('asr#0','asr#1'):
        d=aliases[a];lines.append(f"- {a}：画像输入均值 {d['offered_baseline_service_ms']['mean']:.3f}ms，实际已完成handler相同输入总时长/画像={d['completed_service_over_matching_baseline']:.3f}；音频均值 {d['offered_audio_duration_seconds']['mean']:.3f}s；输出不同于独立画像次数 {d['outputs_differ_from_standalone']}。")
    lines.extend(['','最长已完成 ASR handler 约8.23秒，正常 EOS，未见 max-length/异常生成记录。首个未完成请求是排队链头而非已证实的 hang；CSV 保留 send/write/完成时间和 late response。','',
        '共229条请求未完成（ASR0 17、ASR1 212）。Q 和 U 必须保持 null；−30.52485 是缺少 Q 的局部诊断和，不能当总效用比较。shutdown BrokenPipe 出现在清理阶段，shutdown={}，日志没有异常栈；无法确定具体哪个 pipe 或 worker 退出原因，也没有线程栈快照。', '',
        '本报告仅分析已保存 raw/日志/画像/冻结执行源码，不重跑、不修改任何原始记录、不拟合、不查询/启动 GPU。'])
    (OUTPUT/'report.zh.md').write_text('\n'.join(lines)+'\n')
    code=[receipt(SOURCE/'gsdeploy/runtime'/name) for name in ('deployment.py','transport.py','runner.py')]
    if any(receipt(r['path'])!=r for r in inputs):raise ValueError('Inputs changed')
    write(OUTPUT/'manifest.json',dict(inputs=inputs,execution_code=code,script=receipt(__file__),
        outputs=[receipt(p) for p in sorted(OUTPUT.iterdir()) if p.is_file()],input_bytes_unchanged=True,gpu_started=False))
    print(json.dumps(dict(output=str(OUTPUT),incomplete=report['incomplete_total'],periodic=periodic,gpu_started=False)))


if __name__=='__main__':main()
