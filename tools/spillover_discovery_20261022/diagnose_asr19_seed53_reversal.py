#!/usr/bin/env python3
"""CPU-only observed-time diagnosis; never changes eligibility or estimates a predictor."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'artifacts/static-spillover-study-20261022'
AUDIT = BASE / 'asr19-replication-development-audit-v3'
OUTPUT = BASE / 'asr19-seed53-reversal-diagnostic-v1'
IDS = ('seed53_n2_split', 'seed53_n2_nosplit', 'seed51_n2_split')


def receipt(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def read(path):
    return json.loads(Path(path).read_text())


def summary(xs):
    xs = sorted(xs)
    def pct(p):
        if not xs: return None
        pos = (len(xs)-1)*p; i = int(pos)
        return xs[i] + (xs[min(i+1, len(xs)-1)]-xs[i])*(pos-i)
    return dict(n=len(xs), mean=statistics.fmean(xs) if xs else None,
                median=statistics.median(xs) if xs else None, p95=pct(.95),
                maximum=max(xs, default=None), sum=math.fsum(xs))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def period(seq):
    return next((n for n in range(1, min(len(seq)//2, 256)+1)
                 if all(seq[i] == seq[i % n] for i in range(len(seq)))), None)


def totals(rows):
    b=math.fsum(r['baseline_ms'] for r in rows)
    service=math.fsum(r['service_ms'] for r in rows)
    handler=math.fsum(r['handler_ms'] for r in rows)
    return dict(requests=len(rows), unique_inputs=len({r['sample_id'] for r in rows}),
                baseline_ms=summary([r['baseline_ms'] for r in rows]),
                service_ms=summary([r['service_ms'] for r in rows]),
                handler_ms=summary([r['handler_ms'] for r in rows]),
                service_over_matching_baseline=service/b if b else None,
                handler_over_matching_baseline=handler/b if b else None,
                queue_wait_ms=summary([r['queue_wait_ms'] for r in rows]),
                schedule_delay_ms=summary([r['schedule_delay_ms'] for r in rows]),
                e2e_ms=summary([r['e2e_ms'] for r in rows]),
                generated_tokens=summary([r['generated_tokens'] for r in rows if r['generated_tokens'] is not None]),
                finish_reasons=dict(Counter(r['finish_reason'] for r in rows)),
                cache_reclaims=sum(r['cache_reclaim'] for r in rows),
                changed_outputs=sum(r['differs_from_baseline_outputs'] for r in rows))


def main():
    manifest=read(AUDIT/'manifest.json'); inputs=[receipt(AUDIT/'manifest.json')]
    expected={r['path']:r['sha256'] for r in manifest['inputs']+manifest['outputs']}
    def checked(path):
        rec=receipt(path)
        if expected.get(rec['path']) != rec['sha256']: raise ValueError(f'Unbound/changed: {path}')
        inputs.append(rec); return read(path)
    audit=checked(AUDIT/'audit.json')
    conditions={c['id']:c for c in audit['conditions'] if c['id'] in IDS}
    if set(conditions)!=set(IDS): raise ValueError('Missing selected condition')
    raws={cid:checked(conditions[cid]['run_path']) for cid in IDS}
    baseline={}; profile_paths=set(); baseline_outputs=defaultdict(set)
    request_rows=[]; alias_rows=[]; stage_rows=[]; input_rows=[]; power_rows=[]; fine_rows=[]; reports={}
    for cid,raw in raws.items():
        start=raw['measurement_window']['start_time_ns']; end=raw['measurement_window']['end_time_ns']
        nmap=Counter(raw['replica_to_family'].values())
        catalog=raw['config']['catalog']; records=[]
        for key, phase in [('priming_request_records','priming'),('warmup_request_records','warmup'),('request_records','measurement')]:
            for r in raw[key]:
                if r['status']!='completed': raise ValueError('Diagnostic expects all-completed selected inputs')
                alias=r['model']; fam=raw['replica_to_family'][alias]; task=r['task']
                bk=(fam,task,nmap[fam])
                if bk not in baseline:
                    samples=defaultdict(list)
                    for path in catalog['task_replica_profiles'][fam][task][str(nmap[fam])]['raw_service_samples']:
                        profile_paths.add(path)
                        for line in Path(path).read_text().splitlines():
                            v=json.loads(line)
                            if v.get('success'):
                                samples[v['sample_id']].append(v['service_ms'])
                                baseline_outputs[bk+(v['sample_id'],)].add(v['prediction'])
                    baseline[bk]={s:statistics.median(v) for s,v in samples.items()}
                response=r['worker_response']; sample=r['pool_id']
                # Map a worker wall-clock interval through this request's paired host submit timestamps.
                # Do not subtract the worker wall clock directly from host perf_counter.
                submit_offset=(r['actual_submit_time_ns']-start)/1e9
                ws=submit_offset+(response['start_ns']-response['submit_ns'])/1e9
                we=submit_offset+(response['end_ns']-response['submit_ns'])/1e9
                row=dict(condition=cid,phase=phase,request_id=r['request_id'],alias=alias,family=fam,task=task,
                    sample_id=sample,scheduled_offset_s=r.get('scheduled_offset_seconds'),submit_offset_s=submit_offset,
                    worker_start_offset_s=ws,worker_end_offset_s=we,completion_offset_s=(r['completion_time_ns']-start)/1e9,
                    baseline_ms=baseline[bk][sample],service_ms=response['service_ms'],handler_ms=r['worker_latency_ms'],
                    handler_nonservice_ms=r['worker_latency_ms']-response['service_ms'],queue_wait_ms=r['worker_queue_wait_ms'],
                    schedule_delay_ms=r['schedule_delay_ms'],e2e_ms=r['e2e_latency_ms'],
                    generated_tokens=response.get('generated_tokens'),finish_reason=response.get('finish_reason'),
                    cache_reclaim=response.get('cache_reclamation',{}).get('performed',False),
                    differs_from_baseline_outputs=response['prediction'] not in baseline_outputs[bk+(sample,)])
                records.append(row); request_rows.append(row)
        measured=[r for r in records if r['phase']=='measurement']
        groups=defaultdict(list)
        for r in measured: groups[(r['alias'],r['task'])].append(r)
        alias_detail={}
        for (alias,task),rows in groups.items():
            d=totals(rows); d['input_counts']=dict(Counter(r['sample_id'] for r in rows))
            d['observed_handler_seconds_per_600s']=d['handler_ms']['sum']/600000
            alias_detail[f'{alias}/{task}']=d
            alias_rows.append(dict(condition=cid,alias=alias,task=task,requests=len(rows),
                baseline_mean_ms=d['baseline_ms']['mean'],service_mean_ms=d['service_ms']['mean'],
                handler_mean_ms=d['handler_ms']['mean'],handler_total_s=d['handler_ms']['sum']/1000,
                service_ratio=d['service_over_matching_baseline'],handler_ratio=d['handler_over_matching_baseline'],
                queue_mean_ms=d['queue_wait_ms']['mean'],queue_p95_ms=d['queue_wait_ms']['p95'],
                handler_seconds_per_600s=d['observed_handler_seconds_per_600s']))
            for sample in sorted({r['sample_id'] for r in rows}):
                sub=[r for r in rows if r['sample_id']==sample]; sd=totals(sub)
                input_rows.append(dict(condition=cid,alias=alias,task=task,sample_id=sample,requests=len(sub),
                    baseline_ms=sd['baseline_ms']['mean'],service_mean_ms=sd['service_ms']['mean'],
                    handler_mean_ms=sd['handler_ms']['mean'],service_ratio=sd['service_over_matching_baseline'],
                    queue_mean_ms=sd['queue_wait_ms']['mean']))
        stages={}
        stage_defs=[('priming',None,None),('warmup',None,None)]+[(f'measurement_{lo}_{lo+100}',lo,lo+100) for lo in range(0,600,100)]+[('drain',600,float('inf'))]
        for name,lo,hi in stage_defs:
            groups=defaultdict(list)
            for r in records:
                use=(r['phase']==name) if lo is None else r['phase']=='measurement' and lo<=r['worker_start_offset_s']<hi
                if use: groups[(r['alias'],r['task'])].append(r)
            for (alias,task),rows in groups.items():
                d=totals(rows); stages[f'{name}/{alias}/{task}']=d
                stage_rows.append(dict(condition=cid,stage=name,alias=alias,task=task,requests=len(rows),
                    baseline_mean_ms=d['baseline_ms']['mean'],service_mean_ms=d['service_ms']['mean'],
                    handler_mean_ms=d['handler_ms']['mean'],handler_ratio=d['handler_over_matching_baseline'],
                    queue_mean_ms=d['queue_wait_ms']['mean'],generated_tokens_mean=d['generated_tokens']['mean']))
        for lo in range(0,810,30):
            for alias in ('asr#0','asr#1','generalist_2#0','ocr#0'):
                rows=[r for r in measured if r['alias']==alias and lo<=r['worker_start_offset_s']<lo+30]
                if rows:
                    d=totals(rows)
                    fine_rows.append(dict(condition=cid,start_s=lo,end_s=lo+30,alias=alias,requests=len(rows),
                        handler_ratio=d['handler_over_matching_baseline'],service_mean_ms=d['service_ms']['mean'],
                        queue_mean_ms=d['queue_wait_ms']['mean']))
        asr=[r for r in measured if r['task']=='asr']
        poolseq=[r['sample_id'] for r in asr]; joint=[(r['sample_id'],r['alias']) for r in asr]
        sets={a:sorted({r['sample_id'] for r in asr if r['alias']==a}) for a in sorted({r['alias'] for r in asr})}
        assignment=dict(input_period=period(poolseq),joint_input_alias_period=period(joint),input_sets=sets,
            asr_alias_overlap=sorted(set(sets.get('asr#0',[]))&set(sets.get('asr#1',[]))),
            g2_asr_input_count=len(sets.get('generalist_2#0',[])),
            prefix=[dict(asr_index=i,sample_id=r['sample_id'],alias=r['alias']) for i,r in enumerate(asr[:128])])
        power={}
        ps=raw['energy_evidence']['samples']
        for lo in range(0,900,100):
            vals=[r['watts'] for r in ps if lo<=(r['time_ns']-start)/1e9<lo+100]
            if vals:
                d=summary(vals); power[f'{lo}_{lo+100}']=d
                power_rows.append(dict(condition=cid,start_s=lo,end_s=lo+100,**d))
        # Search scalar/sample field names only; no external telemetry/GPU tools are used.
        def telemetry_keys(obj,p=''):
            found=[]
            if isinstance(obj,dict):
                for k,v in obj.items():
                    if any(s in k.lower() for s in ('temperature','clock','throttle')) and k!='clock': found.append(p+'.'+k)
                    if k not in ('request_records','warmup_request_records','priming_request_records'): found.extend(telemetry_keys(v,p+'.'+k))
            elif isinstance(obj,list) and obj: found.extend(telemetry_keys(obj[0],p+'[0]'))
            return found
        reports[cid]=dict(raw_status=raw['status'],U=conditions[cid]['U'],validity=conditions[cid]['validity'],
            stability=raw['evaluation']['stability'],measurement_window=raw['measurement_window'],aliases=alias_detail,
            stages=stages,input_assignment=assignment,power_watts_by100s=power,
            temperature_clock_throttle_fields=telemetry_keys(raw),
            baseline_scope='Per-input median service_ms from original same-family/task replica-count profile (ASR n2; G2 n1; OCR n1). Handler ratio includes wrapper overhead; service ratio does not.',
            time_scope='Stage by observed worker handler start, mapped through request host-submit perf and paired worker wall timestamp deltas. Measurement cohort may execute during bounded drain.')
    for path in sorted(profile_paths): inputs.append(receipt(path))
    # Same seed has the same task/input cohort but different replica assignments; compare identical inputs globally.
    cross=[]
    ref=[r for r in request_rows if r['condition']=='seed53_n2_nosplit' and r['phase']=='measurement' and r['family']=='asr']
    cur=[r for r in request_rows if r['condition']=='seed53_n2_split' and r['phase']=='measurement' and r['family']=='asr']
    for r in cur:
        matches=[v['service_ms'] for v in ref if v['sample_id']==r['sample_id']]
        if matches: cross.append(dict(alias=r['alias'],sample_id=r['sample_id'],service_ratio_vs_same_input_nosplit=r['service_ms']/statistics.median(matches)))
    within=[]
    for alias in ('asr#0','asr#1'):
        rows=[r for r in cur if r['alias']==alias]
        for sample in sorted({r['sample_id'] for r in rows}):
            sample_rows=[r for r in rows if r['sample_id']==sample]
            early=[r['service_ms'] for r in sample_rows if r['worker_start_offset_s']<400]
            late=[r['service_ms'] for r in sample_rows if 400<=r['worker_start_offset_s']<600]
            drain=[r['service_ms'] for r in sample_rows if r['worker_start_offset_s']>=600]
            within.append(dict(alias=alias,sample_id=sample,early_n=len(early),late_n=len(late),drain_n=len(drain),
                early_mean_service_ms=statistics.fmean(early) if early else None,
                late_mean_service_ms=statistics.fmean(late) if late else None,
                drain_mean_service_ms=statistics.fmean(drain) if drain else None,
                late_over_early=statistics.fmean(late)/statistics.fmean(early) if late and early else None,
                drain_over_early=statistics.fmean(drain)/statistics.fmean(early) if drain and early else None))
    registry_equal={cid:raws[cid]['worker_registry']==raws[IDS[0]]['worker_registry'] for cid in IDS}
    seed53_trace_equal=raws[IDS[0]]['measurement_window']['trace_sha256']==raws[IDS[1]]['measurement_window']['trace_sha256']
    if not seed53_trace_equal: raise ValueError('Same seed trace mismatch')
    report=dict(schema='gsdeploy.asr19_seed53_reversal_diagnostic.v1',conditions=reports,
        worker_registry_equal_to_seed53_split=registry_equal,
        same_seed_trace_equal=seed53_trace_equal,
        same_input_within_split_service_changes=within,
        same_seed_specialist_input_matched_service_ratio={a:summary([r['service_ratio_vs_same_input_nosplit'] for r in cross if r['alias']==a]) for a in ('asr#0','asr#1')},
        scope='Read-only development diagnosis. Observed service changes and periodic input assignment are descriptive; no causal identification of interference, scheduling, clocks or thermal effects.',
        gpu_started=False,rerun=False,fitted=False,stability_rules_changed=False)
    OUTPUT.mkdir(parents=True,exist_ok=False)
    write(OUTPUT/'diagnostic.json',report)
    for name,rows in [('requests.csv',request_rows),('alias-task-summary.csv',alias_rows),('worker-start-stages.csv',stage_rows),('input-alias-summary.csv',input_rows),('board-power-100s.csv',power_rows),('same-seed-input-matched.csv',cross),('worker-start-30s.csv',fine_rows),('same-input-within-split.csv',within)]: csv_write(OUTPUT/name,rows)
    lines=['# ASR1.9 seed53 分流反转诊断','',
        '三条已审计完整原始记录保持原有效性和效用。以下按相同输入匹配原画像服务中位数，区分 worker 内部 service、handler、排队与调度延迟；ASR 基线为原 n2 画像，G2/OCR 为 n1。未重跑、未拟合、未查询 GPU。','',
        '| 条件 | alias / task | 请求 | 画像ms | 实际service ms | service倍率 | handler倍率 | queue均值ms | handler总秒 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in alias_rows:
        lines.append(f"| {r['condition']} | {r['alias']}/{r['task']} | {r['requests']} | {r['baseline_mean_ms']:.2f} | {r['service_mean_ms']:.2f} | {r['service_ratio']:.3f} | {r['handler_ratio']:.3f} | {r['queue_mean_ms']:.2f} | {r['handler_total_s']:.2f} |")
    lines+=['','阶段按 handler 真正开始时间分组，避免把排队等待当服务变化。measurement_0_100 等为窗口内每100秒，drain 为600秒后才开始执行的测量请求。完整阶段见 worker-start-stages.csv。','',
        '| 条件 | 阶段 | ASR0 handler倍率 | ASR1 handler倍率 |', '|---|---|---:|---:|']
    for cid in IDS:
        for stage in ['warmup']+[f'measurement_{lo}_{lo+100}' for lo in range(0,600,100)]+['drain']:
            ds=[reports[cid]['stages'].get(f'{stage}/{a}/asr') for a in ('asr#0','asr#1')]
            if any(ds): lines.append(f"| {cid} | {stage} | "+' | '.join(f"{d['handler_over_matching_baseline']:.3f}" if d else '无请求' for d in ds)+' |')
    lines+=['','输入周期与 alias 分配：']
    for cid in IDS:
        d=reports[cid]['input_assignment'];lines.append(f"- {cid}：输入周期 {d['input_period']}，输入+alias周期 {d['joint_input_alias_period']}；ASR0/1共享输入 {len(d['asr_alias_overlap'])}，G2分到 {d['g2_asr_input_count']} 个固定输入。各输入计数和前128请求见 JSON/CSV。")
    lines+=['','反转发生在服务层，而非单纯评价公式或请求量变化。53split 比53nosplit少送135条ASR给专用，但ASR0/1的实际handler总秒仍从406.89/455.52升至627.82/794.37，均超过单个600秒窗口。均值队列等待从0.538/0.946秒升至78.701/98.233秒。这个实际服务需求与窗口不相容，直接解释了长排队和延迟项恶化；U和原recoverable_congested有效性均保持。', '',
        '较慢运行在warmup就已出现：53split两ASR副本倍率2.760/2.858，而同seed不分流1.032/1.004、51split1.168/1.167。53split前400秒两副本约2.1–2.6倍；400秒后ASR0约0.95，ASR1仍约2.33。反转含明显时间/副本差异，固定单输入服务均值无法解释。same-input-within-split.csv进一步在每个相同输入内部比较前400秒、400–600秒与drain，避免把输入混合变化当速度变化。', '',
        '输入周期32与固定路由共同锁定了本次测量的输入分配：分流时G2固定接收其中4个输入，另28个由专用两副本各固定14个；不分流是两个16输入集合。53split的ASR0基线均值875.33ms，ASR1仅742.70ms，但后者实际服务1682.90ms反而更慢，说明较重输入不是慢副本和反转的充分解释。51split两ASR副本基线768.25/823.18ms、实测855.76/902.06ms，也未出现同样放大。', '',
        'G2并非没有代价：53split增加ASR handler318.37秒，G2总handler407.03秒；51split G2 ASR270.70秒，总约376.77秒。53split G2 ASR相同输入基线倍率2.722，51split1.962；G2 text倍率约5.15在两次split都存在，且53nosplit约4.91，不能单凭背景text慢解释专用突变。OCR在53split也由同seed不分流的2.338升至4.355倍，说明放大不只局限于专用ASR队列，但仍不能由相关性断定GPU干扰的物理机制。', '',
        '排空不等于两专用同时恢复：53split ASR0在600秒前已恢复，drain阶段约0.892倍，629.047秒完成；ASR1 drain仍约2.132倍，797.225秒才完成。G2至601.084秒、OCR至598.694秒结束。不能把ASR0变快归因为停止到达，因为变快早于600秒；也不能说停止到达后ASR1立即恢复。worker-start-30s.csv保留细分。', '',
        '所有专用ASR与G2 ASR输出均与原逐输入画像输出一致、EOS结束，专用ASR没有cache reclaim，service与handler倍率几乎相同；未见额外生成、截断或Python包装开销能解释2倍服务。53split专用send-start调度延迟均值10.752/13.558秒，而另两条约0.0002秒，与大积压及提交背压并存；该值不能代替独立的worker service。', '',
        '功率仅作辅助：53split前400秒各100秒采样均值约183–200W，400–600秒升至222/234W；51split窗口约214–228W；53nosplit约194–213W。这是已有全卡采样的算术均值，非每worker能耗，也非时钟或降频证据。原始记录没有GPU温度、SM/显存频率或throttle字段；不能确定热、频率、调度、并发干扰或外部进程等具体根因。', '',
        '结论是同一名义1.9/.125配置出现了真实、可量化的执行服务状态差异，并叠加固定输入/alias周期；画像输入均值与固定分流率不能代表所有运行状态。已记录证据足以解释排队为何反转，但不足以唯一识别物理原因。此产物只用于开发诊断，不改门禁、不筛掉较差U、不拟合或激活新的预测器。']
    lines+=['','额外核对：三条raw的worker_registry（执行参数、环境、命令、模型/协议标识）完全相同；53两条trace SHA相同。逐个输入的400–600秒/前400秒service均值比，ASR0的14个输入均为0.399–0.479，中位数0.423；ASR1的14个输入为0.869–1.152，中位数1.016。即ASR0恢复在每个输入内部都存在，ASR1没有同样恢复，无法由阶段样本比例变化解释。该前后分段用于描述观察，不是拟合模型或对新门禁的选择。']
    (OUTPUT/'report.zh.md').write_text('\n'.join(lines)+'\n')
    if any(receipt(r['path'])!=r for r in inputs): raise ValueError('Input changed during diagnosis')
    write(OUTPUT/'manifest.json',dict(inputs=inputs,code=[receipt(__file__)],outputs=[receipt(p) for p in sorted(OUTPUT.iterdir())],input_bytes_unchanged=True,gpu_started=False))
    print(json.dumps(dict(output=str(OUTPUT),requests=len(request_rows),gpu_started=False)))


if __name__=='__main__': main()
