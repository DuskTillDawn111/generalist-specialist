#!/usr/bin/env python3
"""Read-only audit of seven ASR1.9 development runs; never final confirmation.

--check-only checks the fixed input roster and packet/profile identities without
opening raw outputs. Missing inputs exit 2 (INCOMPLETE); no values are filled.
Full audit exits 0 or 3 (strict import/identity/arithmetic failure). Output must
be a new directory. No GPU or predictor fitting is invoked.
"""
import argparse
from collections import Counter
import copy
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_interim_data import receipt, read, write, csv_write, close, score, percentile, integrate, TERMS
from gsdeploy.calibration import _import_source, _baseline_identity
from gsdeploy.config import SystemConfig
from gsdeploy.quality import quality_digest
from gsdeploy.static_study import validate_packet
from gsdeploy.trace import make_trace,trace_digest
from scipy.stats import t as student_t

BASE=ROOT/'artifacts/static-spillover-study-20261022'
ORIGINAL=ROOT/'artifacts/static-study-gpu-20261022/execution-packet-v2.json'
PAIR_SEEDS=(51,52,53)


def roster():
    entries=[]
    for seed in PAIR_SEEDS:
        for split in (False,True):
            if seed==51:
                folder=BASE/'development-small-spillover-v2'
                packet=folder/'experiment-packet-asr1p9.json'
                label='asr1p9_specialist_share_'+('0p875' if split else '1p0')
                run=folder/'runs-asr1p9'/(label+'-attempt1')/'run.json'
            else:
                folder=BASE/'development-asr19-replication-v1'
                packet=folder/'experiment-packet.json'
                label=f'asr19_n2_specialist_share_{"0p875" if split else "1p0"}_seed{seed}'
                run=folder/'runs'/(label+'-attempt1')/'run.json'
            entries.append(dict(id=f'seed{seed}_n2_'+('split' if split else 'nosplit'),seed=seed,
                replicas=2,split=split,label=label,packet_path=str(packet),run_path=str(run)))
    folder=BASE/'development-asr19-replication-v1'
    label='asr19_n3_specialist_share_1p0_seed51'
    entries.append(dict(id='seed51_n3_nosplit',seed=51,replicas=3,split=False,label=label,
        packet_path=str(folder/'experiment-packet.json'),run_path=str(folder/'runs'/(label+'-attempt1')/'run.json')))
    return entries


def readiness(entries):
    original=read(ORIGINAL);configs=[];jobs=[]
    for packet_path in sorted({e['packet_path'] for e in entries}):
        packet=read(packet_path)
        check=validate_packet(packet)
        if not check['valid']:raise ValueError('Packet changed: '+repr(check['changed']))
        for key in ('models','quality','safe_factor','capacity_factors','memory_table_mib','task_replica_profiles'):
            if packet['config']['catalog'][key]!=original['config']['catalog'][key]:
                raise ValueError('Original profile/quality identity changed: '+key)
        if _baseline_identity(packet['config']['catalog'])!=_baseline_identity(original['config']['catalog']):
            raise ValueError('Original per-task replica baseline identity changed')
        for key in ('weights','latency_reference_ms','gpu_memory_mib','energy_reference_watts','startup_amortization_seconds','utility_schema','drain_seconds','arrival_process'):
            if packet['config'][key]!=original['config'][key]:raise ValueError('Scoring contract changed: '+key)
        configs.append(receipt(packet_path))
        for entry in (e for e in entries if e['packet_path']==packet_path):
            job=next((j for j in packet['jobs'] if j['labels']==[entry['label']]),None)
            if job is None or job['seed']!=entry['seed'] or job['phase']!='calibration' or 'development' not in job.get('purpose',''):
                raise ValueError('Predeclared development job missing/changed')
            x=job['plan']['deployment'];routing=job['plan']['routing']
            if x!={'asr':entry['replicas'],'generalist_2':1,'ocr':1} or job['identity']['workload']!=dict(text=.05,image=.05,ocr=.05,asr=1.9):
                raise ValueError('Changed workload/deployment')
            close(routing['asr'].get('generalist_2',0)/1.9,.125 if entry['split'] else 0,'split')
            jobs.append(dict(id=entry['id'],job_id=job['id'],purpose=job['purpose'],seed=entry['seed']))
    missing=[e['run_path'] for e in entries if not Path(e['run_path']).is_file()]
    return dict(schema='gsdeploy.asr19_replication_readiness.v1',status='INCOMPLETE' if missing else 'ALL_SEVEN_RAW_PATHS_PRESENT',
        original_profile_identity_verified=True,original_packet=receipt(ORIGINAL),packets=configs,jobs=jobs,
        roster=entries,missing_raw_paths=missing,raw_files_read=False,gpu_started=False,
        purpose='Development repetitions and stronger replica control; not independent validation or final confirmation')


def import_one(entry):
    packet=read(entry['packet_path']);path=Path(entry['run_path'])
    raw_disk=read(path);job=next(j for j in packet['jobs'] if j['labels']==[entry['label']])
    source=dict(id=entry['id'],**receipt(path),packet_path=entry['packet_path'],packet_sha256=receipt(entry['packet_path'])['sha256'])
    raw,utility,import_receipt=_import_source(source,{job['identity']['runtime_code_sha256']})
    if raw['experiment_job_id']!=job['id'] or raw['experiment_phase']!='calibration':raise ValueError('Job identity changed')
    config=SystemConfig(**raw['config']);window=raw['measurement_window'];records=raw['request_records']
    if config.random_seed!=entry['seed'] or window['duration_seconds']!=600 or not utility['eligible']:raise ValueError('Changed seed/H or ineligible run')
    original=read(ORIGINAL)['config']
    for key in ('models','quality','safe_factor','capacity_factors','memory_table_mib','task_replica_profiles'):
        if config.catalog[key]!=original['catalog'][key]:raise ValueError('Measured original profile changed: '+key)
    if _baseline_identity(config.catalog)!=_baseline_identity(original['catalog']):raise ValueError('Measured baseline identity changed')
    trace=make_trace(config.workload,600,entry['seed'],config.arrival_process)
    if len(trace)!=len(records) or trace_digest(trace)!=window['trace_sha256']:raise ValueError('Trace mismatch')
    rows=[];totals={t:[0,0] for t in config.workload};latencies={t:[] for t in config.workload}
    for i,(record,(offset,task)) in enumerate(zip(records,trace)):
        if record['task']!=task or record['request_id']!=f'measurement:{i}' or record['status']!='completed':raise ValueError('Incomplete cohort')
        if record['scheduled_time_ns']!=window['start_time_ns']+round(offset*1e9):raise ValueError('Scheduled timestamp differs')
        response=record['worker_response']
        if response.get('success') is not True:raise ValueError('Missing successful actual output')
        a,b=score(task,response['prediction'],config.catalog['quality']['references'][task][record['pool_id']])
        totals[task][0]+=a;totals[task][1]+=b
        e2e=(record['completion_time_ns']-record['scheduled_time_ns'])/1e6;latencies[task].append(e2e)
        rows.append(dict(run_id=entry['id'],seed=entry['seed'],asr_replicas=entry['replicas'],split=entry['split'],
            request_id=record['request_id'],task=task,pool_id=record['pool_id'],model=record['model'],
            family=raw['replica_to_family'][record['model']],scheduled_offset_seconds=offset,e2e_latency_ms=e2e,
            runtime_latency_ms=(record['completion_time_ns']-record['actual_submit_time_ns'])/1e6,
            worker_latency_ms=record['worker_latency_ms'],queue_wait_ms=record.get('worker_queue_wait_ms'),
            quality_numerator=a,quality_denominator=b,within_window=record['completion_time_ns']<=window['end_time_ns']))
    per_task={t:dict(requests=len(latencies[t]),quality_loss=totals[t][0]/totals[t][1],
        error_numerator=totals[t][0],reference_denominator=totals[t][1],mean_ms=statistics.fmean(latencies[t]),p95_ms=percentile(latencies[t],95)) for t in config.workload}
    weights={t:q/math.fsum(config.workload.values()) for t,q in config.workload.items()}
    mem=raw['memory_evidence'];peak=max(s['owned_used_mib'] for s in [mem['before'],mem['after'],*mem['samples']]);safe=peak*config.catalog['safe_factor']
    energy=raw['energy_evidence'];run_j,_=integrate(energy['samples'],window['start_time_ns'],window['end_time_ns'])
    startup_j,_=integrate(energy['samples'],energy['startup_begin_time_ns'],window['start_time_ns'])
    close(safe,mem['safe_memory_mib'],'safe_memory')
    close(run_j,energy['run_joules'],'run_joules',1e-6)
    close(startup_j,energy['startup_joules'],'startup_joules',1e-6)
    terms=dict(Q=math.fsum(weights[t]*per_task[t]['quality_loss'] for t in config.workload),
        T=1-sum(row['within_window'] for row in rows)/len(rows),
        L=math.fsum(weights[t]*per_task[t]['mean_ms']/config.latency_reference_ms[t] for t in config.workload),
        P=math.fsum(weights[t]*per_task[t]['p95_ms']/config.latency_reference_ms[t] for t in config.workload),
        M=safe/config.gpu_memory_mib,E=(run_j/600+startup_j/config.startup_amortization_seconds)/config.energy_reference_watts)
    contributions={k:-config.weights[k]*terms[k] for k in TERMS};total=math.fsum(contributions.values())
    for other in (utility,raw_disk['evaluation']['utility']):
        for k in TERMS:close(terms[k],other[k+'_norm'],'terms/'+k)
        close(total,other['U'],'U')
    return dict(**entry,source=source,strict_import_receipt=import_receipt,utility=utility,U=total,
        terms=terms,contributions=contributions,per_task=per_task,plan=raw['plan'],
        quality_document_sha256=quality_digest(config.catalog['quality']),
        cohort=dict(trace_sha256=window['trace_sha256'],offered=len(records),
            task_input_sha256=quality_digest([(r['task'],r['pool_id']) for r in records])),
        weights=config.weights,stability=raw['evaluation']['stability']),rows


def paired_statistics(conditions):
    differences=[];pairs=[]
    for seed in PAIR_SEEDS:
        a=next(c for c in conditions if c['seed']==seed and c['replicas']==2 and not c['split'])
        b=next(c for c in conditions if c['seed']==seed and c['replicas']==2 and c['split'])
        if a['cohort']!=b['cohort'] or a['quality_document_sha256']!=b['quality_document_sha256']:raise ValueError('Within-seed cohort/profile mismatch')
        delta=b['U']-a['U'];differences.append(delta)
        pairs.append(dict(seed=seed,nosplit=a['id'],split=b['id'],split_minus_nosplit_U=delta,
            delta_terms={k:b['terms'][k]-a['terms'][k] for k in TERMS},
            delta_contributions={k:b['contributions'][k]-a['contributions'][k] for k in TERMS},cohort=a['cohort']))
    n=len(differences);estimate=statistics.fmean(differences);se=statistics.stdev(differences)/math.sqrt(n)
    half=float(student_t.ppf(.975,n-1))*se;observed=abs(math.fsum(differences))
    p=sum(abs(math.fsum(s*d for s,d in zip(signs,differences)))>=observed-1e-12 for signs in itertools.product((-1,1),repeat=n))/2**n
    return dict(pairs=pairs,n_seed_blocks=n,mean_split_minus_nosplit_U=estimate,standard_error=se,
        student_95_interval=[estimate-half,estimate+half],exact_two_sided_sign_flip_p=p,
        minimum_possible_two_sided_sign_flip_p=2/2**n,all_seed_differences_positive=all(d>0 for d in differences),
        development_selection_caveat='Seed51 selected the repetition study after observing its tiny benefit; development inference is descriptive, not unbiased final confirmation.',
        assumptions='CI assumes independent approximately normal seed differences; exact sign flip assumes independent symmetric null differences; n=3 is very small.',
        independent_unit='seed block, never individual repeated requests',final_confirmation=False)


def run(output,entries,ready):
    conditions=[];requests=[]
    for entry in entries:
        condition,rows=import_one(entry);conditions.append(condition);requests.extend(rows)
    paired=paired_statistics(conditions)
    n3=next(c for c in conditions if c['replicas']==3);controls=[]
    for other in (c for c in conditions if c['seed']==51 and c['replicas']==2):
        if n3['cohort']!=other['cohort']:raise ValueError('n3 seed51 cohort differs')
        controls.append(dict(n3=n3['id'],reference=other['id'],delta_U_n3_minus_reference=n3['U']-other['U'],
            delta_terms={k:n3['terms'][k]-other['terms'][k] for k in TERMS},
            scope='Single seed51 structural change from n2 to n3; descriptive only, no significance or route-only causal claim'))
    report=dict(schema='gsdeploy.asr19_replication_development_audit.v1',status='SEVEN_DEVELOPMENT_RUNS_VERIFIED',
        readiness=ready,conditions=conditions,paired=paired,n3_structural_controls=controls,
        decision_support=dict(mean_split_gain=paired['mean_split_minus_nosplit_U'],
            consistent_positive_seed_direction=paired['all_seed_differences_positive'],
            strongest_seed51_structure=max((c for c in conditions if c['seed']==51),key=lambda c:c['U'])['id'],
            interpretation='Development evidence to decide whether to continue ASR or inspect text backup; no automatic claim of reliable gain or full-pool optimum'),
        quality_direction='All Q/T/L/P/M/E are losses; lower is better; U=-sum(weight*loss), higher is better',
        final_confirmation=False,full_pool_optimality=False,predictor_fitted=False,gpu_started=False)
    write(output/'audit.json',report)
    csv_write(output/'plans-six-terms.csv',[dict(id=c['id'],seed=c['seed'],asr_replicas=c['replicas'],split=c['split'],
        **{k+'_norm':c['terms'][k] for k in TERMS},**{k+'_contribution':c['contributions'][k] for k in TERMS},U=c['U'],
        raw_sha256=c['source']['sha256']) for c in conditions])
    csv_write(output/'request-latencies.csv',requests)
    lines=['# ASR1.9 开发重复审计','', '七条记录严格导入并独立复核实际输出质量、六项及总效用；画像、评分权重与原始配置一致。同 seed 采用共同到达和输入序列。质量损失越小越好，U 越大越好。','',
        '| seed | n2不分流 U | n2分流 U | 分流 ΔU |','|---|---:|---:|---:|']
    for pair in paired['pairs']:
        a=next(c for c in conditions if c['id']==pair['nosplit']);b=next(c for c in conditions if c['id']==pair['split'])
        lines.append(f"| {pair['seed']} | {a['U']:.9f} | {b['U']:.9f} | {pair['split_minus_nosplit_U']:+.9f} |")
    lines.extend(['',f"三 seed 平均 ΔU={paired['mean_split_minus_nosplit_U']:+.9f}，95% t 区间={paired['student_95_interval']}，双侧精确符号翻转 p={paired['exact_two_sided_sign_flip_p']:.3f}。n=3，最小可能双侧 p=0.25。seed51 已用于决定追加实验，这些均为开发数据，不能作为无选择偏差的最终确认。",''])
    for c in controls:lines.append(f"- seed51 n3不分流相对 {c['reference']}：ΔU={c['delta_U_n3_minus_reference']:+.9f}；属于副本结构变化，仅单 seed 描述。")
    lines.extend(['','仅支持“继续研究 ASR 或转向 text 备选”的开发决策，不声称完整模型池最优、稳定分流收益或最终确认。重复请求不是独立质量样本；CI/符号翻转要求见 JSON。工具未修改数据、门禁、画像或预测器，未启动 GPU。'])
    (output/'report.zh.md').write_text('\n'.join(lines)+'\n')
    inputs={r['path']:r for r in [ready['original_packet'],*ready['packets'],
        *[dict(path=c['source']['path'],sha256=c['source']['sha256']) for c in conditions]]}
    if any(receipt(r['path'])!=r for r in inputs.values()):raise ValueError('Audit input bytes changed')
    write(output/'manifest.json',dict(inputs=list(inputs.values()),script=receipt(__file__),
        helper=receipt(Path(__file__).with_name('audit_interim_data.py')),
        outputs=[receipt(p) for p in sorted(output.iterdir()) if p.is_file()],gpu_started=False))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();entries=roster();ready=readiness(entries)
    if args.output:
        args.output.mkdir(parents=True,exist_ok=False);write(args.output/'readiness.json',ready)
    if args.check_only or ready['missing_raw_paths']:
        print(json.dumps(ready,ensure_ascii=False,indent=2))
        raise SystemExit(2 if ready['missing_raw_paths'] else 0)
    if not args.output:parser.error('--output required for complete audit')
    try:
        report=run(args.output,entries,ready)
    except Exception as error:
        write(args.output/'failure.json',dict(status='AUDIT_FAILED',error_type=type(error).__name__,error=str(error),gpu_started=False))
        raise SystemExit(3) from error
    print(json.dumps(dict(status=report['status'],paired=report['paired'],gpu_started=False),ensure_ascii=False))


if __name__=='__main__':main()
