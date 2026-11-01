#!/usr/bin/env python3
"""CPU metadata-derived input exposure and frozen quality replay; never run raw."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'artifacts/static-study-gpu-20261022/execution-source-v5'
sys.path.insert(0,str(SOURCE))
from gsdeploy.quality import predict_quality, sample_totals
from gsdeploy.routing import Router
from gsdeploy.runtime.workload import SamplePool
from gsdeploy.static_study import validate_packet
from gsdeploy.trace import expand_routes, make_trace, trace_digest

DEFAULT_PACKET=ROOT/'artifacts/static-spillover-study-20261022/text26-development-v1/experiment-packet.json'


def receipt(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def observed_period(sequence):
    """Smallest exact period in this finite scheduled sequence, if <=256."""
    for period in range(1,min(256,len(sequence)//2)+1):
        if all(v==sequence[k%period] for k,v in enumerate(sequence)):
            return period
    return None


def exposure_group(records, expected, quality, task, family, alias=None):
    counts=Counter(r['pool_id'] for r in records)
    path=quality['paths'][family][task]
    numerator=denominator=0.0
    for sample,count in counts.items():
        a,b=sample_totals(task,path['samples'][sample]);numerator+=count*a;denominator+=count*b
    loss=numerator/denominator if denominator else None
    return dict(task=task,family=family,alias=alias,offered_requests=len(records),
        pool_size=len(expected),unique_inputs=len(counts),missing_inputs=sorted(set(expected)-set(counts)),
        per_input_counts={sample:counts[sample] for sample in expected},
        positive_input_count_min=min(counts.values()) if counts else None,
        positive_input_count_max=max(counts.values()) if counts else None,
        scheduled_profile_error_numerator=numerator,scheduled_profile_reference_denominator=denominator,
        scheduled_profile_native_loss=loss,
        scheduled_profile_accuracy=(1-loss if loss is not None and task in {'text','image'} else None),
        full_pool_profile_value=path['value'],full_pool_profile_metric=path['metric'],
        quality_scope='METADATA_DERIVED_FROZEN_PROFILE_REPLAY_NOT_NATIVE_OUTPUT')


def audit_job(job, packet, pool_path):
    identity=job['identity'];quality=packet['config']['catalog']['quality']
    if job['phase']!='calibration' or job['seed'] not in range(51,60):
        raise ValueError('Only predeclared development seeds51..59 accepted')
    rates=identity['workload'];seed=job['seed']
    trace=make_trace(rates,identity['horizon_seconds'],seed,identity['arrival_process'])
    if trace_digest(trace)!=identity['trace_sha256']:
        raise ValueError('Rebuilt trace differs from frozen job identity')
    aliases,routes=expand_routes(job['plan']['deployment'],job['plan']['routing'])
    pool=SamplePool(pool_path,list(rates),seed);router=Router(routes)
    taskpos=Counter();records=[]
    for arrival,(offset,task) in enumerate(trace):
        item=pool.next(task);alias=router.choose(task);family=aliases[alias]
        records.append(dict(arrival_index=arrival,scheduled_offset_seconds=offset,task=task,
            task_position=taskpos[task],pool_id=item['pool_id'],alias=alias,family=family,
            replicas=job['plan']['deployment'][family]))
        taskpos[task]+=1
    by_family=defaultdict(list);by_alias=defaultdict(list)
    for record in records:
        by_family[(record['task'],record['family'])].append(record)
        by_alias[(record['task'],record['family'],record['alias'])].append(record)
    family_groups=[exposure_group(seq,quality['input_order'][task],quality,task,family)
        for (task,family),seq in sorted(by_family.items())]
    alias_groups=[exposure_group(seq,quality['input_order'][task],quality,task,family,alias)
        for (task,family,alias),seq in sorted(by_alias.items())]
    sequences={}
    for task in sorted(rates):
        seq=[r for r in records if r['task']==task]
        period=observed_period([r['alias'] for r in seq])
        sequences[task]=dict(offered_requests=len(seq),observed_alias_period=period,
            finite_period_scope='Exact equality over this finite scheduled cohort only; not a global floating-point periodicity proof.',
            first_64_aliases=[r['alias'] for r in seq[:64]],
            first_64_inputs=[r['pool_id'] for r in seq[:64]])
    predicted_Q,evidence=predict_quality(packet['config']['catalog'],job['plan']['deployment'],
        job['plan']['routing'],rates,trace,seed)
    direct={task:math.fsum(g['scheduled_profile_error_numerator'] for g in family_groups if g['task']==task)/
        math.fsum(g['scheduled_profile_reference_denominator'] for g in family_groups if g['task']==task)
        for task in rates}
    rebuilt_Q=math.fsum(rates[t]/sum(rates.values())*direct[t] for t in rates)
    if predicted_Q is None or not math.isclose(predicted_Q,rebuilt_Q,rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError('Runtime-sample exposure disagrees with frozen per-input quality predictor')
    for task in rates:
        if not math.isclose(direct[task],evidence['per_task'][task]['loss'],rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('Task exposure disagrees with frozen quality predictor: '+task)
    return dict(job_id=job['id'],labels=job['labels'],seed=seed,plan=job['plan'],
        scope='METADATA_DERIVED_NOT_NATIVE',offered_requests=len(records),trace_sha256=trace_digest(trace),
        family_groups=family_groups,alias_groups=alias_groups,task_sequences=sequences,
        scheduled_records=records,predicted_quality_Q_loss=predicted_Q,
        predicted_quality_evidence=evidence,direct_exposure_quality_loss=direct,
        direct_vs_predictor_verified=True,U=None,native_outputs_read=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,default=DEFAULT_PACKET)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    packet=json.loads(args.packet.read_text());validation=validate_packet(packet)
    if not validation['valid']:raise ValueError('Frozen packet validation failed')
    pool_path=Path(packet['inputs'])/'runtime-inputs.jsonl'
    if receipt(pool_path)['sha256']!=packet['config']['catalog']['quality']['identity']['input_pool_sha256']:
        raise ValueError('Runtime pool differs from frozen quality identity')
    if args.output.exists():raise FileExistsError('Never overwrite exposure evidence')
    source_receipts=[receipt(SOURCE/p) for p in ['gsdeploy/routing.py','gsdeploy/trace.py',
        'gsdeploy/quality.py','gsdeploy/runtime/workload.py','gsdeploy/runtime/runner.py','gsdeploy/static_study.py']]
    inputs=[receipt(args.packet),receipt(pool_path)]
    tool=receipt(Path(__file__))
    reports=[audit_job(job,packet,pool_path) for job in packet['jobs']]
    for item in inputs+source_receipts+[tool]:
        if receipt(item['path'])!=item:raise ValueError('Metadata/source changed during exposure audit')
    args.output.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        (args.output/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    overview=[]
    for k,r in enumerate(reports):
        filename=f'job-{k+1}-{r["job_id"][:12]}.json';save(filename,r)
        overview.append({**{key:r[key] for key in ['job_id','labels','plan','offered_requests',
            'trace_sha256','family_groups','alias_groups','task_sequences','predicted_quality_Q_loss',
            'direct_exposure_quality_loss','direct_vs_predictor_verified']},'detail_file':filename})
    limitations=['Metadata-derived scheduled exposure only; no run/progress, native output, latency, service or energy read.',
        'Quality reuses frozen historical per-input outputs; concurrent/replica output reproduction is unverified.',
        'Q is native task loss weighted by configured task rates; lower is better. It is not accuracy averaged across incompatible task metrics, nor total utility.',
        'Offered exposure includes requests that a native failed run might never dispatch or complete.',
        'Fixed repeated-pool and deterministic routing can align sample identities with aliases; path averages are input-composition dependent.',
        'No workload, routing, seed, pool, scoring or profile changes; no GPU execution or fitted parameters.']
    save('report.json',dict(schema='gsdeploy.predeclared_exposure.v1',scope='METADATA_DERIVED_NOT_NATIVE',
        jobs=overview,limitations=limitations,raw_files_read=0,progress_read=False,native_outputs_read=0,
        gpu_started=False,U=None,fitted_parameters=False))
    lines=['# text26 预声明输入暴露审计','',
        '本报告只重建冻结packet的计划输入与路由，未读取运行raw、progress或原生输出，未启动GPU。Q为固定画像预测的质量损失，越小越好；不是实测精度或总效用U。','',
        '| 条件 | text请求 | 专用覆盖 | G2覆盖 | 专用各alias覆盖 | 有限轨迹alias周期 | 预测text精度 | 预测Q损失 |',
        '|---|---:|---:|---:|---|---:|---:|---:|']
    for r in reports:
        gs={g['family']:g for g in r['family_groups'] if g['task']=='text'}
        aliases=[g for g in r['alias_groups'] if g['task']=='text' and g['family']=='text']
        lines.append('| '+r['labels'][0]+' | '+str(r['task_sequences']['text']['offered_requests'])+
            ' | '+str(gs['text']['unique_inputs'])+'/32 | '+str(gs['generalist_2']['unique_inputs'])+'/32 | '+
            ', '.join(g['alias']+':'+str(g['unique_inputs'])+'/32' for g in aliases)+
            ' | '+str(r['task_sequences']['text']['observed_alias_period'])+' | '+
            f"{1-r['direct_exposure_quality_loss']['text']:.9f} | {r['predicted_quality_Q_loss']:.9f} |")
    lines+=['','逐输入计数、缺失输入及各alias质量画像重放结果见每job JSON；所有零计数输入均明确列出。',
        '四条件共享相同到达轨迹，但不同路由会改变同一输入由哪一模型／alias处理。因此跨条件服务与输出质量应先核对共同输入，不能将路径均值差异直接归因于分流或共驻干扰。',
        '固定画像的Q用同一运行时代码重建输入与路由，并逐任务核对冻结predict_quality；二者一致不证明原生并发输出将复现历史答案。',
        'Poisson控制到达时刻；按任务的固定输入循环与确定性路由仍可能周期对齐。周期字段只描述有限计划轨迹。',
        '本审计不改变任何实验条件，不挑选运行或路由，不产生U或最优组合结论。','']
    (args.output/'report.md').write_text('\n'.join(lines))
    save('protocol.json',dict(packet_validation=validation,inputs=inputs,source_receipts=source_receipts,
        tool=tool,scope='METADATA_DERIVED_NOT_NATIVE',limitations=limitations,raw_files_read=0,progress_read=False,gpu_started=False))
    save('manifest.json',dict(inputs=inputs,source_receipts=source_receipts,tool=tool,
        files={p.name:receipt(p)['sha256'] for p in args.output.iterdir() if p.is_file()}))
    print(json.dumps(dict(status='METADATA_DERIVED_EXPOSURE_COMPLETE',jobs=len(reports),output=str(args.output.resolve()),raw_files_read=0,gpu_started=False)))


if __name__=='__main__':main()
