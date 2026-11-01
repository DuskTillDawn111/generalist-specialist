"""CPU-only formal input/route exposure for explicitly supplied rates and seeds.

No native run, service, capacity, utility or seed/load selection is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'artifacts/static-study-gpu-20261022/execution-source-v5'
DEPLOYMENT = {'text':2,'generalist_2':1,'ocr':1}
SHARES = (0.0, 1/8, 1/4)


def receipt(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def frozen_api(source=SOURCE):
    """Use an isolated package namespace; never replace process gsdeploy imports."""
    source=Path(source).resolve()
    name='_capacity_exposure_frozen_'+hashlib.sha256(str(source).encode()).hexdigest()[:12]
    if name not in sys.modules:
        package=types.ModuleType(name);package.__path__=[str(source/'gsdeploy')]
        sys.modules[name]=package
    return {key:importlib.import_module(name+'.'+module) for key,module in {
        'trace':'trace','routing':'routing','workload':'runtime.workload','quality':'quality'}.items()}


def finite_period(sequence):
    for size in range(1,min(256,len(sequence)//2)+1):
        if all(value==sequence[index%size] for index,value in enumerate(sequence)):
            return size
    return None


def build_case(config,samples,text_rate,seed,generalist_share,api=None):
    if type(text_rate) not in (int,float) or not math.isfinite(text_rate) or text_rate<=0:
        raise ValueError('Explicit positive finite text rate required')
    if type(seed) is not int or seed<0:raise ValueError('Explicit nonnegative integer seed required')
    if generalist_share not in SHARES:raise ValueError('Only declared shares0,1/8,1/4')
    api=api or frozen_api();quality=config['catalog']['quality']
    rates={'text':text_rate,'image':.01,'asr':.01,'ocr':.01}
    routing={'text':{'text':text_rate*(1-generalist_share),'generalist_2':text_rate*generalist_share},
             'image':{'generalist_2':.01},'asr':{'generalist_2':.01},'ocr':{'ocr':.01}}
    arrival=config['arrival_process']
    trace=api['trace'].make_trace(rates,600,seed,arrival)
    aliases,routes=api['trace'].expand_routes(DEPLOYMENT,routing)
    pool=api['workload'].SamplePool(samples,list(rates),seed)
    router=api['routing'].Router(routes);position=Counter();records=[]
    for index,(offset,task) in enumerate(trace):
        item=pool.next(task);alias=router.choose(task)
        records.append(dict(arrival_index=index,scheduled_offset_seconds=offset,task=task,
                            task_position=position[task],pool_id=item['pool_id'],alias=alias,family=aliases[alias]))
        position[task]+=1
    def group(task,family,alias=None):
        selected=[r for r in records if r['task']==task and r['family']==family and (alias is None or r['alias']==alias)]
        counts=Counter(r['pool_id'] for r in selected);order=quality['input_order'][task]
        path=quality['paths'][family][task];a=b=0.0;fulla=fullb=0.0
        for pool_id in order:
            numerator,denominator=api['quality'].sample_totals(task,path['samples'][pool_id])
            a+=counts[pool_id]*numerator;b+=counts[pool_id]*denominator
            fulla+=numerator;fullb+=denominator
        return dict(task=task,family=family,alias=alias,scheduled_requests=len(selected),pool_size=len(order),
                    unique_inputs=len(counts),missing_inputs=sorted(set(order)-set(counts)),
                    per_input_counts={p:counts[p] for p in order},scheduled_error_numerator=a,
                    scheduled_reference_denominator=b,conditional_scheduled_loss=a/b if b else None,
                    full_pool_replay_loss=fulla/fullb,full_pool_catalog_value=path.get('value'),
                    full_pool_catalog_metric=path['metric'],
                    quality_scope='FROZEN_HISTORICAL_PROFILE_REPLAY_NOT_MEASURED_NATIVE_QUALITY')
    family_groups=[];alias_groups=[]
    for family in sorted(DEPLOYMENT):
        for task in sorted(rates):
            if task not in config['catalog']['models'][family]['supported_tasks']:continue
            family_groups.append(group(task,family))
            for alias in sorted(a for a,f in aliases.items() if f==family):alias_groups.append(group(task,family,alias))
    direct={}
    for task in rates:
        groups=[g for g in family_groups if g['task']==task]
        a=math.fsum(g['scheduled_error_numerator'] for g in groups)
        b=math.fsum(g['scheduled_reference_denominator'] for g in groups)
        direct[task]=a/b if b else None
    predicted,evidence=api['quality'].predict_quality(config['catalog'],DEPLOYMENT,routing,rates,trace,seed)
    for task,loss in direct.items():
        actual=evidence['per_task'][task]['loss']
        if (actual is None)!=(loss is None) or (loss is not None and not math.isclose(actual,loss,abs_tol=1e-12,rel_tol=1e-12)):
            raise ValueError('Manual exposure and frozen quality replay differ: '+task)
    direct_total=math.fsum(rates[t]/math.fsum(rates.values())*direct[t] for t in rates) if all(v is not None for v in direct.values()) else None
    if (predicted is None)!=(direct_total is None) or (predicted is not None and not math.isclose(predicted,direct_total,abs_tol=1e-12,rel_tol=1e-12)):
        raise ValueError('Manual totalQ differs from frozen quality replay')
    text_sets={family:set(r['pool_id'] for r in records if r['task']=='text' and r['family']==family) for family in ['text','generalist_2']}
    return dict(schema='gsdeploy.formal_route_exposure.case.v1',text_rate=text_rate,seed=seed,generalist_text_share=generalist_share,
                deployment=DEPLOYMENT,routing=routing,workload=rates,horizon_seconds=600,arrival_process=arrival,
                trace_sha256=api['trace'].trace_digest(trace),scheduled_records=records,
                family_groups=family_groups,alias_groups=alias_groups,
                text_family_input_intersection=sorted(text_sets['text']&text_sets['generalist_2']),
                finite_alias_period={t:finite_period([r['alias'] for r in records if r['task']==t]) for t in rates},
                finite_period_scope='Exact finite cohort equality only; no infinite floating-point periodicity proof.',
                manual_task_quality_loss=direct,predicted_quality_Q_loss=predicted,manual_total_quality_loss=direct_total,
                frozen_predict_quality_verified=True,quality_evidence=evidence,U=None,raw_files_read=0,GPU_started=False)


def produce(config_path,samples,text_rates,seeds,output):
    output=Path(output)
    if output.exists():raise FileExistsError('Never overwrite exposure evidence')
    if not text_rates or not seeds or len(set(text_rates))!=len(text_rates) or len(set(seeds))!=len(seeds):
        raise ValueError('Explicit nonempty unique rates and seeds required')
    config=json.loads(Path(config_path).read_text());quality=config['catalog']['quality']
    if quality['mode']!='per_sample_replay':raise ValueError('Per-input frozen quality evidence required')
    if receipt(samples)['sha256']!=quality['identity']['input_pool_sha256']:raise ValueError('Input pool hash differs from frozen quality identity')
    api=frozen_api();api['quality']._document(config['catalog'])
    provenance=[receipt(config_path),receipt(samples),receipt(__file__)]
    provenance += [receipt(SOURCE/p) for p in ['gsdeploy/trace.py','gsdeploy/routing.py','gsdeploy/quality.py','gsdeploy/runtime/workload.py']]
    reports=[build_case(config,samples,rate,seed,share,api) for rate in text_rates for seed in seeds for share in SHARES]
    intersections=[]
    for rate in text_rates:
        for seed in seeds:
            cases=[r for r in reports if r['text_rate']==rate and r['seed']==seed]
            cohort=[[(x['arrival_index'],x['scheduled_offset_seconds'],x['task'],x['pool_id']) for x in r['scheduled_records']] for r in cases]
            if any(rows!=cohort[0] for rows in cohort):raise ValueError('Routing changed route-independent cohort')
            for left_index,left in enumerate(cases):
                for right in cases[left_index+1:]:
                    for family in ['text','generalist_2']:
                        a={r['pool_id'] for r in left['scheduled_records'] if r['task']=='text' and r['family']==family}
                        b={r['pool_id'] for r in right['scheduled_records'] if r['task']=='text' and r['family']==family}
                        intersections.append(dict(text_rate=rate,seed=seed,family=family,left_share=left['generalist_text_share'],right_share=right['generalist_text_share'],
                                                  left_unique=len(a),right_unique=len(b),intersection=sorted(a&b),left_only=sorted(a-b),right_only=sorted(b-a)))
    for item in provenance:
        if receipt(item['path'])!=item:raise ValueError('Input/source changed during reconstruction')
    output.mkdir(parents=True,exist_ok=False)
    overview=[]
    for index,report in enumerate(reports,1):
        stem=f'case-{index:03d}'
        with (output/(stem+'-roster.csv')).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['arrival_index','scheduled_offset_seconds','task','task_position','pool_id','alias','family'])
            writer.writeheader();writer.writerows(report['scheduled_records'])
        detail={k:v for k,v in report.items() if k!='scheduled_records'}
        detail['roster_csv']=stem+'-roster.csv'
        with (output/(stem+'.json')).open('x') as stream:json.dump(detail,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
        overview.append({k:detail[k] for k in ['text_rate','seed','generalist_text_share','trace_sha256','manual_task_quality_loss','predicted_quality_Q_loss','roster_csv']})
    result=dict(schema='gsdeploy.formal_route_exposure.v1',scope='METADATA_ONLY_NOT_NATIVE_OR_CAPACITY_VALIDATION',
                explicit_text_rates=text_rates,explicit_seeds=seeds,shares=list(SHARES),cases=overview,
                family_intersections_across_routes=intersections,provenance=provenance,U=None,raw_files_read=0,progress_read=False,GPU_started=False,
                limitations=['No rates/seeds selected or ranked; all supplied cases reconstructed.',
                             'Conditional scheduled quality replays historical outputs; it is not actual concurrent quality or total utility.',
                             'Missing task observations remain unknown; no seed is discarded for missing background samples.',
                             'No service, memory, energy, rho or stability feasibility is inferred.',
                             'Formal SamplePool, arrival trace, expanded alias routing and SWRR are unchanged.'])
    with (output/'report.json').open('x') as stream:json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    with (output/'README.zh.md').open('x') as stream:
        stream.write('仅CPU元数据重建：固定text×2、G2×1、OCR×1，背景各0.01rps，H600；负载与种子均由显式参数提供。\n\n每条件CSV列出原正式到达、输入与alias路由；JSON同时报告条件输入质量、全池画像与跨路由输入交集。Q仅为历史画像重放，U=null；不读取运行数据、不选择有利条件、不作容量或可行性结论。\n')
    manifest=dict(provenance=provenance,files={p.name:receipt(p)['sha256'] for p in sorted(output.iterdir()) if p.is_file()})
    with (output/'manifest.json').open('x') as stream:json.dump(manifest,stream,indent=2);stream.write('\n')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path);parser.add_argument('--samples',required=True,type=Path)
    parser.add_argument('--text-rates',required=True,nargs='+',type=float);parser.add_argument('--seeds',required=True,nargs='+',type=int)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(argv)
    result=produce(args.config,args.samples,args.text_rates,args.seeds,args.output)
    print(json.dumps(dict(cases=len(result['cases']),output=str(args.output.resolve()),GPU_started=False,raw_files_read=0)))


if __name__=='__main__':main()
