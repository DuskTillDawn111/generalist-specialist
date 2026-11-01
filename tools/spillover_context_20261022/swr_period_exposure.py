"""Freeze then replay an explicit CPU-only SWRR/text-pool resonance design.

No native results, queues, service fitting, U, route-domain edits or execution.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/static-spillover-study-20261022'
SOURCE=ROOT/'artifacts/static-study-gpu-20261022/execution-source-v5'
SPEC=importlib.util.spec_from_file_location('_swr_frozen_api',ROOT/'tools/spillover_capacity_20261022/exposure.py')
exposure=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(exposure)
SHARES=['0','1/65','1/64','1/63','1/33','1/32','1/31','1/17','1/16','1/15']
SEEDS=[107,108];DEPLOYMENT={'text':2,'generalist_2':1,'ocr':1}
RATES={'text':1.85,'image':.01,'asr':.01,'ocr':.01}
STATES=[('text','text',2),('generalist_2','text',1),('generalist_2','image',1),('generalist_2','asr',1),('ocr','ocr',1)]
DESTINATION=BASE/'swr-period-exposure-v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def finite_period(sequence):
    """Smallest exact finite-tape period supported by at least two full repeats."""
    for p in range(1,len(sequence)//2+1):
        if all(x==sequence[i%p] for i,x in enumerate(sequence)):return p
    return None


def periodic_residues(period,selected_offsets,pool_size):
    """Hypothetical repeated observed motif, not infinite float-SWRR proof."""
    if period is None:return None
    return sorted({(offset+k*period)%pool_size for offset in selected_offsets for k in range(pool_size//math.gcd(period,pool_size))})


def bsum(values):
    return math.fsum(values) if all(v is not None for v in values) else None


def imbalance(a,b):
    if a is None or b is None:return dict(status='UNKNOWN',absolute_gap=None,relative_gap_to_mean=None,max_min_ratio=None)
    return dict(status='FROZEN_BASELINE_PLANNED_WORK_ONLY',absolute_gap=abs(a-b),
        relative_gap_to_mean=abs(a-b)/((a+b)/2) if a+b else 0.,max_min_ratio=max(a,b)/min(a,b) if min(a,b)>0 else None)


def prepare(output=DESTINATION):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    packet_path=BASE/'text185-fine-development-v1/experiment-packet.json'
    if sha(packet_path)!='649913d0073f6407ed97ce5d2edc826fef166fb386a79818fea771f5b6846f5a':raise ValueError('Fine packet changed')
    packet=json.loads(packet_path.read_text());catalog=packet['config']['catalog']
    inventory_path=BASE/'context-design-inventory-v5/manifest.json'
    if sha(inventory_path)!='81775f9116d666476a8b5a8ac86745a431bbf921f679072b2f2dffaa55b079e8':raise ValueError('Frozen inventory changed')
    inventory=json.loads(inventory_path.read_text());samples=Path(packet['inputs'])/'runtime-inputs.jsonl'
    if sha(samples)!=catalog['quality']['identity']['input_pool_sha256']:raise ValueError('Input pool identity changed')
    if len(catalog['quality']['input_order']['text'])!=32 or catalog['quality']['mode']!='per_sample_replay':raise ValueError('Expected historical text32 per-input quality')
    baseline_files={}
    for family,task,n in STATES:
        for filename in catalog['task_replica_profiles'][family][task][str(n)]['raw_service_samples']:
            path=Path(filename).resolve();expected=inventory['inputs'][str(path)]
            if sha(path)!=expected:raise ValueError('Frozen standalone b source changed')
            baseline_files[str(path)]=expected
    files=[packet_path,inventory_path,samples,Path(__file__).resolve(),Path(exposure.__file__),
        ROOT/'tests/test_swr_period_exposure.py',*[SOURCE/p for p in ['gsdeploy/trace.py','gsdeploy/routing.py','gsdeploy/quality.py','gsdeploy/runtime/workload.py']],
        *map(Path,baseline_files)]
    design=dict(schema='gsdeploy.swr_period_exposure.design.v1',status='FROZEN_BEFORE_REPLAY',prepared_unix_ns=time.time_ns(),
        shares=SHARES,seeds=SEEDS,cases=[dict(share=s,seed=seed) for seed in SEEDS for s in SHARES],case_count=20,
        deployment=DEPLOYMENT,workload=RATES,horizon_seconds=600,arrival_process='poisson',
        packet=ref(packet_path),samples=ref(samples),baseline_states=[dict(family=m,task=t,n=n) for m,t,n in STATES],
        files=[ref(p) for p in sorted(set(files))],
        reconstruction='Unmodified original make_trace -> phase-local SamplePool.next(task) -> expanded-alias float SWRR; task-local route state; no random routing or ideal fractional counts.',
        planned_statistics=['Full request roster with text/task position, alias, pool slot, input ID and frozen b_ms',
            'Every alias/task request counts, unique inputs, per-input frequencies, conditional historical quality, frozen baseline sums and composition bias',
            'G2 text positions mod32, position gaps, finite observed alias/selection periods and gcd/lcm diagnostic',
            'Exact frozen historical Q cross-checked with original predict_quality; text-specialist planned b workload imbalance',
            'All20 cases retained, route-independent cohort equality per seed verified; no route/seed ranking'],
        mathematical_caution='Minimal rational integer weight-sum and repeated observed motif are diagnostic only; actual floating-point tie ordering is replayed and finite evidence does not prove infinite SWRR periodicity.',
        scope='CPU_METADATA_EXPOSURE_ONLY',U=None,formal_rho=None,actual_service=None,predictor_activated=False,
        search_domain_modified=False,packets_modified=False,queue_read=False,raw_run_results_read=False,GPU_started=False)
    output.mkdir(parents=True,exist_ok=False);save(output/'design.json',design)
    save(output/'design-manifest.json',dict(design=ref(output/'design.json'),case_count=20,results_generated=False))
    print(json.dumps(dict(design=ref(output/'design.json'),manifest=ref(output/'design-manifest.json'),cases=20,results_generated=False)))


def baselines(config):
    result={};metadata=[]
    for family,task,n in STATES:
        grouped=defaultdict(list);files=config['catalog']['task_replica_profiles'][family][task][str(n)]['raw_service_samples']
        for filename in files:
            for line in Path(filename).read_text().splitlines():
                if not line.strip():continue
                row=json.loads(line);value=row.get('service_ms')
                if row.get('success') is True and isinstance(value,(int,float)) and math.isfinite(value) and value>0:
                    grouped[row['sample_id']].append(value)
        ids=config['catalog']['quality']['input_order'][task]
        values={p:statistics.median(grouped[p]) if grouped[p] else None for p in ids}
        result[family,task,n]=values
        metadata.append(dict(family=family,task=task,n=n,missing=[p for p,v in values.items() if v is None],
            definition='Median successful standalone service_ms over frozen rounds/replicas for exact family/task/n/input; no current mixed service or fallback',
            mean_b_ms=bsum(list(values.values()))/len(ids) if all(v is not None for v in values.values()) else None,per_input_b_ms=values))
    return result,metadata


def replay_case(config,samples,seed,share,baseline,api):
    share_fraction=Fraction(share);g=float(share_fraction)
    text={'text':RATES['text']*(1-g)}
    if g:text['generalist_2']=RATES['text']*g
    routing=dict(text=text,image={'generalist_2':.01},asr={'generalist_2':.01},ocr={'ocr':.01})
    trace=api['trace'].make_trace(RATES,600,seed,'poisson')
    aliases,routes=api['trace'].expand_routes(DEPLOYMENT,routing)
    pool=api['workload'].SamplePool(samples,list(RATES),seed);router=api['routing'].Router(routes);positions=Counter();rows=[]
    pool_orders={t:[r['pool_id'] for r in items] for t,items in pool.samples.items()}
    for i,(arrival,task) in enumerate(trace):
        position=positions[task];item=pool.next(task);alias=router.choose(task);family=aliases[alias]
        b=baseline[family,task,DEPLOYMENT[family]].get(item['pool_id'])
        rows.append(dict(arrival_index=i,arrival_seconds=arrival,task=task,task_position=position,
            pool_slot=position%len(pool.samples[task]),pool_id=item['pool_id'],alias=alias,family=family,b_ms=b))
        positions[task]+=1
    groups=[];catalog=config['catalog'];quality=catalog['quality'];manual=defaultdict(lambda:[0.,0.])
    for alias,family in aliases.items():
        for task in RATES:
            if task not in catalog['models'][family]['supported_tasks']:continue
            selected=[r for r in rows if r['alias']==alias and r['task']==task];counts=Counter(r['pool_id'] for r in selected)
            path=quality['paths'][family][task];a=b=0.
            for row in selected:
                x,y=api['quality'].sample_totals(task,path['samples'][row['pool_id']]);a+=x;b+=y
            manual[task][0]+=a;manual[task][1]+=b
            ids=quality['input_order'][task];bs=baseline.get((family,task,DEPLOYMENT[family]),{});total_b=bsum([r['b_ms'] for r in selected])
            mean_b=bsum([bs.get(p) for p in ids]);mean_b=mean_b/len(ids) if mean_b is not None else None
            expected=len(selected)*mean_b if mean_b is not None else None
            groups.append(dict(alias=alias,family=family,task=task,requests=len(selected),unique_inputs=len(counts),pool_size=len(ids),
                per_input_counts={p:counts[p] for p in ids},missing_inputs=sorted(set(ids)-set(counts)),
                scheduled_historical_error_numerator=a,scheduled_historical_reference_denominator=b,
                conditional_historical_loss=a/b if b else None,planned_sum_b_ms=total_b,full_pool_mean_b_ms=mean_b,
                same_count_uniform_pool_work_b_ms=expected,input_composition_work_difference_ms=total_b-expected if total_b is not None and expected is not None else None,
                scheduled_mean_b_ms=total_b/len(selected) if total_b is not None and selected else None))
    q,evidence=api['quality'].predict_quality(catalog,DEPLOYMENT,routing,RATES,trace,seed)
    manual_losses={t:a/b if b else None for t,(a,b) in manual.items()}
    for task in RATES:
        if manual_losses[task]!=evidence['per_task'][task]['loss']:raise ValueError('Manual historical task Q mismatch')
    manual_q=math.fsum(RATES[t]/math.fsum(RATES.values())*manual_losses[t] for t in RATES) if all(v is not None for v in manual_losses.values()) else None
    if q is None or not math.isclose(q,manual_q,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Exact historical total Q mismatch')
    text_rows=[r for r in rows if r['task']=='text'];tape=[r['alias'] for r in text_rows]
    period=finite_period(tape);selected=[r['task_position'] for r in text_rows if r['family']=='generalist_2'];flags=[a=='generalist_2#0' for a in tape]
    flag_period=finite_period(flags) if selected else None
    offsets=[i for i in range(period) if tape[i]=='generalist_2#0'] if period else []
    theoretical_weights=[share_fraction,(1-share_fraction)/2,(1-share_fraction)/2]
    lcm=math.lcm(*(w.denominator for w in theoretical_weights));integer=[int(w*lcm) for w in theoretical_weights];divisor=math.gcd(*integer)
    integer=[w//divisor for w in integer]
    specialists={g['alias']:g for g in groups if g['task']=='text' and g['family']=='text'}
    gap=imbalance(specialists['text#0']['planned_sum_b_ms'],specialists['text#1']['planned_sum_b_ms'])
    gap['request_count_gap']=abs(specialists['text#0']['requests']-specialists['text#1']['requests'])
    g2=next(g for g in groups if g['alias']=='generalist_2#0' and g['task']=='text')
    alias_work={a:bsum([group['planned_sum_b_ms'] for group in groups if group['alias']==a]) for a in aliases}
    return dict(seed=seed,share=share,share_float=g,routing=routing,trace_sha256=api['trace'].trace_digest(trace),
        cohort_sha256=digest([(r['arrival_seconds'],r['task'],r['pool_id']) for r in rows]),pool_orders=pool_orders,
        text_requests=len(text_rows),task_counts=dict(positions),alias_task_groups=groups,planned_sum_b_ms_by_alias=alias_work,
        text_specialist_planned_work_imbalance=gap,G2_text_coverage=dict(requests=g2['requests'],unique_inputs=g2['unique_inputs'],
            pool_size=32,fraction=g2['unique_inputs']/32,unique_upper_bound_for_request_count=min(g2['requests'],32),
            unused_possible_unique_slots=min(g2['requests'],32)-g2['unique_inputs']),
        resonance=dict(actual_G2_text_positions=selected,position_gap_counts=dict(Counter(b-a for a,b in zip(selected,selected[1:]))),
            observed_text_position_mod32_counts=dict(Counter(i%32 for i in selected)),finite_alias_tape_period=period,
            finite_G2_selection_period=flag_period,observed_period_G2_offsets=offsets,
            observed_alias_period_gcd32=math.gcd(period,32) if period else None,
            hypothetical_periodic_motif_reachable_pool_slots=periodic_residues(period,offsets,32),
            rational_minimal_integer_weights_G2_text0_text1=integer,rational_weight_sum=sum(integer),
            rational_weight_sum_gcd32=math.gcd(sum(integer),32),
            scope='Actual finite original float-SWRR tape is authoritative. Rational weight-sum and repeated-motif reachability are descriptive diagnostics, not infinite-cycle or performance guarantees.'),
        historical_exact_Q_loss=q,historical_exact_task_loss=manual_losses,historical_quality_evidence=evidence,
        manual_Q_matches_frozen_predict_quality=True,rows=rows,U=None,formal_rho=None,actual_service=None)


def replay(output=DESTINATION):
    output=Path(output).resolve();design_path=output/'design.json';manifest=json.loads((output/'design-manifest.json').read_text())
    if ref(design_path)!=manifest['design']:raise ValueError('Frozen design changed')
    design=json.loads(design_path.read_text())
    if design['shares']!=SHARES or design['seeds']!=SEEDS or design['cases']!=[dict(share=s,seed=seed) for seed in SEEDS for s in SHARES]:raise ValueError('Exact20 predeclared cases required')
    for source in design['files']:
        if ref(source['path'])!=source:raise ValueError('Frozen source changed')
    result=output/'results'
    if result.exists():raise FileExistsError('Never overwrite replay')
    packet=json.loads(Path(design['packet']['path']).read_text());config=packet['config']
    baseline,baseline_metadata=baselines(config);api=exposure.frozen_api()
    cases=[replay_case(config,design['samples']['path'],c['seed'],c['share'],baseline,api) for c in design['cases']]
    for seed in SEEDS:
        hashes={c['cohort_sha256'] for c in cases if c['seed']==seed}
        if len(hashes)!=1:raise ValueError('Route-independent task/input cohort changed')
    for source in design['files']:
        if ref(source['path'])!=source:raise ValueError('Input changed during replay')
    result.mkdir();overview=[]
    for i,case in enumerate(cases,1):
        stem=f'case-{i:02d}';rows=case.pop('rows')
        with (result/(stem+'-roster.csv')).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        case['roster_csv']=stem+'-roster.csv';save(result/(stem+'.json'),case)
        overview.append(dict(case=stem+'.json',seed=case['seed'],share=case['share'],text_requests=case['text_requests'],
            G2_text_coverage=case['G2_text_coverage'],historical_exact_Q_loss=case['historical_exact_Q_loss'],
            finite_alias_period=case['resonance']['finite_alias_tape_period'],finite_G2_period=case['resonance']['finite_G2_selection_period'],
            G2_position_gap_counts=case['resonance']['position_gap_counts'],
            planned_sum_b_ms_by_alias=case['planned_sum_b_ms_by_alias'],text_specialist_planned_work_imbalance=case['text_specialist_planned_work_imbalance']))
    report=dict(schema='gsdeploy.swr_period_exposure.report.v1',status='CPU_DIAGNOSTIC_ONLY',design=ref(design_path),
        cases=overview,case_count=20,baseline_metadata=baseline_metadata,
        cohort_equality_per_seed_verified=True,all_predeclared_cases_retained=True,
        scope='Unmodified runtime arrival/input/router reconstruction and frozen historical b/Q accounting; not mixed native service measurements or formal rho.',
        limitations=['Finite alias/input patterns depend on original floating-point SWRR tie decisions; theoretical rational-weight periods alone are not sufficient.',
            'Coverage is unique inputs in a fixed finite replay, not accuracy, representative sampling or capacity. Historical Q reuses old observed outputs and does not predict reproducibility under current concurrency.',
            'Planned b work sums reuse exact standalone family/task/n/input medians only. Alias imbalance may reflect input assignment as well as counts; it is not measured mixed service, utilization, formal rho or queue latency.',
            'All20 cases retained with no best route/seed choice. New denominators do not modify frozen route domain, search, packets or GPU queue.'],
        U=None,formal_rho=None,predictor_activated=False,search_domain_modified=False,GPU_started=False,
        raw_run_results_read=False,queue_read=False,reserved_seeds_read=False)
    save(result/'report.json',report)
    lines=['# SWRR 与 text32 输入周期暴露诊断','',
        '先封存design.json，再对全部10份额×2开发seed重建原make_trace→SamplePool→expanded-alias SWRR。H600，text1.85，背景各0.01；部署text2+G2+OCR。未读运行结果，无GPU、U或rho，不改搜索域。','',
        '| seed | G2份额 | G2请求 | G2唯一/32 | 有限alias周期 | 历史Q损失 | 专用b工作量差/均值 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for c in overview:
        coverage=c['G2_text_coverage'];gap=c['text_specialist_planned_work_imbalance']['relative_gap_to_mean']
        lines.append(f"| {c['seed']} | {c['share']} | {coverage['requests']} | {coverage['unique_inputs']} | {c['finite_alias_period']} | {c['historical_exact_Q_loss']:.9f} | {gap:.3%} |")
    lines+=['','每份额完整roster和alias/task逐输入计数、G2位置mod32/间距、冻结b工作量均已保存。历史Q由逐请求旧输出算得，并与冻结predict_quality完全核对。专用b差异不代表混合实测服务或排队收益；邻近分母改变覆盖也不能直接推出质量或总效用改善。','',
        'SWRR是任务内独立状态，到达时间与背景不会改变给定text序列的alias次序；seed改变text输入排列和600秒请求数，全部保留。有限重复motif与gcd32只解释观察到的覆盖，浮点tie使最简有理分母不必等于实际alias周期，不外推无限周期。']
    (result/'README.zh.md').write_text('\n'.join(lines)+'\n')
    save(result/'manifest.json',dict(design=ref(design_path),files={p.name:sha(p) for p in result.iterdir() if p.is_file()},
        all_frozen_inputs_unchanged=True,source_results_opened=[]))
    print(json.dumps(dict(output=str(result),report=ref(result/'report.json'),cases=20,overview=overview)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','replay'])
    parser.add_argument('--output',type=Path,default=DESTINATION);args=parser.parse_args()
    prepare(args.output) if args.mode=='prepare' else replay(args.output)
