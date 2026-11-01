#!/usr/bin/env python3
"""CPU standalone-evidence challenge screening, never a validated utility search."""
import argparse
import csv
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'artifacts/static-spillover-study-20261022'
SOURCE=ROOT/'artifacts/static-study-gpu-20261022/execution-source-v5'
sys.path.insert(0,str(SOURCE))
from gsdeploy.quality import predict_quality
from gsdeploy.trace import make_trace
from gsdeploy.config import SystemConfig

RATES={'text':1.85,'image':.01,'asr':.01,'ocr':.01}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def solve_routes(c,services,x,objective='maxrho'):
    """Continuous optimistic capacity relaxation over measured paths only."""
    models=list(x);tasks=list(RATES)
    edges=[(t,m) for t in tasks for m in models if (m,t,x[m]) in services]
    if any(not any(t==tt for tt,m in edges) for t in tasks):return None
    Aeq=np.zeros((4,len(edges)+1));Aub=np.zeros((len(models),len(edges)+1))
    costs=np.zeros(len(edges)+1)
    for j,(t,m) in enumerate(edges):
        Aeq[tasks.index(t),j]=1
        Aub[models.index(m),j]=services[m,t,x[m]]['original_profile_mean_service_ms']/1000/c['capacity_factors'][m][str(x[m])]
        p=c['quality']['paths'][m][t]
        costs[j]=(1-p['value'] if t in {'text','image'} else p['value'])/sum(RATES.values()) if objective=='quality' else 0
    Aub[:,-1]=-1
    if objective=='maxrho':costs[-1]=1
    result=linprog(costs,A_ub=Aub,b_ub=np.zeros(len(models)),A_eq=Aeq,b_eq=list(RATES.values()),
        bounds=[(0,None)]*len(edges)+[(0,.8 if objective=='quality' else None)],method='highs')
    if not result.success:return None
    routing={t:{} for t in tasks}
    for (t,m),q in zip(edges,result.x):
        if q>1e-10:routing[t][m]=float(q)
    return routing

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    config_path=STUDY/'text185-fine-development-v1/system-config.json'
    protocol_path=STUDY/'text185-fine-development-v1/discovery-protocol.json'
    config=json.loads(config_path.read_text());c=config['catalog'];formal_config=SystemConfig(**config)
    assert config['workload']==RATES and config['gpu_memory_mib']==49140 and config['horizon_seconds']==600
    inputs={str(config_path.resolve()):sha(config_path),str(protocol_path.resolve()):sha(protocol_path)}
    route_domain_path=ROOT/'gsdeploy/route_domain.py'
    inputs[str(route_domain_path.resolve())]=sha(route_domain_path)
    import importlib.util
    spec=importlib.util.spec_from_file_location('screen_shared_route_domain',route_domain_path)
    route_domain=importlib.util.module_from_spec(spec);spec.loader.exec_module(route_domain)
    services={};profiles=[]
    for m,ts in c['task_replica_profiles'].items():
        for t,ns in ts.items():
            for sn,p in ns.items():
                samples={};records=0
                for filename in p['raw_service_samples']:
                    path=Path(filename)
                    if not path.is_relative_to(ROOT/'artifacts/retained-replica-profiles-20261021'):
                        raise ValueError('Only frozen retained service profile JSONL accepted')
                    inputs[str(path)]=sha(path)
                    for line in path.read_text().splitlines():
                        if not line.strip():continue
                        row=json.loads(line);records+=1
                        if row.get('success') is not True or not isinstance(row.get('service_ms'),(int,float)):
                            raise ValueError('Incomplete standalone service evidence')
                        samples.setdefault(row['sample_id'],[]).append(row['service_ms'])
                required=set(c['quality']['input_order'][t])
                missing=sorted(required-set(samples))
                value=statistics.mean(statistics.median(samples[k]) for k in required) if not missing else None
                row=dict(model=m,task=t,replicas=int(sn),records=records,unique_inputs=len(samples),
                    required_inputs=len(required),missing_inputs=missing,pool_mean_of_input_medians_ms=value,
                    original_profile_mean_service_ms=p['mean_service_ms'],scope=p['scope'])
                profiles.append(row)
                if value is not None:services[m,t,int(sn)]=row

    def inspect(x,routing=None):
        memory=c['safe_factor']*math.fsum(c['memory_table_mib'][m][str(n)] for m,n in x.items())
        unknown=[dict(model=m,replicas=n,task=t) for m,n in x.items() for t in c['models'][m]['executable_tasks'] if (m,t,n) not in services]
        rho={m:0. for m in x};formal_rho={m:0. for m in x};Q=None
        if routing is not None:
            for t,row in routing.items():
                assert math.isclose(sum(row.values()),RATES[t],abs_tol=1e-8)
                for m,q in row.items():
                    rho[m]+=q*services[m,t,x[m]]['pool_mean_of_input_medians_ms']/1000/c['capacity_factors'][m][str(x[m])]
                    formal_rho[m]+=q*services[m,t,x[m]]['original_profile_mean_service_ms']/1000/c['capacity_factors'][m][str(x[m])]
            Q=sum(q*(1-c['quality']['paths'][m][t]['value'] if t in {'text','image'} else c['quality']['paths'][m][t]['value']) for t,row in routing.items() for m,q in row.items())/sum(RATES.values())
        receipt=None;formal_error=None
        if routing:
            try:receipt=formal_config.validate_plan(x,routing)
            except ValueError as exc:formal_error=str(exc)
        if receipt is not None:
            assert math.isclose(receipt['max_rho'],max(formal_rho.values()),abs_tol=1e-10)
            assert math.isclose(receipt['safe_memory_mib'],memory,abs_tol=1e-6)
        return dict(deployment=x,routing=routing,safe_residency_proxy_mib=memory,memory_proxy_pass=memory<=49140,
            missing_profile_edges=unknown,input_median_mean_rho=rho if routing else None,
            input_median_mean_max_rho=max(rho.values()) if routing else None,
            formal_profile_mean_rho=formal_rho if routing else None,max_rho=max(formal_rho.values()) if routing else None,
            historical_aggregate_Q_proxy=Q,production_U=None,measured_U=None,E=None,
            frozen_config_validate_plan=receipt,frozen_config_validate_plan_error=formal_error,
            proxy_constraints_pass=bool(routing and memory<=49140 and max(formal_rho.values())<=.8+1e-8),
            physical_feasibility='UNKNOWN',validated_utility=False)

    names=list(c['memory_table_mib']);all_states=[];count=0
    for ns in itertools.product(*(list(c['memory_table_mib'][m]) for m in names)):
        count+=1;x={m:int(n) for m,n in zip(names,ns) if int(n)}
        memory=c['safe_factor']*sum(c['memory_table_mib'][m][str(n)] for m,n in x.items())
        if not x or memory>49140:continue
        route=solve_routes(c,services,x);row=inspect(x,route)
        if route is not None and row['max_rho']<=.8+1e-8:status='KNOWN_ROUTE_PROXY_FEASIBLE'
        elif row['missing_profile_edges']:status='UNKNOWN_MISSING_PROFILE'
        else:status='COMPLETE_PROFILE_CAPACITY_PROXY_FAIL'
        row['status']=status;all_states.append(row)
    save(out/'all-memory-proxy-states.json',all_states)
    candidates=[];seen=set()
    def add(label,x,route=None,reason='',relaxation=False):
        if route is None:route=solve_routes(c,services,x)
        key=json.dumps([x,route],sort_keys=True)
        if key in seen:return
        seen.add(key);row=inspect(x,route);row.update(id=label,selection_reason=reason,
            routing_domain='CONTINUOUS_RELAXATION_NOT_NATIVE_GRID_CERTIFICATE' if relaxation else 'EXPLICIT_NATIVE_ROUTE',
            all_pool_state_retained=True)
        row['shared_route_domain_membership']={}
        if route:
            for t,r in route.items():
                models=tuple(sorted(r));K=config['route_grid'];ds=[32,64] if t=='text' else []
                member=any(all(math.isclose(r.get(m,0),RATES[t]*f.numerator/f.denominator,abs_tol=1e-10,rel_tol=1e-10) for m,f in fs)
                    for fs in route_domain.fraction_rows(models,K,ds) if len(fs)==len(models))
                row['shared_route_domain_membership'][t]=dict(member=member,base_K=K,pair_denominators=ds,
                    check='actual current fraction_rows exact rational enumeration; original route numbers unmodified')
        row['all_task_routes_in_declared_shared_domain']=bool(route and all(v['member'] for v in row['shared_route_domain_membership'].values()))
        row['historical_seed_quality']={}
        if route:
            for seed in (107,108):
                trace=make_trace(RATES,600,seed,'poisson');q,e=predict_quality(c,x,route,RATES,trace,seed)
                row['historical_seed_quality'][str(seed)]={'Q':q,'text_EM':1-e['per_task']['text']['loss'] if q is not None else None,
                    'concurrent_reproduction_verified':False}
        candidates.append(row)
    def backgrounds(g):return {'image':{g:.01},'asr':{g:.01},'ocr':{'ocr':.01}}
    for share in (0,1/32):
        rt={'text':{'text':1.85*(1-share)},**backgrounds('generalist_2')}
        if share:rt['text']['generalist_2']=1.85*share
        add('target_text2_G2_OCR_split_'+str(share),{'text':2,'generalist_2':1,'ocr':1},rt,'已冻结局部开发对，作为相同比较起点')
    for n in (1,2,3,4):
        x={'text':n,'image':1,'ocr':1,'asr':1};rt={t:{t:r} for t,r in RATES.items()}
        add('all_specialist_text'+str(n),x,rt,'纯专全任务对照；原始驻留proxy不含旧倍率，不能先删')
    for g in ('generalist_1','generalist_3'):
        for share in (0,1/32):
            rt={'text':{'text':1.85*(1-share)},**backgrounds(g)}
            if share:rt['text'][g]=1.85*share
            add(f'text2_{g}_native_share_{share}',{'text':2,g:1,'ocr':1},rt,
                '与主部署同text副本/同明确路由份额，仅更换通用家族，识别质量与背景干扰竞争')
    add('text1_G2_native_half',{'text':1,'generalist_2':1,'ocr':1},
        {'text':{'text':.925,'generalist_2':.925},**backgrounds('generalist_2')},
        '单text副本以较低驻留和较低rho挑战；质量损失由原输出实测，不能预先弱化')
    add('text3_G2_native_nosplit',{'text':3,'generalist_2':1,'ocr':1},
        {'text':{'text':1.85},**backgrounds('generalist_2')},
        '额外text副本低rho挑战；已有不同负载text26 n3实测内存超限，仅为风险证据，不将其负载迁移为当前已知结果')
    add('pure_G2_G3_native_half',{'generalist_2':1,'generalist_3':1},
        {'text':{'generalist_2':.925,'generalist_3':.925},
            'image':{'generalist_2':.01},'asr':{'generalist_2':.01},'ocr':{'generalist_2':.01}},
        '仅19120MiB驻留的纯通双家族路由；可能凭内存/能耗抵偿精度，需实测')
    add('pure_G3x3_native',{'generalist_3':3},
        {t:{'generalist_3':r} for t,r in RATES.items()},'G3纯通较强历史质量对照，全部路径有n3画像')
    add('text2_G2_image_native_fine',{'text':2,'generalist_2':1,'image':1,'ocr':1},
        {'text':{'text':1.85*31/32,'generalist_2':1.85/32},
            'image':{'image':.01},'asr':{'generalist_2':.01},'ocr':{'ocr':.01}},
        '主细分流同路由增加image专用；质量与背景解耦强对照，需衡量额外驻留代价')
    for g in ('generalist_1','generalist_2','generalist_3'):
        for n in (1,2,3):
            add(f'text{n}_{g}_OCR_minrho',{'text':n,g:1,'ocr':1},reason='其他专用副本数/通用家族；容量余量挑战',relaxation=True)
    for suffix,x in [
        ('G2x2',{'text':2,'generalist_2':2,'ocr':1}),
        ('G2_image',{'text':2,'generalist_2':1,'image':1,'ocr':1}),
        ('G2_asr',{'text':2,'generalist_2':1,'asr':1,'ocr':1}),
        ('G2_G3',{'text':2,'generalist_2':1,'generalist_3':1,'ocr':1}),
        ('text1_G2x2',{'text':1,'generalist_2':2,'ocr':1})]:
        add(suffix,x,reason='低负载背景专用化/额外通用/单text多通用强竞争结构',relaxation=True)
    feasible=[r for r in all_states if r['proxy_constraints_pass']]
    groups={'pure_general':lambda x:all(c['models'][m]['role']=='generalist' for m in x),
        'with_G1':lambda x:'generalist_1' in x,'with_G3':lambda x:'generalist_3' in x,
        'text_absent':lambda x:'text' not in x,'all_known_feasible':lambda x:True}
    for group,accept in groups.items():
        subset=[r for r in feasible if accept(r['deployment'])]
        if not subset:continue
        for key in ('max_rho','historical_aggregate_Q_proxy','safe_residency_proxy_mib'):
            best=min(subset,key=lambda r:r[key]);add(group+'_best_'+key,best['deployment'],best['routing'],
                f'完整驻留proxy枚举的minmax-rho见证中，{group}按单指标{key}挑选；不是该组全路由指标最优或总效用排序',True)
    # Quality LP produces a separate high-accuracy adversary at the unchanged rho limit.
    for base in list(candidates):
        if base['memory_proxy_pass']:
            rt=solve_routes(c,services,base['deployment'],'quality')
            if rt:add(base['id']+'_qualityLP',base['deployment'],rt,'相同部署rho<=.8内历史质量最小损失连续挑战；须另映射当前网格',True)
    save(out/'candidates.json',candidates)
    audit_path=STUDY/'text26-development-audit-v2/audit.json'
    related=None
    if audit_path.is_file():
        inputs[str(audit_path.resolve())]=sha(audit_path)
        audited=json.loads(audit_path.read_text())
        old=next(r for r in audited['conditions'] if r['label']=='text26_n3_specialist_share_4of7')
        related=dict(source=str(audit_path.resolve()),workload=old['workload'],
            deployment=old['plan']['deployment'],memory_evidence=old['memory_evidence'],
            stability=old['stability']['stability_state_v2'],U=old['utility']['U'],
            implication='Observed n3 memory invalid at a different workload; warns original residency proxy is not hardware guarantee. No utility/service transfer to text1.85.')
    counts={s:sum(r['status']==s for r in all_states) for s in sorted({r['status'] for r in all_states})}
    report=dict(schema='gsdeploy.text185_challenger_screen.v1',status='CPU_DIAGNOSTIC_NOT_VALIDATED',
        workload=RATES,horizon_seconds=600,seeds=[107,108],rho_limit=.8,safe_memory_limit_mib=49140,
        count_vectors_including_empty=count,memory_proxy_states=len(all_states),screen_counts=counts,
        known_feasible_states_with_unknown_other_edges=sum(r['proxy_constraints_pass'] and bool(r['missing_profile_edges']) for r in all_states),
        profile_rows=len(profiles),candidate_count=len(candidates),candidates=candidates,related_development_memory_warning=related,
        constraints='formal-profile rho=sum(route_rps*task_replica_profiles.mean_service_ms/1000)/frozen_capacity_factor; separate input-median sensitivity rho uses mean_of_per_input_median_ms; safe_memory=frozen_safe_factor*sum(frozen_residency)',
        full_pool_optimum=False,production_U=None,validated_utility=False,gpu_started=False,
        old_transfer_model_used=False,old_energy_used=False,heldout_read=False,
        limitations=['Standalone closed-loop service is a proxy for new concurrent open arrival, not a bound on real throughput.',
            'Memory proxy failure excludes only this arithmetic model, never proves measured hardware infeasibility.',
            'All memory-proxy states retained; missing n/task profiles remain UNKNOWN, no n1 fallback.',
            'Continuous routes are optimistic screening witnesses, not membership or optimum certificates in the shared native route domain.',
            'No QTLPE/U ranking, no queue tail, no energy; historical Q reproduction unverified.',
            'Only 107/108 development quality replay; no independent validation or confirmation read.'])
    save(out/'profiles.json',profiles);save(out/'report.json',report)
    lines=['# Text1.85 强其他部署 CPU 挑战集合','',
        '状态：CPU诊断，尚无已验证总效用。全七家族全部驻留行枚举；不启GPU、不用旧failed transfer/能耗、不修改任何冻结合同。',
        '',f'负载{RATES}、Poisson/H600；rho≤.8、safe显存≤49140。原始count向量{count}，驻留proxy通过{len(all_states)}，状态{counts}。缺失路径始终UNKNOWN，全部记录见all-memory-proxy-states.json。',
        '', '主容量约束使用config.validate_plan同task_replica_proxy算术：task_replica_profiles.mean_service_ms / capacity_factor。另列全部历史输入的逐输入中位数均值rho作为敏感性，二者不能混称。n状态使用原画像，不插值不回退。显存仅原safe_factor×驻留和，不含旧学习倍率。通过只表示proxy有已知路径见证，真实同卡干扰/峰值/质量/能耗仍UNKNOWN。',
        '', '|候选|部署|formal profile rho最大|input median rho最大|safe MiB|Q历史聚合|proxy通过|未知边数|', '|---|---|---:|---:|---:|---:|---|---:|']
    for r in candidates:
        f=lambda x:'UNKNOWN' if x is None else f'{x:.6f}'
        lines.append(f"|{r['id']}|{json.dumps(r['deployment'],ensure_ascii=False)}|{f(r['max_rho'])}|{f(r['input_median_mean_max_rho'])}|{r['safe_residency_proxy_mib']:.2f}|{f(r['historical_aggregate_Q_proxy'])}|{r['proxy_constraints_pass']}|{len(r['missing_profile_edges'])}|")
    lines+=['','每条精确绝对路由、107/108原SWRR历史质量、缺失边及选点理由见candidates.json；LP路由均标连续松弛，不假装当前共同网格原生可执行或全池证书。全枚举里的“best历史Q”仅对minmax-rho见证选点，不是该组全路由Q最优。',
        '', '建议先保留text2+image1+OCR1+ASR1纯专作为强质量/低背景开销对照，再从G3/G1替代、单text多G2、额外G2或image专用化中按容量余量/质量/驻留多指标选择2—3个。先对候选映射冻结共同网格并做same-trace proxy复核，再由实测同负载Q/T/L/P/M/E决定强弱；不能仅以局部微分流两点差异推称全池最优。',
        '', '少量直接原生强挑战优先：①all_specialist_text2（背景全专用）；②text2_generalist_3_native_share_0.03125（同1/32细分流换G3）；③text1_G2_native_half（单专用副本半分流）；④pure_G2_G3_native_half（低驻留纯通）；备选text2_G2_image_native_fine（改善image质量/干扰）与pure_G3x3_native。G1明确原生0/1/32路由也保留，不用慢历史均值手工删掉。推荐是多指标暴露强替代，不能保证少量实测足以证明全池最优。',
        '', 'text3_G2_native_nosplit虽然原驻留proxy通过，另有text26不同负载n3真实safe_memory=50219.487632、resource_valid=False证据；该条需资源审查并保留UNKNOWN，不将原proxy43000级显存称实测可行，也不把不同负载结果当当前失败。',
        '', '纯通、G1/G3与缺失画像域均被保留；即使proxy容量不足也不是物理失败证明。当前目标四条只能检验固定部署的局部路由差异，尚不能证明算法在全七池主动选中最优。','']
    (out/'README.zh.md').write_text('\n'.join(lines))
    assert all(sha(p)==digest for p,digest in inputs.items())
    save(out/'manifest.json',dict(script={'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},
        inputs=inputs,files={p.name:sha(p) for p in out.iterdir() if p.is_file()},all_inputs_unchanged=True,gpu_started=False))
    print(json.dumps({k:report[k] for k in ('status','count_vectors_including_empty','memory_proxy_states','screen_counts','candidate_count')},ensure_ascii=False))

if __name__=='__main__':main()
