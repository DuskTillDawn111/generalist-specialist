"""Module26 owns new files only; utility and historical workers are immutable."""
from module25.common import *
import os
from datetime import datetime,timezone
OUT=ROOT/'results/module26_positive_probe'
AROOT=ROOT/'results/module24_utility_calibration'
BROOT=ROOT/'results/module24b_high_validation'
def now():return datetime.now(timezone.utc).isoformat()
def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=json_native)+'\n')
def csvsave(name,rows,fields=()):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys([*fields,*(k for r in rows for k in r)])));w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True,default=json_native) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def protected_metadata():
    roots=[ROOT/'results',*[p for p in ROOT.glob('module*') if p.is_dir() and p.name!='module26']]
    return {str(p):[p.stat().st_size,p.stat().st_mtime_ns] for root in roots for p in root.rglob('*')
        if p.is_file() and OUT not in p.parents and '__pycache__' not in p.parts}
def inputs():
    d=read(AROOT/'selected_pool_catalog.json');definition,refs=utility_inputs()
    return d,refs,definition['weights']
def observe(raw,path):
    from module24.calibration import extract
    d,refs,weights=inputs()
    return extract(raw,26,path,refs,weights,d=d,split='INDEPENDENT_REPLICATION_NEVER_TRAIN')
def arithmetic(raw,d,refs,weights):
    """Independent raw arithmetic, not summary-based or completed-subset utility."""
    ph=raw.get('phases',{}).get('measurement');result=dict(U=None,eligible=False)
    if not ph:return result
    rs=ph['request_records'];done=[r for r in rs if r['status']=='completed'];rates=raw['slot']['lambda_rps']
    window=ph['window'];seconds=window['planned_duration_seconds'];total=sum(rates.values());per={}
    for t,q in rates.items():
        rr=[r for r in done if r['task']==t]
        if not rr:continue
        per[t]=dict(count=len(rr),mean_ms=float(np.mean([r['e2e_latency_ms'] for r in rr])),
            p95_ms=float(np.percentile([r['e2e_latency_ms'] for r in rr],95)),
            offered=sum(r['task']==t for r in rs),completed=len(rr))
    idle=[np.mean([v['owned_used_mib'] for v in raw[k]]) for k in ['postwarm_idle','postmeasurement_idle'] if raw.get(k)]
    if len(idle)!=2 or set(per)!=set(rates):return dict(**result,per_task=per)
    safe=float(d['safe_factor']*max(idle));peak=max((v['owned_used_mib'] for v in raw.get('telemetry',{}).get('samples',[])),default=None)
    within=sum(r['completion_time_ns']<=window['end_time_ns'] for r in done)
    T=1-min(1.,len(done)/len(rs),within/(total*seconds))
    L=sum(q/total*per[t]['mean_ms']/refs[t] for t,q in rates.items());P=sum(q/total*per[t]['p95_ms']/refs[t] for t,q in rates.items());M=safe/raw['slot']['safe_memory_budget_mib']
    parts={k+'_contribution':-weights[k]*v for k,v in dict(T=T,L=L,P=P,M=M).items()}
    result.update(raw_arithmetic_U=sum(parts.values()),**parts,T_norm=T,L_norm=L,P_norm=P,M_norm=M,
        per_task=per,safe_memory_mib=safe,sampled_peak_mib=peak,offered=len(rs),completed=len(done),within=within,
        offered_qps=len(rs)/seconds,completed_cohort_qps=len(done)/seconds,within_qps=within/seconds,
        completion_ratio=len(done)/len(rs),planned_seconds=seconds,
        pooled_mean_ms=float(np.mean([r['e2e_latency_ms'] for r in done])),
        pooled_p95_ms=float(np.percentile([r['e2e_latency_ms'] for r in done],95)))
    # Stability and quality are checked independently by the existing classifier.
    ob=observe(raw,Path(raw['_path']) if '_path' in raw else OUT/'unused_path.json') if '_path' in raw else None
    if ob:
        result.update(eligible=ob['eligible'],U=result['raw_arithmetic_U'] if ob['eligible'] else None,hard_unstable=ob['hard_unstable'])
        if ob['eligible']:assert abs(result['U']-ob['U'])<1e-12
    return result
def cohort(raw):
    rs=raw['phases']['measurement']['request_records']
    return [(r['task'],r.get('pool_id'),r.get('sample_id'),r['frozen_offset_seconds'],r['routing_uniform'],r['replica_uniform']) for r in rs]

