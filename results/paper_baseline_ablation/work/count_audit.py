import sys,os,json,csv,hashlib
from pathlib import Path
ROOT=Path('/root/autodl-tmp/generalist-specialist');OUT=Path(os.environ.get('PAPER_EVAL_OUT',str(ROOT/'results/paper_baseline_ablation')));M35=ROOT/'results/module35_utility_routing';M34=ROOT/'results/module34_candidate_pool';sys.dont_write_bytecode=True;sys.path[:0]=[str(M34),str(M35),str(ROOT)]
from m35 import isolation
isolation.ROOT=OUT;isolation.install('supervisor')
from m34.cpu_core import candidate_routes
read=lambda p:json.loads(Path(p).read_text())
base=read(OUT/'baseline_cpu_results.json');eff=read(OUT/'solver_efficiency.json');protocol=read(OUT/'experiment_protocol.json');sp=protocol['candidate_space'];audits=[]
def csvsave(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()})
for b in base:
 s,pool,alg=b['scenario'],b['pool'],b['algorithm']
 if alg!='V3_J_GRID_CONTROL':continue
 f=read(M35/f'cpu/{s}_{pool}_V3_GRID.json');z=f['result'];unique={}
 def add(x):
  active={m:n for m,n in x.items() if n};unique[json.dumps(active,sort_keys=True)]=active
 add(z['seed']['deployment']);add(z['deployment'])
 for it in z['iterations']:
  if 'J_current_seconds' in it:add(it['deployment'])
  for c in it['candidates']:
   if 'J_star_seconds' in c:add(c['mutation']['deployment'])
 for c in z['actual_serving_route_evaluations']:add(c['deployment'])
 assert f['J_route_solves']==2*len(unique),(s,pool,'CACHE_RECONSTRUCTION_MISMATCH')
 rates=next(x['rates'] for x in protocol['scenarios'] if x['scenario']==s);d=read(sp['pools'][pool]['path']);count=sum(sum(1 for _ in candidate_routes(d,x,rates,1)) for x in unique.values())
 b['J_surrogate_evaluations']=count;b['J_unique_route_deployment_states']=len(unique);b['J_count_status']='reconstructed_from_this_frozen_fit_logged_cache_deployments; duplicate_sharedcache_removed'
 audits.append(dict(scenario=s,pool=pool,exact_J_surrogate_evaluations=count,unique_J_cached_deployments=len(unique),legacy_double_cache_count=f['J_route_solves'],reconstruction_asserted=True,method='existing34 count_audit exactrule reused; no optimizerexecution'))
 for e in eff:
  if e['scenario']==s and e['pool']==pool and e['method']==alg:e['J_surrogate_evaluations']=count;e['J_unique_route_deployment_states']=len(unique);e['J_count_status']=b['J_count_status']
for e in eff:
 if e['method'] in ['20A_U_GRID','V3_J_GRID_CONTROL']:
  oracle=next(b for b in base if b['scenario']==e['scenario'] and b['pool']==e['pool'] and b['algorithm']=='CPU_U_ORACLE')
  e['historical_speedup_vs_finite_U_plus_J_oracle']=oracle['seconds']/e['seconds_median']
  e['historical_speedup_scope']='same finite domain in original35 CPUbatch, single historical observations; Oracle includes Jdiagnostic; distinct from new U-only medians'
for name,rows in [('baseline_cpu_results',base),('solver_efficiency',eff)]:
 (OUT/(name+'.json')).write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n');csvsave(name+'.csv',rows)
csvsave('paper_figures/figure1_regret.csv',[b for b in base if b['strict_same_search_domain']]);(OUT/'actual_J_count_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
for name in ['baseline_comparability_audit.md','paper_ready_findings.md']:
 p=OUT/name;s=p.read_text();s+='\n计数补充：V3网格全部12单元的实际J代理评价数已按冻结候选日志和共享缓存键重建，严格检查旧计数为同一缓存的两倍；见 `actual_J_count_audit.json`。原生J内部调用仍未知。20A/V3网格的同批次历史有限U+J枚举加速比另列 `historical_speedup_vs_finite_U_plus_J_oracle`，不与新U-only重复计时混为同一数字。\n';p.write_text(s)
# Preserve this additional countaudit as a distinct documented step in reproduction.
p=OUT/'README.md';s=p.read_text();needle='PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/bin/python /root/autodl-tmp/generalist-specialist/results/paper_baseline_ablation/work/plot.py'
s=s.replace(needle,'PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/gsdeploy/bin/python /root/autodl-tmp/generalist-specialist/results/paper_baseline_ablation/work/count_audit.py\n'+needle);p.write_text(s)
for p in (OUT/'paper_figures').glob('*_raw_data.json'):
 data=read(p)
 for r in data['files']:
  q=Path(r['path']);r.update(sha256=hashlib.sha256(q.read_bytes()).hexdigest(),bytes=q.stat().st_size,mtime_ns=q.stat().st_mtime_ns)
 p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
print('ACTUAL_J_CALL_COUNT_AUDIT_PASS',audits)
