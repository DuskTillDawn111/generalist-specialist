import os,sys,json,csv,time,hashlib,itertools,statistics,platform,copy
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path('/root/autodl-tmp/generalist-specialist'); OUT=Path(os.environ.get('PAPER_EVAL_OUT',str(ROOT/'results/paper_baseline_ablation'))); M35=ROOT/'results/module35_utility_routing'; M34=ROOT/'results/module34_candidate_pool'
sys.stdout=(OUT/'work/timing_resume.log').open('a',buffering=1);sys.stderr=sys.stdout
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
sys.path[:0]=[str(M35),str(M34),str(ROOT)]
# Reuse the existing guard with a process-local destination override; source unchanged.
from m35 import isolation
isolation.ROOT=OUT
isolation.install('supervisor')
import numpy as np
from scipy.stats import t,ttest_1samp
from u_aligned_optimizer import UAlignedOptimizer
from module20.optimizer import predict_scenarios
from m34.cpu_core import candidate_routes

def read(p):return json.loads(Path(p).read_text())
def save(name,x):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
def table(p):
 rows=list(csv.DictReader(Path(p).open()))
 for r in rows:
  for k,v in list(r.items()):
   if v=='':r[k]=None
   elif v in ['True','False']:r[k]=v=='True'
   elif v.startswith(('{','[')):
    try:r[k]=json.loads(v)
    except ValueError:pass
   else:
    try:r[k]=float(v)
    except ValueError:pass
 return rows

def csvsave(name,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r));p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def ref(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return dict(path=str(p),sha256=h.hexdigest(),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
def identity(x,r):return json.dumps([x,{t:{m:round(q,10) for m,q in rr.items() if q>1e-9} for t,rr in r.items()}],sort_keys=True)
fm=read(M34/'frozen_problem_manifest.json');sp=read(fm['space']['path']);refs=read(fm['utility_reference']['path'])['latency_reference_ms'];weights=read(fm['utility_definition']['path'])['weights'];budget=sp['budget_mib']
cases=[(s,x['rates'],'development') for s,x in sp['scenarios'].items()]+[('HOLDOUT_TEXT3',read(M35/'frozen_holdout_manifest.json')['rates'],'historical_holdout')]

base=table(OUT/'baseline_cpu_results.csv'); enriched=base; lookup={(r['scenario'],r['pool'],r['algorithm']):r for r in base};abl=read(OUT/'ablation_cpu_results.json');timings=table(OUT/'solver_timing_repetitions.csv')
# Recover an already completed repetition whose batch CSV was not flushed before process interruption.
for s,rates,split in cases:
 for pool in sp['pools']:
  for rep in range(1,4):
   a=OUT/f'raw_data/timing_{s}_{pool}_U_ONLY_ORACLE_{rep}.json';b=OUT/f'raw_data/timing_{s}_{pool}_M35_{rep}.json'
   if not (a.exists() and b.exists()) or any(r['scenario']==s and r['pool']==pool and r['repetition']==rep for r in timings):continue
   o,f=read(a),read(b)
   timings.append(dict(scenario=s,pool=pool,split=split,method='FINITE_U_ONLY_ORACLE',repetition=rep,seconds=o['seconds'],search_states=o['route_evaluations'],deployment_vectors=o['deployment_vectors'],full_utility_evaluations=o['route_evaluations'],scenario_prediction_calls=4*o['route_evaluations'],worker_prediction_calls=0,state_definition='full_route_leaves',time_scope='finite_full_problem_enumeration_without_J_diagnostic',raw_result=str(a.relative_to(OUT))))
   timings.append(dict(scenario=s,pool=pool,split=split,method='MODULE35_FROZEN',repetition=rep,seconds=f['seconds'],search_states=f['metrics']['search_states'],full_utility_evaluations=f['metrics']['full_prediction_calls']/4,scenario_prediction_calls=f['metrics']['full_prediction_calls'],worker_prediction_calls=f['metrics']['worker_prediction_calls'],state_definition='DPprefixlabels',time_scope='joint_K1_full_solve_including_final_validation',raw_result=str(b.relative_to(OUT)),**{k:v for k,v in f['metrics'].items() if k not in ['search_states','worker_prediction_calls']}))
save('work/timing_resume_event.json',dict(reason='Timing process absent before twelve-cellcompletion, no Pythonexception logged; completed repetitions recovered from immutable rawoutputs, missing repetitions resumed only',original_log='work/cpu.log',completed_repetition_rows=len(timings),no_completed_raw_overwrite=True))
raw_refs=read(OUT/'source_references.json') if (OUT/'source_references.json').exists() else []
# Timing-only U Oracle.
# Enumerate identical route domain; no J-surrogate diagnostics.
for s,rates,split in cases:
 for pool,pref in sp['pools'].items():
  d=read(pref['path']);oracle=lookup[s,pool,'CPU_U_ORACLE'];names=sorted(d['models'])
  for rep in range(1,4):
   if any(r['scenario']==s and r['pool']==pool and r['repetition']==rep and r['method']=='MODULE35_FROZEN' for r in timings):continue
   start=time.perf_counter();best=None;calls=0;feasible=0;vectors=0
   for bits in itertools.product((0,1),repeat=len(names)):
    vectors+=1;x={m:1 for m,b in zip(names,bits) if b};mem=d['safe_factor']*sum(d['memory_table_mib'][m]['1'] for m in x)
    if mem>budget+1e-6 or not set(rates)<={t for m in x for t in d['models'][m]['supported_tasks']}:continue
    for r in candidate_routes(d,x,rates,1):
     calls+=1;p=predict_scenarios(d,x,r,rates,budget,refs,weights)
     if p['eligible'] and p['max_rho']<=.8+1e-8:
      feasible+=1
      if best is None or (p['U'],-mem)>(best['prediction']['U'],-best['prediction']['safe_memory_mib']):best=dict(deployment=x,routing=r,prediction=p)
   sec=time.perf_counter()-start;assert abs(best['prediction']['U']-oracle['predicted_U'])<1e-8
   raw=f'raw_data/timing_{s}_{pool}_U_ONLY_ORACLE_{rep}.json';save(raw,dict(winner=best,seconds=sec,route_evaluations=calls,feasible_routes=feasible,deployment_vectors=vectors))
   timings.append(dict(scenario=s,pool=pool,split=split,method='FINITE_U_ONLY_ORACLE',repetition=rep,seconds=sec,search_states=calls,deployment_vectors=vectors,full_utility_evaluations=calls,scenario_prediction_calls=4*calls,worker_prediction_calls=0,state_definition='full_route_leaves',time_scope='finite_full_problem_enumeration_without_J_diagnostic',raw_result=raw))
   start=time.perf_counter();f=UAlignedOptimizer(d,rates,budget,refs,weights).solve(replica_options={m:[1] for m in d['models']});wall=time.perf_counter()-start
   assert abs(f['winner']['prediction']['U']-oracle['predicted_U'])<1e-8
   raw=f'raw_data/timing_{s}_{pool}_M35_{rep}.json';save(raw,f)
   timings.append(dict(scenario=s,pool=pool,split=split,method='MODULE35_FROZEN',repetition=rep,seconds=f['seconds'],outer_wall_seconds=wall,search_states=f['metrics']['search_states'],full_utility_evaluations=f['metrics']['full_prediction_calls']/4,scenario_prediction_calls=f['metrics']['full_prediction_calls'],worker_prediction_calls=f['metrics']['worker_prediction_calls'],state_definition='DPprefixlabels',time_scope='joint_K1_full_solve_including_final_validation',raw_result=raw,**{k:v for k,v in f['metrics'].items() if k not in ['search_states','worker_prediction_calls']}))
   print('TIMING',s,pool,rep,'UOracle',round(sec,4),'M35',round(f['seconds'],4),flush=True)
  csvsave('solver_timing_repetitions.csv',timings)
save('CPU_completed.json',dict(cells=12,baseline_rows=len(enriched),diagnostic_rows=len(abl),timing_rows=len(timings),no_new_optimizer=True,strict_global_ablation_A=False,strict_global_ablation_B=False,ablation_C_skipped=True))
save('source_references.json',raw_refs)
print('CPU_DONE',flush=True)
