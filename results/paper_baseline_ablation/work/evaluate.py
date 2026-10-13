import os,sys,json,csv,time,hashlib,itertools,statistics,platform,copy
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path('/root/autodl-tmp/generalist-specialist'); OUT=Path(os.environ.get('PAPER_EVAL_OUT',str(ROOT/'results/paper_baseline_ablation'))); M35=ROOT/'results/module35_utility_routing'; M34=ROOT/'results/module34_candidate_pool'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
sys.path[:0]=[str(M35),str(M34),str(ROOT)]
if (OUT/'final_gate.json').exists():raise SystemExit('Output is frozen. Set PAPER_EVAL_OUT to a new empty results directory for reproduction.')
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
# Snapshot scientific history before any evaluation. Exclude transient runtime caches.
protected=[]
for folder in [ROOT/'module20',ROOT/'deployment_core',ROOT/'method_v2',ROOT/'module9bc',ROOT/'results/module20_utility_optimizer',ROOT/'results/module33_mixed_oracle',M34,M35,ROOT/'results/paper_extra_g_only_retention']:
 for p in folder.rglob('*'):
  if p.is_file() and not p.is_symlink() and not set(p.relative_to(folder).parts)&{'work','__pycache__'}:protected.append(p)
protected=list(dict.fromkeys(protected));save('history_sha256_before.json',before:=[ref(p) for p in protected])
assert ref(M35/'u_aligned_optimizer.py')['sha256']=='c6727f4f4a4b18bdfd4e8767b9914f19f65fdf8e0ef8a1a5179c54aab53fb1a3'
protocol=dict(frozen_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),stage='CPU_PROTOCOL_FROZEN_BEFORE_NEW_DIAGNOSTICS',historical_results_already_known=True,
 deadline='2026-10-16 Asia/Shanghai',writing_start='2026-10-17 Asia/Shanghai',candidate_space=sp,weights=weights,latency_reference_ms=refs,budget_mib=budget,K=1,
 inputs=[ref(fm['space']['path']),ref(fm['utility_definition']['path']),ref(fm['utility_reference']['path']),ref(fm['quality_policy']['path']),ref(M35/'u_aligned_optimizer.py'),ref(M35/'frozen_holdout_manifest.json')],
 scenarios=[dict(scenario=s,rates=r,split=z) for s,r,z in cases],baseline_source='Reuse all twelve Module35 historical baseline cells; preserve native/control labels',
 ablation_A=dict(status='NO_INDEPENDENT_COMPONENT_SWITCH',substitute='At fixed original20A deployment, restrict candidate routes to historical20A routes for that deployment, versus unchangedModule35 fixed-deployment route DP. Candidate-coverage diagnostic, not unchanged-global-DP ablation.',oracle_injection=False),
 ablation_B=dict(status='FIXED_DEPLOYMENT_ROUTE_COMPONENT_ONLY',substitute='At fixed Module35 deployment, use same DP solve implementation with bundle mask restricted to fastest legal supported path in that deployment. No global deployment search ablation claimed.'),
 ablation_C=dict(status='SKIPPED_NOT_SINGLE_VARIABLE',reason='Frozen predict_scenarios uses isolated service profiles, replica scaling and FIFO; no independently switchable measured co-location correction. Stress multipliers are not co-location ablation.'),
 hard_gates=dict(complete_task_coverage=True,quality='Frozen supported_tasks after historical quality admission',replicas='binary0/1',memory='safe_factor times frozen measured residentmemory <=49140MiB',routing='full endpoint K1',rho_max=.8,nominal_drain_seconds=60,stress='report only, unchanged from original; not new hard gate',GPU='unverified by CPU'),
 timing=dict(repetitions=3,fresh_cache=True,scope='CPU solve wallclock including final four-scenario validation; exclude input loading and resultserialization',oracle='new U-only finiteenumeration, no J diagnostic; separately preserve historical U+J Oracle timings',counts='DPworkerbundles and full4scenario calls separate; utilitydecision evaluations not equal to stressscenario predictioncalls; MILPinternal nodes unknown'),
 GPU=dict(stage='DECIDE_AFTER_CPU',max_new_scenarios=2,reuse_only_if_exact_protocol=True,no_adaptive_trials=True),stop_after_completion=True)
save('experiment_protocol.json',protocol)
base=table(M35/'cpu_oracle_comparison.csv')+table(M35/'holdout_results.csv'); enriched=[]
lookup={(r['scenario'],r['pool'],r['algorithm']):r for r in base}
raw_refs=[]
for source in [M35/'cpu_oracle_comparison.csv',M35/'holdout_results.csv',M35/'solver_runtime_comparison.csv',M35/'solver_runtime_summary.csv',M35/'routing_gap_attribution.json',M34/'optimizer_evaluation_count_audit.json',ROOT/'results/module33_mixed_oracle/frozen_problem_manifest.json',ROOT/'results/module33_mixed_oracle/module33_gate.json',ROOT/'results/module20_utility_optimizer/module20_gate.json',ROOT/'results/paper_extra_g_only_retention/final_summary.md']:
 if source.exists():raw_refs.append(ref(source));(OUT/'raw_data'/source.name).write_bytes(source.read_bytes())
for s,rates,split in cases:
 for pool,pref in sp['pools'].items():
  d=read(pref['path']);oracle=lookup[s,pool,'CPU_U_ORACLE'];rows=table(M35/f'cpu/{s}_{pool}_oracle_candidates.csv')
  (OUT/f'raw_data/{s}_{pool}_oracle_candidates.csv').write_bytes((M35/f'cpu/{s}_{pool}_oracle_candidates.csv').read_bytes())
  eligible_keys={identity(z['deployment'],z['routing']) for z in rows if z['eligible']}
  for b in [z for z in base if z['scenario']==s and z['pool']==pool]:
   z=dict(b);x=z.get('deployment');r=z.get('routing');p=predict_scenarios(d,x,r,rates,budget,refs,weights) if x else None
   member=bool(x and identity(x,r) in eligible_keys)
   if p and z['predicted_U'] is not None:assert abs(p['U']-z['predicted_U'])<1e-8
   native=z['algorithm'] in ['20A_NATIVE_CONTINUOUS_PRESERVED','V3_NATIVE_CONTINUOUS_PRESERVED']
   z.update(split=split,method_domain='native_continuous' if native else 'finite_K1_binary_replicas',strict_same_search_domain=not native,
    selected_finite_member=member,regret=z['regret'] if not native else None,selected_plan_reference_gap=oracle['predicted_U']-p['U'] if member and p else None,
    regret_scope='same_domain' if not native else 'not_reported_native_domain; selected_plan_reference_gap_only',
    task_rates=rates,routing_fraction={t:{m:q/rates[t] for m,q in rr.items()} for t,rr in r.items()} if r else None,
    quality_gate=p['quality_capability'] if p else None,coverage_gate=set(r)==set(rates) if r else None,conservation_gate=p['route_conservation'] if p else None,
    memory_gate=p['safe_memory_mib']<=budget+1e-6 if p else None,stability_gate=p['max_rho']<=.8+1e-8 and p['incomplete_after_60s_drain']==0 if p else None,
    stress_all_eligible=p['stress_all_eligible'] if p else None,GPU_stability='NOT_ESTABLISHED_BY_CPU',source=str(M35/('holdout_results.csv' if split!='development' else 'cpu_oracle_comparison.csv')),
    time_scope='heuristic_plan_construction_excludes_validation' if 'FASTEST' in z['algorithm'] or 'STATIC' in z['algorithm'] else 'historical_CPU_solve',
    full_utility_evaluations=z.get('full_prediction_calls',0)/4 if z.get('full_prediction_calls') is not None else None,
    search_state_definition='DPprefixlabels' if z['algorithm']=='U_ALIGNED_DP' else 'enumerated_complete_routeplans' if z['algorithm']=='CPU_U_ORACLE' else 'unique_candidates_or_backendcalls; not MILPinternalnodes')
   if native:assert z['regret'] is None
   enriched.append(z)
csvsave('baseline_cpu_results.csv',enriched)
save('baseline_cpu_results.json',enriched)
# Both diagnostics use frozen deployment choices. No candidate defined using an Oracle winner.
class RouteMaskRestricted(UAlignedOptimizer):
 def __init__(self,*args,allowed_masks,**kwargs):super().__init__(*args,**kwargs);self.allowed_masks=allowed_masks;self.mask_restrictions=0
 def bundle(self,m,n,mask):
  if mask not in self.allowed_masks.get(m,set()):self.mask_restrictions+=1;return None
  return super().bundle(m,n,mask)
def masks_for(routes,tasks,x):
 return {m:{sum(1<<i for i,t in enumerate(tasks) if rr[t].get(m,0)>1e-9) for rr in routes} for m in x}
def fastest_route(d,x,rates):
 r={}
 for task,q in rates.items():
  allowed=[m for m in x if task in d['models'][m]['supported_tasks']]
  if not allowed:return None
  m=min(allowed,key=lambda m:(d['models'][m]['service_time_ms'][task]*x[m]/d['capacity_factors'][m][str(x[m])],d['memory_table_mib'][m][str(x[m])],m))
  r[task]={m:q}
 return r
abl=[];timings=[];coverage=[]
def rec_abl(s,pool,split,name,x,r,p,ss,metrics,oracle,full,scope,raw):
 eligible=bool(p and p['eligible'] and p['max_rho']<=.8+1e-8)
 return dict(scenario=s,pool=pool,split=split,ablation=name,scope=scope,deployment=x,routing=r,routing_fraction={t:{m:q/rates[t] for m,q in rr.items()} for t,rr in r.items()} if r else None,
  eligible=eligible,predicted_U=p['U'] if eligible else None,oracle_regret=oracle['predicted_U']-p['U'] if eligible else None,full_method_U=full,
  full_minus_variant_U=full-p['U'] if eligible else None,seconds_median=ss,metrics=metrics,raw_result=raw,
  coverage_gate=bool(r and set(r)==set(rates)),quality_gate=p.get('quality_capability') if p else None,memory_gate=p.get('safe_memory_mib',budget+1)<=budget if p else None,
  stability_gate=eligible,selected_safe_memory_mib=p.get('safe_memory_mib') if p else None,changed_tasks=[t for t in rates if r and r[t]!=lookup[s,pool,'U_ALIGNED_DP']['routing'][t]],
  single_variable_global_ablation=False,not_gpu_evidence=True)
for s,rates,split in cases:
 for pool,pref in sp['pools'].items():
  d=read(pref['path']);oracle=lookup[s,pool,'CPU_U_ORACLE'];new=lookup[s,pool,'U_ALIGNED_DP'];old=read(M35/f'cpu/{s}_{pool}_20A_GRID.json');xold=old['winner']['deployment'];xnew=new['deployment']
  for label,x in [('A_FIXED_20A_DEPLOYMENT_FULL_ROUTE',xold),('B_FIXED_M35_DEPLOYMENT_FULL_ROUTE',xnew)]:
   fits=[UAlignedOptimizer(d,rates,budget,refs,weights).solve(deployment=x) for _ in range(3)];f=fits[0];p=f['winner']['prediction'];r=f['winner']['routing'];raw=f'raw_data/{s}_{pool}_{label}.json';save(raw,dict(repetitions=fits))
   abl.append(rec_abl(s,pool,split,label,x,r,p,statistics.median(z['seconds'] for z in fits),f['metrics'],oracle,new['predicted_U'],'fixed_deployment_only; same_DP_core',raw))
  # Original complete route set from old generator on exactly the old deployment.
  oldroutes=[z['routing'] for z in old['candidates'] if z['deployment']==xold and z['prediction']['eligible'] and z['prediction']['max_rho']<=.8+1e-8]
  legal=[];secs=[]
  for rep in range(3):
   start=time.perf_counter();legal=[(predict_scenarios(d,xold,r,rates,budget,refs,weights),r) for r in oldroutes];secs.append(time.perf_counter()-start)
  p,r=max(legal,key=lambda z:z[0]['U']);full=next(z['predicted_U'] for z in abl if z['scenario']==s and z['pool']==pool and z['ablation']=='A_FIXED_20A_DEPLOYMENT_FULL_ROUTE')
  raw=f'raw_data/{s}_{pool}_A_OLD_GENERATED_ROUTES.json';save(raw,dict(deployment=xold,candidates=[dict(prediction=p,routing=r) for p,r in legal],replay_seconds=secs,original_generation_seconds=old['seconds'],generation_calls=old['generation_calls']))
  abl.append(rec_abl(s,pool,split,'A_OLD_ROUTE_COVERAGE_DIAGNOSTIC',xold,r,p,statistics.median(secs),dict(full_utility_evaluations=len(oldroutes),scenario_prediction_calls=4*len(oldroutes),generator_MILP_calls=42,generation_seconds=old['seconds'],generation_included_in_replay_time=False),oracle,full,'matched_deployment_candidate_coverage; NOT_global_component_ablation',raw))
  cover=dict(scenario=s,pool=pool,split=split,same_old_new_deployment=xold==xnew,old_routes=len(oldroutes),old_U=p['U'],expanded_U=full,recovered_U=full-p['U'],old_ranking_loss=old['winner']['prediction']['U']-p['U'],new_route_in_old_set=any(identity(xold,z)==identity(xold,next(a for a in abl if a['scenario']==s and a['pool']==pool and a['ablation']=='A_FIXED_20A_DEPLOYMENT_FULL_ROUTE')['routing']) for z in oldroutes));coverage.append(cover)
  rfast=fastest_route(d,xnew,rates);masks=masks_for([rfast],list(rates),xnew)
  fits=[]
  for rep in range(3):
   ss=RouteMaskRestricted(d,rates,budget,refs,weights,allowed_masks=masks);f=ss.solve(deployment=xnew);f['restricted_mask_count']=ss.mask_restrictions;fits.append(f)
  f=fits[0];p=f['winner']['prediction'] if f['winner'] else None;r=f['winner']['routing'] if f['winner'] else rfast
  assert not p or identity(xnew,r)==identity(xnew,rfast)
  raw=f'raw_data/{s}_{pool}_B_FASTEST_MASK.json';save(raw,dict(repetitions=fits,allowed_masks={m:sorted(v) for m,v in masks.items()}))
  abl.append(rec_abl(s,pool,split,'B_FIXED_M35_DEPLOYMENT_FASTEST',xnew,r,p,statistics.median(z['seconds'] for z in fits),f['metrics'],oracle,new['predicted_U'],'fixed_deployment_same_DP_core_route_mask_only; NOT_global_deployment_ablation',raw))
  abl.append(dict(scenario=s,pool=pool,split=split,ablation='C_COLOCATION_CORRECTION_OFF',status='SKIPPED_NOT_SINGLE_VARIABLE',predicted_U=None,oracle_regret=None,scope=protocol['ablation_C']['reason']))
  print('DIAGNOSTICS',s,pool,'A recovered',cover['recovered_U'],'B regret',None if not p else oracle['predicted_U']-p['U'],flush=True)
csvsave('ablation_cpu_results.csv',abl);save('ablation_cpu_results.json',abl);csvsave('route_candidate_coverage.csv',coverage)
# Timing-only U Oracle. Enumerate identical route domain; no J-surrogate diagnostics.
for s,rates,split in cases:
 for pool,pref in sp['pools'].items():
  d=read(pref['path']);oracle=lookup[s,pool,'CPU_U_ORACLE'];names=sorted(d['models'])
  for rep in range(1,4):
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
