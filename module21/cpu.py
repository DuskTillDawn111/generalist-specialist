from .common import *
from .optimizer import PoissonUtility
import itertools

def main():
 f=read(OUT/'input_protocol_freeze.json');d=read(f['catalog']['path']);u=read(f['utility_definition']['path']);refs=read(f['utility_references']['path'])['latency_reference_ms'];pred=PoissonUtility(d,refs,u['weights']);start=time.perf_counter();states=[];legal=[]
 for counts in itertools.product(*(list(map(int,d['capacity_factors'][m])) for m in MODELS)):
  x={m:n for m,n in zip(MODELS,counts) if n};mem=d['safe_factor']*sum(d['memory_table_mib'][m][str(n)] for m,n in x.items());cover=set(t for m in x for t in d['models'][m]['supported_tasks']);reasons=[]
  if mem>BUDGET+1e-6:reasons.append('SAFE_MEMORY_BUDGET')
  if set(next(iter(RATES.values())))-cover:reasons.append('TASK_COVERAGE')
  r={'deployment':x,'safe_memory_mib':mem,'reject_reasons':reasons,'measured_counts_verified':all(str(n) in d['memory_table_mib'][m] for m,n in x.items())};states.append(r)
  if not reasons:legal.append(x)
 save('all_deployment_states.json',states);by_stage={};summary=[]
 for stage,rates in RATES.items():
  fits=[]
  for i,x in enumerate(legal):
   t=time.perf_counter();fit=pred.optimize(x,rates);fit['solve_seconds']=time.perf_counter()-t;fits.append(fit);save('cpu/'+stage+f'_deployment_{i:03d}.json',fit)
   print('CPU',stage,i+1,'/',len(legal),'U',fit['winner']['U'] if fit['winner'] else None,flush=True)
  valid=[fit for fit in fits if fit['winner'] and fit['winner']['eligible']];best=max(valid,key=lambda r:(r['winner']['U'],-r['winner']['safe_memory_mib'],json.dumps(r['deployment'],sort_keys=True))) if valid else None
  # Refine every viable deployment, not just the favorable winner.
  fine=PoissonUtility(d,refs,u['weights'],dt=.00025,bins=65536);finefits=[]
  for i,fit in enumerate(valid):
   # Reoptimize around two DE solutions, coarse optimum and all corners at finer resolution.
   x=fit['deployment'];_,bounds=fine.domain(x,rates);free=[j for j,b in enumerate(bounds) if b[0]!=b[1]]
   base=fit['shares'];candidates=[base]
   for corner in itertools.product([0.,1.],repeat=len(free)):
    z=base.copy()
    for j,v in zip(free,corner):z[j]=v
    candidates.append(z)
   from scipy.optimize import minimize
   import numpy as np
   def expand(v):
    z=np.array(base);z[free]=v;return z
   if free:
    res=minimize(lambda v:fine.evaluate(x,rates,expand(v)),np.array(base)[free],method='Powell',bounds=[bounds[j] for j in free],options={'xtol':1e-7,'ftol':1e-9,'maxiter':100});candidates.append(expand(res.x).tolist())
   evaluated=[(fine.evaluate(x,rates,z,True),z) for z in candidates];evaluated=[(p,z) for p,z in evaluated if p.get('eligible')]
   p,z=max(evaluated,key=lambda v:v[0]['U']);new={**fit,'coarse_U':fit['winner']['U'],'shares':list(map(float,z)),'winner':p,'finer_resolution_reoptimized':True};finefits.append(new);save('cpu_fine/'+stage+f'_deployment_{i:03d}.json',new)
  winner=max(finefits,key=lambda r:(r['winner']['U'],-r['winner']['safe_memory_mib'],json.dumps(r['deployment'],sort_keys=True)))
  noM=[r for r in finefits if MMS not in r['deployment']];withM=[r for r in finefits if MMS in r['deployment']]
  if stage=='LOW':control=max(withM,key=lambda r:r['winner']['U']) if withM else None
  elif stage=='MID':control=max(noM,key=lambda r:r['winner']['U']) if noM else None
  else:
   z=winner['shares'].copy();z[2]=0.;p=fine.evaluate(winner['deployment'],rates,z,True);control={'deployment':winner['deployment'],'shares':z,'winner':p,'all_non_LID_routes_same':True} if p.get('eligible') else None
  delta=winner['winner']['U']-control['winner']['U'] if control else None
  pureS=fine.optimize(winner['deployment'],rates,{'lid':0.}) if MMS in winner['deployment'] else None
  by_stage[stage]={'rates':rates,'winner':winner,'control':control,'predicted_delta_U':delta,'coarse_winner_deployment':best['deployment'],'coarse_winner_shares':best['shares'],'fullpool_pure_S_background_reoptimized':pureS,'deployment_results':finefits,'nominal_stable_deployments':len(valid)}
  summary.append({'stage':stage,'deployment':winner['deployment'],'LID_G_fraction':winner['shares'][2],'predicted_U':winner['winner']['U'],'predicted_delta_U':delta,'control_nominal_feasible':control is not None,'safe_memory_mib':winner['winner']['safe_memory_mib']});save('cpu_stage_'+stage+'.json',by_stage[stage]);print('STAGE',stage,summary[-1],flush=True)
 checks={'LOW_no_target_S':MMS not in by_stage['LOW']['winner']['deployment'],'LOW_G_only':by_stage['LOW']['winner']['shares'][2]>=1-1e-6,'MID_autonomous_target_S':MMS in by_stage['MID']['winner']['deployment'],'HIGH_autonomous_mixed':1e-6<by_stage['HIGH']['winner']['shares'][2]<1-1e-6,'all_predicted_paired_gains_gt_001':all(v['predicted_delta_U'] is not None and v['predicted_delta_U']>.01 for v in by_stage.values()),'coarse_and_fine_phase_same':all((MMS in v['coarse_winner_deployment'])==(MMS in v['winner']['deployment']) and (1e-6<v['coarse_winner_shares'][2]<1-1e-6)==(1e-6<v['winner']['shares'][2]<1-1e-6) for v in by_stage.values()),'FFT_truncation_small_at_winners':all(v['winner']['winner']['FFT_tail_mass_last_decile']<1e-6 for v in by_stage.values())}
 result={'at_utc':now(),'Stage_A_passed':all(checks.values()),'checks':checks,'all_states_count':len(states),'budget_coverage_legal_deployments':len(legal),'all_legal_deployments_competed':True,'all_six_task_routes_optimized':True,'continuous_global_optimality_certificate':False,'solver_limit':'Two continuous DE seeds, corners, Powell and all-deployment finer CDF refinement; no proof of global continuous optimum. No finite target ratio grid imposed.','stage_summary':summary,'stages':by_stage,'CPU_seconds':time.perf_counter()-start,'predictor_evaluations_coarse':pred.evals,'utility_definition':f['utility_definition'],'utility_references':f['utility_references'],'GPU_evidence':False}
 save('cpu_fullpool_optimization.json',result)
 if not result['Stage_A_passed']:
  gate={'at_utc':now(),'gate':'THREE_PHASE_CPU_NOT_REPRODUCED','Stage_A_passed':False,'checks':checks,'GPU_started':False,'formal_GPU_runs':0,'reason':'Unconstrained full-pool optimum or frozen positive gain checks fail; stop before GPU.','no_Module22':True}
  save('module21_gate.json',gate);save('frozen_manifest.json',{'at_utc':now(),'status':'GPU_NOT_AUTHORIZED_BY_STAGE_A_GATE','input_protocol':ref(OUT/'input_protocol_freeze.json'),'CPU':ref(OUT/'cpu_fullpool_optimization.json'),'GPU_slots':[],'utility_definition':f['utility_definition'],'utility_references':f['utility_references'],'no_scientific_inputs_changed':True})
  for name,fields in [('gpu_paired_runs.csv',['run_id','stage','arm','repetition','status']),('system_utility_comparison.csv',['stage','repetition','actual_delta_U','evidence_status']),('per_task_metrics.csv',['run_id','task','offered','completed','mean_E2E_ms','P95_E2E_ms']),('worker_stability.csv',['run_id','worker','state','busy_fraction','end_backlog'])]:csvsave(name,[],fields)
 print('STAGE_A_PASS',result['Stage_A_passed'],checks,flush=True)
if __name__=='__main__':main()
