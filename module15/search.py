"""First full-pool three-phase witness under the predeclared capacity-only rules."""
import itertools
from .common import *
from .solver import ResourceSolver
from .baseline import run as baseline

def check(point,g,s,task,stage):
 if not point.get('winner'):return False,'NO_ACCEPTED_FULL_POOL_MINIMUM_MEMORY_J_WINNER'
 w=point['winner'];x=w['deployment'];r=w['routing']
 if stage=='LOW':return (x=={g:1},'FULL_POOL_LOW_NOT_EXACT_G1_ONLY')
 if x!={g:1,s:1}:return False,'FULL_POOL_COMPETITOR_OR_REPLICA_COUNT_REPLACES_CANDIDATE_G1_S1'
 rg=r.get(task+'::'+g,0);rs=r.get(task+'::'+s,0);total=rg+rs
 if stage=='MID':return (rs/total>=.8,'MID_S_FRACTION_BELOW_0_8')
 return (rg>1e-9 and rs>1e-9 and w['max_rho']<=.78,'HIGH_NOT_REAL_SPLIT_OR_MARGIN_INSUFFICIENT')

def main():
 f=read(OUT/'search_protocol_freeze.json');verify(f['implementation']+[f['catalog']]);assert not (OUT/'cpu_search_results.json').exists()
 data=read(OUT/'existing_empirical_catalog.json');solver=ResourceSolver(data);bg=f['background_rps'];bgt=f['background_task'];attempts=[];cache={};selected=None
 def solve(rates,B):
  key=(tuple(sorted(rates.items())),B)
  if key not in cache:cache[key]=solver.solve(rates,B)
  return cache[key]
 for task in f['task_priority']:
  candidates=[]
  for b in f['budgets']:
   for g,gp in data['models'].items():
    if gp['role']!='generalist' or task not in gp['supported_tasks'] or bgt not in gp['supported_tasks']:continue
    CG=(.8*data['capacity_factors'][g]['1']-bg*gp['service_time_ms'][bgt]/1000)/(gp['service_time_ms'][task]/1000)
    for s,sp in data['models'].items():
     if sp['role']!='specialist' or task not in sp['supported_tasks']:continue
     CS=.8*data['capacity_factors'][s]['1']/(sp['service_time_ms'][task]/1000)
     MG=data['safe_factor']*data['memory_table_mib'][g]['1'];MS=data['safe_factor']*data['memory_table_mib'][s]['1'];record={'target_task':task,'Generalist':g,'Specialist':s,'background_task':bgt,'background_rps':bg,'budget':b['id'],'budget_mib':b['safe_memory_budget_mib'],'M_G_safe_mib':MG,'M_S_safe_mib':MS,'C_G_rps':CG,'C_S_rps':CS,'candidate_memory_mib':MG+MS}
     if not 0<CG<CS:attempts.append({**record,'complete_witness':False,'rejection':'CAPACITY_ORDER_0_LT_CG_LT_CS_FAIL'});continue
     if MG+MS>b['safe_memory_budget_mib']:attempts.append({**record,'complete_witness':False,'rejection':'G1_S1_EXCEEDS_EXISTING_BUDGET'});continue
     candidates.append(record)
  candidates.sort(key=lambda r:(r['budget_mib'],r['candidate_memory_mib'],r['Generalist'],r['Specialist']))
  for r in candidates:
   g,s=r['Generalist'],r['Specialist'];CG,CS=r['C_G_rps'],r['C_S_rps'];points=[];failure=[]
   for stage in ['LOW','MID','HIGH']:
    found=None
    for fraction in f['point_fraction_order'][stage]:
     q=fraction*CG if stage=='LOW' else CG+fraction*(CS-CG) if stage=='MID' else CS+fraction*CG
     rates={task:q,bgt:bg};fit=solve(rates,r['budget_mib']);ok,why=check(fit,g,s,task,stage)
     failure.append({'stage':stage,'fraction':fraction,'target_rps':q,'accepted':ok,'rejection':None if ok else why,'full_pool_result':fit})
     if ok:found={'stage':stage,'fraction':fraction,'target_rps':q,'rates':rates,'full_pool_result':fit};break
    if found is None:break
    points.append(found)
   complete=len(points)==3;attempts.append({**r,'complete_witness':complete,'stage_checks':failure})
   print('CPU_CANDIDATE',task,r['budget'],g,s,'complete',complete,flush=True)
   if complete:selected={**r,'points':points};break
  if selected:break
 save('cpu_search_results.json',{'at_utc':now(),'selected':selected,'attempts':attempts,'CPU_unique_point_solves':len(cache),'full_model_pool':list(data['models']),'new_models_searched':0,'no_quality_memory_budget_latency_changes':True})
 if selected is None:
  save('module15_gate.json',{'at_utc':now(),'gate':'THREE_PHASE_NOT_FOUND','scope':'No full-pool theoretical witness under predeclared measured-state search; no global claim about unmeasured replica counts or every real QPS','mathematical_and_competition_rejections':attempts,'GPU_runs':0,'stop':True});print('NO_THEORETICAL_WITNESS_STOP');return
 g,s=selected['Generalist'],selected['Specialist'];B=selected['budget_mib'];details=[]
 for p in selected['points']:
  full=solver.solve(p['rates'],B,competition=10);assert full['winner']['deployment']==p['full_pool_result']['winner']['deployment'];assert abs(full['winner']['J_seconds']-p['full_pool_result']['winner']['J_seconds'])<1e-9
  p['full_pool_result']=full;p['original_V3_CPU_baseline']=baseline(p['rates'],B,solver)
  win=full['winner'];rg=win['routing'].get(selected['target_task']+'::'+g,0);rs=win['routing'].get(selected['target_task']+'::'+s,0)
  details.append({'stage':p['stage'],'target_task':selected['target_task'],'target_rps':p['target_rps'],'background_task':bgt,'background_rps':bg,'budget':selected['budget'],'budget_mib':B,'deployment':win['deployment'],'routing':win['routing'],'target_G_fraction':rg/p['target_rps'],'target_S_fraction':rs/p['target_rps'],'safe_memory_mib':win['safe_memory_mib'],'max_rho':win['max_rho'],'rho':win['rho'],'J_seconds':win['J_seconds'],'minimum_memory_lower_bound_mib':full['minimum_memory_lower_bound_mib'],'original_V3_status':p['original_V3_CPU_baseline'].get('status')})
 selected['M_G_raw_mib']=data['memory_table_mib'][g]['1'];selected['M_S_raw_mib']=data['memory_table_mib'][s]['1'];selected['G_plus_S2_safe_mib']=data['safe_factor']*(data['memory_table_mib'][g]['1']+data['memory_table_mib'][s]['2']) if '2' in data['memory_table_mib'][s] else None
 selected['extra_measured_S_replica_excluded_by_budget']=selected['G_plus_S2_safe_mib'] is not None and B<selected['G_plus_S2_safe_mib'];selected['all_three_same_full_pool']=True;selected['fixed_G_not_forced_in_CPU_optimizer']=True
 save('theoretical_witness.json',{'at_utc':now(),'witness':selected,'CPU_only':True,'scope':'measured operational capacity plus minimum memory; mixed-task and co-resident throughput/stability require GPU','latency_SLO':None});csvsave('cpu_three_phase.csv',details)
 save('baseline_protocol_freeze.json',{'at_utc':now(),'clarifies_search_protocol_baseline_label':'Original V3 is module9bc.frozen_v3.create/snapshot, not the discrete mathematical global-J certificate. Execute original CPU dry-run and record actual J routes, candidates and status separately. No synthetic runtime labels or GPU baseline.','original_entrypoint':ref(ROOT/'module9bc/frozen_v3.py'),'frozen_definition':ref(ROOT/'results/module9bc_e2e_routing_ranking/v3_minimal_frozen.json'),'implementation':ref(ROOT/'module15/baseline.py'),'source_untouched':True})
 save('module15_gate.json',{'at_utc':now(),'gate':'THREE_PHASE_THEORY_ONLY','theoretical_witness':ref(OUT/'theoretical_witness.json'),'GPU_runs':0,'pending_frozen_GPU_validation':True})
 print('THEORETICAL_WITNESS',g,s,'budget',B,'CG/CS',selected['C_G_rps'],selected['C_S_rps'],'QPS',[(p['stage'],p['target_rps']) for p in selected['points']],flush=True)
if __name__=='__main__':main()
