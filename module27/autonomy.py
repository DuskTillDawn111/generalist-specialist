"""Formal Module20 optimizer, unchanged utility, full measured states eligible."""
from .common import *
from module20.optimizer import Generator,predict_scenarios

def main():
 assert read(OUT/'independent_result.json')['Stage_D_permitted']
 d,refs,weights=inputs();f=read(OUT/'frozen_positive_candidate.json');rates=f['rates'];start=time.perf_counter();states=[]
 for ns in itertools.product(*(list(map(int,d['capacity_factors'][m])) for m in MODELS)):
  x={m:n for m,n in zip(MODELS,ns) if n};mem=d['safe_factor']*sum(d['memory_table_mib'][m][str(n)] for m,n in x.items())
  cover={t for m in x for t in d['models'][m]['supported_tasks']};reasons=[]
  if mem>BUDGET:reasons.append('MEMORY')
  if set(rates)-cover:reasons.append('TASK_COVERAGE')
  states.append(dict(deployment=x,safe_memory_mib=mem,reject_reasons=reasons))
 save('optimizer_all_measured_deployment_states.json',states)
 candidates=[];attempts=[];generator=Generator(d,refs)
 for cap in [.8,.95]:
  for fraction in [None,0.,.01,.05,.1,.2,.5,1.]:
   c=generator.solve(rates,BUDGET,TARGET,fraction,cap,weights);attempts.append(c)
   if 'deployment' in c:
    c['prediction']=predict_scenarios(d,c['deployment'],c['routing'],rates,BUDGET,refs,weights)
    c['source']='Unmodified Module20 MILP candidate generation';c['target_G_fraction']=c['routing'][TARGET].get(G,0)/rates[TARGET];candidates.append(c)
 # Official Module20 also retains frozen historical candidates. They compete,
 # with their original predictions and no priority, with all free candidates.
 for fraction in [0.,.1,.2,1.]:
  rr=routing(rates,fraction);x=f['deployment']
  candidates.append(dict(deployment=x,routing=rr,target_G_fraction=fraction,
   prediction=predict_scenarios(d,x,rr,rates,BUDGET,refs,weights),source='Preregistered exploration endpoints and grid; historical candidate seed only'))
 valid=[c for c in candidates if c['prediction']['eligible']]
 ranked=sorted(valid,key=lambda c:(-c['prediction']['U'],c['prediction']['safe_memory_mib'],identity([c['deployment'],c['routing']])))
 winner=ranked[0] if ranked else None
 exact=bool(winner and winner['deployment']==f['deployment'] and all(abs(winner['routing'].get(t,{}).get(m,0)-f['routing'].get(t,{}).get(m,0))<=1e-6*rates[t]
  for t in rates for m in MODELS))
 result=dict(at_utc=now(),status='POSITIVE_SPLIT_AUTONOMOUSLY_SELECTED' if exact else 'POSITIVE_SPLIT_EXISTS_OPTIMIZER_MISSED',
  winner=winner,exact_replicated_configuration_selected=exact,optimizer_autonomous=True,original_Module20_weights_refs_unchanged=True,
  no_forced_GS_no_phase_constraints=True,all_measured_model_count_choices_in_formal_MILP=True,
  total_deployment_states=len(states),candidate_count=len(candidates),ranked=ranked,solve_attempts=attempts,
  search_scope='Formal Module20 heuristic generator and official periodic scorer; no exhaustive continuous utility optimum claim',
  unknown_joint_replica_expansion_still_unverified=True,other_prediction_not_real_GPU_superiority=True,no_prediction_fit=True,
  measured_positive_route_different_from_algorithm_success=True,wall_seconds=time.perf_counter()-start)
 save('optimizer_autonomous_decision.json',result);csvsave('optimizer_candidate_ranking.csv',[dict(rank=i+1,**c) for i,c in enumerate(ranked)])
 print('STAGE_D_RESULT',result['status'],flush=True)

if __name__=='__main__':main()
