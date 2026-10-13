"""Independent measured-state enumeration + LP capacity audit of chosen points."""
import itertools,numpy as np
from scipy.optimize import linprog
from .common import *
def main():
 d=read(OUT/'existing_empirical_catalog.json');w=read(OUT/'theoretical_witness.json')['witness'];rates=w['points'][0]['rates'];models=sorted(m for m,p in d['models'].items() if set(p['supported_tasks'])&set(rates));vectors=itertools.product(*[[int(n) for n in d['memory_table_mib'][m]] for m in models]);memory_states=[];B=w['budget_mib']
 for ns in vectors:
  x={m:n for m,n in zip(models,ns) if n};mem=d['safe_factor']*math.fsum(d['memory_table_mib'][m][str(n)] for m,n in x.items())
  if mem>B+1e-6 or not all(any(t in d['models'][m]['supported_tasks'] for m in x) for t in rates):continue
  memory_states.append((x,mem))
 checks=[]
 for p in w['points']:
  tasks=sorted(p['rates']);possible=[]
  for x,mem in memory_states:
   ms=sorted(x);paths=[(t,m) for t in tasks for m in ms if t in d['models'][m]['supported_tasks']]
   eq=[[float(t==k) for t,m in paths] for k in tasks];ub=[[d['models'][m]['service_time_ms'][t]/1000 if m==k else 0 for t,m in paths] for k in ms];cap=[.8*d['capacity_factors'][m][str(x[m])] for m in ms]
   lp=linprog(np.zeros(len(paths)),A_eq=eq,b_eq=[p['rates'][k] for k in tasks],A_ub=ub,b_ub=cap,bounds=(0,None),method='highs')
   if lp.status==2:continue
   assert lp.success,lp.message
   possible.append({'deployment':x,'safe_memory_mib':mem})
  best=min(r['safe_memory_mib'] for r in possible);win=p['full_pool_result']['winner'];assert abs(best-win['safe_memory_mib'])<1e-6
  checks.append({'stage':p['stage'],'independent_LP_feasible_states':possible,'minimum_memory_mib':best,'MILP_minimum_matches_enumeration':True,'MILP_memory_dual_gap_mib':win['safe_memory_mib']-win['MILP_memory_lower_bound_mib']})
 competition=[]
 for m,p in d['models'].items():
  if p['role']=='generalist':
   if w['background_task'] not in p['supported_tasks'] or w['target_task'] not in p['supported_tasks']:continue
   for n in d['memory_table_mib'][m]:
    if int(n):competition.append({'candidate':'G '+m+' x'+n,'safe_memory_mib':d['safe_factor']*d['memory_table_mib'][m][n],'fits_budget':d['safe_factor']*d['memory_table_mib'][m][n]<=B,'note':'Background capable competitor, no forced exclusion.'})
  elif w['target_task'] in p['supported_tasks']:
   for n in d['memory_table_mib'][m]:
    if int(n):
     cost=w['M_G_safe_mib']+d['safe_factor']*d['memory_table_mib'][m][n];competition.append({'candidate':'G '+w['Generalist']+' x1 + S '+m+' x'+n,'safe_memory_mib':cost,'fits_budget':cost<=B,'note':'Measured state only; unknown replica counts excluded as unknown, not declared physically infeasible.'})
 save('cpu_independent_verification.json',{'at_utc':now(),'active_models':models,'memory_coverage_feasible_vectors':len(memory_states),'checks':checks,'full_pool_other_tasks_eliminated_by_strict_positive_cost_and_no_served_workload':True,'quality_and_data_sources':ref(OUT/'task_path_measurement_audit.json'),'capacity_order_verified':0<w['C_G_rps']<w['C_S_rps'],'unknown_counts_not_extrapolated':True});csvsave('measured_model_competition.csv',competition)
 print('CPU_INDEPENDENT_ENUMERATION_LP_PASS',len(memory_states),[(r['stage'],len(r['independent_LP_feasible_states'])) for r in checks])
if __name__=='__main__':main()
