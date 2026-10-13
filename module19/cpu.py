"""All strategies independently solve identical module-specific frozen cases."""
from .common import *
from .solver import PolicySolver,original_v3,fixed_S_only
from module11.latency import certificate
from module14c.common import catalog

def audit(case,data,fit,strategy):
 w=fit.get('winner')
 if not w:return {'passed':False,'reason':fit['status']}
 c,roles,cal=catalog(data);x=w['deployment'];route={tuple(p.split('::')):v for p,v in w['routing'].items()};rates=case['rates'];ce=certificate(x,route,c,cal)
 checks={'quality_and_original_admission':all(m in c and k in c[m].supported_tasks for (k,m),v in route.items() if v>0),'all_task_coverage':all(any(k==t and v>0 for (t,m),v in route.items()) for k in rates),'route_conservation':all(abs(math.fsum(v for (t,m),v in route.items() if t==k)-q)<1e-8 for k,q in rates.items()),'empirical_replica_counts_only':all(str(n) in data['capacity_factors'][m] and str(n) in data['memory_table_mib'][m] for m,n in x.items()),'correct_14F_safe_memory':abs(w['safe_memory_mib']-data['safe_factor']*math.fsum(data['memory_table_mib'][m][str(n)] for m,n in x.items()))<1e-6,'budget':w['safe_memory_mib']<=case['budget_mib']+1e-6,'operational_rho':ce['passed']}
 targetrole='generalist' if strategy=='BASELINE_A_G_ONLY' else 'specialist' if strategy=='BASELINE_B_S_ORIENTED' else None
 if targetrole:checks['baseline_target_role_constraint']=all(k!=case['target_task'] or roles[m]==targetrole or v==0 for (k,m),v in route.items())
 return {'passed':all(checks.values()),'checks':checks,'original_operational_certificate':ce,'reason':','.join(k for k,v in checks.items() if not v) or None}

def main():
 f=read(OUT/'input_protocol_freeze.json');verify(f['implementation']);results=[];rows=[];ablations=[]
 for case in f['cases']:
  data=read(case['catalog']['path']);record={'case':case,'strategies':{}}
  # Same independent four policies for every case; no historical result is the solve.
  for strategy,role in [('BASELINE_A_G_ONLY','generalist'),('BASELINE_B_S_ORIENTED','specialist'),('BASELINE_C_ORIGINAL_V3','V3'),('PROPOSED',None)]:
   fit=original_v3(data,case['rates'],case['budget_mib']) if role=='V3' else PolicySolver(data,case['target_task'],role).solve(case['rates'],case['budget_mib']);checks=audit(case,data,fit,strategy);w=fit.get('winner');record['strategies'][strategy]={'fit':fit,'audit':checks}
   status='FEASIBLE' if checks['passed'] else 'INFEASIBLE' if fit['status']=='CAPACITY_OR_MEMORY_INFEASIBLE' else 'SOLVER_UNRESOLVED' if not w else 'CONSTRAINT_REJECTED'
   rows.append({'case_id':case['case_id'],'source_module':case['source_module'],'scope':case['scope'],'target_task':case['target_task'],'stage':case['stage'],'strategy':strategy,'status':status,'solver_status':fit['status'],'pool_catalog_sha256':case['catalog']['sha256'],'pool_model_count':len(data['models']),'full_workload':case['rates'],'budget_mib':case['budget_mib'],'deployment':w['deployment'] if w else None,'safe_memory_mib':w['safe_memory_mib'] if w else None,'routing':w['routing'] if w else None,'rho':w['rho'] if w else None,'max_rho':w['max_rho'] if w else None,'predicted_J_seconds':w['J_seconds'] if w else None,'solve_seconds':fit['solve_seconds'],'audit_seconds':fit.get('audit_seconds'),'audit':checks,'actual_model_pool_never_changed':True});print('CPU',case['case_id'],strategy,status,w['deployment'] if w else None,flush=True)
  prop=record['strategies']['PROPOSED'];assert prop['audit']['passed'];p=prop['fit']['winner'];old=case['source_slot'];assert p['deployment']==old['deployment'];assert abs(p['safe_memory_mib']-old['CPU_safe_memory_mib'])<1e-6
  rateold={f'{k}::{m}':v for k,r in old['routing'].items() for m,v in r.items()};assert max(abs(p['routing'].get(k,0)-rateold.get(k,0)) for k in set(p['routing'])|set(rateold))<1e-7,'Historical Proposed route not reproduced; no GPU'
  record['historical_Proposed_reproduced']=True
  if case['stage']=='HIGH':
   ab=fixed_S_only(data,case,p);ab['audit']=audit(case,data,ab,'FIXED_RESIDENCY_S_ONLY');record['routing_ablation_S_only']=ab;ablations.append({'case_id':case['case_id'],'status':ab['status'],'deployment':ab['winner']['deployment'],'routing':ab['winner']['routing'],'rho':ab['winner']['rho'],'predicted_J_seconds':ab['winner']['J_seconds'],'non_target_routing_unchanged':True,'operationally_legal':ab['operationally_legal'],'diagnostic_only':ab['diagnostic_only_when_infeasible'],'audit':ab['audit']});print('ABLATION',case['case_id'],ab['status'],ab['winner']['max_rho'],flush=True)
  results.append(record);save('cpu/'+case['case_id']+'.json',record)
 csvsave('cpu_baseline_comparison.csv',rows);save('cpu_results.json',{'at_utc':now(),'results':results,'fairness':'Original per-module catalog/workload/budget/quality/admission/g/M all equal across policies; target-role restrictions only baseline policy. No mixed Module15/18 comparison.','original_V3_unchanged':True,'all_Proposed_reproduced':True});save('cpu_routing_ablation.json',ablations);print('CPU_COMPLETE',len(rows),'strategies',len(ablations),'HIGH ablations',flush=True)
if __name__=='__main__':main()
