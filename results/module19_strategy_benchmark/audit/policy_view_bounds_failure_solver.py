"""Independent policies under one measured catalog; original V3 is untouched."""
import time,math
from dataclasses import replace
from module16.solver import CertifiedResourceSolver
from module15.solver import ResourceSolver
from deployment_core import Workload,Hardware
from deployment_core.routing.e2e_surrogate import evaluate_surrogate
from module11.latency import certificate
from method_v2 import frozen_config
from module9bc.frozen_v3 import create,snapshot

class PolicySolver(CertifiedResourceSolver):
 def __init__(self,data,target,role=None):
  super().__init__(data);self.original_c=self.c.copy();self.policy_zero_paths=[];self.policy_only_zero_route_models=[]
  if role:
   c={}
   for m,p in self.c.items():
    allowed=p.supported_tasks-{target} if self.roles[m]!=role else p.supported_tasks
    if target in p.supported_tasks and self.roles[m]!=role:self.policy_zero_paths.append((target,m))
    if allowed:c[m]=replace(p,supported_tasks=frozenset(allowed),service_time_ms={k:p.service_time_ms[k] for k in allowed})
    else:self.policy_only_zero_route_models.append(m)
   self.c=c
  self.target=target;self.target_role=role

 def solve(self,rates,budget,competition=0):
  t=time.perf_counter();result=super().solve(rates,budget,competition);result['solve_seconds']=time.perf_counter()-t;result['allowed_pool_members_all']=list(self.data['models']);result['target_route_role_constraint']=self.target_role;result['policy_zero_paths']=[f'{k}::{m}' for k,m in self.policy_zero_paths];result['policy_zero_demand_models']=self.policy_only_zero_route_models;result['model_admission_catalog_unchanged']=True
  if result.get('winner'):
   w=result['winner'];routing={tuple(p.split('::')):v for p,v in w['routing'].items()};check=certificate(w['deployment'],routing,self.original_c,self.cal);assert check['passed'];assert all(k!=self.target or self.target_role is None or self.roles[m]==self.target_role for (k,m),v in routing.items() if v>1e-10)
  return result

def original_v3(data,rates,budget):
 s=ResourceSolver(data);c,roles,cal=s.c,s.roles,s.cal;captured=[];engine=create(None,c,cal,trace_callback=lambda r:captured.append(r.to_dict()));start=time.perf_counter();search_seconds=None
 try:
  result=engine.search(Workload(rates),c,Hardware(budget/data['safe_factor']),replace(frozen_config(),dry_run=True),roles=roles,capacity_mode='empirical',memory_mode='empirical',calibration=cal);search_seconds=time.perf_counter()-start
  audit=time.perf_counter();v=snapshot(engine,result,Workload(rates));fit=engine.serving_routing_adapter.ranker.evaluate(result.deployment,Workload(rates)) if result.deployment and result.canonical else None
  cert=certificate(result.deployment,fit['routing'],c,cal) if fit and fit['converged'] else None
  x={m:n for m,n in (result.deployment or {}).items() if n};mem=data['safe_factor']*math.fsum(cal.memory_table_mib[m][n] for m,n in x.items()) if x else None
  valid=bool(fit and fit['converged'] and cert and cert['passed'] and mem<=budget+1e-6)
  winner={'deployment':x,'safe_memory_mib':mem,'routing':{f'{k}::{m}':q for (k,m),q in fit['routing'].items()},'J_seconds':fit['evaluation']['J_seconds'],'rho':fit['evaluation']['rho'],'max_rho':fit['evaluation']['z'],'J_operational_certificate':cert,'J_gap_seconds':fit.get('optimality_gap_seconds'),'J_accepted':True} if valid else None
  return {'status':'VALID_ORIGINAL_V3' if valid else 'ORIGINAL_V3_UNRESOLVED_NOT_THEORETICAL_INFEASIBILITY','winner':winner,'solve_seconds':search_seconds,'audit_seconds':time.perf_counter()-audit,'end_to_end_seconds':time.perf_counter()-start,'original_snapshot':v,'original_final_fit':{**{k:q for k,q in (fit or {}).items() if k!='routing'},'routing':{f'{k}::{m}':q for (k,m),q in (fit or {}).get('routing',{}).items()}} if fit else None,'original_operational_certificate':cert,'original_V3_implementation_modified':False,'numerical_polish_applied_to_V3':False}
 except (RuntimeError,ValueError,AssertionError) as exc:
  return {'status':'ORIGINAL_V3_NUMERICAL_OR_IMPLEMENTATION_FAILURE','winner':None,'solve_seconds':time.perf_counter()-start,'exception':repr(exc),'partial_search_trace':captured,'original_V3_implementation_modified':False,'numerical_polish_applied_to_V3':False}

def fixed_S_only(data,case,proposed):
 s=ResourceSolver(data);x=proposed['deployment'];task=case['target_task'];rates=case['rates'];Ss=[m for m in x if s.roles[m]=='specialist' and task in s.c[m].supported_tasks];assert len(Ss)==1,'Current frozen HIGH has one target S family; no arbitrary multi-S routing allocation.'
 route={tuple(p.split('::')):v for p,v in proposed['routing'].items() if p.split('::')[0]!=task};route[task,Ss[0]]=rates[task]
 cert=certificate(x,route,s.c,s.cal);ev=evaluate_surrogate(x,route,Workload(rates),s.c,s.cal)
 return {'status':'LEGAL_FIXED_DEPLOYMENT_ROUTING' if cert['passed'] else 'THEORETICALLY_INFEASIBLE_OPERATIONAL_CAPACITY','winner':{'deployment':x,'safe_memory_mib':proposed['safe_memory_mib'],'routing':{f'{k}::{m}':v for (k,m),v in route.items()},'J_seconds':ev['J_seconds'],'rho':ev['rho'],'max_rho':ev['z'],'J_operational_certificate':cert,'J_accepted':False,'J_gap_seconds':None},'non_target_routing_unchanged':True,'same_deployment_and_budget':True,'operationally_legal':cert['passed'],'is_legal_baseline':cert['passed'],'diagnostic_only_when_infeasible':not cert['passed'],'solve_seconds':0.}
