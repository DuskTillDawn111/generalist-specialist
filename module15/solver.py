"""Full-pool measured-memory MILP followed by unchanged frozen J-routing."""
import numpy as np,math
from scipy.optimize import Bounds,LinearConstraint
from module12.solver import precise_milp
from module14c.common import catalog
from deployment_core import Workload
from deployment_core.routing.e2e_router import solve_e2e_routing
from module11.latency import certificate
from .common import *
class ResourceSolver:
 def __init__(self,data):
  self.data=data;self.c,self.roles,self.cal=catalog(data)
 def primary(self,rates,budget,excluded=()):
  models=sorted(m for m,p in self.c.items() if any(k in p.supported_tasks and r>0 for k,r in rates.items()))
  tasks=sorted(k for k,r in rates.items() if r>0)
  selectors=[(m,int(n)) for m in models for n in self.data['capacity_factors'][m]]
  paths=[(k,m) for k in tasks for m in models if k in self.c[m].supported_tasks]
  ns=len(selectors);N=ns+len(paths);A=[];lower=[];upper=[]
  def add(v,lo,hi):A.append(v);lower.append(lo);upper.append(hi)
  for m in models:
   v=np.zeros(N);cap=np.zeros(N)
   for i,(p,n) in enumerate(selectors):
    if p==m:v[i]=1;cap[i]=-.8*self.cal.capacity_factors[m][n]
   for j,(k,p) in enumerate(paths):
    if p==m:cap[ns+j]=self.c[m].service_time_ms[k]/1000
   add(v,1,1);add(cap,-np.inf,0)
  for k in tasks:
   v=np.zeros(N)
   for j,(t,m) in enumerate(paths):
    if t==k:v[ns+j]=1
   add(v,rates[k],rates[k])
  cost=np.zeros(N)
  for i,(m,n) in enumerate(selectors):cost[i]=self.data['safe_factor']*self.cal.memory_table_mib[m][n]
  add(cost/budget,-np.inf,1)
  for x in excluded:
   v=np.zeros(N)
   for i,(m,n) in enumerate(selectors):
    if n==x.get(m,0):v[i]=1
   add(v,-np.inf,len(models)-1)
  fit=precise_milp(cost/budget,integrality=np.r_[np.ones(ns),np.zeros(len(paths))],bounds=Bounds(np.zeros(N),np.r_[np.ones(ns),np.full(len(paths),np.inf)]),constraints=LinearConstraint(np.array(A),lower,upper),options={'mip_rel_gap':0.,'time_limit':30.})
  if fit.status==2:return None
  if not fit.success:raise RuntimeError('Primary memory solver unresolved: '+fit.message)
  x={m:sum(n for i,(p,n) in enumerate(selectors) if p==m and fit.x[i]>.5) for m in models};x={m:n for m,n in x.items() if n}
  route={(k,m):float(fit.x[ns+j]) for j,(k,m) in enumerate(paths) if m in x}
  cert=certificate(x,route,self.c,self.cal);assert cert['passed']
  mem=self.data['safe_factor']*math.fsum(self.cal.memory_table_mib[m][n] for m,n in x.items())
  assert mem<=budget+1e-6 and abs(mem-fit.fun*budget)<1e-6
  for k in tasks:assert abs(math.fsum(v for (t,m),v in route.items() if t==k)-rates[k])<1e-7
  return {'deployment':x,'safe_memory_mib':mem,'MILP_memory_lower_bound_mib':float(fit.mip_dual_bound*budget),'MILP_relative_gap':float(fit.mip_gap),'MILP_status':fit.message,'certificate':cert}
 def route(self,state,rates):
  try:r=solve_e2e_routing(state['deployment'],Workload(rates),self.c,self.cal)
  except (ValueError,ArithmeticError,AssertionError) as exc:return {**state,'J_status':'NUMERICAL_SOLVER_EXCEPTION','J_accepted':False,'exception':repr(exc)}
  ce=certificate(state['deployment'],r['routing'],self.c,self.cal) if r['converged'] else None
  valid=bool(r['converged'] and ce and ce['passed'])
  result={**state,'J_status':r['status'],'J_accepted':valid,'J_gap_seconds':r.get('optimality_gap_seconds'),'J_seconds':r['evaluation']['J_seconds'],'rho':r['evaluation']['rho'],'max_rho':r['evaluation']['z'],'routing':{f'{k}::{m}':v for (k,m),v in (r.get('routing') or {}).items()},'J_operational_certificate':ce}
  return result
 def solve(self,rates,budget,competition=0):
  first=self.primary(rates,budget)
  if first is None:return {'feasible':False,'status':'CAPACITY_OR_MEMORY_INFEASIBLE','winner':None,'competitors':[]}
  memory=first['safe_memory_mib'];states=[];excluded=[];state=first;next_bound=None
  # Exhaust every exact minimum-memory deployment; original J determines winner.
  while state is not None and state['safe_memory_mib']<=memory+1e-7:
   states.append(self.route(state,rates));excluded.append(state['deployment']);state=self.primary(rates,budget,excluded)
  if state is not None:next_bound=state['safe_memory_mib']
  valid=[s for s in states if s['J_accepted']]
  unresolved=[s for s in states if not s['J_accepted']]
  if unresolved:return {'feasible':True,'status':'MINIMUM_MEMORY_FROZEN_ROUTER_UNRESOLVED','winner':None,'minimum_memory_mib':memory,'tied_minimum_states':states,'unresolved':unresolved,'competitors':[]}
  winner=min(valid,key=lambda s:(s['J_seconds'],tuple(sorted(s['deployment'].items()))))
  competitors=[]
  for i in range(competition):
   if state is None:break
   competitors.append(self.route(state,rates));excluded.append(state['deployment']);state=self.primary(rates,budget,excluded)
  return {'feasible':True,'status':'MINIMUM_MEASURED_MEMORY_THEN_FROZEN_J','winner':winner,'minimum_memory_mib':memory,'minimum_memory_lower_bound_mib':first['MILP_memory_lower_bound_mib'],'tied_minimum_states':states,'next_resource_cost_mib':next_bound,'competitors':competitors,'competition_all_exhausted':state is None,'full_pool_participates':True,'unrelated_task_models_eliminated_only_because_positive_cost_and_zero_demand':True,'Generalist_or_Specialist_identity_not_forced':True}
