"""Independent resource policy, same convex J, exact active-face Newton polish.
Original Module9/V3 remain untouched and their rejected numerical fits are kept.
No tolerance relaxation, alternate objective, model restriction or forced route.
"""
import numpy as np
from scipy.optimize import linprog
from module15.solver import ResourceSolver
from module11b.convex_audit import reference_route,TOLERANCE_SECONDS
from deployment_core.routing.e2e_surrogate import evaluate_surrogate
from deployment_core import Workload
from module11.latency import certificate
class CertifiedResourceSolver(ResourceSolver):
 def route(self,state,rates):
  original=super().route(state,rates)
  if original['J_accepted']:return {**original,'solver':'UNCHANGED_MODULE9','original_Module9_fit':original}
  x=state['deployment'];c=self.c;cal=self.cal
  r=reference_route(x,rates,c,cal)
  if r is None:return original
  models=sorted(x);tasks=sorted(rates);pairs=[(k,m) for k in tasks for m in models if k in c[m].supported_tasks]
  eq=np.array([[float(k==t) for k,m in pairs] for t in tasks]);b=np.ones(len(tasks))
  service=np.array([[rates[k]*c[m].service_time_ms[k]/1000 if model==m else 0. for k,m in pairs] for model in models])
  g=np.array([cal.capacity_factors[m][x[m]] for m in models]);total=sum(rates.values())
  u=np.array([r['routing'][k,m]/rates[k] for k,m in pairs]);u[u<1e-7]=0.;u[u>1-1e-7]=1.
  free=np.flatnonzero((u>0)&(u<1));E=eq[:,free];active=np.flatnonzero(np.any(E!=0,axis=1));E=E[active]
  if len(free):u[free]+=np.linalg.lstsq(E,b[active]-eq[active]@u,rcond=None)[0]
  def fun(v):
   d=service@v;return float(1000*np.sum(d/(1-d/g))/total)
  def grad(v):return 1000*service.T@(1/(1-service@v/g)**2)/total
  iterations=0
  for iterations in range(30):
   if not len(free):break
   d=service@u;H=1000*service.T@np.diag(2/g/(1-d/g)**3)@service/total
   K=np.block([[H[np.ix_(free,free)],E.T],[E,np.zeros((len(active),len(active)))]])
   delta=np.linalg.lstsq(K,np.r_[-grad(u)[free],np.zeros(len(active))],rcond=None)[0][:len(free)]
   if np.max(np.abs(delta))<1e-14:break
   step=1.
   while step>1e-10:
    v=u.copy();v[free]+=step*delta
    for row in active:
     idx=free[E[list(active).index(row)]!=0];v[idx[-1]]=1.-sum(v[j] for j in np.flatnonzero(eq[row]) if j!=idx[-1])
    if np.all(v>=0) and np.all(v<=1) and np.all(service@v<=.8*g) and fun(v)<=fun(u)+1e-11:break
    step*=.5
   if step<=1e-10:break
   u=v
  lp=linprog(grad(u),A_eq=eq,b_eq=b,A_ub=service,b_ub=.8*g,bounds=[(0,1)]*len(pairs),method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
  gap=max(0.,float(grad(u)@(u-lp.x))/1000) if lp.success else None
  routing={(k,m):float(v*rates[k]) for (k,m),v in zip(pairs,u)};ce=certificate(x,routing,c,cal);ev=evaluate_surrogate(x,routing,Workload(rates),c,cal)
  accepted=bool(ce['passed'] and np.max(np.abs(eq@u-b))<=1e-12 and gap is not None and gap<=TOLERANCE_SECONDS and ev['domain_valid'])
  return {**state,'J_status':'STRICT_CONVEX_J_CERTIFIED' if accepted else 'NUMERICAL_UNRESOLVED','J_accepted':accepted,'J_gap_seconds':gap,'J_seconds':ev['J_seconds'],'rho':ev['rho'],'max_rho':ev['z'],'routing':{f'{k}::{m}':v for (k,m),v in routing.items()} if accepted else {},'J_operational_certificate':ce,'solver':'INDEPENDENT_SAME_J_ACTIVE_FACE_NEWTON_WITH_GLOBAL_LINEAR_CERTIFICATE','polish_iterations':iterations,'J_tolerance_seconds':TOLERANCE_SECONDS,'original_Module9_fit':original,'original_Module9_modified':False,'reference_solver_initial_status':{'success':r['solver_success'],'gap_seconds':r['gap_seconds'],'message':r['solver_message']}}
