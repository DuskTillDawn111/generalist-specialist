"""Independent enumeration checks for the CPU convex reference; no GPU."""
import itertools,time
from .common import *
from .convex_audit import solve_discrete_J,reference_route
from module11.common import rates,safe_memory

def main():
 c,hw,roles,cal=inputs();old=read(OLD/'input_freeze.json');bg=old['background'];heads=read(OLD/'normalized_directions.json')['heads_by_budget'];space=read(OLD/'deployment_space.json');checks=[]
 cases=[('B1',bg,c,roles,cal,space['B1'],1),('B1',rates(bg,{k:heads['B1'][k] for k in ['asr','lid']},.4),c,roles,cal,space['B1'],1)]
 for bid,lam,cat,rr,cc,states,maxG in cases:
  B=old['contexts'][int(bid[1:])-1]['memory_budget_mib'];best=None;n=0;begin=time.perf_counter()
  for state in states:
   if state['N_G']>maxG:continue
   f=reference_route(state['deployment'],lam,cat,cc)
   if f and f['converged']:
    n+=1
    if best is None or f['evaluation']['J_seconds']<best:best=f['evaluation']['J_seconds']
  outer=solve_discrete_J(lam,cat,rr,cc,B,maxG)
  assert best is not None and abs(best-outer['J_seconds'])<=1e-8,(best,outer)
  checks.append({'budget':bid,'lambda':lam,'independent_enumeration_feasible_optima':n,'enumeration_J_seconds':best,
    'tangent_MILP_J_seconds':outer['J_seconds'],'global_gap_seconds':outer['global_gap_seconds'],'passed':True,'seconds':time.perf_counter()-begin})
  print(json.dumps(checks[-1]),flush=True)
 save('CPU_reference_validation.json',{'passed':True,'checks':checks,'GPU_runs':0,'original_count_space':True,
  'scope':'independent enumerate-all B1 measured states and fixed-state convex solves vs tangent discrete MILP at baseline/pressure; unchanged J/rho<=.8'})
if __name__=='__main__':main()
