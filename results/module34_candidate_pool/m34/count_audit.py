from .common import *
from .cpu_core import candidate_routes

def main():
 sp=read(OUT/'candidate_space_definition.json');oo=read(OUT/'cpu_oracles.json');rows=[]
 for pool,pr in sp['pools'].items():
  d=read(pr['path'])
  for scenario,L in sp['scenarios'].items():
   oracle=next(o for o in oo if o['pool']==pool and o['scenario']==scenario)
   for alg in ['V3_J_GRID_CONTROL','V3_NATIVE_CONTINUOUS_PRESERVED']:
    fit=read(OUT/f'cpu/{scenario}_{pool}_{alg}.json');seen={identity({m:n for m,n in c['deployment'].items() if n},c['routing']) for c in fit['backend_calls']};z=fit.get('result');unique={}
    def add(x):
     active={m:n for m,n in x.items() if n};unique[json.dumps(active,sort_keys=True)]=active
    if z:
     add(z['seed']['deployment']);add(z['deployment'])
     for it in z['iterations']:
      if 'J_current_seconds' in it:add(it['deployment'])
      for c in it['candidates']:
       if 'J_star_seconds' in c:add(c['mutation']['deployment'])
     for c in z['actual_serving_route_evaluations']:add(c['deployment'])
    grid=alg=='V3_J_GRID_CONTROL';Jcalls=None
    if grid:
     assert fit['J_route_solves']==2*len(unique),('reconstruction mismatch',pool,scenario)
     Jcalls=sum(sum(1 for _ in candidate_routes(d,x,L['rates'],1)) for x in unique.values())
    rows.append(dict(pool=pool,scenario=scenario,algorithm=alg,status=fit['status'],search_U_prediction_component_calls=fit.get('objective_calls'),final_validation_U_prediction_component_calls=4 if fit.get('decision') else 0,
       unique_runtime_route_plans=len(seen),unique_runtime_route_fraction=len(seen)/oracle['feasible_routes'] if grid else None,controlled_exact_J_surrogate_calls=Jcalls,
       J_route_cache_count_legacy=fit.get('J_route_solves'),controlled_unique_J_route_states=len(unique) if grid else None,native_inner_J_surrogate_calls='NOT_INSTRUMENTED_PRESERVED',canonical_LP_solves=z.get('lp_solve_count') if z else None,seed_MILP_solves=z.get('milp_solve_count') if z else None,
       audit='Legacy grid J_route_solves adds the same shared cache twice; actual unique states are half. Exact Jcalls independently reconstructed from logged cached deployment set. Original outputs preserved; Ucalls count search component forecasts, not final validation.'))
   fit=read(OUT/f'cpu/{scenario}_{pool}_20A_U_GRID_1.json');legal=[c for c in fit['candidates'] if c['prediction']['eligible']];identities={identity(c['deployment'],c['routing']) for c in legal};rows.append(dict(pool=pool,scenario=scenario,algorithm='20A_U_GRID',search_U_prediction_component_calls=fit['objective_calls'],final_validation_U_prediction_component_calls=4,unique_legal_route_plans=len(identities),unique_legal_route_fraction=len(identities)/oracle['feasible_routes'],MILP_generation_calls=fit['MILP_calls'],audit='Distinct normalized complete deployment+route keys; all generated candidates use4 frozen prediction scenarios.'))
 csvsave('optimizer_evaluation_count_audit.csv',rows);save('optimizer_evaluation_count_audit.json',rows);print('COUNT_AUDIT',len(rows))
if __name__=='__main__':main()
