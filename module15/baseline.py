"""Original frozen V3 CPU dry-run; never label a reference optimizer as V3."""
from dataclasses import replace
from deployment_core import Workload,Hardware
from method_v2 import frozen_config
from module9bc.frozen_v3 import create,snapshot
from module11.latency import certificate
from .common import *
def run(rates,budget,resource_solver):
 c,roles,cal=resource_solver.c,resource_solver.roles,resource_solver.cal;w=Workload(rates);captured=[]
 engine=create(None,c,cal,trace_callback=lambda r:captured.append(r.to_dict()))
 try:
  result=engine.search(w,c,Hardware(budget/resource_solver.data['safe_factor']),replace(frozen_config(),dry_run=True),roles=roles,capacity_mode='empirical',memory_mode='empirical',calibration=cal)
  data=snapshot(engine,result,w);data['baseline_scope']='ORIGINAL_V3_CPU_DRY_RUN_AND_NEIGHBORHOOD_RANKING_NOT_REAL_GPU_SEARCH_OR_GLOBAL_J_OPTIMUM'
  if result.deployment and result.canonical:
   fit=engine.serving_routing_adapter.ranker.evaluate(result.deployment,w)
   data['frozen_serving_J_fit']={**{k:v for k,v in fit.items() if k!='routing'},'routing':{f'{k}::{m}':v for (k,m),v in (fit.get('routing') or {}).items()}}
   data['operational_rho_certificate']=certificate(result.deployment,fit['routing'],c,cal) if fit['converged'] else None
 except (RuntimeError,ValueError,AssertionError) as exc:
  data={'status':'ORIGINAL_V3_CPU_FAILURE','exception':repr(exc),'partial_search_trace':captured,'not_hidden_or_repaired':True}
 data['frozen_V3_implementation_modified']=False;data['GPU_baseline_runs']=0;data['safe_budget_mib']=budget;data['raw_memory_budget_input_mib']=budget/resource_solver.data['safe_factor'];return data
