"""Execute the frozen RC-SG rule, with an unchanged canonical capacity gate."""
import math
from deployment_core import Workload,Hardware,MilpDeploymentSeeder
from deployment_core.replica_calibration import ReplicaPlanningPolicy
from deployment_search.routing import solve_canonical_routing
from deployment_search.result import jsonable
from .common import read,A,F,OUT,write,inputs,utc
TASKS=['text','image','ocr','asr','visual_qa','lid']
def rc_sg(lam,B,catalog,roles,cal,factor,ids):
 def safe(x):return factor*math.fsum(cal.memory_table_mib[m][x.get(m,0)] for m in catalog)
 bases=[]
 for generalist in sorted(m for m in catalog if roles[m]=='generalist'):
  paths={};valid=True
  for t in TASKS:
   if lam[t]<=0:continue
   if t in catalog[generalist].supported_tasks:paths[t]=generalist
   else:
    choices=[m for m in catalog if roles[m]=='specialist' and t in catalog[m].supported_tasks]
    if not choices:valid=False;break
    paths[t]=min(choices,key=lambda m:(catalog[m].service_time_ms[t],m))
  if not valid:continue
  x={m:int(m==generalist or m in paths.values()) for m in ids};memory=safe(x)
  if memory>B:continue
  demand=math.fsum(lam[t]*catalog[paths[t]].service_time_ms[t]/1000 for t in TASKS if lam[t]>0)
  bases.append((demand,memory,generalist,tuple(x[m] for m in ids),x,paths))
 if not bases:return {'status':'BASE_MEMORY_OR_CAPABILITY_INFEASIBLE','deployment':None,'trace':[]}
 chosen=min(bases,key=lambda r:r[:4]);x=dict(chosen[4]);paths=dict(chosen[5]);trace=[{'operation':'BASE','generalist':chosen[2],'M_safe_mib':chosen[1],'assignment':dict(paths),'weighted_worker_demand':chosen[0]}]
 while True:
  choices=[]
  for s in sorted(m for m in catalog if roles[m]=='specialist' and not x[m]):
   improved=[t for t in TASKS if lam[t]>0 and t in catalog[s].supported_tasks and catalog[s].service_time_ms[t]<catalog[paths[t]].service_time_ms[t]]
   gain=math.fsum(lam[t]*(catalog[paths[t]].service_time_ms[t]-catalog[s].service_time_ms[t])/1000 for t in improved);delta=factor*cal.memory_table_mib[s][1];proposal=dict(x);proposal[s]=1
   if gain>0 and safe(proposal)<=B:choices.append((-gain/delta,delta,s,improved,proposal,gain))
  if not choices:break
  row=min(choices,key=lambda r:r[:3]);x=row[4]
  for t in row[3]:paths[t]=row[2]
  trace.append({'operation':'ADD','model':row[2],'value_per_safe_mib':-row[0],'delta_safe_mib':row[1],'gain_seconds_per_second':row[5],'improved_tasks':row[3],'M_safe_mib':safe(x),'assignment':dict(paths)})
 canonical=solve_canonical_routing({m:x[m] for m in catalog},Workload(lam),catalog,.8,replica_bounds=cal.evidence_bounds,capacity_mode='empirical',calibration=cal)
 return {'status':'READY' if canonical else 'PLANNING_CAPACITY_INFEASIBLE','deployment':x,'assignment':paths,'M_safe_mib':safe(x),'trace':trace,'canonical_routing_rps':jsonable(canonical.routing) if canonical else None,'canonical_max_pressure':canonical.max_pressure if canonical else None}
def preflight():
 catalog,_,roles,cal=inputs();factor=read(A/'memory_safety_factor.json')['safety_factor'];ids=read(A/'deployment_space_manifest.json')['model_order'];rows=[]
 for cell in read(F/'campaign_cells.json')['cells']:
  rc=rc_sg(cell['lambda_rps'],cell['memory_budget_mib'],catalog,roles,cal,factor,ids);frozen=cell['preflight']['RC_SG']
  assert rc==frozen,'Frozen RC-SG construction changed: '+cell['cell_id']
  policy=ReplicaPlanningPolicy(catalog,'empirical','empirical',cal);h=Hardware(cell['planning_budget_input_mib'],'Frozen resource input');bounds,_=policy.bounds(h)
  seed=MilpDeploymentSeeder().solve(catalog,Workload(cell['lambda_rps']),h,.8,replica_bounds=bounds,capacity_mode='empirical',memory_mode='empirical',calibration=cal)
  x={m:cell['preflight']['static_v2']['deployment'][m] for m in catalog};assert seed.feasible and seed.deployment==x,cell['cell_id']
  route=solve_canonical_routing(x,Workload(cell['lambda_rps']),catalog,.8,replica_bounds=bounds,capacity_mode='empirical',calibration=cal)
  assert jsonable(route.routing)==cell['preflight']['static_v2']['canonical_routing_rps']
  rows.append({'cell_id':cell['cell_id'],'seed_matches_exactly':True,'seed_route_matches_exactly':True,'RC_SG_matches_exactly':True,'RC_SG_construction':rc})
 write('planning_execution_verification.json',{'status':'PASS','verified_at_utc':utc(),'before_any_GPU':True,'cells':rows,'enumeration_calls':0});return rows
if __name__=='__main__':preflight()
