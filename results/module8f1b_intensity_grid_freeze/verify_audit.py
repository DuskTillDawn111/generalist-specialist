"""Independent constraint and original canonical-LP verification; CPU only."""
import bisect,csv,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));OUT=Path(__file__).resolve().parent;R=ROOT/'results';TOL=1e-9

def read(n):return json.loads((OUT/n).read_text())
def source(n):return json.loads((R/n).read_text())
def close(a,b):return abs(a-b)<=TOL*max(1,abs(a),abs(b))
for r in read('artifact_hashes.json').values():assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
gate=read('resource_campaign_gate.json')
for r in gate['frozen_source_hashes'].values():assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
mixtures=read('base_mixture_freeze.json');base={m['id']:m['lambda_base_rps'] for m in mixtures['mixtures']};plan=source('module8b_quality_pool_finalization/planning_catalog.json');ids=source('module8f1a_memory_breakpoints/deployment_space_manifest.json')['model_order'];bounds=source('replica_capacity_calibration/evidence_replica_bounds.json')['replica_bounds'];safety=source('module8f1a_memory_breakpoints/memory_safety_factor.json');factor=safety['safety_factor'];rho=gate['rho_max'];p=plan['models'];M={m:{int(k):v for k,v in source('replica_capacity_calibration/replica_memory_table.json')['models'][m]['M'].items()} for m in ids};G={m:{int(k):v for k,v in source('replica_capacity_calibration/replica_capacity_factors.json')['models'][m]['g'].items()} for m in ids};roles={m['id']:m['role'] for m in source('candidate_pool_v2_1/candidate_pool_v2_1.json')['models'] if m['id'] in ids};tasks=list(next(iter(base.values())))
for m in mixtures['mixtures']:
 for old in m['legacy_moderate_high_provenance']:
  assert all(close(m['lambda_base_rps'][t]*old['reference_normalized_load_L'],old['lambda_rps'][t]) for t in tasks)
# Recompute every alpha limit through the original unchanged minimax canonical
# pressure relaxation, independent of the new direct max-alpha LP formulation.
from deployment_core.models import ModelProfile
from deployment_core.workload import Workload
from deployment_core.replica_calibration import ReplicaCalibration
from deployment_search.routing import solve_pressure_relaxation
catalog={m:ModelProfile(m,frozenset(v['supported_tasks']),v['memory_mib'],v['service_time_ms'],v['sources']) for m,v in p.items()};cal=ReplicaCalibration({m:G[m] for m in p},{m:M[m] for m in p},{m:bounds[m] for m in p}).validate(catalog)
rows=list(csv.DictReader((OUT/'deployment_capacity_limits.csv').open()));states={};limits={m:{} for m in base};maxrelative=0
for row in rows:
 x=json.loads(row['replica_vector']);i=int(row['state_id']);memory=factor*math.fsum(M[m][x[m]] for m in ids);assert memory==float(row['M_safe_mib']) and memory<=49140;states[i]={'x':x,'memory':memory};m=row['mixture'];a=float(row['alpha_limit']);route=solve_pressure_relaxation({model:x[model] for model in p},Workload(base[m]),catalog,rho,replica_bounds={model:bounds[model] for model in p},capacity_mode='empirical',calibration=cal);assert route is not None
 expected=rho/route.max_pressure;relative=abs(expected-a)/expected;maxrelative=max(maxrelative,relative);assert relative<=1e-8,(m,i,a,expected);limits[m][i]=a
assert len(rows)==5772 and len(states)==1924
print('Independent original canonical LP verified all 5772 intensity limits.',flush=True)
vector_index={tuple(s['x'][m] for m in ids):i for i,s in states.items()}
curve=list(csv.DictReader((OUT/'intensity_breakpoint_curve.csv').open()));bnd=read('alpha_capacity_boundaries.json');reference=bnd['alpha_max_reference_by_mixture'];budgetpop={float(B):[i for i,s in states.items() if s['memory']<=float(B)] for B in bnd['budgets']}
for row in curve:
 m=row['mixture'];B=float(row['budget_mib']);a=float(row['alpha_breakpoint']);pop=budgetpop[B];at=[i for i in pop if limits[m][i]>=a-TOL*max(1,a)];after=[i for i in pop if limits[m][i]>a+TOL*max(1,a)];assert len(at)==int(row['feasible_count_at_breakpoint']) and len(after)==int(row['feasible_count_immediately_after']);assert len(at)/len(pop)==float(row['feasible_fraction_of_zero_load']);assert float(row['beta_breakpoint'])==a/reference[m]
 for model in ('qwen','whisper-small','resnet50'):
  doubles=[i for i in at if states[i]['x'][model]==2];assert len(doubles)==int(row['x2_'+model+'_feasible_count']);useful=sum(limits[m][vector_index[tuple(states[i]['x'][mm]-(1 if mm==model else 0) for mm in ids)]]<a-TOL*max(1,a) for i in doubles);assert useful==int(row['x2_'+model+'_capacity_useful_pair_count'])
for m,bb in bnd['mixtures'].items():
 for B,v in bb.items():
  pop=budgetpop[float(B)];assert v['alpha_max']==max(limits[m][i] for i in pop);x=v['boundary_deployment'];i=vector_index[tuple(x[mm] for mm in ids)];assert close(limits[m][i],v['alpha_max']);r=v['boundary_routing_rps']
  for t in tasks:assert close(math.fsum(r.get(f'{t}::{mm}',0) for mm in p),limits[m][i]*base[m][t])
rep=read('replica_activation_thresholds.json')
for m,bb in rep['mixtures'].items():
 for B,models in bb.items():
  pop=budgetpop[float(B)]
  for model,result in models.items():
   double=[i for i in pop if states[i]['x'][model]==2];pairs=[]
   for i in double:
    j=vector_index[tuple(states[i]['x'][mm]-(1 if mm==model else 0) for mm in ids)];a1=limits[m][j];a2=limits[m][i]
    if a2>a1+TOL*max(1,a1,a2):pairs.append((a1,a2,i,j))
   assert bool(double)==result['x2_memory_feasible'] and len(pairs)==result['capacity_useful_pair_count']
   assert result['activation_infimum_alpha']==(min(v[0] for v in pairs) if pairs else None)
   if result['globally_necessary_after_alpha'] is not None:
    assert result['globally_necessary_after_alpha']==max(limits[m][i] for i in pop if states[i]['x'][model]<=1)
    assert result['globally_necessary_until_alpha']>result['globally_necessary_after_alpha']
final=read('final_memory_budgets.json');budgets=[v['budget_mib'] for v in final['budgets']];assert len(budgets)==6 and all(v['empirical_breakpoint'] for v in final['budgets'][:-1]);assert final['budgets'][-1]['physical_endpoint_exception'];assert all(b['legacy_8f1a_feasible_fraction']-a['legacy_8f1a_feasible_fraction']>=.05 for a,b in zip(final['budgets'],final['budgets'][1:]))
levels=read('frozen_intensity_levels.json');assert len(levels['levels'])==3 and all(a['beta']<b['beta'] for a,b in zip(levels['levels'],levels['levels'][1:]))
for level in levels['levels']:
 for m,values in level['mixtures'].items():
  assert values['alpha']==level['beta']*reference[m];assert all(values['lambda_rps'][t]==values['alpha']*base[m][t] for t in tasks)
  for b in values['budget_feasibility']:
   a=values['alpha'];eligible=[i for i in budgetpop[b['budget_mib']] if limits[m][i]>=a-TOL*max(1,a)];assert len(eligible)==b['feasible_count']
# Baseline selected values are checked against all eligible alternatives at
# each greedy step; empirical capacity never enters the scoring calculation.
def memory(x):return factor*math.fsum(M[m][x.get(m,0)] for m in ids)
def demand(paths,lam):return math.fsum(lam[t]*p[paths[t]]['service_time_ms'][t]/1000 for t in tasks)
def check_route(x,r,lam):
 for t in tasks:assert close(math.fsum(r.get(f'{t}::{m}',0) for m in p),lam[t])
 for k,v in r.items():
  t,m=k.split('::');assert v>=0
  if v>0:assert x[m]>0 and t in p[m]['supported_tasks']
 for m in p:
  d=math.fsum(r.get(f'{t}::{m}',0)*p[m]['service_time_ms'].get(t,0)/1000 for t in tasks);assert d<=rho*G[m][x.get(m,0)]+1e-9
campaign=read('campaign_cells.json');ready_counts={m:0 for m in ['static_v2','Generalist_centric','RC_SG']}
for cell in campaign['cells']:
 B=cell['memory_budget_mib'];lam=cell['lambda_rps'];m=cell['mixture'];alpha=cell['alpha'];available=[i for i in budgetpop[B] if limits[m][i]>=alpha-TOL*max(1,alpha)];assert len(available)==cell['capacity_feasible_space_count']
 for method,pre in cell['preflight'].items():
  if pre['status']=='READY':
   ready_counts[method]+=1;assert memory(pre['deployment'])<=B+1e-9;check_route(pre['deployment'],pre.get('canonical_routing_rps',pre.get('fixed_routing_rps')),lam)
 rc=cell['preflight']['RC_SG'];bases=[]
 for g in p:
  if roles[g]!='generalist':continue
  paths={};ok=True
  for t in tasks:
   if t in p[g]['supported_tasks']:paths[t]=g
   else:
    choices=[s for s in p if roles[s]=='specialist' and t in p[s]['supported_tasks']]
    if not choices:ok=False;break
    paths[t]=min(choices,key=lambda s:(p[s]['service_time_ms'][t],s))
  if not ok:continue
  x={mm:int(mm==g or mm in paths.values()) for mm in ids}
  if memory(x)<=B:bases.append((demand(paths,lam),memory(x),g,tuple(x[mm] for mm in ids),x,paths))
 if not bases:assert rc['status']=='BASE_MEMORY_OR_CAPABILITY_INFEASIBLE';continue
 best=min(bases,key=lambda row:row[:4]);x=dict(best[4]);paths=dict(best[5]);assert rc['trace'][0]['generalist']==best[2]
 for step in rc['trace'][1:]:
  candidates=[]
  for s in p:
   if roles[s]!='specialist' or x[s]:continue
   improved=[t for t in tasks if t in p[s]['supported_tasks'] and p[s]['service_time_ms'][t]<p[paths[t]]['service_time_ms'][t]];gain=math.fsum(lam[t]*(p[paths[t]]['service_time_ms'][t]-p[s]['service_time_ms'][t])/1000 for t in improved);xx=dict(x);xx[s]=1;delta=factor*M[s][1]
   if gain>0 and memory(xx)<=B:candidates.append((-gain/delta,delta,s,improved))
  winner=min(candidates,key=lambda row:row[:3]);assert step['model']==winner[2] and close(step['value_per_safe_mib'],-winner[0]);x[winner[2]]=1
  for t in winner[3]:paths[t]=winner[2]
 assert rc['deployment']==x and rc['assignment']==paths
 for s in p:
  if roles[s]!='specialist' or x[s]:continue
  xx=dict(x);xx[s]=1
  if memory(xx)<=B:assert not any(t in p[s]['supported_tasks'] and p[s]['service_time_ms'][t]<p[paths[t]]['service_time_ms'][t] for t in tasks)
 v2=cell['preflight']['static_v2']
 if v2['status']=='READY':
  bounds_effective=v2['effective_resource_bounds'];permitted=[i for i in available if all(states[i]['x'][mm]<=bounds_effective[mm] for mm in p)];assert close(v2['M_safe_mib'],min(states[i]['memory'] for i in permitted))
A=read('memory_sweep_design.json');B=read('intensity_sweep_design.json');estimate=read('gpu_budget_estimate.json');assert A['cells']==6 and B['cells']==9 and campaign['unique_cell_count']==12;assert estimate['structural_ready_method_cells']==ready_counts and estimate['estimated_new_runs_conservative']==22*ready_counts['static_v2']+5*(ready_counts['Generalist_centric']+ready_counts['RC_SG']);assert estimate['worst_case_timed_runs']==384
assert gate['RESOURCE_STATIC_GPU_CAMPAIGN_READY']==all(gate['checks'].values()) and gate['new_GPU_runs']==0
assert gate['HIGH_model_tradeoff_diagnostic'] and not gate['HIGH_replica_usefulness_diagnostic']
result={'status':'PASS','frozen_source_hashes_unchanged':True,'independent_original_canonical_LP_limits_verified':len(rows),'maximum_relative_difference_direct_vs_canonical_LP':maxrelative,'memory_safe_covered_vectors_per_mixture':len(states),'intensity_curve_rows_verified':len(curve),'replica_open_interval_and_global_necessity_verified':True,'mixture_scalar_invariance_verified':True,'memory_fraction_dedup_verified':True,'RC_SG_all_greedy_step_values_and_tie_breaks_verified':True,'all_ready_routes_and_safe_memory_verified':True,'cold_seed_minimum_memory_checked_against_enumeration':True,'cells':campaign['unique_cell_count'],'ready_method_cells':ready_counts,'estimated_new_runs':estimate['estimated_new_runs_conservative'],'RESOURCE_STATIC_GPU_CAMPAIGN_READY':gate['RESOURCE_STATIC_GPU_CAMPAIGN_READY'],'new_GPU_runs':0,'Proposed_search_calls':0}
(OUT/'validation_results.json').write_text(json.dumps(result,indent=2)+'\n')
art={}
for pth in sorted(OUT.iterdir()):
 if pth.is_file() and pth.name not in ('artifact_hashes.json','execution.log','verification.log'):
  b=pth.read_bytes();art[pth.name]={'path':str(pth),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
(OUT/'artifact_hashes.json').write_text(json.dumps(art,indent=2)+'\n')
print(json.dumps(result),flush=True)
