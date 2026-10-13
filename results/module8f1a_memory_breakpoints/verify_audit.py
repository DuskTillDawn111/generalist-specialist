"""Independent file/constraint checks of the offline audit; no GPU imports/calls."""
import csv,hashlib,itertools,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));OUT=Path(__file__).resolve().parent
def read(n):return json.loads((OUT/n).read_text())
manifest=read('deployment_space_manifest.json');catalog=read('memory_catalog_frozen.json');errors=read('memory_error_audit.json');safety=read('memory_safety_factor.json');minimum=read('minimum_feasible_budgets.json');replica=read('replica_memory_breakpoints.json');candidate=read('candidate_memory_budgets.json');sg=read('specialist_greedy_thresholds.json');certs=read('capacity_feasibility_certificates.json')
ids=manifest['model_order'];bounds=manifest['replica_bounds'];physical=49140;factor=safety['safety_factor'];epsilon=max(max((r['actual_peak_mib']-r['planning_mib'])/r['planning_mib'],0) for r in errors['measurements'])
assert epsilon==safety['epsilon_mem']==errors['epsilon_mem'] and factor==1+epsilon
for r in errors['measurements']:
 p=Path(r['raw_source']['path']);b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==r['raw_source']['sha256'];d=json.loads(b);assert d['memory_observations']['peak_device_mib']==r['actual_peak_mib'];assert d['memory_observations']['planning_idle_mib']==r['planning_mib'];assert d['status']=='completed'
for r in manifest['frozen_input_sources'].values():assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
for r in read('artifact_hashes.json').values():assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
rows=list(csv.DictReader((OUT/'deployment_memory_states.csv').open()));curve=list(csv.DictReader((OUT/'memory_breakpoint_curve.csv').open()));vectors=list(itertools.product(*(range(bounds[m]+1) for m in ids)))
assert len(rows)==len(vectors)==manifest['actual_enumerated_state_count']==13824
witness={(r['workload'],r['state_id']):r for r in certs['witnesses']};planning=json.loads((ROOT/'results/module8b_quality_pool_finalization/planning_catalog.json').read_text());profiles=planning['models'];g=planning['capacity_factors'];rho=certs['rho_max'];rate={w['id']:w['lambda_rps'] for w in manifest['workloads']};false_capacity=[];F={w:set() for w in rate}
for i,(row,v) in enumerate(zip(rows,vectors)):
 assert int(row['state_id'])==i and tuple(int(row['x_'+m]) for m in ids)==v
 mp=math.fsum(catalog['models'][m]['values'][str(n)]['value'] for m,n in zip(ids,v));assert float(row['M_plan_mib'])==mp and float(row['M_safe_mib'])==factor*mp
 assert int(row['N_G'])==sum(n for m,n in zip(ids,v) if catalog['models'][m]['role']=='generalist') and int(row['total_replicas'])==sum(v)
 x=dict(zip(ids,v));disabled_ok=all(x[m]==0 for m in planning['quality_disabled_model_variables'])
 for w,lam in rate.items():
  coverage=disabled_ok and all(any(x[m]>0 and t in p['supported_tasks'] for m,p in profiles.items()) for t,a in lam.items() if a>0)
  if row['feasible_'+w]=='True':
   assert coverage;F[w].add(i);origin=int(row['capacity_certificate_'+w]);c=witness[w,origin];assert all(x[m]>=c['deployment'][m] for m in ids)
   routes={tuple(k.split('::')):v for k,v in c['routing_rps'].items()}
   for (t,m),r in routes.items():assert r>=0 and t in profiles[m]['supported_tasks'] and c['deployment'][m]>0
   for t,a in lam.items():
    if a>0:assert math.isclose(math.fsum(r for (tk,m),r in routes.items() if tk==t),a,rel_tol=1e-7,abs_tol=1e-12)
   for m,p in profiles.items():
    load=math.fsum(r*p['service_time_ms'][t]/1000 for (t,mk),r in routes.items() if mk==m)
    assert load<=rho*g[m][str(x[m])]+1e-9
  elif coverage:false_capacity.append((i,w))
intersection=set.intersection(*F.values());full={i for i in intersection if float(rows[i]['M_safe_mib'])<=physical};assert len(full)==1834
for w,f in F.items():assert min(float(rows[i]['M_safe_mib']) for i in f)==minimum['workloads'][w]['B_min_mib']
assert minimum['B_min_global_mib']==max(v['B_min_mib'] for v in minimum['workloads'].values())
assert {float(r['breakpoint_mib']) for r in curve}=={float(r['M_safe_mib']) for r in rows}
# Recount every physical growth curve point from independent histogram accumulation.
hist={}
for i in full:
 b=float(rows[i]['M_safe_mib']);hist[b]=hist.get(b,0)+1
running=0
for row in curve:
 b=float(row['breakpoint_mib']);running+=hist.get(b,0);assert int(row['N_feasible'])==running;assert float(row['fraction_feasible'])==running/len(full)
for c in candidate['candidates']:
 b=c['budget_mib'];feasible={i for i in full if float(rows[i]['M_safe_mib'])<=b};assert b<=physical and len(feasible)==c['N_feasible'];assert len(feasible)/len(full)==c['fraction_feasible'];assert c['SG_all_memory_feasible']==all(r['B_SG_mib']<=b for r in sg['workloads'])
 assert c['x2_models_feasible']==[m for m in ids if bounds[m]==2 and any(int(rows[i]['x_'+m])==2 for i in feasible)]
assert len({c['budget_mib'] for c in candidate['candidates']})==5
# Independently rerun zero-objective LP for every capability/quality-covered
# state declared capacity-infeasible, guarding the monotonic closure complement.
from deployment_core.models import ModelProfile
from deployment_core.deployment import Deployment
from deployment_core.workload import Workload
from deployment_core.replica_calibration import ReplicaCalibration
from deployment_core.routing_solver import solve_capacity_lp
lp_profiles={m:ModelProfile(m,frozenset(p['supported_tasks']),p['memory_mib'],p['service_time_ms'],p['sources']) for m,p in profiles.items()}
cal=ReplicaCalibration({m:{int(k):v for k,v in g[m].items()} for m in profiles},{m:{int(k):v for k,v in planning['memory_table_mib'][m].items()} for m in profiles},{m:bounds[m] for m in profiles}).validate(lp_profiles)
for i,w in false_capacity:
 x={m:int(rows[i]['x_'+m]) for m in profiles}
 assert solve_capacity_lp(Deployment(x),lp_profiles,Workload(rate[w]),rho,capacity_mode='empirical',calibration=cal) is None,(i,w)
result={'status':'PASS','all_state_vectors_verified':len(rows),'historical_raw_hashes_verified':len(errors['measurements']),'every_feasible_capacity_certificate_verified':True,'capacity_infeasible_covered_states_rechecked':len(false_capacity),'all_breakpoint_curve_counts_verified':len(curve),'frozen_inputs_unchanged':True,'candidate_memory_and_SG_and_x2_semantics_verified':True,'physical_endpoint_exception_verified':not candidate['candidates'][-1]['empirical_breakpoint'],'new_GPU_runs':0,'deployment_optimizer_calls':0}
(OUT/'validation_results.json').write_text(json.dumps(result,indent=2)+'\n')
# Include the verifier and validation result in a fresh hash inventory; logs vary.
art={}
for p in sorted(OUT.iterdir()):
 if p.is_file() and p.name not in ('artifact_hashes.json','execution.log'):
  b=p.read_bytes();art[p.name]={'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
(OUT/'artifact_hashes.json').write_text(json.dumps(art,indent=2)+'\n')
print(json.dumps(result))
