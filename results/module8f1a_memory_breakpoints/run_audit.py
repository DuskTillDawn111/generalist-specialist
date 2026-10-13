"""Module 8F-1A: file-only empirical audit. No serving imports or GPU calls."""
import csv, hashlib, itertools, json, math, statistics, subprocess, bisect
from collections import Counter, defaultdict
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent
PHYSICAL=49140
sources={}
def ref(p,field=None):
 p=Path(p);p=p if p.is_absolute() else ROOT/p
 b=p.read_bytes();r={'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
 sources[str(p)]=r
 return dict(r,**({'field':field} if field else {}))
def read(p):
 r=ref(p);return json.loads(Path(r['path']).read_text())
def write(n,v): (OUT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvwrite(n,rows):
 with (OUT/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def quant(v,p):
 v=sorted(v);i=(len(v)-1)*p;lo=int(i);return v[lo]+(v[min(lo+1,len(v)-1)]-v[lo])*(i-lo)
def stats(v):return {'n':len(v),'min':min(v),'mean':statistics.mean(v),'median':statistics.median(v),'p95':quant(v,.95),'max':max(v)}
POOL='results/candidate_pool_v2_1/candidate_pool_v2_1.json'
MEM='results/replica_capacity_calibration/replica_memory_table.json'
BOUNDS='results/replica_capacity_calibration/evidence_replica_bounds.json'
CAP='results/replica_capacity_calibration/replica_capacity_factors.json'
PLAN='results/module8b_quality_pool_finalization/planning_catalog.json'
WORK='results/module8b_quality_pool_finalization/final_benchmark_workload_matrix.json'
HELD='results/module8e2_heldout_freeze/heldout_workloads.json'
ADM='results/module8b_quality_pool_finalization/quality_admission_matrix.json'
SGDEF='results/module8d_baseline_freeze/baseline_definitions.json'
SGSTATES='results/module8d_baseline_freeze/baseline_deployments.json'
pool=read(POOL);mem=read(MEM);bounds=read(BOUNDS)['replica_bounds'];cap=read(CAP);plan=read(PLAN)
work=read(WORK);held=read(HELD);admission=read(ADM);sgdef=read(SGDEF);sgstates=read(SGSTATES)
for r in sgdef['definition_inputs'].values():
 assert ref(r['path'])['sha256']==r['sha256'],r
for p in sorted((ROOT/'method_v2').rglob('*.py')): ref(p)
for p in sorted((ROOT/'deployment_search').glob('routing*.py')): ref(p)
for name in ['results/module8b_quality_pool_finalization/quality_policy.json','results/module8e2_heldout_freeze/method_comparison_freeze.json'] : read(name)
# Source hashes attest the frozen catalog chain; raw model weights are not read.
for r in mem['source_hashes'].values(): assert ref(r['path'])['sha256']==r['sha256'],r
ids=sorted(pool['formal_model_ids']);models={m['id']:m for m in pool['models'] if m['id'] in ids}
assert set(ids)==set(bounds)==set(mem['models'])==set(cap['models']) and len(ids)==12
M={m:{int(k):v for k,v in mem['models'][m]['M'].items()} for m in ids}
roles={m:models[m]['role'] for m in ids}
measured=read('results/candidate_pool_v2/measured_vram.json');catalog={}
for m in ids:
 assert list(sorted(M[m]))==list(range(bounds[m]+1)) and M[m][1]==models[m]['measured_idle_vram_mib']
 evidence=cap['models'][m]['evidence'];values={}
 for x,value in M[m].items():
  provenance={'derivation':'absent model uses zero memory'} if x==0 else ({'method':measured['method'],'raw_measurement':ref(measured['models'][m]['raw_evidence']),'evidence':evidence['single_memory_source']} if x==1 else {'method':'mean of three measured high-load post-warm idle owned-VRAM observations','observations':evidence['observed_7c1_double_idle_mib'],'replica_evidence':evidence,'raw_measurements':[]})
  if x==2:
   raw=read('results/replica_scaling_real/measurement_validation.json')
   provenance['raw_measurements']=[r for k,r in raw['raw_file_hashes'].items() if f'{m}_r2_high' in k or f'{m}_r2_high' in r['path']]
   assert len(provenance['raw_measurements'])==3,(m,provenance['raw_measurements'])
   for r in provenance['raw_measurements']:assert ref(r['path'])['sha256']==r['sha256']
  values[str(x)]={'value':value,'unit':'MiB','source':ref(MEM,f'models.{m}.M.{x}'),'measurement_provenance':provenance,'value_sha256':hashlib.sha256(json.dumps({'model':m,'x':x,'value':value,'unit':'MiB'},sort_keys=True).encode()).hexdigest(),'measured':x>0}
 catalog[m]={'role':roles[m],'replica_evidence_bound':bounds[m],'values':values,'unmeasured_x2':bounds[m]<2}
write('memory_catalog_frozen.json',{'frozen':True,'models':catalog,'bounds_source':ref(BOUNDS),'no_x2_extrapolation':True,'formal_candidate_pool_source':ref(POOL)})
# Rule is saved before historical errors, deployment enumeration, or runtime outcomes.
rule={'module':'8F-1A','frozen_before_audit_and_enumeration':True,'primary_workload_scope':'six frozen 8B benchmarks plus six frozen 8E heldout workloads; current rates as diagnostic inputs only','primary_space':'intersection of F_w over all 12 workloads; deployment vectors counted once','additional_spaces':['per-workload','union','original-six intersection','heldout-six intersection'],'quality_disabled_models':'preserve frozen planning convention: no admitted routing edge implies x=0; raw enumeration still includes every bound-valid vector','capacity_check':'existing zero-objective capacity LP at frozen rho_max=0.8; no deployment optimizer, canonical route selection, or runtime filter','B1':'B_min_global','B2':'first empirical breakpoint at or above B1 reaching 1/3 of full physical-budget intersection space','B3':'B_SG_global','B4':'first empirical breakpoint at or above B1 reaching 2/3 of full physical-budget intersection space','B5':PHYSICAL,'deduplication':{'order':'B1,B2,B3,B4; reserve physical endpoint B5','trigger':['duplicate','non-increasing order','positive separation smaller than the smallest positive measured per-model memory increment after applying the safety factor'],'minimum_separation':'safety_factor * min(M_m(x+1)-M_m(x)) over frozen tables; data-derived, no arbitrary margin','action':'advance to nearest next unused empirical breakpoint; repeat if order/spacing trigger remains, log skipped range/count and reason; fail if no eligible point below physical endpoint','endpoint':'B5 is fixed by the user; no rounding or replacement'},'endpoint_conflict_policy':'49140 MiB is an explicitly mandated physical endpoint. If it is not an empirical sum breakpoint, retain it and mark the all-points-empirical condition false; never fabricate an empirical breakpoint.','runtime_performance_used':False,'algorithm_enumeration_claim_includes_this_audit':False,'curve_physical_cap':'All empirical sum breakpoints are retained, including above 49140 MiB. Primary N_feasible and fraction_feasible are clipped to the physical cap; unbounded counts are separate columns.'}
write('budget_selection_rule.json',rule)
print('Frozen inputs and selection rule saved.',flush=True)
# Find all saved steady per-measurement labels; never use aggregate steady label as a run.
paths=subprocess.run(['rg','-l','"stability_state_v2": "steady"','results','benchmark/results','--glob','*.json','--glob','!module8f1a_memory_breakpoints/**'],cwd=ROOT,capture_output=True,text=True,check=False).stdout.splitlines()
labels=defaultdict(list)
def walk(o,p):
 if isinstance(o,dict):
  if o.get('stability_state_v2')=='steady' and o.get('raw_sha256') and o.get('raw_result_path'):
   labels[o['raw_result_path']].append({'index_source':str(p),'expected_raw_sha256':o['raw_sha256'],'stability_reason':o.get('stability_reason_v2')})
  for k,v in o.items():
   if k not in ('request_records','warmup_request_records','priming_request_records','telemetry','models','per_task_metrics'):walk(v,p)
 elif isinstance(o,list):
  for v in o:walk(v,p)
for p in sorted(paths):
 p=ROOT/p
 if OUT in p.parents:continue
 walk(json.loads(p.read_text()),p)
rows=[];excluded=[];seen={}
for name,lab in sorted(labels.items()):
 p=Path(name)
 if not p.exists():excluded.append({'path':name,'reason':'missing_raw'});continue
 b=p.read_bytes();h=hashlib.sha256(b).hexdigest()
 if any(l['expected_raw_sha256']!=h for l in lab):raise RuntimeError('Historical raw hash mismatch: '+name)
 d=json.loads(b)
 if 'request_records' not in d or not d.get('measurement_window') or d.get('status')!='completed':excluded.append({'path':name,'reason':'not_completed_raw_timed_measurement'});continue
 obs=d.get('memory_observations') or {};planning=obs.get('planning_idle_mib',d.get('planning_memory_mib'));peak=obs.get('peak_device_mib')
 if planning is None or peak is None or planning<=0:excluded.append({'path':name,'reason':'no_simultaneous_explicit_planning_and_device_peak','steady_evidence':lab});continue
 assert math.isfinite(planning) and math.isfinite(peak)
 if d.get('valid_execution') is False:excluded.append({'path':name,'reason':'invalid_execution'});continue
 identity=(d['measurement_window']['start_time_ns'],d['measurement_window'].get('start_unix_ns'),json.dumps(d.get('worker_identities',{}),sort_keys=True))
 if h in seen or identity in seen:
  rows[seen.get(h,seen.get(identity))]['raw_aliases'].append({'path':name,'sha256':h});continue
 e=(peak-planning)/planning
 r={'measurement_id':h,'raw_source':{'path':name,'sha256':h,'bytes':len(b)},'raw_aliases':[],'steady_evidence':lab,'planning_mib':planning,'actual_peak_mib':peak,'peak_definition':'sampled full-device VRAM peak over recorded lifecycle (conservative physical-budget scope)','peak_owned_mib':obs.get('peak_owned_mib'),'peak_workload_mib':obs.get('peak_workload_mib'),'unit':'MiB','e_i':e,'e_i_positive':max(e,0),'deployment':d.get('deployment'),'measurement_window':d['measurement_window'],'campaign':p.relative_to(ROOT).parts[1]}
 seen[h]=seen[identity]=len(rows);rows.append(r)
assert rows
positive=[r['e_i_positive'] for r in rows];epsilon=max(positive);factor=1+epsilon
write('memory_error_audit.json',{'frozen':True,'n':len(rows),'statistics_on_all_e_i_positive_including_zeros':stats(positive),'signed_error_statistics':stats([r['e_i'] for r in rows]),'positive_only_statistics':stats([e for e in positive if e>0]),'epsilon_mem':epsilon,'max_positive_error':epsilon,'steady_index_files_scanned':len(paths),'per_measurement_steady_references':len(labels),'deduplication':'raw hash then physical measurement start timestamps + worker identities; copied/reused measurements counted once','campaign_counts':dict(Counter(r['campaign'] for r in rows)),'excluded':excluded,'measurements':rows,'latency_used_for_feasibility_or_budget_selection':False})
write('memory_safety_factor.json',{'frozen':True,'epsilon_mem':epsilon,'safety_factor':factor,'definition':'M_safe(x) = (1 + epsilon_mem) * sum_m M_m(x_m)','epsilon_rule':'max_i max((actual_peak_i-planning_i)/planning_i,0) over authenticated steady historical timed measurements with explicit planning and device peak','maximum_error_measurements':[r['raw_source'] for r in rows if r['e_i_positive']==epsilon],'arbitrary_margin':False,'scope':'Empirical maximum of sampled historical observations; no unmeasured peak guarantee.'})
print(f'Historical audit: n={len(rows)}, epsilon={epsilon:.12g}',flush=True)
from deployment_core.models import ModelProfile
from deployment_core.deployment import Deployment
from deployment_core.workload import Workload
from deployment_core.replica_calibration import ReplicaCalibration
from deployment_core.routing_solver import solve_capacity_lp
planning_models=plan['models'];disabled=plan['quality_disabled_model_variables']
admitted={(r['task'],r['model']) for r in admission['rows'] if r['quality_admissible']}
for m,p in planning_models.items():assert set(p['supported_tasks'])=={t for t in models[m]['capabilities_verified'] if (t,m) in admitted}
assert set(disabled)==set(ids)-set(planning_models)
catalog_lp={m:ModelProfile(m,frozenset(p['supported_tasks']),p['memory_mib'],p['service_time_ms'],p['sources']) for m,p in planning_models.items()}
cal=ReplicaCalibration({m:{int(k):v for k,v in cap['models'][m]['g'].items()} for m in catalog_lp},{m:M[m] for m in catalog_lp},{m:bounds[m] for m in catalog_lp})
cal.validate(catalog_lp)
assert all(all(v[b]>=v[a] for a,b in zip(sorted(v),sorted(v)[1:])) for v in cal.capacity_factors.values())
workloads=[{'id':p['name'],'suite':'original_six','lambda_rps':p['lambda_rps'],'source':ref(WORK,f'proposals[{p["name"]}]')} for p in work['proposals']]+[{'id':p['id'],'suite':'heldout_six','lambda_rps':p['lambda_rps'],'source':ref(HELD,f'workloads[{p["id"]}]')} for p in held['workloads']]
W={w['id']:Workload(w['lambda_rps']) for w in workloads}
theoretical=math.prod(bounds[m]+1 for m in ids)
vectors=list(itertools.product(*(range(bounds[m]+1) for m in ids)));assert len(vectors)==theoretical
index={x:i for i,x in enumerate(vectors)}
states=[]
for i,v in enumerate(vectors):
 mp=math.fsum(M[m][n] for m,n in zip(ids,v));ng=sum(n for m,n in zip(ids,v) if roles[m]=='generalist');ns=sum(v)-ng
 states.append({'state_id':i,'vector':v,'M_plan_mib':mp,'M_safe_mib':factor*mp,'N_G':ng,'N_S':ns,'total_replicas':ng+ns,'deployed_model_identities':';'.join(f'{m}#{j}' for m,n in zip(ids,v) for j in range(n))})
def deployment(i):return dict(zip(ids,vectors[i]))
def desc(i):return {'state_id':i,'deployment':deployment(i),'M_plan_mib':states[i]['M_plan_mib'],'M_safe_mib':states[i]['M_safe_mib']}
# Every vector is visited. Capacity feasibility is inherited only from a verified
# componentwise smaller witness; empirical g(x) is nondecreasing in all tables.
F={};CQ={};witness_rows=[];calls={};summaries=[];rho=read('results/module8b_quality_pool_finalization/planning_protocol.json')['rho_max']
for w in workloads:
 name=w['id'];feasible=set();quality=set();origins={};ncalls=0
 for i,v in enumerate(vectors):
  if any(v[ids.index(m)] for m in disabled):continue
  active={m for m,n in zip(ids,v) if n}
  if any(not any(t in planning_models[m]['supported_tasks'] for m in active) for t in W[name].active_tasks):continue
  quality.add(i);predecessor=None
  for j,n in enumerate(v):
   if n:
    prev=v[:j]+(n-1,)+v[j+1:];pi=index[prev]
    if pi in feasible:predecessor=pi;break
  if predecessor is not None:
   feasible.add(i);origins[i]=origins[predecessor];continue
  counts={m:n for m,n in zip(ids,v) if m in catalog_lp}
  route=solve_capacity_lp(Deployment(counts),catalog_lp,W[name],rho,capacity_mode='empirical',calibration=cal);ncalls+=1
  if route is not None:
   feasible.add(i);origins[i]=i
   witness_rows.append({'workload':name,**desc(i),'routing_rps':{f'{k}::{m}':v for (k,m),v in route.routing_rps.items()},'utilization':route.utilization})
 F[name]=feasible;CQ[name]=quality;calls[name]=ncalls
 for s in states:s[f'feasible_{name}']=s['state_id'] in feasible;s[f'capacity_certificate_{name}']=origins.get(s['state_id'])
 physically={i for i in feasible if states[i]['M_safe_mib']<=PHYSICAL}
 summaries.append({'workload':name,'suite':w['suite'],'active_tasks':';'.join(W[name].active_tasks),'vectors_enumerated':theoretical,'quality_capability_feasible_count':len(quality),'planning_capacity_feasible_count':len(feasible),'safe_physical_feasible_count':len(physically),'full_budget_feasible_fraction_of_enumerated':len(physically)/theoretical,'B_min_mib':min(states[i]['M_safe_mib'] for i in feasible),'capacity_checked':True,'rho_max':rho,'zero_objective_capacity_LP_calls':ncalls,'runtime_latency_filter_used':False})
 print(f'Feasibility {name}: {len(feasible)} capacity-feasible, {len(physically)} at physical endpoint, {ncalls} LP certificates/checks.',flush=True)
intersection=set.intersection(*F.values());union=set.union(*F.values());original=set.intersection(*(F[w['id']] for w in workloads if w['suite']=='original_six'));heldspace=set.intersection(*(F[w['id']] for w in workloads if w['suite']=='heldout_six'))
spaces={'all_workloads_intersection':intersection,'any_workload_union':union,'original_six_intersection':original,'heldout_six_intersection':heldspace,**F}
full={name:{i for i in f if states[i]['M_safe_mib']<=PHYSICAL} for name,f in spaces.items()};assert all(full.values())
minimum={}
for name,f in F.items():
 b=min(states[i]['M_safe_mib'] for i in f);mins=sorted(i for i in f if states[i]['M_safe_mib']==b)
 minimum[name]={'B_min_mib':b,'B_min_gib':b/1024,'minimum_deployments':[desc(i) for i in mins]}
bmin=max(v['B_min_mib'] for v in minimum.values())
write('minimum_feasible_budgets.json',{'workloads':minimum,'B_min_global_mib':bmin,'B_min_global_gib':bmin/1024,'definition':'max_w min_{x in F_w} M_safe(x)','intersection_minimum_mib':min(states[i]['M_safe_mib'] for i in intersection),'global_minimum_does_not_generally_imply_common_deployment':True})
# Load frozen SG states, reconstruct heldout states from the frozen cache identity,
# then verify the original argmin-s0 rule without examining performance.
sg={s['workload']:s['deployment'] for s in sgstates['states'] if s['baseline']=='B2'};sg_refs={n:ref(SGSTATES) for n in sg}
for w in workloads:
 if w['suite']!='heldout_six':continue
 p=f'results/module8e3_heldout_validation/{w["id"]}/B2/search_cache/runtime_cache_manifest.json'
 d=read(p);identities=[e['identity']['deployment'] for e in d['configurations'].values()];assert len(identities)==1
 sg[w['id']]=identities[0];sg_refs[w['id']]=ref(p,'configurations.*.identity.deployment')
sgrows=[]
for w in workloads:
 name=w['id'];selected={}
 for t in W[name].active_tasks:
  capable=[m for m in ids if (t,m) in admitted and t in models[m]['capabilities_verified']]
  specialists=[m for m in capable if roles[m]=='specialist'];choices=specialists or capable
  selected[t]=min(choices,key=lambda m:(planning_models[m]['service_time_ms'][t],m))
 expected={m:int(m in selected.values()) for m in ids};loaded={m:sg[name].get(m,0) for m in ids};assert loaded==expected
 i=index[tuple(loaded[m] for m in ids)];b=states[i]['M_safe_mib']
 fixed_load={m:math.fsum(w['lambda_rps'][t]*planning_models[m]['service_time_ms'][t]/1000 for t in selected if selected[t]==m) for m in selected.values()}
 sgrows.append({'workload':name,'B_SG_mib':b,'B_SG_gib':b/1024,'state':desc(i),'source':sg_refs[name],'selected_task_paths':selected,'frozen_fixed_routing_capacity_feasible':all(v<=rho*cal.capacity_factors[m][loaded[m]]+1e-9 for m,v in fixed_load.items()),'fixed_route_utilization':fixed_load,'canonical_capacity_feasible':i in F[name]})
bsg=max(r['B_SG_mib'] for r in sgrows)
write('specialist_greedy_thresholds.json',{'definition_source':ref(SGDEF,'B2'),'workloads':sgrows,'distribution_statistics_mib':stats([r['B_SG_mib'] for r in sgrows]),'distribution_counts':dict(Counter(r['B_SG_mib'] for r in sgrows)),'B_SG_global_mib':bsg,'B_SG_global_gib':bsg/1024,'meaning':'At this budget every frozen SG deployment satisfies the memory budget; fixed-route capacity is reported separately.','runtime_performance_used':False})
rep={}
for m in ids:
 if bounds[m]!=2:continue
 j=ids.index(m);per={}
 for name,f in spaces.items():
  for label,eligible in [('capability_quality',set.intersection(*(CQ.values())) if name=='all_workloads_intersection' else (CQ[name] if name in CQ else (set.union(*CQ.values()) if name=='any_workload_union' else set.intersection(*(CQ[w['id']] for w in workloads if w['suite']==('original_six' if name=='original_six_intersection' else 'heldout_six')))))) ,('planning_capacity',f)]:
   subset=[i for i in eligible if vectors[i][j]==2]
   b=min((states[i]['M_safe_mib'] for i in subset),default=None)
   per.setdefault(name,{})[label]={'B_rep_mib':b,'B_rep_gib':b/1024 if b is not None else None,'below_physical_endpoint':b is not None and b<=PHYSICAL,'minimum_deployments':[desc(i) for i in sorted(subset) if states[i]['M_safe_mib']==b]}
 rep[m]={'global_first_possible_mib':min(p['planning_capacity']['B_rep_mib'] for name,p in per.items() if name in F),'all_workloads_intersection_mib':per['all_workloads_intersection']['planning_capacity']['B_rep_mib'],'workloads_and_spaces':per}
write('replica_memory_breakpoints.json',{'models':rep,'global_first_possible_definition':'minimum over workload-specific thresholds; intersection threshold separately reported','runtime_replication_benefit_inferred':False})
breakpoints=sorted({s['M_safe_mib'] for s in states});assert len(breakpoints)==len({s['M_plan_mib'] for s in states})
memlists={name:sorted(states[i]['M_safe_mib'] for i in f) for name,f in spaces.items()}
def count(name,b):return bisect.bisect_right(memlists[name],b)
def fraction(name,b):return count(name,min(b,PHYSICAL))/len(full[name])
curve=[]
for b in breakpoints:
 row={'breakpoint_mib':b,'breakpoint_gib':b/1024,'within_physical_budget':b<=PHYSICAL,'N_feasible':count('all_workloads_intersection',min(b,PHYSICAL)),'N_feasible_without_physical_cap':count('all_workloads_intersection',b),'fraction_feasible':fraction('all_workloads_intersection',b),'N_feasible_physical_clipped':count('all_workloads_intersection',min(b,PHYSICAL))}
 for name in spaces:
  row[f'N_feasible_{name}']=count(name,b);row[f'fraction_physical_clipped_{name}']=fraction(name,b)
 curve.append(row)
csvwrite('memory_breakpoint_curve.csv',curve)
eligible_breakpoints=[b for b in breakpoints if bmin<=b<PHYSICAL]
one=next(b for b in eligible_breakpoints if count('all_workloads_intersection',b)*3>=len(full['all_workloads_intersection']))
two=next(b for b in eligible_breakpoints if count('all_workloads_intersection',b)*3>=2*len(full['all_workloads_intersection']))
rawbudgets=[bmin,one,bsg,two,PHYSICAL]
min_increment=min(M[m][x+1]-M[m][x] for m in ids for x in range(bounds[m]) if M[m][x+1]>M[m][x]);separation=min_increment*factor
selected=[];dedup=[]
for k,b in enumerate(rawbudgets[:4],1):
 original_b=b;reasons=[];skipped=[]
 while selected and (b<=selected[-1] or b-selected[-1]<separation or b in selected):
  reasons.extend(['duplicate'] if b in selected else ['order_conflict'] if b<=selected[-1] else ['empirically_tiny_spacing'])
  skipped.append(b);idx=bisect.bisect_right(breakpoints,max(b,selected[-1]));assert idx<len(breakpoints)
  b=breakpoints[idx];assert b<PHYSICAL,'Cannot satisfy deterministic dedup below physical endpoint'
 selected.append(b)
 dedup.append({'label':f'B{k}','raw_mib':original_b,'selected_mib':b,'changed':b!=original_b,'reasons':sorted(set(reasons)),'next_breakpoint_steps':len(skipped),'skipped_breakpoints_mib':skipped})
selected.append(PHYSICAL);assert all(a<b for a,b in zip(selected,selected[1:]))
regimes=[];candidate=[]
for k,b in enumerate(selected,1):
 primary=full['all_workloads_intersection'];reachable={i for i in primary if states[i]['M_safe_mib']<=b};assert reachable
 repmodels=[m for m in ids if bounds[m]==2 and any(vectors[i][ids.index(m)]==2 for i in reachable)]
 mini=min(reachable,key=lambda i:(states[i]['M_safe_mib'],i));actualbp=b in breakpoints
 candidate.append({'label':f'B{k}','budget_mib':b,'budget_gib':b/1024,'empirical_breakpoint':actualbp,'physical_endpoint':k==5,'raw_rule_mib':rawbudgets[k-1],'N_feasible':len(reachable),'fraction_feasible':len(reachable)/len(primary),'SG_all_memory_feasible':b>=bsg,'x2_models_feasible':repmodels,'headroom_to_physical_mib':PHYSICAL-b})
 for name,f in full.items():
  subset={i for i in f if states[i]['M_safe_mib']<=b};repms=[m for m in ids if bounds[m]==2 and any(vectors[i][ids.index(m)]==2 for i in subset)]
  scopesg=[s for s in sgrows if name not in F or s['workload']==name]
  if name=='original_six_intersection':scopesg=[s for s in sgrows if s['workload'] in [w['id'] for w in workloads if w['suite']=='original_six']]
  if name=='heldout_six_intersection':scopesg=[s for s in sgrows if s['workload'] in [w['id'] for w in workloads if w['suite']=='heldout_six']]
  regimes.append({'budget_label':f'B{k}','budget_mib':b,'budget_gib':b/1024,'scope':name,'feasible_deployment_count':len(subset),'feasible_fraction':len(subset)/len(f),'minimum_N_G':min((states[i]['N_G'] for i in subset),default=None),'maximum_N_G':max((states[i]['N_G'] for i in subset),default=None),'minimum_N_S':min((states[i]['N_S'] for i in subset),default=None),'maximum_N_S':max((states[i]['N_S'] for i in subset),default=None),'SG_memory_feasible':all(s['B_SG_mib']<=b for s in scopesg),'SG_fixed_routing_capacity_feasible':all(s['frozen_fixed_routing_capacity_feasible'] for s in scopesg),'any_x2_feasible':bool(repms),'x2_models_feasible':';'.join(repms),'representative_minimum_memory_deployment':json.dumps(deployment(min(subset,key=lambda i:(states[i]['M_safe_mib'],i))),sort_keys=True) if subset else None,'empirical_breakpoint':actualbp})
write('candidate_memory_budgets.json',{'status':'MECHANICALLY_DERIVED_CANDIDATES_WITH_EXPLICIT_PHYSICAL_ENDPOINT_EXCEPTION','candidates':candidate,'raw_budgets_mib':rawbudgets,'deduplication_log':dedup,'minimum_separation_mib':separation,'minimum_empirical_increment_mib':min_increment,'all_points_empirical':all(c['empirical_breakpoint'] for c in candidate),'physical_endpoint_exception':PHYSICAL not in breakpoints,'final_workload_intensity_frozen':False,'resource_sweep_started':False})
csvwrite('resource_regime_audit.csv',regimes)
csvwrite('workload_feasibility_summary.csv',summaries)
state_rows=[]
for s in states:
 row={k:v for k,v in s.items() if k!='vector'};row.update({f'x_{m}':n for m,n in zip(ids,s['vector'])});row['safe_physical_feasible']=s['M_safe_mib']<=PHYSICAL;row['feasible_all_workloads']=s['state_id'] in intersection;row['feasible_any_workload']=s['state_id'] in union;state_rows.append(row)
csvwrite('deployment_memory_states.csv',state_rows)
write('capacity_feasibility_certificates.json',{'method':'Existing zero-objective capacity LP; other feasible states inherit a componentwise smaller verified certificate. No deployment optimizer or canonical routing optimization.','rho_max':rho,'LP_calls':calls,'witnesses':witness_rows})
maxsafe=max(states[i]['M_safe_mib'] for i in full['all_workloads_intersection']);minsafe=min(states[i]['M_safe_mib'] for i in intersection)
manifest={'module':'8F-1A','purpose':'offline experimental-design enumeration; excluded from Proposed algorithm enumeration claims','model_order':ids,'replica_bounds':bounds,'theoretical_state_count':theoretical,'actual_enumerated_state_count':len(states),'includes_empty_vector':True,'unique_memory_breakpoints':len(breakpoints),'unique_memory_breakpoints_within_physical':sum(b<=PHYSICAL for b in breakpoints),'unique_nonzero_memory_breakpoints':len(breakpoints)-1,'quality_disabled_models_fixed_zero_for_feasibility':disabled,'workloads':workloads,'space_counts_without_budget':{name:len(f) for name,f in spaces.items()},'full_physical_safe_space_counts':{name:len(f) for name,f in full.items()},'all_workloads_safe_headroom_at_minimum_mib':PHYSICAL-minsafe,'maximum_safe_feasible_memory_mib':maxsafe,'headroom_above_maximum_safe_feasible_state_mib':PHYSICAL-maxsafe,'safe_physical_vector_count_without_quality':sum(s['M_safe_mib']<=PHYSICAL for s in states),'safe_factor_applied_without_adjustment':True,'execution_python':sys.executable,'new_GPU_runs':0,'deployment_optimizer_calls':0,'zero_objective_capacity_LP_calls':sum(calls.values()),'resource_sweep_runs':0,'final_intensity_frozen':False,'frozen_input_sources':dict(sources)}
write('deployment_space_manifest.json',manifest)
# Verify all audited frozen inputs at the end (not the output copies).
assert all(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in sources.values())
lines=['# Module 8F-1A — Empirical Memory Breakpoint Audit','', '完成纯离线审计。static v2、routing、quality policy、candidate pool、replica evidence bounds 均未修改。新 GPU experiment / resource sweep / deployment optimizer 调用均为 0；不冻结最终 workload intensity。','', '主口径：全部 12 个当前冻结 workload 的 capability / quality / frozen planning-capacity feasible 集合交集；逐 workload、原始六项交集、heldout 六项交集及并集也完整保存。无 admitted edge 的 PaddleOCR-VL 保留在 12-model 原始枚举中，但按现有冻结 planning convention 在 feasible 集合中固定 x=0。','', f'1. 实际枚举 **{len(states):,}** 个 vectors，理论状态数 3³ × 2⁹ = {theoretical:,}（含空向量和超过物理预算的向量）。',f'2. 历史独立 steady measurements n={len(rows)}；epsilon_mem={epsilon:.15g}（{epsilon*100:.9f}%），safety factor={factor:.15g}。使用全设备 sampled lifecycle peak；不是 5%/10% margin。positive error min/mean/median/p95/max = '+', '.join(f'{stats(positive)[k]:.12g}' for k in ['min','mean','median','p95','max'])+'。','3. 各 workload B_min（MiB）：','', '| Workload | B_min MiB | GiB |','|---|---:|---:|']
lines += [f'| {name} | {v["B_min_mib"]:.9f} | {v["B_min_gib"]:.9f} |' for name,v in minimum.items()]
lines += ['',f'4. B_min_global = **{bmin:.9f} MiB**。',f'5. Frozen SG B_SG_global = **{bsg:.9f} MiB**；SG distribution 见 specialist_greedy_thresholds.json；所有 frozen fixed routes 的 capacity feasibility = {all(s["frozen_fixed_routing_capacity_feasible"] for s in sgrows)}。','6. x=2 的 first-possible breakpoint（any workload）与全部 workload 交集 threshold：']
lines += [f'   - {m}: {r["global_first_possible_mib"]:.9f} MiB；交集 {r["all_workloads_intersection_mib"]:.9f} MiB。' for m,r in rep.items()]
lines += [f'7. 全枚举 unique memory breakpoints = **{len(breakpoints):,}**（含 0）；物理预算内 {sum(b<=PHYSICAL for b in breakpoints):,}；非零总数 {len(breakpoints)-1:,}。',f'8. Full {PHYSICAL} MiB 下全部 workload 交集 feasible deployments = **{len(full["all_workloads_intersection"]):,}**；并集 = {len(full["any_workload_union"]):,}。','9–12. 五个候选 budget、feasible-space 比例、SG 与 x=2：','', '| Budget | MiB | GiB | Feasible count | Fraction | All SG memory feasible | x=2 models |','|---|---:|---:|---:|---:|---|---|']
lines += [f'| {c["label"]} | {c["budget_mib"]:.9f} | {c["budget_gib"]:.9f} | {c["N_feasible"]} | {c["fraction_feasible"]:.9%} | {c["SG_all_memory_feasible"]} | {", ".join(c["x2_models_feasible"]) or "none"} |' for c in candidate]
lines += ['',f'13. 原始预算 = {rawbudgets} MiB。实测最小正 memory increment={min_increment} MiB，safe separation={separation:.9f} MiB。发生 deterministic dedup 的点：'+(', '.join(d['label']+' ('+','.join(d['reasons'])+')' for d in dedup if d['changed']) or 'none')+'；逐步 next-unused breakpoint 记录见 candidate_memory_budgets.json。','14. B1–B4 完全来自已有 empirical memory、历史误差、冻结 quality/capacity 与提前写定的规则，没有使用 Proposed runtime performance。B5=49140 MiB 是用户指定物理端点，'+('恰好是 empirical breakpoint。' if PHYSICAL in breakpoints else '**不是 empirical sum breakpoint**；“所有预算均为 empirical breakpoint”与“B5 精确等于 physical maximum”无法同时满足，已保留端点并明确记录例外，不伪造断点。'),f'15. 可以进入 Module 8F-1B Workload-Intensity Freeze：memory audit 已完成，保守安全系数后仍有 {len(full["all_workloads_intersection"]):,} 个统一 feasible states，统一 minimum 上方 headroom={PHYSICAL-minsafe:.9f} MiB。最终 resource-sweep budget 全断点条件存在上述 physical-endpoint 例外；本阶段不进入 8F-1B、不冻结 intensity、不运行 sweep。','',f'Physical sanity：所有候选 B_i≤{PHYSICAL} MiB；最大 safe feasible state={maxsafe:.9f} MiB，剩余 {PHYSICAL-maxsafe:.9f} MiB。epsilon_mem 未调整。','', '可复现：`/root/miniconda3/envs/gsdeploy/bin/python results/module8f1a_memory_breakpoints/run_audit.py`。部署 CSV 每个 workload 的 capacity_certificate 列指向 capacity_feasibility_certificates.json 中的实证 LP witness，可复核所有 feasible set。CSV/JSON 保留计算值，Markdown 显示小数不用于预算选择。历史 sampled maximum 仅为已有数据的误差上界，不保证未观测的峰值。','', '完成后停止。']
(OUT/'module8f1a_summary.md').write_text('\n'.join(lines)+'\n')
write('artifact_hashes.json',{p.name:ref(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in ('artifact_hashes.json','execution.log')})
print(json.dumps({'vectors':len(states),'epsilon':epsilon,'breakpoints':len(breakpoints),'full_feasible':len(full['all_workloads_intersection']),'budgets':selected,'B_min_global':bmin,'B_SG_global':bsg}),flush=True)
