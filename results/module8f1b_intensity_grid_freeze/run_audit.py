"""Offline Module 8F-1B. CPU LP only; never imports serving evaluators."""
import bisect,csv,hashlib,itertools,json,math,statistics,sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));R=ROOT/'results';A=R/'module8f1a_memory_breakpoints';sources={};PHYSICAL=49140
TOL=1e-9

def ref(p,field=None):
 p=Path(p);p=p if p.is_absolute() else ROOT/p;b=p.read_bytes();r={'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)};sources[str(p)]=r
 return dict(r,**({'field':field} if field else {}))
def read(p):r=ref(p);return json.loads(Path(r['path']).read_text())
def write(n,v):(OUT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvwrite(n,rows):
 with (OUT/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def close(a,b):return abs(a-b)<=TOL*max(1,abs(a),abs(b))
def cluster(values):
 out=[]
 for v in sorted(values):
  if not out or not close(v,out[-1]):out.append(v)
 return out
f1a=read(A/'deployment_space_manifest.json');assert read(A/'validation_results.json')['status']=='PASS'
for r in read(A/'artifact_hashes.json').values():assert ref(r['path'])['sha256']==r['sha256'],r
comparison=read(R/'module8e2_heldout_freeze/method_comparison_freeze.json')
for r in comparison['frozen_inputs'].values():assert ref(r['path'])['sha256']==r['sha256'],r
v2=read(R/'module8e1_method_revision/method_v2_source_hashes.json')
for r in v2['sources'].values():assert ref(r['path'])['sha256']==r['sha256'],r
for d in ('deployment_core','deployment_search','method_v1','method_v2'):
 for p in sorted((ROOT/d).glob('*.py')):ref(p)
plan=read(R/'module8b_quality_pool_finalization/planning_catalog.json');pool=read(R/'candidate_pool_v2_1/candidate_pool_v2_1.json');service=read(R/'candidate_pool_v2_1/service_demand_v2_1.json');admission=read(R/'module8b_quality_pool_finalization/quality_admission_matrix.json');mem=read(R/'replica_capacity_calibration/replica_memory_table.json');capacity=read(R/'replica_capacity_calibration/replica_capacity_factors.json');bounds=read(R/'replica_capacity_calibration/evidence_replica_bounds.json')['replica_bounds'];safety=read(A/'memory_safety_factor.json');factor=safety['safety_factor'];rho=comparison['search_config']['rho_max'];assert rho==.8
ids=f1a['model_order'];profiles=plan['models'];roles={m['id']:m['role'] for m in pool['models'] if m['id'] in ids};M={m:{int(k):v for k,v in mem['models'][m]['M'].items()} for m in ids};G={m:{int(k):v for k,v in capacity['models'][m]['g'].items()} for m in ids};disabled=plan['quality_disabled_model_variables'];tasks=['text','image','ocr','asr','visual_qa','lid'];admitted={(r['task'],r['model']) for r in admission['rows'] if r['quality_admissible']}
assert ids==sorted(pool['formal_model_ids']) and factor==1+safety['epsilon_mem']
for m,p in profiles.items():
 assert set(p['supported_tasks'])=={t for t in service['models'][m]['capabilities_verified'] if (t,m) in admitted}
 assert p['service_time_ms']=={t:service['models'][m]['service_time_ms'][t] for t in p['supported_tasks']}
held=read(R/'module8e2_heldout_freeze/heldout_workloads.json');generation=read(R/'module8e2_heldout_freeze/heldout_generation_spec.json');base={};base_records=[]
for j in range(1,4):
 members=[w for w in held['workloads'] if w['mixture_id']==j];moderate=next(w for w in members if w['intensity']=='moderate');name=f'mix_{j:02d}';lam={t:moderate['lambda_rps'][t]/moderate['reference_normalized_load_L'] for t in tasks};base[name]=lam
 for w in members:
  assert all(close(w['lambda_rps'][t],w['reference_normalized_load_L']*lam[t]) for t in tasks)
 base_records.append({'id':name,'lambda_base_rps':lam,'base_reference_demand_L':1.0,'composition_reference_demand_share':moderate['reference_demand_share'],'composition_raw_request_share':moderate['raw_request_share'],'legacy_moderate_high_provenance':members,'source':ref(R/'module8e2_heldout_freeze/heldout_workloads.json',f'workloads[mixture_id={j}]'),'scaling':'lambda(alpha)=alpha*lambda_base; old L=.4/.72 are provenance only, not final levels'})
write('base_mixture_freeze.json',{'frozen':True,'mixtures':base_records,'generation_source':ref(R/'module8e2_heldout_freeze/heldout_generation_spec.json'),'composition_changed':False,'new_rng_draws':0,'legacy_intensities_reused_as_final':False})
# Memory selection uses only the already audited 8F-1A curve, before capacity results.
with (A/'memory_breakpoint_curve.csv').open() as f:memory_curve=list(csv.DictReader(f))
ref(A/'memory_breakpoint_curve.csv');bps=[float(r['breakpoint_mib']) for r in memory_curve];fracs=[float(r['fraction_feasible']) for r in memory_curve]
bmin=read(A/'minimum_feasible_budgets.json')['B_min_global_mib'];bsg=read(A/'specialist_greedy_thresholds.json')['B_SG_global_mib'];old=read(A/'candidate_memory_budgets.json')['candidates']
def fraction(B):return fracs[bisect.bisect_right(bps,B)-1]
mrule={'frozen_before_capacity_analysis':True,'minimum_fraction_separation':.05,'fraction_source':'8F-1A all-12-workload safe physical-budget intersection curve; immutable legacy rates used only for resource-axis definition','anchors_priority':['B_min_global','B_SG_global','physical maximum'],'nonanchors_priority':['8F-1A first one-third point B2','8F-1A first two-thirds point B4','8F-1A shifted B3'],'order':'Insert anchors; consider nonanchors in semantic-priority order. If neighboring fractions differ <.05, advance candidate to the smallest next unused empirical breakpoint with >=.05 separation from both surviving neighbors. If impossible, omit nonanchor and log reason. Never move an anchor.','target_count':'4–6','physical_endpoint_exception':PHYSICAL,'runtime_results_used':False}
write('memory_budget_dedup_rule.json',mrule)
selected={bmin:'minimum_feasible_anchor',bsg:'frozen_SG_anchor',PHYSICAL:'physical_endpoint'};memorylog=[]
for label in ['B2','B4','B3']:
 original=next(c['budget_mib'] for c in old if c['label']==label);chosen=None
 for b in bps:
  if b<original or b>=PHYSICAL or b in selected:continue
  neighbors=sorted(selected);lo=max((p for p in neighbors if p<b),default=None);hi=min((p for p in neighbors if p>b),default=None)
  if (lo is None or fraction(b)-fraction(lo)>=.05) and (hi is None or fraction(hi)-fraction(b)>=.05):chosen=b;break
 memorylog.append({'old_label':label,'original_mib':original,'original_fraction':fraction(original),'selected_mib':chosen,'selected_fraction':fraction(chosen) if chosen else None,'changed':chosen!=original,'reason':'retain_semantic_point' if chosen==original else 'adjacent_fraction_gap_below_5pp; move to next qualifying empirical breakpoint' if chosen else 'no_separated_slot; omit_nonanchor'})
 if chosen:selected[chosen]={'B2':'one_third_space','B4':'two_thirds_space','B3':'separated_middle_space'}[label]
budgets=sorted(selected);assert 4<=len(budgets)<=6
# Intensity choice is preregistered before alpha LPs; physical endpoint is reference.
irule={'frozen_before_capacity_analysis':True,'base':'lambda_base has frozen Qwen-reference L=1; alpha is total reference demand L','normalization':'beta=alpha/alpha_max(w,49140 MiB)','reference_budget_mib':PHYSICAL,'candidate_levels':'all positive deployment capacity breakpoints normalized by mixture reference, union over all final budgets and mixtures; cluster numerical equivalents using frozen LP tolerance 1e-9, representative minimum','capacity_equality':'feasible at alpha<=alpha_limit; loses feasibility immediately after. Numerical comparisons use existing LP tolerance 1e-9.','population':'for each mixture/budget, all bound-valid memory-safe capability/quality-covered deployments at alpha -> 0+, not the 8F-1A capacity-filtered set','majority':'strict majority of final memory budgets (floor(n/2)+1), required separately for all three mixtures','I1_LOW_MODERATE':'largest normalized capacity breakpoint preserving >=2/3 of the zero-load population in a strict majority of budgets, with >=1 deployment at EVERY final budget for EVERY mixture','I2_PRESSURED':'largest normalized capacity breakpoint preserving >=1/3 in a strict majority of budgets, with >=2 deployments at every SG-anchor-or-looser budget for every mixture','I3_HIGH':'largest normalized capacity breakpoint preserving >=2 deployments in a strict majority of budgets for every mixture; must exceed I2. Replica usefulness/model trade-offs are audited afterward; no model-targeted rescaling.','distinctness':'Require beta_I1 < beta_I2 < beta_I3; no arbitrary 1x/2x/3x substitute; failure closes gate.','representative_mixture':'medoid of three frozen reference-demand composition vectors under total L1 distance; exact ties by mixture ID, chosen before capacity analysis','experiment_A_intensity':'highest frozen I-level that is feasible for representative mixture at every final memory budget; if none, close gate','experiment_B_budgets':'tight=B_min_global; loose=physical maximum; intermediate=nonanchor empirical budget with 8F-1A fraction closest to .5, ties lower memory','experiment_B_mixture':'same preregistered composition medoid','all_methods_equal_inputs':True,'runtime_results_used':False}
write('intensity_selection_rule.json',irule)
shares={r['id']:r['composition_reference_demand_share'] for r in base_records};dist={m:math.fsum(math.fsum(abs(shares[m][t]-shares[n][t]) for t in tasks) for n in base) for m in base};representative=min(dist,key=lambda m:(dist[m],m))
# RC-SG construction deliberately excludes empirical g from all choices.
rcspec={'frozen_before_capacity_analysis':True,'name':'Resource-Constrained Specialist Greedy (RC-SG)','inputs_for_selection':['frozen quality/capability admission','standalone s0','workload lambda','8F-1A safe empirical memory'],'replicas':'x in {0,1}; never add a second replica or tune after runtime','base_generalist':'For each admitted Generalist g, keep one g. Initially route each task admitted by g to g; repair every uncovered active task with the admitted Specialist minimizing s0[k,s], exact ties model ID. A specialist may repair multiple tasks. Discard incomplete or memory-unsafe bases. Among surviving bases, choose minimum sum_k lambda[k]*s0[k,assigned(k)]/1000; ties total M_safe, Generalist ID, full vector. If no base fits, emit BASE_MEMORY_OR_CAPABILITY_INFEASIBLE; no GPU.','greedy_value':'For each absent admitted Specialist s, improvements are active tasks for which s0[k,s] < s0[k,current_assignment]. Delta safe memory=(1+epsilon_mem)*M_s(1). value(s)=sum_k lambda[k]*(s0[k,current_assignment]-s0[k,s])/1000 / Delta safe memory. Uses actual current fallback (not an inadmissible Generalist edge).','selection':'Among positive-value specialists that fit remaining memory, choose descending value; ties smaller Delta safe memory, then model ID. Add one s; reassign all improved tasks; recompute gains. Stop when none fits with positive gain. Never evict base/repairs, never use g(x) or runtime outcomes in greedy selection.','canonical_routing':'After deployment construction, call the EXISTING unchanged empirical canonical routing routine for execution/preflight, at frozen rho_max. Canonical feasibility is a gate only, never a repair/tuning signal. If infeasible, retain construction and mark PLANNING_CAPACITY_INFEASIBLE; no runtime, substitute, or retry.','generalist_centric':'Retain frozen 8D-1 B1 Qwen-centric + full-six-task quality repair deployment and its frozen task-selected routing. Apply identical safe-memory and capacity gates; preserve infeasible outcomes; never violate the budget.','old_B2_reference':'Existing frozen Specialist-greedy B2 deployments/results retained untouched as loose-resource reference; not used to tune or generate current choices.','static_v2':'Unchanged code/config/routing. Resource budget is an experiment INPUT via equivalent planning budget B/(1+epsilon_mem), not a method revision. Seed/candidate safe-memory checks must be revalidated before execution.','runtime_performance_used':False}
write('resource_constrained_sg_spec.json',rcspec)
print('All selection rules preregistered; composition medoid='+representative+', memory points='+str(budgets),flush=True)
# Reuse raw bound-valid vectors from 8F-1A, not only its F_w sets.
with (A/'deployment_memory_states.csv').open() as f:raw=list(csv.DictReader(f))
ref(A/'deployment_memory_states.csv');states=[]
for row in raw:
 x={m:int(row['x_'+m]) for m in ids}
 if any(x[m] for m in disabled):continue
 if float(row['M_safe_mib'])>PHYSICAL:continue
 if any(not any(x[m]>0 and t in p['supported_tasks'] for m,p in profiles.items()) for t in tasks):continue
 states.append({'state_id':int(row['state_id']),'x':x,'M_safe_mib':float(row['M_safe_mib']),'N_G':int(row['N_G']),'N_S':int(row['N_S'])})
state_by_id={s['state_id']:s for s in states};vector_index={tuple(s['x'][m] for m in ids):s['state_id'] for s in states}
# Direct max-alpha LP: maximize a with conservation sum_m r[k,m]=a*lambda_base[k]
# and demand <= rho_max*g_m(x). This is offline analysis, not Proposed search.
LPcalls=0

def capacity_limit(x,lam):
 global LPcalls
 active=[m for m in profiles if x[m]>0];pairs=[(t,m) for t in tasks for m in active if t in profiles[m]['supported_tasks']];n=len(pairs);aeq=np.zeros((len(tasks),n+1));aub=np.zeros((len(active),n+1))
 for i,t in enumerate(tasks):aeq[i,-1]=-lam[t]
 for j,(t,m) in enumerate(pairs):aeq[tasks.index(t),j]=1;aub[active.index(m),j]=profiles[m]['service_time_ms'][t]/1000
 c=np.zeros(n+1);c[-1]=-1;LPcalls+=1
 result=linprog(c,A_eq=aeq,b_eq=np.zeros(len(tasks)),A_ub=aub,b_ub=np.array([rho*G[m][x[m]] for m in active]),bounds=[(0,None)]*(n+1),method='highs',options={'primal_feasibility_tolerance':TOL,'dual_feasibility_tolerance':TOL})
 if result.status==2:return {'alpha_limit':0,'routing_rps':{},'model_pressure':{},'bottleneck_models':[],'bottleneck_tasks':[]}
 assert result.success,result.message;a=float(result.x[-1]);r={f'{t}::{m}':float(v) for (t,m),v in zip(pairs,result.x[:-1])};pressure={m:math.fsum(r.get(f'{t}::{m}',0)*profiles[m]['service_time_ms'].get(t,0)/1000 for t in tasks)/G[m][x[m]] for m in active}
 assert all(close(math.fsum(r.get(f'{t}::{m}',0) for m in active),a*lam[t]) for t in tasks)
 assert all(v<=rho+TOL*10 for v in pressure.values())
 bn=[m for m in sorted(active) if close(pressure[m],rho)];bt=[]
 for m in bn:
  contrib={t:r.get(f'{t}::{m}',0)*profiles[m]['service_time_ms'].get(t,0)/1000/G[m][x[m]] for t in tasks};mx=max(contrib.values());bt.extend({'model':m,'task':t,'effective_demand':v} for t,v in contrib.items() if v>0 and close(v,mx))
 return {'alpha_limit':a,'routing_rps':r,'model_pressure':pressure,'z':max(pressure.values()),'bottleneck_models':bn,'bottleneck_tasks':bt}
limits={};capacity_rows=[]
for mixture,lam in base.items():
 limits[mixture]={}
 for s in states:
  result=capacity_limit(s['x'],lam);limits[mixture][s['state_id']]=result
  capacity_rows.append({'mixture':mixture,'state_id':s['state_id'],'M_safe_mib':s['M_safe_mib'],'alpha_limit':result['alpha_limit'],'bottleneck_models':';'.join(result['bottleneck_models']),'bottleneck_tasks':json.dumps(result['bottleneck_tasks'],sort_keys=True),'replica_vector':json.dumps(s['x'],sort_keys=True)})
 print(f'{mixture}: max-alpha LP completed for {len(states)} memory-safe covered vectors.',flush=True)
csvwrite('deployment_capacity_limits.csv',capacity_rows)
# Include original candidate budgets and every final budget in the audit.
audit_budgets=sorted(set(budgets+[c['budget_mib'] for c in old]+[bsg]));pops={b:[s for s in states if s['M_safe_mib']<=b] for b in audit_budgets}
alpha_ref={m:max(limits[m][s['state_id']]['alpha_limit'] for s in pops[PHYSICAL]) for m in base}

def max_boundary(m,pop):
 maximum=max((limits[m][s['state_id']]['alpha_limit'] for s in pop),default=0);ties=[s for s in pop if close(limits[m][s['state_id']]['alpha_limit'],maximum)]
 if not ties:return {'alpha_max':0,'boundary_deployment':None}
 chosen=min(ties,key=lambda s:(s['M_safe_mib'],sum(s['x'].values()),tuple(s['x'][i] for i in ids)));c=limits[m][chosen['state_id']]
 return {'alpha_max':maximum,'boundary_state_id':chosen['state_id'],'boundary_deployment':chosen['x'],'replica_vector':chosen['x'],'boundary_M_safe_mib':chosen['M_safe_mib'],'bottleneck_model':c['bottleneck_models'],'bottleneck_task':c['bottleneck_tasks'],'boundary_routing_rps':c['routing_rps'],'boundary_model_pressure':c['model_pressure'],'z':c['z'],'maximizing_state_count_numerical_ties':len(ties)}
boundaries={m:{str(b):max_boundary(m,pops[b]) for b in audit_budgets} for m in base}
write('alpha_capacity_boundaries.json',{'alpha_units':'frozen Qwen-reference total demand L; lambda_base L=1','rho_max':rho,'alpha_max_reference_by_mixture':alpha_ref,'reference_budget_mib':PHYSICAL,'budgets':audit_budgets,'mixtures':boundaries,'offline_design_enumeration_only':True,'Proposed_search_calls':0,'capacity_LP_calls':LPcalls,'numerical_tolerance':TOL,'per_state_limits_source':ref(OUT/'deployment_capacity_limits.csv')})
sg_defs=read(R/'module8d_baseline_freeze/baseline_definitions.json');old_states=read(R/'module8d_baseline_freeze/baseline_deployments.json');sgstate=next(s for s in old_states['states'] if s['baseline']=='B2' and s['workload']=='demand_balanced');gcstate=next(s for s in old_states['states'] if s['baseline']=='B1' and s['workload']=='demand_balanced');sgx=sgstate['deployment'];gcx=gcstate['deployment'];sgpath=sgstate['selected_task_model'];gcpath=gcstate['selected_task_model']
def assigned_limit(x,paths,lam):
 pressure={m:math.fsum(lam[t]*profiles[m]['service_time_ms'][t]/1000 for t in tasks if paths[t]==m)/G[m][x[m]] for m in profiles if x[m]};z=max(pressure.values());return rho/z
sglimits={m:{'fixed_routing_alpha_limit':assigned_limit(sgx,sgpath,lam),'canonical_routing_alpha_limit':limits[m][vector_index[tuple(sgx[i] for i in ids)]]['alpha_limit'],'M_safe_mib':bsg,'deployment':sgx,'routing_paths':sgpath} for m,lam in base.items()}
# Each curve includes inclusive and immediately-right counts: no fake "x=2 first
# becomes feasible" event as load rises; all positive states are feasible near 0.
curves=[];replica_summary={};pairrows=[];transition_summary={};single_ids={s['state_id'] for s in states if all(v<=1 for v in s['x'].values())}
def capped_count(m,pop,a,strict=False):
 return sum(limits[m][s['state_id']]['alpha_limit']>a+TOL*max(1,abs(a)) if strict else limits[m][s['state_id']]['alpha_limit']>=a-TOL*max(1,abs(a)) for s in pop)
def regime_sets(pop):return {'generalist_only':[s for s in pop if s['N_G']>0 and s['N_S']==0],'hybrid':[s for s in pop if s['N_G']>0 and s['N_S']>0],'specialist_only':[s for s in pop if s['N_G']==0],'specialist_heavy':[s for s in pop if s['N_S']>s['N_G']]}
for mixture in base:
 replica_summary[mixture]={};transition_summary[mixture]={}
 for B,pop in pops.items():
  values=cluster([limits[mixture][s['state_id']]['alpha_limit'] for s in pop]);regimes=regime_sets(pop);singles=[s for s in pop if s['state_id'] in single_ids]
  transition_summary[mixture][str(B)]={'first_x1_state_loss_after_alpha':min(limits[mixture][s['state_id']]['alpha_limit'] for s in singles) if singles else None,'all_x1_states_lost_after_alpha':max(limits[mixture][s['state_id']]['alpha_limit'] for s in singles) if singles else None,'regime_last_feasible_alpha':{k:max((limits[mixture][s['state_id']]['alpha_limit'] for s in ss),default=None) for k,ss in regimes.items()},'SG_fixed_first_infeasible':'immediately after fixed_routing_alpha_limit if memory fits; otherwise memory-infeasible at every alpha','SG_fixed_alpha_limit':sglimits[mixture]['fixed_routing_alpha_limit'],'SG_memory_feasible':bsg<=B,'unique_deployment_capacity_breakpoint_count':len(values)}
  replica_summary[mixture][str(B)]={}
  for model in ('qwen','whisper-small','resnet50'):
   doubles=[s for s in pop if s['x'][model]==2];useful=[]
   for s in doubles:
    v=tuple(s['x'][m]-(1 if m==model else 0) for m in ids);prev=vector_index[v];a1=limits[mixture][prev]['alpha_limit'];a2=limits[mixture][s['state_id']]['alpha_limit']
    if a2>a1+TOL*max(1,a1,a2):
     useful.append((a1,a2,s['state_id'],prev));pairrows.append({'mixture':mixture,'budget_mib':B,'model':model,'x2_state_id':s['state_id'],'x1_state_id':prev,'activation_infimum_alpha_open':a1,'useful_through_alpha_inclusive':a2,'capacity_gain_alpha':a2-a1})
   no_double=[s for s in pop if s['x'][model]<=1];no_max=max((limits[mixture][s['state_id']]['alpha_limit'] for s in no_double),default=0);two_max=max((limits[mixture][s['state_id']]['alpha_limit'] for s in doubles),default=0)
   earliest=min(useful,key=lambda v:(v[0],v[1],v[2])) if useful else None
   firstpoint=next((v for v in values if earliest and v>earliest[0]+TOL*max(1,earliest[0]) and v<=earliest[1]+TOL*max(1,earliest[1])),None)
   replica_summary[mixture][str(B)][model]={'x2_memory_feasible':bool(doubles),'memory_feasible_x2_state_count':len(doubles),'capacity_useful_pair_count':len(useful),'activation_infimum_alpha':earliest[0] if earliest else None,'activation_lower_boundary_open':True,'first_empirical_capacity_breakpoint_inside_earliest_useful_interval':firstpoint,'firstpoint_exists':firstpoint is not None,'earliest_useful_interval_alpha':{'lower_open':earliest[0],'upper_closed':earliest[1]} if earliest else None,'representative_x2_deployment':state_by_id[earliest[2]]['x'] if earliest else None,'representative_x1_deployment':state_by_id[earliest[3]]['x'] if earliest else None,'global_no_replica_alpha_max':no_max,'global_x2_alpha_max':two_max,'globally_necessary_after_alpha':no_max if two_max>no_max+TOL*max(1,no_max,two_max) else None,'globally_necessary_until_alpha':two_max if two_max>no_max+TOL*max(1,no_max,two_max) else None,'global_necessity_meaning':'No bound-valid memory-safe feasible deployment without this model at x=2 exists above the open threshold, while at least one x=2 deployment remains feasible.'}
  for a in [0.]+values:
   active=[s for s in pop if limits[mixture][s['state_id']]['alpha_limit']>=a-TOL*max(1,a)];count=len(active)
   row={'mixture':mixture,'budget_mib':B,'budget_is_final':B in budgets,'alpha_breakpoint':a,'beta_breakpoint':a/alpha_ref[mixture],'feasible_count_at_breakpoint':count,'feasible_count_immediately_after':capped_count(mixture,pop,a,True),'zero_load_population':len(pop),'feasible_fraction_of_zero_load':count/len(pop),'x1_only_vector_count':sum(s['state_id'] in single_ids for s in active),'SG_memory_feasible':bsg<=B,'SG_fixed_capacity_feasible':bsg<=B and a<=sglimits[mixture]['fixed_routing_alpha_limit']+TOL*max(1,a)}
   for k,ss in regimes.items():row[k+'_feasible_count']=capped_count(mixture,ss,a)
   for model in ('qwen','whisper-small','resnet50'):
    row['x2_'+model+'_feasible_count']=sum(s['x'][model]==2 for s in active);row['x2_'+model+'_capacity_useful_pair_count']=sum(s['x'][model]==2 and limits[mixture][vector_index[tuple(s['x'][m]-(1 if m==model else 0) for m in ids)]]['alpha_limit']<a-TOL*max(1,a) for s in active)
   curves.append(row)
csvwrite('intensity_breakpoint_curve.csv',curves)
csvwrite('replica_useful_intervals.csv',pairrows)
write('replica_activation_thresholds.json',{'definition':'Onset is an infimum/open lower boundary: x1 is feasible at its maximum, x2 first solves its failure for alpha strictly above that maximum and no larger than paired x2 maximum. No arbitrary offset alpha is manufactured.','memory_feasible_vs_capacity_useful':'x2 memory feasibility is alpha-independent. Capacity usefulness is the paired open-closed interval; globally necessary replication is separately audited over all vectors.','rho_max':rho,'mixtures':replica_summary,'capacity_and_regime_transitions':transition_summary,'frozen_SG_thresholds':sglimits,'all_useful_pairs_source':ref(OUT/'replica_useful_intervals.csv'),'planning_transfer_scope':capacity['transfer_scope'],'runtime_replication_benefit_inferred':False})
# Mechanical intensity selection, no performance reads. Binary search on monotone
# survival conditions over the normalized capacity breakpoint union.
betapoints=cluster([limits[m][s['state_id']]['alpha_limit']/alpha_ref[m] for m in base for B in budgets for s in pops[B]])
majority=len(budgets)//2+1
sortedlimits={(m,B):sorted(limits[m][s['state_id']]['alpha_limit']/alpha_ref[m] for s in pops[B]) for m in base for B in budgets}
def n_at(m,B,beta):v=sortedlimits[m,B];return len(v)-bisect.bisect_left(v,beta-TOL*max(1,beta))
def choose_condition(kind,beta):
 for m in base:
  if kind=='I1':
   if sum(n_at(m,B,beta)*3>=2*len(pops[B]) for B in budgets)<majority or any(n_at(m,B,beta)<1 for B in budgets):return False
  elif kind=='I2':
   if sum(n_at(m,B,beta)*3>=len(pops[B]) for B in budgets)<majority or any(n_at(m,B,beta)<2 for B in budgets if B>=bsg):return False
  elif sum(n_at(m,B,beta)>=2 for B in budgets)<majority:return False
 return True
chosen_beta={}
for label in ('I1','I2','I3'):
 valid=[b for b in betapoints if choose_condition(label,b)];assert valid,label;chosen_beta[label]=max(valid)
assert chosen_beta['I1']<chosen_beta['I2']<chosen_beta['I3'],chosen_beta
intensities=[]
for label,beta in chosen_beta.items():
 records={}
 for m,lam in base.items():
  alpha=beta*alpha_ref[m];stats=[]
  for B in budgets:
   feasible=[s for s in pops[B] if limits[m][s['state_id']]['alpha_limit']>=alpha-TOL*max(1,alpha)];useful={model:sum(s['x'][model]==2 and limits[m][vector_index[tuple(s['x'][n]-(1 if n==model else 0) for n in ids)]]['alpha_limit']<alpha-TOL*max(1,alpha) for s in feasible) for model in ('qwen','whisper-small','resnet50')}
   model_useful={}
   for model in profiles:
    n=0
    for st in feasible:
     if st['x'][model]!=1:continue
     prev=vector_index.get(tuple(st['x'][mm]-(1 if mm==model else 0) for mm in ids))
     if prev is not None and limits[m][prev]['alpha_limit']<alpha-TOL*max(1,alpha):n+=1
    model_useful[model]=n
   stats.append({'budget_mib':B,'feasible_count':len(feasible),'zero_load_fraction':len(feasible)/len(pops[B]),'any_capacity_useful_replica':any(useful.values()),'capacity_useful_replica_pair_counts':useful,'capacity_useful_single_model_pair_counts':model_useful,'any_capacity_useful_model_tradeoff':any(model_useful.values()),'minimum_N_G':min((s['N_G'] for s in feasible),default=None),'maximum_N_G':max((s['N_G'] for s in feasible),default=None),'minimum_N_S':min((s['N_S'] for s in feasible),default=None),'maximum_N_S':max((s['N_S'] for s in feasible),default=None)})
  records[m]={'alpha':alpha,'lambda_rps':{t:alpha*lam[t] for t in tasks},'total_rps':math.fsum(alpha*v for v in lam.values()),'budget_feasibility':stats}
 intensities.append({'id':label,'regime':{'I1':'LOW_MODERATE','I2':'PRESSURED','I3':'HIGH'}[label],'beta':beta,'mixtures':records})
write('frozen_intensity_levels.json',{'frozen':True,'levels':intensities,'alpha_max_reference':alpha_ref,'reference_budget_mib':PHYSICAL,'selection_rule':ref(OUT/'intensity_selection_rule.json'),'numerical_tolerance':TOL,'unified_normalized_beta':True,'old_moderate_high_provenance_only':True,'runtime_read':False})
# Final resource-axis semantics: legacy fraction plus new intensity-conditioned counts.
final_memory=[]
for j,B in enumerate(budgets,1):
 pop=pops[B];x2=[m for m in ('qwen','whisper-small','resnet50') if any(s['x'][m]==2 for s in pop)]
 final_memory.append({'id':f'M{j}','budget_mib':B,'budget_gib':B/1024,'legacy_8f1a_feasible_fraction':fraction(B),'legacy_8f1a_feasible_count':int(memory_curve[bisect.bisect_right(bps,B)-1]['N_feasible']),'zero_load_quality_memory_feasible_count':len(pop),'SG_memory_feasible':B>=bsg,'x2_memory_feasible_models':x2,'semantic_regime':selected[B],'empirical_breakpoint':B in bps,'physical_endpoint_exception':B==PHYSICAL and B not in bps,'planning_budget_input_mib':B/factor,'new_intensity_feasibility':{level['id']:{m:next(v for v in level['mixtures'][m]['budget_feasibility'] if v['budget_mib']==B) for m in base} for level in intensities}})
write('final_memory_budgets.json',{'frozen':True,'budgets':final_memory,'deduplication_log':memorylog,'minimum_fraction_separation':.05,'anchors':{'B_min_global':bmin,'B_SG_global':bsg,'physical_maximum':PHYSICAL},'legacy_B2_reference_preserved':True,'legacy_shifted_B3_reference':next(c for c in old if c['label']=='B3'),'all_nonphysical_points_empirical':all(v['empirical_breakpoint'] for v in final_memory if v['budget_mib']!=PHYSICAL),'selection_rule':ref(OUT/'memory_budget_dedup_rule.json')})
print('Frozen normalized betas='+str(chosen_beta)+'; alpha references='+str(alpha_ref),flush=True)
# Baseline construction implementation and CPU preflight for concrete grid cells.
def safe_memory(x):return factor*math.fsum(M[m][x.get(m,0)] for m in ids)
def rc_sg(lam,B):
 bases=[]
 for generalist in sorted(m for m in profiles if roles[m]=='generalist'):
  paths={};valid=True
  for t in tasks:
   if t in profiles[generalist]['supported_tasks']:paths[t]=generalist
   else:
    choices=[m for m in profiles if roles[m]=='specialist' and t in profiles[m]['supported_tasks']]
    if not choices:valid=False;break
    paths[t]=min(choices,key=lambda m:(profiles[m]['service_time_ms'][t],m))
  if not valid:continue
  x={m:int(m==generalist or m in paths.values()) for m in ids};memory=safe_memory(x)
  if memory>B:continue
  demand=math.fsum(lam[t]*profiles[paths[t]]['service_time_ms'][t]/1000 for t in tasks)
  bases.append((demand,memory,generalist,tuple(x[m] for m in ids),x,paths))
 if not bases:return {'status':'BASE_MEMORY_OR_CAPABILITY_INFEASIBLE','deployment':None,'trace':[]}
 chosen=min(bases,key=lambda row:row[:4]);x=dict(chosen[4]);paths=dict(chosen[5]);trace=[{'operation':'BASE','generalist':chosen[2],'M_safe_mib':chosen[1],'assignment':dict(paths),'weighted_worker_demand':chosen[0]}]
 while True:
  choices=[]
  for s in sorted(m for m in profiles if roles[m]=='specialist' and not x[m]):
   improved=[t for t in tasks if t in profiles[s]['supported_tasks'] and profiles[s]['service_time_ms'][t]<profiles[paths[t]]['service_time_ms'][t]]
   gain=math.fsum(lam[t]*(profiles[paths[t]]['service_time_ms'][t]-profiles[s]['service_time_ms'][t])/1000 for t in improved);delta=factor*M[s][1];proposal=dict(x);proposal[s]=1
   if gain>0 and safe_memory(proposal)<=B:choices.append((-gain/delta,delta,s,improved,proposal,gain))
  if not choices:break
  row=min(choices,key=lambda row:row[:3]);x=row[4]
  for t in row[3]:paths[t]=row[2]
  trace.append({'operation':'ADD','model':row[2],'value_per_safe_mib':-row[0],'delta_safe_mib':row[1],'gain_seconds_per_second':row[5],'improved_tasks':row[3],'M_safe_mib':safe_memory(x),'assignment':dict(paths)})
 return {'status':'CONSTRUCTED','deployment':x,'assignment':paths,'M_safe_mib':safe_memory(x),'trace':trace}
from deployment_core.models import ModelProfile,Hardware
from deployment_core.workload import Workload
from deployment_core.replica_calibration import ReplicaCalibration,ReplicaPlanningPolicy
from deployment_core.milp_seeder import MilpDeploymentSeeder,validate_seed
from deployment_search.routing import solve_canonical_routing
cat={m:ModelProfile(m,frozenset(p['supported_tasks']),p['memory_mib'],p['service_time_ms'],p['sources']) for m,p in profiles.items()};cal=ReplicaCalibration({m:G[m] for m in profiles},{m:M[m] for m in profiles},{m:bounds[m] for m in profiles}).validate(cat);policy=ReplicaPlanningPolicy(cat,'empirical','empirical',cal)
def canonical(x,lam):
 return solve_canonical_routing({m:x.get(m,0) for m in cat},Workload(lam),cat,rho,replica_bounds={m:bounds[m] for m in cat},capacity_mode='empirical',calibration=cal)
valid_A=[l for l in intensities if all(next(v['feasible_count'] for v in l['mixtures'][representative]['budget_feasibility'] if v['budget_mib']==B)>0 for B in budgets)];assert valid_A
Alevel=valid_A[-1]['id'];intermediate=min([v for v in final_memory if v['budget_mib'] not in (bmin,bsg,PHYSICAL)],key=lambda v:(abs(v['legacy_8f1a_feasible_fraction']-.5),v['budget_mib']));Bbudgets=[bmin,intermediate['budget_mib'],PHYSICAL]
methods=['static_v2','Generalist_centric','RC_SG'];cells_by_key={}
for B in budgets:
 key=(representative,Alevel,B);cells_by_key.setdefault(key,[]).append('A')
for B in Bbudgets:
 for l in intensities:cells_by_key.setdefault((representative,l['id'],B),[]).append('B')
method_rows=[];cells=[];milpcalls=0
for j,((mixture,level,B),experiments) in enumerate(sorted(cells_by_key.items(),key=lambda kv:(kv[0][2],kv[0][1])),1):
 cellid=f'C{j:02d}';il=next(l for l in intensities if l['id']==level);lam=il['mixtures'][mixture]['lambda_rps'];alpha=il['mixtures'][mixture]['alpha'];beta=il['beta'];available=[s for s in pops[B] if limits[mixture][s['state_id']]['alpha_limit']>=alpha-TOL*max(1,alpha)];statuses={}
 cell={'cell_id':cellid,'experiments':experiments,'mixture':mixture,'intensity_level':level,'alpha':alpha,'beta':beta,'lambda_rps':lam,'memory_budget_mib':B,'planning_budget_input_mib':B/factor,'physical_device_mib':PHYSICAL,'capacity_feasible_space_count':len(available),'workload_hash':hashlib.sha256(json.dumps(lam,sort_keys=True).encode()).hexdigest(),'preflight':{}}
 # Exact existing cold seed; no MethodV2 object or search loop is constructed.
 hardware=Hardware(B/factor,'Same physical hardware; resource budget is an experiment input',ref(OUT/'final_memory_budgets.json'));effective_bounds,_=policy.bounds(hardware)
 seed=MilpDeploymentSeeder().solve(cat,Workload(lam),hardware,rho,capacity_mode='empirical',memory_mode='empirical',calibration=cal);milpcalls+=seed.milp_call_count
 assert seed.feasible==bool(available),(cellid,'seed/enumeration disagreement')
 if seed.feasible:
  assert not validate_seed(seed,cat,Workload(lam),hardware,rho,effective_bounds,planning_policy=policy)
  x={m:seed.deployment.get(m,0) for m in ids};assert safe_memory(x)<=B+TOL;route=canonical(x,lam);assert route is not None
  status='READY';detail={'deployment':x,'M_safe_mib':safe_memory(x),'seed_planning_memory_mib':policy.deployment_memory(seed.deployment),'canonical_routing_rps':{f'{t}::{m}':r for (t,m),r in route.routing.items()},'canonical_max_pressure':route.max_pressure,'effective_resource_bounds':effective_bounds}
 else:status='PLANNING_CAPACITY_INFEASIBLE';detail={'deployment':None,'reason':'No quality/capability/memory/capacity feasible vector at this frozen cell; no GPU or substitute.'}
 cell['preflight']['static_v2']={'status':status,**detail}
 # Frozen Generalist-centric B1 has fixed task paths, no budget repair.
 gcmemory=safe_memory(gcx);gcmax=assigned_limit(gcx,gcpath,base[mixture]);gcstatus='MEMORY_INFEASIBLE' if gcmemory>B else 'PLANNING_CAPACITY_INFEASIBLE' if alpha>gcmax+TOL*max(1,gcmax) else 'READY'
 cell['preflight']['Generalist_centric']={'status':gcstatus,'deployment':gcx,'M_safe_mib':gcmemory,'fixed_routing_rps':{f'{t}::{gcpath[t]}':lam[t] for t in tasks},'fixed_routing_alpha_limit':gcmax,'source':ref(R/'module8d_baseline_freeze/baseline_deployments.json','B1')}
 rc=rc_sg(lam,B)
 if rc['deployment'] is not None:
  route=canonical(rc['deployment'],lam);rc['status']='READY' if route else 'PLANNING_CAPACITY_INFEASIBLE'
  rc['canonical_routing_rps']={f'{t}::{m}':r for (t,m),r in route.routing.items()} if route else None;rc['canonical_max_pressure']=route.max_pressure if route else None
 cell['preflight']['RC_SG']=rc
 for method in methods:
  p=cell['preflight'][method];method_rows.append({'cell_id':cellid,'method':method,'status':p['status'],'deployment':p.get('deployment'),'M_safe_mib':p.get('M_safe_mib'),'memory_budget_mib':B,'intensity':level,'mixture':mixture})
 cells.append(cell)
 print(f'{cellid} {mixture} {level} {B:.6f}: '+', '.join(f'{m}={cell["preflight"][m]["status"]}' for m in methods),flush=True)
settings=comparison['measurement_settings'];shared_protocol={'methods':methods,'search_config':comparison['search_config'],'v2_max_unique_runtime_states':6,'search_repetition_seeds':comparison['search_seeds'],'confirmation_seeds':comparison['confirmation_seeds'],'v2_confirmations':'2 seed + 2 final; keep independent role samples even if identical vector','fixed_baseline_runs':'3 timed repetitions + 2 fixed-role confirmations','measurement_settings':settings,'objective':comparison['runtime_objective'],'runtime_performance_used_for_design':False,'unsafe_or_capacity_infeasible_cells':'record structural outcome, zero GPU runs; no replacement or workload/budget tuning','baseline_routing_note':'RC-SG and static v2 use unchanged canonical routine; frozen Generalist-centric retains its baseline fixed routing; no routing revision','history_reuse':'Exact identity, exact lambda/routing, same model/environment/hardware, seed and phase contract required. No approximate alpha matches. Existing B2 reference preserved report-only.'}
write('campaign_cells.json',{'frozen':True,'unique_cell_count':len(cells),'cells':cells,'protocol':shared_protocol,'cold_seed_MILP_calls':milpcalls,'Proposed_search_calls':0,'GPU_runs':0})
write('baseline_deployments.json',{'frozen':True,'RC_SG_spec':ref(OUT/'resource_constrained_sg_spec.json'),'cell_method_states':method_rows,'baseline_source_preserved':ref(R/'module8d_baseline_freeze/baseline_deployments.json'),'old_B2_results_deleted':False})
Acell=[c for c in cells if 'A' in c['experiments']];Bcell=[c for c in cells if 'B' in c['experiments']]
write('memory_sweep_design.json',{'experiment':'A','frozen':True,'representative_mixture':representative,'composition_medoid_distance':dist,'intensity_level':Alevel,'alpha':next(l for l in intensities if l['id']==Alevel)['mixtures'][representative]['alpha'],'memory_budgets_mib':budgets,'cells':len(Acell),'cell_ids':[c['cell_id'] for c in Acell],'methods':methods,'selection_rule':ref(OUT/'intensity_selection_rule.json'),'answers':['confirmed latency','N_G/N_S','deployment composition','baseline budget/capacity feasibility'],'full_factorial':False})
write('intensity_sweep_design.json',{'experiment':'B','frozen':True,'representative_mixture':representative,'budget_roles':{'tight':bmin,'intermediate':intermediate['budget_mib'],'loose':PHYSICAL},'levels':[l['id'] for l in intensities],'cells':len(Bcell),'cell_ids':[c['cell_id'] for c in Bcell],'methods':methods,'selection_rule':ref(OUT/'intensity_selection_rule.json'),'answers':['composition','canonical and observed bottleneck','replica allocation','confirmed latency'],'full_factorial':False})
hypotheses={'frozen':True,'primary_metric':'same existing equal-repetition arrival-weighted mean E2E from independent final confirmations; weighted task P95 and frozen stability classification secondary','acceptance_rule':'Existing NoiseAwareRuntimeObjective(1 ms, 5%) and frozen stability priorities; no new optimized significance threshold','H1':{'claim':'Memory constraints systematically change deployment composition.','test':'On Experiment A compare confirmed static-v2 full vectors and N_G/N_S across ordered budgets. Report direction, distinct vectors and matched planning gates. Runtime-best observed states are not claimed globally optimal.'},'H2':{'claim':'Static v2 approaches SG/RC-SG in loose resources.','test':'At physical endpoint compare independent confirmed v2 and RC-SG through unchanged noise-aware objective; close means same stability and absolute mean gap <= max(1 ms, .05*reference mean). Old B2 is provenance/reference only; no cross-workload historical latency substitution.'},'H3':{'claim':'Simple unconstrained per-task Specialist selection is insufficient under tight memory.','test':'Report old SG memory/capacity infeasibility at tight budget; compare budget-compliant RC-SG and v2 feasibility, stability and noise-aware confirmed metrics. Infer runtime superiority only from actual valid matched confirmations.'},'H4':{'claim':'Offered-load increase changes composition or replica allocation.','test':'Experiment B frozen levels at each fixed budget; report all vector and count changes and baseline structural failures. Null changes are legitimate.'},'H5':{'claim':'Activated replicas are consistent with measured empirical scaling.','test':'For each runtime selected x2, compare its paired planning interval and empirical g(2), report task transfer caveat and observed throughput/pressure; no linear doubling assumption. No activation => not observed, not force a new level.'},'not_all_hypotheses_required':True,'post_runtime_grid_changes_allowed':False,'infeasible_cells':'Keep denominators and reason codes; no latency imputation, workload replacement, reruns, or budget growth.'}
write('success_hypotheses.json',hypotheses)
# Conservative caps and known structural skips; future search states are unknown.
N=len(cells);v2ready=sum(c['preflight']['static_v2']['status']=='READY' for c in cells);gcready=sum(c['preflight']['Generalist_centric']['status']=='READY' for c in cells);rcready=sum(c['preflight']['RC_SG']['status']=='READY' for c in cells);active_states=6*v2ready+gcready+rcready;active_runs=22*v2ready+5*(gcready+rcready);full_runs=32*N
old_lambdas=[w['lambda_rps'] for w in held['workloads']];legacy_exact_lambda_hits=sum(c['lambda_rps']==lam for c in cells for lam in old_lambdas)
assert legacy_exact_lambda_hits==0,'Unexpected exact historical workload match: audit full identity before estimation.'
estimate={'frozen':True,'memory_sweep_cells':len(Acell),'intensity_sweep_cells':len(Bcell),'overlap_cells':len(Acell)+len(Bcell)-N,'unique_cells':N,'methods':methods,'logical_method_cells_before_structural_skip':N*3,'hypothetical_full_factorial_cells':3*len(budgets)*3,'hypothetical_full_factorial_method_cells':3*len(budgets)*3*3,'per_cell_worst_case':{'logical_search_or_fixed_evaluations':8,'distinct_deployment_routing_states':8,'timed_runs':32,'v2':'<=6 states ×3 search reps +2 seed +2 final confirmations =22','each_fixed_baseline':'3 timed reps +2 fixed confirmations =5'},'worst_case_logical_evaluations':8*N,'worst_case_physical_states':8*N,'worst_case_timed_runs':full_runs,'structural_ready_method_cells':{'static_v2':v2ready,'Generalist_centric':gcready,'RC_SG':rcready},'structural_skipped_method_cells':N*3-v2ready-gcready-rcready,'planning_adjusted_logical_evaluation_cap':active_states,'planning_adjusted_physical_state_cap':active_states,'planning_adjusted_timed_run_cap':active_runs,'estimated_new_runs_conservative':active_runs,'historical_exact_lambda_reuse_opportunities':legacy_exact_lambda_hits,'confirmed_reused_runs':0,'intra_campaign_reuse':'Identical deployment+canonical route+lambda+context+seed+phase may share search runs across static v2/RC-SG only after independently committed choices. Shared runs still charge logical budgets. No seed-final or search-confirmation sharing. Future v2 states unknown; no speculative reuse discount.','experiment_A_B_overlap':'One physical execution per identical campaign cell; both figures reference the same sample, not independent evidence.','measurement_seconds':settings['measurement_seconds'],'warmup_seconds':settings['warmup_seconds'],'estimated_timed_window_hours':active_runs*settings['measurement_seconds']/3600,'estimated_warmup_plus_timed_hours':active_runs*(settings['measurement_seconds']+settings['warmup_seconds'])/3600,'startup_drain_release_overhead_included_in_time_estimate':False,'hard_physical_attempt_cap':active_runs,'retries_allowed':0,'this_stage_new_GPU_runs':0,'second_hardware_allowed':False}
write('gpu_budget_estimate.json',estimate)
# Close or open the design gate based on concrete CPU evidence, never on a winner.
checks={'8F1A_PASS':True,'frozen_input_hashes_unchanged':all(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in sources.values()),'composition_unchanged':True,'selection_rules_saved_before_capacity_results':True,'three_distinct_normalized_levels':len(set(chosen_beta.values()))==3,'final_memory_count_4_to_6':4<=len(budgets)<=6,'all_nonphysical_budgets_empirical':all(B in bps for B in budgets if B!=PHYSICAL),'fraction_separation_5pp':all(fraction(b)-fraction(a)>=.05 for a,b in zip(budgets,budgets[1:])),'safe_memory_and_capacity_preflight_complete':True,'static_v2_ready_memory_sweep_all_cells':all(c['preflight']['static_v2']['status']=='READY' for c in Acell),'RC_SG_budget_compliant':all(r['M_safe_mib'] is None or r['M_safe_mib']<=r['memory_budget_mib'] for r in method_rows if r['method']=='RC_SG'),'replica_or_model_capacity_tradeoff_observed_in_planning_HIGH':any(v['any_capacity_useful_replica'] or v['any_capacity_useful_model_tradeoff'] for m in base for v in intensities[-1]['mixtures'][m]['budget_feasibility']),'no_new_runtime':True,'bounded_campaign':active_runs<=384}
ready=all(checks.values())
gate={'module':'8F-1B','RESOURCE_STATIC_GPU_CAMPAIGN_READY':ready,'meaning':'Planning grid and budget are frozen and preflighted; this stage does NOT authorize/execute GPU runtime.','HIGH_replica_usefulness_diagnostic':any(v['any_capacity_useful_replica'] for m in base for v in intensities[-1]['mixtures'][m]['budget_feasibility']),'HIGH_model_tradeoff_diagnostic':any(v['any_capacity_useful_model_tradeoff'] for m in base for v in intensities[-1]['mixtures'][m]['budget_feasibility']),'gate_interpretation':'HIGH requires replica OR model trade-off, as requested. A missing individual replica mechanism is a reported result, not a reason to alter alpha or close an otherwise valid campaign.','checks':checks,'physical_gpu_capacity_mib':PHYSICAL,'epsilon_mem':safety['epsilon_mem'],'rho_max':rho,'unchanged_methods':['static_v2','canonical_routing','quality_policy','candidate_pool_v2.1','service_demand','empirical_replica_capacity','evidence_bounds','8F1A_memory_catalog_and_epsilon'],'new_GPU_runs':0,'Proposed_search_calls':0,'execution_started':False,'GPU_execution_authorized_in_this_stage':False,'next_execution_requirements':['Recheck frozen files, hardware/environment/weights/adapters and precision against original manifests.','Revalidate equivalent planning budget B/safety_factor and safe memory for every proposed/baseline state.','Use frozen settings, seeds, state/run caps and phase-aware sharing; no retries or second hardware.','Retain every structurally infeasible cell as an outcome; do not replace it.'],'frozen_source_hashes':dict(sources),'campaign_manifest_source':ref(OUT/'campaign_cells.json')}
write('resource_campaign_gate.json',gate)
lines=['# Module 8F-1B — Workload Intensity Breakpoint Audit + Final Resource Grid Freeze','',f'完成纯 CPU planning 审计。RESOURCE_STATIC_GPU_CAMPAIGN_READY={ready}。本阶段 GPU runs=0、Proposed search calls=0；static v2/canonical routing/quality/pool/s0/g/bounds/epsilon/memory 均未修改。','', '1. 固定 mixture composition，lambda_base 对应旧冻结 Qwen-reference L=1。旧 .4/.72 intensity 只保存 provenance。alpha_max 随显存预算如下（最终预算，MiB）：','', '| Memory MiB | mix_01 alpha_max | mix_02 alpha_max | mix_03 alpha_max |','|---:|---:|---:|---:|']
lines += [f'| {B:.9f} | '+ ' | '.join(f'{boundaries[m][str(B)]["alpha_max"]:.9f}' for m in base)+' |' for B in budgets]
lines += ['', '2. 所有 deployment 的 capacity breakpoints、边界 vector、bottleneck models/tasks 已保存。每个 deployment 在 alpha_limit 本身仍可行，紧随其后失效；intensity_breakpoint_curve.csv 同时保存 inclusive/right-limit count。质量 admission 下 Generalist-only 和 Specialist-only 均不可覆盖全部六任务，不能伪造 regime transition。','3. x=2 memory-feasible 不依赖 alpha；capacity-useful 为同一向量 x=1 capacity boundary 之后、x=2 boundary 以内的 (alpha_1,alpha_2] 区间。全预算空间是否必须复制另存 global necessity；没有 runtime 复制收益推断。','4. Replica activation infimum（开放下界）；逐预算/mixture 完整 intervals 及 global necessity 见 replica_activation_thresholds.json：','', '| Mixture | Memory MiB | Qwen | Whisper-small | ResNet50 |','|---|---:|---:|---:|---:|']
for m in base:
 for B in budgets:
  values=[replica_summary[m][str(B)][n]['activation_infimum_alpha'] for n in ('qwen','whisper-small','resnet50')];lines.append(f'| {m} | {B:.9f} | '+' | '.join(f'{v:.9f}' if v is not None else '无 memory-feasible/useful x2' for v in values)+' |')
lines += ['', '5. 最终 intensity 使用 beta=alpha/alpha_max(w,49140)，每个 beta 都来自真实 capacity breakpoint。LOW/MODERATE 保留宽 feasibility 且全部预算至少一解；PRESSURED 保留至少 1/3 多数预算空间；HIGH 取多数预算仍至少两个解的最大断点。规则在 LP 结果前保存，不针对特定模型调整。','', '| Level | beta | mix_01 alpha | mix_02 alpha | mix_03 alpha |','|---|---:|---:|---:|---:|']
lines += [f'| {l["id"]} {l["regime"]} | {l["beta"]:.12f} | '+' | '.join(f'{l["mixtures"][m]["alpha"]:.9f}' for m in base)+' |' for l in intensities]
lines += ['', '6. B2/B3 按 5 percentage-point information separation 机械去重；保留 one-third B2，将 shifted B3 移到下一个对两侧 surviving neighbors 都分离至少 5pp 的 empirical breakpoint。minimum/SG/physical anchors 全保留。','7–8. 最终 memory budgets/resource regimes：','', '| ID | MiB | GiB | 8F1A fraction | SG memory feasible | x2 memory-feasible | Regime |','|---|---:|---:|---:|---|---|---|']
lines += [f'| {v["id"]} | {v["budget_mib"]:.9f} | {v["budget_gib"]:.9f} | {v["legacy_8f1a_feasible_fraction"]:.6%} | {v["SG_memory_feasible"]} | {", ".join(v["x2_memory_feasible_models"]) or "none"} | {v["semantic_regime"]} |' for v in final_memory]
lines += ['', '物理端点 49140 MiB 明确保留 empirical-breakpoint exception；所有其他点均为 empirical sum breakpoint。上表 fraction 使用冻结 8F1A 口径，新的 intensity-conditioned fractions/counts 完整保存在 final_memory_budgets.json。','HIGH 在部分预算存在单模型增加可解决容量失败的 trade-off，但没有 paired x2 capacity-useful 状态；ResNet50 在全部审计 mixture/budget 内没有 paired capacity gain。这些都是保留的 planning 结果，不调整 intensity 来强制复制。','9. RC-SG：遍历单 Generalist base，以最小 s0 Specialist 修复 uncovered tasks；在预算可行 bases 中按总 lambda×s0、safe memory、Generalist ID、vector 确定基底。每步添加能放入预算、对当前 assignment 有正改善的 Specialist，按总节省 worker-demand / 新增 safe memory 排序，ties 按更小 memory/model ID。不复制、不驱逐、不使用 g 或 runtime 做 selection；构造后调用 unchanged canonical routing，容量不满足只标记不可行、不修补。Generalist-centric 保留旧 B1，越预算时不运行。','10. Memory Sweep: '+f'{len(Acell)} cells，composition medoid={representative}，固定 {Alevel}，扫描所有 {len(budgets)} budgets。','11. Intensity Sweep: '+f'{len(Bcell)} cells，同一 mixture，3 levels × tight/intermediate/loose；与 A 重叠 {len(Acell)+len(Bcell)-N} cells，去重后总 {N} cells ×3 methods。','12. GPU budget：'+f'未扣结构不可行的 worst-case logical evaluations/physical states={8*N}/{8*N}，timed runs={full_runs}。CPU preflight 后 method cells ready: v2={v2ready}, Generalist-centric={gcready}, RC-SG={rcready}；结构不可行跳过 {N*3-v2ready-gcready-rcready} 个 method cells。保守 estimated new runs={active_runs}，physical states≤{active_states}；timed-window hours≈{active_runs*settings["measurement_seconds"]/3600:.3f}（未含启动/释放 overhead）。没有历史 exact-lambda reuse，未来 search reuse 不预先打折。', '13. 全部 mixture/memory/intensity/grid/baseline/hypotheses/预算选择均在任何新 runtime 前冻结；没有查看 v2/B2 performance。',f'14. RESOURCE_STATIC_GPU_CAMPAIGN_READY={ready}。仅设计与 CPU preflight ready；本阶段执行授权为 false，GPU runs=0。','', '可复现：`/root/miniconda3/envs/gsdeploy/bin/python results/module8f1b_intensity_grid_freeze/run_audit.py`。','', '完成后停止；不启动 resource sweep、不修改 method/routing、不进入 dynamic、不使用第二硬件。']
(OUT/'module8f1b_summary.md').write_text('\n'.join(lines)+'\n')
assert checks['frozen_input_hashes_unchanged']
write('artifact_hashes.json',{p.name:ref(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in ('artifact_hashes.json','execution.log','verification.log')})
print(json.dumps({'ready':ready,'memory_budgets':budgets,'betas':chosen_beta,'alpha_reference':alpha_ref,'A_cells':len(Acell),'B_cells':len(Bcell),'unique_cells':N,'estimated_new_runs':active_runs,'checks':checks}),flush=True)
