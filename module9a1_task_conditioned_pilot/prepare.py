import csv,hashlib,json,math
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/module9a1_task_conditioned_pilot';DEV=ROOT/'results/module8f2_resource_static'
def load(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(n,d):
 p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def csvsave(n,rows,cols=None):
 with (OUT/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols or list(rows[0]));w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def now():return datetime.now(timezone.utc).isoformat()
def main():
 if (OUT/'pilot_selection.json').exists():raise RuntimeError('Selection already frozen; do not regenerate')
 assert load(DEV/'final_closeout.json')['STATIC_MAIN_EXPERIMENTS_COMPLETE']
 inv=load(DEV/'artifact_hashes.json');bad=[r['path'] for r in inv.values() if sha(r['path'])!=r['sha256']]
 assert not bad,bad
 protected=load(DEV/'execution_freeze.json');refs=protected['protected_references']+list(protected['execution_sources'].values())
 assert all(sha(r['path'])==r['sha256'] for r in refs)
 from module8c_final_real_search.common import inputs
 catalog,hw,roles,cal=inputs();v2=load(DEV/'v2_results.json')['cells'];pool=load(ROOT/'results/module8f1b_intensity_grid_freeze/campaign_cells.json')['cells'];cells={c['cell_id']:c for c in pool}
 counts=Counter(m for c in v2 for m,n in c['deployment'].items() if n and roles[m]=='generalist');g1=sorted(counts,key=lambda m:(-counts[m],m))[0]
 histories=[];replacement=Counter();evaluated_replacement=Counter()
 for c in v2:
  trace=load(DEV/c['cell_id']/'static_v2/search_trace.json')
  for it in trace['iterations']:
   for cand in it['candidates']:
    x=cand['mutation']['deployment'];typ=cand['mutation'].get('kind',cand['mutation'].get('type',''))
    row={'cell_id':c['cell_id'],'iteration':it['iteration'],'mutation':cand['mutation'],'status':cand.get('status'),'reason':cand.get('reason'),'canonical':cand.get('canonical'),'runtime_evaluated':cand.get('runtime') is not None};histories.append(row)
    # Replacement evidence is a candidate that removes the modal Generalist
    # and introduces a different Generalist, actually evaluated on GPU.
    if cand.get('runtime') is not None and not x.get(g1):
     for m,n in x.items():
      if n and roles[m]=='generalist' and m!=g1:
       evaluated_replacement[m]+=1
       if cand['status']=='ACCEPTED':replacement[m]+=1
 g2=sorted(evaluated_replacement,key=lambda m:(-replacement[m],-evaluated_replacement[m],m))[0];generals=[g1,g2]
 bank=load(DEV/'physical_measurements.json')['runs'];tasksets={m:set() for m in generals};failure=[];raws={}
 for r in bank:
  raw=load(r['raw_path']);raws[r['physical_raw_id']]=raw;config=r['identity']['config'];route=config['routing_rps'];lam=config['workload_rps'];demand={m:{k:rate*catalog[m].service_time_ms[k]/1000 for k,rr in route.items() if (rate:=rr.get(m,0))>0} for m in generals if r['deployment'].get(m)}
  for m in generals:
   for k,rr in route.items():
    if rr.get(m,0)>0 and lam[k]>0:
     assert k in catalog[m].supported_tasks;tasksets[m].add(k)
  metric=raw['system_metrics'];records=[q for q in raw['request_records'] if q['status']=='completed'];tot=sum(lam.values());weighted=sum(lam[k]/tot*np.mean([q['e2e_latency_ms'] for q in records if q['task']==k]) for k in lam if lam[k]>0)
  per_task_p95={k:float(np.quantile([q['e2e_latency_ms'] for q in records if q['task']==k],.95)) for k in lam if lam[k]>0}
  failure.append({'physical_raw_id':r['physical_raw_id'],'cell_id':r['workload'],'method':r['method'],'phase':r['phase'],'role':r['role'],'seed':r['repetition_seed'],'runtime_label':r['stability'],'deployment':r['deployment'],'routing_rps':route,'workload_lambda':lam,'generalists':[m for m,n in r['deployment'].items() if n and roles[m]=='generalist'],'generalist_task_demand_worker_seconds_per_second':demand,'co_resident_specialists':[m for m,n in r['deployment'].items() if n and roles[m]=='specialist'],'mean_e2e_ms':float(weighted),'weighted_p95_ms':sum(lam[k]/tot*v for k,v in per_task_p95.items()),'raw_path':r['raw_path'],'raw_sha256':r['raw_sha256'],'state_identity_sha256':r['state_identity_sha256'],'evidence_use':'DEVELOPMENT_ONLY_NOT_V3_HELDOUT'})
 # One observation per frozen benchmark cell: official static-v2 final route.
 # r[k,m] is routed request rate, so demand=r*s0=lambda*fraction*s0.
 score=defaultdict(float);detail=[]
 for c in v2:
  for key,rate in c['routing'].items():
   k,m=key.split('::')
   if rate>0 and roles[m]=='specialist':
    d=rate*catalog[m].service_time_ms[k]/1000;score[m]+=d;detail.append({'cell_id':c['cell_id'],'model':m,'task':k,'routed_lambda':rate,'s0_ms':catalog[m].service_time_ms[k],'worker_demand':d})
 ranking=sorted(score,key=lambda m:(-score[m],m));co=ranking[:2]
 selection={'frozen_at_utc':now(),'evidence_use':'DEVELOPMENT_ONLY; 8F-2 is permanently ineligible for any future v3 held-out final validation','generalists':generals,'selection_rule':'First: maximum official static-v2 final Generalist occurrence across 12 cells, tie model ID. Second: other Generalist actually evaluated as replacement for first, highest accepted replacements, then evaluated replacements, then model ID. No runtime latency used.','final_generalist_counts':dict(counts),'accepted_replacement_counts':dict(replacement),'evaluated_replacement_counts':dict(evaluated_replacement),'tasks':{m:sorted(tasksets[m]) for m in generals},'task_rule':'Union of every active, quality-admitted positive Generalist route in all frozen 8F-2 physical states, including candidates and confirmations; no task omission.','co_models':co,'co_model_rule':'Top-2 cumulative routed worker demand sum_cell,sum_task r[k,m]*frozen_s0[k,m] over the 12 official static-v2 final routes, one route per cell; tie model ID. Equivalent lambda*route_fraction*s0, not full lambda for unrouted paths.','co_ranking':[{'model':m,'cumulative_worker_demand':score[m]} for m in ranking],'co_demand_components':detail,'co_tasks':{m:sorted({r['task'] for r in detail if r['model']==m}) for m in co},'memory_budgets_modified':False,'workloads_modified':False,'quality_modified':False,'new_GPU_results_before_selection':0}
 csvsave('failure_map.csv',failure);save('candidate_history_freeze.json',{'candidates':histories,'source_8F2_inventory_sha256':sha(DEV/'artifact_hashes.json')});save('pilot_selection.json',selection)
 save('development_evidence_freeze.json',{'frozen_at_utc':now(),'all_8F2_artifacts':inv,'all_prior_protected_references':refs,'frozen_8F2_inventory_sha256':sha(DEV/'artifact_hashes.json'),'workloads':pool,'deployments_and_routings':{'static_v2':v2,'RC_SG':load(DEV/'rc_sg_results.json')['cells']},'future_v3_heldout_validation_eligible':False,'new_GPU_runs':0})
 groups=defaultdict(list)
 for r in bank:
  if r['valid'] and r['stability']=='steady':groups[r['state_identity_sha256']].append(r)
 audits=[];rng=np.random.default_rng(902001);null_samples=10000;floors=[]
 for state,rr in sorted(groups.items()):
  if len(rr)<2:continue
  # Report all latency CVs. Gate floor is the maximum null absolute Delta-S
  # among every selected Generalist/task path with >=2 steady run means.
  paths=defaultdict(list);e2e=[]
  for r in rr:
   raw=raws[r['physical_raw_id']];paths_run=defaultdict(list)
   for q in raw['request_records']:
    if q['status']=='completed':paths_run[(q['model_family'],q['task'])].append(q['worker_latency_ms'])
   e2e.append(next(f['mean_e2e_ms'] for f in failure if f['physical_raw_id']==r['physical_raw_id']))
   for path,values in paths_run.items():paths[path].append({'raw_id':r['physical_raw_id'],'mean_ms':float(np.mean(values)),'n':len(values)})
  for path,values in sorted(paths.items()):
   if len(values)<2:continue
   means=np.array([v['mean_ms'] for v in values]);draw=rng.choice(means,size=(null_samples,4),replace=True);delta=draw[:,0]/draw[:,1]-draw[:,2]/draw[:,3];floor=float(np.quantile(np.abs(delta),.95));boot=rng.choice(means,size=(null_samples,len(means)),replace=True).mean(axis=1)
   row={'state_identity_sha256':state,'model':path[0],'task':path[1],'n_steady_repetitions':len(means),'run_means_ms':means.tolist(),'latency_cv':float(np.std(means,ddof=1)/np.mean(means)),'normalized_slowdown_cv':float(np.std(means/np.mean(means),ddof=1)),'maximum_observed_same_state_ratio_gap':float(np.max(means)/np.min(means)-1),'bootstrap_mean_95CI_ms':np.quantile(boot,[.025,.975]).tolist(),'bootstrap_null_abs_deltaS_p95':floor,'included_in_gate_floor':path[0] in generals and path[1] in tasksets[path[0]],'raw_evidence':values};audits.append(row)
   if row['included_in_gate_floor']:floors.append(floor)
 assert floors
 noise={'frozen_at_utc':now(),'historical_noise_floor':max(floors),'unit':'absolute slowdown-ratio difference (Delta S), dimensionless','definition':'Maximum across all selected Generalist/task path same-state steady groups of bootstrap Q95(|L_a/L_b-L_c/L_d|), where a,b,c,d independently resample empirical repeated-run mean worker latencies. No constant margin; conservative across paths and states.','bootstrap_seed':902001,'bootstrap_resamples':null_samples,'steady_raw_runs':sum(len(r) for r in groups.values()),'same_state_groups_with_repeats':sum(len(r)>=2 for r in groups.values()),'path_audits':audits,'nonsteady_runs_excluded_from_noise_only':sum(r['stability']!='steady' for r in bank),'noise_scope_caveat':'Historical repeats came from deployment co-residency, not this controlled pair protocol; conservative transfer floor. Gate requires an independent new pair CI as well.','new_GPU_runs':0}
 save('noise_audit.json',noise)
 context=load(DEV/'contexts/C01.json');settings=load(ROOT/'results/module8e2_heldout_freeze/method_comparison_freeze.json');items=[json.loads(line) for line in Path(context['input_pool']).read_text().splitlines() if line.strip()];by={k:[i for i in items if i['task_type']==k] for k in set(i['task_type'] for i in items)};seeds=settings['search_seeds'];subsets={}
 for seed in seeds:
  subsets[str(seed)]={}
  for task in sorted(set().union(*tasksets.values(),*(set(selection['co_tasks'][m]) for m in co))):
   order=np.random.default_rng(seed).permutation(len(by[task]))[:30];subsets[str(seed)][task]=[by[task][int(i)] for i in order]
 protocol={'frozen_at_utc':now(),'characterization_only':True,'not_new_benchmark_workload_or_resource_grid':True,'measurement_source':'Existing active-pair primitive: real native workers, warmed task paths, continuously replenished fixed co-backlog, exact worker execution interval overlap. Thin pilot transport adapter supports native Phi4mm VQA and fixed windows; frozen handlers and benchmark routing unchanged.','existing_active_pair_source':{'path':str(ROOT/'joint_deploy/run_active_pair_profile.py'),'sha256':sha(ROOT/'joint_deploy/run_active_pair_profile.py')},'generalists':generals,'tasks':selection['tasks'],'co_models':co,'co_tasks':selection['co_tasks'],'repetition_seeds':seeds,'target_sample_count':30,'warmup_seconds':settings['measurement_settings']['warmup_seconds'],'measurement_seconds':settings['measurement_settings']['measurement_seconds'],'arrival_process':'deterministic_periodic; target sample i at i*210/30 seconds, identical schedule for every task','target_schedule_seconds':[i*210/30 for i in range(30)],'co_arrival_process':'existing closed-loop saturation; sorted canonical co-task cycle, backlog 64, refill on completion, no idle gap','co_backlog':64,'co_gap_ms':0,'minimum_overlap_ratio':.9,'startup_timeout_seconds':300,'drain_timeout_seconds':60,'retries':0,'input_pool':context['input_pool'],'input_pool_sha256':sha(context['input_pool']),'input_subsets':subsets,'registry':{m:context['registry'][m] for m in generals+co},'standalone_ground_truth':'Fresh unprofiled task-path worker latencies, matched task/input/repetition protocol. Formal frozen s0 catalog is retained and never refit; profiled durations never feed real slowdown.','signature_profile_samples':3,'signature_profile_repetitions':3,'bootstrap_resamples':10000,'bootstrap_seed':902002,'ci_level':.95,'statistical_rule':'Hierarchical bootstrap of independent repetition blocks and matched input indices within blocks; DeltaS=S(task_a)-S(task_b). DETECTED iff 95% CI excludes zero AND abs(point DeltaS)>frozen historical_noise_floor. No post-result threshold change.','kernel_structure_rule':'Distributional distances use kernel class JSD, occupancy W1, exact resource-configuration JSD, transition-bigram JSD and count ratio. Stable structural difference requires between-task bootstrap 95% lower JSD bound > empirical within-task repeat-bootstrap Q95 JSD floor and repeated profiles; never kernel-duration weighted. Missing resource fields remain missing.','correlation_rule':'Spearman on unique Generalist/task-pair signature distance versus absolute DeltaS, co-model observations explicitly clustered. Fewer than 3 independent task-pair contrasts => INSUFFICIENT_FOR_CORRELATION.','gate_rule':{'STRONG_SUPPORT':'At least one Generalist has >=2 selected tasks and >=1 fixed co-model contrast satisfying DETECTED, and the same task contrast has stable kernel structural difference.','PARTIAL':'Only structural or only interference stable evidence, or both without a same-contrast match; stop and audit.','NOT_SUPPORTED':'Neither stable structural nor interference differences; stop this direction.','missing_evidence':'Missing profiler/runtime evidence precludes STRONG_SUPPORT; explicitly incomplete, never fabricate a negative scientific result.'},'historical_noise_floor':noise['historical_noise_floor'],'max_unprofiled_runs':len(generals)*len(tasksets[g1])*len(seeds)*(1+len(co)),'new_GPU_runs_before_freeze':0}
 save('measurement_protocol_freeze.json',protocol);save('pilot_state.json',{'status':'PREPARED','GPU_runs':0});print(json.dumps({'selection':{k:selection[k] for k in ['generalists','final_generalist_counts','accepted_replacement_counts','evaluated_replacement_counts','tasks','co_models','co_ranking']},'historical_noise_floor':noise['historical_noise_floor'],'8F2_files_verified':len(inv),'max_unprofiled_runs':protocol['max_unprofiled_runs']},ensure_ascii=False))
if __name__=='__main__':main()
