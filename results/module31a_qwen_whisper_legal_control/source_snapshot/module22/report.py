"""Real request utility only; censored cohorts cannot produce a sortable U."""
from .common import *
from module20.optimizer import components
import statistics,numpy as np
from types import SimpleNamespace
from performance_model.local_calibration.stability_v2 import reconstruct,classify,DEFAULT_THRESHOLDS

def analyze_base(raw,f):
 s=raw['slot'];d=read(f['catalog']['path']);refs=read(f['utility_references']['path'])['latency_reference_ms'];weights=read(f['utility_definition']['path'])['weights'];ph=raw['phases'].get('measurement');base={'run_id':s['run_id'],'stage':s['stage'],'arm':s['arm'],'repetition':s['repetition'],'seed':s['seed'],'deployment':s['deployment'],'routing':s['routing'],'frozen_offered_QPS':s['lambda_rps'],'CPU_predicted_U':s['CPU_predicted_U'],'raw':ref(OUT/'raw_runs'/(s['run_id']+'.json'))}
 if not ph:return {**base,'U':None,'eligible':False,'system_state':'NO_FORMAL_MEASUREMENT','worker_states':{},'hard_unstable':True,'memory_passed':False,'errors':raw['errors']},[],[]
 rs=ph['request_records'];w=ph['window'];done=[r for r in rs if r['status']=='completed'];within=sum(r['completion_time_ns']<=w['end_time_ns'] for r in done);rates=s['lambda_rps'];taskrows=[];taskstats={};workers=[]
 for t,q in rates.items():
  cohort=[r for r in rs if r['task']==t];dd=[r for r in cohort if r['status']=='completed'];lat=[r['e2e_latency_ms'] for r in dd];responses=[r['worker_response'] for r in dd];scored=[v['correct'] for v in responses if isinstance(v.get('correct'),bool)];words=sum(v.get('reference_words',0) for v in responses)
  tr={**{k:base[k] for k in ['run_id','stage','arm','repetition']},'task':t,'offered_QPS':q,'offered':len(cohort),'completed':len(dd),'within_window_completed':sum(r['completion_time_ns']<=w['end_time_ns'] for r in dd),'within_window_QPS':sum(r['completion_time_ns']<=w['end_time_ns'] for r in dd)/120.,'completion_ratio':len(dd)/len(cohort) if cohort else None,'mean_E2E_ms':statistics.mean(lat) if lat else None,'P95_E2E_ms':float(np.percentile(lat,95)) if lat else None,'mean_worker_service_ms':statistics.mean(r['worker_latency_ms'] for r in dd) if dd else None,'P95_worker_service_ms':float(np.percentile([r['worker_latency_ms'] for r in dd],95)) if dd else None,'mean_queue_ms':statistics.mean(r['worker_queue_wait_ms'] for r in dd) if dd else None,'P95_queue_ms':float(np.percentile([r['worker_queue_wait_ms'] for r in dd],95)) if dd else None,'actual_routed_counts':{m:sum(raw['replica_to_family'][r['model']]==m for r in cohort) for m in s['deployment']},'censored':len(dd)<len(cohort),'correct':sum(scored),'scored':len(scored),'accuracy_descriptive':statistics.mean(scored) if scored else None,'corpus_WER_descriptive':sum(v.get('word_errors',0) for v in responses)/words if words else None,'invalid_closed_set_outputs':sum(v.get('valid_closed_set_output') is False for v in responses)}
  taskrows.append(tr)
  if lat:taskstats[t]={'mean_ms':tr['mean_E2E_ms'],'p95_ms':tr['P95_E2E_ms']}
 for worker,identity in raw['worker_identities'].items():
  wr=[r for r in rs if r['model']==worker];m=raw['replica_to_family'][worker];dd=[r for r in wr if r['status']=='completed'];ev=None;cl=None
  if wr:
   tmp={'request_records':wr,'measurement_window':w,'status':'completed','errors':raw['errors'],'drain_timeout_seconds':60,'worker_identities':{worker:identity}};ev=reconstruct(tmp,{worker:SimpleNamespace(service_time_ms=d['models'][m]['service_time_ms'])},DEFAULT_THRESHOLDS);cl=classify(ev,DEFAULT_THRESHOLDS)
  start_unix=w['start_unix_ns'];end_unix=start_unix+int(120e9);busy=sum(max(0,min(r['worker_response']['end_ns'],end_unix)-max(r['worker_response']['start_ns'],start_unix)) for r in dd)/120e9
  workers.append({'run_id':s['run_id'],'stage':s['stage'],'arm':s['arm'],'repetition':s['repetition'],'worker':worker,'family':m,'pid':identity['pid'],'state':cl['stability_state_v2'] if cl else 'IDLE','hard_reasons':cl['hard_reasons'] if cl else [],'warning_reasons':cl['congestion_warning_reasons'] if cl else [],'busy_fraction_completed_handlers_LOWER_BOUND_if_censored':busy,'offered':len(wr),'completed':len(dd),'within_window_QPS':sum(r['completion_time_ns']<=w['end_time_ns'] for r in dd)/120,'mean_service_ms':statistics.mean(r['worker_latency_ms'] for r in dd) if dd else None,'mean_queue_ms':statistics.mean(r['worker_queue_wait_ms'] for r in dd) if dd else None,'end_backlog':ev['end_backlog_requests'] if ev else 0,'peak_backlog':ev['peak_backlog_requests'] if ev else 0,'tail_slope_rps':ev['tail_backlog_slope'] if ev else None,'final_backlog_after_drain':ev['final_backlog_after_drain'] if ev else 0,'recovery_confirmed':cl['final_recovery_confirmed'] if cl else True,'CPU_family_rho':s['CPU_rho'].get(m)})
 idle=[statistics.mean(v['owned_used_mib'] for v in raw[k]) for k in ['postwarm_idle','postmeasurement_idle'] if raw.get(k)];safe=d['safe_factor']*max(idle) if len(idle)==2 else None;peak=max((v['owned_used_mib'] for v in raw['telemetry']['samples']),default=None);memok=safe is not None and peak is not None and safe<=BUDGET+1e-6 and peak<=BUDGET+1e-6
 st=ph['stability']['stability_state_v2'];hard=st=='unstable' or any(r['state']=='unstable' for r in workers);quality=all(r['task'] in d['models'][raw['replica_to_family'][r['model']]]['supported_tasks'] for r in rs) and sum(r['invalid_closed_set_outputs'] for r in taskrows)==0
 complete=len(done)==len(rs);covered=set(taskstats)==set(rates);T=1-min(1.,len(done)/len(rs) if rs else 0.,within/(sum(rates.values())*120));norms=components(taskstats,rates,refs,T,safe/BUDGET,weights) if covered and safe is not None else {'Q_norm':None,'U':None}
 eligible=bool(complete and covered and memok and quality and not hard and raw['valid_session'] and not raw['errors']);ev=ph['stability_evidence'];lat=[r['e2e_latency_ms'] for r in done]
 row={**base,**norms,'U':norms['U'] if eligible else None,'completed_subset_diagnostic_U_NOT_sortable':norms['U'] if not eligible else None,'eligible':eligible,'system_state':st,'worker_states':{r['worker']:r['state'] for r in workers},'all_workers_steady':all(r['state'] in ['steady','IDLE'] for r in workers),'hard_unstable':hard,'quality_admission':quality,'offered':len(rs),'completed':len(done),'within_window_completed':within,'completed_within_window_QPS':within/120,'actual_offered_QPS':len(rs)/120,'completion_ratio':len(done)/len(rs) if rs else None,'window_completion_ratio_actual_offered':within/len(rs) if rs else None,'cohort_QPS_including_drain':len(done)/(120+w['drain_seconds']),'system_mean_E2E_ms':statistics.mean(lat) if lat else None,'system_pooled_P95_ms':float(np.percentile(lat,95)) if lat else None,'observed_safe_memory_mib':safe,'sampled_owned_peak_mib':peak,'memory_passed':memok,'end_backlog':ev['end_backlog_requests'],'tail_slope_rps':ev['tail_backlog_slope'],'final_backlog_after_drain':ev['final_backlog_after_drain'],'drain_seconds':w['drain_seconds'],'censored':not complete,'errors':raw['errors'],'nominal_predicted_U_not_evidence':s['CPU_predicted_U']}
 return row,taskrows,workers

def analyze(raw,f):
 row,tasks,workers=analyze_base(raw,f)
 row['CPU_nominal_feasible']=raw['slot']['CPU_nominal_feasible']
 row['capacity_diagnostic_only']=raw['slot']['capacity_diagnostic_only']
 if not raw['phases'].get('measurement'):
  row.update(all_workers_steady=False,quality_admission=False,censored=True)
  return row,tasks,workers
 rs=raw['phases']['measurement']['request_records'];rates=raw['slot']['lambda_rps']
 refs=read(f['utility_references']['path'])['latency_reference_ms'];catalog=read(f['catalog']['path'])
 row['accepted_submitted']=sum(r['actual_submit_time_ns'] is not None for r in rs)
 row['accepted_QPS']=row['accepted_submitted']/120.
 row['accepted_definition']='host submitted; no independent worker admission ACK'
 for tr in tasks:
  t=tr['task'];cohort=[r for r in rs if r['task']==t];done=[r for r in cohort if r['status']=='completed']
  tr['accepted_submitted']=sum(r['actual_submit_time_ns'] is not None for r in cohort)
  tr['actual_offered_QPS']=tr['offered']/120.;tr['accepted_QPS']=tr['accepted_submitted']/120.;tr['completed_cohort_QPS']=tr['completed']/120.
  tr['offered_weight']=rates[t]/sum(rates.values());tr['reference_ms']=refs[t]
  for name,key in [('L','mean_E2E_ms'),('P','P95_E2E_ms')]:
   tr[name+'_norm_contribution']=tr['offered_weight']*tr[key]/refs[t] if tr[key] is not None else None
   tr[name+'_utility_contribution']=-.3*tr[name+'_norm_contribution'] if tr[name+'_norm_contribution'] is not None else None
  s0=statistics.mean(catalog['models'][raw['replica_to_family'][r['model']]]['service_time_ms'][t] for r in done) if done else None
  tr['frozen_service_mean_for_actual_completed_routes_ms']=s0
  tr['actual_service_minus_frozen_ms']=tr['mean_worker_service_ms']-s0 if s0 is not None else None
  tr['service_ratio_to_frozen_actual_route_mix']=tr['mean_worker_service_ms']/s0 if s0 else None
  tr['CPU_predicted_mean_E2E_ms']=raw['slot']['CPU_per_task'][t]['mean_ms']
  tr['CPU_predicted_P95_E2E_ms']=raw['slot']['CPU_per_task'][t]['p95_ms']
 lid=[r for r in rs if r['task']=='lid'];lid_done=[r for r in lid if r['status']=='completed']
 row['actual_LID_G_offered_fraction']=sum(raw['replica_to_family'][r['model']]==G for r in lid)/len(lid) if lid else None
 row['actual_LID_G_completed_fraction']=sum(raw['replica_to_family'][r['model']]==G for r in lid_done)/len(lid_done) if lid_done else None
 row['LID_G_completed']=sum(raw['replica_to_family'][r['model']]==G for r in lid_done)
 row['LID_S_completed']=sum(raw['replica_to_family'][r['model']]==MMS for r in lid_done)
 for wr in workers:
  cohort=[r for r in rs if r['model']==wr['worker']];done=[r for r in cohort if r['status']=='completed']
  wr['accepted_submitted']=sum(r['actual_submit_time_ns'] is not None for r in cohort)
  wr['actual_offered_QPS']=len(cohort)/120.;wr['accepted_QPS']=wr['accepted_submitted']/120.
  wr['P95_service_ms']=float(np.percentile([r['worker_latency_ms'] for r in done],95)) if done else None
  wr['P95_queue_ms']=float(np.percentile([r['worker_queue_wait_ms'] for r in done],95)) if done else None
  wr['max_queue_ms']=max((r['worker_queue_wait_ms'] for r in done),default=None)
 return row,tasks,workers

def paired(stage,rows,raws):
 pairs=[];rawmap={r['slot']['run_id']:r for r in raws}
 for rep in [1,2,3]:
  a=next((r for r in rows if r['stage']==stage and r['repetition']==rep and r['arm']=='proposed'),None)
  b=next((r for r in rows if r['stage']==stage and r['repetition']==rep and r['arm']=='control'),None)
  p={'stage':stage,'repetition':rep,'delta_U':None,'comparison_valid':False,'feasibility_advantage_separate':False}
  if a is None or b is None:
   p['reason']='Missing run; retain incomplete planned pair';pairs.append(p);continue
  ar,br=rawmap[a['run_id']],rawmap[b['run_id']]
  def signature(d,label):
   ph=d.get('warmup',{}) if label=='warmup' else d['phases'].get('measurement',{})
   return [(r['task'],r['pool_id'],r['sample_id'],r['frozen_offset_seconds'],r['routing_uniform'],r['replica_uniform']) for r in ph.get('request_records',[])]
  same=signature(ar,'measurement')==signature(br,'measurement') and bool(signature(ar,'measurement'))
  warm_same=signature(ar,'warmup')==signature(br,'warmup')
  resident=ar['slot']['deployment']==br['slot']['deployment']
  background=all(ar['slot']['routing'][t]==br['slot']['routing'][t] for t in ar['slot']['routing'] if t!='lid')
  valid=a['eligible'] and b['eligible'] and same and warm_same and not a['capacity_diagnostic_only'] and not b['capacity_diagnostic_only']
  if stage=='HIGH':valid=valid and resident and background
  p.update(comparison_valid=bool(valid),same_offered_trace_and_sample_ids=same,same_warmup_trace=warm_same,
           same_residency=resident,non_LID_routes_identical=background,
           proposed_infeasible=not a['eligible'],control_infeasible=not b['eligible'],
           feasibility_advantage_separate=a['eligible'] and not b['eligible'],
           proposed_QPS=a.get('completed_within_window_QPS'),control_QPS=b.get('completed_within_window_QPS'),
           actual_G_completed_fraction=a.get('actual_LID_G_completed_fraction'),
           all_proposed_workers_steady=a['all_workers_steady'],all_control_workers_steady=b['all_workers_steady'])
  if valid:
   p['delta_U']=a['U']-b['U']
   for name,key in [('mean','L_contribution'),('P95','P_contribution'),('throughput','T_contribution'),('memory','M_contribution')]:
    p['delta_U_'+name]=a[key]-b[key]
   p['component_sum']=sum(p['delta_U_'+name] for name in ['mean','P95','throughput','memory'])
   p['sum_error']=p['component_sum']-p['delta_U'];assert abs(p['sum_error'])<1e-12
  else:p['reason']='No normal utility comparison: eligibility, capacity-diagnostic designation or pairing constraint failed; U remains null when unstable/incomplete'
  pairs.append(p)
 values=[p['delta_U'] for p in pairs if p['delta_U'] is not None];n=len(values)
 mean=statistics.mean(values) if n else None;sd=statistics.stdev(values) if n>1 else None
 ci=[mean-4.302652729911275*sd/math.sqrt(3),mean+4.302652729911275*sd/math.sqrt(3)] if n==3 else None
 positive=n==3 and all(v>0 for v in values)
 if stage=='MID':
  passed=positive and all(p['all_proposed_workers_steady'] for p in pairs)
  no_benefit=n==3 and all(v<=0 for v in values)
  gate='MID_POSITIVE_UTILITY' if passed else 'MID_NO_BENEFIT' if no_benefit else 'MID_INCONCLUSIVE'
 else:
  passed=positive and mean>.01 and all(p['all_proposed_workers_steady'] and p['all_control_workers_steady'] and 0<p['actual_G_completed_fraction']<1 for p in pairs)
  no_benefit=n==3 and all(v<=.01 for v in values)
  gate='HIGH_POSITIVE_SPLIT_UTILITY' if passed else 'HIGH_NO_BENEFIT' if no_benefit else 'HIGH_INCONCLUSIVE'
 return {'stage':stage,'gate':gate,'pairs':pairs,'n_independent_comparable_pairs':n,'values':values,'mean_delta_U':mean,
         'sample_sd':sd,'sample_variance':sd*sd if sd is not None else None,'range':[min(values),max(values)] if n else None,
         'nominal_paired_df2_95CI':ci,'same_positive_direction_all_three':positive,
         'all_three_delta_gt_001':n==3 and all(v>.01 for v in values),
         'component_mean_deltas':{name:statistics.mean(p['delta_U_'+name] for p in pairs if p['comparison_valid']) if n else None for name in ['mean','P95','throughput','memory']},
         'uncertainty':'Independent fresh GPU repetitions are primary units; n<=3, df=2 CI when complete, exploratory and not multiplicity-corrected. No request-level pseudoreplication. CI reported, not an added acceptance gate.',
         'feasibility_only_pair_count':sum(p['feasibility_advantage_separate'] for p in pairs)}

def write_outputs(raws,f):
 rows=[];tasks=[];workers=[];paths=[]
 catalog=read(f['catalog']['path'])
 for raw in raws:
  row,tr,wr=analyze(raw,f);rows.append(row);tasks+=tr;workers+=wr
  rs=raw['phases'].get('measurement',{}).get('request_records',[])
  for worker,identity in raw['worker_identities'].items():
   m=identity['family']
   for t in raw['slot']['lambda_rps']:
    rr=[r for r in rs if r['model']==worker and r['task']==t and r['status']=='completed']
    if not rr:continue
    mean=statistics.mean(r['worker_latency_ms'] for r in rr);s0=catalog['models'][m]['service_time_ms'][t]
    paths.append({'run_id':raw['slot']['run_id'],'stage':raw['slot']['stage'],'arm':raw['slot']['arm'],
                 'task':t,'worker':worker,'completed':len(rr),'frozen_s0_ms':s0,'actual_mean_service_ms':mean,
                 'service_error_ms':mean-s0,'service_ratio':mean/s0,'service_error_percent':100*(mean/s0-1),
                 'actual_P95_service_ms':float(np.percentile([r['worker_latency_ms'] for r in rr],95)),
                 'actual_mean_queue_ms':statistics.mean(r['worker_queue_wait_ms'] for r in rr),
                 'actual_P95_queue_ms':float(np.percentile([r['worker_queue_wait_ms'] for r in rr],95))})
 stages=[paired(stage,rows,raws) for stage in RATES]
 csvsave('gpu_paired_runs.csv',rows,['run_id','stage','arm','repetition','U'])
 csvsave('system_utility_decomposition.csv',[p for st in stages for p in st['pairs']],['stage','repetition','delta_U'])
 csvsave('per_task_e2e.csv',tasks,['run_id','task','offered_QPS','actual_offered_QPS','accepted_QPS','within_window_QPS'])
 csvsave('worker_queue_and_service.csv',workers,['run_id','worker','state'])
 csvsave('actual_path_service_vs_prediction.csv',paths,['run_id','task','worker'])
 save('paired_utility_statistics.json',{'at_utc':now(),'stages':{s['stage']:s for s in stages},'source':ref(OUT/'frozen_experiment_manifest.json')})
 save('real_utility_analysis.json',{'runs':rows,'tasks':tasks,'workers':workers,'paths':paths,'stages':stages})
 save('module22_gate.json',{'at_utc':now(),'MID':stages[0]['gate'],'HIGH':stages[1]['gate'],
      'formal_GPU_runs':sum('measurement' in r['phases'] for r in raws),'attempted_new_sessions':len(raws),
      'maximum_new_sessions':12,'full_three_phase_positive_utility_validated':False,'LOW_not_repaired':True,'no_Module23':True,
      'provisional_while_campaign_pending':len(raws)<12 and not (OUT/'final_integrity_audit.json').exists()})
 return rows
