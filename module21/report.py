"""Real request utility only; censored cohorts cannot produce a sortable U."""
from .common import *
from module20.optimizer import components
import statistics,numpy as np
from types import SimpleNamespace
from performance_model.local_calibration.stability_v2 import reconstruct,classify,DEFAULT_THRESHOLDS

def analyze(raw,f):
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

def stage_result(stage,rows):
 pairs=[]
 for rep in [1,2,3]:
  a=next((r for r in rows if r['repetition']==rep and r['arm']=='proposed'),None);b=next((r for r in rows if r['repetition']==rep and r['arm']=='control'),None)
  if a is None or b is None:continue
  delta=a['U']-b['U'] if a['U'] is not None and b['U'] is not None else None
  pairs.append({'repetition':rep,'delta_U':delta,'control_infeasible':not b['eligible'],'proposed_infeasible':not a['eligible'],'capacity_feasibility_benefit_separate':a['eligible'] and not b['eligible'],'contribution_deltas':{k:a[k]-b[k] for k in ['T_contribution','L_contribution','P_contribution','M_contribution'] if a.get(k) is not None and b.get(k) is not None}})
 values=[r['delta_U'] for r in pairs if r['delta_U'] is not None];mean=statistics.mean(values) if values else None;ci=None
 if len(values)==3:
  se=statistics.stdev(values)/math.sqrt(3);ci=[mean-4.302652729911275*se,mean+4.302652729911275*se]
 passed=len(values)==3 and all(v>.01 for v in values)
 return {'stage':stage,'passed':passed,'pairs':pairs,'values':values,'mean_delta_U':mean,'nominal_paired_95CI':ci,'all_three_same_positive_direction':len(values)==3 and all(v>0 for v in values),'all_three_delta_gt_001':len(values)==3 and all(v>.01 for v in values),'uncertainty':'n=3 independent process repetitions; paired df=2 nominal t interval, exploratory and not multiplicity-adjusted. Fixed common arrival streams within pairs, no favorable seed selection.'}

def write_outputs(raws,f):
 rows=[];tasks=[];workers=[];pairs=[]
 for raw in raws:
  r,tr,wr=analyze(raw,f);rows.append(r);tasks.extend(tr);workers.extend(wr)
 for stage in RATES:
  sr=stage_result(stage,[r for r in rows if r['stage']==stage])
  for p in sr['pairs']:
   rep=p['repetition'];a=next(r for r in rows if r['stage']==stage and r['repetition']==rep and r['arm']=='proposed');b=next(r for r in rows if r['stage']==stage and r['repetition']==rep and r['arm']=='control');ar=next(r for r in raws if r['slot']['run_id']==a['run_id']);br=next(r for r in raws if r['slot']['run_id']==b['run_id']);paired=ar['slot']['trajectories']['measurement']['sha256']==br['slot']['trajectories']['measurement']['sha256'];assert paired
   def signature(raw):return [(r['task'],r['pool_id'],r['frozen_offset_seconds']) for r in raw['phases'].get('measurement',{}).get('request_records',[])]
   same=signature(ar)==signature(br);pairs.append({'stage':stage,**p,'mean_delta_U':sr['mean_delta_U'],'nominal_paired_95CI':sr['nominal_paired_95CI'],'stage_passed':sr['passed'],'same_frozen_trajectory':paired,'same_actual_scheduled_samples':same,'uncertainty':sr['uncertainty']})
 csvsave('gpu_paired_runs.csv',rows,['run_id','stage','arm','repetition','U']);csvsave('system_utility_comparison.csv',pairs,['stage','repetition','delta_U']);csvsave('per_task_metrics.csv',tasks,['run_id','task']);csvsave('worker_stability.csv',workers,['run_id','worker','state']);save('real_utility_analysis.json',{'runs':rows,'tasks':tasks,'workers':workers,'stages':[stage_result(stage,[r for r in rows if r['stage']==stage]) for stage in RATES]})
