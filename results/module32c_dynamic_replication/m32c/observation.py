"""CPU raw accounting and original Module20A utility; no optimizer change."""
import collections,math,statistics,time
from bisect import bisect_right
from types import SimpleNamespace
import numpy as np
from .common import *
from module20.optimizer import components
from module25.common import utility_inputs
from performance_model.local_calibration.stability_v2 import reconstruct,classify,DEFAULT_THRESHOLDS

def stats(rows,key):
 a=[r[key] for r in rows if r.get(key) is not None]
 return dict(n=len(a),mean=float(np.mean(a)),p95=float(np.percentile(a,95))) if a else dict(n=0,mean=None,p95=None)
def journal_audit(raw):
 slot=raw['slot'];path=OUT/'request_journals'/(slot['run_id']+'.jsonl');terminal={};duplicates=[];headers=[]
 try:
  with path.open() as f:
   for line in f:
    row=json.loads(line)
    if row['kind']=='PHASE_WINDOW':headers.append(row['label'])
    if row['kind']=='REQUEST_TERMINAL':
     key=(row['phase'],row['record']['request_id'])
     if key in terminal:duplicates.append(key)
     terminal[key]=row['record']
 except Exception as exc:return dict(passed=False,error=repr(exc))
 cohorts=[raw.get('warmup')]+list(raw.get('phases',{}).values())+[raw.get('aborted_phase')]
 keys=[];missing=[];mismatch=[];fields=['status','task','model','actual_submit_time_ns','completion_time_ns','worker_latency_ms','worker_queue_wait_ms','worker_response']
 for ph in cohorts:
  if not ph:continue
  for r in ph['request_records']:
   key=(r['phase'],r['request_id']);keys.append(key);s=terminal.get(key)
   if s is None:missing.append(key);continue
   if any(s.get(k)!=r.get(k) for k in fields):mismatch.append(key)
 return dict(passed=not missing and not duplicates and not mismatch and len(keys)==len(set(keys)),expected=len(keys),terminal=len(terminal),missing=missing,duplicates=duplicates,mismatch=mismatch,headers=headers)
def observation(raw):
 f=read(OUT/'frozen_gpu_manifest.json');slot=raw['slot'];ph=raw.get('phases',{}).get('measurement');audit=journal_audit(raw)
 out=dict(run_id=slot['run_id'],candidate=slot['candidate'],arm=slot['arm'],Generalist=slot['Generalist'],Specialist=slot['Specialist'],
  status=raw['status'],valid_session=raw.get('valid_session',False),eligible=False,U=None,journal_audit=audit,errors=raw['errors'])
 if not ph:return {**out,'stability':'UNASSESSABLE_ABORT','reason':'No full measurement phase, entire planned cohort retained'}
 seconds=ph['window']['planned_duration_seconds'];rates=slot['lambda_rps'];rs=ph['request_records'];done=[r for r in rs if r['status']=='completed'];mapping=raw['replica_to_family'];window=ph['window']
 taskstats={};per={};paths=[];workers=[]
 for t,q in rates.items():
  rr=[r for r in rs if r['task']==t];dd=[r for r in rr if r['status']=='completed'];e=stats(dd,'e2e_latency_ms')
  per[t]=dict(n_offered=len(rr),n_completed=len(dd),mean_ms=e['mean'],p95_ms=e['p95'],offered_qps=len(rr)/seconds,completed_qps=len(dd)/seconds,
   within_qps=sum(r['completion_time_ns']<=window['end_time_ns'] for r in dd)/seconds)
  if e['n']:taskstats[t]=dict(mean_ms=e['mean'],p95_ms=e['p95'])
  requested_workers={r['model'] for r in rr}
  if t==slot['target_task']:requested_workers.update([slot['Generalist']+'#0',slot['Specialist']+'#0'])
  for worker in sorted(requested_workers):
   pp=[r for r in dd if r['model']==worker];paths.append(dict(worker=worker,model=mapping[worker],task=t,completed=len(pp),
    service=stats(pp,'worker_latency_ms'),wait=stats(pp,'worker_queue_wait_ms'),e2e=stats(pp,'e2e_latency_ms'),host_schedule_delay=stats(pp,'schedule_delay_ms')))
 for worker,ident in raw['worker_identities'].items():
  wr=[r for r in rs if r['model']==worker]
  if not wr:continue
  tmp=dict(request_records=wr,measurement_window=window,status='completed',errors=raw['errors'],drain_timeout_seconds=60,worker_identities={worker:ident})
  ev=reconstruct(tmp,{worker:SimpleNamespace(service_time_ms=f['plan']['models'][mapping[worker]]['service_time_ms'])},DEFAULT_THRESHOLDS)
  cl=classify(ev,DEFAULT_THRESHOLDS);workers.append(dict(worker=worker,classification=cl,evidence=ev,busy_fraction=sum(r['worker_latency_ms'] for r in wr if r['status']=='completed')/(seconds*1000)))
 idle=[float(np.mean([v['owned_used_mib'] for v in raw[k]])) for k in ['postwarm_idle','postmeasurement_idle'] if raw.get(k)]
 safe=f['safe_factor']*max(idle) if len(idle)==2 else None;peak=max((v['owned_used_mib'] for v in raw['telemetry']['samples']),default=None)
 memok=safe is not None and safe<=slot['safe_memory_budget_mib'] and peak is not None and peak<=slot['safe_memory_budget_mib']
 hard=ph['stability']['stability_state_v2']=='unstable' or any(w['classification']['stability_state_v2']=='unstable' for w in workers)
 recovery=all(c.get('final_recovery_confirmed',False) if c['stability_state_v2']=='recoverable_congested' else True for c in [ph['stability']]+[w['classification'] for w in workers])
 quality=all(r['task'] in f['plan']['models'][mapping[r['model']]]['supported_tasks'] for r in rs) and not any(r.get('worker_response',{}).get('valid_closed_set_output') is False for r in done)
 trace=read(slot['trajectories']['measurement']['path'])['events']
 native_timings_ok=all(all(r.get('worker_response',{}).get(k) is not None for k in ['start_ns','end_ns','submit_ns']) and r['worker_latency_ms']>=0 and r['worker_queue_wait_ms']>=0 and abs(r['worker_latency_ms']-(r['worker_response']['end_ns']-r['worker_response']['start_ns'])/1e6)<1e-6 and abs(r['worker_queue_wait_ms']-(r['worker_response']['start_ns']-r['worker_response']['submit_ns'])/1e6)<1e-6 for r in done)
 traceok=len(rs)==len(trace) and all(r['task']==e['task'] and r['sample_id']==e['item']['sample'].get('id') and r['pool_id']==e['item']['pool_id'] and r['frozen_offset_seconds']==e['offset_seconds'] and r['routing_uniform']==e['routing_uniform'] and r['replica_uniform']==e['replica_uniform'] for r,e in zip(rs,trace))
 eligible=raw['valid_session'] and audit['passed'] and traceok and native_timings_ok and quality and memok and not hard and recovery and len(done)==len(rs) and set(taskstats)==set(rates)
 within=sum(r['completion_time_ns']<=window['end_time_ns'] for r in done);T=1-min(1.,len(done)/len(rs),within/(sum(rates.values())*seconds))
 definition,refs=utility_inputs();norms=components(taskstats,rates,refs,T,safe/slot['safe_memory_budget_mib'],definition['weights']) if len(taskstats)==len(rates) and safe is not None else {'U':None}
 # Independent original raw arithmetic, without observation/predictor fitting.
 from module26.common import arithmetic
 check=arithmetic(raw,read(OUT/'selected_pool_catalog.json'),refs,definition['weights'])
 if norms['U'] is not None:assert abs(norms['U']-check['raw_arithmetic_U'])<1e-12
 times=[v for v in raw['telemetry']['samples'] if v['phase']=='measurement' and window['start_time_ns']<=v['time_ns']<=window['end_time_ns']]
 def sensorstat(key):return stats(times,key)
 target=[r for r in done if r['task']==slot['target_task']];g=sum(mapping[r['model']]==slot['Generalist'] for r in target);s=len(target)-g
 out.update(eligible=bool(eligible),U=norms['U'] if eligible else None,components={**norms,'U':norms['U'] if eligible else None},
  diagnostic_completed_subset_U_NOT_sortable=norms['U'] if not eligible else None,per_task=per,paths=paths,workers=workers,
  stability=ph['stability']['stability_state_v2'],hard_unstable=hard,recovery=recovery,quality_admitted=quality,matched_trace=traceok,native_timings_consistent=native_timings_ok,
  offered=len(rs),completed=len(done),within=within,offered_qps=len(rs)/seconds,completed_qps=len(done)/seconds,within_qps=within/seconds,
  G_completed=g,S_completed=s,G_fraction=g/len(target) if target else None,
  G_offered=sum(r['task']==slot['target_task'] and mapping[r['model']]==slot['Generalist'] for r in rs),
  S_offered=sum(r['task']==slot['target_task'] and mapping[r['model']]==slot['Specialist'] for r in rs),
  G_offered_fraction=sum(r['task']==slot['target_task'] and mapping[r['model']]==slot['Generalist'] for r in rs)/sum(r['task']==slot['target_task'] for r in rs),safe_memory_mib=safe,sampled_peak_mib=peak,memory_passed=memok,
  pooled_mean_ms=stats(done,'e2e_latency_ms')['mean'],pooled_p95_ms=stats(done,'e2e_latency_ms')['p95'],drain_seconds=window['drain_seconds'],
  temperature=sensorstat('temperature_gpu_c'),sm_clock=sensorstat('sm_clock_mhz'),memory_clock=sensorstat('memory_clock_mhz'),
  postwarm_temperature_mean=float(np.mean([v['temperature_gpu_c'] for v in raw['postwarm_idle']])) if raw.get('postwarm_idle') else None,
  telemetry_samples=len(times),GPU_uuid=raw['baseline_GPU']['uuid'])
 return out

def backlog(raw):
 ph=raw.get('phases',{}).get('measurement') or raw.get('aborted_phase')
 if not ph:return []
 rs=ph['request_records'];window=ph.get('window',ph);start=window.get('start_time_ns',window.get('planned_start_time_ns'));end=window.get('end_time_ns',window.get('planned_end_time_ns'));abort=ph.get('abort_time_ns');out=[]
 for worker in ['system']+sorted(raw['worker_identities']):
  rr=rs if worker=='system' else [r for r in rs if r['model']==worker]
  schedules=sorted(r['scheduled_time_ns'] for r in rr);enqueues=sorted(r['enqueued_time_ns'] for r in rr if r.get('enqueued_time_ns'));submits=sorted(r['actual_submit_time_ns'] for r in rr if r.get('actual_submit_time_ns'));completions=sorted(r['completion_time_ns'] for r in rr if r['status']=='completed')
  for offset in range(int(window.get('planned_duration_seconds',2400))+1):
   at=start+offset*1000000000
   if abort and at>abort:continue
   done=bisect_right(completions,at)
   out.append(dict(run_id=raw['slot']['run_id'],worker=worker,offset_seconds=offset,offered_backlog=bisect_right(schedules,at)-done,host_backlog=bisect_right(enqueues,at)-done,submitted_backlog=bisect_right(submits,at)-done))
 return out
