import os,traceback,threading,fcntl
from types import SimpleNamespace
from .common import *
from module30 import gpu as nativegpu,telemetry
from . import observation as report
from module31a.campaign import TimedMonitor
from . import validation as r31
from module31a import common as c31
from module21.trajectory import choose
from joint_deploy.measurement.run_measurement import new_record,wait_until
from joint_deploy.measurement.metrics import annotate_records,summarize
from performance_model.local_calibration.stability_v2 import reconstruct,classify,DEFAULT_THRESHOLDS
from vqa_aware_dynamic_router import VQAAwareRouter

def configure():
 for m in [nativegpu,telemetry,report,r31,c31]:m.OUT=OUT;m.save=save;m.csvsave=csvsave
 r31.configure=configure
 return nativegpu,report

def phase(runtime,config,slot,plan,mapping,label,monitor):
 tr=read(slot['trajectories'][label]['path']);assert ref(slot['trajectories'][label]['path'])['sha256']==slot['trajectories'][label]['sha256']
 ev=tr['events'];duration=tr['duration'];router=VQAAwareRouter(read(OUT/'frozen_routing_parameters.json'),read(OUT/'audio_metadata_cache.json'),duration)
 start=time.perf_counter_ns();end=start+round(duration*1e9);unix=time.time_ns();rs=[];decisions=[]
 for i,e in enumerate(ev):
  model=S if slot['arm']=='dynamic' and e['task']=='asr' else choose(e,slot['routing'],slot['deployment'])
  r=new_record(label,i,e['task'],model,e['item'],start+round(e['offset_seconds']*1e9));r.update(frozen_offset_seconds=e['offset_seconds'],routing_uniform=e['routing_uniform'],replica_uniform=e['replica_uniform']);rs.append(r)
 monitor.phase=label;monitor.set_active(dict(label=label,planned_start_time_ns=start,planned_end_time_ns=end,planned_duration_seconds=duration,request_records=rs))
 for r,e in zip(rs,ev):
  wait_until(r['scheduled_time_ns']);monitor.check()
  with runtime.lock:
   if runtime.errors:raise RuntimeError(runtime.errors)
   pending=[x for x in runtime.pending.values() if x['status'] in ('queued','submitted')]
   oldest=max(((time.perf_counter_ns()-x['enqueued_time_ns'])/1e9 for x in pending),default=0)
  if len(pending)>300 or oldest>120:raise RuntimeError('FROZEN_BACKLOG_SAFETY_ABORT')
  if r['task']=='asr':
   _,b=router.cost(e['item']['sample']['input']);
   with runtime.lock:r['router_S_service_estimate_ms']=b['S_p75_ms'] if b else max(x['S_p75_ms'] for x in router.p['cost_bins'])
   if slot['arm']=='dynamic':
    tick=time.perf_counter_ns()
    # Snapshot past observed state and commit admission under the same lock.
    # Only statuses and host times enter the router; no worker_response/START/END/result/label.
    with runtime.lock:
     now=time.perf_counter_ns();proj=[dict(role='G' if x['model']==G else 'S',task=x['task'],submit_perf_ns=x.get('actual_submit_time_ns'),enqueue_perf_ns=x['enqueued_time_ns'],S_service_estimate_ms=x.get('router_S_service_estimate_ms',max(z['S_p75_ms'] for z in router.p['cost_bins']))) for x in runtime.pending.values() if x['model'] in [G,S] and x['status'] in ('queued','submitted')]
     dec=router.decide(now,e['item']['sample']['input'],proj);r['model']=G if dec['select_G'] else S
     runtime.enqueue(r,e['item']['sample'])
    dec.update(request_id=r['request_id'],decision_perf_ns=now,routing_compute_ms=(time.perf_counter_ns()-tick)/1e6,past_state_snapshot=proj,phase=label);decisions.append(dec)
    continue
  runtime.enqueue(r,e['item']['sample'])
 wait_until(end);monitor.check();ae=time.perf_counter_ns();monitor.phase='drain' if label=='measurement' else 'warmup_drain';runtime.drain(rs,60.);drain=time.perf_counter_ns()
 with runtime.lock:annotate_records(rs)
 monitor.flush_active();window=dict(start_time_ns=start,start_unix_ns=unix,end_time_ns=end,planned_end_time_ns=end,duration_seconds=duration,planned_duration_seconds=duration,arrival_loop_end_time_ns=ae,drain_end_time_ns=drain,drain_seconds=(drain-end)/1e9,stop=None,cohort='Full frozen offered cohort, bounded drain; causal routes chosen only at arrival')
 with runtime.lock:per,system,models=summarize(rs,config,window)
 tmp=dict(request_records=rs,measurement_window=window,status='completed' if all(r['status']=='completed' for r in rs) else 'completed_with_failures',errors=runtime.errors,drain_timeout_seconds=60.,worker_identities={i:{'pid':w.process.pid} for i,w in runtime.workers.items()});aliases={i:SimpleNamespace(service_time_ms=plan['models'][m]['service_time_ms']) for i,m in mapping.items()};evc=reconstruct(tmp,aliases,DEFAULT_THRESHOLDS);monitor.set_active(None)
 return dict(window=window,request_records=rs,per_task_metrics=per,system_metrics=system,stability=classify(evc,DEFAULT_THRESHOLDS),stability_evidence=evc,trajectory=slot['trajectories'][label],routing_decisions=decisions)
