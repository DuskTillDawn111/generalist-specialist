import os,time,copy,traceback,fcntl
from types import SimpleNamespace
import numpy as np
from .common import *
from .telemetry_v2 import SafeMonitorV2
from module22.gpu import PoissonConfig
from module21.trajectory import choose
from module11b.measure import prime
from joint_deploy.measurement.deployment_runtime import DeploymentRuntime
from joint_deploy.measurement.run_measurement import new_record,wait_until
from joint_deploy.measurement.metrics import annotate_records,summarize
from replica_scaling.telemetry import snapshot,release
from performance_model.local_calibration.stability_v2 import reconstruct,classify,DEFAULT_THRESHOLDS
class TimedMonitor(SafeMonitorV2):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);self.check_seconds=[];self.publish_seconds=[]
 def check(self):
  s=time.perf_counter()
  try:return super().check()
  finally:self.check_seconds.append(time.perf_counter()-s)
 def set_active(self,v):
  s=time.perf_counter()
  try:return super().set_active(v)
  finally:self.publish_seconds.append(time.perf_counter()-s)
SafeMonitor=TimedMonitor
class AuditedRuntime(DeploymentRuntime):
 def close(self):
  report=super().close()
  live=[name for name,t in self.senders.items() if t.is_alive()]
  if self.collector and self.collector.is_alive():live.append('collector')
  live += ['reader:'+name for name,w in self.workers.items() if w.process and not w.reader_done.is_set()]
  if live:self.errors.append(dict(error='RUNTIME_THREADS_DID_NOT_EXIT',threads=live))
  return report

def phase(runtime,config,slot,plan,mapping,label,monitor):
 trajectory=read(slot['trajectories'][label]['path']);assert ref(slot['trajectories'][label]['path'])['sha256']==slot['trajectories'][label]['sha256']
 events=trajectory['events'];duration=trajectory['duration'];start=time.perf_counter_ns();end=start+round(duration*1e9);unix=time.time_ns()
 records=[]
 for i,e in enumerate(events):
  r=new_record(label,i,e['task'],choose(e,slot['routing'],slot['deployment']),e['item'],start+round(e['offset_seconds']*1e9))
  r.update(frozen_offset_seconds=e['offset_seconds'],routing_uniform=e['routing_uniform'],replica_uniform=e['replica_uniform']);records.append(r)
 # Publish complete planned cohort; collector writes all record mutations under runtime.lock.
 monitor.phase=label;monitor.set_active(dict(label=label,planned_start_time_ns=start,planned_end_time_ns=end,planned_duration_seconds=duration,request_records=records))
 for r,e in zip(records,events):
  wait_until(r['scheduled_time_ns']);monitor.check()
  with runtime.lock:
   if runtime.errors:raise RuntimeError(runtime.errors)
   pending=[p for p in runtime.pending.values() if p['status'] in ('queued','submitted')]
   oldest=max(((time.perf_counter_ns()-p['enqueued_time_ns'])/1e9 for p in pending),default=0)
  if len(pending)>300 or oldest>120:raise RuntimeError('FROZEN_BACKLOG_SAFETY_ABORT')
  runtime.enqueue(r,e['item']['sample'])
 wait_until(end);monitor.check();arrival_end=time.perf_counter_ns();monitor.phase='drain' if label=='measurement' else 'warmup_drain'
 runtime.drain(records,60.);drain=time.perf_counter_ns()
 with runtime.lock:annotate_records(records)
 monitor.flush_active()
 window=dict(start_time_ns=start,start_unix_ns=unix,end_time_ns=end,planned_end_time_ns=end,duration_seconds=duration,planned_duration_seconds=duration,
  arrival_loop_end_time_ns=arrival_end,drain_end_time_ns=drain,drain_seconds=(drain-end)/1e9,stop=None,cohort='All frozen matched offered requests including full bounded drain')
 with runtime.lock:per,system,models=summarize(records,config,window)
 tmp=dict(request_records=records,measurement_window=window,status='completed' if all(r['status']=='completed' for r in records) else 'completed_with_failures',errors=runtime.errors,drain_timeout_seconds=60.,worker_identities={i:{'pid':w.process.pid} for i,w in runtime.workers.items()})
 aliases={i:SimpleNamespace(service_time_ms=plan['models'][m]['service_time_ms']) for i,m in mapping.items()};ev=reconstruct(tmp,aliases,DEFAULT_THRESHOLDS)
 monitor.set_active(None)
 return dict(window=window,request_records=records,per_task_metrics=per,system_metrics=system,stability=classify(ev,DEFAULT_THRESHOLDS),stability_evidence=ev,trajectory=slot['trajectories'][label])

def cool(slot,f,monitor=None):
 from .environment import thermal
 resident=monitor is not None;p=f['thermal_protocol'];limit=p['resident_cool_timeout_seconds'] if resident else p['max_wait_seconds'];maxc=p['resident_premeasurement_temperature_max_c'] if resident else p['prelaunch_temperature_max_c'];required=p['resident_consecutive_samples'] if resident else p['consecutive_idle_samples_required'];begin=time.monotonic();samples=[];consecutive=0
 while time.monotonic()-begin<limit:
  v=monitor.sample() if resident else snapshot()
  if not resident:v.update(thermal())
  samples.append(v)
  if not resident and (v['processes'] or v['used_mib']>128):raise RuntimeError('PRELAUNCH_GPU_ISOLATION_FAILURE')
  consecutive=consecutive+1 if v['temperature_gpu_c']<=maxc else 0
  if consecutive>=required:return dict(passed=True,samples=samples,elapsed_seconds=time.monotonic()-begin)
  if monitor:monitor.check()
  time.sleep(2.)
 raise RuntimeError('THERMAL_LAUNCH_FAILURE')

def safe_idle(raw,key,monitor,f):
 raw[key]=monitor.idle_window(key,2.);monitor.check()
 if f['safe_factor']*np.mean([v['owned_used_mib'] for v in raw[key]])>raw['slot']['safe_memory_budget_mib']:raise RuntimeError('SAFE_MEMORY_BUDGET_EXCEEDED')

def run(slot,f):
 path=OUT/'raw_runs'/(slot['run_id']+'.json');assert not path.exists(),'No retries or overwritten raw'
 save('run_attempts/'+slot['run_id']+'.json',dict(slot=slot,started_unix_ns=time.time_ns(),pid=os.getpid()))
 pre=cool(slot,f);save('thermal_prelaunch/'+slot['run_id']+'.json',pre);baseline=pre['samples'][-1];plan=f['plan']
 assert all(baseline[k]==f['GPU'][k] for k in ['uuid','name','driver','total_mib'])
 for k,v in plan['environment_overrides'].items():os.environ[k]=v
 mapping={f'{m}#{i}':m for m,n in slot['deployment'].items() for i in range(n)}
 routes={t:{i:q/slot['deployment'][m] for m,q in row.items() for i,mm in mapping.items() if mm==m} for t,row in slot['routing'].items()}
 config=PoissonConfig(deployment={i:1 for i in mapping},workload_rps=slot['lambda_rps'],routing_rps=routes,arrival_process='poisson',random_seed=slot['seed'],
  warmup_seconds=30.,measurement_seconds=slot['measurement_seconds'],startup_timeout_seconds=300.,drain_timeout_seconds=60.,shutdown_timeout_seconds=15.,routing_source='Module35 frozen finite benchmark matched Poisson inputs and independent route/replica uniforms')
 config.validate({i:plan['models'][m]['supported_tasks'] for i,m in mapping.items()})
 runtime=AuditedRuntime(config,{i:plan['registry'][m] for i,m in mapping.items()});monitor=SafeMonitor(runtime,slot,slot['safe_memory_budget_mib'])
 raw=dict(slot=slot,status='STARTING',errors=[],phases={},baseline_GPU=baseline,replica_to_family=mapping,worker_identities={},transport_config=config.as_dict(),frozen_manifest=ref(OUT/f.get('manifest_name','frozen_gpu_manifest.json')));begin=time.perf_counter();monitor.start()
 try:
  runtime.start();monitor.check();raw['worker_identities']={i:dict(family=m,pid=runtime.workers[i].process.pid,READY=runtime.ready[i]) for i,m in mapping.items()}
  assert len({r['pid'] for r in raw['worker_identities'].values()})==len(mapping)
  raw['path_probe_records']=prime(runtime,config,plan,mapping,slot);monitor.check()
  raw['warmup']=phase(runtime,config,slot,plan,mapping,'warmup',monitor);assert all(r['status']=='completed' for r in raw['warmup']['request_records'])
  monitor.phase='postwarm_thermal_control';raw['resident_thermal_control']=cool(slot,f,monitor);safe_idle(raw,'postwarm_idle',monitor,f)
  print('MEASUREMENT_START',slot['run_id'],slot['measurement_seconds'],flush=True);raw['phases']['measurement']=phase(runtime,config,slot,plan,mapping,'measurement',monitor)
  safe_idle(raw,'postmeasurement_idle',monitor,f)
  if not all(r['status']=='completed' for r in raw['phases']['measurement']['request_records']):raise RuntimeError('REQUEST_ERROR_OR_INCOMPLETE_COHORT')
  raw['status']='COMPLETED'
 except (Exception,KeyboardInterrupt) as exc:
  raw['status']='FAILED';raw['errors'].append(repr(exc));raw['exception_traceback']=traceback.format_exc()
  with monitor.sample_lock:
   if monitor.active_phase:raw['aborted_phase']=monitor.active_phase;raw['aborted_phase']['abort_time_ns']=time.perf_counter_ns()
 finally:
  # Allow ordered RESULT/BYE/EXIT collection before marking any surviving cohort incomplete.
  try:raw['shutdown']=runtime.close()
  except Exception as exc:raw['errors'].append('shutdown:'+repr(exc))
  if raw.get('aborted_phase'):
   with runtime.lock:
    for r in raw['aborted_phase']['request_records']:
     if r['status'] in ('planned','queued','submitted'):r.update(status='incomplete',error='Entire frozen cohort retained after abort')
    annotate_records(raw['aborted_phase']['request_records'])
   try:monitor.flush_active()
   except Exception as exc:raw['errors'].append('final_journal:'+repr(exc))
  try:monitor.close()
  except Exception as exc:raw['errors'].append('monitor_close:'+repr(exc))
  raw['errors']+=runtime.errors+monitor.errors
  raw['control_latency_seconds']=dict(max_check=max(monitor.check_seconds,default=0),max_publish=max(monitor.publish_seconds,default=0),threshold_seconds=.01)
  raw['control_nonblocking']=max(monitor.check_seconds+monitor.publish_seconds,default=0)<.01
  from .isolation import VIOLATIONS
  raw['unexpected_write_violations']=list(VIOLATIONS)
  raw['telemetry']=dict(samples=monitor.samples,errors=monitor.errors,error_details=monitor.error_details,interval_seconds=.5,
   collector_exited=not monitor.thread.is_alive(),peak_scope='NVML sampled owned high-water, not instantaneous allocator maximum')
  raw['gpu_release']=release(baseline,[w.process.pid for w in runtime.workers.values() if w.process])
  raw['valid_session']=bool(raw['status']=='COMPLETED' and raw['gpu_release']['verified'] and not raw['errors'] and raw['telemetry']['collector_exited'] and len(raw.get('shutdown',{}))==len(mapping) and all(v['exit_code']==0 and not v['forced'] and v['bye_received'] for v in raw['shutdown'].values()))
  raw['elapsed_seconds']=time.perf_counter()-begin;save('raw_runs/'+path.name,raw)
 return raw
