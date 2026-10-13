"""Frozen independent validation harness, explicit local monitor and confined native backends.
Native worker transport/inference, warmup, queues, timeout/thermal rules unchanged.
"""
from module30.gpu import *
from .common import *
from .runtime import phase
from .monitor import IncrementalMonitor
from router_optimized import DeploymentRuntime
class AuditedRuntime(DeploymentRuntime):
 def close(self):
  report=super().close()
  live=[name for name,t in self.senders.items() if t.is_alive()]
  if self.collector and self.collector.is_alive():live.append("collector")
  live += ["reader:"+name for name,w in self.workers.items() if w.process and not w.reader_done.is_set()]
  if live:self.errors.append(dict(error="RUNTIME_THREADS_DID_NOT_EXIT",threads=live))
  return report
SafeMonitor=IncrementalMonitor

def run(slot,f):
 path=OUT/'raw_runs'/(slot['run_id']+'.json');assert not path.exists(),'No retries or overwritten raw'
 save('run_attempts/'+slot['run_id']+'.json',dict(slot=slot,started_unix_ns=time.time_ns(),pid=os.getpid()))
 pre=cool(slot,f);save('thermal_prelaunch/'+slot['run_id']+'.json',pre);baseline=pre['samples'][-1];plan=f['plan']
 assert all(baseline[k]==f['GPU'][k] for k in ['uuid','name','driver','total_mib'])
 for k,v in plan['environment_overrides'].items():os.environ[k]=v
 mapping={f'{m}#{i}':m for m,n in slot['deployment'].items() for i in range(n)}
 routes={t:{i:q/slot['deployment'][m] for m,q in row.items() for i,mm in mapping.items() if mm==m} for t,row in slot['routing'].items()}
 config=PoissonConfig(deployment={i:1 for i in mapping},workload_rps=slot['lambda_rps'],routing_rps=routes,arrival_process='poisson',random_seed=slot['seed'],
  warmup_seconds=30.,measurement_seconds=slot['measurement_seconds'],startup_timeout_seconds=300.,drain_timeout_seconds=60.,shutdown_timeout_seconds=15.,routing_source='Module32G reused matched engineering trajectory; dynamic uses causal observed pending state only; Module32G independent pair; no policy changes')
 config.validate({i:plan['models'][m]['supported_tasks'] for i,m in mapping.items()})
 runtime=AuditedRuntime(config,{i:plan['registry'][m] for i,m in mapping.items()});monitor=SafeMonitor(runtime,slot,slot['safe_memory_budget_mib'])
 raw=dict(slot=slot,status='STARTING',errors=[],phases={},baseline_GPU=baseline,replica_to_family=mapping,worker_identities={},transport_config=config.as_dict(),frozen_manifest=ref(OUT/'frozen_gpu_manifest.json'));begin=time.perf_counter();monitor.start()
 try:
  runtime.start();monitor.check();raw['worker_identities']={i:dict(family=m,pid=runtime.workers[i].process.pid,READY=runtime.ready[i]) for i,m in mapping.items()}
  assert len({r['pid'] for r in raw['worker_identities'].values()})==len(mapping)
  raw['path_probe_records']=prime(runtime,config,plan,mapping,slot);monitor.check()
  conditioning_slot=copy.deepcopy(slot);conditioning_slot['arm']='s_only'
  conditioning_slot['routing']['asr']={slot['Specialist']:slot['lambda_rps']['asr']}
  raw['conditioning']=phase(runtime,config,conditioning_slot,plan,mapping,'conditioning',monitor)
  assert all(r['status']=='completed' for r in raw['conditioning']['request_records'])
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
  raw['telemetry']=dict(samples=monitor.samples,errors=monitor.errors,error_details=monitor.error_details,interval_seconds=.5,
   collector_exited=not monitor.thread.is_alive(),peak_scope='NVML sampled owned high-water, not instantaneous allocator maximum')
  raw['runtime_index']=dict(active_high_water=runtime.active_high_water,final_active=len(runtime.active),archival_count=len(runtime.pending),journal_snapshot_lock_ms=monitor.snapshot_lock_ms)
  raw['control_latency_seconds']=dict(check=monitor.check_seconds,phase_publish=monitor.publish_seconds,
   max_check=max(monitor.check_seconds,default=0),max_publish=max(monitor.publish_seconds,default=0),threshold_seconds=.01)
  raw['control_probes']=monitor.control_probes
  raw['control_nonblocking']=max(monitor.check_seconds+monitor.publish_seconds,default=0)<.01
  raw['all_runtime_records']=list(runtime.pending.values())
  raw['routing_decisions_all_phases']=monitor.routing_logs
  from .isolation import VIOLATIONS
  raw['unexpected_write_violations']=list(VIOLATIONS)
  raw['engineering_only']=False;raw['U']=None;raw['utility_computed']=False
  raw['gpu_release']=release(baseline,[w.process.pid for w in runtime.workers.values() if w.process])
  raw['valid_session']=bool(raw['status']=='COMPLETED' and raw['gpu_release']['verified'] and not raw['errors'] and raw['telemetry']['collector_exited'] and len(raw.get('shutdown',{}))==len(mapping) and all(v['exit_code']==0 and not v['forced'] and v['bye_received'] for v in raw['shutdown'].values()))
  raw['elapsed_seconds']=time.perf_counter()-begin;save('raw_runs/'+path.name,raw)
 return raw
