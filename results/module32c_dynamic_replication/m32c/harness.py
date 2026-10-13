"""Exact Module30 native harness copy; only window config/log/source label adapted.
SafeMonitor is explicitly TimedMonitor(SafeMonitorV2); OUT/save use Module32.
Native worker transport, warmup, path probes, queues, timeout and thermal rules unchanged.
"""
from module30.gpu import *
from .common import *
from .runtime import phase
from module31a.campaign import TimedMonitor
SafeMonitor=TimedMonitor

def run(slot,f):
 path=OUT/'raw_runs'/(slot['run_id']+'.json');assert not path.exists(),'No retries or overwritten raw'
 save('run_attempts/'+slot['run_id']+'.json',dict(slot=slot,started_unix_ns=time.time_ns(),pid=os.getpid()))
 pre=cool(slot,f);save('thermal_prelaunch/'+slot['run_id']+'.json',pre);baseline=pre['samples'][-1];plan=f['plan']
 assert all(baseline[k]==f['GPU'][k] for k in ['uuid','name','driver','total_mib'])
 for k,v in plan['environment_overrides'].items():os.environ[k]=v
 mapping={f'{m}#{i}':m for m,n in slot['deployment'].items() for i in range(n)}
 routes={t:{i:q/slot['deployment'][m] for m,q in row.items() for i,mm in mapping.items() if mm==m} for t,row in slot['routing'].items()}
 config=PoissonConfig(deployment={i:1 for i in mapping},workload_rps=slot['lambda_rps'],routing_rps=routes,arrival_process='poisson',random_seed=slot['seed'],
  warmup_seconds=30.,measurement_seconds=slot['measurement_seconds'],startup_timeout_seconds=300.,drain_timeout_seconds=60.,shutdown_timeout_seconds=15.,routing_source='Module32C independent matched pair trajectory; dynamic uses causal observed pending state only')
 config.validate({i:plan['models'][m]['supported_tasks'] for i,m in mapping.items()})
 runtime=AuditedRuntime(config,{i:plan['registry'][m] for i,m in mapping.items()});monitor=SafeMonitor(runtime,slot,slot['safe_memory_budget_mib'])
 raw=dict(slot=slot,status='STARTING',errors=[],phases={},baseline_GPU=baseline,replica_to_family=mapping,worker_identities={},transport_config=config.as_dict(),frozen_manifest=ref(OUT/'frozen_gpu_manifest.json'));begin=time.perf_counter();monitor.start()
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
  raw['telemetry']=dict(samples=monitor.samples,errors=monitor.errors,error_details=monitor.error_details,interval_seconds=.5,
   collector_exited=not monitor.thread.is_alive(),peak_scope='NVML sampled owned high-water, not instantaneous allocator maximum')
  raw['gpu_release']=release(baseline,[w.process.pid for w in runtime.workers.values() if w.process])
  raw['valid_session']=bool(raw['status']=='COMPLETED' and raw['gpu_release']['verified'] and not raw['errors'] and raw['telemetry']['collector_exited'] and len(raw.get('shutdown',{}))==len(mapping) and all(v['exit_code']==0 and not v['forced'] and v['bye_received'] for v in raw['shutdown'].values()))
  raw['elapsed_seconds']=time.perf_counter()-begin;save('raw_runs/'+path.name,raw)
 return raw
