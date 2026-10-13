"""Native GPU runtime, frozen Poisson cohorts, durable incremental evidence."""
from .common import *
import subprocess
from module22.gpu import phase,PoissonConfig
from module11b.measure import prime
from module16.gpu import ProcessMonitor,memory_values
from joint_deploy.measurement.deployment_runtime import DeploymentRuntime
from joint_deploy.measurement.metrics import annotate_records
from replica_scaling.telemetry import snapshot,release
from module26.gpu import thermal

def cool(slot,f):
 samples=[];consecutive=0;policy=f['thermal_protocol'];begin=time.monotonic()
 while time.monotonic()-begin<=policy['max_wait_seconds']:
  value=snapshot();value.update(thermal());samples.append(value)
  if value['processes'] or value['used_mib']>128:raise RuntimeError('PRELAUNCH_GPU_NOT_ISOLATED')
  consecutive=consecutive+1 if value['temperature_gpu_c']<=policy['prelaunch_temperature_max_c'] else 0
  if consecutive>=policy['consecutive_idle_samples_required']:
   save('thermal_prelaunch/'+slot['run_id']+'.json',dict(passed=True,samples=samples));return value
  time.sleep(policy['between_samples_seconds'])
 save('thermal_prelaunch/'+slot['run_id']+'.json',dict(passed=False,samples=samples));raise RuntimeError('PRELAUNCH_THERMAL_FAILURE')

class DurableMonitor(ProcessMonitor):
 def __init__(self,runtime,slot):
  super().__init__(runtime,.5,BUDGET);self.slot=slot;self.last_progress=0.;self.last_checkpoint=0.;self.persisted=set();self.phases=set()
  self.journal=OUT/'request_journals'/(slot['run_id']+'.jsonl');self.journal.parent.mkdir(parents=True,exist_ok=True)
 def sample(self):
  value=super().sample();value.update(thermal());active=getattr(self,'active_phase',None)
  if not active:return value
  records=active['request_records'];elapsed=(time.perf_counter_ns()-active['planned_start_time_ns'])/1e9
  if time.monotonic()-self.last_checkpoint>=10:
   self.last_checkpoint=time.monotonic();rows=[];label=active['label']
   if label not in self.phases:
    rows.append(dict(kind='PHASE_WINDOW',label=label,start_time_ns=active['planned_start_time_ns'],planned_end_time_ns=active['planned_end_time_ns'],planned_duration_seconds=active['planned_duration_seconds']));self.phases.add(label)
   pending=[]
   for record in records:
    key=(label,record['request_id'])
    if record['status'] in ['completed','error','incomplete']:
     if key not in self.persisted:rows.append(dict(kind='REQUEST_TERMINAL',phase=label,record=copy.deepcopy(record)));self.persisted.add(key)
    elif record.get('actual_submit_time_ns'):pending.append(copy.deepcopy(record))
   rows.append(dict(kind='CHECKPOINT',at_unix_ns=time.time_ns(),phase=label,telemetry=value,pending=pending,full_raw_pending=True))
   with self.journal.open('a') as stream:
    for row in rows:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False,default=json_native)+'\n')
   save('durable_heartbeat.json',dict(at_utc=now(),controller_pid=os.getpid(),run_id=self.slot['run_id'],phase=label,elapsed_seconds=elapsed))
  if time.monotonic()-self.last_progress>=30:
   self.last_progress=time.monotonic();submitted=sum(bool(r.get('actual_submit_time_ns')) for r in records);done=sum(r['status']=='completed' for r in records)
   save('live_progress.json',dict(run_id=self.slot['run_id'],stage=self.slot['stage'],phase=self.phase,elapsed_seconds=elapsed,
    planned_seconds=active['planned_duration_seconds'],planned=len(records),submitted=submitted,completed=done,outstanding=submitted-done,
    temperature_gpu_c=value['temperature_gpu_c'],used_mib=value['owned_used_mib']))
  return value

def resident_cool(raw,monitor,f):
 policy=f['thermal_protocol'];begin=time.monotonic();samples=[];consecutive=0;monitor.phase='postwarm_thermal_control'
 while time.monotonic()-begin<=policy['resident_cool_timeout_seconds']:
  monitor.check();v=monitor.sample();samples.append(v)
  consecutive=consecutive+1 if v['temperature_gpu_c']<=policy['resident_premeasurement_temperature_max_c'] else 0
  if consecutive>=policy['resident_consecutive_samples']:
   raw['resident_thermal_control']=dict(passed=True,samples=samples,elapsed_seconds=time.monotonic()-begin);return
  time.sleep(2.)
 raw['resident_thermal_control']=dict(passed=False,samples=samples);raise RuntimeError('RESIDENT_PREMEASUREMENT_THERMAL_FAILURE')

def idle(raw,label,monitor,f):
 raw[label]=monitor.idle_window(label,2.);monitor.check()
 value=f['safe_factor']*np.mean([r['owned_used_mib'] for r in raw[label]])
 if value>BUDGET+1e-6:raise RuntimeError(f'SAFE_MEMORY_EXCEEDED {label}: {value}')

def run(slot,f):
 path=OUT/'raw_runs'/(slot['run_id']+'.json');assert not path.exists(),'No outcome retry or overwrite'
 baseline=cool(slot,f);plan=f['plan'];assert all(baseline[k]==plan['GPU'][k] for k in ['uuid','name','driver'])
 for k,v in plan['environment_overrides'].items():os.environ[k]=v
 assert not os.environ.get('CUDA_VISIBLE_DEVICES')
 mapping={f'{m}#{i}':m for m,n in slot['deployment'].items() for i in range(n)}
 routes={t:{i:q/slot['deployment'][m] for m,q in rr.items() for i,family in mapping.items() if family==m} for t,rr in slot['routing'].items()}
 config=PoissonConfig(deployment={i:1 for i in mapping},workload_rps=slot['lambda_rps'],routing_rps=routes,arrival_process='poisson',random_seed=slot['seed'],
  warmup_seconds=30.,measurement_seconds=slot['measurement_seconds'],startup_timeout_seconds=300.,drain_timeout_seconds=60.,shutdown_timeout_seconds=15.,
  routing_source='Frozen per-task Poisson; shared inputs/route and replica uniforms; independent stageB/C seeds')
 config.validate({i:plan['models'][m]['supported_tasks'] for i,m in mapping.items()})
 runtime=DeploymentRuntime(config,{i:plan['registry'][m] for i,m in mapping.items()});monitor=DurableMonitor(runtime,slot)
 raw=dict(slot=slot,status='STARTING',errors=[],phases={},baseline_GPU=baseline,replica_to_family=mapping,worker_identities={},
  transport_config=config.as_dict(),frozen_protocol=ref(OUT/'discovery_protocol.json'));begin=time.perf_counter();monitor.start()
 try:
  runtime.start();monitor.check();raw['worker_identities']={i:dict(family=m,pid=runtime.workers[i].process.pid,READY=runtime.ready[i]) for i,m in mapping.items()}
  assert len({v['pid'] for v in raw['worker_identities'].values()})==sum(slot['deployment'].values())
  raw['path_probe_records']=prime(runtime,config,plan,mapping,slot);monitor.check()
  raw['warmup']=phase(runtime,config,slot,plan,mapping,'warmup',monitor)
  assert all(r['status']=='completed' for r in raw['warmup']['request_records']),'Warmup incomplete'
  resident_cool(raw,monitor,f);idle(raw,'postwarm_idle',monitor,f)
  print('MEASUREMENT_START',slot['run_id'],slot['measurement_seconds'],flush=True)
  raw['phases']['measurement']=phase(runtime,config,slot,plan,mapping,'measurement',monitor)
  idle(raw,'postmeasurement_idle',monitor,f);raw['status']='COMPLETED'
 except (Exception,KeyboardInterrupt) as exc:
  raw['status']='FAILED';raw['errors'].append(repr(exc))
  if getattr(monitor,'active_phase',None):raw['aborted_phase']=monitor.active_phase;raw['aborted_phase']['abort_time_ns']=time.perf_counter_ns()
 finally:
  try:raw['shutdown']=runtime.close()
  except Exception as exc:raw['errors'].append('shutdown: '+repr(exc))
  if raw.get('aborted_phase'):
   for r in raw['aborted_phase']['request_records']:
    if r['status'] in ['planned','queued','submitted']:r.update(status='incomplete',error='Entire planned cohort retained; no completed-subset normal U')
   annotate_records(raw['aborted_phase']['request_records'])
  monitor.close();raw['errors']+=runtime.errors+monitor.errors;raw['telemetry']=dict(samples=monitor.samples,errors=monitor.errors,interval_seconds=.5,
   peak_scope='Sampled NVML owned-process high-water; allocator instantaneous peak unavailable')
  raw['gpu_release']=release(baseline,[w.process.pid for w in runtime.workers.values() if w.process])
  raw['valid_session']=bool(raw['status']=='COMPLETED' and raw['gpu_release']['verified'] and not raw['errors'] and all(v['exit_code']==0 and not v['forced'] and v['bye_received'] for v in raw.get('shutdown',{}).values()))
  raw['elapsed_seconds']=time.perf_counter()-begin;save('raw_runs/'+path.name,raw)
 return raw
