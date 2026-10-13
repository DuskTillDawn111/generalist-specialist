"""Bounded GPU experiment; immutable native inference workers and optimizer."""
from common_replication import *
from dynamic_router import DynamicRouter
from module22.gpu import PoissonConfig
from module11b.measure import prime
from module16.gpu import ProcessMonitor
from module26.gpu import thermal
from joint_deploy.measurement.deployment_runtime import DeploymentRuntime
from joint_deploy.measurement.run_measurement import new_record,wait_until
from joint_deploy.measurement.metrics import annotate_records,summarize
from performance_model.local_calibration.stability_v2 import reconstruct,classify,DEFAULT_THRESHOLDS
from replica_scaling.telemetry import snapshot,release
from module21.trajectory import choose
from types import SimpleNamespace
import copy,threading

class VisibleRuntime(DeploymentRuntime):
 def __init__(self,*args):
  super().__init__(*args);self.visible={};self.router=None;self.latest_received_ns=0
 def enqueue(self,record,sample):
  with self.lock:
   super().enqueue(record,sample);self.visible[record['request_id']]=record
 def arrival_snapshot(self,now_ns):
  # Existing native collector marks status completed only after RESULT received.
  pending=collections.defaultdict(list)
  with self.lock:
   for rid,r in list(self.visible.items()):
    if r['status']=='completed' and r['completion_time_ns']<=now_ns:
     self.latest_received_ns=max(self.latest_received_ns,r['completion_time_ns'])
     if self.router:self.router.observe_completion(r,now_ns/1e9)
     del self.visible[rid]
    elif r['status'] in ['queued','submitted']:
     pending[r['model']].append({k:r.get(k) for k in ['request_id','task','model','enqueued_time_ns','actual_submit_time_ns']})
   for rows in pending.values():rows.sort(key=lambda r:r['enqueued_time_ns'])
  return pending

class DurableMonitor(ProcessMonitor):
 def __init__(self,runtime,slot,budget):
  super().__init__(runtime,.5,budget);self.slot=slot;self.last_checkpoint=0.;self.last_progress=0.;self.persisted=set();self.phase_labels=set();self.active_phase=None;self.journal=OUT/'request_journals'/(slot['run_id']+'.jsonl');self.journal.parent.mkdir(parents=True,exist_ok=True)
 def sample(self):
  value=super().sample();value.update(thermal());active=self.active_phase
  if not active:return value
  records=active['request_records'];label=active['label'];elapsed=(time.perf_counter_ns()-active['planned_start_time_ns'])/1e9
  if time.monotonic()-self.last_checkpoint>=10:
   self.last_checkpoint=time.monotonic();rows=[]
   if label not in self.phase_labels:
    rows.append(dict(kind='PHASE_WINDOW',label=label,start_time_ns=active['planned_start_time_ns'],planned_end_time_ns=active['planned_end_time_ns'],planned_duration_seconds=active['planned_duration_seconds']));self.phase_labels.add(label)
   for r in list(records):
    if r['status'] in ['completed','error','incomplete'] and (label,r['request_id']) not in self.persisted:
     rows.append(dict(kind='REQUEST_TERMINAL',phase=label,record=copy.deepcopy(r)));self.persisted.add((label,r['request_id']))
   rows.append(dict(kind='CHECKPOINT',at_unix_ns=time.time_ns(),phase=label,telemetry=value,outstanding=[copy.deepcopy(r) for r in records if r['status'] in ['queued','submitted']],full_raw_pending=True))
   with self.journal.open('a') as stream:
    for row in rows:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False,default=json_native)+'\n')
  if time.monotonic()-self.last_progress>=30:
   self.last_progress=time.monotonic();save('live_progress.json',dict(run_id=self.slot['run_id'],stage=self.slot['stage'],phase=self.phase,elapsed_seconds=elapsed,planned_seconds=active['planned_duration_seconds'],planned=len(records),submitted=sum(bool(r.get('actual_submit_time_ns')) for r in records),completed=sum(r['status']=='completed' for r in records),temperature_gpu_c=value['temperature_gpu_c'],used_mib=value['owned_used_mib']))
  return value

def phase(runtime,config,slot,f,mapping,label,monitor):
 trajectory=json.loads(Path(slot['trajectories'][label]['path']).read_text());assert ref(slot['trajectories'][label]['path'])['sha256']==slot['trajectories'][label]['sha256']
 events=trajectory['events'];seconds=trajectory['duration'];start=time.perf_counter_ns();unix=time.time_ns();end=start+round(seconds*1e9);records=[];monitor.phase=label
 runtime.arrival_snapshot(start);runtime.router=DynamicRouter(f['parameters']) # same frozen initialization each phase/arm
 monitor.active_phase=dict(label=label,planned_start_time_ns=start,planned_end_time_ns=end,planned_duration_seconds=seconds,request_records=records)
 for i,e in enumerate(events):
  r=new_record(label,i,e['task'],choose(e,slot['routing'],slot['deployment']),e['item'],start+round(e['offset_seconds']*1e9));r.update(frozen_offset_seconds=e['offset_seconds'],routing_uniform=e['routing_uniform'],replica_uniform=e['replica_uniform']);records.append(r)
 for r,event in zip(records,events):
  wait_until(r['scheduled_time_ns']);monitor.check();assert not runtime.errors,runtime.errors
  if r['task']==f['target']:
   decision_start=time.perf_counter_ns();pending=runtime.arrival_snapshot(decision_start);router=runtime.router
   capacity_ok=not monitor.resource_failure and not runtime.errors and all(runtime.workers[w].process.poll() is None for w in [f['parameters']['G_worker'],f['parameters']['S_worker']])
   admitted=all(f['target'] in runtime.ready[w]['tasks'] for w in [f['parameters']['G_worker'],f['parameters']['S_worker']])
   if slot['arm']=='dynamic':worker,decision=router.decide(r['request_id'],decision_start/1e9,pending,admitted,capacity_ok)
   else:
    worker=choose(event,slot['routing'],slot['deployment']);decision=router.estimate(pending);decision.update(reason='FROZEN_S_ONLY' if slot['arm']=='s_only' else 'FROZEN_FIXED10_UNIFORM',route=worker,admitted=admitted,capacity_ok=capacity_ok)
   r['model']=worker;r['routing_decision']=dict(**decision,decision_time_ns=decision_start,latest_observed_completion_time_ns=runtime.latest_received_ns,scheduled_time_ns=r['scheduled_time_ns'],overhead_us=(time.perf_counter_ns()-decision_start)/1000,observation_scope='Host enqueued outstanding; only received RESULT service; no live START ACK')
   assert runtime.latest_received_ns<=decision_start
  runtime.enqueue(r,event['item']['sample'])
 wait_until(end);monitor.check();arrival_end=time.perf_counter_ns();monitor.phase='drain' if label=='measurement' else 'warmup_drain';runtime.drain(records,60.);drain=time.perf_counter_ns();runtime.arrival_snapshot(drain);annotate_records(records)
 window=dict(start_time_ns=start,start_unix_ns=unix,end_time_ns=end,planned_end_time_ns=end,duration_seconds=seconds,planned_duration_seconds=seconds,arrival_loop_end_time_ns=arrival_end,drain_end_time_ns=drain,drain_seconds=(drain-end)/1e9,stop=None,cohort='All frozen matched requests scheduled in [start,end) including full drain')
 per_task,system,models=summarize(records,config,window);temp=dict(request_records=records,measurement_window=window,status='completed' if all(r['status']=='completed' for r in records) else 'completed_with_failures',errors=runtime.errors,drain_timeout_seconds=60.,worker_identities={i:{'pid':w.process.pid} for i,w in runtime.workers.items()});aliases={i:SimpleNamespace(service_time_ms=f['plan']['models'][m]['service_time_ms']) for i,m in mapping.items()};ev=reconstruct(temp,aliases,DEFAULT_THRESHOLDS);monitor.active_phase=None
 return dict(window=window,request_records=records,per_task_metrics=per_task,system_metrics=system,stability=classify(ev,DEFAULT_THRESHOLDS),stability_evidence=ev,trajectory=slot['trajectories'][label],prediction_errors=runtime.router.prediction_errors)

def cool(slot,f,resident=False,monitor=None):
 policy=f['thermal_protocol'];limit=policy['resident_cool_timeout_seconds'] if resident else policy['max_wait_seconds'];maxc=policy['resident_premeasurement_temperature_max_c'] if resident else policy['prelaunch_temperature_max_c'];required=policy['resident_consecutive_samples'] if resident else policy['consecutive_idle_samples_required'];begin=time.monotonic();samples=[];consecutive=0
 while time.monotonic()-begin<limit:
  v=monitor.sample() if resident else snapshot();v.update(thermal());samples.append(v)
  if not resident and (v['processes'] or v['used_mib']>128):raise RuntimeError('PRELAUNCH_GPU_NOT_ISOLATED')
  consecutive=consecutive+1 if v['temperature_gpu_c']<=maxc else 0
  if consecutive>=required:return dict(passed=True,samples=samples,elapsed_seconds=time.monotonic()-begin)
  if monitor:monitor.check()
  time.sleep(2.)
 raise RuntimeError('THERMAL_LAUNCH_FAILURE')

def run(slot,f):
 path=OUT/'raw_runs'/(slot['run_id']+'.json');assert not path.exists(),'No result overwrite/retry';pre=cool(slot,f);save('thermal_prelaunch/'+slot['run_id']+'.json',pre);baseline=pre['samples'][-1];assert all(baseline[k]==f['GPU'][k] for k in ['uuid','name','driver','total_mib'])
 plan=f['plan']
 for k,v in plan['environment_overrides'].items():os.environ[k]=v
 mapping={f'{m}#0':m for m in slot['deployment']};routes={t:{m+'#0':q for m,q in row.items()} for t,row in slot['routing'].items()}
 config=PoissonConfig(deployment={i:1 for i in mapping},workload_rps=slot['lambda_rps'],routing_rps=routes,arrival_process='poisson',random_seed=slot['seed'],warmup_seconds=30.,measurement_seconds=slot['measurement_seconds'],startup_timeout_seconds=300.,drain_timeout_seconds=60.,shutdown_timeout_seconds=15.,routing_source='Module28 independent runtime routing; burst trajectories explicitly frozen')
 config.validate({i:plan['models'][m]['supported_tasks'] for i,m in mapping.items()})
 runtime=VisibleRuntime(config,{i:plan['registry'][m] for i,m in mapping.items()});monitor=DurableMonitor(runtime,slot,f['budget_mib']);raw=dict(slot=slot,status='STARTING',errors=[],phases={},baseline_GPU=baseline,replica_to_family=mapping,worker_identities={},transport_config=config.as_dict(),frozen_protocol=ref(OUT/'frozen_replication_manifest.json'));begin=time.perf_counter();monitor.start()
 try:
  runtime.start();monitor.check();raw['worker_identities']={i:dict(family=m,pid=runtime.workers[i].process.pid,READY=runtime.ready[i]) for i,m in mapping.items()};assert len({v['pid'] for v in raw['worker_identities'].values()})==len(mapping)
  raw['path_probe_records']=prime(runtime,config,plan,mapping,slot);monitor.check()
  raw['warmup']=phase(runtime,config,slot,f,mapping,'warmup',monitor);assert all(r['status']=='completed' for r in raw['warmup']['request_records'])
  monitor.phase='postwarm_thermal_control';raw['resident_thermal_control']=cool(slot,f,True,monitor)
  raw['postwarm_idle']=monitor.idle_window('postwarm_idle',2.);monitor.check();assert f['safe_factor']*np.mean([v['owned_used_mib'] for v in raw['postwarm_idle']])<=f['budget_mib']
  print('MEASUREMENT_START',slot['run_id'],slot['measurement_seconds'],flush=True);raw['phases']['measurement']=phase(runtime,config,slot,f,mapping,'measurement',monitor)
  raw['postmeasurement_idle']=monitor.idle_window('postmeasurement_idle',2.);monitor.check();raw['status']='COMPLETED'
 except (Exception,KeyboardInterrupt) as exc:
  raw['status']='FAILED';raw['errors'].append(repr(exc))
  if monitor.active_phase:raw['aborted_phase']=monitor.active_phase;raw['aborted_phase']['abort_time_ns']=time.perf_counter_ns()
 finally:
  try:raw['shutdown']=runtime.close()
  except Exception as exc:raw['errors'].append('shutdown: '+repr(exc))
  if raw.get('aborted_phase'):
   for r in raw['aborted_phase']['request_records']:
    if r['status'] in ['planned','queued','submitted']:r.update(status='incomplete',error='Aborted entire frozen cohort retained')
   annotate_records(raw['aborted_phase']['request_records'])
  monitor.close();raw['errors']+=runtime.errors+monitor.errors;raw['telemetry']=dict(samples=monitor.samples,errors=monitor.errors,interval_seconds=.5,peak_scope='NVML sampled owned high-water, not instantaneous allocator maximum');raw['gpu_release']=release(baseline,[w.process.pid for w in runtime.workers.values() if w.process]);raw['valid_session']=bool(raw['status']=='COMPLETED' and raw['gpu_release']['verified'] and not raw['errors'] and all(v['exit_code']==0 and not v['forced'] and v['bye_received'] for v in raw.get('shutdown',{}).values()));raw['elapsed_seconds']=time.perf_counter()-begin;save('raw_runs/'+path.name,raw)
 return raw
