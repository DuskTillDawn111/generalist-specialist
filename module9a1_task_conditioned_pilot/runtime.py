"""Controlled native-worker active-pair primitive; no deployment/routing optimizer."""
import argparse,itertools,json,os,queue,subprocess,threading,time,sys
from pathlib import Path
from types import SimpleNamespace
from .prepare import ROOT,OUT,load,save,sha
from joint_deploy.measurement.deployment_runtime import ObservedWorker
from integer_replica_pilot.runtime import SafetyMonitor
from replica_scaling.telemetry import snapshot,release
sys.path.insert(0,str(ROOT/'joint_deploy'))
from joint_deploy.run_active_pair_profile import merge_intervals,overlap_with_union

class ProfileWorker(ObservedWorker):
 def __init__(self,name,spec,events,output):super().__init__(name,spec,events);self.output=output
 def start(self):
  env=dict(os.environ,PYTHONUNBUFFERED='1',PILOT_NATIVE_WORKER=self.config['script'])
  cmd=['/usr/local/bin/nsys','profile','--trace=cuda,nvtx','--capture-range=cudaProfilerApi','--capture-range-end=stop','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--force-overwrite=false','-o',str(self.output),self.config['python'],str(ROOT/'module9a1_task_conditioned_pilot/profile_worker.py')]
  self.process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
  threading.Thread(target=self._reader,daemon=True).start()

def run(model,task,seed,co=None,profile=False):
 p=load(OUT/'measurement_protocol_freeze.json');assert model in p['generalists'] and task in p['tasks'][model] and seed in p['repetition_seeds'];assert co is None or co in p['co_models']
 kind='kernel' if profile else ('pair' if co else 'standalone');tag=f'{model}__{task}__{co or "none"}__{seed}';directory=OUT/('task_path_kernel_profiles' if profile else 'pair_measurements')/kind;directory.mkdir(parents=True,exist_ok=True);output=directory/(tag+'.json');assert not output.exists(),'No retry or overwrite'
 baseline=snapshot();carrier=load(ROOT/'results/module8f2_resource_static/campaign_manifest.json')['GPU'];assert not baseline['processes'] and baseline['used_mib']<=128 and all(baseline[k]==carrier[k] for k in ['name','driver','total_mib','uuid'])
 events=queue.Queue();workers={};ready={};results={};submitted={};monitor=None;attempt={'kind':kind,'model':model,'task':task,'co_model':co,'seed':seed,'protocol_sha256':sha(OUT/'measurement_protocol_freeze.json'),'started_at_unix':time.time(),'status':'STARTED','target_records':[],'co_records':[],'warmup_records':[],'profiler_timing_ground_truth':False}
 output.write_text(json.dumps(attempt,indent=2)+'\n');names=[model]+([co] if co else []);co_index=0;warm_index=0;refill=False;phase='load'
 def receive(timeout=.1):
  try:n,k,r=events.get(timeout=max(0,timeout))
  except queue.Empty:return None
  if k=='ERROR':raise RuntimeError(f'{n}: {r}')
  if k=='RESULT':
   rid=r['request_id'];record={**r,'model_family':n,'phase':submitted.get((n,rid),{}).get('phase'),'sample_index':submitted.get((n,rid),{}).get('sample_index')};results[(n,rid)]=record
   if str(rid).startswith('target:'):attempt['target_records'].append(record)
   elif str(rid).startswith('co:'):attempt['co_records'].append(record)
   elif str(rid).startswith('warm:'):attempt['warmup_records'].append(record)
   if refill and n==co and str(rid).startswith('co:'):send_co()
  if k=='EXIT' and phase not in ('shutdown','done'):raise RuntimeError(f'Unexpected worker exit {n}: {r}')
  if monitor:monitor.check()
  return n,k,r
 def send(n,item,rid,sample_index=None):
  request={'request_id':rid,'pool_id':item['pool_id'],'arrival_index':sample_index if sample_index is not None else -1,'task_type':item['task_type'],'sample':item['sample'],'submit_ns':time.time_ns()};submitted[(n,rid)]={'phase':phase,'sample_index':sample_index};workers[n].send(request)
 def one(n,item,rid,index=None):
  send(n,item,rid,index);deadline=time.monotonic()+p['startup_timeout_seconds']
  while (n,rid) not in results:
   if time.monotonic()>deadline:raise TimeoutError(f'Request timeout {n}/{rid}')
   receive()
  return results[(n,rid)]
 def send_co():
  nonlocal co_index
  ct=p['co_tasks'][co][co_index%len(p['co_tasks'][co])];items=p['input_subsets'][str(seed)][ct];item=items[(co_index//len(p['co_tasks'][co]))%len(items)];send(co,item,f'co:{co_index}',co_index);co_index+=1
 def wait_until(t):
  while time.monotonic()<t:receive(min(.05,max(0,t-time.monotonic())))
 try:
  for n in names:
   workers[n]=ProfileWorker(n,p['registry'][n],events,directory/tag) if profile else ObservedWorker(n,p['registry'][n],events);workers[n].start();deadline=time.monotonic()+p['startup_timeout_seconds']
   while n not in ready:
    event=receive()
    if event and event[0]==n and event[1]=='READY':ready[n]=event[2]
    if time.monotonic()>deadline:raise TimeoutError(f'READY timeout {n}')
  holder=SimpleNamespace(workers={n:SimpleNamespace(process=SimpleNamespace(pid=r['pid'])) for n,r in ready.items()});monitor=SafetyMonitor(holder,.25,49140);monitor.start();attempt['ready']=ready;attempt['idle_after_load']=monitor.idle('idle_after_load');phase='warmup';monitor.phase=phase;start=time.monotonic()
  # Native-path warmup serial and excluded; all co-task paths explicitly primed.
  warm_paths=[(model,task)]+([(co,k) for k in p['co_tasks'][co]] if co else [])
  while time.monotonic()-start<p['warmup_seconds'] or warm_index<len(warm_paths):
   n,k=warm_paths[warm_index%len(warm_paths)];items=p['input_subsets'][str(seed)][k];one(n,items[(warm_index//len(warm_paths))%len(items)],f'warm:{warm_index}');warm_index+=1
  attempt['warmup_actual_seconds']=time.monotonic()-start;attempt['warmup_configured_seconds']=p['warmup_seconds'];attempt['idle_before_measurement']=monitor.idle('idle_before_measurement');phase='measurement';monitor.phase=phase
  if profile:
   one(model,{'task_type':'__profile_start__','pool_id':'__profile_start__','sample':{}},'control:start')
   for i,item in enumerate(p['input_subsets'][str(seed)][task][:p['signature_profile_samples']]):one(model,item,f'target:{i}',i)
   one(model,{'task_type':'__pilot_profile_stop__','pool_id':'__profile_stop__','sample':{}},'control:stop')
   attempt['measurement_duration_rule']='metadata capture of exactly 3 matched inputs; profiled timing diagnostic only'
  else:
   if co:
    phase='co_priming';refill=True
    for _ in range(p['co_backlog']):send_co()
    deadline=time.monotonic()+p['startup_timeout_seconds']
    while not attempt['co_records']:
     if time.monotonic()>deadline:raise TimeoutError('Co-model priming timeout')
     receive()
   phase='measurement';start=time.monotonic();end=start+p['measurement_seconds'];attempt['measurement_start_unix_ns']=time.time_ns();attempt['measurement_seconds']=p['measurement_seconds']
   for i,(offset,item) in enumerate(zip(p['target_schedule_seconds'],p['input_subsets'][str(seed)][task])):
    wait_until(start+offset);send(model,item,f'target:{i}',i)
   wait_until(end);phase='drain';monitor.phase='drain';deadline=time.monotonic()+p['drain_timeout_seconds']
   while len(attempt['target_records'])<p['target_sample_count']:
    if time.monotonic()>deadline:raise TimeoutError('Target drain timeout')
    receive()
   refill=False
   while co and len(attempt['co_records'])<co_index:
    if time.monotonic()>deadline:raise TimeoutError('Co-backlog drain timeout')
    receive()
   attempt['actual_window_elapsed_seconds']=time.monotonic()-start;attempt['configured_window_completed']=True
  if co:
   intervals=merge_intervals([(r['start_ns'],r['end_ns']) for r in attempt['co_records']]);attempt['overlap_filter_applied']=False
   for r in attempt['target_records']:
    r['active_overlap_ratio']=overlap_with_union(r['start_ns'],r['end_ns'],intervals)/(r['end_ns']-r['start_ns'])
   attempt['all_targets_meet_overlap_gate']=all(r['active_overlap_ratio']>=p['minimum_overlap_ratio'] for r in attempt['target_records'])
  attempt['status']='COMPLETE'
 except BaseException as exc:attempt['status']='FAILED';attempt['error']=f'{type(exc).__name__}: {exc}'
 finally:
  phase='shutdown';refill=False
  if monitor:monitor.phase='shutdown'
  shutdown={}
  for n,w in workers.items():
   try:
    if w.process.poll() is None:w.shutdown()
    w.process.wait(timeout=120 if profile else 20);shutdown[n]={'exit_code':w.process.returncode,'forced':False}
   except Exception as exc:
    w.process.terminate()
    try:w.process.wait(timeout=10)
    except subprocess.TimeoutExpired:w.process.kill();w.process.wait(timeout=10)
    shutdown[n]={'exit_code':w.process.returncode,'forced':True,'error':str(exc)};attempt['status']='FAILED'
  if monitor:monitor.close();attempt['telemetry']=monitor.samples;attempt['telemetry_errors']=monitor.errors
  attempt['shutdown']=shutdown;attempt['gpu_release']=release(baseline,[r['pid'] for r in ready.values()]);attempt['actual_peak_vram_mib']=max((r['used_mib'] for r in attempt.get('telemetry',[])),default=None);attempt['finished_at_unix']=time.time();attempt['target_records'].sort(key=lambda r:r['sample_index']);attempt['raw_timing_used_for_slowdown']=not profile
  if not attempt['gpu_release']['verified'] or any(v['exit_code']!=0 or v['forced'] for v in shutdown.values()):attempt['status']='FAILED'
  output.write_text(json.dumps(attempt,indent=2,ensure_ascii=False)+'\n')
 if attempt['status']!='COMPLETE':raise RuntimeError(attempt.get('error','Worker shutdown/release failure'))
 if profile:
  rep=directory/(tag+'.nsys-rep');assert rep.exists(),'Missing Nsight report'
  subprocess.run(['/usr/local/bin/nsys','export','--type=sqlite','--output',str(directory/(tag+'.sqlite')),str(rep)],check=True)
 return output

def main():
 a=argparse.ArgumentParser();a.add_argument('--model',required=True);a.add_argument('--task',required=True);a.add_argument('--seed',required=True,type=int);a.add_argument('--co');a.add_argument('--profile',action='store_true');args=a.parse_args();run(args.model,args.task,args.seed,args.co,args.profile)
if __name__=='__main__':main()
