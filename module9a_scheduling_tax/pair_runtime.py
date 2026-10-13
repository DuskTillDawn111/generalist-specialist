"""Unprofiled measurement harness; native workers and runtime stay unchanged."""
import argparse,os,queue,subprocess,sys,threading,time
from types import SimpleNamespace
from .common import *
sys.path.insert(0,str(ROOT/'joint_deploy'))
from joint_deploy.measurement.deployment_runtime import ObservedWorker
from integer_replica_pilot.runtime import SafetyMonitor
from replica_scaling.telemetry import snapshot,release
from joint_deploy.run_active_pair_profile import merge_intervals,overlap_with_union

class Diagnostics:
 def __init__(self):self.samples=[];self.errors=[];self.stop=threading.Event();self.thread=threading.Thread(target=self.collect,daemon=True)
 def collect(self):
  while not self.stop.is_set():
   try:
    v=subprocess.check_output(['nvidia-smi','--query-gpu=clocks.sm,clocks.mem,temperature.gpu,pstate','--format=csv,noheader,nounits'],text=True,timeout=10).strip().split(',');self.samples.append({'unix_ns':time.time_ns(),'sm_clock_mhz':float(v[0]),'memory_clock_mhz':float(v[1]),'temperature_c':float(v[2]),'pstate':v[3].strip()})
   except Exception as e:self.errors.append(str(e))
   self.stop.wait(1)
 def start(self):self.thread.start()
 def close(self):self.stop.set();self.thread.join(timeout=12)

def run(key,rep):
 p=load(OUT/'protocol_identity.json');s=p['states'][key];output=OUT/'raw_runs'/f'{key}__rep{rep}.json';assert not output.exists(),'No overwrite or GPU retry';output.parent.mkdir(exist_ok=True)
 baseline=snapshot();assert not baseline['processes'] and baseline['used_mib']<=128,'GPU isolation failure'
 assert all(baseline[k]==p['GPU'][k] for k in ['name','driver','total_mib']),'GPU environment drift'
 attempt={'state_key':key,'repetition':rep,'protocol_sha256':sha(OUT/'protocol_identity.json'),'started_at_unix':time.time(),'baseline_GPU':baseline,'protocol_family':s['protocol_family'],'status':'STARTED','target_records':[],'co_records':[],'warmup_records':[],'priming_records':[],'submission_records':[],'ready':{},'co_backlog_trace':[],'frozen_subset_changed':False}
 output.write_text(json.dumps(attempt,indent=2)+'\n');events=queue.Queue();workers={};results={};submitted={};monitor=None;diag=Diagnostics();phase='load';refill=False;coindex=0;co_completed=0;target_worker=s['target_worker'];is_replay=s['protocol_family']=='historical_full_state_replay';co=s['co_model']
 def receive(timeout=.05):
  nonlocal co_completed
  try:n,k,r=events.get(timeout=max(timeout,0))
  except queue.Empty:return None
  if k=='ERROR':raise RuntimeError(f'{n}: {r}')
  if k=='RESULT':
   rid=r['request_id'];meta=submitted[(n,rid)];record={**r,**meta,'worker_identity':n};results[(n,rid)]=record
   attempt[{'target':'target_records','co':'co_records','warmup':'warmup_records','priming':'priming_records'}[meta['category']]].append(record)
   if meta['category']=='co' and not is_replay:
    co_completed+=1
    if refill:send_co()
  if k=='EXIT' and phase not in ['shutdown','done']:raise RuntimeError(f'Unexpected worker exit {n}: {r}')
  if monitor:monitor.check()
  return n,k,r
 def send(n,item,rid,category,index,offset=None):
  stamp=time.time_ns();meta={'category':category,'sample_index':index,'phase':phase,'scheduled_offset_seconds':offset};submitted[(n,rid)]=meta
  attempt['submission_records'].append({'worker':n,'request_id':rid,'pool_id':item['pool_id'],'task':item['task_type'],'unix_ns':stamp,**meta})
  workers[n].send({'request_id':rid,'pool_id':item['pool_id'],'arrival_index':index,'task_type':item['task_type'],'sample':item['sample'],'submit_ns':stamp})
 def one(n,item,rid,category,index):
  send(n,item,rid,category,index);deadline=time.monotonic()+300
  while (n,rid) not in results:
   if time.monotonic()>deadline:raise TimeoutError(rid)
   receive()
 def send_co():
  nonlocal coindex
  task=s['co_tasks'][coindex%len(s['co_tasks'])];items=s['co_items'][task];item=items[(coindex//len(s['co_tasks']))%len(items)];send(co,item,f'co:{coindex}','co',coindex);coindex+=1
 def wait_until(deadline):
  while time.monotonic()<deadline:receive(min(.05,max(0,deadline-time.monotonic())))
 def replay(sequence,duration,label):
  start=time.monotonic();attempt[label+'_start_unix_ns']=time.time_ns();target_index=0
  for i,r in enumerate(sequence):
   wait_until(start+r['offset_seconds']);istarget=label=='measurement' and r['worker']==target_worker and r['item']['task_type']==s['task'];category=('target' if istarget else 'co') if label=='measurement' else 'warmup';index=target_index if istarget else i
   if istarget:target_index+=1
   send(r['worker'],r['item'],f'{label}:{i}',category,index,r['offset_seconds'])
  wait_until(start+duration);deadline=time.monotonic()+60
  while any((n,rid) not in results for n,rid in submitted):
   if time.monotonic()>deadline:raise TimeoutError(label+' drain')
   receive()
  attempt[label+'_actual_elapsed_seconds']=time.monotonic()-start
 try:
  diag.start()
  for n,spec in s['worker_specs'].items():
   workers[n]=ObservedWorker(n,spec,events);t=time.monotonic();workers[n].start();deadline=t+300
   while n not in attempt['ready']:
    r=receive()
    if r and r[0]==n and r[1]=='READY':attempt['ready'][n]=r[2];attempt.setdefault('startup_seconds',{})[n]=time.monotonic()-t
    if time.monotonic()>deadline:raise TimeoutError('READY '+n)
  holder=SimpleNamespace(workers={n:SimpleNamespace(process=SimpleNamespace(pid=r['pid'])) for n,r in attempt['ready'].items()});monitor=SafetyMonitor(holder,.25,s['budget_mib']);monitor.start();attempt['idle_after_load']=monitor.idle('idle_after_load');phase='warmup';monitor.phase=phase;t=time.monotonic()
  for i,r in enumerate(s.get('priming_sequence',[])):one(r['worker'],r['item'],f'prime:{i}','priming',i)
  if is_replay:replay(s['warmup_sequence'],s['warmup_seconds'],'warmup')
  else:
   for i,r in enumerate(s['warmup_sequence']):one(r['worker'],r['item'],f'warm:{i}','warmup',i)
  attempt['warmup_actual_seconds']=time.monotonic()-t;attempt['warmup_frozen_count']=s['warmup_count'];assert len(attempt['warmup_records'])==s['warmup_count'];attempt['idle_before_measurement']=monitor.idle('idle_before_measurement')
  phase='measurement';monitor.phase=phase
  if is_replay:
   replay(s['measurement_schedule'],s['measurement_seconds'],'measurement')
  else:
   if co:
    phase='co_priming';refill=True
    for _ in range(s['co_backlog']):send_co()
    deadline=time.monotonic()+300
    while not attempt['co_records']:
     if time.monotonic()>deadline:raise TimeoutError('co priming')
     receive()
   phase='measurement';start=time.monotonic();attempt['measurement_start_unix_ns']=time.time_ns()
   for i,(offset,item) in enumerate(zip(s['target_schedule_seconds'],s['target_items'])):
    wait_until(start+offset);attempt['co_backlog_trace'].append({'target_index':i,'outstanding_co_requests':coindex-co_completed,'co_submitted':coindex,'co_completed':co_completed});send(target_worker,item,f'target:{i}','target',i,offset)
   wait_until(start+s['measurement_seconds']);phase='drain';monitor.phase=phase;refill=False;deadline=time.monotonic()+s['drain_timeout_seconds']
   while any((n,rid) not in results for n,rid in submitted):
    if time.monotonic()>deadline:raise TimeoutError('measurement drain')
    receive()
   attempt['measurement_actual_elapsed_seconds']=time.monotonic()-start
  attempt['measurement_seconds']=s['measurement_seconds'];attempt['configured_window_completed']=True
  if co:
   intervals=merge_intervals([(r['start_ns'],r['end_ns']) for r in attempt['co_records'] if r['worker_identity']!=target_worker]);attempt['overlap_filter_applied']=False
   for r in attempt['target_records']:r['active_overlap_ratio']=overlap_with_union(r['start_ns'],r['end_ns'],intervals)/(r['end_ns']-r['start_ns'])
   if not is_replay:assert all(r['active_overlap_ratio']>=s['minimum_overlap_ratio'] for r in attempt['target_records']),'Existing overlap gate failure'
  assert len(attempt['target_records'])==len(s['target_items']);assert len({r['pid'] for r in attempt['ready'].values()})==len(workers)
  attempt['status']='COMPLETE'
 except BaseException as e:attempt['status']='FAILED';attempt['error']=f'{type(e).__name__}: {e}'
 finally:
  phase='shutdown';refill=False;report={}
  for n,w in workers.items():
   try:
    if w.process.poll() is None:w.shutdown()
    w.process.wait(timeout=20);report[n]={'exit_code':w.process.returncode,'forced':False}
   except Exception as e:
    w.process.terminate()
    try:w.process.wait(timeout=10)
    except subprocess.TimeoutExpired:w.process.kill();w.process.wait(timeout=10)
    report[n]={'exit_code':w.process.returncode,'forced':True,'error':str(e)};attempt['status']='FAILED'
  if monitor:monitor.close();attempt['telemetry']=monitor.samples;attempt['telemetry_errors']=monitor.errors
  diag.close();attempt['clock_thermal_diagnostics']=diag.samples;attempt['diagnostic_errors']=diag.errors;attempt['shutdown']=report;attempt['gpu_release']=release(baseline,[r['pid'] for r in attempt['ready'].values()]);attempt['actual_peak_vram_mib']=max((r['used_mib'] for r in attempt.get('telemetry',[])),default=None);attempt['finished_at_unix']=time.time();attempt['target_records'].sort(key=lambda r:r['sample_index'])
  if not attempt['gpu_release']['verified'] or any(v['exit_code']!=0 or v['forced'] for v in report.values()):attempt['status']='FAILED'
  output.write_text(json.dumps(attempt,ensure_ascii=False,indent=2)+'\n')
 if attempt['status']!='COMPLETE':raise RuntimeError(attempt.get('error','shutdown/release failure'))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--state',required=True);a.add_argument('--rep',type=int,required=True);v=a.parse_args();run(v.state,v.rep)
