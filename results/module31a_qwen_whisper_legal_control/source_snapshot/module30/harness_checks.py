"""Native CPU transport through the exact new phase+journal; no GPU inference."""
import os,time,threading
from types import SimpleNamespace
from .common import *
from .telemetry import SafeMonitor
from .gpu import AuditedRuntime,phase
from .stress import sensor
from joint_deploy.measurement.schema import MeasurementConfig

def atomic_pressure():
 errors=[];stop=threading.Event();name='cpu_atomic_pressure.json'
 def writer(k):
  try:
   for i in range(80):save(name,dict(writer=k,version=i,payload=[k,i]*100))
  except Exception as exc:errors.append(repr(exc))
 def reader():
  while not stop.is_set():
   try:
    p=OUT/name
    if p.exists():
     d=read(p);assert d['payload']==[d['writer'],d['version']]*100
   except Exception as exc:errors.append(repr(exc))
 t=threading.Thread(target=reader);t.start();threads=[threading.Thread(target=writer,args=(k,)) for k in range(4)]
 for tt in threads:tt.start()
 for tt in threads:tt.join(20);assert not tt.is_alive()
 stop.set();t.join(2);assert not errors,errors
 return dict(passed=True,concurrent_writers=4,writes=320,parse_errors=errors)

def run(abort=False):
 cid='cpu_harness_abort' if abort else 'cpu_harness_full'
 events=[dict(offset_seconds=i*.025,task='text',routing_uniform=.5,replica_uniform=.5,item=dict(pool_id='cpu_pool',task_type='text',sample=dict(id=str(i)))) for i in range(80)]
 path='fixtures/'+cid+'.json';save(path,dict(events=events,duration=2.))
 slot=dict(run_id=cid,routing={'text':{'fixture':40}},deployment={'fixture':1},trajectories={'measurement':ref(OUT/path)})
 cfg=MeasurementConfig(deployment={'fixture#0':1},workload_rps={'text':40},routing_rps={'text':{'fixture#0':40}},measurement_seconds=2.,shutdown_timeout_seconds=3.)
 runtime=AuditedRuntime(cfg,{'fixture#0':{'python':os.sys.executable,'script':str(OUT/'fixtures/cpu_worker.py')}})
 monitor=SafeMonitor(runtime,slot,1,interval=.003,checkpoint_seconds=.025,sensor=sensor);runtime.start();monitor.start()
 if abort:
  old=monitor.check;calls=[0]
  def check():
   calls[0]+=1
   if calls[0]==41:raise RuntimeError('INJECTED_PHASE_ABORT')
   old()
  monitor.check=check
 observed_abort=False;ph=None;aborted=None
 try:ph=phase(runtime,cfg,slot,{'models':{'fixture':{'service_time_ms':{'text':1}}}}, {'fixture#0':'fixture'},'measurement',monitor)
 except RuntimeError as exc:
  assert abort and str(exc)=='INJECTED_PHASE_ABORT'
  observed_abort=True;aborted=monitor.active_phase
 finally:
  report=runtime.close()
  if aborted:
   with runtime.lock:
    for row in aborted['request_records']:
     if row['status'] in ('planned','queued','submitted'):row.update(status='incomplete',error='Retained planned cohort after injected abort')
   monitor.flush_active()
  monitor.close()
 assert not runtime.errors and not monitor.errors
 rs=ph['request_records'] if ph else aborted['request_records'];journal=[]
 with monitor.journal.open() as stream:
  for line in stream:
   d=json.loads(line)
   if d['kind']=='REQUEST_TERMINAL':journal.append(d['record'])
 assert len(rs)==len(journal)==80 and len({r['request_id'] for r in journal})==80
 assert all(j['status']==r['status'] for j,r in zip(sorted(journal,key=lambda r:r['arrival_index']),rs))
 if abort:assert observed_abort and sum(r['status']=='incomplete' for r in rs)==40
 else:assert all(r['status']=='completed' for r in rs)
 return dict(passed=True,planned=len(rs),completed=sum(r['status']=='completed' for r in rs),incomplete=sum(r['status']=='incomplete' for r in rs),journal_terminal=len(journal),shutdown=report,abort_observed=observed_abort)

def main():
 full=run();aborted=run(True);atomic=atomic_pressure()
 save('cpu_harness_acceptance.json',dict(passed=True,complete_phase=full,aborted_phase=aborted,atomic_file_pressure=atomic,GPU_measurements=0))
 print('CPU_HARNESS_ACCEPTANCE_PASSED',flush=True)
if __name__=='__main__':main()
