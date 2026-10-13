import copy,json,threading,time,sys
from types import SimpleNamespace
from .common import *
from .telemetry import SafeMonitor

def reproduce():
 row={str(i):i for i in range(30000)};stop=threading.Event();found=[];old=sys.getswitchinterval();sys.setswitchinterval(.000001)
 def writer():
  i=0
  while not stop.is_set():
   row['new'+str(i%100)]=i;row.pop('new'+str((i+1)%100),None);i+=1
 t=threading.Thread(target=writer);t.start()
 try:
  for i in range(500):
   try:copy.deepcopy(row)
   except RuntimeError as exc:
    if 'dictionary changed size during iteration' in str(exc):found.append(repr(exc));break
 finally:stop.set();t.join(2);sys.setswitchinterval(old)
 assert found,'Expected dictionary race not reproduced'
 save('legacy_race_reproduction.json',dict(CPU_only=True,plain_dict_mutation=True,reproduced_error=found,source_failure_not_uniquely_identified_without_original_traceback=True))

def lifecycle():
 from joint_deploy.measurement.deployment_runtime import DeploymentRuntime
 from joint_deploy.measurement.schema import MeasurementConfig
 script=OUT/'fixtures/cpu_worker.py';script.parent.mkdir(exist_ok=True)
 script.write_text('''import json,sys,time
print('READY|'+json.dumps({'tasks':['text']}),flush=True)
for line in sys.stdin:
 r=json.loads(line)
 if r.get('op')=='shutdown':
  print('BYE|{}',flush=True);break
 start=time.time_ns();time.sleep(.001);end=time.time_ns()
 print('RESULT|'+json.dumps(dict(request_id=r['request_id'],task_type=r['task_type'],worker_latency_ms=(end-start)/1e6,queue_wait_ms=(start-r['submit_ns'])/1e6,start_ns=start,end_ns=end)),flush=True)
''')
 config=MeasurementConfig(deployment={'fixture':1},workload_rps={'text':1},routing_rps={'text':{'fixture':1}},shutdown_timeout_seconds=3.)
 runtime=DeploymentRuntime(config,{'fixture':{'python':sys.executable,'script':str(script)}})
 runtime.start();records=[]
 for i in range(200):
  r=dict(request_id=str(i),pool_id='test',arrival_index=i,task='text',model='fixture',status='planned');records.append(r);runtime.enqueue(r,{})
 runtime.drain(records,5.);report=runtime.close()
 assert all(r['status']=='completed' for r in records),(report,runtime.errors)
 assert not runtime.errors and all(v['exit_code']==0 and v['bye_received'] and not v['forced'] for v in report.values())
 assert not runtime.collector.is_alive() and all(not t.is_alive() for t in runtime.senders.values())
 assert all(w.reader_done.is_set() and w.process.poll()==0 for w in runtime.workers.values())
 save('cpu_lifecycle_audit.json',dict(passed=True,cohort_requests=len(records),completed=sum(r['status']=='completed' for r in records),shutdown=report,collector_and_senders_exited=True,readers_exited=True,GPU_measurements=0))

def main():reproduce();lifecycle();print('SHORT_AUDIT_PASSED',flush=True)
if __name__=='__main__':main()
