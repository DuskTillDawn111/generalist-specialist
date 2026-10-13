"""CPU-only concurrency, durability, failure injection and >1997s soak."""
import argparse,copy,json,threading,time,sys
from types import SimpleNamespace
from unittest.mock import patch
from .common import OUT,save
from .telemetry_v2 import SafeMonitorV2 as SafeMonitor

def sensor():
 time.sleep(.025)
 return dict(time_ns=time.perf_counter_ns(),used_mib=0,owned_used_mib=0,foreign_pids=[],temperature_gpu_c=30)
def test_failures():
 r=SimpleNamespace(lock=threading.RLock(),workers={},ready={})
 m=SafeMonitor(r,{'run_id':'stress_v2_failure'},1,sensor=lambda:(_ for _ in ()).throw(OSError('injected_sensor_failure')),interval=.001)
 m.start();m.thread.join(2);m.close()
 try:m.check();raise AssertionError('silent failure')
 except RuntimeError:pass
 assert 'injected_sensor_failure' in m.error_details[0]['traceback']
 m=SafeMonitor(r,{'run_id':'stress_v2_disk_failure'},1,sensor=sensor)
 m.set_active(dict(label='test',planned_start_time_ns=time.perf_counter_ns(),planned_duration_seconds=1,request_records=[{'request_id':'x','status':'completed'}]))
 with patch('module30.telemetry.os.fsync',side_effect=OSError('injected_disk_failure')):
  try:m.sample();raise AssertionError('silent disk failure')
  except OSError:pass
 assert not m.persisted and not m.samples
 # Bounded shutdown fault must be visible.
 m=SafeMonitor(r,{'run_id':'stress_v2_exit_failure'},1,sensor=sensor)
 m.thread=SimpleNamespace(ident=1,join=lambda timeout:None,is_alive=lambda:True)
 try:m.close();raise AssertionError('silent hung collector')
 except RuntimeError:pass
 return dict(sensor_failure_visible=True,disk_failure_does_not_commit=True,hung_collector_visible=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=2100);a=p.parse_args()
 failure_tests=test_failures();check_times=[]
 runtime=SimpleNamespace(lock=threading.RLock(),workers={},ready={})
 rows=[dict(request_id=str(i),status='submitted',actual_submit_time_ns=1,version=0,worker_response={'version':0}) for i in range(64)]
 m=SafeMonitor(runtime,{'run_id':'stress_v2_long'},1,interval=.005,checkpoint_seconds=.02,sensor=sensor)
 m.set_active(dict(label='test',planned_start_time_ns=time.perf_counter_ns(),planned_duration_seconds=a.seconds,request_records=rows))
 stop=threading.Event();counts=[0,0];errors=[]
 def mutate(k):
  try:
   while not stop.is_set():
    with runtime.lock:
     row=rows[counts[k]%len(rows)];v=row['version']+1;row['version']=v;row['worker_response']={'version':v}
     row['mutable_field_'+str(v%3)]=v
     row.pop('mutable_field_'+str((v+1)%3),None)
     row['status']='submitted'
     counts[k]+=1
    time.sleep(.0005)
  except Exception as exc:errors.append(repr(exc))
 def synchronous():
  try:
   while not stop.is_set():
    m.sample();begin_check=time.perf_counter();m.check();check_times.append(time.perf_counter()-begin_check);_,rs=m.coherent_records();assert all(r['version']==r['worker_response']['version'] for r in rs)
    time.sleep(.007)
  except Exception as exc:errors.append(repr(exc))
 threads=[threading.Thread(target=mutate,args=(k,)) for k in range(2)]+[threading.Thread(target=synchronous)]
 start=time.monotonic();m.start()
 for t in threads:t.start()
 while time.monotonic()-start<a.seconds:
  time.sleep(min(30,a.seconds));m.check()
  print('SOAK',round(time.monotonic()-start,1),'mutations',sum(counts),'samples',len(m.samples),flush=True)
 stop.set()
 for t in threads:t.join(3);assert not t.is_alive()
 with runtime.lock:
  for row in rows:row['status']='completed'
 m.flush_active();m.close();m.check()
 journal=[json.loads(s) for s in m.journal.read_text().splitlines()]
 terminal=[r['record'] for r in journal if r['kind']=='REQUEST_TERMINAL']
 assert len(terminal)==len(rows)==len({r['request_id'] for r in terminal})
 for entry in journal:
  rs=[entry['record']] if entry['kind']=='REQUEST_TERMINAL' else entry.get('outstanding',[])
  assert all(r['version']==r['worker_response']['version'] for r in rs)
 assert not errors
 elapsed=time.monotonic()-start
 save('telemetry_stress_result_v2.json',dict(passed=elapsed>=a.seconds and not m.errors,elapsed_seconds=elapsed,requested_seconds=a.seconds,
  original_failure_seconds=1997.196239099,mutations=sum(counts),samples=len(m.samples),journal_rows=len(journal),terminal_records=len(terminal),
  coherent_snapshots=True,max_control_check_seconds=max(check_times),thread_exited=not m.thread.is_alive(),errors=errors+m.errors,failure_injection=failure_tests,GPU_measurements=0))
 print('STRESS_PASSED',elapsed,flush=True)
if __name__=='__main__':main()
