import threading,time
from types import SimpleNamespace
from .common import *
from .telemetry_v2 import SafeMonitorV2

def main():
 results=[]
 for index in range(12):
  started=threading.Event()
  def sensor():
   started.set();time.sleep(.25)
   return dict(time_ns=time.perf_counter_ns(),used_mib=0,owned_used_mib=0,foreign_pids=[])
  runtime=SimpleNamespace(lock=threading.RLock(),workers={},ready={});m=SafeMonitorV2(runtime,{'run_id':'CPU_SLOW_IO_V2_'+str(index)},1,sensor=sensor)
  thread=threading.Thread(target=m.sample);thread.start();assert started.wait(2)
  t=time.perf_counter();m.check();check=time.perf_counter()-t
  t=time.perf_counter();m.set_active(dict(label='CPU_TEST',planned_start_time_ns=time.perf_counter_ns(),planned_duration_seconds=1,request_records=[]));publish=time.perf_counter()-t
  thread.join(3);assert not thread.is_alive()
  assert check<.01 and publish<.01,(check,publish)
  results.append(dict(check_seconds=check,phase_publish_seconds=publish,sensor_io_seconds=.25))
 save('nonblocking_control_acceptance.json',dict(passed=True,CPU_only=True,GPU_measurements=0,repetitions=results))
 print('NONBLOCKING_CONTROL_ACCEPTANCE_PASSED',max(r['check_seconds'] for r in results),flush=True)
if __name__=='__main__':main()
