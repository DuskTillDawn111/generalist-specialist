"""Timing endpoint correction; original diagnostics retained, policy untouched."""
import control_diagnosis as base
from control_diagnosis import *
class TimedLock:
 def __init__(self,lock):self.lock=lock;self.owner=None;self.recent=collections.deque(maxlen=64)
 def __enter__(self):
  self.lock.acquire();self.owner=threading.get_native_id();self.enter_ns=time.perf_counter_ns();return self
 def __exit__(self,*a):
  end=time.perf_counter_ns();self.recent.append((self.enter_ns,end,self.owner));self.owner=None;self.lock.release()
class CorrectedMonitor(base.SegmentedMonitor):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);self.state_lock=TimedLock(self.state_lock.lock)
 def check(self):
  usage0=resource.getrusage(resource.RUSAGE_THREAD)
  tick=time.perf_counter();begin=time.perf_counter_ns();cpu0=time.thread_time_ns();owner=self.state_lock.owner;error=None
  stamps=[(begin,cpu0)]
  try:
   with self.state_lock:
    stamps.append((time.perf_counter_ns(),time.thread_time_ns()));errors=tuple(self.errors);stamps.append((time.perf_counter_ns(),time.thread_time_ns()))
   stamps.append((time.perf_counter_ns(),time.thread_time_ns()))
   if errors or self.error_event.is_set():raise RuntimeError('TELEMETRY_ERROR: '+repr(errors))
   if self.resource_event.is_set() or self.resource_failure:raise RuntimeError('RUNTIME_RESOURCE_INFEASIBLE: actual GPU residency exceeds frozen physical budget')
   if self.foreign_event.is_set():raise RuntimeError('external_gpu_process_interference')
  except BaseException as e:error=repr(e);raise
  finally:
   # CPU endpoint and every segment endpoint are inside the full wall interval.
   cpuend=time.thread_time_ns();end=time.perf_counter_ns();elapsed=time.perf_counter()-tick;stamps.append((end,cpuend));usage1=resource.getrusage(resource.RUSAGE_THREAD);tid=threading.get_native_id()
   other_owners=[dict(begin_ns=a,end_ns=b,native_tid=t) for a,b,t in tuple(self.state_lock.recent) if t!=tid and a<stamps[1][0] and b>begin] if len(stamps)>1 else []
   row=dict(call_id=len(self.detailed)+1,**self.context,phase=self.phase,perf_ns=begin,end_perf_ns=end,wall_ms=elapsed*1000,thread_cpu_ms=(cpuend-cpu0)/1e6,off_cpu_estimate_ms=max(0,elapsed*1000-(cpuend-cpu0)/1e6),timer_boundary_overhead_ms=elapsed*1000-(end-begin)/1e6,state_owner_before_acquire=owner,other_state_lock_owner_intervals=other_owners,thread=threading.current_thread().name,native_tid=tid,voluntary_switches=usage1.ru_nvcsw-usage0.ru_nvcsw,involuntary_switches=usage1.ru_nivcsw-usage0.ru_nivcsw,error=error,threshold_violation=elapsed>=.01,synchronous_telemetry_write_ms=0,timing_version=2)
   for name,a,b in zip(['acquire','state_read','release','predicates'],stamps,stamps[1:]):row[name+'_wall_ms']=(b[0]-a[0])/1e6;row[name+'_cpu_ms']=(b[1]-a[1])/1e6
   self.check_seconds.append(elapsed);self.control_probes.append(row);self.detailed.append(row)
   if self.enforce_control and elapsed>=.01:raise RuntimeError('CONTROL_CHECK_OVER_10MS: '+repr(elapsed))
base.SegmentedMonitor=CorrectedMonitor
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('case');args=p.parse_args();m=read(OUT/'timing_endpoint_correction_manifest.json');base.run(next(c for c in m['cases'] if c['id']==args.case))
