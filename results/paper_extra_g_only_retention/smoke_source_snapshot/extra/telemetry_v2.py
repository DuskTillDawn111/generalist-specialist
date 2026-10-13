"""Nonblocking error checks and phase publication; I/O never holds the control lock."""
import copy,threading,time,traceback
from .telemetry import SafeMonitor

class SafeMonitorV2(SafeMonitor):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.state_lock=threading.Lock();self.resource_event=threading.Event();self.foreign_event=threading.Event();self.error_event=threading.Event()
 def set_active(self,value):
  with self.state_lock:self.active_phase=value
 def coherent_records(self):
  with self.state_lock:active=self.active_phase
  if active is None:return None,[]
  with self.runtime.lock:
   header={k:v for k,v in active.items() if k!='request_records'};records=copy.deepcopy(active['request_records'])
  return header,records
 def sample(self):
  value=super().sample()
  if value['used_mib']>self.capacity or value['owned_used_mib']>self.capacity:self.resource_event.set()
  if value['foreign_pids']:self.foreign_event.set()
  return value
 def collect(self):
  while not self.stop.is_set():
   try:self.sample()
   except Exception as exc:
    detail=dict(error=repr(exc),traceback=traceback.format_exc(),time_ns=time.perf_counter_ns())
    with self.state_lock:self.errors.append(repr(exc));self.error_details.append(detail)
    self.error_event.set();self.stop.set()
   self.stop.wait(self.interval)
 def check(self):
  # Thread-safe Events and a short error-state lock. No NVML, journal or record-copy lock.
  with self.state_lock:errors=tuple(self.errors)
  if errors or self.error_event.is_set():
   raise RuntimeError('TELEMETRY_ERROR: '+repr(errors))
  if self.resource_event.is_set() or self.resource_failure:raise RuntimeError('RUNTIME_RESOURCE_INFEASIBLE: actual GPU residency exceeds frozen physical budget')
  if self.foreign_event.is_set():raise RuntimeError('external_gpu_process_interference')
