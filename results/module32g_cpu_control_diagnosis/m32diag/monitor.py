"""Explicit output ownership; incremental journal, short control lock, full probe log.

No inherited constructor/save/OUT from a historical campaign is invoked.
Check semantics and perf_counter wall timer match TimedMonitor/SafeMonitorV2.
"""
import threading,traceback
from module11b.telemetry import CostMonitor
from replica_scaling.telemetry import snapshot,owned_sample
from .common import *
from .environment import thermal

class IncrementalMonitor(CostMonitor):
 def __init__(self,runtime,slot,budget,interval=.5,checkpoint_seconds=10.,sensor=None,output_dir=None,enforce_control=True):
  super().__init__(runtime,interval,budget)
  self.output_dir=Path(output_dir or OUT).resolve()
  assert self.output_dir.is_relative_to(OUT.resolve()),'Historical output namespace forbidden'
  self.slot=slot;self.sample_lock=threading.RLock();self.state_lock=threading.Lock()
  self.active_phase=None;self.cached={};self.cache_order=[];self.snapshot_lock_ms=[]
  self.checkpoint_seconds=checkpoint_seconds;self.last_checkpoint=0.;self.last_progress=0.
  self.persisted=set();self.phase_labels=set();self.sensor=sensor;self.error_details=[]
  self.resource_event=threading.Event();self.foreign_event=threading.Event();self.error_event=threading.Event()
  self.check_seconds=[];self.publish_seconds=[];self.control_probes=[];self.enforce_control=enforce_control;self.routing_logs={}
  self.journal=self.output_dir/'request_journals'/(slot['run_id']+'.jsonl')
  self.journal.parent.mkdir(parents=True,exist_ok=True)
  # An empty journal is valid evidence of a startup abort, never a successful
  # request window. Avoid losing the primary failure to a report FileNotFound.
  with self.journal.open('a'):pass
 def set_active(self,value):
  tick=time.perf_counter()
  with self.state_lock:
   with self.runtime.lock:
    self.active_phase=value
    self.cached={r['request_id']:dict(r) for r in value['request_records']} if value else {}
    self.cache_order=list(self.cached)
  elapsed=time.perf_counter()-tick;self.publish_seconds.append(elapsed)
  if self.enforce_control and elapsed>=.01:raise RuntimeError('CONTROL_PUBLISH_OVER_10MS: '+repr(elapsed))
 def coherent_records(self):
  with self.state_lock:active=self.active_phase;cached=self.cached;order=self.cache_order
  if active is None:return None,[]
  tick=time.perf_counter_ns()
  with self.runtime.lock:
   # A delayed old-phase snapshot must not consume the new phase's events.
   if self.active_phase is active:
    while self.runtime.terminal_changes:
     r=self.runtime.terminal_changes.popleft()
     if r['request_id'] in cached:cached[r['request_id']]=dict(r)
    for r in self.runtime.active.values():
     if r['request_id'] in cached:cached[r['request_id']]=dict(r)
   header={k:v for k,v in active.items() if k!='request_records'}
  self.snapshot_lock_ms.append((time.perf_counter_ns()-tick)/1e6)
  return header,copy.deepcopy([cached[k] for k in order])
 def journal_snapshot(self,value,force=False):
  now=time.monotonic()
  if not force and now-self.last_checkpoint<self.checkpoint_seconds:return
  header,records=self.coherent_records()
  if header is None:return
  label=header['label'];rows=[];keys=[]
  if label not in self.phase_labels:rows.append(dict(kind='PHASE_WINDOW',**header))
  for r in records:
   key=(label,r['request_id'])
   if r['status'] in ('completed','error','incomplete') and key not in self.persisted:
    rows.append(dict(kind='REQUEST_TERMINAL',phase=label,record=r));keys.append(key)
  rows.append(dict(kind='CHECKPOINT',at_unix_ns=time.time_ns(),phase=label,telemetry=value,planned=len(records),
   status_counts={s:sum(r['status']==s for r in records) for s in ('planned','queued','submitted','completed','error','incomplete')},
   outstanding=[r for r in records if r['status'] in ('queued','submitted')],full_raw_pending=True))
  payload=''.join(json.dumps(r,ensure_ascii=False,allow_nan=False,default=native)+'\n' for r in rows)
  with self.journal.open('a') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
  self.persisted.update(keys);self.phase_labels.add(label);self.last_checkpoint=now
  if now-self.last_progress>=30 or force:
   self.last_progress=now
   save(str((self.output_dir/'live_progress.json').relative_to(OUT)),dict(run_id=self.slot['run_id'],phase=self.phase,
    elapsed_seconds=(time.perf_counter_ns()-header['planned_start_time_ns'])/1e9,planned_seconds=header['planned_duration_seconds'],
    planned=len(records),submitted=sum(r.get('actual_submit_time_ns') is not None for r in records),
    completed=sum(r['status']=='completed' for r in records),temperature_gpu_c=value.get('temperature_gpu_c'),used_mib=value.get('owned_used_mib')))
 def sample(self):
  with self.sample_lock:
   if self.sensor:value=self.sensor()
   else:
    with self.runtime.lock:pids={w.process.pid for w in list(self.runtime.workers.values()) if w.process is not None}
    value=owned_sample(snapshot(),pids);value.update(self.host(pids));value.update(thermal())
   value['phase']=self.phase
   if value['used_mib']>self.capacity or value['owned_used_mib']>self.capacity:self.resource_failure=True;self.resource_event.set()
   if value['foreign_pids']:self.foreign_event.set()
   self.journal_snapshot(value);self.samples.append(value);return value
 def collect(self):
  while not self.stop.is_set():
   try:self.sample()
   except Exception as exc:
    detail=dict(error=repr(exc),traceback=traceback.format_exc(),time_ns=time.perf_counter_ns())
    with self.state_lock:self.errors.append(repr(exc));self.error_details.append(detail)
    self.error_event.set();self.stop.set()
   self.stop.wait(self.interval)
 def check(self):
  cpu_start=time.thread_time_ns();tick=time.perf_counter();begin=time.perf_counter_ns();acquired=None;released=None;error=None
  try:
   # Same predicates, lock and order as SafeMonitorV2.check; only stamps added.
   with self.state_lock:acquired=time.perf_counter_ns();errors=tuple(self.errors)
   released=time.perf_counter_ns()
   if errors or self.error_event.is_set():raise RuntimeError('TELEMETRY_ERROR: '+repr(errors))
   if self.resource_event.is_set() or self.resource_failure:raise RuntimeError('RUNTIME_RESOURCE_INFEASIBLE: actual GPU residency exceeds frozen physical budget')
   if self.foreign_event.is_set():raise RuntimeError('external_gpu_process_interference')
  except BaseException as exc:error=repr(exc);raise
  finally:
   elapsed=time.perf_counter()-tick;cpu_ns=time.thread_time_ns()-cpu_start
   self.check_seconds.append(elapsed)
   self.control_probes.append(dict(phase=self.phase,perf_ns=begin,wall_ms=elapsed*1000,thread_cpu_ms=cpu_ns/1e6,
    state_wait_ms=(acquired-begin)/1e6 if acquired else None,state_hold_ms=(released-acquired)/1e6 if released else None,
    error=error,thread=threading.current_thread().name))
   # Never relax threshold or remove an outlier. Abort before next admission.
   if self.enforce_control and elapsed>=.01:raise RuntimeError('CONTROL_CHECK_OVER_10MS: '+repr(elapsed))
 def flush_active(self):
  with self.state_lock:
   active=self.active_phase
   if active:
    with self.runtime.lock:self.cached={r['request_id']:dict(r) for r in active['request_records']}
  with self.sample_lock:self.journal_snapshot(self.samples[-1] if self.samples else {},True)
 def close(self):
  self.stop.set()
  if self.thread.ident:self.thread.join(timeout=45.)
  if self.thread.is_alive():self.errors.append('TELEMETRY_THREAD_DID_NOT_EXIT');raise RuntimeError(self.errors[-1])
