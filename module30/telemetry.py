"""Replacement for the unsafe Module29 journal. Historical frozen code is untouched."""
import copy,json,threading,time,traceback,os
from replica_scaling.telemetry import Monitor,snapshot,owned_sample
from module11b.telemetry import CostMonitor
from module26.gpu import thermal
from .common import OUT,save,json_native

class SafeMonitor(CostMonitor):
 def __init__(self,runtime,slot,budget,interval=.5,checkpoint_seconds=10.,sensor=None):
  super().__init__(runtime,interval,budget)
  self.slot=slot;self.sample_lock=threading.RLock();self.active_phase=None
  self.checkpoint_seconds=checkpoint_seconds;self.last_checkpoint=0.;self.last_progress=0.
  self.persisted=set();self.phase_labels=set();self.sensor=sensor
  self.journal=OUT/'request_journals'/(slot['run_id']+'.jsonl');self.journal.parent.mkdir(parents=True,exist_ok=True)
  self.error_details=[]
 def set_active(self,value):
  with self.sample_lock:self.active_phase=value
 def coherent_records(self):
  with self.runtime.lock:
   active=self.active_phase
   if active is None:return None,[]
   header={k:v for k,v in active.items() if k!='request_records'}
   records=copy.deepcopy(active['request_records'])
  return header,records
 def journal_snapshot(self,value,force=False):
  header,records=self.coherent_records()
  if header is None:return
  label=header['label'];now=time.monotonic()
  if not force and now-self.last_checkpoint<self.checkpoint_seconds:return
  rows=[];keys=[]
  if label not in self.phase_labels:rows.append(dict(kind='PHASE_WINDOW',**header))
  for r in records:
   key=(label,r['request_id'])
   if r['status'] in ('completed','error','incomplete') and key not in self.persisted:
    rows.append(dict(kind='REQUEST_TERMINAL',phase=label,record=r));keys.append(key)
  rows.append(dict(kind='CHECKPOINT',at_unix_ns=time.time_ns(),phase=label,telemetry=value,
   planned=len(records),status_counts={s:sum(r['status']==s for r in records) for s in ('planned','queued','submitted','completed','error','incomplete')},
   outstanding=[r for r in records if r['status'] in ('queued','submitted')],full_raw_pending=True))
  payload=''.join(json.dumps(r,ensure_ascii=False,allow_nan=False,default=json_native)+'\n' for r in rows)
  with self.journal.open('a') as stream:
   stream.write(payload);stream.flush();os.fsync(stream.fileno())
  # Mark only after a successful durable write; a failed journal invalidates the session.
  self.persisted.update(keys);self.phase_labels.add(label);self.last_checkpoint=now
  if now-self.last_progress>=30 or force:
   self.last_progress=now
   save('live_progress.json',dict(run_id=self.slot['run_id'],phase=self.phase,
    elapsed_seconds=(time.perf_counter_ns()-header['planned_start_time_ns'])/1e9,
    planned_seconds=header['planned_duration_seconds'],planned=len(records),
    submitted=sum(r.get('actual_submit_time_ns') is not None for r in records),completed=sum(r['status']=='completed' for r in records),
    temperature_gpu_c=value.get('temperature_gpu_c'),used_mib=value.get('owned_used_mib')))
 def sample(self):
  # Includes synchronous cooldown sampling and background collector: one producer.
  with self.sample_lock:
   if self.sensor:value=self.sensor()
   else:
    with self.runtime.lock:pids={w.process.pid for w in list(self.runtime.workers.values()) if w.process is not None}
    value=owned_sample(snapshot(),pids);value.update(self.host(pids));value.update(thermal())
   value['phase']=self.phase
   if value['used_mib']>self.capacity or value['owned_used_mib']>self.capacity:self.resource_failure=True
   self.journal_snapshot(value)
   # Publish only a complete sample.
   self.samples.append(value);return value
 def collect(self):
  while not self.stop.is_set():
   try:self.sample()
   except Exception as exc:
    with self.sample_lock:
     self.errors.append(repr(exc));self.error_details.append(dict(error=repr(exc),traceback=traceback.format_exc(),time_ns=time.perf_counter_ns()))
    self.stop.set()
   self.stop.wait(self.interval)
 def check(self):
  with self.sample_lock:
   if self.errors:raise RuntimeError('TELEMETRY_ERROR: '+repr(self.errors))
   super().check()
 def flush_active(self):
  with self.sample_lock:
   self.journal_snapshot(self.samples[-1] if self.samples else {},True)
 def close(self):
  self.stop.set()
  if self.thread.ident:self.thread.join(timeout=45.)
  if self.thread.is_alive():
   self.errors.append('TELEMETRY_THREAD_DID_NOT_EXIT');raise RuntimeError(self.errors[-1])
