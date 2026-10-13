"""Incremental coherent journal snapshots; sensor and disk I/O stay outside runtime lock."""
from .common import *
from module31a.campaign import TimedMonitor
from module30.telemetry_v2 import SafeMonitorV2

class IncrementalMonitor(TimedMonitor):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);self.cached={};self.cache_order=[];self.snapshot_lock_ms=[]
 def set_active(self,value):
  tick=time.perf_counter()
  with self.state_lock:
   with self.runtime.lock:
    self.active_phase=value
    self.cached={r['request_id']:dict(r) for r in value['request_records']} if value else {}
    self.cache_order=list(self.cached)
  self.publish_seconds.append(time.perf_counter()-tick)
 def coherent_records(self):
  with self.state_lock:
   active=self.active_phase
   if active is None:return None,[]
   tick=time.perf_counter_ns()
   with self.runtime.lock:
    # Each terminal transition is consumed once, including send/worker errors.
    while self.runtime.terminal_changes:
     r=self.runtime.terminal_changes.popleft()
     if r['request_id'] in self.cached:self.cached[r['request_id']]=dict(r)
    for r in self.runtime.active.values():
     if r['request_id'] in self.cached:self.cached[r['request_id']]=dict(r)
    header={k:v for k,v in active.items() if k!='request_records'}
   self.snapshot_lock_ms.append((time.perf_counter_ns()-tick)/1e6)
   # Worker payload dictionaries are published once and never mutated in place;
   # scalar record mutations cannot alter these shallow snapshots. Copy outside lock.
   return header,copy.deepcopy([self.cached[k] for k in self.cache_order])
 def journal_snapshot(self,value,force=False):
  # Original monitor copied the full cohort BEFORE testing its 10s interval,
  # consequently holding runtime.lock every 0.5s. Gate before copying.
  if not force and time.monotonic()-self.last_checkpoint<self.checkpoint_seconds:return
  return super().journal_snapshot(value,force)
 def flush_active(self):
  # Final annotation/abort fields are captured after arrivals have stopped.
  with self.state_lock:
   active=self.active_phase
   if active:
    with self.runtime.lock:self.cached={r['request_id']:dict(r) for r in active['request_records']}
  return super().flush_active()
