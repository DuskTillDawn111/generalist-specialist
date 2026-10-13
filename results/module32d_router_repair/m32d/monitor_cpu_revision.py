"""CPU-only v2 after failed engineering preflight; NOT used for any GPU evidence."""
from .monitor import IncrementalMonitor
from .common import *

class RepairedMonitor(IncrementalMonitor):
 def coherent_records(self):
  # Capture references only. Publication replaces caches instead of mutating the
  # old container; sample_lock serializes sensor/checkpoint producers. A phase
  # transition cannot alter this snapshot's header/cache/order references.
  with self.state_lock:
   active=self.active_phase;cached=self.cached;order=self.cache_order
  if active is None:return None,[]
  tick=time.perf_counter_ns()
  with self.runtime.lock:
   # Publication also owns runtime.lock. Under that lock, a single pointer
   # check safely detects a changed phase without acquiring state_lock in
   # reverse order. Never consume a new phase's terminal events into old cache.
   if self.active_phase is active:
    while self.runtime.terminal_changes:
     r=self.runtime.terminal_changes.popleft()
     if r['request_id'] in cached:cached[r['request_id']]=dict(r)
    for r in self.runtime.active.values():
     if r['request_id'] in cached:cached[r['request_id']]=dict(r)
   header={k:v for k,v in active.items() if k!='request_records'}
  self.snapshot_lock_ms.append((time.perf_counter_ns()-tick)/1e6)
  return header,copy.deepcopy([cached[k] for k in order])
