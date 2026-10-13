"""Call frozen Module29 DurableMonitor.sample with CPU sensor stubs and live writes.

Every potential historical save is intercepted; journal belongs to Module30.
"""
import importlib.util,sys,threading,time,traceback
from unittest.mock import patch
from .common import *

def main():
 source=ROOT/'results/module29_dynamic_replication/gpu_replication.py';before=ref(source)
 sys.path.insert(0,str(source.parent))
 spec=importlib.util.spec_from_file_location('module30_legacy29_audit',source);legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
 row={str(i):i for i in range(30000)};row.update(status='submitted',request_id='test_request')
 runtime=type('CPUStubRuntime',(),{'lock':threading.RLock()})()
 monitor=legacy.DurableMonitor.__new__(legacy.DurableMonitor)
 monitor.active_phase=dict(request_records=[row],label='measurement',planned_start_time_ns=time.perf_counter_ns(),planned_end_time_ns=time.perf_counter_ns()+1000000000,planned_duration_seconds=1.)
 monitor.phase='measurement';monitor.last_checkpoint=0.;monitor.last_progress=float('inf');monitor.persisted=set();monitor.phase_labels=set();monitor.journal=OUT/'fixtures/legacy29_cpu_journal.jsonl';monitor.slot={'run_id':'CPU_LEGACY_REPRODUCER','stage':'CPU_ONLY'}
 stop=threading.Event();old=sys.getswitchinterval();sys.setswitchinterval(.000001)
 def writer():
  i=0
  while not stop.is_set():
   with runtime.lock:row['new'+str(i%100)]=i;row.pop('new'+str((i+1)%100),None)
   i+=1
 thread=threading.Thread(target=writer);thread.start();caught=None
 try:
  with patch.object(legacy.ProcessMonitor,'sample',return_value={'temperature_gpu_c':30,'owned_used_mib':0}),patch.object(legacy,'thermal',return_value={}),patch.object(legacy,'save',side_effect=AssertionError('Historical write blocked by CPU audit')):
   for _ in range(500):
    monitor.last_checkpoint=0.
    try:monitor.sample()
    except RuntimeError as exc:
     if 'dictionary changed size during iteration' in str(exc):caught=dict(error=repr(exc),traceback=traceback.format_exc());break
     raise
 finally:stop.set();thread.join(3);sys.setswitchinterval(old)
 assert caught and ref(source)['sha256']==before['sha256']
 save('legacy_exact_method_reproduction.json',dict(passed=True,CPU_only=True,GPU_measurements=0,original_source=before,
  called='Frozen Module29 DurableMonitor.sample',writer_updates_under_runtime_lock=True,legacy_reader_without_runtime_lock=True,
  captured=caught,historical_save_intercepted=True,unique_original_failure_frame_not_recorded=True))
 print('EXACT_FROZEN_METHOD_RACE_REPRODUCED',caught['error'],flush=True)
if __name__=='__main__':main()
