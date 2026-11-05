"""Optional glibc free-heap reclamation after measured priming, before READY.

No live tensor is freed, no GC policy is changed, and no CUDA synchronization
is added. The full reclamation procedure precedes the worker-ready timestamp
and the passive GC lifecycle snapshot, so its time remains in startup/priming.
This is an isolated runtime candidate, not a qualified deployment setting.
"""
import ctypes
import gc
import os
from pathlib import Path
import platform
import resource
import time


def host_memory():
    values={}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:','VmHWM:','RssAnon:','RssFile:','RssShmem:')):
            key,value=line.split(':',1);values[key+'_kib']=int(value.split()[0])
    return values


def trim_procedure(mode,allocator):
    procedure_start=time.perf_counter_ns();before=host_memory();policy=(gc.isenabled(),gc.get_threshold())
    usage_before=resource.getrusage(resource.RUSAGE_SELF)
    call_start=time.perf_counter_ns();result=allocator(0) if mode=='on' else None;call_end=time.perf_counter_ns()
    usage_after=resource.getrusage(resource.RUSAGE_SELF);after=host_memory()
    if policy!=(gc.isenabled(),gc.get_threshold()):raise ValueError('GC policy changed during heap trim')
    if result is not None and result not in (0,1):raise ValueError('Unexpected malloc_trim return value')
    return dict(schema='gsdeploy.postprime_host_heap_trim.v1',mode=mode,api='glibc malloc_trim(0)',
        procedure_start_ns=procedure_start,procedure_end_ns=time.perf_counter_ns(),
        allocator_call_start_ns=call_start,allocator_call_end_ns=call_end,allocator_return=result,
        host_memory_before=before,host_memory_after=after,
        user_cpu_seconds=usage_after.ru_utime-usage_before.ru_utime,
        system_cpu_seconds=usage_after.ru_stime-usage_before.ru_stime,
        gc_enabled=policy[0],gc_thresholds=list(policy[1]),gc_policy_unchanged=True,
        live_allocations_released=False,extra_cuda_sync=False,
        scope='Full procedure included before post-prime worker READY; RSS counters are proc status observations, not exact allocation ownership')


def install(complete,mode,procedure,clock=time.perf_counter_ns):
    original=complete.emit;expected=None;seen=0;done=False
    def emit(kind,payload):
        nonlocal expected,seen,done
        if kind=='READY':
            expected=3*len(payload['tasks'])
            payload=dict(payload,postprime_trim_policy=dict(mode=mode,api='glibc malloc_trim(0)',
                priming_requests=expected,timing='after final priming response, before worker READY',
                libc=list(platform.libc_ver()),forced_gc=False,gc_thresholds_changed=False,extra_cuda_sync=False))
        elif kind=='WORKER_READY' and payload['request_id'].startswith('priming:'):
            if expected is None or done:raise ValueError('Unexpected priming sequence for trim policy')
            seen+=1
            if seen==expected:
                receipt=procedure(mode);done=True
                payload=dict(payload,perf_ns=clock(),postprime_trim_receipt=receipt,
                    scope='after final priming response flush and declared host heap procedure, before next read')
                if payload['perf_ns']<receipt['procedure_end_ns']:raise ValueError('Trim omitted from READY boundary')
        elif kind=='PHASE' and payload.get('event')=='start' and payload.get('segment_id')=='host_setup':
            if not payload['request_id'].startswith('priming:') and not done:
                raise ValueError('Formal request arrived before declared priming trim boundary')
        original(kind,payload)
    complete.emit=emit


def main():
    mode=os.environ.get('GS_POSTPRIME_HOST_TRIM')
    if mode not in ('on','off'):raise ValueError('Explicit on/off post-prime trim policy required')
    library=ctypes.CDLL(None);allocator=library.malloc_trim
    allocator.argtypes=[ctypes.c_size_t];allocator.restype=ctypes.c_int
    import scoped_reconstruct_observed_worker as scoped
    original=scoped.main
    def scoped_main():
        # host_trace.main and gc_trace.main have installed their emit wrappers
        # before entering scoped.main. Trim must run before their GC snapshot.
        import complete_observed_worker as complete
        install(complete,mode,lambda arm:trim_procedure(arm,allocator))
        original()
    scoped.main=scoped_main
    import host_trace_scoped_worker as host
    host.main()


if __name__=='__main__':main()
