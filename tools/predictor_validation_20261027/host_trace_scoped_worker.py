"""Optional host-call brackets on the unchanged passive-GC scoped worker.

CPU time and scheduling counters describe the calling thread. Wall minus CPU
includes blocking, synchronization and descheduling; it is not a pure scheduler
stall estimate. Brackets include existing instrumentation overhead and preserve
all original semantic CUDA events without adding synchronization.
"""
import os
import resource
import threading
import time


def thread_usage():
    value=resource.getrusage(resource.RUSAGE_THREAD)
    return dict(user_cpu_seconds=value.ru_utime,system_cpu_seconds=value.ru_stime,
        minor_page_faults=value.ru_minflt,major_page_faults=value.ru_majflt,
        voluntary_context_switches=value.ru_nvcsw,involuntary_context_switches=value.ru_nivcsw)


def main():
    mode=os.environ.get('GS_PASSIVE_HOST_TRACE')
    if mode not in ('on','off'):raise ValueError('Explicit on/off host trace required')
    if os.environ.get('GS_PASSIVE_GC_TRACE')!='on':raise ValueError('Matched passive GC enabled in both host arms')
    import complete_observed_worker as complete
    original_emit=complete.emit
    def emit(kind,payload):
        if kind=='READY':
            payload=dict(payload,passive_host_policy=dict(mode=mode,
                clock='time.thread_time_ns',usage='resource.RUSAGE_THREAD',
                scope='outer host-call brackets including existing event-recording overhead',
                extra_cuda_sync=False,affinity=sorted(os.sched_getaffinity(0))))
        original_emit(kind,payload)
    complete.emit=emit
    if mode=='on':
        import semantic_observed_worker as semantic
        original_recorder=semantic.DetailRecorder
        class HostRecorder(original_recorder):
            def call(self,original,role,args,kwargs):
                tid=threading.get_ident();index=len(self.records)
                before=thread_usage();wall_start=time.perf_counter_ns();cpu_start=time.thread_time_ns()
                result=super().call(original,role,args,kwargs)
                cpu_end=time.thread_time_ns();wall_end=time.perf_counter_ns();after=thread_usage()
                matches=[event for event in self.records[index:] if event['thread_id']==tid
                         and event['component']==role and wall_start<=event['host_start_ns']<=wall_end]
                if not matches:raise ValueError('Host bracket missing semantic event')
                event=matches[0]
                if not wall_start<=event['host_start_ns']<event['host_end_ns']<=wall_end:
                    raise ValueError('Host bracket does not contain semantic event')
                event['passive_host_bracket']=dict(schema='gsdeploy.passive_host_call_bracket.v1',
                    thread_id=tid,native_thread_id=threading.get_native_id(),
                    wall_start_ns=wall_start,wall_end_ns=wall_end,cpu_start_ns=cpu_start,cpu_end_ns=cpu_end,
                    usage_before=before,usage_after=after,
                    scope='outer wall/CPU bracket; wall minus CPU mixes blocking, synchronization and descheduling')
                return result
        semantic.DetailRecorder=HostRecorder
    import gc_trace_scoped_worker as worker
    worker.main()


if __name__=='__main__':main()
