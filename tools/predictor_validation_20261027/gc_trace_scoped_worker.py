"""Passive cyclic-GC timing in the unchanged scoped semantic runtime.

Does not disable, force, or retune collection. CUDA events remain unchanged.
This diagnostic instrumentation requires its own matched observer comparison.
"""
from collections import deque
import gc
import os
import resource
import threading
import time


class GCTrace:
    def __init__(self, clock=time.perf_counter_ns, capacity=8192):
        self.clock=clock;self.events=deque(maxlen=capacity)
        self.active=None;self.dropped=0;self.errors=[]

    def callback(self, phase, info):
        now=self.clock()
        if phase=='start':
            if self.active is not None:self.errors.append('nested_collection_start')
            self.active=(now,info['generation'],threading.get_ident())
        elif phase=='stop':
            if self.active is None:
                self.errors.append('unmatched_collection_stop');return
            start,generation,thread=self.active;self.active=None
            if generation!=info['generation']:self.errors.append('collection_generation_changed')
            if len(self.events)==self.events.maxlen:self.dropped+=1
            self.events.append((start,now,generation,info.get('collected'),info.get('uncollectable'),thread))
        else:self.errors.append('unknown_collection_callback_phase')

    def snapshot(self, start, end):
        events=list(self.events)
        covered=not self.dropped or bool(events and events[0][0]<=start)
        return dict(schema='gsdeploy.passive_gc_trace.v1',enabled=True,
            clock='host_perf_counter_ns',window_start_ns=start,window_end_ns=end,
            gc_policy_unchanged=True,gc_enabled=gc.isenabled(),gc_thresholds=list(gc.get_threshold()),
            coverage_complete=covered and not self.errors and self.active is None,
            dropped_process_events=self.dropped,errors=list(self.errors),
            events=[dict(start_ns=a,end_ns=b,generation=g,collected=n,uncollectable=u,thread_id=t)
                    for a,b,g,n,u,t in events if a<end and b>start])


def usage():
    r=resource.getrusage(resource.RUSAGE_SELF)
    return dict(user_cpu_seconds=r.ru_utime,system_cpu_seconds=r.ru_stime,
                minor_page_faults=r.ru_minflt,major_page_faults=r.ru_majflt,
                voluntary_context_switches=r.ru_nvcsw,involuntary_context_switches=r.ru_nivcsw)


def main():
    mode=os.environ.get('GS_PASSIVE_GC_TRACE')
    if mode not in ('on','off'):raise ValueError('Explicit on/off passive GC trace required')
    enabled=mode=='on';trace=GCTrace();current=None;start_usage=None
    import complete_observed_worker as complete
    original=complete.emit
    policy=dict(mode=mode,forced_collections=False,gc_disabled=False,thresholds_changed=False,
                timing='passive gc.callbacks; no extra CUDA synchronization')
    def emit(kind,payload):
        nonlocal current,start_usage
        if kind=='READY':
            payload=dict(payload,passive_gc_policy=dict(policy,gc_enabled=gc.isenabled(),thresholds=list(gc.get_threshold())))
        if enabled:
            if kind=='PHASE' and payload.get('event')=='start' and payload.get('segment_id')=='host_setup':
                current=(payload['request_id'],payload['perf_ns']);start_usage=usage()
            elif kind=='RESULT':
                if current is None or payload['request_id']!=current[0]:raise ValueError('GC trace request identity mismatch')
                end=payload['end_perf_ns'];metadata=trace.snapshot(current[1],end)
                metadata.update(usage_at_first_phase_hook=start_usage,usage_at_result_hook=usage(),
                    initial_counter_scope='first PHASE callback, after observer construction; not request-arrival state')
                payload=dict(payload,passive_gc_trace=metadata)
            elif kind=='WORKER_READY':
                if current is not None and payload['request_id']==current[0]:
                    payload=dict(payload,passive_gc_lifecycle_trace=trace.snapshot(current[1],payload['perf_ns']))
                current=None;start_usage=None
        original(kind,payload)
    complete.emit=emit
    if enabled:gc.callbacks.append(trace.callback)
    try:
        import scoped_reconstruct_observed_worker as scoped
        scoped.main()
    finally:
        if enabled:gc.callbacks.remove(trace.callback)


if __name__=='__main__':main()
