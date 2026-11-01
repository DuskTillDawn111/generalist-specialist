"""Experimental causal host-dispatch adapter; frozen static runtime is untouched.

This window is opt-in and is not connected to the existing native experiment
launcher. Static SWRR cohort certificates do not cover these decisions.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
import time

from tools.spillover_capacity_20261022.exposure import frozen_api


def queue_snapshot(runtime, aliases, now_ns, phase=None):
    """Call under runtime.lock; include pipe/worker backlog, not just qsize()."""
    snapshot={a:dict(outstanding=[],last_completed_ns=None) for a in aliases}
    for row in runtime.pending.values():
        if phase is not None and row.get('phase')!=phase:
            continue
        alias=row.get('model')
        if alias not in snapshot:
            continue
        status=row.get('status')
        if status=='completed':
            completed=row.get('completion_time_ns')
            if type(completed) is not int or completed>now_ns:
                raise ValueError('Invalid or future host completion in snapshot')
            previous=snapshot[alias]['last_completed_ns']
            snapshot[alias]['last_completed_ns']=max(previous or completed,completed)
        elif status in {'queued','submitted'}:
            enqueued=row.get('enqueued_time_ns');submitted=row.get('actual_submit_time_ns')
            if type(enqueued) is not int or enqueued>now_ns or (
                    submitted is not None and (type(submitted) is not int or submitted>now_ns)):
                raise ValueError('Invalid or future outstanding admission in snapshot')
            snapshot[alias]['outstanding'].append(dict(request_id=row['request_id'],
                task=row['task'],pool_id=row['pool_id'],enqueued_ns=enqueued,actual_submit_ns=submitted))
    for state in snapshot.values():
        state['outstanding'].sort(key=lambda r:(r['enqueued_ns'],r['request_id']))
    return snapshot


def choose_and_enqueue(runtime, policy, row, sample):
    """The host queue view and enqueue are atomic relative to result collection."""
    with runtime.lock:
        now=time.perf_counter_ns()
        snapshot=queue_snapshot(runtime,policy.aliases,now,row.get('phase'))
        seen=getattr(policy,'_host_completed_ids',set())
        observed=[]
        for completed in runtime.pending.values():
            if completed.get('phase')==row.get('phase') and completed.get('status')=='completed' and completed['request_id'] not in seen:
                observed.append(dict(request_id=completed['request_id'],model=completed['model'],
                    completion_time_ns=completed['completion_time_ns']))
                seen.add(completed['request_id'])
        policy._host_completed_ids=seen
        decision=policy.choose(row['task'],row['pool_id'],now,snapshot)
        alias=decision.get('alias')
        if alias is None:
            row['routing_decision']=dict(decision_time_ns=now,snapshot=snapshot,decision=decision,newly_observed_completions=observed)
            raise RuntimeError('QUEUE_AWARE_NO_KNOWN_EXECUTABLE_CANDIDATE')
        row['model']=alias
        row['routing_decision']=dict(decision_time_ns=now,snapshot=snapshot,decision=decision,newly_observed_completions=observed)
        runtime.enqueue(row,sample)
    return decision


def window(runtime,config,pool_path,phase,duration,records,monitor,policy,metadata=None):
    """Preregister cohort; choose destination only when each arrival is admitted."""
    if records:
        raise ValueError('Fresh phase-local record list required')
    api=frozen_api();runner=importlib.import_module(api['trace'].__package__+'.runtime.runner')
    pool=api['workload'].SamplePool(pool_path,list(config.workload_rps),config.random_seed)
    trace=api['trace'].make_trace(config.workload_rps,duration,config.random_seed,config.arrival_process)
    payloads=[]
    for index,(offset,task) in enumerate(trace):
        item=pool.next(task)
        row=runner.record(phase,index,task,None,item,0)
        row['scheduled_offset_seconds']=offset
        records.append(row);payloads.append(item['sample'])
    start=time.perf_counter_ns();unix=time.time_ns();end=start+round(duration*1e9)
    metadata={} if metadata is None else metadata
    metadata.update(start_time_ns=start,start_unix_ns=unix,end_time_ns=end,duration_seconds=duration,
        offered_requests=len(trace),trace_sha256=api['trace'].trace_digest(trace),
        routing_semantics='causal_queue_aware_experimental_v1',static_SWRR_certificate_applicable=False,
        cohort='All original arrivals/inputs preregistered; alias selected causally at actual dispatch.')
    for row in records:
        row['scheduled_time_ns']=start+round(row['scheduled_offset_seconds']*1e9)
    for row,sample in zip(records,payloads):
        runner.wait_until(row['scheduled_time_ns'],monitor)
        choose_and_enqueue(runtime,policy,row,sample)
    runner.wait_until(end,monitor)
    metadata['arrival_loop_end_time_ns']=time.perf_counter_ns()
    runtime.drain(records,config.drain_timeout_seconds)
    metadata['drain_end_time_ns']=time.perf_counter_ns()
    metadata['drain_seconds']=(metadata['drain_end_time_ns']-end)/1e9
    metadata['dispatch_decisions_sha256']=hashlib.sha256(json.dumps(
        [dict(request_id=r['request_id'],model=r['model'],routing_decision=r['routing_decision']) for r in records],
        sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return metadata
