"""Continuous, explicitly observed native-call spans, including host transitions.

The legacy native worker remains unchanged. Its inference spans still contain
host work/transfers/CUDA; continuous coverage does not make them pure GPU work.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).parent))
from observed_worker import emit, module


class CompleteObserver:
    def __init__(self, torch, request_id, start_ns):
        self.torch,self.rid,self.stages=torch,request_id,[]
        self.current=None
        self._start('host_setup','host',start_ns)

    def _start(self, segment_id, kind, timestamp):
        name=f'GS_STAGE:{self.rid}:{kind}:{segment_id}'
        self.current=dict(id=segment_id,kind=kind,start_ns=timestamp,
            scope='explicit native call and marker overhead; inference includes host/transfers/CUDA')
        self.torch.cuda.nvtx.range_push(name)
        emit('PHASE',dict(request_id=self.rid,event='start',segment_id=segment_id,
                         kind=kind,perf_ns=timestamp,nvtx_name=name))

    def _end(self,timestamp):
        span=self.current
        span['end_ns']=timestamp
        emit('PHASE',dict(request_id=self.rid,event='end',segment_id=span['id'],
                         kind=span['kind'],perf_ns=timestamp))
        self.torch.cuda.nvtx.range_pop()
        self.stages.append(span)
        self.current=None

    def switch(self, segment_id, kind):
        timestamp=time.perf_counter_ns()
        self._end(timestamp)
        self._start(segment_id,kind,timestamp)

    def call(self, segment_id, kind, function):
        self.switch(segment_id,kind)
        try:
            return function()
        finally:
            self.switch('host_after_'+segment_id,'host')

    def finish(self):
        timestamp=time.perf_counter_ns()
        self._end(timestamp)
        return timestamp


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',required=True);p.add_argument('--source-root',required=True)
    p.add_argument('--execute',action='store_true',required=True)
    args=p.parse_args();spec_path=Path(args.spec);spec=json.loads(spec_path.read_text())
    sys.path.insert(0,str(Path(args.source_root).resolve()))
    from gsdeploy.backends.compact_memory import allocator_environment
    for k,v in allocator_environment(spec['execution']).items():
        if os.environ.get(k)!=v:raise ValueError('Frozen allocator environment mismatch: '+k)
    resources=dict(spec['resources'])
    resources['request_output']=str(Path(resources['request_output'])/str(os.getpid()))
    pp=spec.get('adapter')=='retained_ppocrv6'
    if pp:
        source=Path(spec['adapter_source'])
        if hashlib.sha256(source.read_bytes()).hexdigest()!=spec['adapter_source_sha256']:
            raise ValueError('Changed retained OCR adapter')
        adapter=module('retained_pp_complete_observed',source).Adapter(spec,resources)
    else:
        from gsdeploy.backends.compact_native import ProfileAdapter
        adapter=ProfileAdapter(spec,resources)
    torch=adapter.torch
    tasks=['ocr'] if pp else (['text','image','ocr','asr']
        if spec['candidate']['slot'].startswith('generalist') else [spec['candidate']['slot']])
    emit('READY',dict(pid=os.getpid(),tasks=tasks,worker=spec['candidate']['slot'],
        torch=torch.__version__,clock='time.perf_counter_ns',
        spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        observation='continuous native-call spans; inference remains mixed host/transfers/CUDA'))
    for line in sys.stdin:
        request=None;response_flushed=False;observer=None
        try:
            request=json.loads(line)
            if request.get('op')=='shutdown':
                emit('BYE',dict(pid=os.getpid()));return
            rid,sample=request['request_id'],request['sample']
            if request['task_type'] not in tasks or sample['task']!=request['task_type']:
                raise ValueError('Unsupported or mismatched task')
            start_perf,start_unix=time.perf_counter_ns(),time.time_ns()
            observer=CompleteObserver(torch,rid,start_perf)
            emit('START',dict(request_id=rid,perf_ns=start_perf,unix_ns=start_unix,pid=os.getpid()))
            torch.cuda.reset_peak_memory_stats()
            if pp:
                result=observer.call('native_ocr','gpu',lambda:adapter.run(sample))
            else:
                adapter.executor.candidate=spec.get('task_candidates',{}).get(request['task_type'],spec['candidate'])
                observer.call('pre_sync','sync',torch.cuda.synchronize)
                service_start=time.perf_counter_ns()
                result=observer.call('native_inference','gpu',lambda:adapter.executor.infer(sample,evaluate_quality=False))
                observer.call('post_sync','sync',torch.cuda.synchronize)
                from gsdeploy.backends.compact_memory import reclaim_unused_cache
                result['cache_reclamation']=observer.call('cache_reclamation','cleanup',
                    lambda:reclaim_unused_cache(torch,adapter.cache_release_threshold_mib))
                observer.call('final_sync','sync',torch.cuda.synchronize)
                result.update(service_ms=(time.perf_counter_ns()-service_start)/1e6,
                    reference=sample.get('label'),answers=sample.get('answers'),
                    memory_optimizations=copy.deepcopy(getattr(adapter.executor,'memory_audit',{})))
            memory=adapter.memory()
            end_perf=observer.finish();end_unix=time.time_ns()
            result.update(request_id=rid,task_type=request['task_type'],start_ns=start_unix,
                end_ns=end_unix,start_perf_ns=start_perf,end_perf_ns=end_perf,
                observed_stages=observer.stages,memory=memory,
                stage_semantics='continuous native-call spans; gpu kind includes host/transfers/CUDA',
                worker_latency_ms=(end_perf-start_perf)/1e6,
                queue_wait_ms=(start_unix-request.get('submit_ns',start_unix))/1e6)
            emit('RESULT',result);response_flushed=True
        except Exception as exc:
            if observer is not None and observer.current is not None:observer.finish()
            emit('ERROR',dict(request_id=request.get('request_id') if request else None,
                error=repr(exc),traceback=traceback.format_exc()));response_flushed=True
        finally:
            if response_flushed and isinstance(request,dict) and request.get('request_id'):
                emit('WORKER_READY',dict(request_id=request['request_id'],pid=os.getpid(),
                    perf_ns=time.perf_counter_ns(),clock='host_perf_counter_ns',
                    boundary='previous_handler_next_request_ready',scope='after_response_flush_before_next_read'))


if __name__=='__main__':main()
