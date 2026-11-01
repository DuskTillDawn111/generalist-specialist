"""Native retained models with opt-in stream-event forward observations.

Uses the frozen native adapter, generation settings and cache release sequence.
The inference phase includes Python, transfers and CUDA work. NVTX alone does
not establish GPU activity; CUPTI trace association is required separately.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def emit(kind, payload):
    print(kind + '|' + json.dumps(payload, ensure_ascii=False), flush=True)


class Observer:
    def __init__(self, torch, request_id):
        self.torch, self.rid = torch, request_id
        self.stages = []

    def call(self, segment_id, kind, function):
        name = f'GS_STAGE:{self.rid}:{kind}:{segment_id}'
        self.torch.cuda.nvtx.range_push(name)
        start = time.perf_counter_ns()
        emit('PHASE', dict(request_id=self.rid, event='start', segment_id=segment_id,
                           kind=kind, perf_ns=start, nvtx_name=name))
        try:
            return function()
        finally:
            end = time.perf_counter_ns()
            emit('PHASE', dict(request_id=self.rid, event='end', segment_id=segment_id,
                               kind=kind, perf_ns=end, nvtx_name=name))
            self.torch.cuda.nvtx.range_pop()
            self.stages.append(dict(id=segment_id, kind=kind, start_ns=start, end_ns=end,
                                    scope='explicit native call; GPU phase includes host launch work'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', required=True)
    p.add_argument('--source-root', required=True)
    p.add_argument('--execute', action='store_true', required=True)
    args = p.parse_args()
    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text())
    sys.path.insert(0, str(Path(args.source_root).resolve()))
    from gsdeploy.backends.compact_memory import allocator_environment
    for k, v in allocator_environment(spec['execution']).items():
        if os.environ.get(k) != v:
            raise ValueError('Frozen allocator environment mismatch: ' + k)
    resources = dict(spec['resources'])
    resources['request_output'] = str(Path(resources['request_output']) / str(os.getpid()))
    pp = spec.get('adapter') == 'retained_ppocrv6'
    if pp:
        source = Path(spec['adapter_source'])
        if hashlib.sha256(source.read_bytes()).hexdigest() != spec['adapter_source_sha256']:
            raise ValueError('Changed retained OCR adapter')
        adapter = module('retained_pp_observed', source).Adapter(spec, resources)
    else:
        from gsdeploy.backends.compact_native import ProfileAdapter
        adapter = ProfileAdapter(spec, resources)
    torch = adapter.torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.predictor_validation_20261027.device_progress_observer_v1 import observe_language_forwards
    tasks = ['ocr'] if pp else (['text', 'image', 'ocr', 'asr']
             if spec['candidate']['slot'].startswith('generalist') else [spec['candidate']['slot']])
    emit('READY', dict(pid=os.getpid(), tasks=tasks, worker=spec['candidate']['slot'],
                       torch=torch.__version__, clock='time.perf_counter_ns',
                       spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                       observation='explicit native phases; counters unknown until externally collected'))
    for line in sys.stdin:
        request = None
        response_flushed = False
        try:
            request = json.loads(line)
            if request.get('op') == 'shutdown':
                emit('BYE', dict(pid=os.getpid()))
                return
            if request.get('op') == 'profile_start':
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
                emit('RESULT', dict(request_id=request['request_id'], task_type='__profile_start__'))
                response_flushed = True
                continue
            rid, sample = request['request_id'], request['sample']
            if request['task_type'] not in tasks or sample['task'] != request['task_type']:
                raise ValueError('Unsupported or mismatched task')
            torch.cuda.reset_peak_memory_stats()
            observer = Observer(torch, rid)
            start_perf, start_unix = time.perf_counter_ns(), time.time_ns()
            emit('START', dict(request_id=rid, perf_ns=start_perf, unix_ns=start_unix, pid=os.getpid()))
            if pp:
                result = observer.call('native_ocr', 'gpu', lambda: adapter.run(sample))
            else:
                adapter.executor.candidate = spec.get('task_candidates', {}).get(request['task_type'], spec['candidate'])
                observer.call('pre_sync', 'sync', torch.cuda.synchronize)
                service_start = time.perf_counter_ns()
                with observe_language_forwards(adapter.executor, rid, torch=torch, nvtx=torch.cuda.nvtx,
                                                sink=lambda p: emit('PHASE', p)) as semantic:
                    result = observer.call('native_inference', 'gpu',
                                           lambda: adapter.executor.infer(sample, evaluate_quality=False))
                result['semantic_forward_observations'] = semantic.document()
                observer.call('post_sync', 'sync', torch.cuda.synchronize)
                from gsdeploy.backends.compact_memory import reclaim_unused_cache
                result['cache_reclamation'] = observer.call('cache_reclamation', 'cleanup',
                    lambda: reclaim_unused_cache(torch, adapter.cache_release_threshold_mib))
                observer.call('final_sync', 'sync', torch.cuda.synchronize)
                # Query after the existing sync; the observer adds no synchronization.
                semantic.poll()
                result['semantic_forward_observations'] = semantic.document()
                result.update(service_ms=(time.perf_counter_ns()-service_start)/1e6,
                              reference=sample.get('label'), answers=sample.get('answers'),
                              memory_optimizations=copy.deepcopy(getattr(adapter.executor, 'memory_audit', {})))
            end_perf, end_unix = time.perf_counter_ns(), time.time_ns()
            result.update(request_id=rid, task_type=request['task_type'], start_ns=start_unix,
                          end_ns=end_unix, start_perf_ns=start_perf, end_perf_ns=end_perf,
                          observed_stages=observer.stages, memory=adapter.memory(),
                          worker_latency_ms=(end_perf-start_perf)/1e6,
                          queue_wait_ms=(start_unix-request.get('submit_ns', start_unix))/1e6)
            emit('RESULT', result)
            response_flushed = True
        except Exception as exc:
            emit('ERROR', dict(request_id=request.get('request_id') if request else None,
                               error=repr(exc), traceback=traceback.format_exc()))
            response_flushed = True
        finally:
            if response_flushed and isinstance(request, dict) and request.get('request_id'):
                emit('WORKER_READY', dict(request_id=request['request_id'], pid=os.getpid(),
                    perf_ns=time.perf_counter_ns(), clock='host_perf_counter_ns',
                    boundary='previous_handler_next_request_ready',
                    scope='after_response_flush_before_next_read'))


if __name__ == '__main__':
    main()
