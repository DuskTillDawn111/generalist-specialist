"""Opt-in stream-event completion evidence, without new CUDA synchronization.

Events bracket a language forward on the caller's current CUDA stream. They
do not certify other streams, an entire CUDA DAG, fractional unfinished work,
generated-token count, or uninterrupted SM service. Every query retains its
host bracket; parent reader availability is added by the existing transport.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import math
import time

from .semantic_observer import SemanticObserver, language_module, _ACTIVE

SCHEMA = 'gsdeploy.stream_event_forward_observation.v1'


class BracketedNvtx:
    def __init__(self, delegate, clock):
        self.delegate, self.clock = delegate, clock
        self.stack, self.records = [], []

    def range_push(self, name):
        before = self.clock()
        result = self.delegate.range_push(name)
        after = self.clock()
        row = dict(name=name, push_before_ns=before, push_after_ns=after)
        self.stack.append(row); self.records.append(row)
        return result

    def range_pop(self):
        before = self.clock()
        result = self.delegate.range_pop()
        after = self.clock()
        row = self.stack.pop()
        row.update(pop_before_ns=before, pop_after_ns=after)
        return result


class DeviceProgressObserver(SemanticObserver):
    def __init__(self, owner, request_id, *, torch, nvtx=None, sink=None,
                 clock=time.perf_counter_ns):
        self.bracketed_nvtx = BracketedNvtx(nvtx, clock) if nvtx is not None else None
        super().__init__(owner, request_id, nvtx=self.bracketed_nvtx, sink=sink, clock=clock)
        self.torch = torch
        self.device_rows, self.queries, self.pending = [], [], []
        self.sequence = 0

    def _observation(self, row):
        self.sequence += 1
        value = dict(schema=SCHEMA, request_id=self.request_id,
            event='observation', kind='device_event_query',
            segment_id=row['phase_id'], observation_sequence=self.sequence,
            perf_ns=row['host_query_end_ns'], **row)
        self.queries.append(value)
        if self.sink is not None:
            try:
                self.sink(copy.deepcopy(value))
            except Exception as exc:
                self.errors.append(dict(where='device_sink',type=type(exc).__name__))

    def poll(self):
        """A query returning false is unknown progress, never fractional work."""
        remaining = []
        for item in self.pending:
            end = item.get('_end')
            if end is None:
                remaining.append(item); continue
            try:
                before = self.clock()
                done = bool(end.query())
                after = self.clock()
                row = dict(phase_id=item['phase_id'], stream_id=item['stream_id'],
                    event_id=item['event_id'], device_index=item['device_index'],
                    forward_index=item['forward_index'],
                    host_query_begin_ns=before, host_query_end_ns=after,
                    end_event_complete=done, in_flight_remaining_fraction=None,
                    evidence_scope='recorded_current_stream_only',
                    all_request_GPU_work_complete=False,
                    completion_time_ns=None, physical_SM_service_ns=None)
                if done:
                    # Both event records occur in the same stream. An elapsed
                    # interval includes scheduling/queue gaps and other work.
                    elapsed = float(item['_begin'].elapsed_time(end))
                    if not math.isfinite(elapsed) or elapsed < 0:
                        raise ValueError('Finite nonnegative completed stream interval required')
                    row['recorded_stream_elapsed_ms'] = elapsed
                    item['completion_query_bracket_ns'] = [before,after]
                    item['recorded_stream_elapsed_ms'] = row['recorded_stream_elapsed_ms']
                else:
                    row['recorded_stream_elapsed_ms'] = None
                    remaining.append(item)
                self._observation(row)
            except Exception as exc:
                item['observation_error'] = type(exc).__name__
                self.errors.append(dict(where='device_query',type=type(exc).__name__))
                # A broken optional probe is not completion evidence.
                remaining.append(item)
        self.pending = remaining

    def before(self, module, args, kwargs):
        self.poll()
        super().before(module, args, kwargs)
        phase = self.records[-1]
        try:
            stream = self.torch.cuda.current_stream()
            begin = self.torch.cuda.Event(enable_timing=True)
            end = self.torch.cuda.Event(enable_timing=True)
            before = self.clock(); begin.record(stream); after = self.clock()
            item = dict(phase_id=phase['id'], forward_index=phase['forward_index'],
                event_id=self.request_id+':'+phase['id']+':stream_end',
                stream_id=int(stream.cuda_stream), device_index=int(stream.device.index),
                begin_record_host_bracket_ns=[before,after],
                declared_stream_scope='caller_current_stream_at_forward_BEGIN',
                other_stream_dependencies_certified=False,
                _stream=stream, _begin=begin, _end_pending=end, _end=None)
            phase['_device_probe_index'] = len(self.device_rows)
            self.device_rows.append(item); self.pending.append(item)
        except Exception as exc:
            self.errors.append(dict(where='device_record_begin',type=type(exc).__name__))
        return None

    def _finish(self, basis):
        phase, _ = self._stack[-1]
        index = phase.pop('_device_probe_index', None)
        if index is not None:
            item = self.device_rows[index]
            try:
                current = self.torch.cuda.current_stream()
                if (int(current.cuda_stream) != item['stream_id']
                        or int(current.device.index) != item['device_index']):
                    raise ValueError('Current device or stream changed across language forward')
                before = self.clock()
                item['_end_pending'].record(item['_stream'])
                after = self.clock()
                item['_end'] = item['_end_pending']
                item['end_record_host_bracket_ns'] = [before,after]
                item['forward_exit_hook_basis'] = basis
                # always_call also runs when forward raises. This field says
                # nothing about success; the worker's RESULT/ERROR retains it.
                item['forward_success_certified_by_probe'] = False
            except Exception as exc:
                item['observation_error'] = type(exc).__name__
                self.errors.append(dict(where='device_record_end',type=type(exc).__name__))
        super()._finish(basis)
        self.poll()

    def document(self):
        value = super().document()
        value['device_progress'] = dict(schema=SCHEMA,
            records=[{k:v for k,v in item.items() if not k.startswith('_')} for item in self.device_rows],
            queries=list(self.queries), nvtx_host_brackets=[] if self.bracketed_nvtx is None else list(self.bracketed_nvtx.records),
            pending_event_ids=[item['event_id'] for item in self.pending],
            cuda_synchronization_added=False, unfinished_fraction_identified=False,
            completion_is_single_recorded_stream_only=True, physical_SM_service_identified=False,
            timestamps='host query brackets; actual parent receipt supplied by transport; device completion timestamp not observed',
            progress_model_qualified=False)
        return copy.deepcopy(value)


@contextmanager
def observe_language_forwards(executor, request_id, *, torch, nvtx=None, sink=None,
                             clock=time.perf_counter_ns):
    owner = language_module(executor)
    if owner in _ACTIVE:
        raise ValueError('Language module already has an active semantic observer')
    observer = DeviceProgressObserver(owner, request_id, torch=torch, nvtx=nvtx, sink=sink, clock=clock)
    _ACTIVE[owner] = observer
    handles = []
    try:
        handles.append(owner.register_forward_pre_hook(observer.before, with_kwargs=True))
        handles.append(owner.register_forward_hook(observer.after, with_kwargs=True, always_call=True))
        yield observer
    finally:
        while observer._stack:
            observer._finish('context_exit_fallback')
        observer.poll()
        for handle in reversed(handles):
            handle.remove()
        _ACTIVE.pop(owner,None)
