"""In-process real arrival/prepare/batch delivery runtime for one model owner.

This is an execution loop, not a forecast: clocks measure actual operations.
A virtual clock can be injected for CPU contract tests. The factory runs once;
the same owner's prepared requests flow through the public batching planner
and execute_selected_batch into execute_prepared, without retokenizing.
"""
from __future__ import annotations

from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path
import time
import traceback

from .batching import BatchConfiguration, execute_selected_batch, select_batch


SCHEMA = 'gsdeploy.batch_owner_runtime.v2'


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _receipt(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {'path', 'sha256'}:
        raise ValueError('Parent receipt requires exact path/sha256')
    path = Path(value['path']).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != value['sha256']:
        raise ValueError('Parent receipt changed')
    return dict(path=str(path), sha256=digest)


class BatchOwnerFailure(RuntimeError):
    """Original error is __cause__; report preserves completed/pending prefix."""
    def __init__(self, report):
        self.report = report
        super().__init__(report['failure']['error'])


class _SelectedPreparedBackend:
    def __init__(self, backend, selected):
        self.backend, self.selected = backend, selected
        self.execution = None

    def run_batch(self, samples):
        if samples != [row['sample'] for row in self.selected]:
            raise ValueError('Planner changed prepared sample order')
        self.execution = self.backend.execute_prepared([row['_prepared'] for row in self.selected],
                                                       evaluate_quality=False)
        if len(self.execution.results) != len(self.selected):
            raise ValueError('Prepared batch output cardinality changed')
        for row, result in zip(self.selected, self.execution.results):
            if result.get('sample_id') != row['sample']['id']:
                raise ValueError('Prepared backend changed sample identity/order')
        return self.execution.results


def run_arrivals(samples, arrivals, configuration, backend_factory, *, owner='shared_text',
                 parent_receipt=None, clock=time.perf_counter_ns, wait=None,
                 output=None, event_sink=None):
    """Execute public samples and a finite declared arrival schedule to drain.

    samples: list of complete text sample dicts with unique id.
    arrivals: [{request_id, sample_id, arrival_ns, alias?, pool_id?, ...}], where
      arrival_ns is a nonnegative offset from the epoch *after* one owner load.
      Arrival ties retain public list order. Input order need not be sorted.
    configuration: BatchConfiguration. No adaptive route/policy or GPU launch is
      hidden here: backend_factory is the sole explicit model-loading boundary.
    clock returns absolute monotonic integer ns; wait(delay_ns) waits/advances.
    output is an optional new JSON path (exclusive create); failures also save
      the completed prefix and pending IDs. No automatic retries/fallbacks.

    One owner serializes prepare and generate. Arrivals during either operation
    retain their scheduled times; preparation starts only when owner is free.
    Nonpreemptible work can delay a timeout; actual lateness is reported.
    """
    if not isinstance(configuration, BatchConfiguration):
        raise ValueError('Explicit BatchConfiguration required')
    if not isinstance(owner, str) or not owner:
        raise ValueError('Explicit owner identity required')
    samples, arrivals = copy.deepcopy(list(samples)), copy.deepcopy(list(arrivals))
    if not arrivals:
        raise ValueError('Nonempty declared arrival schedule required')
    index = {}
    for sample in samples:
        if (not isinstance(sample, dict) or not isinstance(sample.get('id'), str)
                or not sample['id'] or sample['id'] in index or sample.get('task') != 'text'):
            raise ValueError('Public samples require unique nonempty text IDs')
        index[sample['id']] = sample
    seen = set()
    for i, row in enumerate(arrivals):
        if (not isinstance(row, dict) or not isinstance(row.get('request_id'), str)
                or not row['request_id'] or row['request_id'] in seen
                or row.get('sample_id') not in index or type(row.get('arrival_ns')) is not int
                or row['arrival_ns'] < 0):
            raise ValueError('Invalid arrival identity/sample/nonnegative time')
        if (row.get('task', 'text') != 'text' or not isinstance(row.get('alias', owner), str)
                or not row.get('alias', owner)):
            raise ValueError('Text owner requires explicit logical alias')
        seen.add(row['request_id'])
        row['_arrival_index'] = i
    schedule = sorted(arrivals, key=lambda row:(row['arrival_ns'],row['_arrival_index']))
    public_arrivals = [{k:v for k,v in row.items() if not k.startswith('_')} for row in arrivals]
    target = None if output is None else Path(output).resolve()
    if target is not None and target.exists():
        raise FileExistsError(target)
    parent = _receipt(parent_receipt)
    wait = wait or (lambda delay_ns:time.sleep(min(delay_ns/1e9, .05)))
    last_clock = -1

    def tick():
        nonlocal last_clock
        value = clock()
        if type(value) is not int or value < last_clock:
            raise ValueError('Runtime clock must be integer monotonic ns')
        last_clock = value
        return value

    report = dict(schema=SCHEMA, status='STARTING', owner=owner,
        configuration=json.loads(json.dumps(asdict(configuration))), parent_receipt=parent,
        public_samples_sha256=_hash(samples), public_arrivals_sha256=_hash(public_arrivals),
        public_arrivals=public_arrivals,
        requested=len(arrivals), delivered=0, factory_calls=0,
        results=[], batches=[], events=[], not_for_training=True,
        evidence_scope='Execution mechanics only; separate physical-source/training qualification required',
        final_gain_eligible=False, GPU_launch_scope='Determined only by explicit backend_factory',
        delivery_scope='Whole batch after prepare/generate/finish/cleanup; no row-early delivery')
    pending, arrived = [], []
    cursor = 0
    backend = None
    epoch = None
    active_prepare = None
    active_batch = []

    def event(kind, **values):
        row = dict(event=kind, **values)
        report['events'].append(row)
        if event_sink is not None:
            event_sink(copy.deepcopy(row))

    def write():
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('x') as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write('\n')

    try:
        load_start = tick()
        event('OWNER_LOAD_START', absolute_ns=load_start, time_ns=None)
        report['factory_calls'] += 1
        backend = backend_factory()
        epoch = tick()
        report.update(status='RUNNING', epoch_ns=epoch, owner_load_start_ns=load_start,
                      owner_load_finish_ns=epoch, owner_load_duration_ns=epoch-load_start,
                      backend_identity=copy.deepcopy(backend.identity))
        event('OWNER_READY', time_ns=0, absolute_ns=epoch, owner_next_ready_ns=0)
        while report['delivered'] < len(schedule):
            now = tick()-epoch
            while cursor < len(schedule) and schedule[cursor]['arrival_ns'] <= now:
                arrival = schedule[cursor];cursor += 1;arrived.append(arrival)
                event('ARRIVAL_OBSERVED', request_id=arrival['request_id'],
                      scheduled_arrival_ns=arrival['arrival_ns'], time_ns=now,
                      observation_delay_ns=now-arrival['arrival_ns'])
            selection = select_batch(pending, now, configuration)
            if selection['status'] == 'UNSUPPORTED':
                raise ValueError(selection['reason'])
            if selection['status'] == 'READY':
                _receipt(parent)
                members = set(selection['members'])
                selected = [row for row in pending if row['request_id'] in members]
                if [r['request_id'] for r in selected] != selection['members']:
                    raise ValueError('Batch planner reordered/missed FIFO members')
                dispatch = tick()-epoch
                adapter = _SelectedPreparedBackend(backend, selected)
                active_batch = list(selection['members'])
                event('BATCH_START', time_ns=dispatch, batch_id=selection['batch_id'],
                      request_ids=selection['members'], owner_next_ready_ns=None)
                delivered = execute_selected_batch(selection, pending, adapter)
                complete = tick()-epoch
                metrics = copy.deepcopy(adapter.execution.metrics)
                if (metrics.get('batch_size') != selection['batch_size']
                        or metrics.get('padding_tokens') != selection['padding_tokens']):
                    raise ValueError('Executed batch dimensions differ from the actual selection')
                batch_record = dict(batch_id=selection['batch_id'], selection=copy.deepcopy(selection),
                    dispatch_ns=dispatch, completed_ns=complete, owner_next_ready_ns=complete,
                    parent_receipt_ns=complete, worker_next_ready_ns=complete,
                    actual_owner_busy_ns=complete-dispatch, backend_metrics=metrics,
                    timeout_lateness_ns=max(0, dispatch-(selected[0]['ready_ns']+configuration.max_wait_ns)))
                for planned, actual in zip(selected, delivered):
                    entry = dict(request_id=planned['request_id'], sample_id=planned['sample']['id'],
                        alias=planned['alias'], pool_id=planned['pool_id'],
                        arrival_index=planned['arrival_index'], batch_id=selection['batch_id'],
                        arrival_ns=planned['arrival_ns'], prepare_start_ns=planned['prepare_start_ns'],
                        ready_ns=planned['ready_ns'], batch_dispatch_ns=dispatch, completed_ns=complete,
                        parent_receipt_ns=complete, worker_next_ready_ns=complete,
                        pre_prepare_wait_ns=planned['prepare_start_ns']-planned['arrival_ns'],
                        preparation_ns=planned['ready_ns']-planned['prepare_start_ns'],
                        formation_wait_ns=dispatch-planned['ready_ns'],
                        queue_wait_ns=dispatch-planned['arrival_ns'], end_to_end_ns=complete-planned['arrival_ns'],
                        result=actual['result'])
                    report['results'].append(entry)
                report['batches'].append(batch_record)
                report['delivered'] += len(delivered)
                pending = [row for row in pending if row['request_id'] not in members]
                active_batch = []
                event('BATCH_COMPLETE', time_ns=complete, batch_id=selection['batch_id'],
                      request_ids=selection['members'], owner_next_ready_ns=complete)
                continue
            if arrived:
                arrival = arrived.pop(0)
                active_prepare = arrival['request_id']
                _receipt(parent)
                started = tick()-epoch
                event('PREPARE_START', time_ns=started, request_id=arrival['request_id'])
                prepared = backend.prepare(index[arrival['sample_id']])
                ready = tick()-epoch
                pending.append(dict(request_id=arrival['request_id'], sample=index[arrival['sample_id']],
                    owner=owner, alias=arrival.get('alias',owner), task='text',
                    execution_key=prepared.compatibility_key, input_tokens=prepared.input_tokens,
                    ready_ns=ready, prepare_start_ns=started, arrival_ns=arrival['arrival_ns'],
                    pool_id=arrival.get('pool_id',arrival['sample_id']), arrival_index=arrival['_arrival_index'],
                    _prepared=prepared))
                event('PREPARE_READY', time_ns=ready, request_id=arrival['request_id'],
                      input_tokens=prepared.input_tokens, execution_key=prepared.compatibility_key,
                      owner_next_ready_ns=ready)
                active_prepare = None
                continue
            future = []
            if selection.get('next_event_ns') is not None:
                future.append(selection['next_event_ns'])
            if cursor < len(schedule):
                future.append(schedule[cursor]['arrival_ns'])
            if not future:
                raise RuntimeError('Undelivered requests without a next scheduling event')
            next_event = min(future)
            if next_event <= now:
                raise RuntimeError('Batch scheduling made no progress')
            event('WAIT', time_ns=now, next_event_ns=next_event,
                  worker_next_ready_ns=selection.get('next_event_ns'),
                  next_arrival_ns=schedule[cursor]['arrival_ns'] if cursor < len(schedule) else None)
            wait(next_event-now)
        if parent != _receipt(parent):
            raise ValueError('Parent receipt identity changed')
        report.update(status='COMPLETED', completed_ns=tick()-epoch,
                      all_request_ids=[row['request_id'] for row in report['results']],
                      pending_request_ids=[], next_arrival_request_ids=[], owner_next_ready_ns=tick()-epoch)
        if set(report['all_request_ids']) != seen or len(report['all_request_ids']) != len(seen):
            raise ValueError('Delivered request identity coverage differs from public schedule')
        write()
        return report
    except Exception as exc:
        report.update(status='FAILED', failure=dict(error=repr(exc), traceback=traceback.format_exc()),
            active_prepare_request_id=active_prepare, active_batch_request_ids=active_batch,
            pending_request_ids=[row['request_id'] for row in pending],
            arrived_request_ids=[row['request_id'] for row in arrived],
            next_arrival_request_ids=[row['request_id'] for row in schedule[cursor:]],
            retry_permitted=False)
        try:
            write()
        except Exception as persistence_error:
            report['persistence_error'] = repr(persistence_error)
        raise BatchOwnerFailure(report) from exc


def run_retained_arrivals(samples, arrivals, configuration, spec_path, source_root, *,
                          execute=False, expected_spec_sha256=None, **runtime_options):
    """Explicit real-model convenience entry, sharing run_arrivals' one load.

    Calling/importing without execute=True cannot load a model. This creates
    new execution evidence; existing frozen plans/transport are not modified.
    """
    if execute is not True:
        raise RuntimeError('Retained model runtime requires explicit execute=True')
    from ..backends.shared_text_batch import load_retained
    return run_arrivals(samples, arrivals, configuration,
        lambda:load_retained(spec_path, source_root, execute=True,
                            expected_spec_sha256=expected_spec_sha256), **runtime_options)
