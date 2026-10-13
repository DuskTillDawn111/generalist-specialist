"""Instrument existing Worker transport; model inference stays in the old workers."""

from collections import OrderedDict, deque
import queue
import subprocess
import threading
import time

from joint_deploy.run_static_hybrid import Worker, WORKERS
from joint_deploy.run_pure_qwen import PYTHON, WORKER_SCRIPT


def default_registry():
    return {**WORKERS, 'qwen': {'python': PYTHON, 'script': WORKER_SCRIPT}}


class TimestampedEvents(queue.Queue):
    def put(self, item, block=True, timeout=None):
        # Called directly by the existing stdout reader, before collector queue delay.
        super().put((*item, time.perf_counter_ns(), time.time_ns()), block, timeout)


class ObservedWorker(Worker):
    """Reuse the parser and transport, adding ordered EOF and reader lifecycle observation."""
    def __init__(self, *args):
        super().__init__(*args)
        self.reader_done = threading.Event()

    def _reader(self):
        try:
            super()._reader()
        finally:
            self.event_queue.put((self.name, 'EXIT', {'exit_code': self.process.poll()}))
            self.reader_done.set()


class DeploymentRuntime:
    def __init__(self, config, registry=None):
        self.config = config
        self.registry = default_registry() if registry is None else registry
        self.events = TimestampedEvents()
        self.workers, self.ready, self.pending, self.queues = {}, {}, {}, {}
        self.senders, self.send_locks = {}, {}
        self.errors, self.byes, self.shutdown_report = [], set(), {}
        self.lock = threading.RLock()
        self.stopping = threading.Event()
        self.collect_stop = threading.Event()
        self.collector = None
        self.active = OrderedDict()
        self.active_models = {}
        self.terminal_changes = deque()
        self.active_high_water = 0

    def refresh(self, record):
        """Caller holds the existing RLock; archive and transport stay unchanged."""
        key = record['request_id']
        group = self.active_models.setdefault(record['model'], OrderedDict())
        if record['status'] in ('queued', 'submitted'):
            self.active[key] = record
            group[key] = record
            self.active_high_water = max(self.active_high_water, len(self.active))
        else:
            self.active.pop(key, None)
            group.pop(key, None)
            self.terminal_changes.append(record)

    def pressure(self, now_ns):
        head = next(iter(self.active.values()), None)
        return len(self.active), (now_ns-head['enqueued_time_ns'])/1e9 if head else 0.

    def projection(self, G, S, default_cost):
        # Merge in original archive insertion order, NOT worker or submit order.
        # Original decide retains its exact sort and floating-point summation.
        candidates = [x for x in self.active.values() if x['model'] in (G, S)]
        return [dict(role='G' if x['model']==G else 'S', task=x['task'],
                     submit_perf_ns=x.get('actual_submit_time_ns'),
                     enqueue_perf_ns=x['enqueued_time_ns'],
                     S_service_estimate_ms=x.get('router_S_service_estimate_ms', default_cost))
                for x in candidates]

    def start(self):
        for name, count in self.config.deployment.items():
            if count > 1:
                raise ValueError('Runtime prototype currently supports x_m in {0,1}; schema permits larger counts')
            if count and name not in self.registry:
                raise ValueError(f'No worker launcher registered for {name}')
        for name, count in sorted(self.config.deployment.items()):
            if not count:
                continue
            worker = ObservedWorker(name, self.registry[name], self.events)
            self.workers[name] = worker
            worker.start()
            deadline = time.monotonic() + self.config.startup_timeout_seconds
            while name not in self.ready:
                if worker.process.poll() is not None:
                    raise RuntimeError(f'{name} exited before READY: exit={worker.process.returncode}')
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'{name} READY timeout')
                try:
                    m, kind, payload, _, _ = self.events.get(timeout=0.1)
                except queue.Empty:
                    continue
                if kind == 'READY':
                    expected = {task for task, route in self.config.routing_rps.items() if route.get(m, 0) > 0}
                    if not expected <= set(payload.get('tasks', [])):
                        raise RuntimeError(f'{m} READY capabilities do not cover configured routing')
                    self.ready[m] = payload
                elif kind == 'ERROR':
                    raise RuntimeError(f'{m} startup error: {payload}')
            self.queues[name] = queue.Queue(maxsize=self.config.max_pending_per_model)
            self.send_locks[name] = threading.Lock()
        self.collector = threading.Thread(target=self._collect, daemon=True)
        self.collector.start()
        for name in self.workers:
            self.senders[name] = threading.Thread(target=self._send, args=(name,), daemon=True)
            self.senders[name].start()

    def enqueue(self, record, sample):
        with self.lock:
            self.pending[record['request_id']] = record
            record['enqueued_time_ns'] = time.perf_counter_ns()
            record['status'] = 'queued'
            self.refresh(record)
        try:
            self.queues[record['model']].put_nowait((record, sample))
        except queue.Full:
            with self.lock:
                record.update(status='error', error='host send queue full; request not submitted')
                self.refresh(record)

    def _send(self, model):
        while not self.stopping.is_set():
            try:
                record, sample = self.queues[model].get(timeout=0.1)
            except queue.Empty:
                continue
            with self.lock:
                if record['status'] != 'queued':
                    continue
            try:
                with self.send_locks[model]:
                    with self.lock:
                        if record['status'] != 'queued' or self.stopping.is_set():
                            continue
                        record.update(actual_submit_time_ns=time.perf_counter_ns(),
                                      actual_submit_unix_ns=time.time_ns(), status='submitted')
                        self.refresh(record)
                    self.workers[model].send({
                        'request_id': record['request_id'], 'pool_id': record['pool_id'],
                        'arrival_index': record['arrival_index'], 'task_type': record['task'],
                        'sample': sample, 'submit_ns': record['actual_submit_unix_ns'],
                    })
                    with self.lock:
                        record['submit_write_completed_time_ns'] = time.perf_counter_ns()
            except Exception as exc:
                with self.lock:
                    if record['status'] not in ('completed', 'incomplete'):
                        record.update(status='error', error=f'send failed: {exc}')
                        self.refresh(record)

    def _collect(self):
        while not self.collect_stop.is_set() or not self.events.empty():
            try:
                model, kind, payload, received_ns, unix_ns = self.events.get(timeout=0.05)
            except queue.Empty:
                continue
            with self.lock:
                if kind == 'BYE':
                    self.byes.add(model)
                    continue
                if kind == 'EXIT':
                    for record in list(self.active_models.get(model, {}).values()):
                        if record['model'] == model and record['status'] in ('queued', 'submitted'):
                            record.update(status='error', error='worker stdout closed before response')
                            self.refresh(record)
                    continue
                if kind not in ('RESULT', 'ERROR'):
                    continue
                record = self.pending.get(payload.get('request_id'))
                if record is None or record['model'] != model or record['status'] not in ('submitted', 'incomplete'):
                    self.errors.append({'model': model, 'kind': kind, 'payload': payload,
                                        'error': 'unmatched or duplicate response'})
                    continue
                if record['status'] == 'incomplete':
                    record['late_response_after_drain'] = payload
                    record['late_response_time_ns'] = received_ns
                    continue
                record['completion_time_ns'] = received_ns
                record['completion_unix_ns'] = unix_ns
                record['worker_response'] = payload
                if kind == 'ERROR':
                    record.update(status='error', error=payload.get('error', 'worker ERROR'))
                elif payload.get('task_type') != record['task']:
                    record.update(status='error', error='worker task_type mismatch')
                else:
                    record.update(status='completed', worker_latency_ms=payload.get('worker_latency_ms'),
                                  worker_queue_wait_ms=payload.get('queue_wait_ms'))
                self.refresh(record)

    def drain(self, records, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if not any(r['status'] in ('queued', 'submitted') for r in records):
                    return
            time.sleep(0.005)
        with self.lock:
            for record in records:
                if record['status'] in ('queued', 'submitted'):
                    record.update(status='incomplete', error='drain timeout')
                    self.refresh(record)

    def close(self):
        self.stopping.set()
        # Shutdown writes may block behind a full pipe. Bound all waits and kill only owned children.
        def graceful(name, worker):
            lock = self.send_locks.get(name, threading.Lock())
            with lock:
                worker.shutdown()
        for name, worker in self.workers.items():
            threading.Thread(target=graceful, args=(name, worker), daemon=True).start()
        deadline = time.monotonic() + self.config.shutdown_timeout_seconds
        for name, worker in self.workers.items():
            process = worker.process
            if process is None:
                continue
            forced = False
            try:
                process.wait(timeout=max(0.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                forced = True
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            self.shutdown_report[name] = {'exit_code': process.returncode, 'forced': forced}
        for thread in self.senders.values():
            thread.join(timeout=1)
        # Readers deliver RESULT, BYE, EXIT in order before the collector is stopped.
        for worker in self.workers.values():
            if worker.process:
                worker.reader_done.wait(timeout=2)
        self.collect_stop.set()
        if self.collector:
            self.collector.join(timeout=2)
        for name, worker in self.workers.items():
            if name in self.shutdown_report:
                self.shutdown_report[name]['bye_received'] = name in self.byes
            if worker.process:
                worker.process.stdin.close()
                worker.process.stdout.close()
        return self.shutdown_report
