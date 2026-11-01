"""Manage worker lifecycles, send queues and timestamped request completion."""

import os
import queue
import subprocess
import threading
import time
from .transport import Worker
from ..resources import worker_environment
from .admission import admission_for_registry


class TimestampedEvents(queue.Queue):

    def put(self, item, block=True, timeout=None):
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
            self.event_queue.put(
                (self.name, "EXIT", {"exit_code": self.process.poll()})
            )
            self.reader_done.set()


class DeploymentRuntime:

    def __init__(self, config, registry):
        self.config = config
        self.registry = registry
        self.events = TimestampedEvents()
        self.workers, self.ready, self.pending, self.queues = ({}, {}, {}, {})
        self.senders, self.send_locks = ({}, {})
        self.errors, self.byes, self.shutdown_report = ([], set(), {})
        self.lock = threading.RLock()
        self.stopping = threading.Event()
        self.collect_stop = threading.Event()
        self.collector = None
        self.admission = None

    def start(self):
        for name, count in self.config.deployment.items():
            if count > 1:
                raise ValueError(
                    "Each transport ID represents one process; expand family replicas into IDs"
                )
            if count and name not in self.registry:
                raise ValueError(f"No worker launcher registered for {name}")
            if count:
                # Validate every binding before starting even the first worker.
                worker_environment(self.registry[name], os.environ)
        self.admission = admission_for_registry(
            {name: self.registry[name] for name, count in self.config.deployment.items() if count})
        for name, count in sorted(self.config.deployment.items()):
            if not count:
                continue
            worker = ObservedWorker(name, self.registry[name], self.events)
            self.workers[name] = worker
            worker.start()
            deadline = time.monotonic() + self.config.startup_timeout_seconds
            while name not in self.ready:
                if worker.process.poll() is not None:
                    raise RuntimeError(
                        f"{name} exited before READY: exit={worker.process.returncode}"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{name} READY timeout")
                try:
                    m, kind, payload, _, _ = self.events.get(timeout=0.1)
                except queue.Empty:
                    continue
                if kind == "READY":
                    expected = {
                        task
                        for (task, route) in self.config.routing_rps.items()
                        if route.get(m, 0) > 0
                    }
                    if not expected <= set(payload.get("tasks", [])):
                        raise RuntimeError(
                            f"{m} READY capabilities do not cover configured routing"
                        )
                    self.ready[m] = payload
                elif kind == "ERROR":
                    raise RuntimeError(f"{m} startup error: {payload}")
            self.queues[name] = queue.Queue(maxsize=self.config.max_pending_per_model)
            self.send_locks[name] = threading.Lock()
        self.collector = threading.Thread(target=self._collect, daemon=True)
        self.collector.start()
        for name in self.workers:
            self.senders[name] = threading.Thread(
                target=self._send, args=(name,), daemon=True
            )
            self.senders[name].start()

    def enqueue(self, record, sample):
        with self.lock:
            self.pending[record["request_id"]] = record
            record["enqueued_time_ns"] = time.perf_counter_ns()
            record["status"] = "queued"
        try:
            self.queues[record["model"]].put_nowait((record, sample))
        except queue.Full:
            with self.lock:
                record.update(
                    status="error", error="host send queue full; request not submitted"
                )

    def _sync_admission_record(self, record):
        """Copy exact gate transitions; record-copy time is not an event time."""
        with self.lock:
            events = self.admission.request_events(record["request_id"])
            record["admission_events"] = events
            for event in events:
                field = {"enqueue": "admission_wait_start_time_ns",
                         "grant": "admission_granted_time_ns",
                         "release": "admission_released_time_ns",
                         "cancel": "admission_cancelled_time_ns"}.get(event["kind"])
                if field is not None:
                    record[field] = event["time_ns"]
                if event["kind"] == "release":
                    record["admission_release_reason"] = event["reason"]

    def _release_admission(self, record, reason):
        self.admission.release(record["request_id"], reason=reason)
        self._sync_admission_record(record)

    def _send(self, model):
        while not self.stopping.is_set():
            try:
                record, sample = self.queues[model].get(timeout=0.1)
            except queue.Empty:
                continue
            with self.lock:
                if record["status"] != "queued":
                    continue
            try:
                if self.admission is not None:
                    granted = self.admission.acquire(model, record["request_id"])
                    self._sync_admission_record(record)
                    if not granted:
                        continue
                with self.send_locks[model]:
                    with self.lock:
                        if record["status"] != "queued" or self.stopping.is_set():
                            if self.admission is not None:
                                self._release_admission(record, "submission_cancelled")
                            continue
                        record.update(
                            actual_submit_time_ns=time.perf_counter_ns(),
                            actual_submit_unix_ns=time.time_ns(),
                            status="submitted",
                        )
                    self.workers[model].send(
                        {
                            "request_id": record["request_id"],
                            "pool_id": record["pool_id"],
                            "arrival_index": record["arrival_index"],
                            "task_type": record["task"],
                            "sample": sample,
                            "submit_ns": record["actual_submit_unix_ns"],
                        }
                    )
                    with self.lock:
                        record["submit_write_completed_time_ns"] = (
                            time.perf_counter_ns()
                        )
            except Exception as exc:
                if self.admission is not None:
                    self._release_admission(record, "send_failure")
                with self.lock:
                    if record["status"] not in ("completed", "incomplete"):
                        record.update(status="error", error=f"send failed: {exc}")

    def _collect(self):
        while not self.collect_stop.is_set() or not self.events.empty():
            try:
                model, kind, payload, received_ns, unix_ns = self.events.get(
                    timeout=0.05
                )
            except queue.Empty:
                continue
            collector_dequeued_ns = time.perf_counter_ns()
            with self.lock:
                collector_processing_ns = time.perf_counter_ns()
                if kind == "BYE":
                    self.byes.add(model)
                    continue
                if kind == "EXIT":
                    if self.admission is not None:
                        releases = self.admission.release_alias(model, reason="worker_exit")
                        for event in releases:
                            released_record = self.pending.get(event["request_id"])
                            if released_record is not None:
                                self._sync_admission_record(released_record)
                    for record in self.pending.values():
                        if record["model"] == model and record["status"] in (
                            "queued",
                            "submitted",
                        ):
                            record.update(
                                status="error",
                                error="worker stdout closed before response",
                            )
                    continue
                if kind == "WORKER_READY":
                    record = self.pending.get(payload.get("request_id"))
                    ready_ns = payload.get("perf_ns")
                    expected_pid = self.ready.get(model, {}).get("pid")
                    if (record is None or record["model"] != model
                            or payload.get("pid") != expected_pid or expected_pid is None
                            or payload.get("clock") != "host_perf_counter_ns"
                            or payload.get("boundary") != "previous_handler_next_request_ready"
                            or type(ready_ns) is not int or ready_ns < 0 or ready_ns > received_ns
                            or "previous_handler_next_request_ready" in record):
                        self.errors.append({"model": model, "kind": kind,
                                            "error": "unmatched or invalid worker readiness observation"})
                        continue
                    end_ns = record.get("worker_response", {}).get("end_perf_ns")
                    if end_ns is not None and ready_ns < end_ns:
                        self.errors.append({"model": model, "kind": kind,
                                            "error": "worker readiness precedes handler end"})
                        continue
                    record["previous_handler_next_request_ready"] = {
                        **payload, "received_time_ns": received_ns,
                        "collector_dequeued_time_ns": collector_dequeued_ns,
                        "collector_processing_time_ns": collector_processing_ns,
                    }
                    continue
                if kind in ("START", "PHASE"):
                    record = self.pending.get(payload.get("request_id"))
                    if (record is None or record["model"] != model
                            or record["status"] not in ("submitted", "incomplete")):
                        self.errors.append({"model": model, "kind": kind,
                                            "error": "unmatched execution observation"})
                        continue
                    event = {**payload, "received_time_ns": received_ns,
                             "collector_dequeued_time_ns": collector_dequeued_ns,
                             "collector_processing_time_ns": collector_processing_ns}
                    record.setdefault("execution_events", []).append(event)
                    if kind == "START":
                        record["observed_worker_start_time_ns"] = payload.get("perf_ns")
                    continue
                if kind not in ("RESULT", "ERROR"):
                    continue
                record = self.pending.get(payload.get("request_id"))
                if (
                    record is None
                    or record["model"] != model
                    or record["status"] not in ("submitted", "incomplete")
                ):
                    self.errors.append(
                        {
                            "model": model,
                            "kind": kind,
                            "payload": payload,
                            "error": "unmatched or duplicate response",
                        }
                    )
                    continue
                record["response_reader_event_enqueue_time_ns"] = received_ns
                record["response_collector_dequeued_time_ns"] = collector_dequeued_ns
                record["response_collector_processing_time_ns"] = collector_processing_ns
                if record["status"] == "incomplete":
                    if self.admission is not None:
                        self._release_admission(record, "late_worker_" + kind.lower())
                    record["late_response_after_drain"] = payload
                    record["late_response_time_ns"] = received_ns
                    continue
                record["completion_time_ns"] = received_ns
                record["completion_unix_ns"] = unix_ns
                record["worker_response"] = payload
                if self.admission is not None:
                    self._release_admission(record, "worker_" + kind.lower())
                if kind == "ERROR":
                    record.update(
                        status="error", error=payload.get("error", "worker ERROR")
                    )
                elif payload.get("success") is False:
                    record.update(status="error", error=payload.get("failure_reason", "worker reported unsuccessful inference"))
                elif payload.get("task_type") != record["task"]:
                    record.update(status="error", error="worker task_type mismatch")
                else:
                    record.update(
                        status="completed",
                        worker_latency_ms=payload.get("worker_latency_ms"),
                        worker_queue_wait_ms=payload.get("queue_wait_ms"),
                    )

    def drain(self, records, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if not any((r["status"] in ("queued", "submitted") for r in records)):
                    return
            time.sleep(0.005)
        with self.lock:
            for record in records:
                if record["status"] in ("queued", "submitted"):
                    record.update(status="incomplete", error="drain timeout")

    def close(self):
        self.stopping.set()
        if self.admission is not None:
            self.admission.close()

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
            self.shutdown_report[name] = {
                "exit_code": process.returncode,
                "forced": forced,
            }
        for thread in self.senders.values():
            thread.join(timeout=1)
        for worker in self.workers.values():
            if worker.process:
                worker.reader_done.wait(timeout=2)
        self.collect_stop.set()
        if self.collector:
            self.collector.join(timeout=2)
        for name, worker in self.workers.items():
            if name in self.shutdown_report:
                self.shutdown_report[name]["bye_received"] = name in self.byes
            if worker.process:
                worker.process.stdin.close()
                worker.process.stdout.close()
        return self.shutdown_report
