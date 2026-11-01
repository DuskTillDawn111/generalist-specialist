"""Gate transition replay and actual collector handoff, with no GPU workload."""

import threading
import queue
import unittest

from gsdeploy.runtime.admission import RequestAdmission
from gsdeploy.runtime.deployment import DeploymentRuntime, TimestampedEvents
from tests.test_resource_control import resources


class RuntimeEventContractTests(unittest.TestCase):
    def test_send_failure_releases_owned_permit_with_failure_reason(self):
        runtime = DeploymentRuntime(None, {})
        runtime.admission = RequestAdmission(resources("request_admission"))
        runtime.send_locks["s#0"] = threading.Lock()
        runtime.queues["s#0"] = queue.Queue()
        row = {"request_id": "one", "model": "s#0", "task": "text", "status": "queued",
               "pool_id": "pool", "arrival_index": 0}
        runtime.queues["s#0"].put((row, {}))

        class FailedTransport:
            def send(self, payload):
                runtime.stopping.set()
                raise OSError("test pipe failure")

        runtime.workers["s#0"] = FailedTransport()
        runtime._send("s#0")
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["admission_release_reason"], "send_failure")
        self.assertEqual(runtime.admission.snapshot()["active"], {})
        self.assertEqual([e["kind"] for e in row["admission_events"]], ["enqueue", "grant", "release"])

    def test_fifo_replay_records_transition_states_and_incremental_cursor(self):
        gate = RequestAdmission(resources("request_admission"))
        gate.acquire("s#0", "one")
        result = []
        sender = threading.Thread(target=lambda: result.append(gate.acquire("g#0", "two")))
        sender.start()
        with gate.condition:
            self.assertTrue(gate.condition.wait_for(lambda: bool(gate.waiting), timeout=2))
        cursor = gate.snapshot()["last_sequence"]
        released = gate.release("one", reason="worker_result")
        sender.join(2)
        self.assertFalse(sender.is_alive())
        self.assertEqual(result, [True])
        gate.release("two", reason="worker_error")
        self.assertIsNone(gate.release("two", reason="duplicate"))
        gate.close()
        snapshot = gate.snapshot()
        events = snapshot["events"]
        self.assertEqual([e["sequence"] for e in events], list(range(1, len(events) + 1)))
        self.assertEqual([e["time_ns"] for e in events], sorted(e["time_ns"] for e in events))
        active, waiting = {}, []
        for event in events:
            rid, kind = event["request_id"], event["kind"]
            if kind == "enqueue":
                waiting.append(rid)
            elif kind == "grant":
                self.assertEqual(waiting.pop(0), rid)
                self.assertFalse(active)
                active[rid] = event["alias"]
            elif kind == "release":
                self.assertEqual(active.pop(rid), event["alias"])
            self.assertEqual(len(active), event["active_count_after"])
            self.assertEqual(len(waiting), event["waiting_count_after"])
        self.assertEqual(active, snapshot["active"])
        self.assertEqual(gate.snapshot(cursor)["events"], events[cursor:])
        self.assertEqual(gate.snapshot(snapshot["last_sequence"])["events"], [])
        second_grant = next(e for e in events if e["kind"] == "grant" and e["request_id"] == "two")
        self.assertGreater(second_grant["sequence"], released["sequence"])
        for cursor in (-1, True, len(events) + 1):
            with self.assertRaises(ValueError):
                gate.snapshot(cursor)

    def test_shutdown_cancellation_and_alias_exit_do_not_invent_release(self):
        gate = RequestAdmission(resources("request_admission"))
        gate.acquire("s#0", "running")
        result = []
        sender = threading.Thread(target=lambda: result.append(gate.acquire("g#0", "waiting")))
        sender.start()
        with gate.condition:
            self.assertTrue(gate.condition.wait_for(lambda: bool(gate.waiting), timeout=2))
        gate.close()
        sender.join(2)
        self.assertEqual(result, [False])
        self.assertEqual([e["kind"] for e in gate.request_events("waiting")], ["enqueue", "cancel"])
        self.assertEqual(gate.request_events("waiting")[-1]["reason"], "gate_closed")
        released = gate.release_alias("s#0", reason="worker_exit")
        self.assertEqual(len(released), 1)
        self.assertEqual(released[0]["reason"], "worker_exit")
        self.assertEqual(gate.release_alias("s#0", reason="worker_exit"), [])
        self.assertFalse(gate.acquire("s#0", "closed"))
        self.assertEqual(gate.request_events("closed")[0]["kind"], "reject")

    def test_reader_receipt_while_collector_blocked_does_not_release_permit(self):
        got_event = threading.Event()

        class TrackedEvents(TimestampedEvents):
            def get(self, *args, **kwargs):
                event = super().get(*args, **kwargs)
                got_event.set()
                return event

        runtime = DeploymentRuntime(None, {})
        runtime.events = TrackedEvents()
        runtime.admission = RequestAdmission(resources("request_admission"))
        runtime.admission.acquire("s#0", "one")
        row = {"request_id": "one", "model": "s#0", "task": "text", "status": "submitted"}
        runtime.pending["one"] = row
        collector = threading.Thread(target=runtime._collect)
        try:
            with runtime.lock:
                runtime.events.put(("s#0", "RESULT", {"request_id": "one", "task_type": "text"}))
                collector.start()
                self.assertTrue(got_event.wait(2))
                self.assertEqual(runtime.admission.snapshot()["active"], {"one": "s#0"})
                self.assertNotIn("admission_released_time_ns", row)
            with runtime.admission.condition:
                self.assertTrue(runtime.admission.condition.wait_for(lambda: not runtime.admission.active, timeout=2))
            with runtime.lock:
                self.assertEqual(row["status"], "completed")
                self.assertEqual(row["completion_time_ns"], row["response_reader_event_enqueue_time_ns"])
                self.assertLessEqual(row["completion_time_ns"], row["response_collector_dequeued_time_ns"])
                self.assertLessEqual(row["response_collector_dequeued_time_ns"], row["response_collector_processing_time_ns"])
                self.assertLessEqual(row["response_collector_processing_time_ns"], row["admission_released_time_ns"])
                self.assertEqual(row["admission_release_reason"], "worker_result")
                self.assertEqual(row["admission_events"], runtime.admission.request_events("one"))
        finally:
            runtime.collect_stop.set()
            if collector.ident:
                collector.join(2)
        self.assertFalse(collector.is_alive())


if __name__ == "__main__":
    unittest.main()
