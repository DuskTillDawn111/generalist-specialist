"""Exercise actual admission behavior and CPU subprocesses, without any GPU."""

import copy
import json
from pathlib import Path
import sys
import threading
import time
import unittest

from gsdeploy.resources import prepare_registry, validate_resources
from gsdeploy.runtime.admission import RequestAdmission, admission_for_registry
from gsdeploy.runtime.deployment import DeploymentRuntime
from gsdeploy.runtime.runner import record
from gsdeploy.runtime.schema import MeasurementConfig
from tests.test_resource_control import resources


class AdmissionTests(unittest.TestCase):
    def test_group_permit_waits_for_release_and_wakes_on_close(self):
        gate = RequestAdmission(resources("request_admission"))
        self.assertTrue(gate.acquire("s#0", "first"))
        done = threading.Event(); results = []
        thread = threading.Thread(target=lambda: (results.append(gate.acquire("g#0", "second")), done.set()))
        thread.start()
        with gate.condition:
            self.assertTrue(gate.condition.wait_for(lambda: len(gate.waiting) == 1, timeout=2))
        self.assertFalse(done.is_set())
        gate.release("first")
        self.assertTrue(done.wait(2)); thread.join(2)
        self.assertEqual(results, [True])
        self.assertEqual(gate.snapshot()["peak_active"], 1)
        gate.release("second"); gate.close()
        self.assertFalse(gate.acquire("s#0", "after-close"))

    def test_shutdown_cancels_blocked_senders_without_starting_them(self):
        gate = RequestAdmission(resources("request_admission")); gate.acquire("s#0", "running")
        result = []; thread = threading.Thread(target=lambda: result.append(gate.acquire("g#0", "waiting")))
        thread.start()
        with gate.condition:
            self.assertTrue(gate.condition.wait_for(lambda: len(gate.waiting) == 1, timeout=2))
        gate.close(); thread.join(2)
        self.assertFalse(thread.is_alive()); self.assertEqual(result, [False])
        self.assertEqual(gate.snapshot()["active"], {"running": "s#0"})
        gate.release_alias("s#0"); self.assertEqual(gate.snapshot()["active"], {})

    def test_overlapping_groups_and_per_alias_caps_are_both_enforced(self):
        document = resources("request_admission", ("s#0", "s#1", "g#0"))
        document["groups"] = {"all": {"aliases": ["s#0", "s#1", "g#0"], "max_active": 2},
                              "specialists": {"aliases": ["s#0", "s#1"], "max_active": 1}}
        gate = RequestAdmission(document); gate.acquire("s#0", "s")
        self.assertFalse(gate._available("s#0")); self.assertFalse(gate._available("s#1"))
        self.assertTrue(gate._available("g#0")); gate.acquire("g#0", "g")
        self.assertFalse(gate._available("g#0")); gate.release("s"); gate.release("g")

    def test_duplicate_admission_and_unknown_alias_are_rejected(self):
        gate = RequestAdmission(resources("request_admission")); gate.acquire("s#0", "same")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            gate.acquire("g#0", "same")
        with self.assertRaisesRegex(ValueError, "Known alias"):
            gate.acquire("other", "new")
        gate.release("same"); gate.release("same")

    def test_invalid_limits_missing_background_and_duplicate_members_rejected(self):
        original = resources("request_admission")
        for mutate in (lambda d: d["allocations"]["s#0"].update(max_inflight=0),
                       lambda d: d["allocations"]["s#0"].update(max_inflight=True),
                       lambda d: d["groups"]["all"].update(max_active=0),
                       lambda d: d["groups"]["all"].update(aliases=["s#0"]),
                       lambda d: d["groups"]["all"].update(aliases=["s#0", "s#0"])):
            document = copy.deepcopy(original); mutate(document)
            with self.assertRaises(ValueError):
                validate_resources(document, original["allocations"])

    def test_registry_requires_one_agreed_configuration(self):
        registry = {a: {"command": [sys.executable, "-c", "pass"]} for a in ("s#0", "g#0")}
        bound = prepare_registry(registry, resources("request_admission"))
        self.assertIsInstance(admission_for_registry(bound), RequestAdmission)
        bound["g#0"]["resource_binding"]["configuration_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            admission_for_registry(bound)

    def test_cpu_workers_really_serialize_and_wait_counts_from_original_arrival(self):
        names = ("s#0", "g#0")
        registry = {a: {"command": [sys.executable, "-m", "gsdeploy.backends.mock"],
                       "cwd": str(Path(__file__).resolve().parents[1]), "env": {"GS_TASKS": json.dumps(["text"])}}
                    for a in names}
        registry = prepare_registry(registry, resources("request_admission"))
        config = MeasurementConfig(deployment={a: 1 for a in names}, workload_rps={"text": 1},
                                   routing_rps={"text": {a: .5 for a in names}},
                                   shutdown_timeout_seconds=5)
        runtime = DeploymentRuntime(config, registry); rows = []
        try:
            runtime.start(); arrived = time.perf_counter_ns()
            for i in range(6):
                row = record("measurement", i, "text", names[i % 2],
                             {"pool_id": str(i), "sample": {"id": str(i)}}, arrived)
                rows.append(row)
                runtime.enqueue(row, {"id": str(i), "mock_delay_seconds": .02})
            runtime.drain(rows, 5)
            self.assertTrue(all(r["status"] == "completed" for r in rows), rows)
            ordered = sorted(rows, key=lambda r: r["actual_submit_time_ns"])
            self.assertTrue(all(b["actual_submit_time_ns"] >= a["completion_time_ns"]
                                for a, b in zip(ordered, ordered[1:])))
            self.assertTrue(all(r["scheduled_time_ns"] == arrived for r in rows))
            self.assertGreater(ordered[-1]["actual_submit_time_ns"] - arrived, 50_000_000)
            self.assertEqual(runtime.admission.snapshot()["peak_active"], 1)
            self.assertTrue(all(r["enqueued_time_ns"] <= r["admission_wait_start_time_ns"]
                                <= r["admission_granted_time_ns"] <= r["actual_submit_time_ns"] for r in rows))
        finally:
            shutdown = runtime.close()
        self.assertTrue(all(r["bye_received"] and not r["forced"] and r["exit_code"] == 0 for r in shutdown.values()))


if __name__ == "__main__":
    unittest.main()
