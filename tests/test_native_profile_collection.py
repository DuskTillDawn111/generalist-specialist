"""CPU-only native collection contract checks; no CUDA import or GPU probe."""

import ctypes as C
import os
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gsdeploy.runtime.deployment import DeploymentRuntime
from gsdeploy.runtime.runner import record
from gsdeploy.runtime.schema import MeasurementConfig
from tools.predictor_validation_20261027.native_collect import BoardMonitor, ProcessInfo, descendant_or_self
from tools.predictor_validation_20261027.observed_worker import Observer


class FakeNVML:
    def __init__(self, processes=(), power_mw=123000, error=0):
        self.processes, self.power_mw, self.error = list(processes), power_mw, error
        self.shutdown = False

    def nvmlInit_v2(self):
        return 0

    def nvmlDeviceGetHandleByIndex_v2(self, index, out):
        out._obj.value = 1
        return 0

    def nvmlDeviceGetPowerUsage(self, handle, out):
        out._obj.value = self.power_mw
        return self.error

    def nvmlDeviceGetComputeRunningProcesses_v2(self, handle, count, output):
        count._obj.value = len(self.processes)
        for i, (pid, used_mib) in enumerate(self.processes):
            output[i].pid = pid
            output[i].used = round(used_mib * 2 ** 20)
        return 0

    def nvmlShutdown(self):
        self.shutdown = True
        return 0


class BoardMonitorTests(unittest.TestCase):
    def runtime(self, pids):
        return SimpleNamespace(workers={name: SimpleNamespace(process=SimpleNamespace(pid=pid)) for name, pid in pids.items()},
                               ready={name: {"pid": pid} for name, pid in pids.items()})

    def test_nvml_process_info_v2_abi_matches_64bit_header(self):
        # nvmlProcessInfo_v2_t: unsigned int, unsigned long long, uint, uint.
        self.assertEqual(C.sizeof(ProcessInfo), 24)
        self.assertEqual(ProcessInfo.pid.offset, 0)
        self.assertEqual(ProcessInfo.used.offset, 8)
        self.assertEqual(ProcessInfo.gpu_instance.offset, 16)
        self.assertEqual(ProcessInfo.compute_instance.offset, 20)

    def test_owned_memory_is_sum_over_all_residents_not_board_memory(self):
        runtime = self.runtime({"text#0": os.getpid(), "ocr#0": os.getppid()})
        fake = FakeNVML([(os.getpid(), 100), (os.getppid(), 200)])
        with patch("tools.predictor_validation_20261027.native_collect.C.CDLL", return_value=fake):
            monitor = BoardMonitor(runtime)
            row = monitor.sample()
        self.assertEqual(row["owned_used_mib"], 300)
        self.assertEqual(row["owned_aliases"], ["ocr#0", "text#0"])
        self.assertEqual(row["watts"], 123)
        self.assertEqual(row["foreign_pids"], [])
        self.assertGreaterEqual(row["query_duration_ns"], 0)
        monitor.check()

    def test_foreign_process_cannot_be_hidden_in_owned_peak(self):
        runtime = self.runtime({"text#0": os.getpid()})
        fake = FakeNVML([(os.getpid(), 100), (2147483647, 200)])
        with patch("tools.predictor_validation_20261027.native_collect.C.CDLL", return_value=fake):
            monitor = BoardMonitor(runtime)
            row = monitor.sample()
        self.assertEqual(row["owned_used_mib"], 100)
        self.assertEqual(row["foreign_pids"], [2147483647])
        with self.assertRaisesRegex(RuntimeError, "Foreign"):
            monitor.check()

    def test_nvml_failure_stays_error_not_zero_power(self):
        runtime = self.runtime({"text#0": os.getpid()})
        with patch("tools.predictor_validation_20261027.native_collect.C.CDLL", return_value=FakeNVML(error=3)):
            monitor = BoardMonitor(runtime)
            self.assertIsNone(monitor.sample())
        self.assertEqual(monitor.samples, [])
        with self.assertRaisesRegex(RuntimeError, "Telemetry failed"):
            monitor.check()

    def test_trace_wrapper_pid_does_not_replace_ready_gpu_owner(self):
        # Fake nsys wrapper launcher PID differs from READY actual worker PID.
        runtime = self.runtime({"text#0": 2147483646})
        runtime.ready["text#0"]["pid"] = os.getpid()
        fake = FakeNVML([(os.getpid(), 300)])
        with patch("tools.predictor_validation_20261027.native_collect.C.CDLL", return_value=fake):
            monitor = BoardMonitor(runtime)
            row = monitor.sample()
        self.assertEqual(row["owned_aliases"], ["text#0"])
        self.assertEqual(row["owned_used_mib"], 300)
        self.assertEqual(row["foreign_pids"], [])

    def test_owned_identity_requires_live_self_or_descendant(self):
        self.assertTrue(descendant_or_self(os.getpid(), os.getpid()))
        self.assertFalse(descendant_or_self(2147483647, os.getpid()))
        self.assertFalse(descendant_or_self(os.getppid(), os.getpid()))
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            self.assertTrue(descendant_or_self(child.pid, os.getpid()))
            self.assertFalse(descendant_or_self(os.getpid(), child.pid))
        finally:
            child.terminate()
            child.wait(timeout=3)


class ObservedPhaseTests(unittest.TestCase):
    def torch(self):
        events = []
        nvtx = SimpleNamespace(range_push=lambda name: events.append(("push", name)),
                               range_pop=lambda: events.append(("pop",)))
        return SimpleNamespace(cuda=SimpleNamespace(nvtx=nvtx)), events

    def test_observer_emits_paired_host_timestamps_and_preserves_value(self):
        torch, ranges = self.torch()
        observer = Observer(torch, "r1")
        with patch("tools.predictor_validation_20261027.observed_worker.emit") as emitted:
            self.assertEqual(observer.call("inference", "gpu", lambda: 42), 42)
        self.assertEqual([c.args[1]["event"] for c in emitted.call_args_list], ["start", "end"])
        self.assertLessEqual(observer.stages[0]["start_ns"], observer.stages[0]["end_ns"])
        self.assertEqual([e[0] for e in ranges], ["push", "pop"])
        self.assertIn("GPU phase includes host", observer.stages[0]["scope"])

    def test_observer_exception_still_closes_range_and_phase(self):
        torch, ranges = self.torch()
        observer = Observer(torch, "r1")
        def fail():
            raise RuntimeError("native inference failed")
        with patch("tools.predictor_validation_20261027.observed_worker.emit") as emitted:
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                observer.call("inference", "gpu", fail)
        self.assertEqual(len(emitted.call_args_list), 2)
        self.assertEqual(len(observer.stages), 1)
        self.assertEqual(ranges[-1], ("pop",))

    def test_real_cpu_worker_start_and_phase_are_not_results(self):
        script = '''
import json, os, sys, time
def emit(kind, obj):
    print(kind + "|" + json.dumps(obj), flush=True)
emit("READY", {"tasks":["text"],"pid":os.getpid()})
for line in sys.stdin:
    r=json.loads(line)
    if r.get("op")=="shutdown":
        emit("BYE", {"pid":os.getpid()}); break
    rid=r["request_id"]
    emit("START", {"request_id":rid,"perf_ns":time.perf_counter_ns(),"pid":os.getpid()})
    emit("PHASE", {"request_id":rid,"event":"start","segment_id":"native","kind":"gpu","perf_ns":time.perf_counter_ns()})
    time.sleep(.005)
    emit("PHASE", {"request_id":rid,"event":"end","segment_id":"native","kind":"gpu","perf_ns":time.perf_counter_ns()})
    emit("RESULT", {"request_id":rid,"task_type":"text","success":True,"worker_latency_ms":5})
'''
        registry = {"text#0": {"command": [sys.executable, "-u", "-c", script]}}
        config = MeasurementConfig(deployment={"text#0": 1}, workload_rps={"text": 1},
                                   routing_rps={"text": {"text#0": 1}}, shutdown_timeout_seconds=3)
        runtime = DeploymentRuntime(config, registry)
        try:
            runtime.start()
            row = record("profile", 0, "text", "text#0", {"pool_id": "x", "sample": {}}, time.perf_counter_ns())
            runtime.enqueue(row, {})
            runtime.drain([row], 3)
            self.assertEqual(row["status"], "completed")
            events = row["execution_events"]
            self.assertEqual(len(events), 3)
            self.assertIn("observed_worker_start_time_ns", row)
            self.assertEqual([e.get("event") for e in events], [None, "start", "end"])
            self.assertTrue(all(e["perf_ns"] <= e["received_time_ns"] for e in events))
            self.assertEqual(runtime.errors, [])
        finally:
            runtime.close()


if __name__ == "__main__":
    unittest.main()
