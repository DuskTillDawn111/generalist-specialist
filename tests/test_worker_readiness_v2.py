"""Real CPU subprocess response flush/readiness and parser integration."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

from gsdeploy.resources import prepare_registry
from gsdeploy.runtime.deployment import DeploymentRuntime
from gsdeploy.runtime.runner import record
from gsdeploy.runtime.schema import MeasurementConfig
from gsdeploy.forecasting.lifecycle import export_training_observations, fit_lifecycle, LifecycleProvider
from tests.test_resource_control import resources
from tests.test_forecast_lifecycle_v2 import CLOCK, CONTEXT
from tools.predictor_validation_20261027.trace_export import anchors_from_worker_lines


class WorkerReadinessTests(unittest.TestCase):
    def test_protocol_emits_readiness_after_success_and_error_but_not_shutdown(self):
        requests = [dict(request_id="one", pool_id="p", arrival_index=0, task_type="text", sample={}),
                    dict(request_id="two", pool_id="p", arrival_index=1, task_type="text", sample={"mock_error": True}),
                    {"op": "shutdown"}]
        result = subprocess.run([sys.executable, "-m", "gsdeploy.backends.mock"],
            input="\n".join(json.dumps(r) for r in requests) + "\n", text=True, capture_output=True,
            env={**os.environ, "GS_TASKS": '["text"]'}, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        messages = [(kind, json.loads(payload)) for kind, payload in
                    (line.split("|", 1) for line in result.stdout.splitlines())]
        self.assertEqual([kind for kind, _ in messages],
                         ["READY", "RESULT", "WORKER_READY", "ERROR", "WORKER_READY", "BYE"])
        self.assertLessEqual(messages[1][1]["end_perf_ns"], messages[2][1]["perf_ns"])
        self.assertEqual(messages[2][1]["request_id"], "one")
        self.assertEqual(messages[4][1]["request_id"], "two")
        self.assertEqual(messages[2][1]["pid"], messages[0][1]["pid"])
        self.assertEqual(messages[2][1]["clock"], "host_perf_counter_ns")
        # Readiness is a host lifecycle boundary, never a phase/kernel anchor.
        self.assertEqual(anchors_from_worker_lines(result.stdout.splitlines()), [])

    def test_actual_collector_records_ready_and_training_export_can_fit_it(self):
        aliases = {a: {"family": "specialist" if a == "s#0" else "generalist"} for a in ("s#0", "g#0")}
        registry = prepare_registry({a: {"command": [sys.executable, "-m", "gsdeploy.backends.mock"],
            "cwd": str(Path(__file__).resolve().parents[1]), "env": {"GS_TASKS": '["text"]'}} for a in aliases},
            resources("request_admission"))
        config = MeasurementConfig(deployment={a: 1 for a in aliases}, workload_rps={"text": 1},
            routing_rps={"text": {a: .5 for a in aliases}}, shutdown_timeout_seconds=5)
        runtime, rows = DeploymentRuntime(config, registry), []
        try:
            runtime.start()
            for index in range(2):
                row = record("measurement", index, "text", "s#0", {"pool_id": "p", "sample": {}}, time.perf_counter_ns())
                row["split"] = "training"
                rows.append(row)
                runtime.enqueue(row, {"mock_delay_seconds": .001})
            runtime.drain(rows, 5)
        finally:
            runtime.close()
        self.assertEqual(runtime.errors, [])
        self.assertTrue(all(row["status"] == "completed" for row in rows))
        self.assertTrue(all("previous_handler_next_request_ready" in row for row in rows))
        raw = {"kind": "cpu_fixture", "split": "training", "execution_context_sha256": CONTEXT,
               "admission_enabled": True, "request_records": rows,
               "admission_snapshot_after": runtime.admission.snapshot()}
        observations = export_training_observations(raw, aliases, execution_context_sha256=CONTEXT,
            clock_contract=CLOCK, source="actual-cpu-subprocess")
        provider = LifecycleProvider(fit_lifecycle(observations, CONTEXT), CONTEXT, allow_cpu_fixture=True)
        delays = provider.delays({"alias": "s#0", "task": "text", "pool_id": "p"}, aliases, "mean")
        self.assertTrue(delays["known"])
        self.assertGreaterEqual(delays["permit_release_ms"], delays["parent_receipt_ms"])
        self.assertGreaterEqual(delays["worker_next_ready_ms"], 0.)
        self.assertFalse(provider.physical_evidence)


if __name__ == "__main__":
    unittest.main()
