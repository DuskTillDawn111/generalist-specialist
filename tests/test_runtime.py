"""Real subprocess/pipe tests without models or GPUs."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from gsdeploy import DeploymentSystem, SystemConfig
from gsdeploy.evaluation import evaluate_run

ROOT = Path(__file__).resolve().parents[1]


class RuntimeTests(unittest.TestCase):
    def test_full_cohort_shutdown_and_offline_evaluation(self):
        system = DeploymentSystem(SystemConfig.load(ROOT / "configs/demo.json"))
        raw = system.run(
            system.plan(),
            ROOT / "examples/demo_samples.jsonl",
            duration=0.15,
            warmup=0,
            startup_timeout=20,
        )
        self.assertEqual(raw["status"], "completed", raw["errors"])
        self.assertEqual(raw["system_metrics"]["completion_ratio"], 1.0)
        self.assertEqual(
            {r["task"] for r in raw["request_records"]}, set(system.config.workload)
        )
        self.assertTrue(
            all(
                v["exit_code"] == 0 and not v["forced"] and v["bye_received"]
                for v in raw["shutdown"].values()
            )
        )
        self.assertTrue(
            all(
                r["scheduled_time_ns"]
                <= r["actual_submit_time_ns"]
                <= r["completion_time_ns"]
                for r in raw["request_records"]
            )
        )
        self.assertEqual(raw["evaluation"], evaluate_run(system.config, raw))
        self.assertIsNone(raw["evaluation"]["utility"]["U"])
        self.assertFalse(raw["evaluation"]["utility"]["eligible"])

    def test_multiple_replicas_are_distinct_processes(self):
        data = SystemConfig.load(ROOT / "configs/demo.json").as_dict()
        data["catalog"]["capacity_factors"]["generalist"]["2"] = 1.7
        data["catalog"]["memory_table_mib"]["generalist"]["2"] = 96.0
        system = DeploymentSystem(SystemConfig(**data))
        plan = system.plan(deployment={"generalist": 2})
        raw = system.run(
            plan,
            ROOT / "examples/demo_samples.jsonl",
            duration=0.12,
            warmup=0,
            startup_timeout=20,
        )
        self.assertEqual(raw["status"], "completed", raw["errors"])
        self.assertEqual(
            set(raw["replica_to_family"]), {"generalist#0", "generalist#1"}
        )
        self.assertEqual(set(raw["shutdown"]), set(raw["replica_to_family"]))
        self.assertEqual(raw["system_metrics"]["completion_ratio"], 1.0)

    def test_worker_failure_retains_evidence_and_closes(self):
        system = DeploymentSystem(SystemConfig.load(ROOT / "configs/demo.json"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            items = [
                json.loads(line)
                for line in (ROOT / "examples/demo_samples.jsonl")
                .read_text()
                .splitlines()
            ]
            for item in items:
                item["sample"]["mock_error"] = True
            path.write_text("\n".join(json.dumps(item) for item in items) + "\n")
            raw = system.run(
                system.plan(),
                path,
                duration=0.1,
                warmup=0,
                startup_timeout=20,
                drain_timeout=1,
            )
        self.assertEqual(raw["status"], "failed")
        self.assertTrue(raw["errors"])
        self.assertTrue(
            any(r["status"] == "error" for r in raw["priming_request_records"])
        )
        self.assertTrue(
            all(
                v["exit_code"] == 0 and v["bye_received"]
                for v in raw["shutdown"].values()
            )
        )
        self.assertIsNone(raw["measurement_window"])


if __name__ == "__main__":
    unittest.main()
