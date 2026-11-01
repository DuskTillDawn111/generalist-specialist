"""Static-study priming covers resident task paths without shifting replay."""
import json
from pathlib import Path
import tempfile
import unittest

from gsdeploy import DeploymentSystem, SystemConfig
from unittest.mock import patch
from gsdeploy.runtime.runner import run
from gsdeploy.runtime.workload import SamplePool, largest_inputs


ROOT = Path(__file__).resolve().parents[1]


class StaticPrimingTests(unittest.TestCase):
    def test_memory_cost_uses_sampled_peak_between_endpoints(self):
        class Monitor:
            def __init__(self, *args, **kwargs):
                self.failed = False
                self.samples = [{"owned_used_mib": 512.0, "foreign_pids": []}]

            def sample(self):
                return {"owned_used_mib": 10.0, "foreign_pids": []}

            def start(self):
                pass

            def check(self):
                pass

            def close(self):
                pass

        system = DeploymentSystem(SystemConfig.load(ROOT / "configs/demo.json"))
        with patch("gsdeploy.runtime.runner.MemoryMonitor", Monitor):
            raw = system.run(system.plan(), ROOT / "examples/demo_samples.jsonl",
                             duration=.05, warmup=0, startup_timeout=20)
        self.assertEqual(raw["status"], "completed", raw["errors"])
        self.assertEqual(raw["memory_evidence"]["safe_memory_mib"], 512.0)

    def test_restricted_certificate_keeps_its_search_domain(self):
        system = DeploymentSystem(SystemConfig.load(ROOT / "configs/demo.json"))
        plan = system.plan(deployment={"generalist": 1})
        self.assertTrue(plan["optimality_certified"])
        with patch("gsdeploy.runtime.runner.run", return_value={}):
            raw = system.run(plan, ROOT / "examples/demo_samples.jsonl")
        contract = raw["planning_contract"]
        self.assertTrue(contract["optimality_transferred"])
        self.assertFalse(contract["unrestricted_optimality_transferred"])
        self.assertFalse(contract["physical_optimality_certified"])
        self.assertEqual(contract["search_restrictions"]["fixed_deployment"], {"generalist": 1})

    def test_shared_worker_warms_unrouted_tasks_and_preserves_replay(self):
        config = SystemConfig.load(ROOT / "configs/demo.json")
        # All text is sent to the specialist, but the resident generalist must
        # still prime its supported text path before the measured arrivals.
        route = {task: {"text_specialist" if task == "text" else "generalist": rate}
                 for task, rate in config.workload.items()}
        plan = {"deployment": {"generalist": 1, "text_specialist": 1}, "routing": route}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            media = directory / "media.bin"
            media.write_bytes(b"x" * 100)
            items = []
            for task in config.workload:
                for index in range(4):
                    items.append({"pool_id": f"{task}-{index}", "task_type": task,
                                  "sample": {"task": task, "input": str(media),
                                             "context": "x" * index, "question": "?"}})
            samples = directory / "inputs.jsonl"
            samples.write_text("\n".join(json.dumps(i) for i in items) + "\n")
            replay = SamplePool(samples, list(config.workload), 42)
            self.assertEqual([i["pool_id"] for i in largest_inputs(replay, "text")],
                             ["text-3", "text-2", "text-1"])
            self.assertTrue(all(n == 0 for n in replay.indices.values()))
            raw = run(config, plan, samples, duration=.1, warmup=0,
                      priming_policy="largest_three_per_task", startup_timeout=20)
            self.assertEqual(raw["status"], "completed", raw["errors"])
            for worker in raw["replica_to_family"]:
                primed = [r for r in raw["priming_request_records"]
                          if r["model"] == worker and r["task"] == "text"]
                self.assertEqual([r["pool_id"] for r in primed], ["text-3", "text-2", "text-1"])
            for record in raw["request_records"]:
                self.assertEqual(record["pool_id"], replay.next(record["task"])["pool_id"])


if __name__ == "__main__":
    unittest.main()
