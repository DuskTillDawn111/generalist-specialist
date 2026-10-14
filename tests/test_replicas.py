"""Static workload feasibility with the existing empirical replica domain."""

from pathlib import Path
import unittest

from gsdeploy import DeploymentSystem, SystemConfig

ROOT = Path(__file__).resolve().parents[1]


class ReplicaDomainTests(unittest.TestCase):
    def test_unchanged_single_copy_data_and_bounded_domain(self):
        old = SystemConfig.load(ROOT / "configs/low18.json")
        new = SystemConfig.load(ROOT / "configs/static_replicas.json")
        for model in old.catalog["models"]:
            self.assertEqual(
                old.catalog["models"][model]["supported_tasks"],
                new.catalog["models"][model]["supported_tasks"],
            )
            self.assertEqual(
                old.catalog["models"][model]["service_time_ms"],
                new.catalog["models"][model]["service_time_ms"],
            )
            for table in ["capacity_factors", "memory_table_mib"]:
                self.assertEqual(
                    old.catalog[table][model]["1"], new.catalog[table][model]["1"]
                )
        with self.assertRaisesRegex(ValueError, "Invalid fixed deployment"):
            DeploymentSystem(new).plan(deployment={"roberta_sst2": 3})
        self.assertIn("ocr", new.catalog["models"]["qwen"]["capabilities_verified"])
        self.assertNotIn("ocr", new.catalog["models"]["qwen"]["supported_tasks"])

    def test_high_load_requires_available_replicas(self):
        old = SystemConfig.load(ROOT / "configs/low18.json").as_dict()
        new = SystemConfig.load(ROOT / "configs/static_replicas.json").as_dict()
        for data in [old, new]:
            data["workload"] = {
                task: rate * 30 for task, rate in data["workload"].items()
            }
        self.assertIsNone(DeploymentSystem(SystemConfig(**old)).plan()["winner"])
        config = SystemConfig(**new)
        winner = DeploymentSystem(config).plan()["winner"]
        self.assertIsNotNone(winner)
        self.assertGreater(winner["deployment"]["phi4mm"], 1)
        self.assertTrue(
            config.validate_plan(winner["deployment"], winner["routing"])[
                "constraints_valid"
            ]
        )


if __name__ == "__main__":
    unittest.main()
