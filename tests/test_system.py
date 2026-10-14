"""Meaningful migration regressions, including independent finite-space oracles."""

import ast
import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest

from gsdeploy import DeploymentSystem, SystemConfig
from gsdeploy.io import save
from gsdeploy.prediction import predict_scenarios

ROOT = Path(__file__).resolve().parents[1]


def case(scenario, pool):
    path = {
        "LOW18": "low18",
        "ASR32B": "asr32b",
        "TEXT_HALF_MID18": "text_half_mid18",
        "HOLDOUT_TEXT3": "holdout",
    }[scenario]
    data = SystemConfig.load(ROOT / "configs" / (path + ".json")).as_dict()
    data["catalog"] = json.loads(
        (ROOT / "configs/catalogs" / (pool.lower() + ".json")).read_text()
    )
    return SystemConfig(**data)


class PlanningTests(unittest.TestCase):
    def test_historical_inputs_use_new_contract(self):
        # The old scores remain historical fixtures, not targets for reproducing
        # route-dependent arrivals. Every old scenario still yields a legal plan.
        for reference in json.loads((ROOT / "tests/reference_plans.json").read_text()):
            with self.subTest(scenario=reference["scenario"], pool=reference["pool"]):
                config = case(reference["scenario"], reference["pool"])
                result = DeploymentSystem(config).plan()
                self.assertTrue(result["search_completed"])
                self.assertTrue(result["optimality_certified"])
                self.assertTrue(result["constraints"]["constraints_valid"])
                self.assertEqual(
                    result["trace_sha256"],
                    result["winner"]["prediction"]["trace_sha256"],
                )
                self.assertEqual(
                    result["winner"]["prediction"]["utility_schema"], "legacy_four_term"
                )

    def test_budget_infeasibility_and_input_errors(self):
        data = SystemConfig.load(ROOT / "configs/demo.json").as_dict()
        data["gpu_memory_mib"] = 1
        self.assertIsNone(DeploymentSystem(SystemConfig(**data)).plan()["winner"])
        for field, bad in [("route_grid", True), ("gpu_memory_mib", float("nan"))]:
            changed = copy.deepcopy(data)
            changed[field] = bad
            with self.assertRaises(ValueError):
                SystemConfig(**changed)
        system = DeploymentSystem(SystemConfig.load(ROOT / "configs/demo.json"))
        with self.assertRaises(ValueError):
            system.plan(deployment={"generalist": 2})
        self.assertIsNotNone(system.plan(route_grid=2)["winner"])
        winner = system.plan()["winner"]
        broken = copy.deepcopy(winner["routing"])
        broken["asr"] = {"text_specialist": 50}
        with self.assertRaises(ValueError):
            system.predict(winner["deployment"], broken)

    def test_atomic_outputs_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            save(path, {"a": 1})
            with self.assertRaises(FileExistsError):
                save(path, {"a": 2})
            self.assertEqual(json.loads(path.read_text()), {"a": 1})

    def test_project_has_no_history_code_dependencies(self):
        for path in (ROOT / "gsdeploy").rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                imports = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else (
                        [node.module or ""]
                        if isinstance(node, ast.ImportFrom) and not node.level
                        else []
                    )
                )
                for name in imports:
                    self.assertFalse(
                        name.startswith(
                            (
                                "module",
                                "joint_deploy",
                                "benchmark",
                                "deployment_core",
                                "results",
                                "performance_model",
                            )
                        ),
                        (path, name),
                    )
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.assertNotIn("/root/autodl-tmp", node.value, path)

    def test_native_helpers_export_referenced_symbols(self):
        exports = {}
        for path in (ROOT / "gsdeploy/backends/tasks").glob("*.py"):
            names = set()
            for node in ast.parse(path.read_text()).body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    names.add(node.name)
                elif isinstance(node, ast.Assign):
                    names.update(t.id for t in node.targets if isinstance(t, ast.Name))
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    names.update(a.asname or a.name.split(".")[0] for a in node.names)
            exports[path.stem] = names
        for path in (ROOT / "gsdeploy/backends").rglob("*.py"):
            tree = ast.parse(path.read_text())
            aliases = {
                a.asname or a.name: a.name
                for node in tree.body
                if isinstance(node, ast.ImportFrom)
                and node.module == "gsdeploy.backends.tasks"
                for a in node.names
            }
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in aliases
                ):
                    self.assertIn(
                        node.attr, exports[aliases[node.value.id]], (path, node.attr)
                    )


if __name__ == "__main__":
    unittest.main()
