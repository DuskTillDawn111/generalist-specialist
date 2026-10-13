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
    def test_twelve_frozen_decisions_and_utility(self):
        for reference in json.loads((ROOT / "tests/reference_plans.json").read_text()):
            with self.subTest(scenario=reference["scenario"], pool=reference["pool"]):
                config = case(reference["scenario"], reference["pool"])
                result = DeploymentSystem(config).plan()
                self.assertEqual(result["status"], "OPTIMAL_K1_DP")
                winner = result["winner"]
                self.assertAlmostEqual(
                    winner["prediction"]["U"], reference["U"], places=8
                )
                self.assertEqual(winner["deployment"], reference["deployment"])
                self.assertEqual(winner["routing"], reference["routing"])
                self.assertTrue(result["constraints"]["constraints_valid"])

    def test_joint_plan_matches_independent_small_oracle(self):
        config = SystemConfig.load(ROOT / "configs/demo.json")
        models = list(config.catalog["models"])
        values = []
        for bits in itertools.product((0, 1), repeat=len(models)):
            x = {m: 1 for m, count in zip(models, bits) if count}
            choices = [
                [m for m in x if t in config.catalog["models"][m]["supported_tasks"]]
                for t in config.workload
            ]
            if not x or any(not options for options in choices):
                continue
            for assignment in itertools.product(*choices):
                route = {
                    t: {m: config.workload[t]}
                    for t, m in zip(config.workload, assignment)
                }
                try:
                    config.validate_plan(x, route)
                except ValueError:
                    continue
                p = predict_scenarios(
                    config.catalog,
                    x,
                    route,
                    config.workload,
                    config.gpu_memory_mib,
                    config.latency_reference_ms,
                    config.weights,
                )
                if p["eligible"]:
                    values.append(p["U"])
        self.assertAlmostEqual(
            DeploymentSystem(config).plan()["winner"]["prediction"]["U"],
            max(values),
            places=9,
        )

    def test_fixed_split_grid_matches_independent_oracle(self):
        config = case("ASR32B", "B")
        system = DeploymentSystem(config)
        x = system.plan()["winner"]["deployment"]
        options = []
        for task in config.workload:
            models = [
                m
                for m in sorted(x)
                if task in config.catalog["models"][m]["supported_tasks"]
            ]
            rows = []
            for units in itertools.product(range(3), repeat=len(models)):
                if sum(units) == 2:
                    rows.append(
                        {
                            m: config.workload[task] * u / 2
                            for m, u in zip(models, units)
                            if u
                        }
                    )
            options.append(rows)
        best = None
        for selected in itertools.product(*options):
            routing = dict(zip(config.workload, selected))
            try:
                config.validate_plan(x, routing)
            except ValueError:
                continue
            prediction = system.predict(x, routing)
            if prediction["eligible"]:
                best = prediction["U"] if best is None else max(best, prediction["U"])
        result = system.plan(deployment=x, route_grid=2)
        self.assertEqual(result["status"], "OPTIMAL_FIXED_SPLIT_GRID")
        self.assertAlmostEqual(result["winner"]["prediction"]["U"], best, places=9)

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
        with self.assertRaises(ValueError):
            system.plan(route_grid=2)
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
