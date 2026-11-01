"""Cache isolation, immutability and exact frozen-predictor equivalence."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from gsdeploy.quality import quality_digest
from gsdeploy.search_session import DigestMemo, SearchSession, freeze
from tests.test_calibrated_replay_v2 import fitted_replay_fixture
from tests.test_calibrated_search_bounds_v6 import context, optimizer, prediction, search_fixture
from tests.test_milp import grid_plans

ROOT = Path(__file__).resolve().parents[1]


def exact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class SearchSessionTests(unittest.TestCase):
    def test_normal_mutators_rejected_and_original_input_detached(self):
        c = search_fixture()
        raw = c.as_dict()
        with SearchSession(ROOT, raw) as session:
            frozen = session.input
            with self.assertRaises(TypeError):
                frozen["catalog"] = {}
            with self.assertRaises(TypeError):
                frozen["catalog"].update({"safe_factor": 2})
            with self.assertRaises(TypeError):
                frozen["catalog"]["models"]["generalist"]["supported_tasks"].append("ocr")
            with self.assertRaises(TypeError):
                frozen["catalog"]["models"]["generalist"]["supported_tasks"][0] = "ocr"
            with self.assertRaises(TypeError):
                frozen.__init__({})
            raw["catalog"]["safe_factor"] = 999
            self.assertNotEqual(session.config.catalog["safe_factor"], 999)
            self.assertEqual(quality_digest(session.input), session.memo(session.input))

    def test_temporary_structures_are_keyed_by_value_not_temporary_identity(self):
        memo = DigestMemo(quality_digest)
        tree = freeze({"x": [1, 2], "y": {"a": 3}})
        for index in range(200):
            temporary = {"x": tree["x"], "s": index, "y": {"nested": tree["y"]}}
            self.assertEqual(memo(temporary), quality_digest(temporary))
            self.assertEqual(memo(dict(temporary)), quality_digest(temporary))
        self.assertEqual(memo.misses, 200)
        self.assertEqual(memo.hits, 200)
        # A mutable payload changing between calls must produce a new digest.
        ordinary = {"a": [1.]}
        before = memo(ordinary)
        ordinary["a"][0] = -0.
        self.assertNotEqual(memo(ordinary), before)
        self.assertNotEqual(memo(-0.), memo(0.))
        other_tree = freeze({"x": [9]})
        self.assertNotEqual(memo(tree["x"]), memo(other_tree["x"]))

    def test_public_module_gates_and_digests_never_patched_and_session_revalidates(self):
        c = fitted_replay_fixture()
        raw = c.as_dict()
        import gsdeploy.quality as public_quality
        import gsdeploy.calibration_v2 as public_v2
        original_digest, original_gate = public_quality.quality_digest, public_v2.validated_replay_spec
        with SearchSession(ROOT, raw) as session:
            prefix = session.prefix
            self.assertIs(public_quality.quality_digest, original_digest)
            self.assertIs(public_v2.validated_replay_spec, original_gate)
            raw["catalog"]["quality"]["paths"]["generalist"]["text"]["value"] = .123
            with self.assertRaises(ValueError):
                SearchSession(ROOT, raw)
            with self.assertRaises(ValueError):
                public_v2.validated_replay_spec(raw["catalog"], raw["workload"])
            self.assertIs(public_quality.quality_digest, original_digest)
        self.assertFalse(any(name == prefix or name.startswith(prefix + ".") for name in sys.modules))

    def test_v1_v2_complete_predictions_and_bounds_equal_bit_for_bit(self):
        fixtures = [search_fixture(), replace(fitted_replay_fixture(quality_mode="per_sample_replay"),
                                             horizon_seconds=20.)]
        for c in fixtures:
            with SearchSession(ROOT, c.as_dict()) as session:
                original_milp = optimizer(__import__("gsdeploy.milp", fromlist=["JointMilpOptimizer"]).JointMilpOptimizer, c)
                cached_milp = session.optimizer("milp")
                original_milp.start = cached_milp.start = time.monotonic()
                for x, rows in list(grid_plans(c, 2))[:40]:
                    self.assertEqual(exact(prediction(c, x, rows)), exact(session.predict(x, rows)))
                    for t, row in rows.items():
                        self.assertEqual(original_milp.floor(t, x, row).hex(),
                                         cached_milp.floor(t, x, row).hex())
                self.assertGreater(session.stats()["digest_hits"], session.stats()["digest_misses"])

    def test_three_algorithms_share_session_math_and_same_optimum(self):
        c = replace(fitted_replay_fixture(), horizon_seconds=20.)
        scores = [p["U"] for x, rows in grid_plans(c, 2)
                  if (p := prediction(c, x, rows))["constraints_eligible"] and p["U"] is not None]
        for algorithm in ("milp", "dfs", "plain_dfs"):
            with SearchSession(ROOT, c.as_dict()) as session:
                result = session.optimizer(algorithm).solve(K=2)
                self.assertTrue(result["optimality_certified"], result["status"])
                self.assertEqual(result["winner"]["score"].hex(), max(scores).hex())
                self.assertGreater(session.memo.hits, 0)

    def test_six_roles_preserve_confirmation_problem_and_restrictions(self):
        from gsdeploy.confirmation import _check_search
        c = replace(search_fixture(), route_grid=2, milp_time_limit_seconds=10.,
                    milp_capacity_witness=False)
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config = base / "config.json"
            config.write_text(json.dumps(c.as_dict()))
            completed = subprocess.run([sys.executable, str(ROOT / "scripts/solve_static_session_study.py"),
                "--source-root", str(ROOT), "--config", str(config), "--output", str(base / "roles")],
                text=True, capture_output=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            bundle = json.loads((base / "roles/search-results.json").read_text())
            roles = _check_search(c, bundle)
            self.assertEqual(len(roles), 6)
            self.assertTrue(bundle["engine"]["source_unchanged"])
            self.assertTrue(bundle["engine"]["wrapper_unchanged"])
            winner = roles["free_search_winner"]["winner"]
            self.assertEqual(roles["strongest_alternative_deployment"]["search_restrictions"]
                             ["exclude_deployments"], [winner["deployment"]])
            self.assertEqual(roles["strongest_alternative_route"]["search_restrictions"]
                             ["fixed_deployment"], winner["deployment"])
            self.assertTrue(roles["best_no_split"]["search_restrictions"]["no_split"])
            for result in roles.values():
                self.assertEqual(result["problem"]["workload"], c.workload)
                self.assertGreater(result["session_cache"]["digest_hits"], 0)

    def test_session_cli_and_fresh_process_benchmark_share_contract(self):
        c = replace(search_fixture(), route_grid=2, milp_time_limit_seconds=10.,
                    milp_capacity_witness=False)
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config = base / "config.json"
            config.write_text(json.dumps(c.as_dict()))
            commands = [
                ["scripts/run_search_session.py", "--output", str(base / "single.json"),
                 "--algorithm", "plain_dfs", "--replicas", "1"],
                ["scripts/compare_calibrated_pool_solvers.py", "--output", str(base / "comparison"),
                 "--session", "--models", "generalist", "specialist", "--repeats", "1"],
            ]
            for command in commands:
                completed = subprocess.run([sys.executable, str(ROOT / command[0]), *command[1:],
                    "--source-root", str(ROOT), "--config", str(config), "--route-grid", "2",
                    "--time-limit", "10"], text=True, capture_output=True, timeout=60)
                self.assertEqual(completed.returncode, 0, completed.stderr)
            single = json.loads((base / "single.json").read_text())
            report = json.loads((base / "comparison/report.json").read_text())
            self.assertTrue(single["mathematical_source_unchanged"])
            self.assertTrue(single["result"]["optimality_certified"])
            self.assertTrue(report["benchmark_complete"], report["per_repeat"])
            self.assertTrue(report["same_contract"])
            self.assertTrue(report["session_wrapper_unchanged"])
            self.assertEqual({r["algorithm"] for r in report["records"]}, {"milp", "dfs", "plain_dfs"})
            self.assertTrue(all(r["session_cache"]["digest_hits"] > 0 for r in report["records"]))
            self.assertEqual(single["result"]["winner"]["score"],
                             report["records"][0]["result"]["winner"]["score"])


if __name__ == "__main__":
    unittest.main()
