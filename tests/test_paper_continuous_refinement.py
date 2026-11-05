"""The original Joint seed and continuous refinement share the stage oracle."""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from gsdeploy.config import SystemConfig
from gsdeploy.continuous import (Budget, FixedDeploymentRoutingEvaluator,
    ImmutableContinuousProblem, JointMILPContinuousRefinement, SessionPredictorOracle)

ROOT = Path(__file__).resolve().parents[1]


def config():
    return SystemConfig.load(ROOT / "configs/paper_fifo_cpu_fixture.json")


class PaperContinuousRefinementTests(unittest.TestCase):
    def test_both_searches_use_new_engine_and_preserve_joint_seed(self):
        for method in ("de", "local"):
            with self.subTest(method=method):
                r = JointMILPContinuousRefinement(config(), seconds=5,
                    max_predictions=32, max_routes=16, method=method).solve()
                self.assertIsNotNone(r["initial_winner"])
                self.assertGreaterEqual(r["winner"]["utility"], r["initial_winner"]["utility"])
                self.assertTrue(r["deployment_preserved"])
                self.assertTrue(all(h["milp_calls"] == 0 for h in r["histories"]))
                self.assertEqual(r["budget"]["milp_solve_calls"], r["initial_counts"]["milp"])
                prediction = r["winner"]["prediction"]
                self.assertEqual(prediction["prediction_scope"], "paper_profiled_fifo_v1")
                self.assertFalse(prediction["native_eligible"])
                self.assertFalse(prediction["independent_accuracy_qualified"])
                self.assertFalse(prediction["final_gain_eligible"])
                self.assertFalse(r["continuous_optimality"])

    def test_off_grid_and_endpoints_keep_residency_and_use_no_deployment_solver(self):
        p = ImmutableContinuousProblem(config())
        with SessionPredictorOracle(p) as oracle:
            e = FixedDeploymentRoutingEvaluator(p, oracle, {"g": 1, "s": 1}, Budget(5, 16, 16))
            rows = [e.evaluate({"text": {"g": v, "s": 1-v}})
                    for v in (0., .3141592653589793, 1.)]
            for row in rows:
                self.assertEqual(row["deployment"], {"g": 1, "s": 1})
                self.assertEqual(row["milp_solve_calls"], 0)
                self.assertTrue(row["feasible"])
                self.assertEqual(row["prediction"]["traces"][0]["prediction"]["safe_memory_mib"], 40.)
            self.assertEqual(rows[1]["routing_shares"]["text"]["g"], .3141592653589793)
            self.assertEqual(e.evaluate({"text": {"g": 1., "s": 0.}})["full_prediction_calls"], 0)
            engine = oracle.session.module("paper_forecast").session(oracle._config.catalog)
            self.assertEqual(engine.engine_calls, 3)
            self.assertEqual(e.budget.counts["milp"], 0)

    def test_unknown_stress_does_not_become_a_robust_seed(self):
        r = JointMILPContinuousRefinement(replace(config(), search_policy="robust"),
            seconds=5, max_routes=8).solve()
        self.assertIsNone(r["winner"])
        self.assertFalse(r["initial"]["optimality_certified"])

    def test_changed_phase_source_invalidates_oracle(self):
        p = ImmutableContinuousProblem(config())
        with SessionPredictorOracle(p) as oracle:
            original = Path.read_text

            def changed(path, *args, **kwargs):
                value = original(path, *args, **kwargs)
                if path == ROOT / "gsdeploy/forecasting/simulator.py":
                    value += "\n# changed implementation\n"
                return value

            with patch.object(Path, "read_text", changed):
                with self.assertRaisesRegex(ValueError, "ORACLE_SOURCE_CHANGED"):
                    oracle.check()


if __name__ == "__main__":
    unittest.main()
