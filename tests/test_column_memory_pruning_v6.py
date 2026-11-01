"""Early task-column pruning uses valid memory lower bounds only."""

from dataclasses import replace
import math
import time
import unittest
from unittest.mock import patch

from gsdeploy.milp import JointMilpOptimizer
from tests.test_calibrated_search_bounds_v6 import optimizer, prediction, search_fixture
from tests.test_calibrated_replay_v2 import fitted_replay_fixture
from tests.test_milp import grid_plans
from tests.test_profiled_memory_search_v6 import profiled_problem


class ColumnMemoryPruningTests(unittest.TestCase):
    def check_pruning_and_oracle(self, c):
        c = replace(c, gpu_memory_mib=2.1, horizon_seconds=20.)
        solver = optimizer(JointMilpOptimizer, c)
        solver.start = time.monotonic()
        original_floor = solver.floor
        evaluated = []

        def floor(task, x, row):
            memory = math.fsum(solver.state_memory(m, n) for m, n in x.items())
            self.assertLessEqual(memory, c.gpu_memory_mib + 1e-6)
            evaluated.append((task, x))
            return original_floor(task, x, row)

        with patch.object(solver, "floor", side_effect=floor):
            columns = solver.columns({m: [0, 1, 2] for m in c.catalog["models"]}, 3)
        self.assertTrue(columns)
        self.assertTrue(evaluated)
        self.assertGreater(solver.metrics["column_memory_pruned"], 0)
        self.assertTrue(any(next(iter(col["deployment"].values())) == 2 for col in columns))
        scores = []
        for x, rows in grid_plans(c, 3):
            p = prediction(c, x, rows)
            if p["constraints_eligible"] and p["U"] is not None:
                scores.append(p["U"])
        self.assertTrue(scores)
        result = optimizer(JointMilpOptimizer, c).solve(K=3)
        self.assertTrue(result["optimality_certified"], result["status"])
        self.assertGreater(result["metrics"]["column_memory_pruned"], 0)
        self.assertAlmostEqual(result["winner"]["score"], max(scores), places=9)

    def test_validated_v1_memory_prunes_before_floor_and_preserves_optimum(self):
        self.check_pruning_and_oracle(search_fixture())

    def test_validated_v2_memory_prunes_before_floor_and_preserves_optimum(self):
        self.check_pruning_and_oracle(fitted_replay_fixture())

    def test_profiled_nonadditive_memory_never_uses_standalone_column_pruning(self):
        c = profiled_problem()
        solver = optimizer(JointMilpOptimizer, c)
        solver.start = time.monotonic()
        columns = solver.columns({m: [0, 1] for m in c.catalog["models"]}, 2)
        self.assertTrue(any(len(col["deployment"]) == 2 for col in columns))
        self.assertEqual(solver.metrics["column_memory_pruned"], 0)
        scores = []
        for x, rows in grid_plans(c, 2):
            p = prediction(c, x, rows)
            if p["constraints_eligible"] and p["U"] is not None:
                scores.append(p["U"])
        result = optimizer(JointMilpOptimizer, c).solve(K=2)
        self.assertTrue(result["optimality_certified"], result["status"])
        self.assertEqual(result["metrics"]["column_memory_pruned"], 0)
        self.assertEqual(result["winner"]["deployment"], {"g": 1, "s": 1})
        self.assertAlmostEqual(result["winner"]["score"], max(scores), places=9)

    def test_pruned_columns_do_not_consume_retained_column_limit(self):
        c = replace(search_fixture(), gpu_memory_mib=2.1, horizon_seconds=20.)
        choices = {m: [0, 1, 2] for m in c.catalog["models"]}
        solver = optimizer(JointMilpOptimizer, c)
        solver.start = time.monotonic()
        columns = solver.columns(choices, 3)
        limit = len(columns)
        exact = optimizer(JointMilpOptimizer, c, milp_max_columns=limit)
        exact.start = time.monotonic()
        self.assertEqual(len(exact.columns(choices, 3)), limit)
        self.assertGreater(exact.metrics["column_memory_pruned"], 0)


if __name__ == "__main__":
    unittest.main()
