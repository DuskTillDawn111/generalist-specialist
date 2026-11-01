"""Residual feedback preserves the finite domain and exact complete scores."""

import itertools
import unittest
from dataclasses import replace
from unittest.mock import patch

from gsdeploy import DeploymentSystem
from gsdeploy.milp import residual_identity_cut
from tests.test_milp import add_energy, grid_plans
from tests.test_static_dfs_v6 import six_term_config
from tests.test_profiled_memory_search_v6 import profiled_problem


def legacy_identity_cut(base_row, theta, selected, blocks, residual):
    cost = residual - sum(base_row.get(i, 0.0) for i in selected)
    return {**{i: -cost for i in selected}, theta: 1.0}, -cost * (blocks - 1)


class ResidualMilpCutTests(unittest.TestCase):
    def test_integer_cut_is_exact_at_identity_and_valid_elsewhere(self):
        # Two one-hot blocks, two options per block. Every cost is >= B(z).
        costs = [0.2, 0.5, 0.3, 0.1]
        base = {i: -cost for i, cost in enumerate(costs)}
        selected, blocks, residual = (0, 2), 2, 0.4
        row, lower = residual_identity_cut(base, 4, selected, blocks, residual)
        for a, b in itertools.product([0, 1], [2, 3]):
            x = [float(i in [a, b]) for i in range(4)]
            B = costs[a] + costs[b]
            required_theta = lower - sum(row[i] * x[i] for i in range(4))
            if (a, b) == selected:
                self.assertAlmostEqual(required_theta, B + residual)
            else:
                self.assertLessEqual(required_theta, B + 1e-12)

    def test_fractional_relaxation_dominates_old_identity_cut(self):
        base = {0: -0.2, 1: -0.5, 2: -0.3, 3: -0.1}
        new, new_lower = residual_identity_cut(base, 4, (0, 2), 2, 0.4)
        old, old_lower = legacy_identity_cut(base, 4, (0, 2), 2, 0.4)
        for a, b in itertools.product([0, 0.25, 0.5, 0.75, 1], repeat=2):
            x = [a, 1 - a, b, 1 - b]
            base_bound = -sum(base[i] * x[i] for i in range(4))
            new_bound = max(base_bound, new_lower - sum(new.get(i, 0) * x[i] for i in range(4)))
            old_bound = max(base_bound, old_lower - sum(old.get(i, 0) * x[i] for i in range(4)))
            self.assertGreaterEqual(new_bound + 1e-12, old_bound)

    def test_legacy_residual_dfs_and_cartesian_agree_in_both_memory_modes(self):
        for c in [six_term_config(), profiled_problem()]:
            c = replace(c, solver="milp")
            scores = [DeploymentSystem(c).predict(x, r)["U"] for x, r in grid_plans(c, 2)]
            expected = max(v for v in scores if v is not None)
            optimized = DeploymentSystem(c).plan(route_grid=2)
            with patch("gsdeploy.milp.residual_identity_cut", legacy_identity_cut):
                old = DeploymentSystem(c).plan(route_grid=2)
            dfs = DeploymentSystem(replace(c, solver="dfs")).plan(route_grid=2)
            for result in [optimized, old, dfs]:
                self.assertTrue(result["optimality_certified"], result["status"])
                self.assertAlmostEqual(result["winner"]["score"], expected, places=9)
                self.assertLessEqual(result["optimality_gap"], 1e-8)
            self.assertGreater(optimized["metrics"]["residual_identity_cuts"], 0)
            self.assertLessEqual(optimized["metrics"]["identity_residual_sum"],
                                 optimized["metrics"]["identity_full_cost_sum"] + 1e-9)

    def test_missing_evidence_gap_survives_residual_feedback(self):
        c = six_term_config()
        # Only G-only has energy. Unknown lower-cost identities survive the gap.
        c.catalog["energy_profiles"] = {}
        add_energy(c, [({"g": 1}, {"text": {"g": 1.0}})], lambda *_: 1.0)
        result = DeploymentSystem(replace(c, solver="milp")).plan(route_grid=2)
        self.assertFalse(result["optimality_certified"])
        self.assertGreater(result["metrics"]["unknown_candidates"], 0)
        self.assertGreaterEqual(result["utility_upper_bound"], result["winner"]["score"])
        self.assertTrue(result["unresolved_candidates"])


if __name__ == "__main__":
    unittest.main()
