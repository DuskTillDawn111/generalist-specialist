"""Measured whole-plan memory must not be pruned by standalone residency sums."""

from dataclasses import replace
import unittest

from gsdeploy import DeploymentSystem
from gsdeploy.performance import profile_key
from tests.test_milp import add_energy, grid_plans
from tests.test_static_dfs_v6 import six_term_config


def profiled_problem():
    c = six_term_config()
    c.catalog["performance_mode"] = "profiled"
    c.catalog["memory_mode"] = "profiled"
    for m in c.catalog["models"]:
        c.catalog["memory_table_mib"][m]["1"] = 120.0
    # Each standalone residency sum exceeds the 100 MiB budget. Complete
    # measurements, not that proxy, determine actual plan eligibility and M.
    for x, routing in grid_plans(c, 2):
        key = profile_key(c.catalog, x, routing, c.workload, c.horizon_seconds,
                          c.random_seed, c.arrival_process)
        c.catalog.setdefault("colocation_profiles", {})[key] = dict(
            kind="synthetic_fixture", scope="whole_gpu_colocation", coverage_valid=True,
            service_time_ms={m: {"text": 2.0} for m in x},
            safe_memory_mib=10.0 if len(x) == 2 else 80.0,
            provenance="CPU-only memory counterexample fixture",
        )
    add_energy(c, list(grid_plans(c, 2)), lambda *_: 1.0)
    return c


class ProfiledMemorySearchTests(unittest.TestCase):
    def test_profiled_lower_memory_plan_not_removed_and_scores_match_oracle(self):
        c = profiled_problem()
        oracle = []
        for x, routing in grid_plans(c, 2):
            p = DeploymentSystem(c).predict(x, routing)
            if p["U"] is not None:
                oracle.append(p["U"])
        self.assertTrue(oracle)
        results = [DeploymentSystem(replace(c, solver=solver)).plan(route_grid=2)
                   for solver in ["milp", "dfs"]]
        for result in results:
            self.assertTrue(result["optimality_certified"], result["status"])
            self.assertEqual(result["winner"]["deployment"], {"g": 1, "s": 1})
            self.assertAlmostEqual(result["winner"]["score"], max(oracle), places=9)
            self.assertAlmostEqual(result["winner"]["prediction"]["M_norm"], 0.1)
            self.assertIn("zero", result["domain"]["memory_bound"])

    def test_missing_plan_memory_is_unknown_never_infeasible_or_optimal(self):
        c = profiled_problem()
        # Keep one fully evidenced candidate but leave the better two-model
        # identities unknown. Their safe memory cannot be replaced by a sum.
        for x, routing in grid_plans(c, 2):
            if len(x) != 2:
                continue
            key = profile_key(c.catalog, x, routing, c.workload, c.horizon_seconds,
                              c.random_seed, c.arrival_process)
            c.catalog["colocation_profiles"][key].pop("safe_memory_mib")
            p = DeploymentSystem(c).predict(x, routing)
            self.assertTrue(p["memory_unknown"])
            self.assertFalse(p["performance_unknown"] if "performance_unknown" in p else False)
            self.assertIsNone(p["U"])
            self.assertFalse(c.validate_plan(x, routing)["constraints_valid"])
        # Rebind all energy records to the now-frozen memory-profile version.
        add_energy(c, list(grid_plans(c, 2)), lambda *_: 1.0)
        for solver in ["milp", "dfs"]:
            result = DeploymentSystem(replace(c, solver=solver)).plan(route_grid=2)
            self.assertFalse(result["optimality_certified"])
            self.assertIsNotNone(result["winner"])
            self.assertGreater(result["metrics"]["unknown_candidates"], 0)
            self.assertGreaterEqual(result["utility_upper_bound"], result["winner"]["score"])
            self.assertTrue(result["unresolved_candidates"])


if __name__ == "__main__":
    unittest.main()
