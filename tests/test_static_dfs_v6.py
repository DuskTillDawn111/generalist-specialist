"""DFS grid/proof regression cases; all evidence here is synthetic CPU data."""

import unittest
from dataclasses import replace

from gsdeploy.optimizer import UtilityOptimizer
from gsdeploy.performance import profile_key
from gsdeploy.prediction import predict_scenarios
from gsdeploy import DeploymentSystem
from tests.test_milp import add_energy, grid_plans
from tests.test_search_contract import config


def solve(c, K=2, **options):
    solve_options = {k: options.pop(k) for k in
                     ("exclude_deployments", "exclude_plans", "no_split") if k in options}
    return UtilityOptimizer(
        c.catalog, c.workload, c.gpu_memory_mib, c.latency_reference_ms, c.weights,
        horizon=c.horizon_seconds, seed=c.random_seed,
        arrival_process=c.arrival_process, drain_seconds=c.drain_seconds,
        energy_reference_watts=c.energy_reference_watts,
        startup_amortization_seconds=c.startup_amortization_seconds,
        milp_capacity_witness=False, **options,
    ).solve(K=K, **solve_options)


def add_service(c, x, routing, milliseconds=2.0):
    key = profile_key(c.catalog, x, routing, c.workload, c.horizon_seconds,
                      c.random_seed, c.arrival_process)
    c.catalog.setdefault("colocation_profiles", {})[key] = dict(
        kind="synthetic_fixture", scope="whole_gpu_colocation", coverage_valid=True,
        service_time_ms={m: {t: milliseconds for t in c.catalog["models"][m]["supported_tasks"]}
                         for m in x},
        provenance="CPU fixture, not hardware evidence",
    )


def six_term_config():
    c = config({"text": 1.0}, {"g": {"text": 2.0}, "s": {"text": 4.0}}, horizon=3)
    c.catalog["capability_contract"] = "verified_executable_v1"
    for spec in c.catalog["models"].values():
        spec["executable_tasks"] = spec["supported_tasks"][:]
    c.catalog["quality"] = dict(
        schema_version=1, mode="aggregate_proxy",
        identity={k: "synthetic CPU fixture" for k in (
            "input_pool_sha256", "reference_sha256", "protocol_sha256", "source")},
        paths={m: {"text": dict(metric="em", unit="fraction", value=value,
                                  checkpoint_sha256="synthetic fixture", source="CPU fixture")}
               for m, value in [("g", 0.2), ("s", 0.99)]},
    )
    c = replace(c, weights={k: 1 / 6 for k in "QTLPME"},
                utility_schema="six_term_quality_v1")
    add_energy(c, list(grid_plans(c, 2)), lambda *_: 1.0)
    return c


class StaticDfsContractTests(unittest.TestCase):
    def test_public_replica_restriction_preserves_evidence_and_certificate_scope(self):
        c = six_term_config()
        original = c.as_dict()
        for solver in ("milp", "dfs"):
            result = DeploymentSystem(replace(c, solver=solver)).plan(
                replica_options={"g": [1], "s": []}, route_grid=2)
            self.assertEqual(result["winner"]["deployment"], {"g": 1})
            self.assertEqual(result["search_restrictions"]["replica_options"], {"g": [1], "s": []})
            self.assertTrue(result["optimality_certified"])
        self.assertEqual(original, c.as_dict())

    def test_grid_only_does_not_silently_insert_capacity_witness(self):
        c = config({"text": 1.3}, {"g": {"text": 1000.0}, "s": {"text": 1000.0}}, horizon=4)
        c.catalog["capacity_factors"]["s"]["1"] = 0.7
        result = solve(c)
        self.assertIsNone(result["winner"])
        self.assertFalse(result["domain"]["capacity_boundary_witnesses"])
        self.assertEqual(result["domain"]["routing"], "union of grids with denominators 1..K")

    def test_complete_profile_domain_matches_cartesian_oracle_despite_slow_standalone(self):
        c = config({"text": 1.0}, {"g": {"text": 5000.0}, "s": {"text": 6000.0}}, horizon=3)
        c.catalog["performance_mode"] = "profiled"
        plans = list(grid_plans(c, 2))
        for x, routing in plans:
            add_service(c, x, routing, 2.0 if "g" in x else 4.0)
        scores = []
        for x, routing in plans:
            p = predict_scenarios(c.catalog, x, routing, c.workload, c.gpu_memory_mib,
                                  c.latency_reference_ms, c.weights, horizon=c.horizon_seconds)
            if p["U"] is not None:
                scores.append(p["U"])
        result = solve(c)
        self.assertTrue(result["optimality_certified"], result["status"])
        self.assertAlmostEqual(result["winner"]["score"], max(scores), places=10)
        self.assertGreater(result["metrics"]["full_prediction_calls"], 0)
        self.assertIn("zero", result["domain"]["service_bound"])

    def test_missing_colocation_is_unresolved_with_valid_upper_bound(self):
        c = config({"text": 1.0}, {"g": {"text": 5000.0}, "s": {"text": 6000.0}}, horizon=3)
        c.catalog["performance_mode"] = "profiled"
        result = solve(c)
        self.assertEqual(result["status"], "INCOMPLETE_PERFORMANCE_DOMAIN")
        self.assertTrue(result["search_completed"])
        self.assertFalse(result["optimality_certified"])
        self.assertIsNone(result["winner"])
        self.assertGreater(result["metrics"]["unknown_candidates"], 0)
        self.assertEqual(result["metrics"]["infeasible_candidates"], 0)
        self.assertGreaterEqual(result["utility_upper_bound"], -0.15 / 100)
        self.assertTrue(result["unresolved_candidates"])

    def test_known_incumbent_does_not_delete_unknown_candidates(self):
        c = config({"text": 1.0}, {"g": {"text": 5000.0}, "s": {"text": 6000.0}}, horizon=3)
        c.catalog["performance_mode"] = "profiled"
        add_service(c, {"g": 1}, {"text": {"g": 1.0}}, 2.0)
        result = solve(c)
        self.assertEqual(result["status"], "FEASIBLE_INCOMPLETE_EVIDENCE_DOMAIN")
        self.assertIsNotNone(result["winner"])
        self.assertFalse(result["optimality_certified"])
        self.assertGreaterEqual(result["utility_upper_bound"], result["winner"]["score"])
        self.assertGreater(result["optimality_gap"], 0)

    def test_time_and_state_limits_preserve_unseen_domain_bound(self):
        c = config()
        for options, marker in [({"dfs_time_limit_seconds": 0}, "TIME_LIMIT"),
                                ({"search_max_states": 1}, "BUDGET_LIMIT")]:
            result = solve(c, **options)
            self.assertFalse(result["search_completed"])
            self.assertFalse(result["optimality_certified"])
            self.assertIn(marker, result["status"])
            self.assertIsNotNone(result["utility_upper_bound"])

    def test_optional_plain_dfs_and_branch_bound_agree(self):
        c = config()
        bounded, plain = solve(c, K=3), solve(c, K=3, dfs_bound_pruning=False)
        self.assertTrue(bounded["optimality_certified"])
        self.assertTrue(plain["optimality_certified"])
        self.assertAlmostEqual(bounded["winner"]["score"], plain["winner"]["score"], places=12)
        self.assertLessEqual(bounded["metrics"]["full_prediction_calls"], plain["metrics"]["full_prediction_calls"])

    def test_six_terms_choose_quality_and_match_milp_and_independent_cartesian(self):
        c = six_term_config()
        predictions = [DeploymentSystem(c).predict(x, routing)
                       for x, routing in grid_plans(c, 2)]
        expected = max(p["U"] for p in predictions if p["U"] is not None)
        dfs = DeploymentSystem(c).plan(route_grid=2)
        milp = DeploymentSystem(replace(c, solver="milp")).plan(route_grid=2)
        self.assertTrue(dfs["optimality_certified"], dfs["status"])
        self.assertTrue(milp["optimality_certified"], milp["status"])
        self.assertEqual(dfs["winner"]["deployment"], {"s": 1})
        self.assertEqual(dfs["winner"]["prediction"]["utility_schema"], "six_term_quality_v1")
        self.assertAlmostEqual(dfs["winner"]["score"], expected, places=10)
        self.assertAlmostEqual(dfs["winner"]["score"], milp["winner"]["score"], places=8)

    def test_missing_quality_is_unknown_not_missing_energy_diagnostic(self):
        c = six_term_config()
        for paths in c.catalog["quality"]["paths"].values():
            for record in paths.values():
                record.pop("value")
        result = solve(c)
        self.assertEqual(result["status"], "DIAGNOSTIC_ONLY_INCOMPLETE_UTILITY")
        self.assertFalse(result["optimality_certified"])
        self.assertIsNone(result["winner"]["prediction"]["U"])
        self.assertIsNotNone(result["winner"]["prediction"]["E_norm"])
        self.assertIsNone(result["winner"]["prediction"]["Q_norm"])
        self.assertGreater(result["metrics"]["unknown_candidates"], 0)
        self.assertIsNotNone(result["utility_upper_bound"])

    def test_deleting_quality_path_identity_also_invalidates_bound_energy(self):
        c = six_term_config()
        c.catalog["quality"]["paths"] = {}
        result = solve(c)
        self.assertFalse(result["optimality_certified"])
        self.assertIsNone(result["winner"]["prediction"]["U"])
        self.assertIsNone(result["winner"]["prediction"]["Q_norm"])
        self.assertIsNone(result["winner"]["prediction"]["E_norm"])

    def test_reoptimized_alternatives_exclude_full_identities(self):
        c = six_term_config()
        best = DeploymentSystem(c).plan(route_grid=2)["winner"]
        alternative = solve(c, exclude_deployments=[best["deployment"]])
        self.assertTrue(alternative["optimality_certified"])
        self.assertNotEqual(alternative["winner"]["deployment"], best["deployment"])
        plan_alternative = solve(c, exclude_plans=[best], no_split=True)
        self.assertTrue(plan_alternative["optimality_certified"])
        self.assertGreater(plan_alternative["metrics"]["excluded_plans"], 0)
        self.assertNotEqual((plan_alternative["winner"]["deployment"], plan_alternative["winner"]["routing"]),
                            (best["deployment"], best["routing"]))
        self.assertTrue(all(len(row) == 1 for row in plan_alternative["winner"]["routing"].values()))

    def test_solo_fifo_floor_bounds_shared_queues_with_actual_poisson_counts(self):
        c = replace(config({"text": 7.0, "image": 6.0},
                           {"g": {"text": 45.0, "image": 50.0}, "s": {"text": 55.0}},
                           horizon=3), arrival_process="poisson")
        optimizer = UtilityOptimizer(c.catalog, c.workload, c.gpu_memory_mib,
                                     c.latency_reference_ms, c.weights,
                                     horizon=c.horizon_seconds,
                                     arrival_process=c.arrival_process,
                                     milp_capacity_witness=False)
        for x, routing in grid_plans(c, 2):
            memory = sum(c.catalog["memory_table_mib"][m][str(n)] for m, n in x.items())
            p = predict_scenarios(c.catalog, x, routing, c.workload, c.gpu_memory_mib,
                                  c.latency_reference_ms, c.weights, horizon=c.horizon_seconds,
                                  arrival_process=c.arrival_process)
            if p["U"] is not None:
                self.assertGreaterEqual(optimizer.upper(x, routing, memory) + 1e-9, p["U"])
        plain = solve(c, K=2, dfs_solo_fifo_bound=False)
        strengthened = solve(c, K=2)
        self.assertAlmostEqual(strengthened["winner"]["score"], plain["winner"]["score"], places=10)


if __name__ == "__main__":
    unittest.main()
