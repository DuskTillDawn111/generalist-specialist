"""Independent finite-domain checks and counterexamples for full-utility MILP."""

import itertools
import unittest
from dataclasses import replace
from unittest.mock import patch
from gsdeploy import DeploymentSystem
from gsdeploy.energy import energy_key
from gsdeploy.performance import profile_key
from tests.test_search_contract import config, oracle_score


def grid_plans(c, K=1):
    """Independent Cartesian oracle, not MILP task/count columns or DFS bounds."""
    models = sorted(c.catalog["models"])
    for counts in itertools.product(
        *(sorted(map(int, c.catalog["capacity_factors"][m])) for m in models)
    ):
        x = {m: n for m, n in zip(models, counts) if n}
        if not x:
            continue
        options = []
        for task, rate in sorted(c.workload.items()):
            supported = [
                m for m in x if task in c.catalog["models"][m]["supported_tasks"]
            ]
            rows = {}
            for d in range(1, K + 1):
                for units in itertools.product(range(d + 1), repeat=len(supported)):
                    if sum(units) != d:
                        continue
                    fractions = tuple(u / d for u in units)
                    rows[fractions] = {
                        m: rate * u / d for m, u in zip(supported, units) if u
                    }
            options.append(list(rows.values()))
        for selected in itertools.product(*options):
            routing = dict(zip(sorted(c.workload), selected))
            yield x, routing


def legal_oracle(c, K=1):
    for x, routing in grid_plans(c, K):
        try:
            v = c.validate_plan(x, routing)
        except ValueError:
            continue
        if v["constraints_valid"]:
            yield x, routing, oracle_score(c, x, routing)


def add_energy(c, plans, joules, multipliers=(1.0,)):
    profiles = c.catalog.setdefault("energy_profiles", {})
    for x, routing in plans:
        for s in multipliers:
            key = energy_key(
                x,
                routing,
                c.workload,
                c.horizon_seconds,
                c.random_seed,
                c.arrival_process,
                s,
                catalog=c.catalog,
            )
            profiles[key] = dict(
                kind="synthetic_fixture",
                scope="whole_gpu_board",
                coverage_valid=True,
                run_joules=joules(x, routing, s),
                startup_joules=0.0,
                provenance="CPU test fixture; no GPU evidence",
            )


class JointMilpTests(unittest.TestCase):
    def test_poisson_burst_fails_drain_and_feedback_finds_legal_alternative(self):
        c = replace(
            config(
                {"text": 3.0},
                {"g": {"text": 250.0}, "s": {"text": 1.0}},
                weights={"Q": 0.0, "T": 0.0, "L": 0.025, "P": 0.025, "M": 0.95},
                horizon=1,
            ),
            solver="milp",
            arrival_process="poisson",
            random_seed=0,
            drain_seconds=0,
            latency_reference_ms={"text": 1000.0},
        )
        c.catalog["memory_table_mib"]["s"]["1"] = 100.0
        system = DeploymentSystem(c)
        slow = system.predict({"g": 1}, {"text": {"g": 3.0}})
        self.assertLess(slow["max_rho"], 0.8)
        self.assertGreater(slow["incomplete_after_drain"], 0)
        result = system.plan()
        self.assertTrue(result["optimality_certified"])
        self.assertEqual(result["winner"]["deployment"], {"s": 1})
        self.assertGreater(result["metrics"]["infeasible_cuts"], 0)

    def test_replicas_splits_and_queue_cost_match_independent_oracle(self):
        for K in (1, 2, 3):
            c = replace(
                config(
                    {"text": 7.0, "image": 6.0},
                    {"g": {"text": 45.0, "image": 50.0}, "s": {"text": 55.0}},
                    horizon=4,
                ),
                solver="milp",
                milp_time_limit_seconds=10,
            )
            c.catalog["capacity_factors"]["g"]["2"] = 1.7
            c.catalog["memory_table_mib"]["g"]["2"] = 2.0
            result = DeploymentSystem(c).plan(route_grid=K)
            expected = max(score for _, _, score in legal_oracle(c, K))
            self.assertTrue(result["optimality_certified"], result["status"])
            self.assertAlmostEqual(result["winner"]["score"], expected, places=8)
            self.assertLessEqual(result["optimality_gap"], c.milp_absolute_gap)
            self.assertGreater(result["metrics"]["feedback_cuts"], 0)
            self.assertTrue(
                any(
                    h["candidate_U"] is not None
                    and -h["candidate_U"] > h["candidate_floor"] + 1e-6
                    for h in result["history"]
                )
            )

    def test_all_specialist_and_capability_gates(self):
        c = replace(
            config(
                services={
                    "g": {"text": 50.0, "image": 50.0},
                    "s": {"text": 1.0, "image": 1.0},
                }
            ),
            solver="milp",
        )
        result = DeploymentSystem(c).plan()
        self.assertEqual(result["winner"]["deployment"], {"s": 1})
        c = replace(
            config(
                {"text": 1.0, "visual_qa": 1.0},
                {"g": {"text": 5.0, "visual_qa": 5.0}, "s": {"text": 1.0}},
            ),
            solver="milp",
        )
        result = DeploymentSystem(c).plan(route_grid=2)
        self.assertEqual(result["winner"]["routing"]["visual_qa"], {"g": 1.0})
        with self.assertRaises(ValueError):
            DeploymentSystem(c).predict(
                {"s": 1}, {"text": {"s": 1}, "visual_qa": {"s": 1}}
            )

    def test_required_replica_and_split_and_explicit_grid_domain(self):
        c = replace(
            config(
                {"text": 1.6, "image": 0.5},
                {"g": {"text": 1000.0, "image": 1000.0}, "s": {"text": 1000.0}},
                horizon=20,
            ),
            solver="milp",
        )
        c.catalog["capacity_factors"]["g"]["2"] = 1.8
        c.catalog["memory_table_mib"]["g"]["2"] = 2.0
        self.assertIsNone(DeploymentSystem(c).plan(route_grid=1)["winner"])
        result = DeploymentSystem(c).plan(route_grid=2)
        self.assertEqual(result["winner"]["deployment"], {"g": 2, "s": 1})
        self.assertEqual(len(result["winner"]["routing"]["text"]), 2)
        self.assertTrue(result["optimality_certified"])
        # A feasible continuous/boundary route need not belong to K=2.
        c = replace(
            config(
                {"text": 1.3},
                {"g": {"text": 1000.0}, "s": {"text": 1000.0}},
                horizon=20,
            ),
            solver="milp",
        )
        c.catalog["capacity_factors"]["s"]["1"] = 0.7
        result = DeploymentSystem(c).plan(route_grid=2)
        self.assertEqual(result["status"], "NO_FEASIBLE_GRID_PLAN")
        self.assertFalse(result["domain"]["continuous_optimality"])
        self.assertIsNotNone(
            DeploymentSystem(replace(c, solver="dfs")).plan(route_grid=2)["winner"]
        )
        augmented = DeploymentSystem(replace(c, milp_capacity_witness=True)).plan(
            route_grid=2
        )
        self.assertIsNotNone(augmented["winner"])
        self.assertTrue(augmented["domain"]["capacity_boundary_witnesses"])
        self.assertTrue(augmented["constraints"]["constraints_valid"])
        self.assertAlmostEqual(
            sum(augmented["winner"]["routing"]["text"].values()), 1.3
        )

    def test_five_term_nonadditive_energy_matches_oracle(self):
        c = replace(
            config(weights={"Q": 0.0, **{k: 0.2 for k in "TLPME"}}, horizon=2),
            solver="milp",
        )
        plans = list(grid_plans(c, 2))

        def joules(x, r, s):
            return 0.1 if len(x) == 2 else 1200.0

        add_energy(c, plans, joules)
        expected = max(
            score
            - c.weights["E"]
            * joules(x, r, 1)
            / c.horizon_seconds
            / c.energy_reference_watts
            for x, r, score in legal_oracle(c, 2)
        )
        result = DeploymentSystem(c).plan(route_grid=2)
        self.assertTrue(result["optimality_certified"], result["status"])
        self.assertEqual(len(result["winner"]["deployment"]), 2)
        self.assertAlmostEqual(result["winner"]["score"], expected, places=8)
        self.assertEqual(result["winner"]["prediction"]["utility_schema"], "five_term")

    def test_missing_and_partial_energy_never_become_zero(self):
        c = replace(
            config(weights={"Q": 0.0, **{k: 0.2 for k in "TLPME"}}), solver="milp"
        )
        result = DeploymentSystem(c).plan(route_grid=2)
        self.assertEqual(result["status"], "DIAGNOSTIC_ONLY_NO_ENERGY")
        self.assertFalse(result["optimality_certified"])
        self.assertTrue(result["diagnostic_optimality_certified"])
        self.assertIsNone(result["winner"]["prediction"]["U"])
        x, r = {"g": 1}, {t: {"g": q} for t, q in c.workload.items()}
        add_energy(c, [(x, r)], lambda *_: 9000.0)
        result = DeploymentSystem(c).plan()
        self.assertIsNotNone(result["winner"]["prediction"]["U"])
        self.assertGreater(result["metrics"]["unknown_candidates"], 0)
        self.assertFalse(result["optimality_certified"])
        self.assertGreater(result["optimality_gap"], 0)
        self.assertGreaterEqual(
            result["utility_upper_bound"], result["winner"]["score"]
        )

    def test_robust_policy_uses_complete_scenario_energy_and_worst_cost(self):
        c = replace(
            config(
                {"text": 2.0},
                {"g": {"text": 100.0}, "s": {"text": 150.0}},
                weights={"Q": 0.0, **{k: 0.2 for k in "TLPME"}},
                horizon=2,
            ),
            solver="milp",
            search_policy="robust",
        )
        plans = list(grid_plans(c))
        add_energy(
            c, plans, lambda x, r, s: 1.0 if "s" in x else 3000.0, (1.0, 1.25, 2.0)
        )
        result = DeploymentSystem(c).plan()
        self.assertTrue(result["optimality_certified"], result["status"])
        p = result["winner"]["prediction"]
        self.assertAlmostEqual(result["winner"]["score"], min(p["scenario_U_values"]))
        c.catalog["energy_profiles"].clear()
        add_energy(c, plans, lambda *_: 1.0)
        missing = DeploymentSystem(c).plan()
        self.assertFalse(missing["optimality_certified"])
        self.assertIsNone(missing["winner"]["prediction"]["scenario_U_low"])

    def test_iteration_build_and_solver_limits_preserve_honest_bounds(self):
        c = replace(
            config(
                {"text": 7.0, "image": 7.0},
                {"g": {"text": 50.0, "image": 50.0}, "s": {"text": 55.0}},
                horizon=4,
            ),
            solver="milp",
            milp_max_iterations=1,
        )
        result = DeploymentSystem(c).plan(route_grid=2)
        self.assertFalse(result["optimality_certified"])
        self.assertFalse(result["search_completed"])
        self.assertEqual(result["status"], "MILP_BUDGET_LIMIT")
        self.assertGreaterEqual(
            result["utility_upper_bound"],
            max(score for _, _, score in legal_oracle(c, 2)),
        )
        self.assertEqual(
            DeploymentSystem(replace(c, milp_max_columns=1)).plan()["status"],
            "MASTER_BUILD_BUDGET_LIMIT",
        )
        self.assertEqual(
            DeploymentSystem(replace(c, milp_time_limit_seconds=1e-12)).plan()[
                "status"
            ],
            "MASTER_BUILD_BUDGET_LIMIT",
        )
        from scipy.optimize import milp as real_milp

        def limited(*args, **kwargs):
            fit = real_milp(*args, **kwargs)
            fit.status = 1
            fit.mip_dual_bound = None
            return fit

        with patch("gsdeploy.milp.milp", limited):
            result = DeploymentSystem(replace(c, milp_max_iterations=20)).plan()
        self.assertFalse(result["optimality_certified"])
        self.assertIsNotNone(result["winner"])

    def test_missing_colocation_never_silently_uses_standalone(self):
        c = replace(config(), solver="milp")
        c.catalog["performance_mode"] = "profiled"
        result = DeploymentSystem(c).plan()
        self.assertEqual(result["status"], "INCOMPLETE_PERFORMANCE_DOMAIN")
        self.assertIsNone(result["winner"])
        self.assertGreater(result["metrics"]["unknown_candidates"], 0)
        x, r = {"g": 1}, {t: {"g": q} for t, q in c.workload.items()}
        p = DeploymentSystem(c).predict(x, r)
        self.assertEqual(p["reason"], "MISSING_COLOCATION_PROFILE")
        self.assertIsNone(p["U"])
        self.assertFalse(c.validate_plan(x, r)["capacity_verified"])
        with self.assertRaisesRegex(ValueError, "unknown colocation capacity"):
            DeploymentSystem(c).run(
                {"winner": {"deployment": x, "routing": r}}, "unused-sample-path"
            )

    def test_long_cohort_lower_bound_still_matches_full_oracle(self):
        c = replace(
            config(
                {"text": 7.0, "image": 6.0},
                {"g": {"text": 45.0, "image": 50.0}, "s": {"text": 55.0}},
                horizon=30,
            ),
            solver="milp",
        )
        result = DeploymentSystem(c).plan(route_grid=2)
        self.assertTrue(result["optimality_certified"], result["status"])
        expected = max(score for _, _, score in legal_oracle(c, 2))
        self.assertAlmostEqual(result["winner"]["score"], expected, places=8)

    def test_whole_plan_colocation_changes_choice_and_energy_identity(self):
        c = replace(
            config({"text": 1.0}, {"g": {"text": 2.0}, "s": {"text": 4.0}}, horizon=2),
            solver="milp",
        )
        c.catalog["performance_mode"] = "profiled"
        c.catalog["colocation_profiles"] = {}
        for x, r in grid_plans(c, 2):
            key = profile_key(
                c.catalog,
                x,
                r,
                c.workload,
                c.horizon_seconds,
                c.random_seed,
                c.arrival_process,
            )
            # Synthetic counterexample: standalone G is fast, actual profiled G
            # is slow. Same deployment has different services at different splits.
            times = {
                m: {"text": (100.0 + 50.0 * r["text"].get("g", 0)) if m == "g" else 4.0}
                for m in x
            }
            c.catalog["colocation_profiles"][key] = dict(
                kind="synthetic_fixture",
                scope="whole_gpu_colocation",
                coverage_valid=True,
                service_time_ms=times,
                provenance="CPU invented route-specific interference fixture",
            )
        result = DeploymentSystem(c).plan(route_grid=2)
        self.assertEqual(result["winner"]["deployment"], {"s": 1})
        self.assertTrue(result["optimality_certified"])
        self.assertEqual(
            result["winner"]["prediction"]["performance_evidence"]["kind"],
            "synthetic_fixture",
        )
        x, r = {"g": 1}, {"text": {"g": 1.0}}
        key = profile_key(c.catalog, x, r, c.workload, 2, 42, c.arrival_process)
        old = energy_key(x, r, c.workload, 2, 42, c.arrival_process, catalog=c.catalog)
        old_stress = energy_key(
            x, r, c.workload, 2, 42, c.arrival_process, 2.0, catalog=c.catalog
        )
        c.catalog["colocation_profiles"][key]["service_time_ms"]["g"]["text"] = 160.0
        new = energy_key(x, r, c.workload, 2, 42, c.arrival_process, catalog=c.catalog)
        self.assertNotEqual(old, new)
        self.assertNotEqual(
            old_stress,
            energy_key(
                x, r, c.workload, 2, 42, c.arrival_process, 2.0, catalog=c.catalog
            ),
        )
        p = DeploymentSystem(replace(c, random_seed=43)).predict(x, r)
        self.assertEqual(p["reason"], "MISSING_COLOCATION_PROFILE")
        with self.assertRaisesRegex(ValueError, "DFS reference"):
            DeploymentSystem(replace(c, solver="dfs")).plan()

    def test_invalid_limits_sparse_cohorts_and_stable_order(self):
        c = replace(config(), solver="milp")
        for field, value in [
            ("solver", "bogus"),
            ("milp_max_iterations", True),
            ("milp_time_limit_seconds", float("nan")),
            ("milp_absolute_gap", -1),
        ]:
            with self.assertRaises(ValueError):
                replace(c, **{field: value})
        sparse = replace(config({"text": 1e-8}, {"g": {"text": 2.0}}), solver="milp")
        self.assertEqual(
            DeploymentSystem(sparse).plan()["status"], "INSUFFICIENT_COHORT"
        )
        a = DeploymentSystem(c).plan(route_grid=2)
        b = DeploymentSystem(
            replace(c, workload=dict(reversed(list(c.workload.items()))))
        ).plan(route_grid=2)
        self.assertEqual(a["winner"]["deployment"], b["winner"]["deployment"])
        self.assertEqual(a["winner"]["routing"], b["winner"]["routing"])
        self.assertEqual(a["trace_sha256"], b["trace_sha256"])


if __name__ == "__main__":
    unittest.main()
