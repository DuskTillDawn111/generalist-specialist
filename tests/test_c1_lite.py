"""Independent finite-domain and progressive-observation counterexamples."""

import copy
from dataclasses import replace
import math
import unittest

from gsdeploy.continuous import Budget, ContinuousProblem, ContinuousRoutingEvaluator
from gsdeploy.continuous.c1_lite import C1LiteOptimizer
from gsdeploy.continuous.contracts import BudgetExhausted
from gsdeploy.continuous.deployment_pool import DeploymentCandidatePool
from gsdeploy.continuous.progressive import ProgressiveFixedRoutingSolver
from tests.test_continuous_routing import TableOracle, tiny, fixture


def evaluator(function, *, problem=None, predictions=100, outer=200, milp=200):
    return ContinuousRoutingEvaluator(problem or tiny(), TableOracle(function),
                                     budget=Budget(10., predictions, outer, milp))


def pure(e, model="g"):
    return {t: {model: 1.} for t in e.problem.config.workload}


class CandidatePoolTests(unittest.TestCase):
    def test_replica_identity_eviction_and_incumbent_protection(self):
        _, p = fixture()
        pool = DeploymentCandidatePool(p, max_size=2)
        pool.add({"g": 1}, source="test", utility=-1.)
        pool.add({"g": 2}, source="test", utility=-2.)
        self.assertEqual(len(pool.entries), 2)
        pool.add({"s_shared": 1}, source="test", utility=-3.)
        self.assertIn((("g", 1),), pool.entries)
        self.assertEqual(len(pool.entries), 2)
        self.assertFalse(pool.events[-1]["safe_prune"])
        with self.assertRaises(ValueError):
            pool.add({"g": True}, source="test")
        with self.assertRaises(ValueError):
            pool.add({"g": 9}, source="test")

    def test_total_activation_and_defensive_copy(self):
        e = evaluator(lambda *args: -1.)
        pool = DeploymentCandidatePool(e.problem)
        pool.add({"g": 1}, source="test")
        pool.add({"g": 1, "s_shared": 1}, source="test")
        routing = {"text": {"g": 0., "s_shared": 8.}, "image": {"g": 8.}}
        rows = pool.compatible(routing)
        self.assertEqual(rows, [{"g": 1, "s_shared": 1}])
        rows[0]["g"] = 5
        self.assertEqual(pool.compatible(routing)[0]["g"], 1)


class ProgressiveTests(unittest.TestCase):
    def test_exact_non_grid_matches_independent_enumeration(self):
        fun = lambda x, r, t: -sum(x.values())-.1*r["text"].get("g", 0.)
        e = evaluator(fun)
        shares = {t: {"g": .3141592653589793, "s_shared": 1-.3141592653589793}
                  for t in e.problem.config.workload}
        solver = ProgressiveFixedRoutingSolver(e, DeploymentCandidatePool(e.problem), use_bounds=False)
        row = solver.evaluate(shares, exact=True)
        ref = evaluator(fun).evaluate(shares)
        self.assertTrue(row["certified"])
        self.assertEqual(row["best_feasible_utility"], ref["best_feasible_utility"])
        self.assertEqual(row["deployment"], ref["deployment"])

    def test_refinement_keeps_ledger_and_skips_old_identity(self):
        e = evaluator(lambda x, r, t: -1. if len(x) == 2 else -4.)
        solver = ProgressiveFixedRoutingSolver(e, DeploymentCandidatePool(e.problem),
                                               inner_predictions=1, use_bounds=False)
        a = solver.evaluate(pure(e), force=True)
        self.assertFalse(a["certified"])
        self.assertEqual(a["full_prediction_calls"], 1)
        b = solver.evaluate(pure(e), exact=True)
        self.assertTrue(b["certified"])
        self.assertEqual(b["best_feasible_utility"], -1.)
        self.assertEqual(e.budget.counts["predictions"], 2)
        self.assertGreaterEqual(b["inner_sessions"], 2)
        self.assertEqual(e.budget.limits["predictions"], 100)

    def test_pool_value_is_lower_bound_and_new_route_repredicted(self):
        e = evaluator(lambda x, r, t: -sum(x.values())-r["text"].get("g", 0.)/100)
        pool = DeploymentCandidatePool(e.problem)
        pool.add({"g": 1}, source="test")
        solver = ProgressiveFixedRoutingSolver(e, pool, exploration_period=100, use_bounds=False)
        a = solver.evaluate(pure(e))
        self.assertFalse(a["certified"])
        b = solver.evaluate({t: {"g": .7, "s_shared": .3} for t in e.problem.config.workload})
        self.assertTrue(b["feasible"])
        self.assertGreater(b["full_prediction_calls"], 0)
        self.assertNotEqual(a["best_feasible_utility"], b["best_feasible_utility"])

    def test_unknown_identity_preserves_upper_and_blocks_certificate(self):
        e = evaluator(lambda x, r, t: "unknown" if len(x) == 2 else -2.)
        solver = ProgressiveFixedRoutingSolver(e, DeploymentCandidatePool(e.problem), use_bounds=False)
        row = solver.evaluate(pure(e), exact=True)
        self.assertTrue(row["master_domain_exhausted"])
        self.assertFalse(row["certified"])
        self.assertEqual(row["valid_upper_bound"], 0.)
        self.assertEqual(len(row["unresolved_candidates"]), 1)
        self.assertEqual(row["best_feasible_utility"], -2.)

    def test_all_unknown_is_not_infeasible(self):
        e = evaluator(lambda *args: "unknown")
        row = ProgressiveFixedRoutingSolver(e, DeploymentCandidatePool(e.problem), use_bounds=False).evaluate(pure(e), exact=True)
        self.assertIsNone(row["best_feasible_utility"])
        self.assertEqual(row["constraint_state"], "UNKNOWN")
        self.assertFalse(row["certified"])

    def test_no_global_floor_and_unknown_means_no_upper(self):
        e = evaluator(lambda x, r, t: "unknown" if len(x) == 2 else -2.)
        e.oracle.cost_lower_bound = None
        row = ProgressiveFixedRoutingSolver(e, DeploymentCandidatePool(e.problem), use_bounds=False).evaluate(pure(e), exact=True)
        self.assertIsNone(row["valid_upper_bound"])
        self.assertFalse(row["certified"])

    def test_proven_infeasible_requires_full_search(self):
        e = evaluator(lambda *args: "infeasible")
        row = ProgressiveFixedRoutingSolver(e, DeploymentCandidatePool(e.problem), use_bounds=False).evaluate(pure(e), exact=True)
        self.assertTrue(row["certified"])
        self.assertEqual(row["status"], "PROVEN_INFEASIBLE")

    def test_pool_eviction_does_not_remove_global_winner_or_route_ledger(self):
        e = evaluator(lambda x, r, t: -.5 if len(x) == 2 else -1.)
        pool = DeploymentCandidatePool(e.problem, max_size=1)
        solver = ProgressiveFixedRoutingSolver(e, pool, use_bounds=False)
        row = solver.evaluate(pure(e), exact=True)
        pool.add({"s_shared": 1}, source="other", utility=-3.)
        again = solver.evaluate(pure(e))
        self.assertEqual(again["best_feasible_utility"], row["best_feasible_utility"])
        self.assertEqual(again["full_prediction_calls"], 0)
        self.assertEqual(solver.best["best_feasible_utility"], -.5)

    def test_static_memory_floor_can_exclude_but_missing_floor_cannot(self):
        e = evaluator(lambda *args: -2.)
        e.oracle.memory_states = lambda p: {(m, n): 1000.*n for m, ns in p.states.items() for n in ns}
        row = ProgressiveFixedRoutingSolver(e, DeploymentCandidatePool(e.problem), use_bounds=False).evaluate(pure(e), exact=True)
        self.assertTrue(row["certified"])
        self.assertFalse(row["feasible"])
        self.assertEqual(row["full_prediction_calls"], 0)

    def test_positive_utility_allowed_when_global_cost_floor_is_negative(self):
        e = evaluator(lambda x, r, t: .25 if len(x) == 1 else -.25)
        e.oracle.cost_lower_bound = -1.
        row = ProgressiveFixedRoutingSolver(e, DeploymentCandidatePool(e.problem), use_bounds=False).evaluate(pure(e), exact=True)
        self.assertTrue(row["certified"])
        self.assertEqual(row["best_feasible_utility"], .25)
        self.assertEqual(row["valid_upper_bound"], .25)

    def test_local_milp_limit_restored_and_global_limit_never_renewed(self):
        e = evaluator(lambda *args: -1., milp=1)
        solver = ProgressiveFixedRoutingSolver(e, DeploymentCandidatePool(e.problem), use_bounds=False)
        row = solver.evaluate(pure(e), force=True)
        self.assertFalse(row["certified"])
        solver.evaluate(pure(e), force=True)
        self.assertEqual(e.budget.counts["milp"], 1)
        self.assertEqual(e.budget.limits["milp"], 1)

    def test_mutated_problem_rejected(self):
        e = evaluator(lambda *args: -1.)
        solver = ProgressiveFixedRoutingSolver(e, DeploymentCandidatePool(e.problem))
        e.problem.config.catalog["safe_factor"] = 3.
        with self.assertRaisesRegex(ValueError, "PROBLEM_CHANGED"):
            solver.evaluate(pure(e))

    def test_cross_route_ledger_isolation(self):
        e = evaluator(lambda x, r, t: -r["text"].get("g", 0.)-.1*len(x))
        solver = ProgressiveFixedRoutingSolver(e, DeploymentCandidatePool(e.problem), use_bounds=False)
        a = solver.evaluate(pure(e), exact=True)
        b = solver.evaluate(pure(e, "s_shared"), exact=True)
        self.assertNotEqual(a["best_feasible_utility"], b["best_feasible_utility"])
        self.assertEqual(len(solver.routes), 2)
        self.assertTrue(a["certified"] and b["certified"])

    def test_partial_pool_refresh_does_not_spend_exploration_cadence(self):
        e = evaluator(lambda *args: -1.)
        pool = DeploymentCandidatePool(e.problem)
        pool.add({"g": 1}, source="test")
        solver = ProgressiveFixedRoutingSolver(e, pool, exploration_period=10, use_bounds=False)
        solver.evaluate(pure(e))
        for _ in range(3):
            solver.evaluate(pure(e), pool_only=True)
        self.assertEqual(solver.trigger.calls, 1)

    def test_bound_violation_never_produces_certificate(self):
        e = evaluator(lambda *args: -1.)
        e.oracle.cost_lower_bound = 2.
        row = ProgressiveFixedRoutingSolver(e, DeploymentCandidatePool(e.problem), use_bounds=False).evaluate(pure(e), exact=True)
        self.assertEqual(row["status"], "SOLVER_ERROR")
        self.assertFalse(row["certified"])
        self.assertIsNone(row["valid_upper_bound"])

    def test_current_reference_v2_phase_predictor_matches_enumeration(self):
        from pathlib import Path
        from gsdeploy.config import SystemConfig
        from gsdeploy.continuous import ImmutableContinuousProblem, SessionPredictorOracle
        c = SystemConfig.load(Path(__file__).resolve().parents[1]/"configs/paper_fifo_reference_v2_cpu_fixture.json")
        p = ImmutableContinuousProblem(c)
        with SessionPredictorOracle(p) as oracle:
            e = ContinuousRoutingEvaluator(p, oracle, budget=Budget(5., 64, 20))
            solver = ProgressiveFixedRoutingSolver(e, DeploymentCandidatePool(p))
            shares = {"text": {"g": .3141592653589793, "s": 1-.3141592653589793}}
            row = solver.evaluate(shares, exact=True)
            ref = ContinuousRoutingEvaluator(p, oracle, budget=Budget(5., 64, 20)).evaluate(shares)
            self.assertTrue(row["certified"])
            self.assertAlmostEqual(row["best_feasible_utility"], ref["best_feasible_utility"])
            self.assertFalse(row["native_qualified"])


class ProgressiveDETests(unittest.TestCase):
    def test_partial_observations_continue_into_interior_and_preserve_best(self):
        e = evaluator(lambda x, r, t: -sum(x.values())-.01*sum(abs(q-4.) for row in r.values() for q in row.values()), outer=40)
        opt = C1LiteOptimizer(e, seed=3, population_size=4, endpoint_limit=1, face_limit=0,
                              inner_predictions=1, use_bounds=False, exploration_period=2)
        result = opt.solve()
        self.assertIsNotNone(result["winner"])
        self.assertEqual(result["evaluation_semantics"], "lower_bound_guided")
        self.assertFalse(result["continuous_optimality"])
        self.assertTrue(any(not r["certified"] for r in result["history"]))
        self.assertTrue(any(any(0 < s < 1 for row in r["routing_shares"].values() for s in row.values())
                            for r in result["history"]))
        self.assertTrue(result["selections"])
        for row in result["selections"]:
            self.assertTrue(row["parent_certified"] or row["parent_version"] == row["pool_version"])
            self.assertTrue(row["trial_certified"] or row["trial_version"] == row["pool_version"])

    def test_pool_expansion_refreshes_old_population(self):
        e = evaluator(lambda x, r, t: -sum(x.values()), outer=55)
        result = C1LiteOptimizer(e, seed=7, population_size=4, endpoint_limit=1,
            face_limit=0, exploration_period=2, inner_predictions=1, use_bounds=False).solve()
        refreshes = [r for r in result["history"] if r["termination_reason"] == "POOL_REFRESH"]
        self.assertTrue(refreshes)
        for row in result["selections"]:
            self.assertTrue(row["parent_certified"] or row["parent_version"] == row["pool_version"])
            self.assertTrue(row["trial_certified"] or row["trial_version"] == row["pool_version"])

    def test_deterministic_with_count_budget(self):
        values = []
        for _ in range(2):
            e = evaluator(lambda x, r, t: -sum(x.values()), outer=25)
            r = C1LiteOptimizer(e, seed=9, endpoint_limit=2, face_limit=1, use_bounds=False).solve()
            values.append((r["winner"]["deployment"], r["winner"]["routing_shares"],
                           [h["routing_shares"] for h in r["history"]]))
        self.assertEqual(values[0], values[1])

    def test_unknown_never_becomes_numeric_objective(self):
        e = evaluator(lambda *args: "unknown", outer=15)
        r = C1LiteOptimizer(e, population_size=4, endpoint_limit=0, face_limit=0, use_bounds=False).solve()
        self.assertIsNone(r["winner"])
        for row in r["selections"]:
            self.assertIsNone(row["parent_lb"])
            self.assertIsNone(row["trial_lb"])
        self.assertFalse(r["optimality_certified"])


if __name__ == "__main__":
    unittest.main()
