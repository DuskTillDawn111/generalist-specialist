"""Certified GP targets, activation transitions, mesh polls and budget safety."""

import copy
import math
from pathlib import Path
import unittest

import numpy as np

from gsdeploy.config import SystemConfig
from gsdeploy.continuous import (ActivationAwareBilevelOptimizer, Budget,
    ContinuousRoutingEvaluator, ImmutableContinuousProblem, SessionPredictorOracle)
from gsdeploy.continuous.activation_search import ActivationSpaces
from gsdeploy.continuous.bayesian_search import CertifiedRouteGP
from gsdeploy.continuous.mesh_search import MeshAdaptivePoll
from tests.test_c1_lite import evaluator, pure


ROOT = Path(__file__).resolve().parents[1]


class ActivationSpaceTests(unittest.TestCase):
    def test_covering_subsets_keep_exact_zeros_and_all_task_activation(self):
        e = evaluator(lambda *args: -1.)
        spaces = ActivationSpaces(e.domain, e.problem.config.catalog)
        spaces.initialize()
        self.assertIn(("g",), spaces.spaces)
        self.assertIn(("s_shared",), spaces.spaces)
        shares = spaces.center(("g",))
        self.assertTrue(all(row["s_shared"] == 0 for row in shares.values()))
        routing = {"text": {"g": 0., "s_shared": 1.}, "image": {"g": 1.}}
        self.assertEqual(spaces.features(e.domain.validate({t: {m: q for m,q in row.items()}
                         for t,row in routing.items()}))[-2:].tolist(), [1., 1.])

    def test_model_toggle_and_cap_do_not_claim_exhaustiveness(self):
        e = evaluator(lambda *args: -1.)
        spaces = ActivationSpaces(e.domain, e.problem.config.catalog, max_spaces=1)
        spaces.initialize(); spaces.expand(tuple(spaces.models))
        self.assertEqual(len(spaces.spaces), 1)
        self.assertGreater(spaces.report()["capacity_drops"], 0)
        self.assertFalse(spaces.report()["exhaustive"])

    def test_random_routes_conserve_qps_and_stay_on_selected_support(self):
        e = evaluator(lambda *args: -1.)
        spaces = ActivationSpaces(e.domain, e.problem.config.catalog)
        spaces.initialize()
        rng = np.random.default_rng(3)
        for key in spaces.spaces:
            shares = spaces.random(key, rng)
            for task,row in shares.items():
                self.assertAlmostEqual(sum(row.values()), 1.)
                self.assertTrue(all(v == 0 for m,v in row.items() if m not in key))


class BayesianSearchTests(unittest.TestCase):
    def row(self, x, u, certified=True):
        return dict(routing_shares=x, best_feasible_utility=u, certified=certified, status="OK")

    def test_partial_lower_bounds_and_unknown_never_train_gp(self):
        gp = CertifiedRouteGP()
        gp.fit([self.row([0.], -1.), self.row([1.], 1000., False),
                self.row([.3], None, False), self.row([.5], None)], np.asarray)
        self.assertEqual(gp.count, 1)
        self.assertEqual(gp.excluded_partial, 2)
        self.assertEqual(gp.x.tolist(), [[0.]])
        self.assertTrue(np.all(np.isfinite(gp.expected_improvement([[.1], [.9]]))))

    def test_gp_novelty_and_retains_best_under_training_cap(self):
        gp = CertifiedRouteGP(max_training=2)
        gp.fit([self.row([0.], -1.), self.row([.2], -3.), self.row([.4], -2.)], np.asarray)
        self.assertEqual(gp.x.tolist(), [[0.], [.4]])
        scores = gp.expected_improvement([[0.], [1.]])
        self.assertGreater(scores[1], scores[0])

    def test_no_exact_observations_uses_exploration(self):
        gp = CertifiedRouteGP()
        gp.fit([self.row([0.], -1., False)], np.asarray)
        self.assertIsNone(gp.factor)
        self.assertTrue(np.all(gp.expected_improvement([[0.], [1.]]) == 1))


class MeshSearchTests(unittest.TestCase):
    def test_poll_integer_mesh_positive_spanning_and_varying_directions(self):
        poll = MeshAdaptivePoll(3, seed=7, level=5)
        a, b = poll.basis(), poll.basis()
        self.assertEqual(np.linalg.matrix_rank(a), 3)
        self.assertFalse(np.array_equal(a, b))
        for point in poll.candidates([.5, .5, .5]):
            step = (point-.5)/poll.mesh
            np.testing.assert_allclose(step, np.round(step), atol=1e-12)
        self.assertLess(poll.mesh, poll.frame)

    def test_bounds_are_rejected_and_corner_has_feasible_poll(self):
        poll = MeshAdaptivePoll(3, seed=3)
        points = poll.candidates([0., 0., 0.])
        self.assertTrue(points)
        self.assertTrue(all(np.all((p >= 0) & (p <= 1)) for p in points))
        self.assertFalse(any(np.all(p == 0) for p in points))

    def test_inconclusive_does_not_contract_mesh(self):
        poll = MeshAdaptivePoll(2)
        mesh = poll.mesh
        poll.update("inconclusive")
        self.assertEqual(poll.mesh, mesh)
        poll.update("failure")
        self.assertLess(poll.mesh, mesh)
        poll.update("success")
        self.assertEqual(poll.mesh, mesh)
        self.assertFalse(poll.report()["continuous_stationarity_certified"])


class BilevelControllerTests(unittest.TestCase):
    def test_partial_comparison_cannot_become_proven_poll_failure(self):
        parent = dict(best_feasible_utility=-1., valid_upper_bound=0., certified=False)
        trial = dict(best_feasible_utility=-2., valid_upper_bound=0., certified=False)
        self.assertEqual(ActivationAwareBilevelOptimizer._comparison(parent, trial), ("inconclusive", True))
        trial["valid_upper_bound"] = -1.5
        self.assertEqual(ActivationAwareBilevelOptimizer._comparison(parent, trial), ("failure", False))
        trial.update(best_feasible_utility=-.5, valid_upper_bound=0.)
        self.assertEqual(ActivationAwareBilevelOptimizer._comparison(parent, trial), ("success", True))

    def test_endpoints_internal_routes_and_both_search_mechanisms(self):
        e = evaluator(lambda x,r,t: -1.-sum((row.get("g", 0.)/8.-.37)**2 for row in r.values()), outer=70)
        result = ActivationAwareBilevelOptimizer(e, seed=11, inner_predictions=8,
                                                poll_evaluations=2).solve()
        self.assertIsNotNone(result["winner"])
        self.assertFalse(result["optimality_certified"])
        self.assertFalse(result["native_qualified"])
        sources = {r["source"] for r in result["events"]}
        self.assertIn("bo_expected_improvement", sources)
        self.assertIn("mads_poll", sources)
        self.assertGreater(result["surrogate"]["training_observations"], 0)
        self.assertTrue(any(len(r["deployment"]) == 1 for r in result["history"] if r["deployment"]))
        self.assertTrue(any(any(0 < v < 1 for row in r["routing_shares"].values() for v in row.values())
                            for r in result["history"]))

    def test_partial_solver_refines_without_false_gp_targets(self):
        e = evaluator(lambda x,r,t: -1. if len(x) == 2 else -3., outer=45)
        result = ActivationAwareBilevelOptimizer(e, seed=9, inner_predictions=1,
            inner_milp_calls=1, promotion_period=2).solve()
        self.assertIsNotNone(result["winner"])
        self.assertTrue(any(not r["certified"] for r in result["events"] if "certified" in r))
        self.assertTrue(any(r["source"] in {"inner_promotion", "mads_inner_refinement"} for r in result["events"]))
        exact = sum(r["certified"] and r["best_feasible_utility"] is not None for r in result["history"])
        self.assertEqual(result["surrogate"]["training_observations"], exact)
        for event in result["events"]:
            if event["source"] == "mads_comparison":
                self.assertTrue(event["parent_version"] <= event["pool_version"])
                self.assertTrue(event["trial_version"] <= event["pool_version"])

    def test_all_unknown_remains_unknown_and_gp_empty(self):
        e = evaluator(lambda *args: "unknown", outer=20)
        result = ActivationAwareBilevelOptimizer(e, seed=7).solve()
        self.assertIsNone(result["winner"])
        self.assertEqual(result["surrogate"]["training_observations"], 0)
        self.assertFalse(result["optimality_certified"])
        for poll in result["polls"]:
            self.assertTrue(all(p["outcome"] != "failure" for p in poll["polls"]))

    def test_global_budget_is_not_renewed(self):
        e = evaluator(lambda *args: -1., predictions=2, outer=15, milp=4)
        old_limits, old_deadline = e.budget.limits.copy(), e.budget.deadline
        result = ActivationAwareBilevelOptimizer(e, inner_predictions=1, inner_milp_calls=1).solve()
        self.assertLessEqual(result["budget"]["full_prediction_calls"], 2)
        self.assertLessEqual(result["budget"]["milp_solve_calls"], 4)
        self.assertEqual(e.budget.limits, old_limits)
        self.assertEqual(e.budget.deadline, old_deadline)
        self.assertIsNotNone(result["winner"])

    def test_deterministic_count_budget(self):
        results = []
        for _ in range(2):
            e = evaluator(lambda x,r,t: -1.-sum(row.get("g", 0.)/100 for row in r.values()), outer=30)
            result = ActivationAwareBilevelOptimizer(e, seed=3, poll_evaluations=2).solve()
            results.append((result["winner"]["routing_shares"],
                [r["routing_shares"] for r in result["events"] if "routing_shares" in r]))
        self.assertEqual(results[0], results[1])

    def test_exact_inner_against_independent_finite_enumeration(self):
        fun = lambda x,r,t: -sum(x.values())-.001*sum(row.get("g", 0.) for row in r.values())
        e = evaluator(fun, outer=12)
        result = ActivationAwareBilevelOptimizer(e, exact_inner=True).solve()
        for row in result["history"]:
            ref = evaluator(fun).evaluate(row["routing_shares"])
            self.assertTrue(row["certified"])
            self.assertEqual(row["best_feasible_utility"], ref["best_feasible_utility"])

    def test_current_v2_oracle_and_optional_joint_initializer(self):
        c = SystemConfig.load(ROOT/"configs/paper_fifo_reference_v2_cpu_fixture.json")
        p = ImmutableContinuousProblem(c)
        with SessionPredictorOracle(p) as oracle:
            e = ContinuousRoutingEvaluator(p, oracle, budget=Budget(8., 64, 24))
            result = ActivationAwareBilevelOptimizer(e, joint_seed_k=1, seed=11).solve()
        self.assertIsNotNone(result["joint_initializer"])
        self.assertTrue(result["joint_initializer"]["shared_global_budget"])
        self.assertGreaterEqual(result["winner"]["best_feasible_utility"], -3.294666667)
        self.assertFalse(result["native_qualified"])

    def test_single_route_terminates_without_continuous_search(self):
        from dataclasses import replace
        from gsdeploy.continuous import ContinuousProblem
        from tests.test_continuous_routing import tiny, TableOracle
        config = tiny().config
        catalog = copy.deepcopy(config.catalog)
        catalog["models"] = {"g": catalog["models"]["g"]}
        p = ContinuousProblem(replace(config, catalog=catalog), replica_options={"g": [1]})
        e = ContinuousRoutingEvaluator(p, TableOracle(lambda *args: -1.), budget=Budget(5., 20, 20))
        result = ActivationAwareBilevelOptimizer(e).solve()
        self.assertEqual(result["status"], "SINGLE_ROUTE_COMPLETE")
        self.assertEqual(result["budget"]["full_prediction_calls"], 1)
        self.assertEqual(result["surrogate"]["training_observations"], 1)

    def test_joint_incumbent_survives_zero_remaining_outer_budget(self):
        from unittest.mock import patch
        from gsdeploy.continuous.contracts import BudgetExhausted
        c = SystemConfig.load(ROOT/"configs/paper_fifo_reference_v2_cpu_fixture.json")
        p = ImmutableContinuousProblem(c)
        with SessionPredictorOracle(p) as oracle:
            e = ContinuousRoutingEvaluator(p, oracle, budget=Budget(8., 64, 24))
            optimizer = ActivationAwareBilevelOptimizer(e, joint_seed_k=1)
            # An interruption immediately after Joint initialization must not
            # discard an already fully evaluated feasible grid incumbent.
            with patch.object(optimizer.inner, "evaluate", side_effect=BudgetExhausted("OUTER_LIMIT")):
                result = optimizer.solve()
        self.assertEqual(result["status"], "OUTER_LIMIT")
        self.assertIsNotNone(result["winner"])
        self.assertEqual(result["winner"]["source"], "joint_initializer")
        self.assertFalse(result["optimality_certified"])

    def test_gp_handles_positive_certified_values(self):
        e = evaluator(lambda *args: .25, outer=10)
        e.oracle.cost_lower_bound = -1.
        result = ActivationAwareBilevelOptimizer(e).solve()
        self.assertEqual(result["winner"]["best_feasible_utility"], .25)
        self.assertGreater(result["surrogate"]["training_observations"], 0)

    def test_oracle_bound_violation_stops_controller(self):
        e = evaluator(lambda *args: -1.)
        e.oracle.cost_lower_bound = 2.
        result = ActivationAwareBilevelOptimizer(e).solve()
        self.assertEqual(result["status"], "INNER_SOLVER_ERROR")
        self.assertFalse(result["optimality_certified"])
        self.assertEqual(result["surrogate"]["training_observations"], 0)

    def test_reject_invalid_settings_and_unsupported_joint_initializer(self):
        for settings in (dict(use_bo=False, use_mads=False), dict(poll_evaluations=0),
                         dict(joint_seed_k=True), dict(max_spaces=0)):
            with self.assertRaises(ValueError):
                ActivationAwareBilevelOptimizer(evaluator(lambda *args: -1.), **settings)
        with self.assertRaisesRegex(ValueError, "JOINT_INITIALIZER_REQUIRES"):
            ActivationAwareBilevelOptimizer(evaluator(lambda *args: -1.), joint_seed_k=1).solve()


if __name__ == "__main__":
    unittest.main()
