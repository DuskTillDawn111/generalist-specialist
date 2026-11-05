"""Closed-domain, independent inner-oracle and budget/evidence counterexamples."""

import copy
from dataclasses import replace
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest

from gsdeploy import SystemConfig
from gsdeploy.continuous import (Budget, ContinuousProblem, TraceSpec,
    ContinuousRouteDomain, ContinuousRoutingEvaluator, DERoutingOptimizer,
    LegacyPredictorOracle, SyntheticColocationOracle)
from gsdeploy.continuous.activation import active_models, deployment_choices
from gsdeploy.continuous.contracts import BudgetExhausted, digest, load_experiment
from gsdeploy.continuous.benchmark import (legacy_baseline, make_problem,
    oracle_from_spec, materialize_grid_catalog, run_experiment)
from gsdeploy.prediction import predict_scenarios
from gsdeploy.trace import make_trace, expand_routes
from gsdeploy.routing import Router


CONFIG = Path(__file__).resolve().parents[1]/"configs/continuous_fixture.json"


def fixture():
    return load_experiment(CONFIG)


def tiny(policy="nominal"):
    spec, problem = fixture()
    c = problem.config
    catalog = copy.deepcopy(c.catalog)
    catalog["models"] = {m: v for m, v in catalog["models"].items() if m != "s_text"}
    return ContinuousProblem(replace(c, catalog=catalog, search_policy=policy),
        replica_options={"g": [1], "s_shared": [1]})


class TableOracle:
    version = "independent-unit-table-v1"
    evidence_kind = "synthetic_fixture"
    supports_stress = True
    cost_lower_bound = 0.0

    def __init__(self, function):
        self.function = function

    def check(self):
        pass

    def state_costs(self, problem):
        return {}

    def memory_states(self, problem):
        return {}

    def predict(self, problem, deployment, routing, trace, budget, deadline=None):
        budget.consume("predictions", deadline)
        budget.counts["predictor_leaves"] += 1
        value = self.function(deployment, routing, trace)
        if value == "unknown":
            return dict(U=None, constraints_eligible=False, evidence_unknown=True)
        if value == "infeasible":
            return dict(U=None, constraints_eligible=False, stress_constraints_valid=False)
        return dict(U=value, constraints_eligible=True, scenario_U_low=value,
                    stress_constraints_valid=True, stress_evidence_complete=True)


class ContinuousDomainTests(unittest.TestCase):
    def setUp(self):
        self.spec, self.problem = fixture()
        self.domain = ContinuousRouteDomain(self.problem)

    def test_arbitrary_float_is_not_quantized(self):
        value = .3141592653589793
        shares = self.domain.decode([value, .2718281828459045, .6180339887498948])
        self.assertEqual(shares["image"]["g"], value)
        qps = self.domain.to_qps(shares)
        self.assertEqual(qps["image"]["g"], 8*value)
        self.assertNotEqual(digest(shares), digest(self.domain.decode([value+1e-14, .2718281828459045, .6180339887498948])))

    def test_endpoints_cover_g_only_s_only_and_mixed(self):
        modes = {active_models(self.domain.to_qps(s)) for s in self.domain.endpoints()}
        self.assertIn(("g",), modes)
        self.assertIn(("s_shared",), modes)
        self.assertIn(("s_shared", "s_text"), modes)
        self.assertIn(("g", "s_text"), modes)

    def test_three_model_simplex_and_all_faces(self):
        self.assertEqual(len(self.domain.edges["text"]), 3)
        faces = list(self.domain.faces())
        self.assertEqual(len(faces), self.domain.face_count)
        for face in faces:
            shares = self.domain.decode([.382] * sum(len(v)-1 for v in face.values()), face)
            for row in shares.values():
                self.assertAlmostEqual(sum(row.values()), 1)
        self.assertEqual(self.domain.decode([0, 0, 0])["text"]["s_text"], 1)

    def test_multiple_generalists_are_capability_driven(self):
        c = self.problem.config
        catalog = copy.deepcopy(c.catalog)
        catalog["models"]["s_shared"]["role"] = "generalist"
        d = ContinuousRouteDomain(ContinuousProblem(replace(c, catalog=catalog)))
        self.assertEqual(d.edges, self.domain.edges)

    def test_model_activation_sums_all_tasks_and_retains_idle_residents(self):
        q = {"text": {"g": 0, "s_shared": 1}, "image": {"g": 1}}
        self.assertEqual(active_models(q), ("g", "s_shared"))
        choices = deployment_choices(self.problem, q)
        self.assertNotIn(0, choices["s_shared"])
        self.assertEqual(choices["s_text"], (0, 1, 2))

    def test_invalid_shares_rejected(self):
        for bad in ({"text": {"g": 1}},
                    {"text": {"g": 1}, "image": {"absent": 1}},
                    {"text": {"g": math.nan}, "image": {"g": 1}},
                    {"text": {"g": 1.01, "s_text": -.01}, "image": {"g": 1}},
                    {"text": {"g": .9}, "image": {"g": 1}}):
            with self.assertRaises(ValueError):
                self.domain.validate(bad)

    def test_tiny_positive_flow_is_active_and_underflow_is_explicit(self):
        q = self.domain.to_qps({"text": {"g": 1-1e-14, "s_text": 1e-14}, "image": {"g": 1}})
        self.assertIn("s_text", active_models(q))
        c = replace(self.problem.config, workload={"text": 5e-324})
        p = ContinuousProblem(c, [TraceSpec("tiny", 1, ((0, "text"),))])
        with self.assertRaisesRegex(ValueError, "QPS_UNDERFLOW"):
            ContinuousRouteDomain(p).to_qps({"text": {"g": .1, "s_text": .9}})


class InnerCorrectnessTests(unittest.TestCase):
    def assert_inner_agreement(self, problem, oracle, routing):
        rows = []
        for inner in ("enumeration", "milp"):
            e = ContinuousRoutingEvaluator(problem, oracle, inner, Budget(20, 2000, 200))
            rows.append(e.evaluate(routing))
        self.assertEqual(rows[0]["status"], rows[1]["status"])
        self.assertEqual(rows[0]["best_feasible_utility"], rows[1]["best_feasible_utility"])
        self.assertEqual(rows[0]["certified"], rows[1]["certified"])
        return rows

    def test_non_grid_inner_matches_independent_cartesian_predictor(self):
        spec, p = fixture()
        oracle = oracle_from_spec(spec)
        shares = {"text": {"g": .3141592653589793, "s_text": 1-.3141592653589793},
                  "image": {"s_shared": 1}}
        rows = self.assert_inner_agreement(p, oracle, shares)
        routing = ContinuousRouteDomain(p).to_qps(shares)
        values = []
        names = sorted(p.states)
        for ns in itertools.product(*(p.states[m] for m in names)):
            x = {m: n for m, n in zip(names, ns) if n}
            if any(m not in x for row in routing.values() for m in row):
                continue
            prediction = oracle.predict(p, x, routing, p.train_traces[0], Budget())
            if prediction["U"] is not None:
                values.append(prediction["U"])
        self.assertEqual(rows[0]["best_feasible_utility"], max(values))
        self.assertFalse(rows[1]["outer_continuous_optimality"])

    def test_idle_resident_can_be_optimal_in_nonseparable_oracle(self):
        p = tiny()
        rows = self.assert_inner_agreement(p, TableOracle(lambda x,r,t: -.1 if "s_shared" in x else -.9),
            {"text": {"g": 1}, "image": {"g": 1}})
        self.assertIn("s_shared", rows[0]["deployment"])

    def test_unknown_does_not_become_infeasible_or_certified(self):
        rows = self.assert_inner_agreement(tiny(), TableOracle(lambda x,r,t:
            "unknown" if "s_shared" in x else -.3),
            {"text": {"g": 1}, "image": {"g": 1}})
        for row in rows:
            self.assertTrue(row["feasible"])
            self.assertFalse(row["certified"])
            self.assertEqual(row["status"], "INCOMPLETE_EVIDENCE")
            self.assertEqual(row["valid_utility_upper_bound"], 0)
            self.assertEqual(len(row["unresolved_candidates"]), 1)

    def test_all_infeasible_is_distinct_from_all_unknown(self):
        for value, status in (("infeasible", "PROVEN_INFEASIBLE"), ("unknown", "INCOMPLETE_EVIDENCE")):
            rows = self.assert_inner_agreement(tiny(), TableOracle(lambda x,r,t: value),
                {"text": {"g": 1}, "image": {"g": 1}})
            self.assertEqual(rows[0]["status"], status)

    def test_no_memory_feasible_deployment(self):
        spec, p = fixture()
        p = ContinuousProblem(replace(p.config, gpu_memory_mib=1.0))
        rows = self.assert_inner_agreement(p, oracle_from_spec(spec),
            {"text": {"g": 1}, "image": {"g": 1}})
        self.assertEqual(rows[0]["status"], "PROVEN_INFEASIBLE")

    def test_missing_quality_and_energy_never_become_formal_utility(self):
        p = tiny()
        rows = self.assert_inner_agreement(p, LegacyPredictorOracle(),
            {"text": {"g": 1}, "image": {"g": 1}})
        self.assertIsNone(rows[0]["best_feasible_utility"])
        self.assertEqual(rows[0]["status"], "INCOMPLETE_EVIDENCE")

    def test_prediction_budget_preserves_partial_incumbent(self):
        p, oracle = tiny(), TableOracle(lambda x,r,t: -.5)
        for inner in ("enumeration", "milp"):
            e = ContinuousRoutingEvaluator(p, oracle, inner, Budget(20, 1, 10))
            row = e.evaluate({"text": {"g": 1}, "image": {"g": 1}})
            self.assertEqual(row["status"], "BUDGET_LIMIT_WITH_INCUMBENT")
            self.assertEqual(row["best_feasible_utility"], -.5)
            self.assertFalse(row["certified"])
            self.assertEqual(e.budget.counts["predictions"], 1)

    def test_inner_timeout_is_not_a_certified_low_score(self):
        e = ContinuousRoutingEvaluator(tiny(), TableOracle(lambda x,r,t: -.5),
            budget=Budget(20), inner_seconds=1e-12)
        row = e.evaluate({"text": {"g": 1}, "image": {"g": 1}})
        self.assertEqual(row["status"], "BUDGET_LIMIT_NO_INCUMBENT")
        self.assertFalse(row["inner_search_completed"])

    def test_missing_global_bound_does_not_assume_nonpositive_utility(self):
        oracle = TableOracle(lambda x,r,t: 2.0 if "s_shared" in x else 1.0)
        oracle.cost_lower_bound = None
        rows = self.assert_inner_agreement(tiny(), oracle,
            {"text": {"g": 1}, "image": {"g": 1}})
        self.assertEqual(rows[1]["best_feasible_utility"], 2.0)

    def test_multitrace_one_deployment_not_mean_of_per_trace_optima(self):
        c = tiny().config
        traces = [TraceSpec("a", 1, ((0, "text"), (.1, "image"))),
                  TraceSpec("b", 2, ((0, "text"), (.1, "image")))]
        p = ContinuousProblem(c, traces, {"g": [1], "s_shared": [1]})
        def value(x, r, t):
            return (-1 if "s_shared" in x else -10) if t.seed == 1 else (-8 if "s_shared" in x else -1)
        rows = self.assert_inner_agreement(p, TableOracle(value),
            {"text": {"g": 1}, "image": {"g": 1}})
        self.assertEqual(rows[0]["best_feasible_utility"], -4.5)

    def test_unsupported_robust_stays_unknown(self):
        oracle = TableOracle(lambda x,r,t: -.1)
        oracle.supports_stress = False
        rows = self.assert_inner_agreement(tiny("robust"), oracle,
            {"text": {"g": 1}, "image": {"g": 1}})
        self.assertEqual(rows[0]["status"], "INCOMPLETE_EVIDENCE")

    def test_synthetic_robust_uses_original_stress(self):
        spec, p = fixture()
        p = ContinuousProblem(replace(p.config, search_policy="robust"))
        self.assert_inner_agreement(p, oracle_from_spec(spec),
            {"text": {"s_text": 1}, "image": {"s_shared": 1}})

    def test_proven_state_floor_can_reduce_milp_predictions(self):
        p = tiny()
        def cost(x):
            return .1 + (.8 if "s_shared" in x else 0)
        oracle = TableOracle(lambda x,r,t: -cost(x)-.01)
        oracle.state_costs = lambda p: {("g",0):0, ("g",1):.1,
                                      ("s_shared",0):0, ("s_shared",1):.8}
        rows = self.assert_inner_agreement(p,oracle,
            {"text":{"g":1},"image":{"g":1}})
        self.assertLess(rows[1]["full_prediction_calls"],rows[0]["full_prediction_calls"])
        self.assertEqual(rows[1]["safely_pruned_deployments"],1)

    def test_invalid_bound_is_not_promoted_to_a_certificate(self):
        oracle = TableOracle(lambda x,r,t: -.01)
        oracle.state_costs = lambda p: {("g",0):0, ("g",1):.5,
                                      ("s_shared",0):0, ("s_shared",1):.5}
        e = ContinuousRoutingEvaluator(tiny(),oracle,"milp")
        row = e.evaluate({"text":{"g":1},"image":{"g":1}})
        self.assertEqual(row["status"],"SOLVER_ERROR")
        self.assertFalse(row["certified"])

    def test_robust_multitrace_aggregates_mean_of_minima(self):
        from gsdeploy.continuous.predictor_adapter import normalize_prediction
        class Scenarios(TableOracle):
            def predict(self,p,x,r,t,b,deadline=None):
                b.consume("predictions",deadline)
                b.counts["predictor_leaves"]+=3
                values = [-1,-9,-2] if t.seed==1 else [-8,-1,-2]
                return dict(U=values[0],constraints_eligible=True,
                    scenario_U_low=min(values),stress_constraints_valid=True,
                    stress_evidence_complete=True)
        c = tiny("robust").config
        p = ContinuousProblem(c,[TraceSpec("a",1,((0,"text"),(.1,"image"))),
                                 TraceSpec("b",2,((0,"text"),(.1,"image")))],
                              {"g":[1],"s_shared":[1]})
        rows = self.assert_inner_agreement(p,Scenarios(None),
            {"text":{"g":1},"image":{"g":1}})
        self.assertEqual(rows[0]["best_feasible_utility"],-8.5)

    def test_uncertain_legacy_stress_failure_is_unknown(self):
        from gsdeploy.continuous.predictor_adapter import normalize_prediction
        state,value=normalize_prediction(dict(U=-.1,constraints_eligible=True,
            scenario_U_low=None,stress_constraints_valid=False),"robust")
        self.assertEqual(state,"UNKNOWN")
        self.assertIsNone(value)


class CacheAndOuterTests(unittest.TestCase):
    def test_exact_cache_and_returned_object_isolation(self):
        e = ContinuousRoutingEvaluator(tiny(), TableOracle(lambda x,r,t: -.1), budget=Budget(20))
        shares = {"text": {"g": .3141592653589793, "s_shared": 1-.3141592653589793}, "image": {"g": 1}}
        a = e.evaluate(shares)
        a["prediction"]["traces"].clear()
        b = e.evaluate(shares)
        self.assertTrue(b["cached"])
        self.assertEqual(b["full_prediction_calls"], 0)
        self.assertTrue(b["prediction"]["traces"])
        shares["text"]["g"] += 1e-14
        shares["text"]["s_shared"] -= 1e-14
        self.assertFalse(e.evaluate(shares)["cached"])

    def test_input_snapshot_is_owned_and_changed_session_rejected(self):
        c = tiny().config
        p = ContinuousProblem(c)
        c.catalog["models"]["g"]["service_time_ms"]["text"] *= 2
        p.check()
        p.config.catalog["safe_factor"] = 2
        with self.assertRaisesRegex(ValueError, "PROBLEM_CHANGED"):
            p.check()

    def test_public_candidate_rejects_bad_rates_and_undeclared_traces(self):
        e = ContinuousRoutingEvaluator(tiny(), TableOracle(lambda x,r,t: -.1))
        with self.assertRaises(ValueError):
            e.candidate({"g":1},{"text":{"g":1},"image":{"g":8}})
        c=e.problem.config
        valid={t:{"g":q} for t,q in c.workload.items()}
        with self.assertRaisesRegex(ValueError,"UNDECLARED_TRACE"):
            e.candidate({"g":1},valid,traces=[TraceSpec("other",9,((0,"text"),(.1,"image")))])

    def test_trace_split_leakage_and_malformed_trace_rejected(self):
        c = tiny().config
        with self.assertRaisesRegex(ValueError, "TRACE_SPLIT_LEAKAGE"):
            ContinuousProblem(c, [TraceSpec("a", 1, ((0,"text"),(1,"image"))),
                TraceSpec("b", 1, ((0,"text"),(1,"image")), "test")])
        with self.assertRaisesRegex(ValueError, "INVALID_FROZEN_TRACE"):
            ContinuousProblem(c, [TraceSpec("a", 1, ((1,"text"),(0,"image")))])

    def test_fifo_mapping_changes_with_replica_count(self):
        routing = {"text": {"g": 1}, "image": {"g": 1}}
        _, route = expand_routes({"g": 2}, routing)
        router = Router(route)
        self.assertEqual([router.choose("text") for _ in range(4)], ["g#0","g#1","g#0","g#1"])
        self.assertEqual(router.choose("image"), "g#0")

    def test_source_version_and_cache_scope(self):
        e = ContinuousRoutingEvaluator(tiny(), TableOracle(lambda x,r,t: -.1))
        e.oracle.version = "changed"
        with self.assertRaisesRegex(ValueError, "EVALUATOR_IDENTITY_CHANGED"):
            e.evaluate({"text": {"g": 1}, "image": {"g": 1}})

    def test_fixed_seed_de_reproduces_candidates_and_winner(self):
        results = []
        histories = []
        for _ in range(2):
            e = ContinuousRoutingEvaluator(tiny(), TableOracle(lambda x,r,t:
                -(r["text"].get("g", 0)-3.141592653589793)**2), budget=Budget(20,1000,32))
            fit = DERoutingOptimizer(e, seed=29, popsize=3, maxiter=3).solve()
            results.append(fit["winner"]["best_feasible_utility"])
            histories.append([r["routing_shares"] for r in e.history])
            self.assertFalse(fit["continuous_optimality"])
        self.assertEqual(results[0], results[1])
        self.assertEqual(histories[0], histories[1])
        # This independent objective has an interior optimum outside the old
        # K2 grid. Reproducibility must not merely compare endpoint winners.
        coarse = max(-(q-3.141592653589793)**2 for q in (0,4,8))
        self.assertGreater(results[0], coarse)

    def test_zero_dimensional_search(self):
        p = tiny()
        catalog = copy.deepcopy(p.config.catalog)
        del catalog["models"]["s_shared"]
        p = ContinuousProblem(replace(p.config, catalog=catalog))
        e = ContinuousRoutingEvaluator(p, TableOracle(lambda x,r,t: -.2))
        result = DERoutingOptimizer(e).solve()
        self.assertEqual(result["budget"]["outer_evaluations"], 1)
        self.assertEqual(result["winner"]["best_feasible_utility"], -.2)

    def test_incomplete_inner_stops_de_without_fake_v_observation(self):
        e = ContinuousRoutingEvaluator(tiny(), TableOracle(lambda x,r,t: -.2),
            budget=Budget(20,1,10))
        fit = DERoutingOptimizer(e).solve()
        self.assertEqual(fit["status"], "INNER_EVALUATION_INCOMPLETE")
        self.assertFalse(fit["exact_inner_comparison"])

    def test_placeholder_never_becomes_feasible(self):
        p = tiny()
        d = copy.deepcopy(p.config.catalog)
        d.update(performance_mode="route_load_replay_v1", memory_mode="route_load_replay_v1",
            route_load_replay_v1={"schema":"gsdeploy.route_load_replay.placeholder.v1", "status":"UNIMPLEMENTED"})
        e = ContinuousRoutingEvaluator(ContinuousProblem(replace(p.config,catalog=d)), LegacyPredictorOracle())
        result = DERoutingOptimizer(e).solve()
        self.assertIsNone(result["winner"])
        self.assertEqual(result["status"], "INCOMPLETE_PREDICTOR_EVIDENCE")

    def test_synthetic_profile_baseline_matches_same_candidate_oracle(self):
        spec, p = fixture()
        oracle = oracle_from_spec(spec)
        data, count = materialize_grid_catalog(p, oracle, 1, Budget(20), p.train_traces[0])
        self.assertGreater(count, 0)
        x = {"g":1,"s_shared":1}
        r = {"text":{"g":8.0},"image":{"s_shared":8.0}}
        c, trace = p.config, p.train_traces[0]
        direct = oracle.predict(p,x,r,trace,Budget())
        frozen = predict_scenarios(data,x,r,c.workload,c.gpu_memory_mib,c.latency_reference_ms,c.weights,
            trace=trace.arrivals,horizon=c.horizon_seconds,seed=trace.seed,
            arrival_process=c.arrival_process,drain_seconds=c.drain_seconds,
            startup_amortization_seconds=c.startup_amortization_seconds)
        self.assertEqual(direct["scenario_U_values"], frozen["scenario_U_values"])

    def test_multitrace_old_baseline_is_not_misrepresented(self):
        spec, p = fixture()
        p = make_problem(p.config, {"train":[101,102]})
        row = legacy_baseline(p,oracle_from_spec(spec),"milp",1,10,100)
        self.assertEqual(row["status"], "UNSUPPORTED_MULTITRACE_TRAINING_OBJECTIVE")


if __name__ == "__main__":
    unittest.main()
