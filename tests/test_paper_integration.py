"""Tiny analytical fixtures, explicitly NOT fabricated GPU measurements."""

import copy
from dataclasses import replace
import itertools
import math
import unittest
from unittest.mock import patch

from gsdeploy.config import SystemConfig
from gsdeploy.forecasting.profiles import context_key
from gsdeploy.paper_forecast import MODE, SCHEMA, session
from gsdeploy.paper_algorithms import J3OInspiredOptimizer, UsherResourcePlanner, fixed_deployment_lp
from gsdeploy.prediction import predict, predict_scenarios
from gsdeploy.system import DeploymentSystem
from tests.test_static_dfs_v6 import six_term_config


def fixture_config():
    c = replace(six_term_config(), workload={"text": 2.}, horizon_seconds=1., drain_seconds=2.,
                solver="milp", milp_time_limit_seconds=2., milp_max_iterations=32,
                milp_capacity_witness=False, route_grid=2)
    data = copy.deepcopy(c.catalog)
    data["performance_mode"] = data["memory_mode"] = MODE
    data[MODE] = dict(schema=SCHEMA, allow_cpu_fixture=True, input_order={"text": ["x"]},
                      scenarios={"1.0": "mean"}, deployments=[], max_requests=20, max_events=1000)
    for x in ({"g": 1}, {"s": 1}, {"g": 1, "s": 1}):
        context = "analytical-only-" + "-".join(sorted(x))
        provenance = dict(kind="cpu_fixture", split="training", sources=["hand-computable-software-test"])
        profiles = dict(schema="gsdeploy.segment_resource_profiles.v1", execution_context_sha256=context,
                        provenance=provenance, profiles=[], contexts={})
        lifecycle = dict(schema="gsdeploy.handler_lifecycle_profiles.v2", execution_context_sha256=context,
                         provenance=provenance, profiles=[])
        for model in sorted(x):
            profiles["profiles"].append(dict(id=model, family=model, task="text", pool_id="x",
                execution=dict(batch_size=1, backend="cpu_fixture", weight_reuse=False),
                segments=[dict(id="gpu", kind="gpu", duration_ms={"mean": 80. if model == "g" else 100.},
                               demand=dict(sm=.8 if model == "g" else .1,
                                           dram=.1 if model == "g" else .8, l2=.2))]))
            lifecycle["profiles"].append(dict(family=model, task="text", pool_id="x", permit_scope="not_applicable",
                scenarios={"mean": dict(parent_receipt_ms=5. if model == "g" else 10.,
                                        permit_release_ms=None, worker_next_ready_ms=20.)},
                evidence=["hand-computable-software-test"]))
        for active in itertools.product((False, True), repeat=len(x)):
            members = {f"{model}#0": dict(profile_id=model, segment_id="gpu")
                       for model, enabled in zip(sorted(x), active) if enabled}
            profiles["contexts"][context_key(members)] = dict(members=members, evidence="analytical-only",
                scenarios={"mean": dict(progress={a: .5 if len(members) == 2 else 1. for a in members},
                                        board_watts=20. + 40. * len(members), peak_memory_mib=20. * len(x))})
        data[MODE]["deployments"].append(dict(deployment=x, profiles=profiles, lifecycle=lifecycle,
            capacity_service_ms={m: {"text": 80. if m == "g" else 100.} for m in x},
            capacity_evidence="analytical-original-rho-service-basis", startup_joules=2. * len(x)))
    return replace(c, catalog=data)


def evaluate(c, x, route, **kwargs):
    return predict(c.catalog, x, route, c.workload, c.gpu_memory_mib,
                   c.latency_reference_ms, c.weights, **DeploymentSystem(c).context(), **kwargs)


class PaperFifoIntegrationTests(unittest.TestCase):
    def test_hand_calculated_overlap_reprices_remaining_work_and_receipt(self):
        c = fixture_config()
        p = evaluate(c, {"g": 1, "s": 1}, {"text": {"g": 1., "s": 1.}}, trace=((0., "text"), (0., "text")))
        self.assertTrue(p["eligible"], p["reason"])
        # Both slow to half speed. g ends at160ms; s then runs its last20ms
        # at full speed. Parent receipt is a separate5/10ms boundary.
        finishes = p["forecast"]["finishes"]
        self.assertEqual(finishes["0"]["handler_end_ns"], 160_000_000)
        self.assertEqual(finishes["0"]["finish_ns"], 165_000_000)
        self.assertEqual(finishes["1"]["handler_end_ns"], 180_000_000)
        self.assertEqual(finishes["1"]["finish_ns"], 190_000_000)
        self.assertAlmostEqual(p["per_task"]["text"]["mean_ms"], 177.5)
        self.assertFalse(p["native_eligible"])

    def test_fifo_next_ready_is_not_parent_receipt_and_idle_model_stays_resident(self):
        c = fixture_config()
        route = {"text": {"s": 2., "g": 0.}}
        p = evaluate(c, {"g": 1, "s": 1}, route, trace=((0., "text"), (0., "text")))
        self.assertEqual(p["forecast"]["finishes"]["1"]["start_ns"], 120_000_000)
        self.assertEqual(p["forecast"]["finishes"]["1"]["finish_ns"], 230_000_000)
        single = evaluate(c, {"s": 1}, {"text": {"s": 2.}}, trace=((0., "text"), (0., "text")))
        self.assertEqual(p["safe_memory_mib"], 40.)
        self.assertEqual(single["safe_memory_mib"], 20.)
        self.assertTrue(all(r["alias"] == "s#0" for r in p["request_mapping"]))

    def test_equal_time_future_burst_preserves_numeric_request_fifo_order(self):
        c = fixture_config()
        p = evaluate(c, {"g": 1}, {"text": {"g": 2.}}, trace=tuple((.1, "text") for _ in range(12)))
        rows = p["request_mapping"]
        starts = [p["forecast"]["finishes"][r["request_id"]]["start_ns"] for r in rows]
        self.assertEqual(starts, [100_000_000 + i * 100_000_000 for i in range(12)])

    def test_missing_co_run_context_and_stress_remain_unknown(self):
        c = fixture_config()
        entry = c.catalog[MODE]["deployments"][-1]
        entry["profiles"]["contexts"] = {k: v for k, v in entry["profiles"]["contexts"].items()
                                           if len(v["members"]) < 2}
        p = evaluate(c, {"g": 1, "s": 1}, {"text": {"g": 1., "s": 1.}}, trace=((0., "text"), (0., "text")))
        self.assertIsNone(p["U"])
        self.assertTrue(p["performance_unknown"])
        p = predict_scenarios(c.catalog, {"s": 1}, {"text": {"s": 2.}}, c.workload,
                              c.gpu_memory_mib, c.latency_reference_ms, c.weights, **DeploymentSystem(c).context())
        self.assertIsNotNone(p["U"])
        self.assertEqual(p["scenario_U_values"][1:], [None, None])
        self.assertFalse(p["stress_all_eligible"])
        c = fixture_config()
        c.catalog[MODE]["deployments"][1]["capacity_service_ms"] = {}
        p = evaluate(c, {"s": 1}, {"text": {"s": 2.}})
        self.assertTrue(p["performance_unknown"])
        self.assertIn("UNKNOWN_ORIGINAL_RHO_SERVICE", p["reason"])

    def test_explicit_whole_request_resource_admission_uses_original_fifo(self):
        c = fixture_config()
        entry = c.catalog[MODE]["deployments"][-1]
        entry["resources"] = dict(schema="gsdeploy.resource_configuration.v1", mode="request_admission",
            device="0", allocations={"g#0": {"max_inflight": 1}, "s#0": {"max_inflight": 1}},
            groups={"gpu": dict(aliases=["s#0", "g#0"], max_active=1)})
        for p in entry["lifecycle"]["profiles"]:
            p["permit_scope"] = "request_admission"
            p["scenarios"]["mean"]["permit_release_ms"] = 7. if p["family"] == "g" else 12.
        p = evaluate(c, {"g": 1, "s": 1}, {"text": {"g": 1., "s": 1.}}, trace=((0., "text"), (0., "text")))
        self.assertTrue(p["eligible"], p["reason"])
        self.assertEqual(p["forecast"]["finishes"]["0"]["finish_ns"], 85_000_000)
        self.assertEqual(p["forecast"]["finishes"]["1"]["start_ns"], 87_000_000)
        self.assertEqual(p["forecast"]["finishes"]["1"]["finish_ns"], 197_000_000)

    def test_shared_predictor_cache_identity_and_defensive_copy(self):
        c = fixture_config()
        s = session(c.catalog)
        before = s.hits
        route = {"text": {"s": 2.}}
        p = evaluate(c, {"s": 1}, route)
        p["request_mapping"][0]["alias"] = "tampered"
        again = evaluate(c, {"s": 1}, route)
        self.assertEqual(again["request_mapping"][0]["alias"], "s#0")
        self.assertGreater(s.hits, before)
        changed = copy.deepcopy(c.catalog)
        changed["safe_factor"] = 1.1
        self.assertIsNot(session(changed), s)
        self.assertIs(session(s.data), s)
        with self.assertRaises(TypeError):
            s.data["safe_factor"] = 2.

    def test_cache_byte_budget_and_forecast_budget_do_not_truncate_requests(self):
        c = fixture_config()
        c.catalog[MODE]["cache_max_bytes"] = 100
        s = session(c.catalog)
        p = evaluate(c, {"s": 1}, {"text": {"s": 2.}})
        self.assertTrue(p["eligible"])
        self.assertEqual(s.cache_bytes, 0)
        self.assertEqual(len(s.cache), 0)
        c.catalog[MODE]["max_requests"] = 1
        p = evaluate(c, {"s": 1}, {"text": {"s": 2.}})
        self.assertTrue(p["performance_unknown"])
        self.assertEqual(p["offered_requests"], 2)
        self.assertIn("NO_TRUNCATION", p["reason"])

    def test_actual_public_milp_and_dfs_use_event_forecast_and_match_oracle(self):
        c = fixture_config()
        oracle = []
        for x in ({"g": 1}, {"s": 1}, {"g": 1, "s": 1}):
            routes = ([{"text": {next(iter(x)): 2.}}] if len(x) == 1 else
                      [{"text": {"g": 2 * f, "s": 2 * (1-f)}} for f in (0., .5, 1.)])
            for route in routes:
                p = evaluate(c, x, route)
                if p["eligible"]:
                    oracle.append(p["U"])
        # Poison old standalone bounds: the new mode must never use them.
        for m in c.catalog["models"]:
            c.catalog["models"][m]["service_time_ms"]["text"] = 1_000_000.
            c.catalog["memory_table_mib"][m]["1"] = 100_000.
        with patch("gsdeploy.prediction.resolve_service", side_effect=AssertionError("old scalar path")):
            for algorithm in ("milp", "dfs"):
                result = DeploymentSystem(replace(c, solver=algorithm)).plan()
                self.assertIsNotNone(result["winner"], result["status"])
                self.assertTrue(result["optimality_certified"], result["status"])
                self.assertAlmostEqual(result["winner"]["score"], max(oracle))
                self.assertIn("forecast", result["winner"]["prediction"])
                self.assertTrue(result["domain"]["coupled_bounds"]["complete_prediction_implemented"])

    def test_unsupported_stress_is_not_a_robust_infeasibility_certificate(self):
        c = replace(fixture_config(), search_policy="robust")
        for algorithm in ("milp", "dfs"):
            result = DeploymentSystem(replace(c, solver=algorithm)).plan()
            self.assertIsNone(result["winner"])
            self.assertFalse(result["optimality_certified"])
            self.assertGreater(result["metrics"]["unknown_candidates"], 0)

    def test_original_rho_is_enforced_and_native_execution_is_separate(self):
        c = fixture_config()
        c.catalog[MODE]["deployments"][1]["capacity_service_ms"]["s"]["text"] = 500.
        p = evaluate(c, {"s": 1}, {"text": {"s": 2.}})
        self.assertEqual(p["max_rho"], 1.)
        self.assertFalse(p["constraints_eligible"])
        with self.assertRaisesRegex(ValueError, "rho"):
            c.validate_plan({"s": 1}, {"text": {"s": 2.}})
        with self.assertRaisesRegex(ValueError, "native execution qualification"):
            DeploymentSystem(c).run({}, "not-opened")

    def test_fixture_opt_in_and_quality_input_alignment_are_required(self):
        c = fixture_config()
        data = copy.deepcopy(c.catalog)
        data[MODE]["allow_cpu_fixture"] = False
        with self.assertRaisesRegex(ValueError, "CPU fixture"):
            replace(c, catalog=data)
        data = copy.deepcopy(c.catalog)
        data["quality"].update(mode="per_sample_replay", input_order={"text": ["other"]})
        with self.assertRaisesRegex(ValueError, "same input order"):
            replace(c, catalog=data)
        data = copy.deepcopy(c.catalog)
        data[MODE]["deployments"][0]["profiles"]["profiles"][0]["execution"]["batch_size"] = 2
        with self.assertRaisesRegex(ValueError, "batch-aware"):
            replace(c, catalog=data)


class PaperAlgorithmTests(unittest.TestCase):
    def test_usher_complementary_grouping_and_real_configuration_budgets(self):
        demands = {"c": dict(compute_fraction=.8, memory_fraction=.1),
                   "m": dict(compute_fraction=.1, memory_fraction=.8),
                   "c2": dict(compute_fraction=.7, memory_fraction=.1)}
        groups = UsherResourcePlanner.groups(demands, max_group_size=2)
        self.assertIn(("c", "m"), groups)
        options = {m: [dict(replicas=1, compute_fraction=v["compute_fraction"],
                            memory_mib=100*v["memory_fraction"], evidence="cpu-fixture-only")]
                   for m, v in list(demands.items())[:2]}
        devices = {"0": dict(compute_capacity=1., memory_mib=100., cost=1.)}
        result = UsherResourcePlanner.enumerate_configurations(options, devices)
        proposal = result["configurations"][0]["placement"]
        self.assertEqual(proposal["status"], "PLACEMENT_PROPOSAL")
        self.assertAlmostEqual(proposal["free"]["0"][0], .1)
        self.assertEqual(len(proposal["placements"]), 2)
        self.assertFalse(proposal["physical_feasibility_certified"])
        options["c"] *= 3
        limited = UsherResourcePlanner.enumerate_configurations(options, devices, max_configurations=2)
        self.assertEqual(limited["original_combinations"], 3)
        self.assertFalse(limited["domain_complete"])

    def test_lp_obeys_original_capacity_and_does_not_quantize(self):
        c = fixture_config()
        entry = c.catalog[MODE]["deployments"][-1]
        entry["capacity_service_ms"] = {"g": {"text": 700.}, "s": {"text": 700.}}
        p = fixed_deployment_lp(c, {"g": 1, "s": 1})
        self.assertIsNotNone(p)
        row = p["routing"]["text"]
        self.assertAlmostEqual(sum(row.values()), 2.)
        self.assertTrue(all(q * .7 <= .8 + 1e-9 for q in row.values()))
        self.assertTrue(any(not math.isclose(q / 2 * 2, round(q / 2 * 2)) for q in row.values()))

    def test_j3o_adaptation_keeps_seed_and_charges_total_budget(self):
        c = fixture_config()
        result = J3OInspiredOptimizer(c, max_evaluations=24, time_limit_seconds=3.,
                                      seed_seconds=1., max_rounds=1).solve()
        self.assertIsNotNone(result["winner"], result["status"])
        self.assertGreaterEqual(result["winner"]["score"], result["seed_winner"]["score"])
        self.assertFalse(result["optimality_certified"])
        self.assertLessEqual(result["full_prediction_calls"], 24)
        self.assertGreater(result["lp_calls"], 0)
        self.assertIn("deployment_ordering", result)

    def test_tiny_budget_and_unknown_resource_demand_remain_explicit(self):
        c = fixture_config()
        for entry in c.catalog[MODE]["deployments"]:
            for profile in entry["profiles"]["profiles"]:
                profile["segments"][0]["demand"] = dict(sm=None, dram=None, l2=None)
        result = J3OInspiredOptimizer(c, max_evaluations=1, time_limit_seconds=1.).solve()
        self.assertLessEqual(result["full_prediction_calls"], 1)
        self.assertEqual(len(result["deployment_ordering"]), 3)
        self.assertTrue(all(not r["resource_demand_known"] for r in result["deployment_ordering"]))
        from gsdeploy.io import dumps
        dumps(result)  # Unknown priority is JSON null, never Infinity.


if __name__ == "__main__":
    unittest.main()
