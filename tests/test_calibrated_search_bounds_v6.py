"""Independent Cartesian checks of frozen-v1 six-term search bounds.

Every value here is an explicit synthetic CPU fixture, never GPU evidence.
"""

import copy
import math
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

from gsdeploy.calibration import _baseline_identity, fit_calibrated_surrogate
from gsdeploy.config import SystemConfig
from gsdeploy.milp import JointMilpOptimizer
from gsdeploy.optimizer import UtilityOptimizer
from gsdeploy.prediction import predict_scenarios
from gsdeploy.quality import quality_digest
from gsdeploy.surrogate_bounds import surrogate_bounds
from tests.test_calibrated_surrogate import calibration_fixture
from tests.test_milp import grid_plans


def seal(data):
    spec = data["calibrated_surrogate"]
    spec["version_sha256"] = quality_digest({k: v for k, v in spec.items()
                                             if k != "version_sha256"})


def search_fixture():
    original, imported, sources = calibration_fixture()
    with patch("gsdeploy.calibration._import_source",
               side_effect=lambda source, accepted, phases=None: imported[source["id"]]):
        raw, _ = fit_calibrated_surrogate(original, sources, [str(i) for i in range(6)],
                                         ["6", "7"], ["CPU fixture"])
    data = raw["catalog"]
    services = {"generalist": {"text": 140., "image": 100.},
                "specialist": {"text": 90., "image": 180.}}
    for m, per_task in services.items():
        data["models"][m].update(supported_tasks=list(per_task), executable_tasks=list(per_task),
                                  service_time_ms={t: 9999. for t in per_task})
        data["capacity_factors"][m]["2"] = 2.
        data["memory_table_mib"][m]["2"] = 1.8
        data["task_replica_profiles"][m] = {
            t: {str(n): {"mean_service_ms": value * (1.35 if n == 2 else 1.)}
                for n in (1, 2)} for t, value in per_task.items()
        }
        data["quality"]["paths"][m]["image"] = copy.deepcopy(data["quality"]["paths"][m]["text"])
        data["quality"]["paths"][m]["image"]["metric"] = "accuracy"
    spec = data["calibrated_surrogate"]
    spec.update(baseline_identities=_baseline_identity(data), workload={"text": 4., "image": 3.},
                quality_document_sha256=quality_digest(data["quality"]),
                power_coefficients=[30., 50., 15.], startup_joules_per_worker=3.,
                training_deployment_memory={})
    seal(data)
    raw.update(workload=spec["workload"], horizon_seconds=3.,
               latency_reference_ms={"text": 400., "image": 400.})
    return SystemConfig(**raw)


def context(c, policy="nominal"):
    return dict(horizon=c.horizon_seconds, seed=c.random_seed, arrival_process=c.arrival_process,
                drain_seconds=c.drain_seconds, energy_reference_watts=c.energy_reference_watts,
                startup_amortization_seconds=c.startup_amortization_seconds,
                milp_capacity_witness=False, search_policy=policy)


def optimizer(cls, c, policy="nominal", **overrides):
    options = context(c, policy)
    options.update(overrides)
    return cls(c.catalog, c.workload, c.gpu_memory_mib, c.latency_reference_ms,
               c.weights, **options)


def prediction(c, x, routing):
    ctx = context(c)
    ctx.pop("search_policy")
    ctx.pop("milp_capacity_witness")
    return predict_scenarios(c.catalog, x, routing, c.workload, c.gpu_memory_mib,
                             c.latency_reference_ms, c.weights, **ctx)


class CalibratedSearchBoundsTests(unittest.TestCase):
    def test_exact_energy_decomposition_includes_busy_idle_workers_and_startup(self):
        c = replace(search_fixture(), startup_amortization_seconds=9.)
        bounds = surrogate_bounds(c.catalog, c.workload, **context(c))
        self.assertIsNotNone(bounds)
        for x, routing in grid_plans(c, 3):
            p = prediction(c, x, routing)
            exact = (bounds.constant_energy + bounds.state_energy(sum(x.values()))
                     + math.fsum(bounds.route_energy(t, x, row) for t, row in routing.items()))
            self.assertAlmostEqual(exact, p["E_norm"], places=12)

    def test_every_partial_branch_and_milp_column_bound_is_valid(self):
        c = search_fixture()
        for policy in ("nominal", "robust"):
            dfs, milp = optimizer(UtilityOptimizer, c, policy), optimizer(JointMilpOptimizer, c, policy)
            milp.start = time.monotonic()
            checked = 0
            for x, routing in grid_plans(c, 3):
                p = prediction(c, x, routing)
                score = p["U"] if policy == "nominal" else p["scenario_U_low"]
                if score is None:
                    continue
                checked += 1
                memory = math.fsum(milp.state_memory(m, n) for m, n in x.items())
                for partial in ({}, {"text": routing["text"]}, {"image": routing["image"]}, routing):
                    self.assertGreaterEqual(dfs.upper(x, partial, memory) + 1e-9, score)
                column_floor = math.fsum(milp.floor(t, {m: x[m] for m in row}, row)
                                          for t, row in routing.items())
                floor = (column_floor + c.weights["M"] * memory / c.gpu_memory_mib
                         + c.weights["E"] * milp.surrogate.state_energy(sum(x.values())))
                self.assertLessEqual(floor, -score + 1e-9)
                for m, n in x.items():
                    rho = math.fsum(row.get(m, 0) * milp.family_service(m, t, n) / 1000 / n
                                    for t, row in routing.items())
                    self.assertAlmostEqual(rho, p["rho"][m], places=12)
            self.assertGreater(checked, 0)

    def test_missing_path_keeps_other_bounds_but_cannot_become_measured_or_certified(self):
        c = search_fixture()
        # Define a separate sparse synthetic predictor, with its own frozen
        # baseline identity. Existing real evidence is never rewritten here.
        del c.catalog["task_replica_profiles"]["generalist"]["text"]["1"]
        c.catalog["calibrated_surrogate"]["baseline_identities"] = _baseline_identity(c.catalog)
        seal(c.catalog)
        bounds = surrogate_bounds(c.catalog, c.workload, **context(c))
        self.assertIsNotNone(bounds)
        self.assertEqual(bounds.family_service("generalist", "text", 1), 0.)
        self.assertGreater(bounds.family_service("generalist", "image", 1), 0.)
        self.assertGreater(bounds.route_energy("image", {"generalist": 1}, {"generalist": 3.}), 0.)
        self.assertGreater(bounds.constant_energy, 0.)
        for cls in (UtilityOptimizer, JointMilpOptimizer):
            search = optimizer(cls, c)
            self.assertTrue(search.separable_service)
            result = search.solve(deployment={"generalist": 1}, K=2)
            self.assertFalse(result["optimality_certified"])
            self.assertIsNone(result["winner"])
            self.assertGreater(result["metrics"]["unknown_candidates"], 0)
            self.assertIsNotNone(result["utility_upper_bound"])

    def test_milp_dfs_and_cartesian_oracle_agree_for_six_terms_and_replica_states(self):
        for process in ("deterministic_periodic", "poisson"):
            c = replace(search_fixture(), arrival_process=process)
            for policy in ("nominal", "robust"):
                oracle = []
                for x, routing in grid_plans(c, 3):
                    p = prediction(c, x, routing)
                    if p["constraints_eligible"] and (policy != "robust" or p["stress_constraints_valid"]):
                        score = p["U"] if policy == "nominal" else p["scenario_U_low"]
                        if score is not None:
                            oracle.append(score)
                self.assertTrue(oracle)
                results = [optimizer(cls, c, policy).solve(K=3)
                           for cls in (UtilityOptimizer, JointMilpOptimizer)]
                results.append(optimizer(UtilityOptimizer, c, policy, dfs_bound_pruning=False).solve(K=3))
                for result in results:
                    self.assertTrue(result["optimality_certified"], result["status"])
                    self.assertAlmostEqual(result["winner"]["score"], max(oracle), places=9)
                    self.assertIsNotNone(result["domain"]["surrogate_bounds_version"])
                    self.assertIn("exact nonnegative", result["domain"]["energy_bound"])

    def test_plain_dfs_evaluates_every_capacity_admissible_identity_without_utility_bounds(self):
        c = search_fixture()
        expected_calls = 0
        scores = []
        for x, routing in grid_plans(c, 3):
            p = prediction(c, x, routing)
            if p["max_rho"] <= .8 + 1e-8 and p["safe_memory_mib"] <= c.gpu_memory_mib:
                expected_calls += 1
            if p["constraints_eligible"] and p["U"] is not None:
                scores.append(p["U"])
        result = optimizer(UtilityOptimizer, c, dfs_bound_pruning=False).solve(K=3)
        self.assertTrue(result["optimality_certified"])
        self.assertEqual(result["metrics"]["full_prediction_calls"], expected_calls)
        self.assertEqual(result["metrics"]["utility_bound_evaluations"], 0)
        self.assertEqual(result["metrics"]["bound_pruned"], 0)
        self.assertEqual(result["metrics"]["quality_pattern_evaluations"], 0)
        self.assertEqual(result["domain"]["search_strategy"], "exhaustive_dfs_without_utility_pruning")
        self.assertAlmostEqual(result["winner"]["score"], max(scores), places=12)

    def test_busy_specialist_actively_spills_to_generalist(self):
        c = search_fixture()
        c = replace(c, workload={"text": 8.}, latency_reference_ms={"text": 400.})
        c.catalog["calibrated_surrogate"]["workload"] = c.workload
        seal(c.catalog)
        fixed = {"generalist": 1, "specialist": 1}
        # Each model alone exceeds rho .8 at this offered rate. A split is
        # feasible in the same finite domain and must be selected actively.
        for cls in (UtilityOptimizer, JointMilpOptimizer):
            result = optimizer(cls, c).solve(deployment=fixed, K=3)
            self.assertTrue(result["optimality_certified"], result["status"])
            row = result["winner"]["routing"]["text"]
            self.assertEqual(set(row), set(fixed))
            self.assertGreater(row["generalist"], 0)
            self.assertLess(row["specialist"], c.workload["text"])

    def test_invalid_version_status_context_and_baselines_cannot_supply_bounds(self):
        c = search_fixture()
        cases = []
        for key, value in (("status", "VALIDATION_FAILED"), ("status", "FITTED_UNVALIDATED"),
                           ("schema", "gsdeploy.calibrated_surrogate.v2"),
                           ("memory_multiplier", .5), ("power_coefficients", [1., -1., 1.])):
            data = copy.deepcopy(c.catalog)
            data["calibrated_surrogate"][key] = value
            seal(data)
            cases.append(data)
        data = copy.deepcopy(c.catalog)
        data["calibrated_surrogate"]["version_sha256"] = "changed"
        cases.append(data)
        for field in ("gpu_uuid", "runtime_checkpoint_sha256"):
            data = copy.deepcopy(c.catalog)
            data["measurement_context"][field] = "changed"
            cases.append(data)
        data = copy.deepcopy(c.catalog)
        data["task_replica_profiles"]["generalist"]["text"]["1"]["mean_service_ms"] *= .5
        cases.append(data)
        for data in cases:
            self.assertIsNone(surrogate_bounds(data, c.workload, **context(c)))
            for cls in (UtilityOptimizer, JointMilpOptimizer):
                instance = cls(data, c.workload, c.gpu_memory_mib, c.latency_reference_ms,
                               c.weights, **context(c))
                self.assertFalse(instance.separable_service)
                self.assertTrue(instance.profiled_memory)


if __name__ == "__main__":
    unittest.main()
