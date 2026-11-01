"""Frozen per-input v2 bound/oracle regressions, using synthetic CPU fits only."""

import copy
import math
import json
from pathlib import Path
import time
import unittest
from dataclasses import replace

from gsdeploy.calibration_v2 import MODE, SCHEMA, _sealed, replay_input_ids
from gsdeploy.milp import JointMilpOptimizer
from gsdeploy.config import SystemConfig
from gsdeploy.optimizer import UtilityOptimizer
from gsdeploy.surrogate_bounds import surrogate_bounds
from gsdeploy.trace import make_trace
from tests.test_calibrated_replay_v2 import fitted_replay_fixture
from tests.test_calibrated_search_bounds_v6 import context, optimizer, prediction
from tests.test_milp import grid_plans


class ReplaySearchBoundsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fitted = fitted_replay_fixture()

    def config(self, **changes):
        return replace(self.fitted, catalog=copy.deepcopy(self.fitted.catalog),
                       horizon_seconds=changes.pop("horizon_seconds", 20.), **changes)

    def test_input_minimum_pool_mean_and_power_baseline_remain_distinct(self):
        c = self.config(horizon_seconds=2.)
        for seed in range(100):
            trace = make_trace(c.workload, c.horizon_seconds, seed)
            if (sorted(t for _, t in trace) == sorted(c.workload)
                    and replay_input_ids(c.catalog, trace, seed) == ["a"] * len(trace)):
                c = replace(c, random_seed=seed)
                break
        else:
            self.fail("Synthetic fixture needs a seed selecting short inputs")
        bounds = surrogate_bounds(c.catalog, c.workload, **context(c))
        self.assertIsNotNone(bounds)
        self.assertAlmostEqual(bounds.family_service("generalist", "text", 1), 66.)
        self.assertAlmostEqual(bounds.pressure_service("generalist", "text", 1), 165.)
        self.assertAlmostEqual(bounds.services["generalist", "text", 1], 150.)
        x = {"generalist": 1, "specialist": 1}
        rows = {"text": {"generalist": c.workload["text"]},
                "image": {"specialist": c.workload["image"]}}
        p = prediction(c, x, rows)
        self.assertIsNotNone(p["U"])
        # Pool means exceed these actual short requests, so substituting means
        # for the service floor would produce an invalid latency bound.
        wrong_mean_cost = math.fsum(
            c.workload[t] / sum(c.workload.values()) / c.latency_reference_ms[t]
            * (c.weights["L"] + c.weights["P"])
            * bounds.pressure_service(m, t, 1)
            for t, row in rows.items() for m in row
        )
        actual_latency_cost = c.weights["L"] * p["L_norm"] + c.weights["P"] * p["P_norm"]
        self.assertGreater(wrong_mean_cost, actual_latency_cost)
        dfs = optimizer(UtilityOptimizer, c)
        memory = c.catalog[MODE]["memory_multiplier"] * sum(c.catalog["memory_table_mib"][m]["1"] for m in x)
        self.assertGreaterEqual(dfs.upper(x, rows, memory) + 1e-9, p["U"])

    def test_all_partial_and_column_bounds_hold_for_variable_input_fifo(self):
        checked = 0
        for process in ("deterministic_periodic", "poisson"):
            c = self.config(arrival_process=process)
            for policy in ("nominal", "robust"):
                dfs = optimizer(UtilityOptimizer, c, policy)
                milp = optimizer(JointMilpOptimizer, c, policy)
                milp.start = time.monotonic()
                for x, rows in grid_plans(c, 3):
                    p = prediction(c, x, rows)
                    score = p["U"] if policy == "nominal" else p["scenario_U_low"]
                    if score is None:
                        continue
                    checked += 1
                    memory = math.fsum(milp.state_memory(m, n) for m, n in x.items())
                    for partial in ({}, {"text": rows["text"]}, {"image": rows["image"]}, rows):
                        self.assertGreaterEqual(dfs.upper(x, partial, memory) + 1e-9, score)
                    column_cost = math.fsum(milp.floor(t, {m: x[m] for m in row}, row)
                                            for t, row in rows.items())
                    total_floor = (column_cost + c.weights["M"] * memory / c.gpu_memory_mib
                                   + c.weights["E"] * milp.surrogate.state_energy(sum(x.values())))
                    self.assertLessEqual(total_floor, -score + 1e-9)
                    energy = (milp.surrogate.constant_energy
                              + milp.surrogate.state_energy(sum(x.values()))
                              + math.fsum(milp.surrogate.route_energy(t, x, row) for t, row in rows.items()))
                    self.assertAlmostEqual(energy, p["E_norm"], places=12)
                    for m, n in x.items():
                        rho = math.fsum(row.get(m, 0.) * milp.pressure_service(m, t, n) / 1000 / n
                                        for t, row in rows.items())
                        self.assertAlmostEqual(rho, p["rho"][m], places=12)
        self.assertGreater(checked, 0)

    def test_milp_bounded_and_plain_dfs_match_independent_cartesian_oracle(self):
        for process in ("deterministic_periodic", "poisson"):
            c = self.config(arrival_process=process)
            for policy in ("nominal", "robust"):
                scores = []
                for x, rows in grid_plans(c, 3):
                    p = prediction(c, x, rows)
                    if p["constraints_eligible"] and (policy != "robust" or p["stress_constraints_valid"]):
                        scores.append(p["U"] if policy == "nominal" else p["scenario_U_low"])
                self.assertTrue(scores)
                solvers = [optimizer(JointMilpOptimizer, c, policy), optimizer(UtilityOptimizer, c, policy),
                           optimizer(UtilityOptimizer, c, policy, dfs_bound_pruning=False)]
                for solver in solvers:
                    result = solver.solve(K=3)
                    self.assertTrue(result["optimality_certified"], result["status"])
                    self.assertAlmostEqual(result["winner"]["score"], max(scores), places=9)
                    self.assertEqual(result["domain"]["surrogate_bounds_schema"], SCHEMA)
                    self.assertIn("minimum service", result["domain"]["service_bound"])
                    self.assertIn("pool-mean", result["domain"]["rho_bound"])

    def test_task_columns_use_pool_mean_for_rho_not_the_latency_minimum(self):
        c = self.config()
        milp = optimizer(JointMilpOptimizer, c)
        milp.start = time.monotonic()
        columns = milp.columns({m: [0, 1, 2] for m in c.catalog["models"]}, 3)
        self.assertTrue(columns)
        for column in columns:
            for model, q in column["routing"].items():
                n = column["deployment"][model]
                mean = milp.pressure_service(model, column["task"], n)
                minimum = milp.family_service(model, column["task"], n)
                self.assertGreater(mean, minimum)
                self.assertAlmostEqual(column["pressure"][model], q * mean / 1000 / n)

    def test_per_sample_quality_and_service_replay_match_oracle_and_partial_bounds(self):
        c = replace(fitted_replay_fixture(quality_mode="per_sample_replay"), horizon_seconds=20.)
        milp, dfs = optimizer(JointMilpOptimizer, c), optimizer(UtilityOptimizer, c)
        milp.start = time.monotonic()
        scores, distinct_quality = [], set()
        for x, rows in grid_plans(c, 3):
            p = prediction(c, x, rows)
            self.assertIsNotNone(p["Q_norm"])
            distinct_quality.add(round(p["Q_norm"], 12))
            if p["constraints_eligible"] and p["U"] is not None:
                scores.append(p["U"])
                memory = sum(milp.state_memory(m, n) for m, n in x.items())
                for partial in ({}, {"text": rows["text"]}, {"image": rows["image"]}, rows):
                    self.assertGreaterEqual(dfs.upper(x, partial, memory) + 1e-9, p["U"])
        self.assertGreater(len(distinct_quality), 1)
        for cls, options in ((JointMilpOptimizer, {}), (UtilityOptimizer, {}),
                             (UtilityOptimizer, {"dfs_bound_pruning": False})):
            result = optimizer(cls, c, **options).solve(K=3)
            self.assertTrue(result["optimality_certified"], result["status"])
            self.assertAlmostEqual(result["winner"]["score"], max(scores), places=9)
            self.assertEqual(result["winner"]["prediction"]["quality_evidence"]["per_task"]["text"]["kind"],
                             "offline_per_sample_replay")

    def test_busy_energy_uses_original_standalone_mean_feature(self):
        c = self.config()
        bounds = surrogate_bounds(c.catalog, c.workload, **context(c))
        # Exercise the mathematical feature with a positive coefficient even
        # if NNLS makes that coefficient zero in the fitted CPU fixture. This
        # changes only a local bound object, never a formal fitted config.
        bounds.busy = 13.
        x = {"generalist": 1, "specialist": 2}
        actual = (bounds.route_energy("text", x, {"generalist": .4})
                  + bounds.route_energy("image", x, {"specialist": .3}))
        expected = 13. * (.4 * .150 + .3 * .198) / c.energy_reference_watts
        self.assertAlmostEqual(actual, expected, places=12)

    def test_long_trace_minimum_multiset_bound_is_safe(self):
        c = self.config(horizon_seconds=4000.)
        x = {"generalist": 2, "specialist": 1}
        rows = {t: {"generalist": q / 3, "specialist": q * 2 / 3} for t, q in c.workload.items()}
        p = prediction(c, x, rows)
        self.assertIsNotNone(p["scenario_U_low"])
        milp = optimizer(JointMilpOptimizer, c, "robust")
        milp.start = time.monotonic()
        memory = sum(milp.state_memory(m, n) for m, n in x.items())
        cost = (sum(milp.floor(t, x, row) for t, row in rows.items())
                + c.weights["M"] * memory / c.gpu_memory_mib
                + c.weights["E"] * milp.surrogate.state_energy(sum(x.values())))
        self.assertLessEqual(cost, -p["scenario_U_low"] + 1e-9)
        self.assertEqual(milp.metrics["isolated_queue_bounds"], 0)
        self.assertGreater(milp.metrics["service_pattern_evaluations"], 0)
        dfs = optimizer(UtilityOptimizer, c, "robust")
        self.assertGreaterEqual(dfs.upper(x, rows, memory) + 1e-9, p["scenario_U_low"])

    def test_exact_input_bound_is_stronger_than_pool_minimum_for_both_trace_lengths(self):
        for horizon in (20., 4000.):
            for process in ("deterministic_periodic", "poisson"):
                c = self.config(horizon_seconds=horizon, arrival_process=process)
                x = {"generalist": 2, "specialist": 1}
                rows = {t: {"generalist": q / 3, "specialist": q * 2 / 3} for t, q in c.workload.items()}
                for policy in ("nominal", "robust"):
                    dfs, old_dfs = optimizer(UtilityOptimizer, c, policy), optimizer(UtilityOptimizer, c, policy)
                    milp, old_milp = optimizer(JointMilpOptimizer, c, policy), optimizer(JointMilpOptimizer, c, policy)
                    old_dfs.task_input_ids = old_milp.task_input_ids = None
                    milp.start = old_milp.start = time.monotonic()
                    p = prediction(c, x, rows)
                    score = p["U"] if policy == "nominal" else p["scenario_U_low"]
                    self.assertIsNotNone(score)
                    for task, row in rows.items():
                        self.assertGreater(dfs.service_floor(x, task, row), old_dfs.service_floor(x, task, row))
                        self.assertGreater(milp.floor(task, x, row), old_milp.floor(task, x, row))
                    memory = sum(milp.state_memory(m, n) for m, n in x.items())
                    for partial in ({}, {"text": rows["text"]}, {"image": rows["image"]}, rows):
                        self.assertGreaterEqual(dfs.upper(x, partial, memory) + 1e-9, score)
                    cost = (sum(milp.floor(t, x, row) for t, row in rows.items())
                            + c.weights["M"] * memory / c.gpu_memory_mib
                            + c.weights["E"] * milp.surrogate.state_energy(sum(x.values())))
                    self.assertLessEqual(cost, -score + 1e-9)

    def test_request_lookup_matches_frozen_samples_and_unknown_path_stays_zero(self):
        c = self.config()
        bounds = surrogate_bounds(c.catalog, c.workload, **context(c))
        trace = make_trace(c.workload, c.horizon_seconds, c.random_seed, c.arrival_process)
        per_task = bounds.task_input_ids(trace, c.random_seed)
        selected = replay_input_ids(c.catalog, trace, c.random_seed)
        for task in c.workload:
            self.assertEqual(per_task[task], [sample for (_, t), sample in zip(trace, selected) if t == task])
            for sample in c.catalog["quality"]["input_order"][task]:
                self.assertEqual(bounds.request_service("generalist", task, 1, sample),
                    c.catalog[MODE]["input_baselines"]["generalist"][task]["1"]["samples"][sample]["median_service_ms"]
                    * c.catalog[MODE]["service_multiplier"])
        self.assertEqual(bounds.request_service("missing", "text", 1, "a"), 0.)

    def test_completed_real_predictor_candidate_partial_bounds(self):
        study = Path(__file__).resolve().parents[1] / "artifacts/static-study-gpu-20261022"
        source = study / "surrogate-v2/system-config.json"
        if not source.exists():
            self.skipTest("Real completed v2 artifacts are not distributed with unit tests")
        actual = SystemConfig(**json.loads(source.read_text()))
        sources = json.loads((study / "surrogate-v2/sources.json").read_text())
        plans = [json.loads(Path(row["path"]).read_text())["plan"] for row in sources]
        # Add an actual measured replica state, to exercise multiple alias queues.
        self.assertIn("2", actual.catalog["capacity_factors"]["generalist_2"])
        plans.append({"deployment": {"generalist_2": 2},
                      "routing": {t: {"generalist_2": q} for t, q in actual.workload.items()}})
        checked = 0
        for process in ("deterministic_periodic", "poisson"):
            c = replace(actual, horizon_seconds=600., arrival_process=process)
            solvers = {policy: (optimizer(UtilityOptimizer, c, policy), optimizer(JointMilpOptimizer, c, policy))
                       for policy in ("nominal", "robust")}
            for _, milp in solvers.values():
                milp.start = time.monotonic()
            for plan in plans:
                x, rows = plan["deployment"], plan["routing"]
                p = prediction(c, x, rows)
                for policy, (dfs, milp) in solvers.items():
                    # Costs remain meaningful for constraint-ineligible complete
                    # candidates too; formal eligibility remains unchanged.
                    score = (p["diagnostic_partial_U"] if policy == "nominal"
                             else min(p["scenario_diagnostic_partial_U"]))
                    memory = sum(milp.state_memory(m, n) for m, n in x.items())
                    tasks = list(rows)
                    for mask in range(1 << len(tasks)):
                        partial = {t: rows[t] for i, t in enumerate(tasks) if mask & (1 << i)}
                        self.assertGreaterEqual(dfs.upper(x, partial, memory) + 1e-8, score)
                        checked += 1
                    cost = (sum(milp.floor(t, {m: x[m] for m in row}, row) for t, row in rows.items())
                            + c.weights["M"] * memory / c.gpu_memory_mib
                            + c.weights["E"] * milp.surrogate.state_energy(sum(x.values())))
                    self.assertLessEqual(cost, -score + 1e-8)
        self.assertEqual(checked, 11 * 2 * 2 * 16)

    def test_sparse_input_path_preserves_known_bounds_and_unknown_complete_status(self):
        c = self.config()
        # Separate sparse synthetic predictor; no historical execution config
        # or measurement record is changed by this counterexample.
        spec = c.catalog[MODE]
        del spec["input_baselines"]["generalist"]["text"]["1"]["samples"]["a"]
        # Rebuild this separate synthetic fixture's full baseline/protocol
        # identity. Merely resealing a tampered spec must fail the real gate.
        frozen = {**spec["frozen_baseline_contract"], "input_baselines": spec["input_baselines"],
                  "sources": spec["baseline_sources"]}
        spec["frozen_baselines_sha256"] = _sealed(frozen)
        spec["protocol"]["frozen_baselines_sha256"] = spec["frozen_baselines_sha256"]
        spec["protocol"]["sha256"] = _sealed(spec["protocol"])
        spec["version_sha256"] = _sealed(spec, "version_sha256")
        bounds = surrogate_bounds(c.catalog, c.workload, **context(c))
        self.assertIsNotNone(bounds)
        self.assertEqual(bounds.family_service("generalist", "text", 1), 0.)
        self.assertEqual(bounds.pressure_service("generalist", "text", 1), 0.)
        # Even the known sample on an incomplete path stays conservative zero;
        # the domain and complete predictor retain that unknown candidate.
        self.assertEqual(bounds.request_service("generalist", "text", 1, "b"), 0.)
        self.assertGreater(bounds.family_service("generalist", "image", 1), 0.)
        for cls in (UtilityOptimizer, JointMilpOptimizer):
            result = optimizer(cls, c).solve(deployment={"generalist": 1}, K=3)
            self.assertFalse(result["optimality_certified"])
            self.assertIsNone(result["winner"])
            self.assertGreater(result["metrics"]["unknown_candidates"], 0)
            self.assertIsNotNone(result["utility_upper_bound"])

    def test_unvalidated_failed_or_changed_v2_never_activates_formal_bounds(self):
        for mutation in ("unvalidated", "failed", "context", "quality", "schema", "version"):
            c = self.config()
            spec = c.catalog[MODE]
            if mutation in {"unvalidated", "failed"}:
                spec["status"] = "FITTED_UNVALIDATED" if mutation == "unvalidated" else "VALIDATION_FAILED"
                spec["version_sha256"] = _sealed(spec, "version_sha256")
            elif mutation == "context":
                c.catalog["measurement_context"]["gpu_uuid"] = "changed"
            elif mutation == "quality":
                c.catalog["quality"]["paths"]["generalist"]["text"]["value"] = .99
            elif mutation == "schema":
                spec["schema"] = "gsdeploy.calibrated_surrogate.v1"
                spec["version_sha256"] = _sealed(spec, "version_sha256")
            else:
                spec["version_sha256"] = "changed"
            self.assertIsNone(surrogate_bounds(c.catalog, c.workload, **context(c)))
            for cls in (UtilityOptimizer, JointMilpOptimizer):
                solver = optimizer(cls, c)
                self.assertFalse(solver.separable_service)
                self.assertTrue(solver.profiled_memory)
                result = solver.solve(deployment={"generalist": 1}, K=2)
                self.assertFalse(result["optimality_certified"])
                self.assertIsNone(result["winner"])


if __name__ == "__main__":
    unittest.main()
