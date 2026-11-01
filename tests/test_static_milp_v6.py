"""Six-term joint search integration; every numerical fixture is CPU synthetic."""

import copy
import unittest
from dataclasses import replace

from gsdeploy import DeploymentSystem
from gsdeploy.energy import energy_key
from gsdeploy.performance import resolve_service
from tests.test_milp import add_energy, grid_plans
from tests.test_search_contract import config
from tests.test_static_dfs_v6 import six_term_config


class StaticMilpV6Tests(unittest.TestCase):
    def test_specialist_backlog_is_offloaded_and_quality_penalty_can_stop_it(self):
        c = config({"text": 4.8, "image": 3.0},
                   {"g": {"text": 200.0, "image": 50.0}, "s": {"text": 160.0}},
                   horizon=120)
        c.catalog["capability_contract"] = "verified_executable_v1"
        for spec in c.catalog["models"].values():
            spec["executable_tasks"] = spec["supported_tasks"][:]
        c.catalog["quality"] = {
            "schema_version": 1, "mode": "aggregate_proxy",
            "identity": {k: "CPU synthetic fixture" for k in
                         ("input_pool_sha256", "reference_sha256", "protocol_sha256", "source")},
            "paths": {"g": {"text": {"metric": "em", "value": 0.8},
                              "image": {"metric": "accuracy", "value": 0.9}},
                      "s": {"text": {"metric": "em", "value": 0.95}}},
        }
        for tasks in c.catalog["quality"]["paths"].values():
            for record in tasks.values():
                record.update(unit="fraction", checkpoint_sha256="CPU fixture", source="CPU fixture")
        c = replace(c, weights={k: 1 / 6 for k in "QTLPME"}, solver="milp",
                    utility_schema="six_term_quality_v1", arrival_process="poisson",
                    latency_reference_ms={"text": 1000.0, "image": 250.0},
                    milp_time_limit_seconds=15)
        add_energy(c, list(grid_plans(c, 4)), lambda *_: 12000.0)
        best = DeploymentSystem(c).plan(route_grid=4)
        self.assertTrue(best["optimality_certified"], best["status"])
        self.assertEqual(set(best["winner"]["routing"]["text"]), {"g", "s"})
        unsplit = DeploymentSystem(c).plan(route_grid=4, no_split=True)
        self.assertGreater(best["winner"]["score"], unsplit["winner"]["score"])
        quality_first = replace(c, weights={k: (0.9 if k == "Q" else 0.02) for k in "QTLPME"})
        protected = DeploymentSystem(quality_first).plan(route_grid=4)
        self.assertEqual(protected["winner"]["routing"]["text"], {"s": 4.8})

    def test_quality_aware_master_and_dfs_agree(self):
        c = six_term_config()
        a = DeploymentSystem(replace(c, solver="milp")).plan(route_grid=2)
        b = DeploymentSystem(replace(c, solver="dfs")).plan(route_grid=2)
        self.assertTrue(a["optimality_certified"], a["status"])
        self.assertTrue(b["optimality_certified"], b["status"])
        self.assertAlmostEqual(a["winner"]["score"], b["winner"]["score"], places=9)
        self.assertEqual(a["winner"]["deployment"], {"s": 1})

    def test_challengers_reoptimize_and_keep_restrictions(self):
        c = replace(six_term_config(), solver="milp")
        system = DeploymentSystem(c)
        best = system.plan(route_grid=2)["winner"]
        alternative = system.plan(route_grid=2, exclude_deployments=[best["deployment"]])
        self.assertNotEqual(alternative["winner"]["deployment"], best["deployment"])
        self.assertTrue(alternative["optimality_certified"], alternative["status"])
        self.assertLessEqual(alternative["winner"]["score"], best["score"] + 1e-9)
        routed = system.plan(deployment=best["deployment"], route_grid=2,
                             exclude_plans=[best], no_split=True)
        self.assertIsNone(routed["winner"])
        self.assertEqual(routed["status"], "NO_FEASIBLE_GRID_PLAN")
        self.assertTrue(routed["search_restrictions"]["no_split"])

    def test_missing_quality_cannot_certify_partial_energy_diagnostic(self):
        c = replace(six_term_config(), solver="milp")
        c.catalog["quality"]["paths"] = {}
        c.catalog.pop("energy_profiles")
        result = DeploymentSystem(c).plan(route_grid=2)
        self.assertFalse(result["optimality_certified"])
        self.assertFalse(result["diagnostic_optimality_certified"])
        self.assertEqual(result["status"], "INCOMPLETE_QUALITY_DOMAIN")
        self.assertTrue(result["unresolved_candidates"])

    def test_replica_means_are_task_specific_without_double_scaling(self):
        c = six_term_config()
        c.catalog["performance_mode"] = "task_replica_proxy"
        c.catalog["capacity_factors"]["g"]["2"] = 2.0
        c.catalog["memory_table_mib"]["g"]["2"] = 2.0
        c.catalog["task_replica_profiles"] = {
            "g": {"text": {"1": {"mean_service_ms": 2.0},
                              "2": {"mean_service_ms": 9.0}}},
            "s": {"text": {"1": {"mean_service_ms": 4.0}}},
        }
        service, evidence = resolve_service(c.catalog, {"g": 2}, {"text": {"g": 1}}, c.workload)
        self.assertEqual(service["g"]["text"], 9.0)
        self.assertIn("unmeasured", evidence["scope"])
        add_energy(c, list(grid_plans(c, 2)), lambda *_: 1.0)
        milp = DeploymentSystem(replace(c, solver="milp")).plan(route_grid=2)
        dfs = DeploymentSystem(replace(c, solver="dfs")).plan(route_grid=2)
        self.assertTrue(milp["optimality_certified"], milp["status"])
        self.assertAlmostEqual(milp["winner"]["score"], dfs["winner"]["score"], places=9)
        del c.catalog["task_replica_profiles"]["g"]["text"]["2"]
        service, evidence = resolve_service(c.catalog, {"g": 2}, {"text": {"g": 1}}, c.workload)
        self.assertIsNone(service)
        self.assertEqual(evidence["reason"], "MISSING_TASK_REPLICA_PROFILE")

    def test_new_measurement_identity_is_pool_independent_and_condition_bound(self):
        c = six_term_config()
        x, routing = {"s": 1}, {"text": {"s": 1.0}}
        def key(catalog):
            return energy_key(x, routing, c.workload, c.horizon_seconds, c.random_seed,
                              c.arrival_process, catalog=catalog)
        original = key(c.catalog)
        reduced = copy.deepcopy(c.catalog)
        for field in ("models", "capacity_factors", "memory_table_mib"):
            reduced[field].pop("g")
        self.assertEqual(original, key(reduced))
        reduced["measurement_context"] = {"gpu_uuid": "different", "input_sha256": "changed"}
        self.assertNotEqual(original, key(reduced))


if __name__ == "__main__":
    unittest.main()
