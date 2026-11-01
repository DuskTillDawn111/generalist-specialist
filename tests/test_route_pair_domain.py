"""CPU finite-domain correctness only; invented utility is not GPU evidence."""

import itertools
import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

from gsdeploy.config import SystemConfig
from gsdeploy.route_domain import fraction_rows
from gsdeploy.static_study_benchmark import grid_plans
from gsdeploy.system import DeploymentSystem
from gsdeploy.optimizer import UtilityOptimizer
from tests.test_milp import add_energy
from tests.test_static_dfs_v6 import six_term_config


def toy_problem():
    c = replace(six_term_config(), route_grid=2, horizon_seconds=64,
                route_pair_denominators_by_task={"text": [32, 64]},
                milp_capacity_witness=False, milp_time_limit_seconds=30,
                milp_max_iterations=1000, search_max_states=100000)
    c.catalog["capacity_factors"]["s"]["2"] = 1.65
    c.catalog["memory_table_mib"]["s"]["2"] = 2.0
    c.catalog["energy_profiles"] = {}
    plans = list(grid_plans(c))
    # Invented complete energy evidence gives a unique fine-grid optimum.
    # This tests solver coverage, and asserts nothing about real hardware.
    def joules(x, r, _):
        target = (x == {"g": 1, "s": 2}
                  and r["text"].get("g") == 1 / 64)
        return 1.0 if target else 1_000_000.0
    add_energy(c, plans, joules)
    return c, plans


class PairDomainTests(unittest.TestCase):
    def test_default_exactly_matches_independent_legacy_cartesian_grid(self):
        for size in (0, 1, 2, 3, 4):
            models = [str(i) for i in range(size)]
            for K in (1, 2, 4):
                expected = set()
                for d in range(1, K + 1):
                    for units in itertools.product(range(d + 1), repeat=size):
                        if sum(units) == d:
                            expected.add(tuple((m, Fraction(u, d))
                                               for m, u in zip(models, units) if u))
                actual = list(fraction_rows(models, K))
                self.assertEqual(set(actual), expected)
                self.assertEqual(len(actual), len(expected))

    def test_pair_extensions_are_exact_nested_and_never_refine_three_family_rows(self):
        models = ["g1", "g2", "s1", "s2"]
        base = set(fraction_rows(models, 3))
        extended = list(fraction_rows(models, 3, [32, 64, 32]))
        self.assertTrue(base.issubset(set(extended)))
        self.assertEqual(len(extended), len(set(extended)))
        for pair in itertools.combinations(models, 2):
            self.assertIn(((pair[0], Fraction(1, 64)),
                           (pair[1], Fraction(63, 64))), extended)
        self.assertTrue(all(len(row) == 2 for row in set(extended) - base))
        self.assertEqual({row for row in extended if len(row) >= 3},
                         {row for row in base if len(row) >= 3})

    def test_config_rejects_invalid_extension_and_roundtrips(self):
        c = six_term_config()
        for invalid in ([], {"unknown": [32]}, {"text": "64"},
                        {"text": [True]}, {"text": [1]}, {"text": [32.0]}):
            with self.assertRaises(ValueError):
                replace(c, route_pair_denominators_by_task=invalid)
        updated = replace(c, route_pair_denominators_by_task={"text": [32, 64]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(updated.as_dict()))
            self.assertEqual(SystemConfig.load(path).as_dict(), updated.as_dict())
            old = c.as_dict()
            del old["route_pair_denominators_by_task"]
            path.write_text(json.dumps(old))
            self.assertEqual(SystemConfig.load(path).route_pair_denominators_by_task, {})

    def test_milp_bounded_plain_dfs_and_full_oracle_agree_on_fine_optimum(self):
        c, plans = toy_problem()
        system = DeploymentSystem(c)
        scores = [system.predict(x, row)["U"] for x, row in plans]
        expected = max(score for score in scores if score is not None)
        results = [DeploymentSystem(replace(c, solver=solver)).plan()
                   for solver in ("milp", "dfs")]
        plain = UtilityOptimizer(c.catalog, c.workload, c.gpu_memory_mib,
            c.latency_reference_ms, c.weights, **system.context(),
            route_pair_denominators_by_task=c.route_pair_denominators_by_task,
            milp_capacity_witness=False, dfs_bound_pruning=False).solve(K=c.route_grid)
        results.append(plain)
        for result in results:
            self.assertTrue(result["optimality_certified"], result["status"])
            self.assertAlmostEqual(result["winner"]["score"], expected, places=10)
            self.assertEqual(result["winner"]["deployment"], {"g": 1, "s": 2})
            self.assertEqual(result["winner"]["routing"]["text"]["g"], 1 / 64)
            self.assertEqual(result["domain"]["route_pair_denominators_by_task"],
                             {"text": [32, 64]})
            self.assertFalse(result["domain"]["capacity_boundary_witnesses"])
            self.assertFalse(result["domain"]["continuous_optimality"])

    def test_pure_and_one_replica_controls_remain_searchable(self):
        c, _ = toy_problem()
        for solver in ("milp", "dfs"):
            system = DeploymentSystem(replace(c, solver=solver))
            pure = system.plan(no_split=True)
            self.assertTrue(pure["optimality_certified"])
            self.assertEqual(len(pure["winner"]["routing"]["text"]), 1)
            small = system.plan(replica_options={"s": [1], "g": [1]})
            self.assertTrue(small["optimality_certified"])
            self.assertLessEqual(small["winner"]["deployment"].get("s", 0), 1)

    def test_search_session_propagates_fine_domain_to_all_algorithms(self):
        from gsdeploy.search_session import SearchSession
        from tests.test_calibrated_search_bounds_v6 import search_fixture
        c = replace(search_fixture(), route_pair_denominators_by_task={"text": [64]})
        root = Path(__file__).resolve().parents[1]
        with SearchSession(root, c.as_dict()) as session:
            for algorithm in ("milp", "dfs", "plain_dfs"):
                optimizer = session.optimizer(algorithm)
                self.assertEqual(optimizer.pair_denominators, {"text": [64]})
                self.assertNotIn("route_pair_denominators_by_task", optimizer.context)

    def test_search_session_keeps_old_frozen_source_workflow_and_rejects_extension(self):
        from gsdeploy.search_session import SearchSession
        root = Path(__file__).resolve().parents[1]
        source = root/'artifacts/static-study-gpu-20261022/execution-source-v5'
        raw = json.loads((root/'artifacts/static-study-gpu-20261022/search-v2-v5-K8/system-config.json').read_text())
        raw['route_pair_denominators_by_task'] = {}
        with SearchSession(source, raw) as session:
            self.assertNotIn('route_pair_denominators_by_task', session.config.__dataclass_fields__)
            for algorithm in ('milp', 'dfs', 'plain_dfs'):
                optimizer = session.optimizer(algorithm)
                self.assertNotIn('route_pair_denominators_by_task', optimizer.context)
            with self.assertRaisesRegex(ValueError, 'SOURCE_ROUTE_DOMAIN_UNSUPPORTED'):
                session.optimizer('milp', route_pair_denominators_by_task={'text': [64]})
        raw['route_pair_denominators_by_task'] = {'text': [64]}
        with self.assertRaisesRegex(ValueError, 'SOURCE_ROUTE_DOMAIN_UNSUPPORTED'):
            SearchSession(source, raw)

    def test_static_swrr_realizes_small_share_with_two_specialist_replicas(self):
        from collections import Counter
        from gsdeploy.routing import Router
        from gsdeploy.trace import expand_routes
        aliases, routes = expand_routes({"g": 1, "s": 2},
            {"text": {"g": 1 / 64, "s": 63 / 64}})
        router = Router(routes)
        observed = Counter(router.choose("text") for _ in range(128))
        self.assertEqual(observed, {"g#0": 2, "s#0": 63, "s#1": 63})
        self.assertEqual(aliases, {"g#0": "g", "s#0": "s", "s#1": "s"})

    def test_legacy_domain_membership_rejects_extended_certificates_explicitly(self):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "tools/spillover_discovery_20261022/target_specialist.py"
        spec = importlib.util.spec_from_file_location("legacy_target_pair_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        plan = {"routing": {"text": {"g": 0.5, "s": 0.5}}}
        self.assertEqual(module.route_grid_membership(plan, {"text": 1.0}, 2)["K"], 2)
        with self.assertRaisesRegex(ValueError, "PAIR_ROUTE_DOMAIN_UNSUPPORTED"):
            module.route_grid_membership(plan, {"text": 1.0}, 2,
                route_pair_denominators_by_task={"text": [64]})
        plan["domain"] = {"route_pair_denominators_by_task": {"text": [64]}}
        with self.assertRaisesRegex(ValueError, "PAIR_ROUTE_DOMAIN_UNSUPPORTED"):
            module.route_grid_membership(plan, {"text": 1.0}, 2)


if __name__ == "__main__":
    unittest.main()
