"""Independent contracts for the manufactured bounded algorithm benchmark."""
import itertools
import unittest

import numpy as np

from experiments.bilevel_scalability_benchmark import (
    PROTOCOL, QueueCostOracle, ControlledOptimizer, FullDomainSpaces,
    build_problem, coarse_routes, initial_routes, read, reference,
    reference_deployments, reference_objective)
from gsdeploy.continuous import ContinuousRoutingEvaluator
from gsdeploy.continuous.contracts import Budget, digest
from gsdeploy.continuous.route_domain import ContinuousRouteDomain


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = read(PROTOCOL)
        cls.problem = build_problem(3, cls.protocol)

    def test_reference_covers_all_memory_feasible_replica_states(self):
        p = build_problem(7, self.protocol)
        expected = set()
        names = sorted(p.states)
        for counts in itertools.product(range(3), repeat=7):
            x = {m: n for m, n in zip(names, counts) if n}
            if 0 < sum(counts) <= 2 and all(any(t in p.config.catalog['models'][m]['supported_tasks'] for m in x)
                                          for t in p.config.workload):
                expected.add(tuple(sorted(x.items())))
        self.assertEqual(expected, {tuple(sorted(x.items())) for x in reference_deployments(p)})

    def test_independent_gradient_and_tangent_bound(self):
        p, x = self.problem, {'g': 1, 's_shared': 1}
        z = np.array([.13, .41])  # image, text QPS
        cost, grad, route, bounds = reference_objective(p, x, z)
        self.assertAlmostEqual(cost, -QueueCostOracle().score(p, x, route)['U'], places=12)
        for i in range(2):
            step = np.zeros(2); step[i] = 1e-6
            numerical = (reference_objective(p, x, z+step)[0]-reference_objective(p, x, z-step)[0])/2e-6
            self.assertAlmostEqual(grad[i], numerical, places=7)
        for a in np.linspace(0, .35, 7):
            for b in np.linspace(0, .65, 7):
                trial = np.array([a, b])
                self.assertGreaterEqual(reference_objective(p, x, trial)[0]+1e-12, cost+grad@(trial-z))

    def test_reference_interval_contains_dense_grid_and_has_small_gap(self):
        p = self.problem
        ref = reference(p, self.protocol['reference'])
        self.assertLess(ref['continuous_reference_gap'], 1e-6)
        oracle = QueueCostOracle()
        for x in reference_deployments(p):
            for route in coarse_routes(p):
                q = ContinuousRouteDomain(p).to_qps(route)
                if all(m in x for row in q.values() for m in row):
                    self.assertLessEqual(oracle.score(p, x, q)['U'], ref['upper_U']+1e-10)

    def test_common_initialization_and_post_initialization_switches(self):
        signatures = []
        for method in ('BO-MADS', 'No-activation', 'No-pool', 'No-progressive'):
            e = ContinuousRoutingEvaluator(self.problem, QueueCostOracle(), budget=Budget(10, 64, 4, 128))
            opt = ControlledOptimizer(e, method, 11, self.protocol)
            result = opt.solve()
            signatures.append(digest(opt.initial_signature))
            self.assertEqual(result['budget']['outer_evaluations'], 4)
            self.assertEqual(opt.pool.disabled, method == 'No-pool')
            self.assertEqual(opt.exact_inner, method == 'No-progressive')
            if method == 'No-activation':
                self.assertIsInstance(opt.spaces, FullDomainSpaces)
                # Same legal domain, includes sparse routes and every endpoint.
                for point in e.domain.endpoints():
                    opt.spaces.domain.validate(point)
                self.assertEqual(opt.spaces.domain.dimension, e.domain.dimension)
        self.assertEqual(len(set(signatures)), 1)

    def test_state_floor_is_valid_for_positive_and_idle_models(self):
        p, oracle = self.problem, QueueCostOracle()
        for x in reference_deployments(p):
            floor = sum(oracle.state_costs(p)[m, n] for m, n in x.items())
            for route in initial_routes():
                q = ContinuousRouteDomain(p).to_qps(route)
                if all(m in x for row in q.values() for m in row):
                    self.assertLessEqual(oracle.score(p, x, q)['U'], -floor+1e-12)


if __name__ == '__main__':
    unittest.main()
