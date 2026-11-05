"""Semantic checks only; no extra benchmark searches or tuning runs."""
import copy
import unittest
import numpy as np

from experiments.compare_crossdomain_strategies_20261105 import (
    PROTOCOL, ARCHIVE, read, build_problem, legal, neighbors, repaired_route,
    route_neighbors, MixedCodec, de_trial, LoggedOracle, DirectSearch, initialize,
    ScheduledControlledOptimizer, ContinuousRoutingEvaluator, Budget, BudgetExhausted)


class CrossDomainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = read(ARCHIVE/'plan.json')['protocol']
        cls.cfg = read(PROTOCOL)
        cls.problem = build_problem(7, cls.base)

    def evaluator(self, predictions=64):
        return ContinuousRoutingEvaluator(self.problem, LoggedOracle(), budget=Budget(30, predictions, 256, 128))

    def test_protocol_fixed_and_hardware_excluded(self):
        c = self.cfg
        self.assertEqual(len(c['sizes'])*len(c['seeds'])*len(c['methods']), c['maximum_searches'])
        self.assertEqual(c['maximum_searches'], 24)
        self.assertEqual(c['real_predictor_calls'], 0)
        self.assertEqual(c['new_gpu_runs'], 0)

    def test_neighbors_include_replica_transfer_and_respect_coverage(self):
        ns = neighbors(self.problem, {'g': 1})
        self.assertIn({'g': 2}, ns)
        self.assertIn({'g': 1, 's_text': 1}, ns)
        self.assertIn({'s_shared': 1}, ns)
        self.assertNotIn({'s_text': 1}, ns)
        self.assertTrue(all(legal(self.problem, n) for n in ns))
        self.assertIn({'g': 1, 's_shared': 1}, neighbors(self.problem, {'g': 2}))

    def test_flow_repair_and_endpoint_poll_preserve_load(self):
        e = self.evaluator()
        previous = e.domain.validate({t: {'g': 1.} for t in e.domain.edges})
        route = repaired_route(self.problem, e.domain, {'g': 1, 's_text': 1}, previous)
        self.assertEqual(route['image']['g'], 1.)
        self.assertEqual(route['text']['s_text'], .5)
        trials = route_neighbors(self.problem, e.domain, {'g': 1, 's_text': 1}, route, .5)
        self.assertTrue(any(r['text']['g'] == 0 for r in trials))
        self.assertTrue(any(r['text']['s_text'] == 0 for r in trials))
        for r in trials:
            qps = e.domain.to_qps(r)
            for t, row in qps.items():self.assertAlmostEqual(sum(row.values()), self.problem.config.workload[t])

    def test_mixed_decode_repairs_unsupported_singleton_and_roundtrips(self):
        e = self.evaluator();codec = MixedCodec(self.problem, e.domain)
        z = np.zeros(codec.dimension);z[codec.names.index('s_text')] = 1
        d, route = codec.decode(z)
        self.assertTrue(legal(self.problem, d))
        cases = [({'g': 2}, {t: {'g': 1.} for t in e.domain.edges}),
                 ({'g': 1, 's_shared': 1}, {'text': {'g': .17, 's_shared': .83}, 'image': {'g': 0., 's_shared': 1.}})]
        for d, r in cases:
            decoded, shares = codec.decode(codec.encode(d, r))
            self.assertEqual(decoded, d)
            self.assertEqual(e.domain.to_qps(shares), e.domain.to_qps(r))
        for z in np.random.default_rng(10).random((100, codec.dimension)):
            d, r = codec.decode(z)
            self.assertTrue(legal(self.problem, d))
            self.assertTrue(all(m in d for row in e.domain.to_qps(r).values() for m in row))

    def test_de_forces_mutant_coordinate_at_zero_crossover(self):
        pop = np.arange(32).reshape(8, 4)/40
        trial = de_trial(pop, 0, np.random.default_rng(7), .8, 0.)
        self.assertTrue(np.any(trial != pop[0]))
        self.assertTrue(np.all((trial >= 0) & (trial <= 1)))

    def test_common_initialization_cache_and_no_unpaid_oracle(self):
        e = self.evaluator(predictions=19)
        opt = ScheduledControlledOptimizer(e, 'BO-MADS', 11, self.base)
        initialize(opt)
        prior = read(ARCHIVE.parent/'bilevel-scheduler-improvement-20261105/7-balanced-11.json')
        self.assertEqual(opt.initial_signature, prior['initial_signature'])
        self.assertEqual(len(e.oracle.calls), 18)
        search = DirectSearch(e, opt, 11, self.cfg)
        d = {'g': 1, 's_shared': 1}
        r = {'text': {'g': .123, 's_shared': .877}, 'image': {'g': .3, 's_shared': .7}}
        search.evaluate(d, r, 'unit_check')
        self.assertEqual(len(e.oracle.calls), 19)
        search.evaluate(d, r, 'unit_check_cache')
        self.assertEqual(len(e.oracle.calls), 19)
        r['text'] = {'g': .124, 's_shared': .876}
        with self.assertRaises(BudgetExhausted):search.evaluate(d, r, 'over_budget')
        self.assertEqual(len(e.oracle.calls), 19)
        self.assertEqual(e.budget.counts['predictions'], 19)


if __name__ == '__main__':unittest.main()
