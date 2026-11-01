import unittest

from tools.queue_coupled_utility_20261023.model import UnknownEvidence
from tools.queue_coupled_utility_20261023.search import (
    SharedRateEnvironment, candidates, evaluate_cohort, execute_search, identity, structural_groups,
)


class Provider:
    def service(self, request, active, aliases, scenario):
        base = 100. if aliases[request['alias']]['family'] == 'text' else 25.
        return dict(known=True, context_supported=True, service_ms=base*len(active)*(1.2 if scenario == 'p90' else 1.))

    def residual(self, request, age_ms, active, aliases, scenario):
        service = self.service(request, active, aliases, scenario)['service_ms']
        return dict(known=age_ms < service, tail_supported=age_ms < service, remaining_ms=service-age_ms)

    def power(self, active, aliases, scenario):
        return dict(known=True, watts=100.*len(active)+100. if active else 50.)


class CoupledSearchTests(unittest.TestCase):
    def aliases(self):
        return {'text#0':dict(family='text', supported_tasks=['text']),
                'text#1':dict(family='text', supported_tasks=['text']),
                'generalist_2#0':dict(family='generalist_2', supported_tasks=['text'])}

    def row(self, rid, arrival=0):
        return dict(request_id=rid, task='text', pool_id='x', arrival_ns=arrival)

    def contract(self):
        return dict(aliases=self.aliases(), deployment={'text':2, 'generalist_2':1}, text_rate=1.85,
                    quality_loss={'text':{'text':0.}, 'generalist_2':{'text':.02}},
                    weights={t:1/6 for t in 'QTLPME'}, latency_refs_ms={'text':100.},
                    background_routing={}, horizon_seconds=1.)

    def test_new_arrival_reprices_existing_work_and_completion_reprices_survivor(self):
        environment = SharedRateEnvironment(self.aliases(), Provider(), 1_000_000_000)
        environment.enqueue(self.row('a'), 'text#0')
        environment.advance(50_000_000)
        environment.enqueue(self.row('b', 50_000_000), 'generalist_2#0')
        environment.advance(1_000_000_000)
        self.assertAlmostEqual(environment.results['b']['finish_ns'], 100_000_000.)
        self.assertAlmostEqual(environment.results['a']['finish_ns'], 125_000_000.)
        # .05s*200W+.05s*300W+.025s*200W+.875s*50W, board once.
        self.assertAlmostEqual(environment.energy_joules, 73.75)
        self.assertGreater(environment.rate_changes, 2)

    def test_fifo_starts_next_head_at_prior_completion(self):
        environment = SharedRateEnvironment(self.aliases(), Provider(), 100_000_000)
        environment.enqueue(self.row('a'), 'text#0')
        environment.enqueue(self.row('b'), 'text#0')
        environment.advance(100_000_000)
        environment.drain()
        self.assertEqual(environment.results['b']['start_ns'], 100_000_000.)
        self.assertEqual(environment.results['b']['finish_ns'], 200_000_000.)
        self.assertAlmostEqual(environment.energy_joules, 20.)

    def test_missing_environment_evidence_is_unknown_never_fallback(self):
        class Missing(Provider):
            def service(self, *args):
                return dict(known=False, context_supported=False, service_ms=None, status='MISSING')
        environment = SharedRateEnvironment(self.aliases(), Missing(), 1_000_000_000)
        with self.assertRaisesRegex(UnknownEvidence, 'UNKNOWN_ENVIRONMENT_SERVICE'):
            environment.enqueue(self.row('a'), 'text#0')
        candidate = next(candidates(self.contract()['deployment']))
        result = evaluate_cohort(self.contract(), {}, [self.row('a')], candidate, provider=Missing())
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertIsNone(result['eligible'])
        self.assertIsNone(result['U'])
        self.assertTrue(result['unsupported_not_infeasible'])

    def test_finite_domain_and_symbolic_control_groups(self):
        items = list(candidates(self.contract()['deployment']))
        self.assertEqual(len(items), 24)
        groups = structural_groups(items)
        self.assertEqual(len(groups), 14)
        nonsingletons = [g for g in groups if len(g['members']) > 1]
        self.assertEqual([len(g['members']) for g in nonsingletons], [6, 6])
        self.assertTrue(all(not g['proof']['objective_used'] for g in nonsingletons))
        self.assertEqual(len({identity(c) for c in items}), 24)

    def test_independent_dfs_and_quotient_run_actual_simulators_and_match_ties(self):
        contract = self.contract()
        cohort = [self.row(str(i), i*5_000_000) for i in range(6)]
        calls = []
        def evaluate(candidate, seed):
            calls.append((identity(candidate), seed))
            return evaluate_cohort(contract, {}, cohort, candidate, provider=Provider(), memory_norm=.8)
        items = list(candidates(contract['deployment']))
        dfs = execute_search(items, (313, 314), evaluate)
        optimized = execute_search(items, (313, 314), evaluate, optimized=True)
        self.assertEqual(dfs['oracle_calls'], 48)
        self.assertEqual(optimized['oracle_calls'], 28)
        self.assertEqual(len(calls), 76)
        self.assertEqual(dfs['winner_id'], optimized['winner_id'])
        self.assertEqual(dfs['tie_ids'], optimized['tie_ids'])
        self.assertTrue(dfs['whole_declared_cpu_domain_certified'])
        self.assertFalse(dfs['physical_optimality'])
        original = {r['candidate_id']:r for r in dfs['records']}
        for row in optimized['records']:
            self.assertEqual(row['objective'], original[row['candidate_id']]['objective'])
            for a, b in zip(row['runs'], original[row['candidate_id']]['runs']):
                self.assertEqual(a['route_trace_sha256'], b['route_trace_sha256'])
                self.assertEqual(a['completion_trace_sha256'], b['completion_trace_sha256'])

    def test_unknown_blocks_certificate_even_when_known_winner_exists(self):
        items = list(candidates(self.contract()['deployment']))
        def evaluate(candidate, seed):
            if candidate['allow_spill']:
                return dict(status='UNKNOWN', U=None, eligible=None)
            return dict(status='COMPLETE', U=-1., eligible=True)
        result = execute_search(items, (313, 314), evaluate, optimized=True)
        self.assertIsNotNone(result['winner_id'])
        self.assertEqual(len(result['unknown_ids']), 12)
        self.assertFalse(result['whole_declared_cpu_domain_certified'])
        self.assertEqual(result['strict_bound_candidates_pruned'], 0)

    def test_memory_and_physical_utility_remain_unknown_without_measured_evidence(self):
        contract = self.contract()
        candidate = next(candidates(contract['deployment']))
        result = evaluate_cohort(contract, {}, [self.row('a')], candidate, provider=Provider())
        self.assertEqual(result['status'], 'COMPLETE_CONDITIONAL_CPU_PROXY')
        self.assertIsNone(result['terms']['M'])
        self.assertNotIn('M', result['objective_terms'])
        self.assertIsNone(result['physical_U'])
        self.assertIsNone(result['formal_U'])


if __name__ == '__main__':
    unittest.main()
