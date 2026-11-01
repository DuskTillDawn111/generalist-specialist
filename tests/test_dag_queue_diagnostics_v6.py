import unittest

from tools.predictor_validation_20261027.dag_queue_diagnostics_v6 import DagQueueDiagnostics


class DagQueueDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.aliases = {'owner': {'family': 'text', 'supported_tasks': ['text']}}
        self.plans = {('owner', 'text', pool): {'nodes': [
            {'id': 'gpu', 'kind': 'gpu', 'fitted_ns': duration}]} for pool, duration in (
                ('long', 100_000_000), ('short', 50_000_000))}
        self.calls = 0

    def provider(self, node, row):
        self.calls += 1
        return node['fitted_ns']

    def row(self, rid, pool, arrival):
        return dict(request_id=rid, alias='owner', task='text', pool_id=pool, arrival_ns=arrival)

    def test_long_head_consumes_work_once_and_short_request_recovers(self):
        diagnostic = DagQueueDiagnostics(self.aliases, self.plans, self.provider)
        first, second = self.row('first', 'long', 0), self.row('second', 'short', 25_000_000)
        diagnostic.register(first)
        diagnostic.mark_started(first)
        diagnostic.observe(0, 25_000_000, {'owner': []}, {'owner': {'row': first}},
            {('owner', 'gpu'): {'kind': 'gpu', 'remaining_ns': 100_000_000}}, {('owner', 'gpu'): 1.})
        diagnostic.register(second)
        diagnostic.observe(25_000_000, 75_000_000, {'owner': [second]}, {'owner': {'row': first}},
            {('owner', 'gpu'): {'kind': 'gpu', 'remaining_ns': 75_000_000}}, {('owner', 'gpu'): 1.})
        diagnostic.mark_started(second)
        diagnostic.observe(100_000_000, 50_000_000, {'owner': []}, {'owner': {'row': second}},
            {('owner', 'gpu'): {'kind': 'gpu', 'remaining_ns': 50_000_000}}, {('owner', 'gpu'): 1.})
        result = diagnostic.result({'first': first, 'second': second})
        self.assertEqual(result['max_remaining_activity_work_ms'], 125.)
        self.assertEqual(result['remaining_activity_work_ms'], 0.)
        self.assertEqual(result['max_oldest_waiting_age_ms'], 75.)
        self.assertEqual(result['owner_fifo_request_wait_ms']['owner'], 75.)
        self.assertEqual(result['owner_fifo_head_blocked_ms']['owner'], 75.)
        self.assertEqual(result['queue_recovery_episodes'], [dict(
            backlog_start_ns=25_000_000, queue_clear_ns=100_000_000, recovery_ms=75.)])
        self.assertEqual(result['offered_work_envelopes'][0]['max_offered_solo_work_ms'], 150.)
        self.assertIsNone(result['max_normalized_resource_work_ms']['dram'])
        self.assertFalse(result['global_admission_hol_applicable'])
        self.assertFalse(result['six_term_utility_changed'])

    def test_model_work_not_elapsed_wall_time_and_explicit_resource_demands(self):
        diagnostic = DagQueueDiagnostics(self.aliases, self.plans, self.provider,
            resource_demand_provider=lambda node, row: dict(sm=.25, dram=.5, l2=0.))
        row = self.row('first', 'long', 0)
        diagnostic.register(row)
        diagnostic.mark_started(row)
        diagnostic.observe(0, 200_000_000, {'owner': []}, {'owner': {'row': row}},
            {('owner', 'gpu'): {'kind': 'gpu', 'remaining_ns': 100_000_000}}, {('owner', 'gpu'): .5})
        result = diagnostic.result({'first': row})
        self.assertEqual(result['max_normalized_resource_work_ms'], dict(sm=25., dram=50., l2=0.))
        self.assertEqual(result['remaining_activity_work_ms'], 0.)
        self.assertEqual(result['unknown_reasons'], [])

    def test_profile_cache_is_shared_with_execution_duration_provider(self):
        diagnostic = DagQueueDiagnostics(self.aliases, self.plans, self.provider)
        first = self.row('first', 'long', 0)
        second = self.row('second', 'long', 0)
        diagnostic.register(first)
        diagnostic.register(second)
        node = self.plans['owner', 'text', 'long']['nodes'][0]
        self.assertEqual(diagnostic.duration(node, first), 100_000_000)
        self.assertEqual(diagnostic.duration(node, second), 100_000_000)
        self.assertEqual(self.calls, 1)
        with self.assertRaises(ValueError):
            diagnostic.register(first)
        with self.assertRaises(ValueError):
            diagnostic.duration(dict(node, kind='copy'), first)

    def test_shortened_run_keeps_full_offered_envelope_unknown(self):
        diagnostic = DagQueueDiagnostics(self.aliases, self.plans, self.provider)
        first, second = self.row('first', 'long', 0), self.row('second', 'short', 25_000_000)
        diagnostic.register(first)
        result = diagnostic.result({'first': first, 'second': second})
        self.assertEqual(result['offered_work_envelopes'][0]['max_arrival_count'], 2)
        self.assertIsNone(result['offered_work_envelopes'][0]['max_offered_solo_work_ms'])

    def test_handoff_blocking_is_distinct_from_active_handler(self):
        diagnostic = DagQueueDiagnostics(self.aliases, self.plans, self.provider)
        row = self.row('first', 'long', 0)
        diagnostic.register(row)
        diagnostic.observe(0, 3_000_000, {'owner': [row]}, {}, {}, {})
        result = diagnostic.result({'first': row})
        self.assertEqual(result['owner_fifo_blocking_causes_ms']['owner'], {'handoff_or_worker_not_ready': 3.})
        with self.assertRaises(ValueError):
            diagnostic.observe(3_000_000, 1, {'owner': []}, {}, {}, {})

    def test_nonphysical_resource_demand_is_rejected(self):
        diagnostic = DagQueueDiagnostics(self.aliases, self.plans, self.provider,
            resource_demand_provider=lambda node, row: dict(sm=1.1, dram=0., l2=0.))
        with self.assertRaises(ValueError):
            diagnostic.register(self.row('first', 'long', 0))

    def test_integrated_fifo_area_equals_full_request_wait_with_handoff(self):
        from tools.predictor_validation_20261027.cuda_dag_simulator_v6 import DagFullHorizonModel

        class Lifecycle:
            def delays(self, *args):
                return dict(known=True, parent_receipt_ms=50., worker_next_ready_ms=30.)

        aliases = {a: self.aliases['owner'] for a in ('owner', 'other')}
        plans = {(a, task, pool): dict(profile, edges=[], cyclic=False, unresolved=[])
                 for a in aliases for (_, task, pool), profile in self.plans.items()}
        rows = [self.row('a1', 'long', 0), self.row('a2', 'short', 10_000_000),
                self.row('a3', 'long', 25_000_000),
                dict(self.row('b1', 'long', 0), alias='other'),
                dict(self.row('b2', 'short', 5_000_000), alias='other')]
        observer = DagQueueDiagnostics(aliases, plans, self.provider)
        options = dict(allow_context_time_slice_hypothesis=True, lifecycle_provider=Lifecycle())
        observed = DagFullHorizonModel(aliases, plans, duration_provider=observer.duration,
            queue_diagnostics=observer, **options).forecast(rows, 0, 1_000_000_000)
        plain = DagFullHorizonModel(aliases, plans, duration_provider=self.provider,
            **options).forecast(rows, 0, 1_000_000_000)
        self.assertEqual(observed['finishes'], plain['finishes'])
        self.assertEqual(observed['starts'], plain['starts'])
        q = observed['queue_diagnostics']
        self.assertAlmostEqual(sum(q['owner_fifo_request_wait_ms'].values()),
            sum(observed['queue_wait_ns'].values()) / 1e6)
        self.assertAlmostEqual(q['remaining_activity_work_ms'], 0.)
        self.assertEqual(q['registered_request_count'], 5)
        self.assertEqual(q['registered_request_activity_node_instances'], 5)
        self.assertEqual(q['unique_cached_activity_nodes'], 4)
        self.assertGreater(q['owner_fifo_blocking_causes_ms']['owner']['handoff_or_worker_not_ready'], 0.)


if __name__ == '__main__':
    unittest.main()
