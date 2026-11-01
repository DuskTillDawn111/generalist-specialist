"""Independent scheduler/owner-ready counterexamples; synthetic CPU only."""
import unittest

from gsdeploy.forecast_errors import UnknownEvidence

from tools.predictor_validation_20261027.conditional_availability_scheduler_v8 import ConditionalAvailabilityScheduler
from tools.predictor_validation_20261027.dag_phase_scheduler_v8 import DagPhaseFullHorizonModel, LegacyEqualContextDiagnosticScheduler
from tools.predictor_validation_20261027.cuda_dag_simulator_v9 import DagFullHorizonModel as V9Model
from tools.predictor_validation_20261027.dag_checkpoint_v9 import decode, encode, receipt
from tools.predictor_validation_20261027.dag_checkpoint_v7 import digest


class Lifecycle:
    def __init__(self, receipt_ns=0, ready_ns=0):
        self.receipt_ns, self.ready_ns = receipt_ns, ready_ns

    def delays(self, *args):
        return dict(known=True, parent_receipt_ms=self.receipt_ns / 1e6,
                    worker_next_ready_ms=self.ready_ns / 1e6)


class AbsoluteStartSupply:
    def __call__(self, row):
        return row['arrival_ns']


def row(rid, *, arrival=0):
    return dict(request_id=rid, alias=rid[0], task='text', pool_id='x', arrival_ns=arrival)


def plan(work):
    return dict(nodes=[dict(id='gpu', kind='gpu', duration_ns=work, release_ns=0)], edges=[])


def conditional():
    state = lambda owners: [[a, 'text', 'x', ['p']] for a in owners]
    return ConditionalAvailabilityScheduler(dict(
        schema='gsdeploy.conditional_context_availability_policy.v8',
        reference_work_domain='SOLO_ONLY_REFERENCE_NS', training_only=True, whole_corun_multiplier_used=False,
        cells=[dict(frontier=state('ab'), current_owner=None, reason='initial', next_owner='a', grant_budget_ns=7, handoff_ns=0),
               dict(frontier=state('b'), current_owner='a', reason='yield', next_owner='b', grant_budget_ns=20, handoff_ns=2),
               dict(frontier=state('a'), current_owner='b', reason='yield', next_owner='a', grant_budget_ns=20, handoff_ns=2)]),
        allow_unvalidated_training_candidate=True)


class SchedulerIndependentReview(unittest.TestCase):
    def test_same_clock_next_request_retains_context_grant(self):
        aliases = {a: dict(supported_tasks=['text']) for a in 'ab'}
        scheduler = conditional()
        model = DagPhaseFullHorizonModel(aliases, {('a', 'text', 'x'): plan(1), ('b', 'text', 'x'): plan(10)},
            scheduler=scheduler, allow_profiled_replay=True, phase_provider=lambda n, r: 'p')
        result = model.forecast([row('a1'), row('a2'), row('b1')], 0, 100, drain_deadline_ns=200)
        self.assertEqual({rid: r['handler_end_ns'] for rid, r in result['finishes'].items()},
                         dict(a1=1, a2=2, b1=14))
        self.assertEqual([(t['clock_ns'], t['reason']) for t in scheduler.transitions], [(0, 'initial'), (2, 'yield')])

    def single_model(self, work=10, *, receipt_ns=0):
        return DagPhaseFullHorizonModel({'a': dict(supported_tasks=['text'])}, {('a', 'text', 'x'): plan(work)},
            allow_profiled_replay=True, lifecycle_provider=Lifecycle(receipt_ns),
            scheduler=LegacyEqualContextDiagnosticScheduler(allow_legacy_equal_context_diagnostic=True))

    def test_handler_and_receipt_at_exact_drain_cutoff_are_included(self):
        result = self.single_model().forecast([row('a1')], 0, 5, drain_deadline_ns=10)
        self.assertEqual(result['finishes']['a1']['finish_ns'], 10)
        self.assertEqual(result['finishes']['a1']['handler_end_ns'], 10)

    def test_delayed_parent_receipt_exactly_at_drain_is_included(self):
        result = self.single_model(5, receipt_ns=5).forecast([row('a1')], 0, 7, drain_deadline_ns=10)
        self.assertEqual(result['finishes']['a1']['finish_ns'], 10)
        self.assertEqual(result['finishes']['a1']['handler_end_ns'], 5)

    def test_work_after_exact_drain_boundary_is_not_advanced(self):
        result = self.single_model(11).forecast([row('a1')], 0, 5, drain_deadline_ns=10)
        self.assertNotIn('a1', result['finishes'])
        self.assertEqual(result['completed_dag_nodes'], 0)

    def test_same_clock_zero_work_chain_cannot_bypass_event_budget(self):
        aliases = {'a': dict(supported_tasks=['text'])}
        plans = {('a', 'text', 'x'): dict(nodes=[dict(id='end', kind='handler_end')], edges=[])}
        model = DagPhaseFullHorizonModel(aliases, plans, max_events=1)
        with self.assertRaisesRegex(UnknownEvidence, 'EVENT_BUDGET'):
            model.forecast([row('a' + str(i)) for i in range(100)], 0, 1, drain_deadline_ns=2)


class OwnerReadyCheckpointIndependentReview(unittest.TestCase):
    def model(self, start_provider=None):
        aliases = {a: dict(supported_tasks=['text']) for a in 'ab'}
        model = V9Model(aliases, {('a', 'text', 'x'): plan(5), ('b', 'text', 'x'): plan(100)},
            allow_profiled_replay=True, allow_context_time_slice_hypothesis=True,
            lifecycle_provider=Lifecycle(ready_ns=2), request_start_provider=start_provider,
            checkpoint_binding=dict(execution_context_sha256='a' * 64, evidence=[receipt(__file__)]))
        return model, [row('a1'), row('a2'), row('b1', arrival=10)]

    @staticmethod
    def reseal(checkpoint, state):
        checkpoint['state'] = encode(state)
        checkpoint['checkpoint_sha256'] = digest({k: v for k, v in checkpoint.items() if k != 'checkpoint_sha256'})

    def test_historical_start_must_follow_previous_owner_ready_with_any_provider(self):
        for provider in (None, AbsoluteStartSupply()):
            with self.subTest(provider=type(provider).__name__):
                model, rows = self.model(provider)
                checkpoint = model.forecast(rows, 0, 50, drain_deadline_ns=200,
                                            pause_after_event_count=4)['checkpoint']
                state = decode(checkpoint['state'])
                self.assertEqual(state['starts']['a2'], 7)
                self.assertEqual(state['clock'], 10)
                state['finishes']['a1']['worker_next_ready_ns'] = 8
                self.reseal(checkpoint, state)
                model, rows = self.model(provider)
                with self.assertRaises(ValueError):
                    model.forecast(rows, 0, 50, drain_deadline_ns=200, checkpoint=checkpoint)

    def test_historical_owner_start_order_must_match_original_fifo(self):
        model, rows = self.model()
        checkpoint = model.forecast(rows, 0, 50, drain_deadline_ns=200,
                                    pause_after_event_count=4)['checkpoint']
        state = decode(checkpoint['state'])
        finished = state['finishes'].pop('a1')
        finished['request_id'] = 'a2'
        state['finishes']['a2'] = finished
        state['running']['a']['row']['request_id'] = 'a1'
        state['starts']['a1'], state['starts']['a2'] = state['starts']['a2'], state['starts']['a1']
        state['queue_wait']['a1'], state['queue_wait']['a2'] = state['queue_wait']['a2'], state['queue_wait']['a1']
        self.reseal(checkpoint, state)
        model, rows = self.model()
        with self.assertRaises(ValueError):
            model.forecast(rows, 0, 50, drain_deadline_ns=200, checkpoint=checkpoint)

    def test_pending_owner_ready_must_match_bound_lifecycle_provider(self):
        model, rows = self.model()
        checkpoint = model.forecast(rows, 0, 50, drain_deadline_ns=200,
                                    pause_after_event_count=2)['checkpoint']
        state = decode(checkpoint['state'])
        self.assertEqual(state['clock'], 5)
        self.assertEqual(state['receipt_pending']['a1']['worker_next_ready_ns'], 7)
        state['receipt_pending']['a1']['worker_next_ready_ns'] = 6
        state['unready']['a'] = 6
        state['wake_events'] = [(6, 'a')]
        self.reseal(checkpoint, state)
        model, rows = self.model()
        with self.assertRaises(ValueError):
            model.forecast(rows, 0, 50, drain_deadline_ns=200, checkpoint=checkpoint)


if __name__ == '__main__':
    unittest.main()
