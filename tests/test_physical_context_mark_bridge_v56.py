"""Physical mark labels must survive request-domain changes without filtering."""
import copy
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tests import test_learned_lifecycle_scheduler_v49_independent_review as clock_fixture
from tests.test_projected_handler_context_v42_independent_review import candidate
from tools.predictor_validation_20261027.physical_context_mark_bridge_v56 import PhysicalContextMark
from tools.predictor_validation_20261027.projected_handler_bridge_v44 import ObservableReferenceMark, feature_projection
from tools.predictor_validation_20261027.learned_lifecycle_scheduler_v49 import LearnedLifecycleScheduler
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import OwnerLifecycle, engine_states
from tools.predictor_validation_20261027.whole57_shared_observable_risk_v31 import OWNERS


class PhysicalMarkTests(unittest.TestCase):
    def mark(self, states, cls=PhysicalContextMark):
        mark = object.__new__(cls)
        mark.candidate, mark.states, mark.clock = candidate(), states, 10
        return mark

    def query(self, mark, holder='text#1', **kwargs):
        return mark.predict_given_save_now(held_owner=holder, context_age_ns=1000,
            pending_frontier=[], **kwargs)

    def scheduler(self, holder='text#1'):
        helper = clock_fixture.IndependentReviewTests()
        helper.setUp(); self.addCleanup(helper.doCleanups)
        oracle = helper.scheduler().oracle
        oracle.next_owner = self.mark(None)
        oracle.next_owner.clock = None
        scheduler = LearnedLifecycleScheduler(OWNERS, oracle=oracle)
        scheduler._event(0, 'RESTORE_START', holder, 'CPU_FIXTURE_NOT_MEASURED')
        scheduler._event(0, 'SERVICE_READY', holder, 'CPU_FIXTURE_NOT_MEASURED')
        return helper, scheduler

    def test_sole_held_eligibility_no_longer_converts_small_held_mass_to_one(self):
        states = engine_states(OWNERS, {'text#1':object()}, {})
        old = self.query(self.mark(states, ObservableReferenceMark))
        new = self.query(self.mark(states))
        self.assertEqual(old['most_likely_next_owner'], 'text#1')
        self.assertEqual(old['next_owner_probabilities']['text#1'], 1.)
        self.assertEqual(new['most_likely_next_owner'], 'text#0')
        self.assertFalse(new['physical_target_currently_service_eligible'])
        physical = self.mark(states).candidate.next_owner(feature_projection('text#1', None, 1000, states))
        self.assertEqual(new['next_owner_probabilities'], physical['physical_owner_probabilities'])
        self.assertGreater(new['next_owner_probabilities']['text#1'], 0)
        self.assertFalse(new['same_holder_probability_removed'])
        self.assertIsNone(new['active_reference_target'])

    def test_issuance_and_frontier_are_not_physical_target_filters(self):
        states = engine_states(OWNERS, {}, {})
        base = self.query(self.mark(states))
        altered = copy.deepcopy(states)
        altered['text#0'] = OwnerLifecycle(False, None, False)
        self.assertEqual(base['next_owner_probabilities'], self.query(self.mark(altered))['next_owner_probabilities'])
        # Unobserved handler status is a required physical model feature.
        altered['text#0'] = OwnerLifecycle(None, None, False)
        with self.assertRaisesRegex(ValueError, 'Complete worker-handler boolean'):
            self.query(self.mark(altered))
        self.assertEqual(base, self.query(self.mark(states), public_workload=[['future-outcome', 99999999999]]))

    def test_inactive_physical_restore_does_not_advance_another_owners_work(self):
        helper, scheduler = self.scheduler()
        first = scheduler.decide(helper.view(0, handlers=('text#1',), frontier=('text#1',)))
        crossing = first.next_transition_ns
        decision = scheduler.decide(helper.view(crossing, handlers=('text#1',), frontier=('text#1',)))
        self.assertEqual(decision.gpu_rates, {('text#1','node'):0.})
        self.assertEqual(scheduler.pending_switch.target_owner, 'text#0')
        budget = scheduler.oracle.hazard['hazard_budget']
        self.assertEqual(scheduler.oracle.hazard['consumed_hazard'], budget)
        pending = copy.deepcopy(scheduler.pending_switch)
        mid = scheduler.decide(helper.view(crossing+1, handlers=('text#1',), frontier=('text#1',)))
        self.assertEqual(mid.gpu_rates, {('text#1','node'):0.})
        self.assertEqual(scheduler.pending_switch, pending)
        restored = scheduler.decide(helper.view(crossing+2, handlers=('text#1',), frontier=('text#1',)))
        self.assertEqual(restored.gpu_rates, {('text#1','node'):0.})
        self.assertEqual(scheduler.held_owner, 'text#0')
        self.assertEqual(scheduler.restore_start_ns, crossing+2)
        self.assertGreater(restored.next_transition_ns, crossing+2)

    def test_raw_same_holder_and_skipped_save_remain_unknown(self):
        for holder, late, reason in [('text#0', 0, 'SAME_CONTEXT_REACQUISITION'),
                ('text#1', 100, 'ENGINE_SKIPPED_PREDICTED_SAVE_BOUNDARY_NO_REWIND')]:
            with self.subTest(holder=holder):
                helper, scheduler = self.scheduler(holder)
                crossing = scheduler.decide(helper.view(0, handlers=())).next_transition_ns
                with self.assertRaisesRegex(UnknownEvidence, reason):
                    scheduler.decide(helper.view(crossing+late, handlers=()))
                self.assertIsNone(scheduler.pending_switch)
                self.assertEqual(scheduler.oracle.decisions, [])


if __name__ == '__main__':
    unittest.main()
