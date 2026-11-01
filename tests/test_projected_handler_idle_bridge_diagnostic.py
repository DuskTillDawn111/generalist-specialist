"""Reproduce frozen v44 rejection of a SAVE with no engine-owned work."""
import copy
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tests import test_learned_lifecycle_scheduler_v49_independent_review as numeric_fixture
from tests.test_projected_handler_context_v42_independent_review import candidate
from tools.predictor_validation_20261027 import projected_handler_bridge_v44 as bridge
from tools.predictor_validation_20261027.learned_lifecycle_scheduler_v27 import LearnedLifecycleScheduler
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import OwnerLifecycle, engine_states
from tools.predictor_validation_20261027.projected_handler_context_v42 import OWNERS


class IdleBridgeDiagnostic(unittest.TestCase):
    def test_settled_no_handler_or_activity_means_false_not_unknown(self):
        idle = engine_states(OWNERS, {}, {})
        self.assertTrue(all(state == OwnerLifecycle(False, False, False) for state in idle.values()))
        self.assertTrue(all(state.active_service_eligible is False for state in idle.values()))
        # A handler awaiting future DAG release is still owned work. An empty
        # GPU-ready set alone must never be interpreted as all-owner idle.
        waiting = engine_states(OWNERS, {'text#0':object()}, {})
        self.assertEqual(waiting['text#0'], OwnerLifecycle(True, None, False))
        self.assertIs(waiting['text#0'].active_service_eligible, True)
        self.assertIs(OwnerLifecycle(None, False, False).active_service_eligible, None)

    def test_frozen_bridge_requires_active_target_even_when_physical_mark_is_complete(self):
        mark = object.__new__(bridge.ObservableReferenceMark)
        mark.candidate = candidate()
        mark.states = engine_states(OWNERS, {}, {})
        mark.clock = 10
        physical = mark.candidate.next_owner(bridge.feature_projection('text#1', None, 10, mark.states))
        self.assertEqual(set(physical['physical_owner_probabilities']), set(OWNERS))
        self.assertAlmostEqual(sum(physical['physical_owner_probabilities'].values()), 1.)
        self.assertEqual(physical['most_likely_physical_owner'], 'text#0')
        original = copy.deepcopy(physical)
        with self.assertRaisesRegex(UnknownEvidence, 'NO_VISIBLE_ACTIVE_REFERENCE_TARGET'):
            mark.predict_given_save_now(held_owner='text#1', context_age_ns=10, pending_frontier=[])
        self.assertEqual(mark.candidate.next_owner(
            bridge.feature_projection('text#1', None, 10, mark.states)), original)

    def test_causal_handler_end_then_exact_idle_SAVE_reaches_frozen_guard(self):
        # Reuse fixture construction only, never inherit its unrelated tests.
        fixture = numeric_fixture.IndependentReviewTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        provisional = fixture.scheduler()
        scheduler = LearnedLifecycleScheduler(OWNERS, oracle=provisional.oracle)
        scheduler._event(0, 'RESTORE_START', 'text#0', 'CPU_FIXTURE_MODELLED_STATE')
        scheduler._event(0, 'SERVICE_READY', 'text#0', 'CPU_FIXTURE_MODELLED_STATE')
        scheduler.decide(fixture.view(0, handlers=('text#1',)))
        # Handler END is causally observed at 1000, before the predicted SAVE;
        # all lifecycle inputs are supplied at their own current boundary.
        scheduler.decide(fixture.view(1000, handlers=()))
        crossing = scheduler.oracle.hazard['absolute_save_candidate_ns']
        self.assertGreater(crossing, 1000)
        budget = scheduler.oracle.hazard['hazard_budget']
        with self.assertRaisesRegex(UnknownEvidence, 'NO_VISIBLE_ACTIVE_REFERENCE_TARGET'):
            scheduler.decide(fixture.view(crossing, handlers=()))
        self.assertEqual(scheduler.oracle.hazard['consumed_hazard'], budget)
        self.assertEqual(scheduler.oracle.lifecycle_clock_ns, crossing)
        self.assertEqual(scheduler.restore_start_ns, 0)
        self.assertIsNone(scheduler.pending_switch)
        self.assertEqual(scheduler.oracle.decisions, [])
        self.assertTrue(all(state.active_service_eligible is False
            for state in scheduler.oracle.lifecycle.values()))


if __name__ == '__main__':
    unittest.main()
