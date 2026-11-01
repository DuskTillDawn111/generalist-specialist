"""Independent causal/UNKNOWN checks of the explicit all-known-idle bridge."""
import ast
import copy
from pathlib import Path
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tests import test_learned_lifecycle_scheduler_v49_independent_review as fixture
from tests.test_projected_handler_context_v42_independent_review import candidate
from tools.predictor_validation_20261027 import idle_physical_mark_bridge_v51 as idle
from tools.predictor_validation_20261027 import projected_handler_bridge_v44 as old
from tools.predictor_validation_20261027 import projected_handler_forecast_v50 as v50
from tools.predictor_validation_20261027 import projected_handler_forecast_v52 as v52
from tools.predictor_validation_20261027 import projected_handler_preflight_v50 as p50
from tools.predictor_validation_20261027 import projected_handler_preflight_v52 as p52
from tools.predictor_validation_20261027.learned_lifecycle_scheduler_v49 import LearnedLifecycleScheduler
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import OwnerLifecycle, FIELDS, engine_states
from tools.predictor_validation_20261027.projected_handler_context_v42 import OWNERS


class IndependentIdleBridgeTests(unittest.TestCase):
    def mark(self, states=None):
        mark = object.__new__(idle.ObservableReferenceMark)
        mark.candidate = candidate()
        mark.states = engine_states(OWNERS, {}, {}) if states is None else states
        mark.clock = 1000
        return mark

    def query(self, mark, **kwargs):
        return mark.predict_given_save_now(held_owner='text#1', context_age_ns=1000,
            pending_frontier=[], **kwargs)

    def scheduler(self, holder='text#1'):
        helper = fixture.IndependentReviewTests()
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        original = helper.scheduler()
        original.oracle.next_owner = self.mark()
        original.oracle.next_owner.clock = None
        original.oracle.next_owner.states = None
        scheduler = LearnedLifecycleScheduler(OWNERS, oracle=original.oracle)
        scheduler._event(0, 'RESTORE_START', holder, 'CPU_FIXTURE_MODELLED_STATE')
        scheduler._event(0, 'SERVICE_READY', holder, 'CPU_FIXTURE_MODELLED_STATE')
        return helper, scheduler

    def test_only_complete_false_empty_domain_uses_raw_probabilities_and_no_service_target(self):
        mark = self.mark()
        result = self.query(mark)
        physical = mark.candidate.next_owner(old.feature_projection('text#1', None, 1000, mark.states))
        self.assertEqual(result['physical_owner_probabilities'], physical['physical_owner_probabilities'])
        self.assertEqual(result['next_owner_probabilities'], physical['physical_owner_probabilities'])
        self.assertEqual(result['most_likely_next_owner'], 'text#0')
        self.assertIsNone(result['active_reference_probabilities'])
        self.assertIsNone(result['active_reference_target'])
        self.assertFalse(result['reference_service_progress_assigned'])
        self.assertIn('UNVALIDATED_ALL_IDLE_PHYSICAL_TRANSITION_BRIDGE', result['ood_flags'])
        # Public future identity/arrival/outcome fields cannot change the mark.
        self.assertEqual(result, self.query(mark, public_workload=[['ocr#0','future-output',999999999]]))
        with self.assertRaisesRegex(UnknownEvidence, 'ALL_IDLE_LIFECYCLE_CONFLICTS_WITH_GPU_FRONTIER'):
            mark.predict_given_save_now(held_owner='text#1', context_age_ns=1000,
                pending_frontier=[['text#0','text','pool',['decode']]])

    def test_each_unknown_field_remains_UNKNOWN_and_nonidle_fallback_matches_v44(self):
        for owner in OWNERS:
            for field in FIELDS:
                with self.subTest(owner=owner, field=field):
                    states = engine_states(OWNERS, {}, {})
                    fields = dict(handler_active=False, issued_outstanding=False, gpu_ready=False)
                    fields[field] = None
                    states[owner] = OwnerLifecycle(**fields)
                    # Unknown handler projection is rejected earlier by the
                    # frozen feature validator; other unknown fields reach
                    # the frozen UNKNOWN owner-domain guard.
                    error = ValueError if field == 'handler_active' else UnknownEvidence
                    message = 'Complete worker-handler boolean' if field == 'handler_active' else 'UNKNOWN_OWNER_WORK_DOMAIN'
                    with self.assertRaisesRegex(error, message):
                        self.query(self.mark(states))
        states = engine_states(OWNERS, {'text#1':object()}, {})
        new = self.mark(states)
        frozen = object.__new__(old.ObservableReferenceMark)
        frozen.candidate, frozen.states, frozen.clock = new.candidate, states, new.clock
        self.assertEqual(self.query(new), self.query(frozen))
        self.assertEqual(self.query(new)['most_likely_next_owner'], 'text#1')

    def test_idle_switch_and_pending_arrival_preserve_budget_age_gap_and_immutability(self):
        helper, scheduler = self.scheduler()
        scheduler.decide(helper.view(0, handlers=('text#0',)))
        scheduler.decide(helper.view(1000, handlers=()))
        hazard = scheduler.oracle.hazard
        self.assertGreater(hazard['consumed_hazard'], 0)
        crossing, budget = hazard['absolute_save_candidate_ns'], hazard['hazard_budget']
        decision = scheduler.decide(helper.view(crossing, handlers=()))
        self.assertEqual(decision.gpu_rates, {})
        self.assertEqual(decision.next_transition_ns, crossing+2)
        self.assertEqual(scheduler.oracle.hazard['consumed_hazard'], budget)
        self.assertEqual(scheduler.oracle.hazard['origin_restore_ns'], 0)
        self.assertEqual(scheduler.oracle.hazard['restore_age_ns'], crossing)
        transition = scheduler.oracle.decisions[-1]
        self.assertIsNone(transition['mark']['active_reference_target'])
        self.assertEqual(transition['next_owner'], 'text#0')
        self.assertEqual(transition['handoff']['save_to_restore_gap_ns'], 2)
        pending, events = copy.deepcopy(scheduler.pending_switch), copy.deepcopy(scheduler.context_events)
        arrived = scheduler.decide(helper.view(crossing+1, handlers=('text#0',), frontier=('text#0',)))
        self.assertEqual(arrived.gpu_rates, {('text#0','node'):0.})
        self.assertEqual(arrived.next_transition_ns, crossing+2)
        self.assertEqual(scheduler.pending_switch, pending)
        self.assertEqual(scheduler.context_events, events)
        self.assertEqual(scheduler.oracle.hazard['consumed_hazard'], budget)
        self.assertEqual(len(scheduler.oracle.decisions), 1)
        restored = scheduler.decide(helper.view(crossing+2, handlers=('text#0',), frontier=('text#0',)))
        self.assertEqual(restored.gpu_rates, {('text#0','node'):1.})
        self.assertEqual(scheduler.restore_start_ns, crossing+2)
        self.assertEqual(scheduler.oracle.hazard['origin_restore_ns'], crossing+2)
        self.assertEqual(scheduler.oracle.hazard['observation_ns'], crossing+2)
        self.assertFalse(scheduler.oracle.hazard['budget_redrawn'])

    def test_idle_raw_same_holder_and_late_SAVE_keep_original_UNKNOWN_guards(self):
        for holder, late, reason in [('text#0',0,'SAME_CONTEXT_REACQUISITION'),
                ('text#1',100,'ENGINE_SKIPPED_PREDICTED_SAVE_BOUNDARY_NO_REWIND')]:
            with self.subTest(holder=holder, late=late):
                helper, scheduler = self.scheduler(holder)
                crossing = scheduler.decide(helper.view(0, handlers=())).next_transition_ns
                with self.assertRaisesRegex(UnknownEvidence, reason):
                    scheduler.decide(helper.view(crossing+late, handlers=()))
                self.assertIsNone(scheduler.pending_switch)
                self.assertEqual(scheduler.oracle.decisions, [])

    def test_v52_factory_and_dedicated_process_runner_only_change_declared_wiring(self):
        pairs = [(v50,v52), (p50,p52)]
        for before, after in pairs:
            expected = Path(before.__file__).read_text().replace('v50','v52').replace('V50','V52').replace('version=50','version=52')
            expected = expected.replace('from .projected_handler_bridge_v44 import ObservableSaveHazard, ObservableReferenceMark',
                'from .idle_physical_mark_bridge_v51 import ObservableSaveHazard, ObservableReferenceMark')
            if before is v50:
                expected = expected.replace('expired_empty_frontier_reviews_republished=True,',
                    'expired_empty_frontier_reviews_republished=True, all_known_idle_physical_transitions_separated_from_service=True,')
            trees = [ast.parse(text) for text in (expected, Path(after.__file__).read_text())]
            for tree in trees:tree.body = tree.body[1:]
            self.assertEqual(ast.dump(trees[0]), ast.dump(trees[1]))


if __name__ == '__main__':
    unittest.main()
