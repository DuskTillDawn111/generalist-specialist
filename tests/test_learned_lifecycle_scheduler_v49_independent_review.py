"""Independent real numeric lifecycle/controller regressions for v49."""
import copy
import math
from pathlib import Path
import tempfile
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tests.test_learned_context_oracle_v14 import Handoff
from tests.test_projected_handler_context_v42_independent_review import candidate
from tools.predictor_validation_20261027 import projected_handler_bridge_v44 as bridge
from tools.predictor_validation_20261027.cumulative_lifecycle_hazard_v24 import CumulativeLifecycleHazard
from tools.predictor_validation_20261027.dag_phase_scheduler_v8 import ActiveGpuNode
from tools.predictor_validation_20261027.learned_lifecycle_scheduler_v49 import LearnedLifecycleOracle, LearnedLifecycleScheduler
from tools.predictor_validation_20261027.owner_lifecycle_engine_v21 import LifecycleSchedulingView
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import OwnerLifecycle, receipt
from tools.predictor_validation_20261027.projected_handler_context_v42 import ObservableContextCandidate, OWNERS, EDGES_NS
from tools.predictor_validation_20261027.profiling import digest


class IndependentReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.source = Path(temporary.name) / 'numeric-fixture'
        self.source.write_text('Real v42 numeric CPU fixture, no fitted artifact')

    def scheduler(self, *, lower_rate_on_peer=False):
        numeric = candidate()
        if lower_rate_on_peer:
            model = copy.deepcopy(numeric.model)
            hazard = model['fitted']['hazard']
            hazard['columns'].append('handler:text#1:true')
            hazard['coefficients'].append(-math.log(4.))
            model['model_sha256'] = digest({k:v for k,v in model.items() if k != 'model_sha256'})
            numeric = ObservableContextCandidate(model)
        hazard = object.__new__(bridge.ObservableSaveHazard)
        hazard.candidate = numeric
        controller = CumulativeLifecycleHazard(hazard, age_edges_ns=EDGES_NS,
            sources=[receipt(self.source)], candidate_identity='a'*64,
            allow_unvalidated_approximation=True)
        mark = object.__new__(bridge.ObservableReferenceMark)
        mark.candidate = numeric
        mark.states = mark.clock = None
        oracle = LearnedLifecycleOracle(initial_prior=None, save_controller=controller,
            next_owner=mark, handoff=Handoff(), public_setup={}, public_workload=[],
            source_identity='a'*64, initial_quantile=.5,
            initial_readiness_rule='lower_bound_hypothesis',
            handoff_readiness_rule='lower_bound_reference_availability_hypothesis',
            allow_initial_pooling=True, allow_observer_transfer=True,
            allow_flagged_statistical_extrapolation=True)
        scheduler = LearnedLifecycleScheduler(OWNERS, oracle=oracle)
        scheduler._event(0, 'RESTORE_START', 'text#0', 'CPU_FIXTURE_MODELLED_STATE')
        scheduler._event(0, 'SERVICE_READY', 'text#0', 'CPU_FIXTURE_MODELLED_STATE')
        return scheduler

    def view(self, clock, *, frontier=(), handlers=('text#1',)):
        states = {owner:OwnerLifecycle(owner in handlers,
            None if owner in handlers else False, False) for owner in OWNERS}
        active = tuple(ActiveGpuNode((owner,'node'), 1e9, 'decode', 'r', 'text', 'pool')
            for owner in frontier)
        return LifecycleSchedulingView(clock, active, tuple(states.items()), (), ())

    def test_absent_and_expired_review_publish_future_event_with_empty_frontier(self):
        for old in (None, 1):
            with self.subTest(old=old):
                scheduler = self.scheduler()
                scheduler.review_deadline_ns = old
                decision = scheduler.decide(self.view(1))
                future = scheduler.oracle.hazard['next_review_ns']
                self.assertGreater(future, 1)
                self.assertEqual(decision.next_transition_ns, future)
                self.assertEqual(scheduler.review_deadline_ns, future)
                self.assertEqual(decision.gpu_rates, {})
                self.assertEqual(scheduler.expired_review_republication_count, int(old is not None))

    def test_future_review_is_kept_when_new_hazard_SAVE_moves_later(self):
        scheduler = self.scheduler(lower_rate_on_peer=True)
        original = scheduler.decide(self.view(0, handlers=('text#0',))).next_transition_ns
        changed = scheduler.decide(self.view(1000, handlers=('text#0','text#1')))
        hazard = scheduler.oracle.hazard
        self.assertGreater(hazard['absolute_save_candidate_ns'], original)
        self.assertEqual(changed.next_transition_ns, original)
        self.assertEqual(scheduler.review_deadline_ns, original)
        self.assertEqual(scheduler.expired_review_republication_count, 0)
        # The old candidate review is a required engine wakeup. On its expiry,
        # publish the later candidate without discarding the accumulated risk.
        budget, consumed = hazard['hazard_budget'], hazard['consumed_hazard']
        advanced = scheduler.decide(self.view(original, handlers=('text#0','text#1')))
        self.assertGreater(advanced.next_transition_ns, original)
        self.assertEqual(advanced.next_transition_ns, scheduler.oracle.hazard['absolute_save_candidate_ns'])
        self.assertEqual(scheduler.oracle.hazard['hazard_budget'], budget)
        self.assertGreater(scheduler.oracle.hazard['consumed_hazard'], consumed)
        self.assertEqual(scheduler.oracle.hazard['origin_restore_ns'], 0)
        self.assertEqual(scheduler.oracle.hazard['restore_age_ns'], original)
        self.assertFalse(scheduler.oracle.hazard['budget_redrawn'])
        self.assertEqual(scheduler.expired_review_republication_count, 1)

    def test_active_frontier_ordinary_deadline_and_reference_rate_preserved(self):
        scheduler = self.scheduler()
        scheduler.review_deadline_ns = 1
        decision = scheduler.decide(self.view(1, frontier=('text#0','text#1')))
        self.assertEqual(decision.next_transition_ns, scheduler.oracle.hazard['next_review_ns'])
        self.assertEqual(decision.gpu_rates, {('text#0','node'):1., ('text#1','node'):0.})
        self.assertEqual(scheduler.expired_review_republication_count, 0)

    def test_exact_SAVE_during_empty_frontier_keeps_real_mark_and_pending_switch(self):
        scheduler = self.scheduler()
        scheduler.review_deadline_ns = 1
        crossing = scheduler.decide(self.view(1)).next_transition_ns
        at_save = scheduler.decide(self.view(crossing))
        self.assertEqual(at_save.next_transition_ns, crossing+2)
        pending, events = copy.deepcopy(scheduler.pending_switch), copy.deepcopy(scheduler.context_events)
        before_restore = scheduler.decide(self.view(crossing+1))
        self.assertEqual(before_restore.next_transition_ns, crossing+2)
        self.assertEqual(scheduler.pending_switch, pending)
        self.assertEqual(scheduler.context_events, events)
        self.assertEqual(scheduler.oracle.decisions[0]['next_owner'], 'text#1')
        self.assertEqual(scheduler.oracle.decisions[0]['mark']['physical_owner_probabilities']['text#0'], .6)

    def test_same_held_UNKNOWN_is_preserved(self):
        scheduler = self.scheduler()
        crossing = scheduler.decide(self.view(0, handlers=('text#0',))).next_transition_ns
        with self.assertRaisesRegex(UnknownEvidence, 'SAME_CONTEXT_REACQUISITION'):
            scheduler.decide(self.view(crossing, handlers=('text#0',)))
        self.assertEqual(scheduler.oracle.decisions, [])
        self.assertEqual(scheduler.expired_review_republication_count, 0)

    def test_late_SAVE_keeps_no_rewind_UNKNOWN_before_repair(self):
        scheduler = self.scheduler()
        scheduler.review_deadline_ns = 1
        crossing = scheduler.decide(self.view(1)).next_transition_ns
        with self.assertRaisesRegex(UnknownEvidence, 'ENGINE_SKIPPED_PREDICTED_SAVE_BOUNDARY_NO_REWIND'):
            scheduler.decide(self.view(crossing+1000))
        self.assertEqual(scheduler.restore_start_ns, 0)
        self.assertEqual(scheduler.oracle.decisions, [])
        self.assertEqual(scheduler.expired_review_republication_count, 1)


if __name__ == '__main__':
    unittest.main()
