"""Physical identity is separate from request readiness and retained switch state."""
import copy
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tests import test_learned_lifecycle_scheduler_v49_independent_review as numeric_fixture
from tools.predictor_validation_20261027.physical_context_mark_bridge_v56 import PhysicalContextMark
from tools.predictor_validation_20261027.learned_lifecycle_scheduler_v49 import LearnedLifecycleScheduler
from tools.predictor_validation_20261027.whole57_shared_observable_risk_v31 import OWNERS


class IndependentPhysicalSemanticsTests(unittest.TestCase):
    def scheduler(self, holder='text#1'):
        fixture=numeric_fixture.IndependentReviewTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        oracle=fixture.scheduler().oracle
        original=oracle.next_owner
        mark=object.__new__(PhysicalContextMark)
        mark.candidate,mark.clock,mark.states=original.candidate,None,None
        oracle.next_owner=mark
        scheduler=LearnedLifecycleScheduler(OWNERS,oracle=oracle)
        scheduler._event(0,'RESTORE_START',holder,'INDEPENDENT_CPU_FIXTURE')
        scheduler._event(0,'SERVICE_READY',holder,'INDEPENDENT_CPU_FIXTURE')
        return fixture,scheduler

    def test_handler_change_during_pending_never_retargets_or_credits_idle_owner(self):
        fixture,scheduler=self.scheduler()
        crossing=scheduler.decide(fixture.view(0,handlers=('text#1',),frontier=('text#1',))).next_transition_ns
        decision=scheduler.decide(fixture.view(crossing,handlers=('text#1',),frontier=('text#1',)))
        self.assertEqual(scheduler.pending_switch.target_owner,'text#0')
        self.assertEqual(decision.gpu_rates,{('text#1','node'):0.})
        pending=copy.deepcopy(scheduler.pending_switch)
        hazard=copy.deepcopy(scheduler.oracle.hazard)
        # The pending target becomes runnable and its competitor disappears,
        # but a pending physical SAVE/RESTORE cannot be replaced or advanced.
        middle=scheduler.decide(fixture.view(crossing+1,handlers=('text#0',),frontier=('text#0',)))
        self.assertEqual(middle.gpu_rates,{('text#0','node'):0.})
        self.assertEqual(scheduler.pending_switch,pending)
        self.assertEqual(scheduler.oracle.hazard,hazard)
        restored=scheduler.decide(fixture.view(crossing+2,handlers=('text#1',),frontier=('text#1',)))
        self.assertEqual(scheduler.held_owner,'text#0')
        self.assertEqual(restored.gpu_rates,{('text#1','node'):0.})
        # The restored owner's eligibility remains false; the mark has not
        # fabricated work or given its service to another owner's request.
        self.assertFalse(scheduler.oracle.next_owner.states['text#0'].active_service_eligible)
        runnable=scheduler.decide(fixture.view(crossing+3,handlers=('text#0',),frontier=('text#0',)))
        self.assertEqual(runnable.gpu_rates,{('text#0','node'):1.})

    def test_all_four_active_frontiers_retained_without_global_request_cap(self):
        fixture,scheduler=self.scheduler('text#0')
        decision=scheduler.decide(fixture.view(0,handlers=OWNERS,frontier=OWNERS))
        self.assertEqual(set(decision.gpu_rates),{(owner,'node') for owner in OWNERS})
        self.assertEqual(decision.gpu_rates[('text#0','node')],1.)
        self.assertEqual(sum(decision.gpu_rates.values()),1.)
        # Binary reference progress is a physical-context approximation;
        # it does not drop or reject the other three concurrent requests.
        self.assertEqual(len(scheduler.snapshot(fixture.view(0,handlers=OWNERS,frontier=OWNERS)).pending_frontier),4)

    def test_same_holder_mass_is_retained_then_explicitly_refused(self):
        fixture,scheduler=self.scheduler('text#0')
        crossing=scheduler.decide(fixture.view(0,handlers=(),frontier=())).next_transition_ns
        with self.assertRaisesRegex(UnknownEvidence,'SAME_CONTEXT_REACQUISITION'):
            scheduler.decide(fixture.view(crossing,handlers=(),frontier=()))
        mark=scheduler.oracle.next_owner.predict_given_save_now(held_owner='text#0',context_age_ns=crossing,pending_frontier=[])
        self.assertEqual(mark['most_likely_next_owner'],'text#0')
        self.assertGreater(mark['next_owner_probabilities']['text#0'],.5)
        self.assertFalse(mark['same_holder_probability_removed'])
        self.assertIsNone(scheduler.pending_switch)
        self.assertEqual(scheduler.oracle.decisions,[])


if __name__=='__main__':unittest.main()
