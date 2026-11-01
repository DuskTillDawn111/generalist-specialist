"""Independent held/pending context and absolute event boundary cases."""
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027.dag_phase_scheduler_v8 import ActiveGpuNode,SchedulingView
from tools.predictor_validation_20261027.held_context_scheduler_v12_1 import ContextAvailabilityDecision,HeldContextScheduler


def view(clock,owners):
    return SchedulingView(clock,tuple(ActiveGpuNode((owner,'gpu'),100.,'phase',owner,'text','exact') for owner in owners))


class HeldContextIndependentTests(unittest.TestCase):
    def scheduler(self,callback):
        class Oracle:
            def decide(self,q):return callback(q)
        scheduler=HeldContextScheduler(('a','b'),oracle=Oracle(),allow_source_replay=True)
        scheduler.observe_context_event(0,'RESTORE_START','a');scheduler.observe_context_event(0,'SERVICE_READY','a')
        return scheduler

    def test_same_clock_atomic_save_restore_service_and_empty_age_are_preserved(self):
        scheduler=self.scheduler(lambda q:ContextAvailabilityDecision('switch',20,'b',5,5,5))
        result=scheduler.decide(view(5,('a','b')))
        self.assertEqual(result.gpu_rates,{('a','gpu'):0.,('b','gpu'):1.})
        self.assertEqual(scheduler.restore_start_ns,5)
        self.assertEqual([r['kind'] for r in scheduler.transitions[-3:]],['SAVE_END','RESTORE_START','SERVICE_READY'])
        empty=scheduler.snapshot(view(12,()))
        self.assertEqual(empty.context_age_ns,7);self.assertEqual(empty.held_owner,'b')
        again=scheduler.decide(view(12,('a','b')))
        self.assertEqual(again.next_transition_ns,20);self.assertEqual(scheduler.restore_start_ns,5)
        self.assertEqual(len(scheduler.queries),1)

    def test_host_calls_do_not_postpone_pending_absolute_context_events(self):
        scheduler=self.scheduler(lambda q:ContextAvailabilityDecision('switch',30,'b',10,12,15)
            if q.clock_ns==1 else ContextAvailabilityDecision('hold',30))
        self.assertEqual(scheduler.decide(view(1,('a','b'))).next_transition_ns,10)
        for clock in (2,2,5,9):
            scheduler.decide(view(clock,('b',)))
            self.assertEqual(scheduler.held_owner,'a');self.assertEqual(scheduler.restore_start_ns,0)
        self.assertEqual(scheduler.decide(view(10,('b',))).next_transition_ns,12)
        self.assertIsNone(scheduler.held_owner)
        self.assertEqual(scheduler.decide(view(12,('b',))).next_transition_ns,15)
        self.assertEqual(scheduler.restore_start_ns,12);self.assertFalse(scheduler.service_available)
        self.assertEqual(scheduler.decide(view(15,('b',))).gpu_rates,{('b','gpu'):1.})
        self.assertEqual(len(scheduler.queries),2)

    def test_rejected_decision_does_not_publish_context_switch_or_renew_review(self):
        def oracle(q):
            if q.clock_ns==1:return ContextAvailabilityDecision('hold',20)
            return ContextAvailabilityDecision('switch',100,'b',50,60,70)
        scheduler=self.scheduler(oracle);scheduler.decide(view(1,('a',)))
        previous=(scheduler.held_owner,scheduler.restore_start_ns,scheduler.review_deadline_ns,
                  list(scheduler.transitions),list(scheduler.queries),scheduler.pending_switch,list(scheduler.context_events))
        with self.assertRaises(UnknownEvidence):scheduler.decide(view(5,('a','b')))
        actual=(scheduler.held_owner,scheduler.restore_start_ns,scheduler.review_deadline_ns,
                scheduler.transitions,scheduler.queries,scheduler.pending_switch,scheduler.context_events)
        self.assertEqual(actual,previous)


if __name__=='__main__':unittest.main()
