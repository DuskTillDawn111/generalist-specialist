import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tests.test_held_context_scheduler_v12_1 import view
from tools.predictor_validation_20261027.held_context_scheduler_v12_1 import ContextAvailabilityDecision
from tools.predictor_validation_20261027.held_context_scheduler_v14 import HeldContextScheduler, InitialContextDecision


class Oracle:
    def __init__(self, initial, action):
        self.initial = initial; self.action = action; self.initial_queries = []; self.queries = []
    def initialize(self, query):
        self.initial_queries.append(query); return self.initial
    def decide(self, query):
        self.queries.append(query); return self.action(query)


class InitialContextSchedulerTests(unittest.TestCase):
    def test_holder_and_restore_age_are_joint_model_decision_not_only_ready_owner(self):
        oracle = Oracle(InitialContextDecision('text#1',12,0,'a'*64),
                        lambda q: ContextAvailabilityDecision('hold',20))
        scheduler = HeldContextScheduler(('text#0','text#1'),oracle=oracle)
        decision = scheduler.decide(view(0))
        self.assertEqual(decision.gpu_rates[('text#0','gpu')],0)
        self.assertEqual(decision.next_transition_ns,20)
        self.assertEqual(scheduler.restore_start_ns,-12)
        self.assertEqual(oracle.queries[0].context_age_ns,12)
        scheduler.decide(view(5)); self.assertEqual(len(oracle.initial_queries),1)
        self.assertEqual(scheduler.restore_start_ns,-12)
        self.assertFalse(scheduler.provenance()['initial_restore_is_observed_history'])
        with self.assertRaises(ValueError): scheduler.observe_context_event(6,'SAVE_END','text#1')

    def test_initial_readiness_delay_cannot_be_overwritten_by_host_calls(self):
        oracle = Oracle(InitialContextDecision('text#0',7,10,'b'*64),
                        lambda q: ContextAvailabilityDecision('hold',20))
        scheduler = HeldContextScheduler(('text#0','text#1'),oracle=oracle)
        for time,owners in ((0,('text#0',)),(2,('text#1',)),(3,()),(9,('text#0','text#1'))):
            result = scheduler.decide(view(time,owners))
            self.assertEqual(result.next_transition_ns,10)
            self.assertFalse(any(result.gpu_rates.values()))
            self.assertEqual(scheduler.restore_start_ns,-7)
            self.assertEqual(oracle.queries,[])
        self.assertEqual(scheduler.decide(view(10)).gpu_rates[('text#0','gpu')],1)
        self.assertEqual(oracle.queries[0].context_age_ns,17)

    def test_save_restore_events_advance_through_empty_ready_sets(self):
        oracle = Oracle(InitialContextDecision('text#0',2,0,'c'*64),
                        lambda q: ContextAvailabilityDecision('switch',20,'text#1',3,5,7))
        scheduler = HeldContextScheduler(('text#0','text#1'),oracle=oracle)
        scheduler.decide(view(0))
        self.assertEqual(scheduler.decide(view(1,())).next_transition_ns,3)
        self.assertEqual(scheduler.decide(view(3,())).next_transition_ns,5)
        self.assertIsNone(scheduler.held_owner)
        self.assertEqual(scheduler.decide(view(5,())).next_transition_ns,7)
        self.assertEqual(scheduler.held_owner,'text#1')
        self.assertEqual(scheduler.decide(view(7,())).next_transition_ns,20)
        self.assertTrue(scheduler.service_available)

    def test_no_oracle_or_incomplete_decision_remains_unknown(self):
        with self.assertRaisesRegex(UnknownEvidence,'INITIAL_CONTEXT_PREDICTION_UNAVAILABLE'):
            HeldContextScheduler(('text#0',)).decide(view(0))
        for initial in (InitialContextDecision('foreign',0,0,'d'*64),
                        InitialContextDecision('text#0',-1,0,'d'*64),
                        InitialContextDecision('text#0',0,-1,'d'*64),
                        InitialContextDecision('text#0',0,0,'not-a-model')):
            scheduler = HeldContextScheduler(('text#0',),oracle=Oracle(initial,lambda q:None))
            with self.assertRaises(UnknownEvidence): scheduler.decide(view(0))
            self.assertEqual(scheduler.held_knowledge,'UNKNOWN')

    def test_interface_presence_does_not_qualify_a_complete_oracle(self):
        class Incomplete:
            initialize = None
        scheduler = HeldContextScheduler(('text#0',),oracle=Incomplete())
        result = scheduler.provenance()
        self.assertFalse(result['initialization_entrypoint_available'])
        self.assertFalse(result['complete_source_bound_oracle_qualified'])
        with self.assertRaisesRegex(UnknownEvidence,'INITIAL_CONTEXT_PREDICTION_UNAVAILABLE'):
            scheduler.decide(view(0))


if __name__ == '__main__': unittest.main()
