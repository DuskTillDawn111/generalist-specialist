import copy
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027.conditional_availability_scheduler_v8 import ConditionalAvailabilityScheduler
from tools.predictor_validation_20261027.context_envelope_diagnostic_v8 import IntervalUnion,partition
from tools.predictor_validation_20261027.dag_phase_scheduler_v8 import ActiveGpuNode,SchedulingView


def view(t, owners=('a','b'),phase='prefill'):
    return SchedulingView(t,tuple(ActiveGpuNode((a,'gpu'),10.,phase,a,'text','x') for a in owners))


def scheduler():
    state=[[a,'text','x',['prefill']] for a in ('a','b')]
    return ConditionalAvailabilityScheduler(dict(schema='gsdeploy.conditional_context_availability_policy.v8',
        reference_work_domain='SOLO_ONLY_REFERENCE_NS',training_only=True,whole_corun_multiplier_used=False,
        cells=[dict(frontier=state,current_owner=None,reason='initial',next_owner='a',grant_budget_ns=7,handoff_ns=0),
               dict(frontier=state,current_owner='a',reason='grant_expired',next_owner='b',grant_budget_ns=5,handoff_ns=2),
               dict(frontier=state,current_owner='b',reason='grant_expired',next_owner='a',grant_budget_ns=7,handoff_ns=2),
               dict(frontier=[state[1]],current_owner='a',reason='yield',next_owner='b',grant_budget_ns=5,handoff_ns=2)]),
        allow_unvalidated_training_candidate=True)


class StatefulAvailabilityTests(unittest.TestCase):
    def test_host_events_and_same_clock_calls_do_not_reset_grant_or_handoff(self):
        s=scheduler();d=s.decide(view(0));self.assertEqual(d.next_transition_ns,7)
        for t in (0,1,2,6):
            d=s.decide(view(t));self.assertEqual(d.next_transition_ns,7);self.assertEqual(d.gpu_rates[('a','gpu')],1.)
        self.assertEqual(len(s.transitions),1)
        for t in (7,7,8):
            d=s.decide(view(t));self.assertEqual(d.next_transition_ns,9);self.assertEqual(set(d.gpu_rates.values()),{0.})
        for t in (9,9,10,13):
            d=s.decide(view(t));self.assertEqual(d.next_transition_ns,14);self.assertEqual(d.gpu_rates[('b','gpu')],1.)
        self.assertEqual(len(s.transitions),2)

    def test_early_yield_releases_grant_but_charges_one_handoff(self):
        s=scheduler();s.decide(view(0));d=s.decide(view(3,('b',)))
        self.assertEqual(d.next_transition_ns,5)
        self.assertEqual(s.decide(view(3,('b',))),d)
        self.assertEqual(s.decide(view(5,('b',))).next_transition_ns,10)

    def test_empty_frontier_keeps_absolute_grant_budget(self):
        s=scheduler();s.decide(view(0));self.assertEqual(s.decide(view(2,())).gpu_rates,{})
        self.assertEqual(s.decide(view(5)).next_transition_ns,7)

    def test_phase_change_does_not_reset_grant_but_unknown_next_state_rejects(self):
        s=scheduler();s.decide(view(0));self.assertEqual(s.decide(view(3,phase='decode')).next_transition_ns,7)
        with self.assertRaisesRegex(UnknownEvidence,'UNSUPPORTED_CONDITIONAL'):
            s.decide(view(7,phase='decode'))

    def test_backwards_clock_and_unbound_phase_reject(self):
        s=scheduler();s.decide(view(2))
        with self.assertRaisesRegex(UnknownEvidence,'MONOTONIC'):s.decide(view(1))
        with self.assertRaisesRegex(UnknownEvidence,'STATIC_BOUND'):scheduler().decide(view(0,phase='UNBOUND_PHASE'))

    def test_interval_partition_preserves_missing_and_overlapping_envelopes(self):
        own=IntervalUnion([(0,4),(3,6),(10,12)]);other=IntervalUnion([(5,9),(11,14)])
        self.assertEqual(partition(0,15,own,other,own.intersection(other)),
            dict(own_only_ns=6,other_only_ns=5,overlapping_envelopes_ns=2,unresolved_ns=2))
        for a in range(15):
            for b in range(a,16):
                expected=sum(any(lo<=t<hi for lo,hi in own.intervals) for t in range(a,b))
                self.assertEqual(own.overlap(a,b),expected)


if __name__=='__main__':unittest.main()
