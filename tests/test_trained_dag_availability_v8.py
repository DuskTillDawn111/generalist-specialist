import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027.dag_pending_frontier_v8 import completion_upper_bounds
from tools.predictor_validation_20261027.dag_phase_scheduler_v8 import ActiveGpuNode,SchedulingView
from tools.predictor_validation_20261027.trained_dag_availability_v8 import TrainedDagAvailabilityScheduler


def view(t,owners='ab'):
    return SchedulingView(t,tuple(ActiveGpuNode((p,'gpu'),100.,'prefill',p,'text','x') for p in owners))


def make():
    state=[[p,'text','x',['prefill']] for p in 'ab'];solo=[state[0]]
    return TrainedDagAvailabilityScheduler(dict(schema='gsdeploy.trained_DAG_availability_policy.v8',training_only=True,
        reference_work_domain='SOLO_ONLY_REFERENCE_NS',
        initial_uncontended_cells=[dict(frontier=solo,dispatch_wait_ns=2)],
        uncontended_cells=[dict(frontier=solo,dispatch_wait_ns=1)],
        cells=[dict(current_owner='a',reason='contention_entry',frontier=state,next_owner='a',grant_budget_ns=7,handoff_ns=0),
               dict(current_owner='a',reason='grant_expired',frontier=state,next_owner='b',grant_budget_ns=5,handoff_ns=2)]),
        allow_unvalidated_training_candidate=True)


class TrainedAvailabilityTests(unittest.TestCase):
    def test_initial_dispatch_and_contention_residual_are_different(self):
        s=make();d=s.decide(view(0,'a'));self.assertEqual(d.next_transition_ns,2);self.assertEqual(d.gpu_rates[('a','gpu')],0.)
        self.assertEqual(s.decide(view(2,'a')).gpu_rates[('a','gpu')],1.)
        d=s.decide(view(4));self.assertEqual(d.next_transition_ns,11)
        for t in (4,6,10):self.assertEqual(s.decide(view(t)).next_transition_ns,11)

    def test_peer_return_uses_existing_residual_deadline(self):
        s=make();s.decide(view(0,'a'));s.decide(view(4))
        s.decide(view(5,'a'));self.assertEqual(s.decide(view(6)).next_transition_ns,11)
        self.assertEqual(len(s.transitions),2)

    def test_handoff_not_restarted_and_unsupported_state_rejects(self):
        s=make();s.decide(view(0,'a'));s.decide(view(4))
        for t in (11,11,12):self.assertEqual(s.decide(view(t)).next_transition_ns,13)
        self.assertEqual(s.decide(view(13)).next_transition_ns,18)
        with self.assertRaisesRegex(UnknownEvidence,'UNKNOWN_TRAINED'):s.decide(view(18))

    def test_copy_predecessor_and_virtual_completion_bound_do_not_use_zero(self):
        nodes={'copy':dict(kind='copy',observed_start_ns=2,observed_end_ns=9),
               'event':dict(kind='event_record'),'wait':dict(kind='event_wait'),
               'gpu':dict(kind='gpu',observed_start_ns=15,observed_end_ns=20)}
        edges=[dict(before=a,after=b) for a,b in [('copy','event'),('event','wait'),('wait','gpu')]]
        bound,preds,order=completion_upper_bounds(nodes,edges)
        self.assertEqual(bound['copy'],9);self.assertEqual(bound['event'],15);self.assertEqual(bound['wait'],15)
        # A candidate depending only on the unobserved virtual operation cannot
        # be called ready before the first observed started consumer at 15.
        self.assertEqual(max(bound[p] for p in preds['gpu']),15)
        self.assertEqual(completion_upper_bounds({'event':dict(kind='event_record')},[])[0]['event'],None)


if __name__=='__main__':unittest.main()
