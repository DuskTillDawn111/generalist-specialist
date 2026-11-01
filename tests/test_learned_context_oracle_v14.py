import copy
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tests.test_held_context_scheduler_v12_1 import view
from tools.predictor_validation_20261027.learned_context_oracle_v14 import LearnedContextOracle, LearnedContextScheduler


class Prior:
    def __init__(self,missing=0):self.missing=missing
    def predict(self,**kwargs):
        scenario=dict(probability=1.,holder_identified=True,held_owner='text#1',context_age_ns=100,
            service_ready_delay_bounds_ns={'lower_ns':0,'upper_ns':0},
            service_ready_point_identified=True,scenario_id='source-joint-scenario')
        return dict(scenarios=[scenario],unknown_holder_mass=self.missing,probability_sum=1.,
                    unresolved_point_readiness_mass=0,observer_transfer_hypothesis=True,
                    conditioning_tier='EXACT_FIRST_PENDING_FRONTIER')


class Controller:
    def __init__(self):self.starts=[]
    def start_restore(self,owner,restore,frontier,key,*,observation_ns):
        self.starts.append((owner,restore,observation_ns))
        state=dict(crossing=observation_ns+5,last=observation_ns,frontier=frontier)
        return self.result(state,observation_ns)
    def update(self,state,now,frontier):
        state=copy.deepcopy(state);state['last']=now;state['frontier']=frontier
        return self.result(state,now)
    def result(self,state,now):
        due=now>=state['crossing']
        return dict(state=state,next_review_ns=state['crossing'],
            status='SAVE_BUDGET_REACHED' if due else 'CUMULATIVE_SAVE_APPROXIMATION',
            absolute_save_candidate_ns=state['crossing'],uncertainty_reasons=[],hazard_budget=.693)
    def provenance(self):return dict(kind='EXPLICIT_CPU_FAKE_CONTROLLER')


class Mark:
    def __init__(self):self.calls=[]
    def predict_given_save_now(self,**kwargs):
        self.calls.append(kwargs)
        return dict(most_likely_next_owner='text#0' if kwargs['held_owner']=='text#1' else 'text#1',ood_flags=[])


class Handoff:
    def query(self,**kwargs):
        return dict(save_to_restore_gap_ns=2,service_ready_delay_after_restore_ns=0,ood_flags=[])


def oracle(missing=0):
    return LearnedContextOracle(initial_prior=Prior(missing),save_controller=Controller(),next_owner=Mark(),
        handoff=Handoff(),public_setup={},public_workload=[['text#0','text','x',0]],
        source_identity='a'*64,initial_quantile=.5,initial_readiness_rule='lower_bound_hypothesis',
        handoff_readiness_rule='lower_bound_reference_availability_hypothesis',
        allow_initial_pooling=True,allow_observer_transfer=True,allow_flagged_statistical_extrapolation=True)


class ComposedOracleTests(unittest.TestCase):
    def test_conditional_initial_age_and_mark_only_at_actual_save(self):
        model=oracle();s=LearnedContextScheduler(('text#0','text#1'),oracle=model)
        self.assertEqual(s.decide(view(0)).next_transition_ns,5)
        self.assertEqual(model.controller.starts,[('text#1',-100,0)])
        s.decide(view(3))
        self.assertEqual(model.next_owner.calls,[])
        s.decide(view(5))
        self.assertEqual(model.next_owner.calls[0]['context_age_ns'],105)
        self.assertIsNone(model.next_owner.calls[0]['previous_held_owner'])
        self.assertEqual(s.decide(view(7)).gpu_rates[('text#0','gpu')],1.)
        self.assertEqual(model.previous_held_owner,'text#1')
        self.assertEqual(model.controller.starts[-1],('text#0',7,7))

    def test_empty_frontier_is_observed_and_can_mark_an_already_due_save(self):
        model=oracle();s=LearnedContextScheduler(('text#0','text#1'),oracle=model)
        s.decide(view(0));s.decide(view(3,()))
        self.assertEqual(model.last_frontier,[])
        result=s.decide(view(5,()))
        self.assertEqual(result.next_transition_ns,7)
        self.assertEqual(model.next_owner.calls[0]['pending_frontier'],[])
        self.assertIsNone(s.held_owner)

    def test_changed_frontier_at_service_preserves_post_switch_review(self):
        model=oracle();s=LearnedContextScheduler(('text#0','text#1'),oracle=model)
        s.decide(view(0));s.decide(view(5))
        # The switch's service boundary is 7 and its existing review is 8.
        # A newly pending owner at SERVICE must not replace that earlier
        # review with the new context's SAVE candidate at 12.
        result=s.decide(view(7,('text#0','text#1')))
        self.assertEqual(result.next_transition_ns,8)
        self.assertEqual(model.hazard['next_review_ns'],12)
        self.assertEqual(s.restore_start_ns,7)
        self.assertEqual(s.decide(view(8,('text#0','text#1'))).next_transition_ns,12)
        self.assertEqual(model.controller.starts[-1],('text#0',7,7))
        self.assertEqual(len(model.controller.starts),2)

    def test_unknown_prior_mass_cannot_be_conditioned_away(self):
        model=oracle(.25);s=LearnedContextScheduler(('text#0','text#1'),oracle=model)
        with self.assertRaisesRegex(UnknownEvidence,'UNRESOLVED_INITIAL_HOLDER_MASS'):
            s.decide(view(0))
        self.assertEqual(model.initial_prediction['unknown_holder_mass'],.25)
        self.assertIsNone(s.held_owner)

    def test_missed_save_must_not_rewind_or_use_future_mark_state(self):
        model=oracle();s=LearnedContextScheduler(('text#0','text#1'),oracle=model)
        s.decide(view(0))
        with self.assertRaisesRegex(UnknownEvidence,'SKIPPED_PREDICTED_SAVE'):
            s.decide(view(6))
        self.assertEqual(model.next_owner.calls,[])


if __name__ == '__main__':unittest.main()
