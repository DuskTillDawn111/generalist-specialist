import copy
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from tests.test_held_context_scheduler_v12_1 import view
from tests.test_learned_context_oracle_v14 import Controller,Mark,Handoff
from tools.predictor_validation_20261027.learned_context_oracle_v15 import LearnedContextOracle,LearnedContextScheduler


class GapPrior:
    def __init__(self,upper=None,unknown=0):self.upper=upper;self.unknown=unknown
    def predict(self,**kwargs):
        scenario=dict(state_kind='IN_TRANSITION',state_identified=True,probability=1.,
            held_owner=None,context_age_ns=None,holder_identified=False,scenario_id='frozen-joint-gap',
            previous_held_owner='text#0',elapsed_gap_ns=10,next_owner='text#1',residual_restore_gap_ns=2,
            service_ready_delay_bounds_ns={'lower_ns':2,'upper_ns':None if self.upper is None else 2+self.upper},
            service_ready_delay_after_restore_bounds_ns={'lower_ns':0,'upper_ns':self.upper},service_ready_point_identified=False)
        return dict(scenarios=[scenario],unknown_state_mass=self.unknown,probability_sum=1.,
            unresolved_point_readiness_mass=1.,observer_transfer_hypothesis=False,
            conditioning_tier='EXACT_FIRST_PENDING_FRONTIER')


def oracle(rule='lower_bound_hypothesis',upper=None,unknown=0):
    return LearnedContextOracle(initial_prior=GapPrior(upper,unknown),save_controller=Controller(),
        next_owner=Mark(),handoff=Handoff(),public_setup={},public_workload=[['text#0','text','x',0]],
        source_identity='a'*64,initial_quantile=.5,initial_readiness_rule=rule,
        handoff_readiness_rule='lower_bound_reference_availability_hypothesis',
        allow_initial_pooling=True,allow_observer_transfer=True,allow_flagged_statistical_extrapolation=True)


class InitialTransitionOracleTests(unittest.TestCase):
    def test_no_fabricated_holder_restore_or_age_before_residual_restore(self):
        model=oracle();scheduler=LearnedContextScheduler(('text#0','text#1'),oracle=model)
        first=scheduler.decide(view(100))
        self.assertEqual(first.next_transition_ns,102)
        self.assertEqual(first.gpu_rates[('text#0','gpu')],0.)
        self.assertIsNone(scheduler.held_owner);self.assertIsNone(scheduler.restore_start_ns)
        self.assertIsNone(scheduler.snapshot(view(101)).context_age_ns)
        self.assertEqual(scheduler.transitions,[])
        self.assertEqual(model.controller.starts,[])
        scheduler.decide(view(102))
        self.assertEqual(scheduler.held_owner,'text#1')
        self.assertEqual(scheduler.restore_start_ns,102)
        self.assertEqual(model.controller.starts,[('text#1',102,102)])
        self.assertEqual(model.previous_held_owner,'text#0')
        self.assertEqual([e['kind'] for e in scheduler.transitions],['RESTORE_START','SERVICE_READY'])

    def test_next_owner_need_not_be_pending_and_future_empty_events_continue(self):
        model=oracle();scheduler=LearnedContextScheduler(('text#0','text#1'),oracle=model)
        scheduler.decide(view(100))
        empty=scheduler.decide(view(101,()))
        self.assertEqual(empty.next_transition_ns,102)
        restored=scheduler.decide(view(102,()))
        self.assertEqual(scheduler.held_owner,'text#1')
        self.assertTrue(scheduler.service_available)
        self.assertEqual(restored.gpu_rates,{})
        scheduler.decide(view(103))
        self.assertEqual(model.controller.starts,[('text#1',102,102)])

    def test_separate_restore_and_service_boundaries_keep_absolute_age(self):
        model=oracle(rule='upper_bound_hypothesis',upper=5)
        scheduler=LearnedContextScheduler(('text#0','text#1'),oracle=model)
        self.assertEqual(scheduler.decide(view(100)).next_transition_ns,102)
        self.assertEqual(scheduler.decide(view(102)).next_transition_ns,107)
        self.assertFalse(scheduler.service_available)
        self.assertEqual(scheduler.snapshot(view(105)).context_age_ns,3)
        scheduler.decide(view(107))
        self.assertTrue(scheduler.service_available)
        self.assertEqual(model.controller.starts,[('text#1',102,107)])
        self.assertEqual(scheduler.snapshot(view(108)).context_age_ns,6)

    def test_unwitnessed_default_bounds_and_upper_point_are_rejected(self):
        for rule in ('bounds_only','witnessed_point_only','upper_bound_hypothesis'):
            model=oracle(rule);scheduler=LearnedContextScheduler(('text#0','text#1'),oracle=model)
            with self.assertRaises(UnknownEvidence):scheduler.decide(view(100))
            self.assertIsNone(scheduler.held_owner)
            self.assertEqual(scheduler.transitions,[])

    def test_unresolved_mass_not_dropped_and_source_replay_disabled(self):
        model=oracle(unknown=.1);scheduler=LearnedContextScheduler(('text#0','text#1'),oracle=model)
        with self.assertRaisesRegex(UnknownEvidence,'UNRESOLVED_INITIAL_STATE_MASS'):scheduler.decide(view(100))
        with self.assertRaises(ValueError):scheduler.observe_context_event(101,'RESTORE_START','text#0')

    def test_prediction_does_not_read_future_events_and_hypotheses_are_recorded(self):
        model=oracle();scheduler=LearnedContextScheduler(('text#0','text#1'),oracle=model)
        with patch('sqlite3.connect',side_effect=AssertionError('future event lookup')):
            scheduler.decide(view(100));scheduler.decide(view(102));scheduler.decide(view(107))
        self.assertEqual(model.next_owner.calls[0]['previous_held_owner'],'text#0')
        self.assertIn('INITIAL_SERVICE_READINESS_LOWER_BOUND_HYPOTHESIS',model.uncertainty)
        self.assertIn('INITIAL_SCHEDULER_IN_TRANSITION_STATISTICAL_SCENARIO',model.uncertainty)
        self.assertFalse(scheduler.provenance()['initial_gap_is_physical_no_execution'])


if __name__=='__main__':unittest.main()
