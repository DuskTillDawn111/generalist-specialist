import copy
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027.context_initial_prior_v15 import (
    SCHEMA,POLICY,JointInitialContextPrior,annotate_gap,realize_joint_initial_scenario)
from tools.predictor_validation_20261027.profiling import digest

SETUP=dict(residents={'a':'text','b':'text'},execution={},threads={},device='CPU_FIXTURE',
           measurement_domain='fixture',profiler='none',capture_mode='fixture')
F=[['a','text','pool',['decode']]]


def original(capture='one',sid='gap'):
    return dict(scenario_id=sid,capture=capture,source_job_id=sid,first_pending_frontier=F,
        public_setup=SETUP,held_owner=None,context_age_ns=None,holder_identified=False,
        service_ready_delay_bounds_ns={'lower_ns':None,'upper_ns':None},
        service_ready_point_identified=False)


def gap(**kwargs):
    config=dict(intervals=[(0,40,'a'),(60,100,'b')],issues=[],all_contexts_mapped=True,kernel_starts={})
    config.update(kwargs)
    return annotate_gap(original(),{'clock_ns':50},**config)


def fixture(rows):
    model=dict(schema=SCHEMA,training_only=True,policy=POLICY,scenarios=rows,CPU_fixture=True)
    model['model_sha256']=digest(model);return model


class JointInitialPriorTests(unittest.TestCase):
    def test_gap_target_ages_and_missing_readiness_are_joint_not_imputed(self):
        row=gap()
        self.assertEqual(row['state_kind'],'IN_TRANSITION')
        self.assertEqual((row['previous_held_owner'],row['elapsed_gap_ns'],row['next_owner'],row['residual_restore_gap_ns']),('a',10,'b',10))
        self.assertIsNone(row['held_owner']);self.assertIsNone(row['context_age_ns'])
        self.assertEqual(row['service_ready_delay_bounds_ns'],{'lower_ns':10,'upper_ns':None})
        self.assertEqual(row['service_ready_delay_after_restore_bounds_ns'],{'lower_ns':0,'upper_ns':None})

    def test_actual_same_envelope_witness_gives_upper_not_later_restore_kernel(self):
        row=gap(kernel_starts={'b':[110]})
        self.assertIsNone(row['service_ready_delay_bounds_ns']['upper_ns'])
        row=gap(kernel_starts={'b':[70,110]})
        self.assertEqual(row['service_ready_delay_bounds_ns'],{'lower_ns':10,'upper_ns':20})

    def test_unpaired_unmapped_missing_sides_overlap_keep_unknown(self):
        for kwargs in (dict(issues=[{'timestamp':45}]),dict(all_contexts_mapped=False),
                       dict(intervals=[(60,100,'b')]),dict(intervals=[(0,40,'a')]),
                       dict(intervals=[(0,55,'a'),(50,100,'b')])):
            self.assertEqual(gap(**kwargs)['state_kind'],'UNKNOWN')

    def test_capture_weights_and_original_order_preserve_gap_mass(self):
        held=dict(original('one','held'),state_kind='HELD_CONTEXT',state_identified=True,
                  held_owner='a',holder_identified=True,context_age_ns=5,
                  service_ready_delay_bounds_ns={'lower_ns':0,'upper_ns':0},service_ready_point_identified=True)
        transition=gap();transition['capture']='two'
        unknown=dict(original('two','unknown'),state_kind='UNKNOWN',state_identified=False)
        prior=JointInitialContextPrior(fixture([held,transition,unknown]),allow_cpu_fixture=True)
        result=prior.predict(first_pending_frontier=F,public_execution_setup=SETUP)
        self.assertEqual([r['scenario_id'] for r in result['scenarios']],['held','gap','unknown'])
        self.assertEqual([r['probability'] for r in result['scenarios']],[.5,.25,.25])
        self.assertEqual(result['in_transition_mass'],.25)
        self.assertEqual(result['unknown_state_mass'],.25)
        self.assertEqual(result['probability_sum'],1.)

    def test_prediction_and_realization_never_read_source_events(self):
        prior=JointInitialContextPrior(fixture([gap()]),allow_cpu_fixture=True)
        with patch('sqlite3.connect',side_effect=AssertionError('future source event read')),patch('pathlib.Path.read_bytes',side_effect=AssertionError('prediction file I/O')):
            result=prior.predict(first_pending_frontier=F,public_execution_setup=SETUP)
            point=realize_joint_initial_scenario(result['scenarios'][0],query_clock_ns=1000)
        self.assertEqual(point['previous_SAVE_ns'],990)
        self.assertEqual(point['next_RESTORE_ns'],1010)
        self.assertIsNone(point['service_ready_ns'])
        self.assertFalse(point['physical_nonexecution_claimed'])

    def test_missing_readiness_cannot_become_upper_or_point(self):
        row=gap()
        for rule in ('upper_bound_hypothesis','witnessed_point_only'):
            with self.assertRaises(UnknownEvidence):realize_joint_initial_scenario(row,query_clock_ns=1000,readiness_rule=rule)
        lower=realize_joint_initial_scenario(row,query_clock_ns=1000,readiness_rule='lower_bound_hypothesis')
        self.assertEqual(lower['service_ready_ns'],1010)
        self.assertTrue(lower['numerical_readiness_is_hypothesis'])
        self.assertIsNone(lower['service_ready_delay_bounds_ns']['upper_ns'])

    def test_unknown_mass_not_silently_realized_and_fixture_must_be_explicit(self):
        unknown=dict(original(),state_kind='UNKNOWN',state_identified=False)
        with self.assertRaises(UnknownEvidence):realize_joint_initial_scenario(unknown,query_clock_ns=1000)
        with self.assertRaises(ValueError):JointInitialContextPrior(fixture([gap()]))
        with self.assertRaises(TypeError):JointInitialContextPrior(fixture([gap()]),allow_cpu_fixture=True).predict(first_pending_frontier=F,public_execution_setup=SETUP,query_clock_ns=1000)


if __name__=='__main__':unittest.main()
