import copy
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027.context_initial_prior_v14 import (
    SCHEMA, POLICY, InitialContextPrior, scenario_from_observation, realize_initial_scenario,
)
from tools.predictor_validation_20261027.profiling import digest


class InitialContextPriorTests(unittest.TestCase):
    def setup(self):
        return dict(residents={'A':'text', 'B':'generalist_2'},
            execution=dict(batch_size=1, admission='default_shared_no_permit'),
            threads={'OMP_NUM_THREADS':'1'}, device='fixture-device',
            measurement_domain='light', profiler='fixture-observer', capture_mode='common_session')

    def row(self, capture='one', age=10, holder='A', unknown=False, readiness_upper=100):
        return dict(capture=capture, job_id=capture+str(age), job_name='public-job', clock_ns=100,
            pending_frontier=[['B', 'text', 'short', ['prefill']]],
            status='UNKNOWN_INITIAL_CONTEXT_GAP' if unknown else 'OBSERVED_PAIRED_CONTEXT_STATE',
            held_owner=None if unknown else holder, context_age_ns=None if unknown else age,
            restore_start_ns=None if unknown else 100-age,
            service_ready_bounds=None if unknown else dict(lower_ns=100-age, upper_ns=readiness_upper, exact=False))

    def fixture(self):
        observations = [self.row(), self.row('two', 20, holder='B'),
            self.row('two', 30, unknown=True)]
        model = dict(schema=SCHEMA, policy=copy.deepcopy(POLICY), training_only=True, CPU_fixture=True,
            scenarios=[scenario_from_observation(row, self.setup()) for row in observations])
        model['model_sha256'] = digest(model)
        return model

    def predict(self, prior, **options):
        return prior.predict(first_pending_frontier=self.row()['pending_frontier'],
            public_execution_setup=self.setup(), **options)

    def test_joint_scenarios_and_missing_mass_are_capture_balanced(self):
        result = self.predict(InitialContextPrior(self.fixture(), allow_cpu_fixture=True))
        self.assertEqual([r['probability'] for r in result['scenarios']], [.5, .25, .25])
        self.assertEqual(result['unknown_holder_mass'], .25)
        self.assertEqual(result['probability_sum'], 1.)
        self.assertFalse(result['missing_mass_renormalized_away'])
        self.assertEqual([(r['held_owner'],r['context_age_ns']) for r in result['scenarios']],
            [('A',10), ('B',20), (None,None)])
        self.assertFalse(result['first_pending_owner_used_as_default_holder'])

    def test_future_job_composition_does_not_change_initial_scenario(self):
        before = self.row()
        after = dict(before, public_workload=[['future-owner', 'image', 'future-input', 99999999]],
            whole_future_job_count=999)
        self.assertEqual(scenario_from_observation(before, self.setup()),
            scenario_from_observation(after, self.setup()))

    def test_unknown_holder_never_becomes_first_pending_owner(self):
        scenario = scenario_from_observation(self.row(unknown=True), self.setup())
        for rule in ('witnessed_point_only', 'lower_bound_hypothesis', 'upper_bound_hypothesis'):
            with self.assertRaisesRegex(UnknownEvidence, 'HOLDER_UNIDENTIFIED'):
                realize_initial_scenario(scenario, query_clock_ns=100, readiness_rule=rule)

    def test_unbounded_readiness_cannot_be_given_a_finite_upper_time(self):
        scenario = scenario_from_observation(self.row(readiness_upper=None), self.setup())
        with self.assertRaisesRegex(UnknownEvidence, 'UPPER_BOUND_UNAVAILABLE'):
            realize_initial_scenario(scenario, query_clock_ns=100, readiness_rule='upper_bound_hypothesis')
        with self.assertRaisesRegex(UnknownEvidence, 'NOT_POINT_IDENTIFIED'):
            realize_initial_scenario(scenario, query_clock_ns=100, readiness_rule='witnessed_point_only')
        candidate = realize_initial_scenario(scenario, query_clock_ns=100, readiness_rule='lower_bound_hypothesis')
        self.assertEqual(candidate['service_ready_delay_ns'], 0)
        self.assertTrue(candidate['readiness_hypothesis_used'])

    def test_actual_past_kernel_witness_needs_no_readiness_hypothesis(self):
        scenario = scenario_from_observation(self.row(readiness_upper=95), self.setup())
        decision = realize_initial_scenario(scenario, query_clock_ns=1000, readiness_rule='witnessed_point_only')
        self.assertEqual(decision['restore_start_ns'], 990)
        self.assertEqual(decision['service_ready_delay_ns'], 0)
        self.assertFalse(decision['readiness_hypothesis_used'])

    def test_frontier_pooling_and_observer_transfer_require_explicit_hypotheses(self):
        prior = InitialContextPrior(self.fixture(), allow_cpu_fixture=True)
        frontier = [['B', 'text', 'new-public-input', ['decode']]]
        with self.assertRaisesRegex(UnknownEvidence, 'POOLING_HYPOTHESIS_REQUIRED'):
            prior.predict(first_pending_frontier=frontier, public_execution_setup=self.setup())
        result = prior.predict(first_pending_frontier=frontier, public_execution_setup=self.setup(),
            allow_unvalidated_frontier_pooling=True)
        self.assertEqual(result['conditioning_tier'], 'POOLED_INPUT_AND_PHASE_HYPOTHESIS')
        natural = dict(self.setup(), profiler='none', measurement_domain='natural')
        with self.assertRaisesRegex(UnknownEvidence, 'OBSERVER_TRANSFER'):
            prior.predict(first_pending_frontier=self.row()['pending_frontier'], public_execution_setup=natural)
        result = prior.predict(first_pending_frontier=self.row()['pending_frontier'], public_execution_setup=natural,
            allow_unvalidated_observer_transfer=True)
        self.assertTrue(result['observer_transfer_hypothesis'])

    def test_public_execution_change_is_not_silently_bridged(self):
        prior = InitialContextPrior(self.fixture(), allow_cpu_fixture=True)
        changed = dict(self.setup(), execution=dict(batch_size=2))
        with self.assertRaisesRegex(UnknownEvidence, 'EXECUTION_SETUP_UNSUPPORTED'):
            prior.predict(first_pending_frontier=self.row()['pending_frontier'], public_execution_setup=changed,
                allow_unvalidated_observer_transfer=True, allow_unvalidated_frontier_pooling=True)

    def test_whole_capture_exclusion_and_single_capture_diagnostic_are_explicit(self):
        model = self.fixture()
        with self.assertRaisesRegex(ValueError, 'whole-training-capture'):
            InitialContextPrior(model, allow_cpu_fixture=True, exclude_training_capture='two')
        prior = InitialContextPrior(model, allow_cpu_fixture=True, exclude_training_capture='two',
            allow_single_capture_training_diagnostic=True)
        result = self.predict(prior)
        self.assertEqual(result['supporting_capture_ids'], ['one'])
        self.assertEqual(result['unknown_holder_mass'], 0.)
        self.assertEqual(result['scenarios'][0]['held_owner'], 'A')


if __name__ == '__main__':
    unittest.main()
