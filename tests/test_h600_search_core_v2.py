"""Real CPU forecasts connect numerical-v2 search to its prediction entrypoint."""
import copy
import json
import unittest
from unittest import mock

from tests import test_flow_validation as fixtures
from tools.predictor_validation_20261027 import h600_event_predict_v2 as prediction
from tools.predictor_validation_20261027 import h600_search_core_v2 as search
from tools.predictor_validation_20261027.profiling import digest
from tools.predictor_validation_20261027.statistical_profiles import phase_count


class H600SearchCoreV2Tests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.FlowValidationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.plan = copy.deepcopy(fixture.plan)
        self.model = json.loads(fixture.model_path.read_text())
        self.resources = json.loads(fixture.resources_path.read_text())
        self.quality = json.loads(fixture.quality_path.read_text())

    def rehash_model(self):
        self.model.pop('model_sha256', None)
        self.model['model_sha256'] = digest(self.model)

    def assert_prediction_equal(self, candidate):
        predicted = prediction.predict(self.plan, candidate, self.model, self.resources, self.quality)
        evaluated = search.evaluate(self.plan, candidate, self.model, self.resources, self.quality)
        self.assertIsNone(predicted['unknown_reason'])
        self.assertEqual(evaluated['status'], 'COMPLETE')
        self.assertEqual(evaluated['comparison_U'], predicted['comparison_U'])
        self.assertEqual(evaluated['absolute_U'], predicted['metrics']['full_U'])
        self.assertEqual(evaluated['terms'], predicted['metrics']['terms'])
        self.assertEqual(evaluated['cohort_size'], len(predicted['requests']))
        self.assertEqual(evaluated['incomplete_ids'], predicted['metrics']['incomplete_ids'])
        self.assertIsNone(evaluated['physical_eligibility'])
        return predicted, evaluated

    def test_search_and_prediction_score_the_same_complete_arrivals_and_fifo_wait(self):
        predicted, evaluated = self.assert_prediction_equal(self.plan['candidates'][0])
        self.assertEqual([r['service_ms'] for r in predicted['requests']], [100., 100.])
        self.assertEqual([r['wait_ms'] for r in predicted['requests']], [0., 50.])
        self.assertTrue(evaluated['model_eligible'])
        self.assertIsNone(evaluated['planning_eligible'])

    def test_plain_and_bounded_construct_fresh_rollouts_and_retain_actual_ties(self):
        with mock.patch.object(search, 'FullHorizonModel', wraps=search.FullHorizonModel) as engine:
            plain = search.search(self.plan, self.model, self.resources, self.quality)
            bounded = search.search(self.plan, self.model, self.resources, self.quality, bounded=True)
        self.assertEqual(engine.call_count, 4)
        self.assertEqual(plain['all_tie_ids'], ['alternative', 'base'])
        self.assertEqual(plain['all_tie_ids'], bounded['all_tie_ids'])
        self.assertEqual(plain['maximum_model_comparison_U'], bounded['maximum_model_comparison_U'])
        self.assertEqual(plain['unknown_ids'], [])
        self.assertEqual(bounded['bound_pruned'], [])
        for record in plain['records']:
            self.assertLessEqual(record['result']['comparison_U'], record['bound']['U_upper'])

    def test_real_two_stage_long_clock_finishes_with_original_event_budget(self):
        self.plan['trace'] = self.plan['trace'][:1]
        self.plan.update(window_seconds=600., drain_seconds=300.)
        for candidate in self.plan['candidates']:
            candidate['routes'] = {'r1': 's'}
        stages = [('prefill', 571791.2283316356), ('finalize', .4716724721000537)]
        self.model['profiles'][0]['segments'] = [
            {'id': name, 'kind': 'host', 'duration_ms': duration, 'demand': {}}
            for name, duration in stages]
        self.model['semantic_models'] = []
        for name, _ in stages:
            key, members = phase_count([({'family': 'text'}, {'phase': name})])
            self.model['phase_contexts'][key] = members
            self.model['semantic_models'].append(
                {'target': ['text', 'text', 'known', name], 'progress': {key: 1.}})
        self.rehash_model()
        predicted, evaluated = self.assert_prediction_equal(self.plan['candidates'][0])
        self.assertEqual(predicted['requests'][0]['finish_ns'], round(571791700004.1077))
        self.assertEqual(predicted['original_event_budget'], 100000)
        self.assertLess(evaluated['transition_count'], 5)
        self.assertTrue(evaluated['model_constraints']['drain_complete'])
        # A finite-service/window diagnostic cannot grant or reject original admission.
        self.assertFalse(evaluated['finite_forecast_load_proxy_below_point8'])
        self.assertTrue(evaluated['model_eligible'])
        self.assertIsNone(evaluated['planning_eligible'])

    def test_joint_batch_and_weight_reuse_require_real_executor_evidence(self):
        for execution in ({'batch_size': 2, 'weight_reuse': False},
                          {'batch_size': 1, 'weight_reuse': True}):
            with self.subTest(execution=execution):
                self.model['profiles'][0]['execution'].update(execution)
                self.rehash_model()
                predicted = prediction.predict(self.plan, self.plan['candidates'][0],
                    self.model, self.resources, self.quality)
                evaluated = search.evaluate(self.plan, self.plan['candidates'][0],
                    self.model, self.resources, self.quality)
                self.assertIn('JOINT_BATCH_OR_MERGED_WEIGHT_EXECUTOR_REQUIRED', predicted['unknown_reason'])
                self.assertEqual(evaluated['status'], 'UNKNOWN')
                self.assertIsNone(evaluated['model_eligible'])
                self.assertIsNone(evaluated['comparison_U'])
                self.assertIsNone(evaluated['physical_eligibility'])
                self.assertTrue(all(r['finish_ns'] is None for r in predicted['requests']))

    def test_unseen_input_remains_in_full_domain_as_unknown(self):
        self.plan['trace'][1]['pool_id'] = 'unprofiled'
        result = search.search(self.plan, self.model, self.resources, self.quality, bounded=True)
        self.assertEqual(result['candidate_count'], 2)
        self.assertEqual(result['fresh_forecast_calls'], 2)
        self.assertEqual(result['unknown_ids'], ['alternative', 'base'])
        self.assertIsNone(result['maximum_model_comparison_U'])
        self.assertEqual(result['model_ineligible_ids'], [])
        self.assertEqual(result['all_tie_ids'], [])


if __name__ == '__main__':
    unittest.main()
