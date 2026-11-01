import copy
import unittest

import numpy as np

from tools.predictor_validation_20261027.handoff_candidate_v14 import (
    HandoffCandidate, PARAMETERS, READINESS_RULES, SCHEMA, baseline_predict,
    feature_row, names, ridge_fit, ridge_predict, verify_labels, weights,
)
from tools.predictor_validation_20261027.profiling import digest


class HandoffCandidateTests(unittest.TestCase):
    def query(self, age=1_000_000):
        return dict(held_owner='text#0', next_owner='generalist_2#0',
            context_age_ns=age,
            pending_frontier=[['text#0', 'text', 'short', ['prefill']]])

    def fixture(self):
        width = len(names({'short': 'text'}))
        def fit(intercept):
            return dict(center=[0.]*width, scale=[1.]*width,
                coefficients=[intercept]+[0.]*(width-1))
        model = dict(schema=SCHEMA, training_only=True, CPU_fixture=True,
            parameters=copy.deepcopy(PARAMETERS), public_pool_tasks={'short': 'text'},
            feature_names=names({'short': 'text'}), feature_minimum=[0.]*width,
            feature_maximum=[1.]*width, gap_model=fit(np.log1p(42.)),
            witness_delay_model=fit(np.log1p(2.2)), witness_presence_model=fit(0.),
            owner_pair_counts={'text#0->generalist_2#0':
                dict(rows=10, prequeued_witness=7, missing=3)})
        model['model_sha256'] = digest(model)
        return model

    def candidate(self):
        return HandoffCandidate(self.fixture(), allow_cpu_fixture=True)

    def test_readiness_requires_explicit_hypothesis_and_nonpending_mark_allowed(self):
        candidate = self.candidate()
        lower = candidate.query(**self.query(), readiness_rule=READINESS_RULES[0])
        upper = candidate.query(**self.query(), readiness_rule=READINESS_RULES[1])
        self.assertEqual(lower['save_to_restore_gap_ns'], 42_000)
        self.assertEqual(lower['service_ready_delay_after_restore_ns'], 0)
        self.assertEqual(upper['service_ready_delay_after_restore_ns'], 2200)
        self.assertTrue(upper['readiness_is_unvalidated_hypothesis'])
        self.assertFalse(upper['exact_service_ready_predicted'])
        self.assertFalse(upper['predicted_delay_is_counterfactual_bound'])
        with self.assertRaises(ValueError):
            candidate.query(**self.query(), readiness_rule='RESTORE_IS_READY')

    def test_missing_witness_mass_retained_and_unseen_pair_extrapolates(self):
        query = dict(self.query(10**14), next_owner='ocr#0')
        result = self.candidate().query(**query, readiness_rule=READINESS_RULES[1])
        self.assertEqual(result['missing_prequeued_witness_probability'], .5)
        self.assertEqual(result['owner_pair_training_support']['rows'], 0)
        self.assertIn('OWNER_PAIR_UNOBSERVED_STATISTICAL_TRANSFER', result['ood_flags'])
        self.assertIn('NO_OBSERVED_PREQUEUED_WITNESS_FOR_OWNER_PAIR', result['ood_flags'])
        self.assertGreater(result['save_to_restore_gap_ns'], 0)

    def test_unseen_pair_baseline_falls_back_to_scalar_global_centre(self):
        train = [dict(capture='a', held_owner='text#0', next_owner='text#1'),
            dict(capture='b', held_owner='text#0', next_owner='text#1')]
        test = [dict(held_owner='text#1', next_owner='text#0')]
        predicted, missing = baseline_predict(train, np.asarray([1000., 3000.]), test, pair=True)
        self.assertEqual(predicted.shape, (1,))
        self.assertEqual(missing, 1)
        self.assertAlmostEqual(predicted[0], (np.sqrt(8)-1)*1000)

    def test_capture_balancing_and_continuous_regression_are_finite(self):
        balanced = weights(['a', 'a', 'a', 'b'])
        self.assertAlmostEqual(balanced[:3].sum(), balanced[3:].sum())
        matrix = np.asarray([feature_row(**self.query(i*1_000_000),
            pool_tasks={'short': 'text'}) for i in range(12)])
        fitted = ridge_fit(matrix, np.arange(12)*100.+1000., np.asarray(['a']*9+['b']*3))
        self.assertTrue(np.isfinite(ridge_predict(fitted, matrix)).all())
        with self.assertRaises(ValueError):
            ridge_fit(matrix, np.full(12, np.nan), np.asarray(['a']*12))

    def test_no_future_fields_and_integer_age_required(self):
        with self.assertRaises(TypeError):
            feature_row(**self.query(), pool_tasks={'short': 'text'}, future_kernel_residual_ns=1)
        for invalid_age in (True, 1.2, -1):
            with self.assertRaises(ValueError):
                feature_row(**self.query(invalid_age), pool_tasks={'short': 'text'})

    def test_missing_witness_never_becomes_zero_service_label(self):
        row = dict(capture='a', save_end_ns=100, next_restore_start_ns=200,
            save_to_restore_gap_ns=100, exact_service_ready_observed=False,
            service_after_restore_interval_ns=dict(lower_ns=0, upper_ns=None),
            prequeued_kernel_witness=None, missing_reason='NO_PREQUEUED_WITNESS')
        document = dict(schema='gsdeploy.prequeued_context_handoff_witness.v14',
            training_only=True, restore_is_exact_service_ready=False,
            observations_are_exogenous_counterfactual_bounds=False, labels=[row])
        verify_labels(document)
        row['service_after_restore_interval_ns']['upper_ns'] = 0
        with self.assertRaisesRegex(ValueError, 'Missing witness'):
            verify_labels(document)

    def test_fixture_and_invalid_model_cannot_masquerade_as_real_support(self):
        with self.assertRaisesRegex(ValueError, 'CPU fixture'):
            HandoffCandidate(self.fixture())
        model = self.fixture()
        model['gap_model']['scale'][0] = 0
        model['model_sha256'] = digest({k: v for k, v in model.items() if k != 'model_sha256'})
        with self.assertRaisesRegex(ValueError, 'Positive'):
            HandoffCandidate(model, allow_cpu_fixture=True)


if __name__ == '__main__':
    unittest.main()
