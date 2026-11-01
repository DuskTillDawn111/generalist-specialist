import copy
import unittest

import numpy as np

from tools.predictor_validation_20261027.context_next_owner_v13 import (
    SCHEMA, PARAMETERS, RESIDENTS, NextOwnerCandidate, evaluate, feature_names,
    features, fit_matrix, predict_matrix,
)
from tools.predictor_validation_20261027.profiling import digest


class NextOwnerCandidateTests(unittest.TestCase):
    def query(self, age=1_000_000):
        return dict(held_owner='text#0', context_age_ns=age,
            pending_frontier=[['text#0', 'text', 'short', ['prefill']]],
            public_workload=[['text#0', 'text', 'short', 0]],
            pending_age_ns_by_owner=None, pending_reference_work_ns_by_owner=None)

    def fixture(self):
        names = feature_names({'short':'text'})
        width = len(names)
        coefficients = np.zeros((width, len(RESIDENTS)))
        coefficients[0, RESIDENTS.index('generalist_2#0')] = 5
        model = dict(schema=SCHEMA, training_only=True, CPU_fixture=True,
            parameters=copy.deepcopy(PARAMETERS), residents=list(RESIDENTS),
            public_pool_tasks={'short':'text'}, feature_names=names,
            fitted=dict(center=[0.]*width, scale=[1.]*width,
                coefficients=coefficients.tolist(), feature_minimum=[0.]*width,
                feature_maximum=[1.]*width, held_log_age_ranges={'text#0':[0.,10.]},
                label_counts={owner:int(owner=='generalist_2#0') for owner in RESIDENTS}))
        model['model_sha256'] = digest(model)
        return model

    def test_nonpending_target_is_possible_and_all_resident_classes_returned(self):
        model = NextOwnerCandidate(self.fixture(), allow_cpu_fixture=True)
        prediction = model.predict_given_save_now(**self.query())
        self.assertEqual(prediction['most_likely_next_owner'], 'generalist_2#0')
        self.assertEqual(set(prediction['next_owner_probabilities']), set(RESIDENTS))
        self.assertAlmostEqual(sum(prediction['next_owner_probabilities'].values()), 1.)
        self.assertFalse(prediction['service_ready_time_predicted'])
        self.assertFalse(prediction['complete_availability_oracle'])

    def test_no_future_target_timestamps_or_residual_labels_in_features(self):
        for key in ('next_owner', 'save_end_ns', 'next_restore_start_ns', 'first_kernel_start_ns'):
            bad = dict(self.query(), **{key:'future'})
            with self.assertRaisesRegex(ValueError, 'future label'):
                features(bad, {'short':'text'})
        for key in ('pending_reference_work_ns_by_owner', 'pending_age_ns_by_owner'):
            bad = dict(self.query(), **{key:{'text#0':100}})
            with self.assertRaisesRegex(ValueError, 'no independently reconstructed'):
                features(bad, {'short':'text'})

    def test_ood_continuous_age_reported_without_exact_cell_lookup(self):
        model = NextOwnerCandidate(self.fixture(), allow_cpu_fixture=True)
        prediction = model.predict_given_save_now(**self.query(10**14))
        self.assertIn('HELD_OWNER_AGE_OUTSIDE_TRAINING_SUPPORT', prediction['ood_flags'])
        self.assertIn('EXTRAPOLATION', prediction['status'])
        self.assertEqual(len(prediction['next_owner_probabilities']), 4)

    def test_unknown_public_identity_is_rejected(self):
        query = self.query()
        query['pending_frontier'][0][2] = 'foreign-pool'
        with self.assertRaisesRegex(ValueError, 'public resident/task/input'):
            features(query, {'short':'text'})

    def test_cpu_fixture_cannot_masquerade_as_measured_model(self):
        with self.assertRaisesRegex(ValueError, 'CPU fixture'):
            NextOwnerCandidate(self.fixture())

    def test_grouped_logit_learns_continuous_age_without_future_features(self):
        matrix = np.asarray([features(self.query(i*1_000_000), {'short':'text'}) for i in range(40)])
        labels = np.asarray([0 if i < 20 else 3 for i in range(40)])
        fit = fit_matrix(matrix, labels, np.asarray(['capture1']*20+['capture2']*20))
        predicted = predict_matrix(fit, matrix)
        self.assertTrue(np.isfinite(predicted).all())
        self.assertGreater(np.mean(predicted.argmax(axis=1) == labels), .9)
        self.assertTrue(fit['optimizer']['converged'])

    def test_unknown_coverage_is_counted_against_full_transition_domain(self):
        result = evaluate(np.asarray([0, 1, 2]), np.asarray([0, -1, 3]),
            [dict(next_owner_is_pending=False)]*3)
        self.assertEqual(result['requested_transitions'], 3)
        self.assertEqual(result['predicted_transitions'], 2)
        self.assertEqual(result['unknown_transitions'], 1)
        self.assertEqual(result['accuracy'], 1/3)
        self.assertEqual(result['confusion'][RESIDENTS[1]]['UNKNOWN'], 1)


if __name__ == '__main__':
    unittest.main()
