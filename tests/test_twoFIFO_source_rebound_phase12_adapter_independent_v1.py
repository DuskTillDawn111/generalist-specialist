"""Independent CPU contract checks for the rebound phase12 row adapter."""
import copy
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from tests.test_twoFIFO_training_intake_v4 import IntakeTests, documents
from tools.predictor_validation_20261027 import twoFIFO_source_rebound_phase12_adapter_v1 as adapter


def fixture():
    IntakeTests.setUpClass()
    plans = IntakeTests.plans
    old = documents(plans)
    new = copy.deepcopy(old)
    for doc in new:
        doc.update(schema=adapter.MACRO_SCHEMA, fitted_model_qualified=False, source_rebound_manifest={})
        for request in doc['requests']:
            request['pool_id'] = request['macros'][0]['pool_id']
    return plans, old, new


class IndependentAdapterTests(unittest.TestCase):
    def test_all_546_rows_equal_frozen_feature_math(self):
        plans, old, new = fixture()
        expected = adapter.frozen.prepared_rows(old, adapter.FEATURES)
        actual = adapter.prepared_rows_new(new, adapter.FEATURES)
        self.assertEqual(len(actual), 546)
        for left, right in zip(expected, actual):
            self.assertEqual(set(left), set(right))
            for key in left:
                if isinstance(left[key], np.ndarray):
                    np.testing.assert_array_equal(left[key], right[key])
                else:
                    self.assertEqual(left[key], right[key])

    def test_shifted_path_exposure_is_rejected(self):
        plans, old, new = fixture()
        piece = new[0]['requests'][0]['macros'][0]['causal_covariate_path'][0]
        piece['left_ns'] += 5
        piece['right_ns'] += 5
        piece['state']['cutoff_processing_ns'] += 5
        with self.assertRaises(ValueError):
            adapter.prepared_rows_new(new, adapter.FEATURES)

    def test_request_macro_gap_is_rejected(self):
        plans, old, new = fixture()
        request = new[0]['requests'][0]
        first = request['macros'][0]
        second = copy.deepcopy(first)
        second['macro_id'] = 'ready_tail'
        duration = first['response_wall_ns']
        for key in ('left_ns', 'right_ns'):
            second[key] += duration + 10
        piece = second['causal_covariate_path'][0]
        for key in ('left_ns', 'right_ns'):
            piece[key] += duration + 10
        piece['state']['cutoff_processing_ns'] = piece['left_ns']
        request['macros'].append(second)
        request['worker_start_to_ready_ns'] += duration
        with self.assertRaises(ValueError):
            adapter.prepared_rows_new(new, adapter.FEATURES)

    def test_actual_bundle_keeps_full_and_three_whole_folds_without_fit(self):
        plans, old, new = fixture()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(adapter, '_load_report', return_value={}), patch.object(adapter, '_rederive', return_value=(plans, new, [])):
                bundle = adapter.build_training_rows('fixture', workdir=directory)
        self.assertEqual(len(bundle['full']['rows']), 546)
        self.assertEqual(len(bundle['whole_block_folds']), 3)
        for fold in bundle['whole_block_folds']:
            self.assertEqual(len(fold['training_attempt_ids']), 8)
            self.assertEqual(len(fold['heldout_attempt_ids']), 4)
            self.assertTrue(set(fold['training_attempt_ids']).isdisjoint(fold['heldout_attempt_ids']))
            self.assertTrue(all(row['capture'] != fold['heldout_block_id'] for row in fold['training_rows']))
            self.assertTrue(all(row['capture'] == fold['heldout_block_id'] for row in fold['heldout_rows']))
            self.assertFalse(fold['fit_performed'])
        for flag in ('root_numerical_qualification', 'USHER_physical_qualification', 'numerical_fit_run', 'GPU_started', 'validation_labels_read', 'main_queue_provider_eligible', 'final_gain_eligible'):
            self.assertFalse(bundle[flag])


if __name__ == '__main__':
    unittest.main()
