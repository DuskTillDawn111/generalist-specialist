import copy
import json
import unittest

from tests import test_flow_validation as fixtures
from tools.predictor_validation_20261027.flow_crossfit_report import (
    predict, compare, ranking, scored_heldout)
from tools.predictor_validation_20261027.probe_validation import receipt


class CrossfitReportTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.FlowValidationTests('test_full_arrivals_parent_receipt_and_wait_service_separated')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_full_arrival_crossfit_retains_parent_tail_error(self):
        f = self.fixture
        load = lambda path: json.loads(path.read_text())
        prediction = predict(f.plan, f.plan['candidates'][0], load(f.model_path),
                             load(f.resources_path), load(f.quality_path))
        result = compare(f.plan, f.plan['candidates'][0], prediction, f.raw())
        self.assertEqual(result['known'], 2)
        self.assertEqual(prediction['requests'][1]['wait_ms'], 50)
        self.assertEqual(result['timing_errors']['e2e']['mae'], 10)
        self.assertFalse(result['eligible_for_gain_claim'])

    def test_missing_input_does_not_disappear_from_denominator(self):
        f = self.fixture
        model = json.loads(f.model_path.read_text())
        candidate = f.plan['candidates'][0]
        f.plan['trace'][1]['pool_id'] = 'unseen'
        prediction = predict(f.plan, candidate, model, json.loads(f.resources_path.read_text()),
                             json.loads(f.quality_path.read_text()))
        self.assertEqual(len(prediction['requests']), 2)
        self.assertIsNone(prediction['requests'][1]['e2e_ms'])
        self.assertIsNotNone(prediction['unknown_reason'])

    def test_unknown_utilities_are_not_counted_as_correct_ties(self):
        rows = [{'candidate_id': 'b', 'predicted_comparison_U': -1., 'observed_U': -2.},
                {'candidate_id': 'good', 'predicted_comparison_U': -.5, 'observed_U': -1.},
                {'candidate_id': 'unknown', 'predicted_comparison_U': None, 'observed_U': -1.}]
        value = ranking(rows, 'b')
        self.assertEqual(value['known_delta_pairs'], 1)
        self.assertEqual(value['known_rank_pairs'], 1)
        self.assertEqual(value['rank_accuracy'], 1.)
        with self.assertRaises(ValueError):
            ranking(rows + rows[:1], 'b')

    def test_scored_overlay_cannot_change_native_timing(self):
        f = self.fixture
        raw = f.raw(); quality = raw.pop('quality_evidence')
        path = f.write('raw.json', raw)
        scored = {**raw, 'native_raw': receipt(path), 'quality_evidence': quality}
        scored_path = f.write('scored-raw.json', scored)
        result, ref = scored_heldout(receipt(path))
        self.assertEqual(result['quality_evidence'], quality)
        self.assertEqual(ref, receipt(scored_path))
        bad = copy.deepcopy(scored)
        bad['request_records'][0]['completion_time_ns'] += 10
        f.write('scored-raw.json', bad)
        with self.assertRaisesRegex(ValueError, 'immutable'):
            scored_heldout(receipt(path))


if __name__ == '__main__':
    unittest.main()
