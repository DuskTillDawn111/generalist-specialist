import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.predictor_validation_20261027.flow_validation_run import prepare
from tools.predictor_validation_20261027.native_flow import receipt


class IndependentRunPacketTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.plan = self.write('plan.json', {'phase': 'predictor_validation', 'seed': 100,
                            'candidates': [{'id': 'a'}, {'id': 'b'}]})
        self.sealed = self.root / 'sealed'; self.sealed.mkdir()
        self.prediction = {'sources': {'plan': receipt(self.plan)}, 'candidates': [
            {'candidate_id': cid, 'unknown_reason': None, 'metrics': {'six_term_comparison_complete': True},
             'requests': [{'e2e_ms': 1.}]} for cid in ('a', 'b')]}
        self.write('sealed/predictions.json', self.prediction)
        self.write('sealed/training-audit.json', {})
        self.qualification = self.write('qualification.json', {'status': 'FROZEN'})
        mock = patch('tools.predictor_validation_20261027.flow_validation_run.check_sealed', return_value={})
        mock.start(); self.addCleanup(mock.stop)

    def write(self, name, value):
        path = self.root / name; path.write_text(json.dumps(value)); return path

    def test_suite_binds_complete_predictions_and_predeclares_no_retry_order(self):
        value = prepare(self.plan, self.sealed, self.qualification, self.root / 'runsuite')
        self.assertEqual({j['candidate_id'] for j in value['jobs']}, {'a', 'b'})
        self.assertEqual(value['sources']['qualification'], receipt(self.qualification))
        self.assertTrue(value['no_retries'])
        self.assertFalse(value['GPU_started'])

    def test_unknown_predictions_prevent_gpu_suite(self):
        self.prediction['candidates'][0]['requests'][0]['e2e_ms'] = None
        self.write('sealed/predictions.json', self.prediction)
        with self.assertRaises(ValueError):
            prepare(self.plan, self.sealed, self.qualification, self.root / 'runsuite')
        self.assertFalse((self.root / 'runsuite').exists())


if __name__ == '__main__':
    unittest.main()
