import unittest
from unittest.mock import patch

from tests import test_flow_validation as fixtures
from tools.predictor_validation_20261027 import flow_trained_validation as entry


class TransitiveFlowEntryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.FlowValidationTests('test_full_arrivals_parent_receipt_and_wait_service_separated')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.fixture.plan['phase'] = 'predictor_validation'
        self.fixture.write('plan.json', self.fixture.plan)
        self.valid = {'verified': True, 'training_flow_count': 8, 'omitted_flow_ids': [], 'omitted_seed': None}

    def test_crossfit_models_cannot_be_used_as_full_independent_model(self):
        for change in ({'training_flow_count': 7}, {'omitted_flow_ids': ['flow']},
                       {'omitted_seed': 1}, {'verified': False}):
            with patch('tools.predictor_validation_20261027.flow_fit.verify_training_model',
                       return_value={**self.valid, **change}):
                with self.assertRaises(ValueError):
                    entry.audit_model(self.fixture.model_path)

    def test_report_rechecks_transitive_training_verifier(self):
        f = self.fixture
        with patch('tools.predictor_validation_20261027.flow_fit.verify_training_model', return_value=self.valid):
            sealed = entry.seal(f.plan_path, f.model_path, f.resources_path, f.quality_path, f.root / 'sealed')
        raws = [f.write(cid + '.json', f.raw(cid, sealed['created_ns'] + 2 * fixtures.NS))
                for cid in ('base', 'alternative')]
        with patch('tools.predictor_validation_20261027.flow_fit.verify_training_model',
                   side_effect=ValueError('Changed transitive native raw')):
            with self.assertRaisesRegex(ValueError, 'transitive native raw'):
                entry.report(f.root / 'sealed', raws, f.root / 'report')
        self.assertFalse((f.root / 'report').exists())

    def test_complete_report_preserves_independent_parent_receipt_metrics(self):
        f = self.fixture
        with patch('tools.predictor_validation_20261027.flow_fit.verify_training_model', return_value=self.valid):
            sealed = entry.seal(f.plan_path, f.model_path, f.resources_path, f.quality_path, f.root / 'sealed')
            raws = [f.write(cid + '.json', f.raw(cid, sealed['created_ns'] + 2 * fixtures.NS))
                    for cid in ('base', 'alternative')]
            result = entry.report(f.root / 'sealed', raws, f.root / 'report')
        self.assertEqual(result['status'], 'ASSESSED')
        self.assertEqual(result['candidates'][0]['timing_errors']['e2e']['mae'], 10)
        self.assertTrue((f.root / 'report' / 'training-audit.json').is_file())
        self.assertFalse(result['final_four_groups_enabled'])


if __name__ == '__main__':
    unittest.main()
