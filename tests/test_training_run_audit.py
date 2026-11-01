import copy
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027.training_run_audit import quality_totals, verify_quality_annotation


class TrainingRunAuditTest(unittest.TestCase):
    def setUp(self):
        self.catalog_ref = {'path': '/frozen/catalog.json', 'sha256': 'catalog'}
        self.plan = {'trace': [{'request_id': 'a'}, {'request_id': 'b'}], 'catalog': self.catalog_ref}
        self.rows = [dict(request_id=rid, task='text', pool_id=rid, status='completed',
            worker_response={'prediction': rid}) for rid in ('a', 'b')]
        self.catalog = {'quality': {'references': {'text': {'a': 'a', 'b': 'b'}}}}
        self.totals = {rid: dict(task='text', numerator=1., denominator=1.) for rid in ('a', 'b')}
        self.raw = {'request_records': self.rows, 'quality_evidence': dict(kind='measured_per_request',
            errors=[], source_receipts=[self.catalog_ref], totals_by_request=copy.deepcopy(self.totals))}

    def test_recompute_preserves_native_and_checks_full_cohort(self):
        before = copy.deepcopy(self.raw)
        with patch('tools.predictor_validation_20261027.training_run_audit.score_result', return_value={}), \
                patch('tools.predictor_validation_20261027.training_run_audit.sample_totals', return_value=(1., 1.)):
            self.assertEqual(quality_totals(self.rows, self.catalog), self.totals)
            self.assertEqual(verify_quality_annotation(self.raw, self.plan, self.catalog)['count'], 2)
            self.assertEqual(self.raw, before)
            bad = copy.deepcopy(self.raw)
            bad['quality_evidence']['totals_by_request']['a']['numerator'] = 0.
            with self.assertRaisesRegex(ValueError, 'recomputed'):
                verify_quality_annotation(bad, self.plan, self.catalog)
            with self.assertRaisesRegex(ValueError, 'every offered'):
                verify_quality_annotation(self.raw, {**self.plan, 'trace': self.plan['trace'][:1]}, self.catalog)

    def test_duplicate_incomplete_or_foreign_catalog_rejected(self):
        with patch('tools.predictor_validation_20261027.training_run_audit.score_result', return_value={}), \
                patch('tools.predictor_validation_20261027.training_run_audit.sample_totals', return_value=(1., 1.)):
            with self.assertRaisesRegex(ValueError, 'Unique completed'):
                quality_totals([self.rows[0], self.rows[0]], self.catalog)
            incomplete = [{**self.rows[0], 'status': 'failed'}]
            with self.assertRaisesRegex(ValueError, 'Unique completed'):
                quality_totals(incomplete, self.catalog)
            bad = copy.deepcopy(self.raw)
            bad['quality_evidence']['source_receipts'] = []
            with self.assertRaisesRegex(ValueError, 'exact frozen plan catalog'):
                verify_quality_annotation(bad, self.plan, self.catalog)


if __name__ == '__main__':
    unittest.main()
