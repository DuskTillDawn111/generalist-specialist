import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027 import complete_training_index_v12 as module


class CompleteTrainingIndexTests(unittest.TestCase):
    def capture(self):
        return dict(kind='measured', status='completed', execution_context_sha256='context',
            request_records=[dict(request_id='measured', pool_id='pool', task='text',
                status='completed', split='training')],
            priming_request_records=[dict(request_id='primer', pool_id='pool', task='text',
                status='completed', phase='priming')])

    def records(self, raw=None):
        return module.records_for_capture(raw or self.capture(),
            dict(split='training', training_only=True, execution_context_sha256='context'),
            dict(path='/original/raw.json', sha256='original-sha'))

    def test_unsplit_completed_priming_is_conservative_identity_only(self):
        rows, measured, primers = self.records()
        self.assertEqual((measured, primers), (1, 1))
        self.assertEqual({r['run_id'] for r in rows}, {'raw-sha256:original-sha'})
        self.assertTrue(all(set(r) == module.FIELDS for r in rows))

    def test_measured_missing_split_and_validation_are_rejected(self):
        for split in (None, 'validation'):
            raw = self.capture()
            raw['request_records'][0]['split'] = split
            with self.assertRaisesRegex(ValueError, 'measured row'):
                self.records(raw)

    def test_unsplit_nonpriming_or_failed_primers_rejected(self):
        for key, value in [('phase', 'measurement'), ('status', 'failed'), ('split', 'validation')]:
            raw = self.capture()
            raw['priming_request_records'][0][key] = value
            with self.assertRaisesRegex(ValueError, 'priming'):
                self.records(raw)

    def test_raw_identity_deduplicates_repeated_review_not_human_labels(self):
        rows, _, _ = self.records()
        first = dict(run_id=rows[0]['run_id'], reviews=[{'path':'review1'}], original_run_label='label1')
        second = dict(run_id=rows[0]['run_id'], reviews=[{'path':'review2'}], original_run_label='label2')
        registry = {}
        self.assertTrue(module.merge_capture(registry, rows, first))
        self.assertFalse(module.merge_capture(registry, copy.deepcopy(rows), second))
        self.assertEqual(len(registry), 1)
        self.assertEqual(len(first['reviews']), 2)
        conflicting = copy.deepcopy(rows)
        conflicting[0]['task'] = 'ocr'
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            module.merge_capture(registry, conflicting, second)

    def test_extra_record_fields_and_repeated_request_rejected(self):
        rows, _, _ = self.records()
        for bad in ([dict(rows[0], phase='measurement')], [rows[0], rows[0]]):
            with self.assertRaises(ValueError):
                module.validate_index(dict(schema=module.DATA_SCHEMA, split='training', records=bad))

    def test_public_disjointness_and_actual_content_mismatch(self):
        rows, _, _ = self.records()
        index = dict(schema=module.DATA_SCHEMA, split='training', records=rows)
        catalog = {'pool':dict(pool_id='pool', task_type='text', sample=dict(context='original', question='q'))}
        declaration = {'pool':dict(task='text', exact_input_sha256=module.input_identity(catalog['pool']))}
        with tempfile.TemporaryDirectory() as temporary:
            public = Path(temporary)/'public.json'
            def job(request_id='fresh', run_id='fresh-run'):
                public.write_text(json.dumps(dict(forecast_inputs=dict(queues={},
                    future_workload=dict(declared_arrivals=[dict(request_id=request_id, pool_id='pool', task='text')])))))
                return [dict(run_id=run_id, candidate_id='candidate', public_inputs=module.receipt(public))]
            result = module.public_disjointness(index, job(), catalog, declaration)
            self.assertEqual(result[0]['requests'], 1)
            for arguments in (dict(request_id='primer'), dict(run_id='raw-sha256:original-sha')):
                with self.assertRaisesRegex(ValueError, 'overlap'):
                    module.public_disjointness(index, job(**arguments), catalog, declaration)
            changed = copy.deepcopy(catalog)
            changed['pool']['sample']['context'] = 'different'
            with self.assertRaisesRegex(ValueError, 'trained and public'):
                module.public_disjointness(index, job(), changed, declaration)

    def test_nontraining_review_plan_rejected_before_opening_raw(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'plan.json').write_text(json.dumps(dict(split='validation', training_only=False)))
            (root/'context.json').write_text('{}')
            review = dict(schema='gsdeploy.native_queue_training_strict_review.v8', status='PASS',
                training_only=True, evidence=dict(raw={'path':'/does-not-exist/raw.json', 'sha256':'x'},
                    plan=module.receipt(root/'plan.json'), context=module.receipt(root/'context.json')))
            (root/'review.json').write_text(json.dumps(review))
            with self.assertRaisesRegex(ValueError, 'before opening'):
                module.review_capture(module.receipt(root/'review.json'))

    def test_smoke_fit_reference_is_unresolved_but_closure_only_is_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = dict(path=str(root/'smoke/raw.json'), sha256='not-opened')
            model = root/'model.json'
            model.write_text(json.dumps(dict(schema='model', captures=[dict(raw=raw)])))
            result = module.lineage_audit([module.receipt(model)], [])
            self.assertEqual(len(result['unresolved_raw_references']), 1)
            model.write_text(json.dumps(dict(schema='model', evidence_closure=dict(materialized_sources=[raw]))))
            result = module.lineage_audit([module.receipt(model)], [])
            self.assertEqual(result['unresolved_raw_references'], [])
            self.assertEqual(result['raw_references'][0]['classification'],
                'INHERITED_RUNTIME_SMOKE_PROVENANCE_ONLY_NOT_INDEXED')


if __name__ == '__main__':
    unittest.main()
