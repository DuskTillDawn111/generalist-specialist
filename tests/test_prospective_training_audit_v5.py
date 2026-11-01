import copy
import unittest

from tools.predictor_validation_20261027 import prospective_training_audit_v5 as tool


def fixture():
    public = dict(inputs=[dict(pool_id='heldout', exact_input_sha256='new')])
    plan = dict(split='training', inputs=[dict(task='text', pool_id='train', exact_input_sha256='old'),
                                        dict(task='text', pool_id='priming-only', exact_input_sha256='priming')])
    raw = dict(status='completed', request_records=[dict(request_id='r', pool_id='train', task='text',
                                                       split='training', status='completed')])
    report = dict(schema='gsdeploy.startup_training_collection_strict_review.v4', status='PASS', issues=[],
                  training_only=True, raw=dict(sha256='raw'), capture_run_id='raw-sha256:raw')
    index = dict(schema=tool.DATA_SCHEMA, split='training', records=[dict(run_id='raw-sha256:raw', request_id='r',
                                                                       pool_id='train', task='text')])
    return index, public, [(report, plan, raw)]


class ProspectiveTrainingAuditTests(unittest.TestCase):
    def test_includes_full_model_training_and_priming_exclusions(self):
        index, public, reviews = fixture()
        result = tool.check_documents(index, public, reviews, [copy.deepcopy(index)])
        self.assertEqual(result['declared_training_inputs'], 2)
        self.assertEqual(result['required_unique_training_requests'], 1)

    def test_new_model_capture_cannot_be_omitted_from_union_index(self):
        index, public, reviews = fixture()
        second = copy.deepcopy(reviews[0])
        second[0].update(raw=dict(sha256='second'), capture_run_id='raw-sha256:second')
        with self.assertRaisesRegex(ValueError, 'omits real model'):
            tool.check_documents(index, public, reviews + [second], [])

    def test_nested_model_index_cannot_add_unlisted_training(self):
        index, public, reviews = fixture()
        nested = copy.deepcopy(index)
        nested['records'][0]['run_id'] = 'unlisted'
        with self.assertRaisesRegex(ValueError, 'omits or duplicates nested'):
            tool.check_documents(index, public, reviews, [nested])

    def test_renamed_training_content_and_priming_content_are_not_holdout(self):
        index, public, reviews = fixture()
        for sha in ('old', 'priming'):
            changed = copy.deepcopy(public)
            changed['inputs'][0]['exact_input_sha256'] = sha
            with self.assertRaisesRegex(ValueError, 'overlaps model training content'):
                tool.check_documents(index, changed, reviews, [])

    def test_nontraining_or_duplicate_request_cannot_enter_model_lineage(self):
        index, public, reviews = fixture()
        for change in ('split', 'duplicate', 'status'):
            changed = copy.deepcopy(reviews)
            if change == 'duplicate':
                changed[0][2]['request_records'].append(copy.deepcopy(changed[0][2]['request_records'][0]))
            elif change == 'status':
                changed[0][0]['status'] = 'FAIL'
            else:
                changed[0][2]['request_records'][0]['split'] = 'validation'
            with self.assertRaises(ValueError):
                tool.check_documents(index, public, changed, [])

    def test_index_cannot_add_unaudited_or_heldout_training_rows(self):
        index, public, reviews = fixture()
        index['records'].append(dict(run_id='unaudited', request_id='x', pool_id='heldout', task='text'))
        with self.assertRaisesRegex(ValueError, 'real model training'):
            tool.check_documents(index, public, reviews, [])

    def test_duplicate_plan_id_cannot_hide_training_content(self):
        index, public, reviews = fixture()
        reviews[0][1]['inputs'].insert(0, dict(task='text', pool_id='train', exact_input_sha256='new'))
        with self.assertRaisesRegex(ValueError, 'Duplicate declared'):
            tool.check_documents(index, public, reviews, [])

    def test_actual_priming_cannot_use_unlisted_input(self):
        index, public, reviews = fixture()
        reviews[0][2]['priming_request_records'] = [dict(pool_id='heldout', task='text', status='completed')]
        with self.assertRaisesRegex(ValueError, 'Actual priming'):
            tool.check_documents(index, public, reviews, [])


if __name__ == '__main__':
    unittest.main()
