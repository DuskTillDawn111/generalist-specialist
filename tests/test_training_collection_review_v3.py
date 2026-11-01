from copy import deepcopy
import unittest

from tools.predictor_validation_20261027.training_collection_review_v3 import coverage


class TrainingCollectionCoverageTests(unittest.TestCase):
    def fixture(self):
        job=dict(id='j',split='training',inputs={'text#0':'p'},tasks={'text#0':'text'})
        row=dict(request_id='r',profile_job_id='j',model='text#0',pool_id='p',task='text',split='training')
        return {'jobs':[job]}, {'request_records':[row],'batches':[dict(job_id='j',request_ids=['r'])]}, {'jobs':[job]}

    def test_exact_training_denominator(self):
        plan,raw,executed=self.fixture()
        self.assertEqual(coverage(plan,raw,executed),[])
        raw['request_records']=[]
        self.assertIn('EXACT_TRAINING_REQUEST_COVERAGE_FAILED',coverage(plan,raw,executed))

    def test_wrong_job_batch_and_split_do_not_pass(self):
        plan,raw,executed=self.fixture()
        raw['request_records'][0]['split']='validation'
        self.assertIn('EXACT_TRAINING_REQUEST_COVERAGE_FAILED',coverage(plan,raw,executed))
        raw['batches'][0]['job_id']='other'
        self.assertIn('EXACT_TRAINING_JOB_COVERAGE_FAILED',coverage(plan,raw,executed))


if __name__=='__main__':unittest.main()
