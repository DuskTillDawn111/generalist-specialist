import unittest

from tests.test_kernel_endpoint_candidate_v27 import row
from tools.predictor_validation_20261027.issued_queue_endpoint_candidate_v28 import (
    IssuedQueuePrefix,features,fit,prediction)


class IssuedQueueTests(unittest.TestCase):
    def test_later_end_time_does_not_change_past_queue_prefix(self):
        first=IssuedQueuePrefix({('a',1):[(10,100)],('b',2):[(20,30)]})
        changed=IssuedQueuePrefix({('a',1):[(10,1000)],('b',2):[(20,30)]})
        self.assertEqual(first.counts(95),changed.counts(95))
        self.assertEqual(first.counts(100)[('a',1)],0)
        self.assertEqual(changed.counts(100)[('a',1)],1)
        self.assertEqual(first.counts(15)[('b',2)],0)

    def test_endpoint_candidate_uses_issued_not_only_started_work(self):
        rows=[]
        for queue in (1,2,4,8,16):
            value=row(False,remaining=10*queue)
            value['prefix_features'].update(measured_issued_unfinished_own_count=queue,
                measured_issued_unfinished_other_owner_count=0,
                measured_issued_unfinished_same_stream_count=queue)
            rows.append(value)
        model=fit(rows,'END_if_not_ended')
        means={tuple(value['key']):value['value'] for value in model['signature_mean_log1p_ns']}
        low,_=prediction(model,rows[0]['prefix_features'],means)
        high,_=prediction(model,rows[-1]['prefix_features'],means)
        self.assertGreater(high,low)
        self.assertEqual(len(features(rows[0]['prefix_features'])),len(model['feature_names']))


if __name__=='__main__':unittest.main()
