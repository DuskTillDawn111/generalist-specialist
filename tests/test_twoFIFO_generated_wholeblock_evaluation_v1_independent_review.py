"""Independent generated fold evaluation dataflow and support accounting."""
import copy
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
import unittest
from tests.test_twoFIFO_generated_wholeblock_evaluation_v1 import public_case,past,law,scoring,QUALITY
from tools.predictor_validation_20261027 import twoFIFO_generated_wholeblock_evaluation_v1 as m


class GeneratedFoldIndependentReview(unittest.TestCase):
    def test_future_completion_change_changes_only_post_prediction_labels_and_errors(self):
        case=public_case();snapshot=past();donors=law()
        class BodyMustNeverBeRead(dict):
            def __getitem__(self,k):
                if k in ('execution_events','previous_handler_next_request_ready','worker_response'):
                    raise AssertionError('heldout future service body entered evaluation/generation')
                return super().__getitem__(k)
            def get(self,k,*default):
                if k in ('execution_events','previous_handler_next_request_ready','worker_response'):
                    raise AssertionError('heldout future service body entered evaluation/generation')
                return super().get(k,*default)
        first=dict(request_records=[BodyMustNeverBeRead(request_id=a['request_id'],status='completed',
            completion_time_ns=1000+a['arrival_ns']+2000) for a in case['arrivals']])
        second=dict(request_records=[BodyMustNeverBeRead(request_id=a['request_id'],status='completed',
            completion_time_ns=1000+a['arrival_ns']+9000) for a in case['arrivals']])
        with tempfile.TemporaryDirectory() as tmp:
            p1=m.evaluate_case(first,case,1000,snapshot,donors,scoring(),QUALITY,Path(tmp)/'first',donor_blocks=donors['blocks'])
            p2=m.evaluate_case(second,case,1000,snapshot,donors,scoring(),QUALITY,Path(tmp)/'second',donor_blocks=donors['blocks'])
            self.assertEqual(Path(p1['prediction']['path']).read_bytes(),Path(p2['prediction']['path']).read_bytes())
            self.assertNotEqual(Path(p1['labels']['path']).read_bytes(),Path(p2['labels']['path']).read_bytes())
            self.assertNotEqual(p1['errors'],p2['errors'])
            self.assertEqual(p1['ensemble_summary']['worlds'],8)
            self.assertEqual(p2['errors']['offered_requests'],2)

    def test_partial_donor_support_keeps_probability_and_conditional_error_scope(self):
        donor=law();donor['vectors']=[v for v in donor['vectors'] if v['training_block_id']=='training-block-1']
        case=public_case();prediction=m.compact_prediction(m.forecast_case(case,1000,past(),donor,scoring(),QUALITY))
        raw=dict(request_records=[dict(request_id=a['request_id'],status='completed',completion_time_ns=1000+a['arrival_ns']+2000)for a in case['arrivals']])
        labels=m.native_labels(raw,case,1000,scoring(),QUALITY);error=m.decode(m.errors(prediction,labels))
        self.assertEqual(error['offered_requests'],2);self.assertEqual(error['point_supported_requests'],0)
        for request in error['requests']:
            self.assertEqual(len(request['draws']),8)
            self.assertEqual(request['known_complete_probability_mass'],F(1,2))
            self.assertEqual(request['unknown_world_probability_mass'],F(1,2))
            self.assertIsNone(request['predicted_mean_reader_E2E_ns'])
            self.assertIsNone(request['error_ns'])
            self.assertIsNotNone(request['conditional_on_known_complete_mean_ns'])
        summary=m.fold_summary([dict(errors=error)])
        self.assertEqual(summary['offered_requests'],2)
        self.assertEqual(summary['unknown_or_incomplete_requests'],2)
        self.assertIsNone(summary['mean_absolute_reader_E2E_error_ns'])
        self.assertTrue(summary['metrics_conditioned_on_point_supported_subset'])

    def test_public_arrival_order_and_request_id_do_not_reassign_plan_slots(self):
        case=public_case();case['arrivals'].reverse()
        requests=m.public_case(case,1000);draws,selections=m.compile_case_draws(requests,law())
        renamed=[dict(r,request_id='completely-new-'+str(i))for i,r in enumerate(requests)]
        again,other=m.compile_case_draws(renamed,law())
        self.assertEqual(selections,other)
        self.assertEqual([d['inputs']for d in draws],[d['inputs']for d in again])
        self.assertTrue(all([v['public_key_arrival_ordinal']for v in s['selections']]==[0,1]for s in selections))
        expected={F(1,8),F(3,8),F(5,8),F(7,8)}
        self.assertEqual({m.public_quantile(('text#0','text','input'),0,i)for i in range(4)},expected)
        leaked=law();leaked['blocks'].append('training-block-3')
        with self.assertRaises(ValueError):m.compile_case_draws(requests,leaked)

    def test_late_native_result_retains_offered_truth_missing_instead_of_late_completion(self):
        case=public_case();deadline=1000+m.MEASUREMENT_NS+m.DRAIN_NS
        raw=dict(request_records=[dict(request_id=a['request_id'],status='completed',completion_time_ns=deadline+1)for a in case['arrivals']])
        labels=m.native_labels(raw,case,1000,scoring(),QUALITY)
        self.assertEqual(labels['original_score']['cohort_size'],2)
        self.assertEqual(len(labels['original_score']['incomplete_ids']),2)
        self.assertIsNone(labels['complete_cohort_reader_E2E_P95_ns'])
        self.assertTrue(all(not r['completed_within_fixed_drain'] and r['reader_E2E_ns'] is None for r in labels['requests']))

    def test_real_atomic_prediction_fsync_precedes_case_evaluation_label_extraction(self):
        import os
        from unittest.mock import patch
        from tools.predictor_validation_20261027 import twoFIFO_owned_collect_v3 as collector
        case=public_case();raw=dict(request_records=[dict(request_id=a['request_id'],status='completed',
            completion_time_ns=1000+a['arrival_ns']+2000)for a in case['arrivals']])
        synced=[];original_fsync=os.fsync;original_labels=m.native_labels
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'case'
            def fsync(fd):
                synced.append(Path(os.readlink('/proc/self/fd/'+str(fd))).name)
                return original_fsync(fd)
            def labels(*args):
                self.assertIn('.prediction.json.tmp',synced)
                self.assertTrue((folder/'prediction.json').exists())
                return original_labels(*args)
            with patch.object(collector.os,'fsync',side_effect=fsync),patch.object(m,'native_labels',side_effect=labels):
                out=m.evaluate_case(raw,case,1000,past(),law(),scoring(),QUALITY,folder,
                    donor_blocks=['training-block-1','training-block-2'])
            self.assertTrue(out['durable_before_case_evaluation_label_extraction'])
            self.assertNotIn('predictions_durable_before_future_label_access',out)
            label=json.loads((folder/'labels.json').read_text())
            self.assertFalse(label['actual_native_quality_validated'])
            self.assertTrue(label['quality_from_frozen_training_expectations_not_native_outcome'])
