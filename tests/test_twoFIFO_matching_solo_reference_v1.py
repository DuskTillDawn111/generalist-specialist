import copy
import json
from pathlib import Path
import tempfile
import unittest
from tools.predictor_validation_20261027 import twoFIFO_matching_solo_reference_v1 as m
from tools.predictor_validation_20261027.stage_response_features_v1 import receipt


class SoloReferenceTests(unittest.TestCase):
    def fixture(self,root):
        context=root/'context.json';context.write_text(json.dumps(dict(worker_bindings={'text':dict(spec_sha256='s',command=['worker'])})))
        pool=root/'pool.jsonl';pool.write_text(json.dumps(dict(pool_id='p',sample=dict(task='text',text='exact')))+'\n')
        plans=[];raws=[]
        for b in range(3):
            for s in range(5):
                aid=str(b)+'-'+str(s);arrivals=[dict(request_id=aid+'-'+str(i),owner='text#0',task='text',pool_id='p') for i in range(2)]
                plan=dict(attempt_id=aid,execution_id=aid,capture_id=str(b),prepared_pool=receipt(pool),context=receipt(context),
                    cases=[dict(case_id='solo',mode='solo_same_owner_burst',arrivals=arrivals)] if s==0 else [])
                rows=[dict(request_id=a['request_id'],model='text#0',task='text',pool_id='p',observed_worker_start_time_ns=10+i*10,
                    previous_handler_next_request_ready=dict(perf_ns=20+i*10+b)) for i,a in enumerate(arrivals)] if s==0 else []
                raw=dict(attempt_id=aid,execution_id=aid,priming_request_records=[],request_records=rows)
                plans.append(plan);raws.append(raw)
        return plans,raws

    def test_three_block_weighting_nopeer_from_fullhistory_and_conditional_ratio(self):
        with tempfile.TemporaryDirectory() as temp:
            plans,raws=self.fixture(Path(temp));result=m.extract_rows(plans,raws)
        key=result['keys'][0];self.assertEqual(key['supported_samples'],6);self.assertEqual(key['supported_blocks'],3)
        self.assertEqual(key['equally_weighted_block_mean_host_ns'],11);self.assertEqual(key['block_mean_range_ns'],[10,12])
        result.update(schema=m.SCHEMA,source_evidence_authenticated=True,target_bindings=[dict(owner='text#0',task='text',pool_id='p',three_block_reference_available=True)],missing_reference_collection_agenda=[])
        comparison=m.decode(m.slowdown_labels(dict(requests=[dict(request_id='v',owner='text#0',task='text',pool_id='p',actual_START_to_READY_ns=22)]),m.encode(result),
            m.encode(dict(requests=[dict(request_id='v',predicted_mean_START_to_READY_ns=33)]))))
        row=comparison['requests'][0];self.assertEqual(row['conditional_host_solo_reference_ratio'],2)
        self.assertEqual(row['predicted_conditional_host_solo_reference_ratio'],3);self.assertEqual(row['conditional_ratio_error'],1)
        self.assertFalse(row['physical_slowdown_qualified'])

    def test_priming_and_othercase_overlap_invalidate_solo_not_calendar_intent(self):
        with tempfile.TemporaryDirectory() as temp:
            plans,raws=self.fixture(Path(temp));raws[0]['priming_request_records'].append(dict(request_id='primingpeer',model='ocr#0',
                observed_worker_start_time_ns=0,previous_handler_next_request_ready=dict(perf_ns=25)))
            result=m.extract_rows(plans,raws)
        key=result['keys'][0];self.assertEqual(key['supported_samples'],4);self.assertEqual(key['supported_blocks'],2)
        self.assertIsNone(key['equally_weighted_block_mean_host_ns']);self.assertTrue(result['rows'][0]['other_owner_overlaps'])
        self.assertEqual(result['rows'][0]['full_attempt_history_requests'],3)

    def test_missing_any_history_READY_refuses_reference_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            plans,raws=self.fixture(Path(temp));raws[1]['priming_request_records'].append(dict(request_id='pendingpeer',model='ocr#0',observed_worker_start_time_ns=1))
            with self.assertRaisesRegex(ValueError,'full-attempt'):m.extract_rows(plans,raws)

    def test_one_of_two_overlap_never_renormalizes_planned_block(self):
        with tempfile.TemporaryDirectory() as temp:
            plans,raws=self.fixture(Path(temp));raws[0]['priming_request_records'].append(dict(request_id='peer',model='ocr#0',
                observed_worker_start_time_ns=11,previous_handler_next_request_ready=dict(perf_ns=12)))
            result=m.extract_rows(plans,raws)
        key=result['keys'][0];self.assertEqual(key['supported_samples'],5);self.assertEqual(key['supported_blocks'],2)
        self.assertEqual(len(key['block_summaries']),3);block=key['block_summaries'][0]
        self.assertEqual((block['required_samples'],block['planned_samples'],block['samples']),(2,2,1))
        self.assertIsNone(block['mean_ns']);self.assertEqual(block['conditional_supported_sample_mean_ns'],10)
        self.assertFalse(block['complete_planned_block']);self.assertIsNone(key['equally_weighted_block_mean_host_ns'])
        self.assertFalse(key['samples_within_block_renormalized'])

if __name__=='__main__':unittest.main()
