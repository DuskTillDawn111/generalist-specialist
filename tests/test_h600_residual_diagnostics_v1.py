import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

from tools.predictor_validation_20261027 import h600_residual_diagnostics_v1 as module


class ResidualDiagnosticTests(unittest.TestCase):
    def sample(self,predicted,actual,third=0):
        return dict(alias='a',task='ocr',pool_id='input',arrival_third=third,
            predicted_service_ms=predicted,actual_service_ms=actual,
            predicted_e2e_ms=predicted,actual_e2e_ms=actual+1,
            predicted_wait_ms=0. if predicted is not None else None,actual_wait_ms=0.,
            parent_tail_ms=1.,submit_to_START_ms=.5,scheduled_to_submit_ms=.25,
            W_vocabulary=dict(status='UNKNOWN_W_VOCABULARY'))

    def test_signed_bias_service_sum_and_P95_use_predicted_minus_actual(self):
        rows=[self.sample(12.,10.),self.sample(18.,20.)]
        value=module.summarize(rows,input_group=True)
        self.assertEqual(value['service']['paired_signed_bias_ms'],0.)
        self.assertEqual(value['service']['full_cohort_sum_predicted_minus_actual_ms'],0.)
        self.assertAlmostEqual(value['service']['full_predicted_p95_ms'],17.7)
        self.assertAlmostEqual(value['service']['full_actual_p95_ms'],19.5)
        self.assertAlmostEqual(value['service']['full_p95_signed_difference_ms'],-1.8)
        self.assertEqual(value['e2e']['paired_signed_bias_ms'],-1.)
        self.assertTrue(value['sparse_input_group'])

    def test_unknown_never_leaves_offered_denominator_or_full_P95(self):
        rows=[self.sample(12.,10.),self.sample(None,20.)]
        value=module.summarize(rows,input_group=True)
        self.assertEqual(value['offered_count'],2);self.assertEqual(value['unknown_count'],1)
        self.assertEqual(value['service']['paired_known_count'],1)
        self.assertEqual(value['service']['paired_sum_predicted_minus_actual_ms'],2.)
        self.assertIsNone(value['service']['full_cohort_sum_predicted_minus_actual_ms'])
        self.assertIsNone(value['service']['full_predicted_p95_ms'])
        self.assertAlmostEqual(value['service']['full_actual_p95_ms'],19.5)
        self.assertEqual(value['service']['paired_subset_actual_p95_ms'],10.)

    def test_minimum_input_n5_and_public_arrival_third_groups(self):
        rows=[self.sample(1.,1.,third=i%3) for i in range(5)]
        value=module.grouped(rows)
        self.assertFalse(value['by_alias_task_input'][0]['sparse_input_group'])
        self.assertEqual(sum(r['offered_count'] for r in value['by_alias_task_input_arrival_third']),5)
        self.assertTrue(all(r['sparse_input_group'] for r in value['by_alias_task_input_arrival_third']))

    def test_OCR_observed_vocabulary_unknown_does_not_imply_forecast_unknown(self):
        provider=Mock();provider.plan.return_value={'segments':[{'id':'ocr_handler'}]}
        row=dict(alias='a',task='ocr',pool_id='input',segments=[{'phase':'ocr_native'}])
        diagnostic=module.vocabulary_audit(provider,{'a':{'family':'ocr'}},row)
        self.assertEqual(diagnostic['status'],'UNKNOWN_W_VOCABULARY')
        self.assertIsNone(diagnostic['fresh_forecast_implication'])
        sample=self.sample(1.,1.);sample['W_vocabulary']=diagnostic
        self.assertEqual(module.summarize([sample])['fresh_forecast_known_count'],1)
        provider.plan.assert_called_once_with(dict(alias='a',task='ocr',pool_id='input'),{'a':{'family':'ocr'}},'mean')

    def join_fixture(self):
        plan=dict(window_seconds=3.,aliases={'a':{'family':'ocr'}},trace=[
            dict(request_id=str(i),task='ocr',pool_id='input',offset_ns=i*1_000_000_000) for i in range(3)])
        candidate=dict(routes={r['request_id']:'a' for r in plan['trace']})
        origin=100_000_000_000;records=[];mapped=[];predicted=[]
        for row in plan['trace']:
            scheduled=origin+row['offset_ns']
            records.append(dict(request_id=row['request_id'],task='ocr',pool_id='input',model='a',status='completed',
                scheduled_time_ns=scheduled,actual_submit_time_ns=scheduled+2_000_000,
                observed_worker_start_time_ns=scheduled+5_000_000,
                worker_response=dict(end_perf_ns=scheduled+15_000_000),completion_time_ns=scheduled+16_000_000))
            mapped.append(dict(native_request_id=row['request_id'],request_id='scoped:'+row['request_id'],alias='a',task='ocr',pool_id='input',segments=[dict(phase='ocr_native')]))
            predicted.append(dict(request_id=row['request_id'],service_ms=10.,wait_ms=4.,e2e_ms=14.))
        raw=dict(phase_start_ns=origin,observation_end_ns=origin+4_000_000_000,request_records=records)
        prediction=dict(requests=predicted,unknown_reason=None)
        mapping=dict(traces=[dict(requests=mapped)])
        provider=Mock();provider.plan.return_value=dict(segments=[dict(id='ocr_handler')])
        return plan,candidate,prediction,raw,mapping,provider

    def test_actual_submit_START_parent_tail_and_public_thirds_remain_separate(self):
        rows=module.join_rows(*self.join_fixture())
        self.assertEqual([r['arrival_third'] for r in rows],[0,1,2])
        for r in rows:
            self.assertEqual(r['actual_service_ms'],10.)
            self.assertEqual(r['actual_wait_ms'],5.)
            self.assertEqual(r['actual_e2e_ms'],16.)
            self.assertEqual(r['parent_tail_ms'],1.)
            self.assertEqual(r['submit_to_START_ms'],3.)
            self.assertEqual(r['scheduled_to_submit_ms'],2.)
        self.assertEqual(module.summarize(rows)['e2e']['paired_signed_bias_ms'],-2.)

    def test_missing_submit_stays_unknown_and_bad_submit_clock_rejected(self):
        fixture=self.join_fixture();fixture[3]['request_records'][0].pop('actual_submit_time_ns')
        rows=module.join_rows(*fixture)
        self.assertIsNone(rows[0]['submit_to_START_ms'])
        self.assertEqual(module.summarize(rows)['actual_submit_to_START_ms']['count'],2)
        fixture[3]['request_records'][0]['actual_submit_time_ns']=fixture[3]['request_records'][0]['observed_worker_start_time_ns']+1
        with self.assertRaisesRegex(ValueError,'host submit'):module.join_rows(*fixture)

    def test_missing_forecast_row_cannot_silently_trim_cohort(self):
        fixture=self.join_fixture();fixture[2]['requests'].pop()
        with self.assertRaisesRegex(ValueError,'Every offered'):module.join_rows(*fixture)

    def test_prepare_does_not_launch_model_and_missing_report_no_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'output'
            with patch.object(module.frozen.core,'predict') as predict:
                with self.assertRaisesRegex(ValueError,'Complete frozen'):module.run(Path(temp)/'missing.json','unused',out)
                predict.assert_not_called();self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
