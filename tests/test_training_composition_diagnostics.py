import unittest
from tools.predictor_validation_20261027.training_composition_diagnostics import summarize


class CompositionTests(unittest.TestCase):
    def test_input_mix_reweighting_separates_mix_and_marks_missing_support(self):
        rows=[]
        for third,samples in enumerate(([('fast',10),('fast',10),('slow',100)],
            [('fast',10),('slow',100),('slow',100)],[('fast',10),('fast',10)])):
            for i,(pool,service) in enumerate(samples):
                t=(third*100+i)*1_000_000
                rows.append(dict(status='completed',scheduled_time_ns=t,model='a',task='text',pool_id=pool,
                    observed_worker_start_time_ns=t,worker_response={'end_perf_ns':t+service*1_000_000}))
        result=summarize(dict(request_records=rows,phase_start_ns=0,deadline_ns=300_000_000),
            dict(aliases={'a':{'family':'text'}}))
        thirds=result['groups'][0]['arrival_thirds']
        self.assertEqual([x['observed_service']['mean_ms'] for x in thirds],[40.,70.,10.])
        self.assertEqual([x['full_cohort_mix_standardized_service_mean_ms'] for x in thirds],[43.75,43.75,None])
        self.assertEqual(thirds[2]['missing_reference_inputs'],['slow'])
        self.assertEqual(result['request_count'],8)


if __name__=='__main__':unittest.main()
