import unittest
from tools.predictor_validation_20261027.training_idle_diagnostics import summarize


class IdleHistoryTests(unittest.TestCase):
    def row(self,rid,start,end,phase='measurement',scheduled=0):
        return dict(request_id=rid,model='a',task='text',pool_id='same',phase=phase,status='completed',
            observed_worker_start_time_ns=start*1_000_000,scheduled_time_ns=scheduled*1_000_000,
            worker_response={'end_perf_ns':end*1_000_000})

    def test_real_priming_predecessor_and_no_sparse_cell_fallback(self):
        raw=dict(phase_start_ns=100_000_000,deadline_ns=400_000_000,
            priming_request_records=[self.row('prime',10,20,'priming')],warmup_request_records=[],
            request_records=[self.row('a',100,110,scheduled=100),self.row('b',150,170,scheduled=150)])
        value=summarize(raw,{})
        self.assertEqual([r['idle_gap_ms'] for r in value['records']],[80.,40.])
        self.assertEqual(value['records'][0]['prior_phase'],'priming')
        self.assertEqual(value['missing_prior_count'],0)
        group=value['groups'][0]
        self.assertIsNone(group['correlation_log1p_idle_service'])
        self.assertEqual(group['correlation_unknown_reason'],'BELOW_FIXED_MIN_CELL_N')
        self.assertFalse(group['each_arrival_third_has_minimum_n'])
        self.assertEqual(group['service']['mean_ms'],15.)

    def test_foreign_overlapping_worker_history_rejected(self):
        raw=dict(phase_start_ns=100_000_000,deadline_ns=400_000_000,
            priming_request_records=[self.row('prime',10,110,'priming')],
            request_records=[self.row('a',100,120,scheduled=100)])
        with self.assertRaisesRegex(ValueError,'cannot overlap'):summarize(raw,{})


if __name__=='__main__':unittest.main()
