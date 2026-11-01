import unittest
from tools.predictor_validation_20261027.long_flow_diagnostics_v14 import summarize


class LongFlowSummaryTests(unittest.TestCase):
    def test_buckets_use_public_arrival_not_finish_and_keep_drain_tail(self):
        rows=[]
        for i,scheduled in enumerate([0,40,80]):
            rows.append({'status':'completed','model':'a','task':'text','pool_id':'x',
                'scheduled_time_ns':scheduled*1000000,'observed_worker_start_time_ns':(scheduled+10)*1000000,
                'completion_time_ns':(scheduled+50)*1000000,
                'worker_response':{'end_perf_ns':(scheduled+45)*1000000}})
        value=summarize({'request_records':rows,'phase_start_ns':0,'deadline_ns':100000000},
            {'aliases':{'a':{'family':'text'}}})
        self.assertEqual([r['arrival_third'] for r in value['by_arrival_third']],[0,1,2])
        group=value['by_input'][0]
        self.assertEqual(group['service_ms']['mean'],35.)
        self.assertEqual(group['e2e_ms']['mean'],50.)
        self.assertEqual(group['wait_to_worker_ms']['mean'],10.)
        self.assertEqual(group['parent_tail_ms']['mean'],5.)
        self.assertEqual(group['e2e_ms']['n'],3)

    def test_incomplete_request_not_silently_trimmed(self):
        with self.assertRaises(ValueError):
            summarize({'request_records':[{'status':'timeout'}]}, {'aliases':{}})


if __name__=='__main__':unittest.main()
