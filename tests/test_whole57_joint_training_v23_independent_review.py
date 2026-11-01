"""Independent host-clock/whole-cohort fixtures; no model or profiler starts."""
import unittest
from tools.predictor_validation_20261027 import whole57_joint_training_v23 as collect
from tools.predictor_validation_20261027 import whole57_joint_review_v23 as review


class Whole57IndependentTests(unittest.TestCase):
    def fixture(self,origin=1_000_000_000_000):
        inputs=[dict(task=t,pool_id=t+str(i)) for t in ('text','ocr') for i in (0,1)]
        trace=collect.make_trace(inputs,'whole57-training-independent')
        routes,_=collect.routes_for(trace,'balanced')
        plan=dict(trace=trace,routes=routes)
        owners={owner:dict(pid=100+i) for i,owner in enumerate(collect.RESIDENTS)}
        rows=[]
        for declared in trace:
            rid=declared['request_id'];owner=routes[rid];pid=owners[owner]['pid']
            scheduled=origin+declared['offset_ns'];begin=scheduled+5;end=begin+20
            def event(stamp,**kw):
                return dict(request_id=rid,pid=pid,perf_ns=stamp,received_time_ns=stamp+1,
                    collector_dequeued_time_ns=stamp+2,collector_processing_time_ns=stamp+3,**kw)
            rows.append(dict(request_id=rid,task=declared['task'],pool_id=declared['pool_id'],model=owner,
                split='training',validation_view='TRAINING_ONLY_NOT_VALIDATION',status='completed',
                scheduled_time_ns=scheduled,enqueued_time_ns=scheduled+1,actual_submit_time_ns=scheduled+2,
                submit_write_completed_time_ns=begin+100,
                completion_time_ns=end+2,response_reader_event_enqueue_time_ns=end+2,
                response_collector_dequeued_time_ns=end+3,response_collector_processing_time_ns=end+4,
                execution_events=[event(begin),event(begin+5,event='start',kind='gpu',segment_id='native'),
                    event(begin+10,event='end',kind='gpu',segment_id='native')],
                previous_handler_next_request_ready=event(end+1,clock='host_perf_counter_ns',boundary='previous_handler_next_request_ready'),
                worker_response=dict(request_id=rid,task_type=declared['task'],success=True,start_perf_ns=begin,
                    end_perf_ns=end,service_ms=.01,worker_latency_ms=.01,
                    semantic_forward_observations=dict(records=[dict(start_ns=begin+5,end_ns=begin+10)],instrumentation_errors=[]))))
        raw=dict(phase_start_ns=origin,planned_public_cohort=trace,request_records=rows,priming_request_records=[],ready=owners,
            deadline_ns=origin+30_000_000_000,arrival_loop_end_ns=origin+30_000_000_001,observation_end_ns=origin+31_000_000_000)
        return plan,raw

    def test_original_clock_offsets_and_all57_are_preserved(self):
        plan,raw=self.fixture();issues,rows,anchors=review.host_lifecycle_rows(plan,raw)
        self.assertEqual(issues,[]);self.assertEqual(len(rows),57);self.assertEqual(len(anchors),57)
        for row,source in zip(rows,raw['request_records']):
            self.assertEqual(row['offsets_ns']['dispatch_write_completed_ns'],source['submit_write_completed_time_ns']-raw['phase_start_ns'])
            self.assertEqual(row['offsets_ns']['parent_receipt_ns'],source['completion_time_ns']-raw['phase_start_ns'])
            self.assertFalse(row['host_queue_wait_removed'])
            self.assertFalse(row['first_semantic_is_first_CUDA_release'])
        shifted_plan,shifted_raw=self.fixture(origin=2_000_000_000_321)
        self.assertEqual(rows,review.host_lifecycle_rows(shifted_plan,shifted_raw)[1])

    def test_dropped_or_reidentified_request_is_refused(self):
        plan,raw=self.fixture();raw['request_records'].pop()
        self.assertIn('FULL57_EXECUTED_COHORT_AND_ORIGIN_REQUIRED',review.host_lifecycle_rows(plan,raw)[0])
        plan,raw=self.fixture();raw['request_records'][-1]['pool_id']='other'
        self.assertTrue(any('DECLARED_REQUEST_IDENTITY' in x for x in review.host_lifecycle_rows(plan,raw)[0]))

    def test_missing_write_and_shortened_window_fail_closed(self):
        plan,raw=self.fixture();del raw['request_records'][0]['submit_write_completed_time_ns']
        self.assertTrue(any('MISSING_ORIGINAL_HOST_BOUNDARY' in x for x in review.host_lifecycle_rows(plan,raw)[0]))
        plan,raw=self.fixture();raw['deadline_ns']-=1
        self.assertIn('EXACT_ORIGINAL_WINDOW_OFFER_AND_TAIL_DRAIN_BOUNDARIES_REQUIRED',review.host_lifecycle_rows(plan,raw)[0])

    def test_worker_next_ready_is_not_silently_replaced_with_handler_end(self):
        plan,raw=self.fixture();raw['request_records'][0]['previous_handler_next_request_ready']['perf_ns']-=5
        self.assertTrue(any('WORKER_NEXT_READY_CHAIN_INVALID' in x for x in review.host_lifecycle_rows(plan,raw)[0]))


if __name__=='__main__':unittest.main()
