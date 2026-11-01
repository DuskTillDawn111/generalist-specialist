import copy
import unittest

from tools.predictor_validation_20261027.stage_host_partition_v1 import partition_request


def clock(at):
    return dict(perf_ns=at, received_time_ns=at+2, collector_dequeued_time_ns=at+3,
        collector_processing_time_ns=at+4)


def fixture():
    events = []
    for edge, phase, at in [('start','native_inference',110), ('start','prefill_0000',120),
            ('end','prefill_0000',140), ('start','decode_0001',150), ('end','decode_0001',160),
            ('end','native_inference',180), ('start','post_sync',181), ('end','post_sync',190)]:
        events.append(dict(request_id='r',event=edge,kind='sync' if phase=='post_sync' else 'gpu',
            segment_id=phase, semantic_phase='prefill' if phase=='prefill_0000' else None, **clock(at)))
    return dict(request_id='r', model='text#0', task='text', pool_id='input', profile_job_id='job',
        status='completed', worker_response=dict(success=True, start_perf_ns=100, end_perf_ns=200),
        observed_worker_start_time_ns=100, completion_time_ns=215,
        response_reader_event_enqueue_time_ns=215, response_collector_dequeued_time_ns=216,
        response_collector_processing_time_ns=217, execution_events=events,
        previous_handler_next_request_ready=dict(request_id='r',
            boundary='previous_handler_next_request_ready',scope='after_response_flush_before_next_read',
            **clock(210)))


class PartitionTests(unittest.TestCase):
    def test_parent_is_partitioned_not_added_to_children(self):
        row=fixture();before=copy.deepcopy(row);result=partition_request(row)
        self.assertEqual(row,before)
        self.assertEqual(sum(x['duration_ns'] for x in result['pieces']),110)
        self.assertEqual(result['worker_handler_to_response_end_ns'],100)
        self.assertEqual(result['response_end_to_ready_ns'],10)
        gaps=[x for x in result['pieces'] if x['label']=='forward_external_gap_in_native_inference']
        self.assertEqual(sum(x['duration_ns'] for x in gaps),40)
        self.assertFalse(result['independent_sync_and_device_wait_addition_permitted'])

    def test_response_and_ready_reader_order_is_not_worker_order(self):
        row=fixture();row['previous_handler_next_request_ready'].update(clock(230))
        result=partition_request(row)
        self.assertEqual(result['worker_handler_to_ready_ns'],130)
        self.assertEqual(result['reader_response_end_to_receipt_ns'],15)
        self.assertEqual(result['reader_receipt_to_READY_receipt_ns'],17)

    def test_queries_never_add_extra_time(self):
        row=fixture()
        query=dict(request_id='r', event='observation', kind='device_event_query',**clock(145))
        row['execution_events'].insert(3,query)
        self.assertEqual(partition_request(row)['worker_handler_to_ready_ns'],110)

    def test_crossing_nested_phases_and_unclosed_stages_refused(self):
        row=fixture();row['execution_events'][2].update(clock(155))
        row['execution_events'].sort(key=lambda x:x['perf_ns'])
        with self.assertRaisesRegex(ValueError,'Crossing'):partition_request(row)
        row=fixture();row['execution_events'].pop()
        with self.assertRaisesRegex(ValueError,'Unclosed'):partition_request(row)

    def test_malformed_identity_and_future_clock_refused(self):
        for mode in ('request','ready','clock','availability','success'):
            row=fixture()
            if mode=='request':row['execution_events'][0]['request_id']='other'
            elif mode=='ready':row['previous_handler_next_request_ready']['request_id']='other'
            elif mode=='clock':row['worker_response']['end_perf_ns']=250
            elif mode=='availability':row['execution_events'][0]['received_time_ns']=1
            else:row['worker_response']['success']=False
            with self.subTest(mode=mode),self.assertRaises(ValueError):partition_request(row)


if __name__=='__main__':unittest.main()
