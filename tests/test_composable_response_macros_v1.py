import unittest

from tests.test_stage_host_partition_v1 import fixture, clock
from tools.predictor_validation_20261027.composable_response_macros_v1 import request_macros,visible_macro_prefix
from tools.predictor_validation_20261027.stage_response_features_v1 import ordered_events


class MacroTests(unittest.TestCase):
    def row(self):
        row=fixture()
        for e in row['execution_events']:
            if e['segment_id']=='decode_0001':e['semantic_phase']='decode'
        return row

    def extract(self,row):
        return request_macros(row,ordered_events([row]),'capture','natural')

    def test_forward_macro_includes_carry_in_wait_exactly_once(self):
        row=self.row();r=self.extract(row)
        self.assertEqual([x['response_wall_ns'] for x in r['macros']],[20,30,30,30])
        self.assertEqual(sum(x['response_wall_ns'] for x in r['macros']),110)
        prefill=r['macros'][1]
        # Its host call ends140; the macro continues to the next BEGIN150.
        self.assertEqual(prefill['right_ns'],150)
        self.assertFalse(prefill['additional_stream_elapsed_or_sync_wait_may_be_added'])
        self.assertTrue(prefill['includes_previous_forward_device_tail_and_selection'])

    def test_query_notice_can_arrive_after_macro_physical_end(self):
        row=self.row()
        query=dict(request_id='r',event='observation',kind='device_event_query',segment_id='prefill_0000',
            phase_id='prefill_0000',end_event_complete=True,**clock(145))
        query.update(received_time_ns=163,collector_dequeued_time_ns=164,collector_processing_time_ns=165)
        row['execution_events'].insert(3,query)
        for e in row['execution_events'][4:]:
            e.update(received_time_ns=e['perf_ns']+16,
                collector_dequeued_time_ns=e['perf_ns']+17,collector_processing_time_ns=e['perf_ns']+18)
        r=self.extract(row);macro=r['macros'][1]
        q=next(x for x in macro['internal_training_milestones'] if x['kind']=='device_event_query')
        self.assertGreater(q['clocks']['processed_ns'],macro['right_ns'])
        self.assertFalse(q['runtime_outcome_known_at_start'])
        self.assertEqual(sum(x['duration_ns'] for x in macro['causal_covariate_path']),30)

    def test_coarse_ocr_remains_explicit_and_clock_complete(self):
        row=self.row();row['model']='ocr#0';row['task']='ocr'
        row['execution_events']=[dict(request_id='r',event=edge,kind='gpu',segment_id='native_ocr',**clock(at))
            for edge,at in [('start',110),('end',180)]]
        r=self.extract(row)
        self.assertEqual([x['response_wall_ns'] for x in r['macros']],[10,70,30])
        self.assertEqual(r['macros'][1]['kind'],'ocr_macro')
        self.assertFalse(r['macros'][1]['true_SM_service_identified'])

    def test_missing_parent_and_mixed_execution_refused(self):
        row=self.row();row['execution_events']=[e for e in row['execution_events'] if not(e['segment_id']=='native_inference' and e['event']=='end')]
        with self.assertRaises(ValueError):self.extract(row)

    def test_successful_query_preserves_selection_tail_until_next_phase_or_native_END(self):
        row=self.row()
        row['execution_events'].insert(3,dict(request_id='r',event='observation',kind='device_event_query',
            phase_id='prefill_0000',segment_id='prefill_0000',end_event_complete=True,**clock(145)))
        events=ordered_events([row])
        state=visible_macro_prefix(events,149,'other','prefill')
        self.assertEqual(state['peer_visible_macro_owner_count'],1)
        self.assertEqual(state['peer_macros'][0]['phase'],'prefill')
        state=visible_macro_prefix(events,154,'other','decode')
        self.assertEqual(state['peer_same_visible_macro_phase_owner_count'],1)
        self.assertEqual(visible_macro_prefix(events,184,'other','decode')['peer_visible_macro_owner_count'],0)
        self.assertFalse(state['visible_macro_is_actual_SM_activity'])

    def test_future_macro_notification_does_not_change_earlier_prefix(self):
        row=self.row();events=ordered_events([row])
        before=visible_macro_prefix(events,149,'other','prefill')
        for e in events:
            if e['collector_processing_time_ns']>149:e['semantic_phase']='future_mutation'
        self.assertEqual(visible_macro_prefix(events,149,'other','prefill'),before)
        row=self.row();row['execution_events'].append(dict(request_id='r',event='start',kind='gpu',segment_id='native_ocr',**clock(195)))
        with self.assertRaises(ValueError):self.extract(row)


if __name__=='__main__':unittest.main()
