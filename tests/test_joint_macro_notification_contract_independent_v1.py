"""Independent CPU boundaries; no fitted provider or native clock inference."""
import copy
import unittest
from tests import test_composable_response_macros_v1 as author
from tools.predictor_validation_20261027 import response_notification_posterior_reference_v1 as posterior
from tools.predictor_validation_20261027.composable_response_macros_v1 import request_macros, visible_macro_prefix
from tools.predictor_validation_20261027.stage_response_features_v1 import ordered_events


class JointMacroNotificationContractTests(unittest.TestCase):
    def row(self):return author.MacroTests().row()

    def test_host_END_inside_macro_does_not_make_response_complete(self):
        row=self.row();result=request_macros(row,ordered_events([row]),'fixture','natural')
        macro=next(m for m in result['macros'] if m['macro_id']=='prefill_0000')
        host_end=next(e['clocks']['occurrence_ns'] for e in macro['internal_training_milestones'] if e['event']=='end' and e['phase_id']=='prefill_0000')
        cutoff=host_end+1
        self.assertLess(cutoff,macro['right_ns'])
        self.assertEqual(macro['response_wall_ns'],30)
        self.assertFalse(macro['additional_stream_elapsed_or_sync_wait_may_be_added'])
        self.assertFalse(macro['true_SM_service_identified'])
        self.assertGreater(macro['right_ns']-cutoff,0)
        # Resetting macro age at internal host END would change remaining wall.
        self.assertNotEqual(macro['right_ns']-cutoff,macro['response_wall_ns']-(cutoff-host_end))

    def test_late_query_notification_survives_physical_macro_boundary(self):
        row=self.row()
        query=dict(request_id='r',event='observation',kind='device_event_query',phase_id='prefill_0000',
            segment_id='prefill_0000',perf_ns=145,received_time_ns=163,
            collector_dequeued_time_ns=164,collector_processing_time_ns=165,end_event_complete=True)
        row['execution_events'].insert(3,query)
        for e in row['execution_events'][4:]:
            e.update(received_time_ns=e['perf_ns']+16,collector_dequeued_time_ns=e['perf_ns']+17,
                collector_processing_time_ns=e['perf_ns']+18)
        result=request_macros(row,ordered_events([row]),'fixture','natural')
        first=next(m for m in result['macros'] if m['macro_id']=='prefill_0000')
        next_macro=next(m for m in result['macros'] if m['macro_id']=='decode_0001')
        q=next(m for m in first['internal_training_milestones'] if m['kind']=='device_event_query')
        self.assertLess(q['clocks']['occurrence_ns'],first['right_ns'])
        self.assertGreater(q['clocks']['processed_ns'],next_macro['left_ns'])
        self.assertFalse(q['runtime_outcome_known_at_start'])
        self.assertEqual(sum(m['response_wall_ns'] for m in result['macros']),110)

    def test_same_no_terminal_notice_retains_macro_complete_and_ongoing(self):
        delay=lambda ns:dict(reader_delay_ns=ns,processing_extra_ns=0)
        worlds=[dict(id='macro_done_notice_pending',mass=.5,begin_ns=120,response_ns=30,
                     shape_template_id='qualified_prior',BEGIN=delay(1),END=delay(20)),
                dict(id='macro_ongoing',mass=.5,begin_ns=120,response_ns=60,
                     shape_template_id='qualified_prior',BEGIN=delay(1),END=delay(10))]
        # Here END denotes the selected macro terminal (next BEGIN/native END),
        # never the internal forward host END.
        facts=[dict(edge='BEGIN',occurrence_ns=120,available_ns=121,shape_template_id='qualified_prior')]
        result=posterior.condition(worlds,facts,cutoff_ns=160,visibility_endpoint='parent_processing',
            registry_complete_through_ns=160)
        self.assertEqual(result['completed_response_mass'],.5)
        self.assertEqual(result['mean_latent_remaining_host_response_ns'],10)
        self.assertIsNone(result['all_request_GPU_work_complete'])

    def test_stream_query_cannot_be_substituted_for_macro_END(self):
        with self.assertRaisesRegex(ValueError,'already-visible'):
            posterior.condition([], [dict(edge='QUERY_TRUE',occurrence_ns=10,available_ns=20,shape_template_id=None)],
                cutoff_ns=30,visibility_endpoint='parent_processing',registry_complete_through_ns=30)

    def test_G2_last_forward_query_true_keeps_native_TTS_tail_visible(self):
        row=self.row();row['model']='generalist#0'
        query=dict(request_id='r',event='observation',kind='device_event_query',phase_id='decode_0001',
            segment_id='decode_0001',end_event_complete=True,**author.clock(165))
        row['execution_events'].insert(5,query)
        events=ordered_events([row])
        for cutoff in (169,179,183):
            state=visible_macro_prefix(events,cutoff,'other','decode')
            self.assertEqual(state['peer_visible_macro_owner_count'],1)
            self.assertEqual(state['peer_macros'][0]['phase_id'],'decode_0001')
            self.assertFalse(state['macro_absence_or_presence_certifies_physical_completion'])
        # Removal is by processed native END184, not physical native END180.
        self.assertEqual(visible_macro_prefix(events,184,'other','decode')['peer_visible_macro_owner_count'],0)

    def test_late_old_query_does_not_clear_or_revert_new_macro(self):
        row=self.row()
        query=dict(request_id='r',event='observation',kind='device_event_query',phase_id='prefill_0000',
            segment_id='prefill_0000',end_event_complete=True,**author.clock(155))
        # Polling an old recorded event after next BEGIN is FIFO-consistent.
        # Do not invent reordered delivery within a single producer stream.
        row['execution_events'].insert(4,query)
        state=visible_macro_prefix(ordered_events([row]),159,'other','decode')
        self.assertEqual(state['peer_same_visible_macro_phase_owner_count'],1)
        self.assertEqual(state['peer_macros'][0]['phase_id'],'decode_0001')

    def test_unavailable_event_body_never_accessed_or_used(self):
        class FutureBodyPoison(dict):
            def get(self,key,*args):raise AssertionError('Future body accessed')
            def __getitem__(self,key):
                if key!='collector_processing_time_ns':raise AssertionError('Future body accessed')
                return super().__getitem__(key)
        events=ordered_events([self.row()]);cutoff=149
        expected=visible_macro_prefix(events,cutoff,'other','prefill')
        changed=copy.deepcopy(events)
        for i,event in enumerate(changed):
            if event['collector_processing_time_ns']>cutoff:
                changed[i]=FutureBodyPoison(collector_processing_time_ns=event['collector_processing_time_ns'])
        self.assertEqual(visible_macro_prefix(changed,cutoff,'other','prefill'),expected)

    def test_internal_milestone_raw_identity_is_lossless(self):
        row=self.row();result=request_macros(row,ordered_events([row]),'capture_identity','natural')
        milestones=[e for macro in result['macros'] for e in macro['internal_training_milestones']]
        self.assertEqual(len(milestones),len(row['execution_events']))
        for event in milestones:
            ordinal=event['raw_event_ordinal']
            self.assertEqual(event['raw_event'],row['execution_events'][ordinal])
            self.assertEqual(event['raw_event_pointer'],dict(capture_sha256='capture_identity',
                request_id='r',execution_event_ordinal=ordinal))


if __name__=='__main__':unittest.main()
