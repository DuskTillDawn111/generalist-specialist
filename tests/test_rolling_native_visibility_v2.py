"""Reader delay, host semantic boundaries and role qualification remain explicit."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import rolling_native_visibility_v2 as v
from tests import test_rolling_visible_queue_v1 as f
from gsdeploy.forecasting.observed_state import feedback_event


class NativeVisibilityTests(unittest.TestCase):
    def test_delayed_start_is_neither_waiting_proof_nor_current_cursor(self):
        record=dict(request_id='a',model='s',execution_events=[dict(request_id='a',perf_ns=0,received_time_ns=10*f.MS)])
        early,tick=v.snapshot(f.session(),[record],5*f.MS)
        row=tick['request_states']['a']
        self.assertEqual(row['status'],'ARRIVED_START_VISIBILITY_UNKNOWN')
        self.assertFalse(row['no_START_is_waiting_proof'])
        self.assertEqual(tick['visible_feedback_events'],0)
        later,tick=v.snapshot(early,[record],15*f.MS)
        self.assertEqual(tick['request_states']['a']['status'],'BUSY_CURRENT_DEVICE_PROGRESS_UNKNOWN')
        self.assertIsNone(tick['request_states']['a']['current_device_fraction'])
        self.assertEqual(tick['max_visible_reader_delay_ns'],10*f.MS)
        self.assertEqual(tick['positive_delay_visible_events'],1)
        self.assertFalse(tick['numeric_main_continuation_eligible'])
        self.assertEqual(early['observer']['last_feedback_sequence'],0)

    def test_visible_forward_end_is_not_device_completion_and_future_body_untouched(self):
        class Forbidden(dict):
            def get(self,*args):raise AssertionError('future response consumed')
        record=dict(request_id='a',model='s',execution_events=[
            dict(request_id='a',perf_ns=0,received_time_ns=0),
            dict(request_id='a',perf_ns=1*f.MS,received_time_ns=2*f.MS,event='start',segment_id='decode_0001',kind='gpu'),
            dict(request_id='a',perf_ns=3*f.MS,received_time_ns=4*f.MS,event='end',segment_id='decode_0001',kind='gpu')],
            response_reader_event_enqueue_time_ns=100*f.MS,worker_response=Forbidden())
        _,tick=v.snapshot(f.session(),[record],50*f.MS)
        row=tick['request_states']['a']
        self.assertEqual(row['visible_host_phase_end_ids'],['decode_0001'])
        self.assertFalse(row['host_phase_end_is_device_completion'])
        self.assertIsNone(row['exact_current_remaining_work'])
        self.assertEqual(tick['exact_current_progress_receipts'],0)

    def test_ready_visibility_is_required_and_entire_public_cohort_is_retained(self):
        record=dict(request_id='a',model='s',execution_events=[dict(request_id='a',perf_ns=0,received_time_ns=0)],
            response_reader_event_enqueue_time_ns=10*f.MS,worker_response=dict(end_perf_ns=9*f.MS),
            previous_handler_next_request_ready=dict(request_id='a',perf_ns=11*f.MS,received_time_ns=20*f.MS,
                clock='host_perf_counter_ns',boundary='previous_handler_next_request_ready',
                scope='after_response_flush_before_next_read'))
        session,tick=v.snapshot(f.session(),[record],15*f.MS)
        self.assertEqual(tick['request_states']['a']['status'],'HANDOFF_RESIDUAL_UNKNOWN')
        _,tick=v.snapshot(session,[record],25*f.MS)
        self.assertEqual(tick['request_states']['a']['status'],'COMPLETED_READY_VISIBLE')
        self.assertEqual(tick['public_cohort_size'],3)
        self.assertEqual(tick['status_counts']['FUTURE_PUBLIC_ARRIVAL'],2)

    def test_validation_role_refused_before_raw_open(self):
        plan={'path':'plan','sha256':'p'};context={'path':'context','sha256':'c'};raw={'path':'raw-forbidden','sha256':'x'}
        seen=[]
        def read(ref):
            seen.append(ref)
            if ref==raw:raise AssertionError('validation raw opened')
            return dict(split='validation',training_only=False)
        with patch.object(v,'read',side_effect=read),self.assertRaisesRegex(ValueError,'before raw read'):
            v.qualify_capture(plan,context,raw)
        self.assertNotIn(raw,seen)


if __name__=='__main__':unittest.main()
