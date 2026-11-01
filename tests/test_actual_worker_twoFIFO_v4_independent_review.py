"""Independent actual worker release, queue and public draw conservation."""
from dataclasses import replace
from fractions import Fraction as F
import copy
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tests import test_actual_worker_twoFIFO_reference_v4 as fixtures
from tools.predictor_validation_20261027 import actual_worker_twoFIFO_reference_v4 as v4
from tools.predictor_validation_20261027.worker_stdin_fifo_contract_v1 import TransportDraw


class ActualTwoFIFOIndependentReview(unittest.TestCase):
    def inputs(self,**kwargs):return fixtures.ActualWorkerTwoFIFOTests().inputs(**kwargs)

    def test_ready_processing_does_not_gate_second_START_or_clear_new_request(self):
        args=self.inputs();world=fixtures.decode(v4.simulate(**args))
        self.assertEqual(world['starts']['r1'],12)
        self.assertLess(world['starts']['r1'],world['RESULT_processed']['r0'])
        self.assertEqual(world['maximum_parallel_handlers'],1)
        self.assertFalse(world['main_queue_provider_eligible']);self.assertFalse(world['native_assimilation_eligible'])
        self.assertFalse(world['busy_transport_transfer_qualified'])

    def test_result_reader_completion_at_cutoff_not_lost_to_late_processing(self):
        args=self.inputs(cutoff=19);world=fixtures.decode(v4.simulate(**args))
        self.assertEqual(world['RESULT_receipts'],{'r0':19})
        self.assertEqual(world['unfinished_request_ids'],['r1'])
        self.assertFalse(world['RESULT_processed'])
        self.assertEqual(world['active_response_states']['owner']['request_id'],'r1')

    def test_initial_worker_busy_release_and_sender_busy_are_independent(self):
        args=self.inputs();args['initial_worker_read_release_ns']={'owner':30}
        world=fixtures.decode(v4.simulate(**args))
        self.assertEqual(world['starts']['r0'],31)
        self.assertEqual(world['parent_dispatches'],{'r0':0,'r1':1})
        args=self.inputs();args['initial_sender_available_ns']={'owner':30}
        world=fixtures.decode(v4.simulate(**args))
        self.assertEqual(world['parent_dispatches']['r0'],30)
        self.assertEqual(world['starts']['r0'],31)

    def test_initial_protocol_reader_and_collector_backlog_never_release_worker(self):
        args=self.inputs();args['initial_protocol_state'].update(reader_free={'owner':50},collector_free=70,last_collector_arrival=0)
        world=fixtures.decode(v4.simulate(**args))
        self.assertEqual(world['starts'],{'r0':1,'r1':12})
        self.assertEqual(world['RESULT_receipts']['r0'],60)
        self.assertFalse(world['RESULT_processed'])
        first=next(n for n in world['pending_notifications'] if n['body']['action']=='result')
        self.assertEqual(first['available_ns'],170)

    def test_cutoff_before_START_keeps_reserved_owner_and_future_draws(self):
        args=self.inputs(cutoff=5);args['initial_worker_read_release_ns']={'owner':20}
        before=copy.deepcopy(args);world=fixtures.decode(v4.simulate(**args))
        self.assertFalse(world['starts']);self.assertFalse(world['response_exposure'])
        self.assertEqual(world['pending_worker_starts'][0]['at_ns'],21)
        self.assertEqual(world['per_owner_FIFO']['owner'],['r1'])
        self.assertEqual(len(world['consumed_public_key_joint_draws']),1)
        self.assertEqual(args,before)

    def test_nonzero_world_start_is_translation_equivalent(self):
        args=self.inputs();base=fixtures.decode(v4.simulate(**args));origin=1000
        for row in args['requests']:row['arrival_ns']+=origin
        for field in ('window_start_ns','window_end_ns','drain_end_ns','cutoff_ns'):args[field]+=origin
        args['initial_sender_available_ns']={'owner':origin};args['initial_worker_read_release_ns']={'owner':origin}
        args['initial_protocol_state'].update(collector_free=origin,last_collector_arrival=origin)
        shifted=fixtures.decode(v4.simulate(**args))
        self.assertEqual(shifted['task_E2E_latency_arrays_ns'],base['task_E2E_latency_arrays_ns'])
        for rid,at in base['starts'].items():self.assertEqual(shifted['starts'][rid],at+origin)

    def test_missing_or_extra_transport_draw_rejects_full_offered_cohort_at_cutoff(self):
        for action in ('missing','extra'):
            args=self.inputs(cutoff=0)
            if action=='missing':del args['transport_draws'][('owner','text','input',1)]
            else:args['transport_draws'][('owner','text','input',2)]=TransportDraw(0,0,0,0,0)
            with self.subTest(action=action),self.assertRaises((ValueError,UnknownEvidence)):v4.simulate(**args)

    def test_missing_message_slack_cannot_be_zero(self):
        args=self.inputs();first=args['plans'][('owner','text','input')][0]
        del args['protocol_slacks'][first]['result0']
        with self.assertRaisesRegex(UnknownEvidence,'SLACK'):v4.simulate(**args)

    def test_request_ids_are_not_transport_or_template_draw_lookup_keys(self):
        args=self.inputs();before=fixtures.decode(v4.simulate(**args))
        for i,row in enumerate(args['requests']):row['request_id']='renamed:'+str(1-i)
        world=fixtures.decode(v4.simulate(**args))
        self.assertEqual(list(world['starts'].values()),list(before['starts'].values()))
        self.assertEqual(world['task_E2E_latency_arrays_ns'],before['task_E2E_latency_arrays_ns'])


if __name__=='__main__':unittest.main()
