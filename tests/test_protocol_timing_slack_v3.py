from fractions import Fraction
import copy
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tests import test_joint_request_response_transport_candidate_v2 as fixtures
from tools.predictor_validation_20261027 import protocol_timing_slack_v3 as timing
from tools.predictor_validation_20261027 import joint_request_response_transport_candidate_v2 as work
from tools.predictor_validation_20261027 import joint_request_protocol_transport_candidate_v3 as candidate
from tools.predictor_validation_20261027.macro_protocol_timing_reference_v3 import simulate
from tools.predictor_validation_20261027.macro_response_queue_reference_v1 import simulate as fixed_simulate


def timing_vector():
    vector=fixtures.vector()
    for node in vector['nodes']:
        for m in node['marks']:
            m['protocol_slacks']=dict(reader_slack_ns=m['reader_delay_ns'],
                collector_dequeue_slack_ns=m['dequeue_extra_ns'],collector_processing_slack_ns=m['processing_extra_ns'])
    vector['ready_protocol_slacks']=dict(reader_slack_ns=1,collector_dequeue_slack_ns=0,collector_processing_slack_ns=0)
    return vector


class ProtocolTimingTests(unittest.TestCase):
    def run_world(self,vector,cutoff=None):
        plan,slacks=candidate.to_reference(vector,public_dispatch_ordinal=0)
        key=('g2','text','exact-content')
        request=dict(request_id='fresh',owner=key[0],task=key[1],pool_id=key[2],arrival_ns=0)
        return simulate([request],{key:plan},work.rate_table([0,0]),window_start_ns=0,window_end_ns=1,
            drain_end_ns=100,cutoff_ns=cutoff,producer_fifo=(plan.ready_producer_id,),protocol_slacks={plan:slacks})

    def test_source_clocks_reconstruct_with_cross_owner_reader_order(self):
        rows=[dict(identity='a',producer='a',occurrence_ns=0,reader_ns=10,dequeue_ns=12,processing_ns=15),
              dict(identity='b',producer='b',occurrence_ns=1,reader_ns=2,dequeue_ns=3,processing_ns=4)]
        derived=timing.derive(rows)
        self.assertEqual([r['identity'] for r in derived],['b','a'])
        self.assertEqual(derived[1]['protocol_slacks']['collector_dequeue_slack_ns'],2)

    def test_shared_parent_station_changes_other_owner_processing(self):
        s=timing.TimingStations();slack=dict(reader_slack_ns=0,collector_dequeue_slack_ns=0,collector_processing_slack_ns=10)
        self.assertEqual(s.accept_reader(s.issue(0,'a',slack),slack),(0,10))
        self.assertEqual(s.accept_reader(s.issue(1,'b',slack),slack),(10,20))
        self.assertEqual(s.collector_free,20)

    def test_reader_queue_does_not_silently_clamp_old_delay(self):
        s=timing.TimingStations();large=dict(reader_slack_ns=20,collector_dequeue_slack_ns=0,collector_processing_slack_ns=0)
        small=dict(reader_slack_ns=1,collector_dequeue_slack_ns=0,collector_processing_slack_ns=0)
        self.assertEqual(s.issue(1,'a',large),21)
        self.assertEqual(s.issue(2,'a',small),22)
        # Independent old delay gives3; the new law explicitly adds service-like
        # timing slack after the queued predecessor, a different timing model.
        self.assertNotEqual(22,2+small['reader_slack_ns'])

    def test_negative_slack_or_unobserved_missing_message_is_rejected(self):
        with self.assertRaises(ValueError):timing.checked(dict(reader_slack_ns=1,collector_dequeue_slack_ns=-1,
            collector_processing_slack_ns=2))
        with self.assertRaises(ValueError):timing.derive([dict(identity='bad',producer='a',
            occurrence_ns=5,reader_ns=4,dequeue_ns=6,processing_ns=7)])
        with self.assertRaises(ValueError):timing.checked(dict(reader_slack_ns=1))

    def test_new_protocol_law_composes_old_fixed_delay_FIFO_conflict(self):
        vector=timing_vector();h=vector['nodes'][1]['marks'][1]
        h['reader_delay_ns']=20;h['protocol_slacks']['reader_slack_ns']=20
        plan,slacks=candidate.to_reference(vector,public_dispatch_ordinal=0)
        key=('g2','text','exact-content');request=dict(request_id='fresh',owner='g2',task='text',pool_id='exact-content',arrival_ns=0)
        with self.assertRaises(UnknownEvidence):
            fixed_simulate([request],{key:plan},work.rate_table([0,0]),window_start_ns=0,window_end_ns=1,
                drain_end_ns=100,producer_fifo=(plan.ready_producer_id,))
        output=self.run_world(vector)
        self.assertEqual(output['unfinished_request_ids'],[])
        self.assertFalse(output['protocol_slacks_are_identified_CPU_service'])

    def test_cutoff_keeps_pending_reader_and_actual_completed_macro(self):
        vector=timing_vector();vector['nodes'][1]['marks'][0]['protocol_slacks']['reader_slack_ns']=30
        output=self.run_world(vector,cutoff=13)
        self.assertEqual(output['active_response_states']['g2']['macro_id'],'native_end_to_worker_ready')
        self.assertTrue(output['pending_protocol_reader_events'])
        self.assertEqual(output['visible_macro_evidence']['g2']['phase'],'entry')
        self.assertFalse(output['query_success_ends_macro'])

    def test_collector_receives_reader_order_not_actual_occurrence_order(self):
        s=timing.TimingStations();a=dict(reader_slack_ns=20,collector_dequeue_slack_ns=0,collector_processing_slack_ns=1)
        b=dict(reader_slack_ns=1,collector_dequeue_slack_ns=0,collector_processing_slack_ns=1)
        ar=s.issue(0,'a',a);br=s.issue(1,'b',b)
        self.assertEqual((ar,br),(20,2))
        self.assertEqual(s.accept_reader(br,b),(2,3))
        self.assertEqual(s.accept_reader(ar,a),(20,21))


if __name__=='__main__':unittest.main()
