"""Independent station recurrence and exact protocol-mark conservation."""
import copy
from fractions import Fraction as F
import unittest

from tests import test_protocol_timing_slack_v3 as fixtures
from tools.predictor_validation_20261027 import protocol_timing_slack_v3 as station
from tools.predictor_validation_20261027 import joint_request_protocol_transport_candidate_v3 as candidate
from tools.predictor_validation_20261027 import joint_request_response_transport_candidate_v2 as work
from tools.predictor_validation_20261027.macro_protocol_timing_reference_v3 import simulate


class ProtocolStationIndependentReview(unittest.TestCase):
    def test_missing_boolean_negative_slack_and_nonserial_source_reject(self):
        for values in ({},dict(reader_slack_ns=True,collector_dequeue_slack_ns=0,collector_processing_slack_ns=0),
                dict(reader_slack_ns=0,collector_dequeue_slack_ns=0,collector_processing_slack_ns=-1)):
            with self.subTest(values=values),self.assertRaises(ValueError):station.checked(values)
        rows=[dict(identity='a',producer='a',occurrence_ns=0,reader_ns=1,dequeue_ns=1,processing_ns=20),
            dict(identity='b',producer='b',occurrence_ns=0,reader_ns=2,dequeue_ns=2,processing_ns=3)]
        with self.assertRaises(ValueError):station.derive(rows)

    def test_exact_station_queue_recurrence_can_add_delay_without_overwriting_slack(self):
        t=station.TimingStations(F(1,2));slack=dict(reader_slack_ns=F(3,2),collector_dequeue_slack_ns=2,collector_processing_slack_ns=3)
        before=copy.deepcopy(slack)
        first=t.issue(F(1,2),'a',slack);second=t.issue(1,'a',slack)
        self.assertEqual((first,second),(2,F(7,2)))
        self.assertEqual(t.accept_reader(first,slack),(4,7))
        self.assertEqual(t.accept_reader(second,slack),(9,12))
        self.assertEqual(slack,before)

    def test_source_messages_keeps_START_and_all_event_kinds_PLUS_RESULT_READY(self):
        raw=dict(request_records=[dict(model='a',request_id='original',execution_events=[
            dict(perf_ns=1,received_time_ns=2,collector_dequeued_time_ns=3,collector_processing_time_ns=4),
            dict(kind='device_event_query',perf_ns=5,received_time_ns=6,collector_dequeued_time_ns=7,collector_processing_time_ns=8)],
            worker_response=dict(end_perf_ns=9),response_reader_event_enqueue_time_ns=10,
            response_collector_dequeued_time_ns=11,response_collector_processing_time_ns=12,
            previous_handler_next_request_ready=dict(perf_ns=13,received_time_ns=14,collector_dequeued_time_ns=15,collector_processing_time_ns=16))])
        messages=candidate.source_messages(raw)
        self.assertEqual(len(messages),4)
        self.assertEqual([m['identity'][-1] for m in messages],['event:0','event:1','RESULT','worker_READY'])
        self.assertEqual(len(station.derive(messages)),4)

    def test_new_protocol_compiler_does_not_mutate_joint_vector_or_impute_missing(self):
        vector=fixtures.timing_vector();before=copy.deepcopy(vector)
        plan,slacks=candidate.to_reference(vector,public_dispatch_ordinal=2)
        self.assertEqual(vector,before)
        self.assertEqual(set(slacks),{m.identity for macro in plan.macros for m in macro.marks}|{'worker_READY'})
        del vector['ready_protocol_slacks']
        with self.assertRaises((ValueError,KeyError)):candidate.to_reference(vector,public_dispatch_ordinal=2)

    def test_reader_completed_RESULT_before_collector_processing_survives_cutoff(self):
        vector=fixtures.timing_vector()
        for node in vector['nodes']:
            for mark in node['marks']:
                if mark['action']=='result':mark['protocol_slacks']['collector_processing_slack_ns']=40
        plan,timings=candidate.to_reference(vector,public_dispatch_ordinal=0)
        request=dict(request_id='new',owner='g2',task='text',pool_id='exact-content',arrival_ns=0)
        kwargs=dict(requests=[request],plans={('g2','text','exact-content'):plan},rates=work.rate_table([0,0]),
            window_start_ns=0,window_end_ns=1,drain_end_ns=100,protocol_slacks={plan:timings})
        full=simulate(**kwargs);at=full['RESULT_receipts']['new'];cutoff=F(at['numerator'],at['denominator'])
        result=simulate(**kwargs,cutoff_ns=cutoff)
        self.assertEqual(result['unfinished_request_ids'],[])
        self.assertFalse(result['RESULT_processed'])
        self.assertFalse(result['main_queue_provider_eligible']);self.assertFalse(result['dispatch_contract_matches_original_sender'])


if __name__=='__main__':unittest.main()
