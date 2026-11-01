import unittest
import copy
from fractions import Fraction
from unittest.mock import patch

from tests.test_twoFIFO_joint_request_law_v2 import source_fixture
from tools.predictor_validation_20261027 import twoFIFO_joint_request_law_v4 as m
from tools.predictor_validation_20261027 import twoFIFO_paired_transport_v3 as transport
from tools.predictor_validation_20261027 import actual_worker_twoFIFO_reference_v5_2 as engine
from tests.test_actual_worker_twoFIFO_reference_v5 import decode


class JointLawV4Tests(unittest.TestCase):
    def test_compiled_complete_vector_preserves_all_nodes_marks_and_protocol(self):
        plan,raw,requests,fit,vector=source_fixture();out=m.compile_draw_v4(vector,0)
        self.assertIs(type(out['transport_draw']),transport.PairedTransportConstraints)
        self.assertFalse(out['latent_line_or_continuation_point_selected'])
        self.assertEqual(len(out['plan'].macros),len(vector['nodes']))
        self.assertEqual(sum(len(n.marks) for n in out['plan'].macros),sum(len(n['marks']) for n in vector['nodes']))
        self.assertEqual(len(out['protocol_slacks']),sum(len(n['marks']) for n in vector['nodes'])+1)
        self.assertEqual(out['plan'].dispatch_to_worker_start_ns,0)
        self.assertFalse(out['changed_schedule_transport_transfer_qualified'])

    def test_compiled_real_joint_vector_enters_engine_with_actual_previous_ready(self):
        plan,raw,requests,fit,vector=source_fixture();out=m.compile_draw_v4(vector,0)
        row=raw['request_records'][0];prior=raw['priming_request_records'][-1]
        key=tuple(vector['public_key']);origin=row['scheduled_time_ns']
        kw=dict(requests=[dict(request_id='generated-request',owner=key[0],task=key[1],pool_id=key[2],arrival_ns=origin)],
            plans={key:out['plan']},rates={(key[0],n.phase,0,0):Fraction(1) for n in out['plan'].macros},
            protocol_slacks={out['plan']:out['protocol_slacks']},transport_draws={(*key,0):out['transport_draw']},
            window_start_ns=origin,window_end_ns=origin+1,drain_end_ns=origin+10_000_000_000,
            initial_sender_available_ns={key[0]:prior['submit_write_completed_time_ns']},
            initial_worker_read_release_ns={key[0]:prior['previous_handler_next_request_ready']['perf_ns']},
            initial_protocol_state=dict(reader_free={},collector_free=origin,last_collector_arrival=origin,pending_messages=[]))
        world=decode(engine.simulate(**kw))
        self.assertEqual(world['offered_requests'],1)
        self.assertEqual(world['starts']['generated-request'],row['observed_worker_start_time_ns'])
        self.assertEqual(len([x for x in world['trace'] if x['kind']=='latent_occurrence']),sum(len(n['marks']) for n in vector['nodes'])+1)

    def test_builder_still_enforces_complete_twelve_attempt_fit_gate(self):
        with self.assertRaises(ValueError):m.build_joint_law_v4([],[],[],dict(response_law_compilation_eligible=True))
        with patch.object(m.response,'load_complete',return_value=dict(status=m.response.CONTAMINATION_HOLD,documents=None)):
            with self.assertRaisesRegex(ValueError,m.response.CONTAMINATION_HOLD):m.run('oldcampaign',[],'missingfit','missingoutput')

    def test_identical_response_slots_keep_distinct_paired_protocol_slacks(self):
        *_,vector=source_fixture();second=copy.deepcopy(vector)
        for v in (vector,second):v['transport_constraints'].update(enqueue_after_arrival_ns=0,sender_turn_extra_ns=0,
            submit_to_write_record_ns=1,observed_submit_to_START_ns=2,previous_READY_minus_submit_ns=1)
        # Make two complete response shapes identical, with a changed RESULT
        # reader slack only. Both public slots must survive a plan-keyed map.
        for node in second['nodes']:
            for mark in node['marks']:
                if mark['action']=='result':mark['protocol_slacks']['reader_slack_ns']+=100
        one,two=m.compile_draw_v4(vector,0),m.compile_draw_v4(second,1)
        self.assertNotEqual(one['plan'],two['plan'])
        timing={one['plan']:one['protocol_slacks'],two['plan']:two['protocol_slacks']}
        self.assertEqual(len(timing),2)
        key=tuple(vector['public_key']);owner=key[0]
        kw=dict(requests=[dict(request_id='slot'+str(i),owner=owner,task=key[1],pool_id=key[2],arrival_ns=i) for i in range(2)],
            plans={key:(one['plan'],two['plan'])},rates={(owner,n.phase,0,0):Fraction(1) for n in one['plan'].macros},
            protocol_slacks=timing,transport_draws={(*key,0):one['transport_draw'],(*key,1):two['transport_draw']},
            window_start_ns=0,window_end_ns=2,drain_end_ns=10_000_000_000,
            initial_sender_available_ns={owner:0},initial_worker_read_release_ns={owner:2},
            initial_protocol_state=dict(reader_free={},collector_free=0,last_collector_arrival=0,pending_messages=[]))
        world=decode(engine.simulate(**kw))
        results=[r for r in world['trace'] if r['kind']=='latent_occurrence' and r['action']=='result']
        self.assertEqual(len(results),2)
        observed=[]
        for r in results:
            plan=one['plan'] if r['request_id']=='slot0' else two['plan']
            observed.append(timing[plan][r['identity']]['reader_slack_ns'])
        self.assertEqual(observed[1]-observed[0],100)
        self.assertEqual(world['offered_requests'],2)


if __name__=='__main__':unittest.main()
