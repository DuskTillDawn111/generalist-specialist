import copy
import unittest
from tests.test_twoFIFO_joint_request_law_v2 import source_fixture
from tools.predictor_validation_20261027 import twoFIFO_joint_request_law_v4 as m


class SlotScopedLawIndependentReview(unittest.TestCase):
    def test_full_public_slot_prevents_input_key_and_ready_slack_plan_alias(self):
        *_,first=source_fixture();second=copy.deepcopy(first)
        second['public_key'][2]='another-exact-input'
        second['ready_protocol_slacks']['reader_slack_ns']+=321
        one=m.compile_draw_v4(first,0);two=m.compile_draw_v4(second,0)
        self.assertNotEqual(one['plan'],two['plan'])
        timing={one['plan']:one['protocol_slacks'],two['plan']:two['protocol_slacks']}
        self.assertEqual(len(timing),2)
        self.assertEqual(timing[two['plan']]['worker_READY']['reader_slack_ns']-
            timing[one['plan']]['worker_READY']['reader_slack_ns'],321)
        a=[mark for node in one['plan'].macros for mark in node.marks]
        b=[mark for node in two['plan'].macros for mark in node.marks]
        self.assertEqual([(x.work_ns,x.action,x.phase,x.query_complete,x.clear_scope) for x in a],
            [(x.work_ns,x.action,x.phase,x.query_complete,x.clear_scope) for x in b])
        self.assertTrue(set(x.identity for x in a).isdisjoint(x.identity for x in b))
        self.assertTrue(set(x.event_id for x in a if x.event_id).isdisjoint(x.event_id for x in b if x.event_id))
        for compiled,original in ((one,first),(two,second)):
            expected=[mark['identity'] for node in original['nodes'] for mark in node['marks']]
            actual=[compiled['generated_to_source_message_identity'][mark.identity]
                    for node in compiled['plan'].macros for mark in node.marks]
            self.assertEqual(actual,expected)
            self.assertFalse(compiled['latent_line_or_continuation_point_selected'])
