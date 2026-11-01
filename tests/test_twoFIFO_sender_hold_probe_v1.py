import unittest

from tools.predictor_validation_20261027.twoFIFO_sender_hold_probe_v1 import SenderHoldProbe


class SenderProbeTests(unittest.TestCase):
    def test_explicit_hold_only_sender_clock_and_delegate_payload_unchanged(self):
        class Transport:
            process='existing-owned-process'
            def __init__(self):self.payloads=[]
            def send(self,payload):self.payloads.append(payload);return 'sent'
        now=[100];events=[];transport=Transport()
        def wait(deadline):now[0]=max(now[0],deadline)
        probe=SenderHoldProbe(transport,'owner',{0:20_000_000},clock_ns=lambda:now[0],wait_until_ns=wait,events=events)
        payload=dict(request_id='r0',task_type='text',sample={'context':'unchanged'})
        self.assertEqual(probe.send(payload),'sent')
        self.assertIs(transport.payloads[0],payload)
        self.assertEqual([events[0]['hold_begin_ns'],events[0]['hold_end_ns']],[100,20_000_100])
        self.assertFalse(events[0]['natural_transport_fit_eligible'])
        self.assertEqual(probe.process,'existing-owned-process')
        probe.send(dict(payload,request_id='r1'))
        self.assertEqual(events[1]['requested_hold_ns'],0)

    def test_control_messages_cannot_consume_measured_probe_ordinal(self):
        probe=SenderHoldProbe(object(),'owner',{0:1},clock_ns=lambda:0,wait_until_ns=lambda _:None,events=[])
        with self.assertRaises(ValueError):probe.send(dict(request_id='control',task_type='__profile_start__'))
        self.assertEqual(probe.ordinal,0)


if __name__=='__main__':unittest.main()
