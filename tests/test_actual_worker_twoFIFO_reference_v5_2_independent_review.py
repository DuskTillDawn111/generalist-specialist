"""Independent CPU transport-set composition checks for the isolated v5.2."""
import copy
import json
import unittest
from gsdeploy.forecast_errors import UnknownEvidence
from tests.test_actual_worker_twoFIFO_reference_v5_2 import paired_inputs
from tests.test_actual_worker_twoFIFO_reference_v5 import decode
from tools.predictor_validation_20261027 import actual_worker_twoFIFO_reference_v5_2 as m
from tools.predictor_validation_20261027.twoFIFO_paired_transport_v3 import PairedTransportConstraints as C,UnknownTransportInterval


class PairedReferenceIndependentReview(unittest.TestCase):
    def test_unknown_later_generated_ready_survives_JSON_pause_with_same_full_denominator(self):
        kw=paired_inputs();kw['transport_draws'][('owner','text','input',1)]=C(0,0,10,12,-3)
        with self.assertRaises(UnknownTransportInterval) as direct:m.simulate(**kw)
        first=m.simulate(**dict(kw,cutoff_ns=5))
        state=json.loads(json.dumps(first['continuation_state']))
        with self.assertRaises(UnknownTransportInterval) as resumed:m.simulate(**dict(kw,continuation_state=state))
        self.assertEqual(direct.exception.evidence,resumed.exception.evidence)
        e=resumed.exception.evidence
        self.assertEqual(e['target_previous_READY_ns'],11)
        self.assertEqual(e['START_ns_bounds'],[13,23])
        self.assertEqual(e['offered_request_ids'],['r0','r1'])
        self.assertFalse(e['latent_point_selected'])

    def test_singleton_binding_retains_full_paired_constraints_without_line_or_continuation_point(self):
        kw=paired_inputs(seeded=True,write=5);world=decode(m.simulate(**kw))
        for draw in world['supplied_transport_draws']:
            self.assertNotIn('submit_to_stdin_line_available_ns',draw)
            self.assertNotIn('worker_release_to_START_extra_ns',draw)
            self.assertIn('previous_READY_minus_submit_ns',draw)
        for send in world['transport_schedule'].values():
            self.assertIsNone(send['stdin_line_available_ns'])
            self.assertFalse(send['stdin_line_availability_known'])
            self.assertFalse(send['latent_line_or_continuation_point_selected'])
        p=world['continuation_state']['payload']
        self.assertEqual(p['initial_history']['producer_next'],{'owner':3,'idle:stdout':3})
        self.assertEqual(p['producer_next'],{'owner':7,'idle:stdout':3})
        self.assertFalse(world['busy_transport_transfer_qualified'])
        self.assertFalse(world['main_queue_provider_eligible'])
        changed=dict(kw);changed['transport_draws']=copy.deepcopy(kw['transport_draws'])
        changed['transport_draws'][('owner','text','input',1)]=C(0,0,1,10,8)
        # Quantitative source set changes remain bound even if its projection
        # might coincidentally leave some already executed clocks unchanged.
        part=m.simulate(**dict(kw,cutoff_ns=105))
        with self.assertRaises(UnknownEvidence):m.simulate(**dict(changed,continuation_state=part['continuation_state']))
