import copy
import json
import unittest
from fractions import Fraction

from gsdeploy.forecast_errors import UnknownEvidence
from tests.test_actual_worker_twoFIFO_reference_v5 import inputs,decode,coupled_inputs
from tests.test_twoFIFO_settled_history_initial_v1 import seeded_inputs
from tools.predictor_validation_20261027 import actual_worker_twoFIFO_reference_v5_2 as m
from tools.predictor_validation_20261027 import actual_worker_twoFIFO_reference_v5_1 as previous
from tools.predictor_validation_20261027.twoFIFO_paired_transport_v3 import PairedTransportConstraints as C,UnknownTransportInterval


def paired_inputs(*,seeded=False,write=1):
    kw=seeded_inputs() if seeded else inputs(write=write)
    kw['transport_draws']={('owner','text','input',0):C(0,0,write,1,-5),
        ('owner','text','input',1):C(0,0,1,10,9)}
    return kw


class ReferenceV52Tests(unittest.TestCase):
    def test_idle_then_busy_point_identified_without_latent_line(self):
        kw=paired_inputs(write=5);world=decode(m.simulate(**kw))
        self.assertEqual(world['starts'],{'r0':1,'r1':12})
        self.assertEqual(world['parent_dispatches'],{'r0':0,'r1':5})
        self.assertEqual(world['RESULT_receipts'],{'r0':19,'r1':30})
        for send in world['transport_schedule'].values():
            self.assertIsNone(send['stdin_line_available_ns']);self.assertFalse(send['stdin_line_availability_known'])
            self.assertFalse(send['latent_line_or_continuation_point_selected'])
        identified=[r for r in world['trace'] if r['kind']=='paired_transport_START_identification']
        self.assertEqual(len(identified),2)
        self.assertTrue(all(r['identification']['status']=='CONDITIONALLY_POINT_IDENTIFIED_START' for r in identified))
        self.assertFalse(world['READY_processing_gates_worker_START'])

    def test_interval_unknown_has_exact_bounds_and_entire_offered_cohort(self):
        kw=paired_inputs();kw['transport_draws'][('owner','text','input',0)]=C(0,0,5,10,9)
        with self.assertRaises(UnknownTransportInterval) as caught:m.simulate(**kw)
        e=caught.exception.evidence
        self.assertEqual(e['START_ns_bounds'],[1,6]);self.assertEqual(e['offered_request_ids'],['r0','r1'])
        self.assertEqual(e['offered_requests'],2);self.assertTrue(e['missing_predictions_retained_as_UNKNOWN'])
        self.assertFalse(e['latent_point_selected'])

    def test_queued_request_uses_generated_ready_before_deciding_support(self):
        kw=paired_inputs();kw['transport_draws'][('owner','text','input',1)]=C(0,0,10,12,-3)
        with self.assertRaises(UnknownTransportInterval) as caught:m.simulate(**kw)
        e=caught.exception.evidence
        self.assertEqual(e['public_key_arrival_ordinal'],1)
        self.assertEqual(e['target_previous_READY_ns'],11)
        self.assertEqual(e['START_ns_bounds'],[13,23])
        self.assertEqual(e['offered_requests'],2)

    def test_exact_resume_keeps_identified_projection_and_seed_offsets(self):
        for seeded in (False,True):
            kw=paired_inputs(seeded=seeded);full=m.simulate(**kw);origin=100 if seeded else 0
            for offset in (0,Fraction(1,2),1,5,11,12,19,30,119):
                partial=m.simulate(**dict(kw,cutoff_ns=origin+offset))
                saved=json.loads(json.dumps(partial['continuation_state']))
                resume=dict(kw,continuation_state=saved)
                if seeded:resume.pop('settled_initial_state')
                self.assertEqual(m.simulate(**resume),full)
            w=decode(full)
            if seeded:
                first=next(r for r in w['trace'] if r['kind']=='latent_occurrence')
                self.assertEqual(first['producer_sequence'],3);self.assertEqual(first['owner_issuance_ordinal'],3)
                self.assertEqual(w['continuation_state']['payload']['sequence'],10)

    def test_explicit_latent_branch_preserves_existing_ownerphase_response(self):
        for kw in (inputs(),coupled_inputs(),seeded_inputs()):
            new,old=decode(m.simulate(**kw)),decode(previous.simulate(**kw))
            for field in ('trace','starts','READY_occurrences','RESULT_receipts','response_exposure','transport_schedule'):
                self.assertEqual(new[field],old[field])

    def test_modified_paired_constraints_cannot_resume_bound_checkpoint(self):
        kw=paired_inputs();partial=m.simulate(**dict(kw,cutoff_ns=5))
        kw['transport_draws'][('owner','text','input',1)]=C(0,0,1,11,9)
        with self.assertRaises(UnknownEvidence):m.simulate(**dict(kw,continuation_state=partial['continuation_state']))


if __name__=='__main__':unittest.main()
