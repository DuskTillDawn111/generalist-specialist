"""Independent CPU-only fit failure and joint transport contract negatives."""
import copy
from fractions import Fraction
import unittest
from unittest.mock import patch

from tests import test_twoFIFO_phase12_fit_v1 as response_fixture
from tests import test_twoFIFO_joint_request_law_v2 as law_fixture
from tools.predictor_validation_20261027 import twoFIFO_phase12_fit_v1 as response
from tools.predictor_validation_20261027 import twoFIFO_joint_request_law_v2 as joint


class PhaseJointIndependentTests(unittest.TestCase):
    def test_optimizer_failure_returns_unknown_without_parameters_or_column_fallback(self):
        rows,_=response_fixture.synthetic_rows()
        with patch.object(response.numeric,'fit',side_effect=FloatingPointError('no root')) as numeric:
            result,parameters=response.fitted_rows(rows)
        numeric.assert_called_once();self.assertIsNone(parameters)
        self.assertFalse(result['numerical_qualification']);self.assertEqual(len(result['features']),12)
        self.assertEqual(result['macro_rows'],len(rows))

    def test_explicit_transport_requires_all_five_components_and_exact_vector_pair(self):
        *_,vector=law_fixture.source_fixture();c=vector['transport_constraints']
        line=Fraction(min(c['submit_to_write_record_ns'],c['observed_submit_to_START_ns']))
        extra=Fraction(c['observed_submit_to_START_ns'])-max(Fraction(c['previous_READY_minus_submit_ns']),line)
        draw=joint.TransportDraw(Fraction(c['enqueue_after_arrival_ns']),Fraction(c['sender_turn_extra_ns']),
            Fraction(c['submit_to_write_record_ns']),line,extra)
        support=dict(status='SUPPLIED_FEASIBLE_LATENT_SCENARIO',vector_id=vector['vector_id'],public_key_ordinal=0,draw=draw)
        with self.assertRaises(ValueError):joint.compile_draw_v2(vector,0,dict(support,vector_id='another-request-vector'))
        with self.assertRaises(ValueError):joint.compile_draw_v2(vector,0,dict(support,draw=dict(submit_to_START=0)))
        self.assertTrue(joint.compile_draw_v2(vector,0)['status'].startswith('UNKNOWN'))

    def test_selected_training_block_cannot_borrow_unseen_input_from_another_block(self):
        *_,vector=law_fixture.source_fixture()
        vector['training_block_id']='training-block-1'
        law=dict(schema=joint.SCHEMA,blocks=['training-block-2','training-block-3'],vectors=[vector])
        result=joint.choose_vector(law,vector['public_key'],0,block_quantile=0,within_block_quantile=0)
        self.assertEqual(result['status'],'UNKNOWN_EXACT_PUBLIC_INPUT_IN_SELECTED_WHOLE_BLOCK')


if __name__=='__main__':unittest.main()
