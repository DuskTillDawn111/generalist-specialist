"""Independent six-term full-world endpoint, UNKNOWN and scope checks."""
import copy
from fractions import Fraction as F
import unittest

from tests import test_macro_six_term_bridge_v1 as fixtures
from tools.predictor_validation_20261027.macro_six_term_bridge_v1 import whole_score
from tools.predictor_validation_20261027.macro_response_queue_reference_v1 import encode


class OriginalSixTermIndependentReview(unittest.TestCase):
    def inputs(self):return fixtures.SixTermBridgeTests().inputs(work=400_000_000)

    def test_UNKNOWN_startup_energy_cannot_be_zero_in_absolute_whole_world_E(self):
        ref,rows,plan,q,res=self.inputs();res['startup_energy_joules']=None
        result=whole_score(ref,rows,plan,q,res)
        self.assertIsNone(result['score']['terms']['E'])
        self.assertFalse(result['score']['six_term_comparison_complete'])

    def test_each_raw_resource_UNKNOWN_stays_unknown(self):
        for field,term in [('measurement_board_energy_joules','E'),
                ('startup_through_phase_start_owned_peak_mib','M'),('phase_start_through_drain_owned_peak_mib','M')]:
            ref,rows,plan,q,res=self.inputs();res[field]=None
            with self.subTest(field=field):self.assertIsNone(whole_score(ref,rows,plan,q,res)['score']['terms'][term])

    def test_reader_endpoint_controls_throughput_and_latency_even_with_late_processing(self):
        ref,rows,plan,q,res=self.inputs()
        ref['RESULT_processed']={rid:encode(F(10_000_000_000)) for rid in ref['RESULT_receipts']}
        result=whole_score(ref,rows,plan,q,res)
        self.assertEqual(result['score']['terms']['T'],0)
        self.assertAlmostEqual(result['score']['terms']['L'],.4)
        self.assertFalse(result['main_queue_provider_eligible'])

    def test_censored_cohort_cannot_use_completed_subset_P95_or_full_U(self):
        ref,rows,plan,q,res=fixtures.SixTermBridgeTests().inputs()
        score=whole_score(ref,rows,plan,q,res)['score']
        self.assertEqual(score['cohort_size'],2);self.assertEqual(score['incomplete_ids'],['b'])
        self.assertIsNone(score['terms']['P']);self.assertIsNone(score['full_U'])

    def test_nonzero_origin_and_fractional_reader_roundup_are_translation_consistent(self):
        ref,rows,plan,q,res=self.inputs();base=whole_score(ref,rows,plan,q,res)
        origin=1_000_000_000_000
        ref['original_window']=encode([F(origin),F(origin+1_000_000_000)])
        ref['original_drain_end_ns']=encode(F(origin+3_000_000_000));ref['cutoff_ns']=ref['original_drain_end_ns']
        for rid,stamp in ref['RESULT_receipts'].items():
            ref['RESULT_receipts'][rid]=encode(F(stamp['numerator'],stamp['denominator'])+origin+F(1,3))
        for row in rows:row['arrival_ns']+=origin
        res.update(original_window_ns=[origin,origin+1_000_000_000],original_drain_end_ns=origin+3_000_000_000)
        changed=whole_score(ref,rows,plan,q,res)
        self.assertAlmostEqual(changed['maximum_timestamp_roundup_ns'],2/3)
        self.assertAlmostEqual(changed['score']['terms']['L']-base['score']['terms']['L'],1e-9)

    def test_memory_max_is_across_startup_and_drain_not_sum(self):
        ref,rows,plan,q,res=self.inputs();res['phase_start_through_drain_owned_peak_mib']=80
        score=whole_score(ref,rows,plan,q,res)['score']
        self.assertAlmostEqual(score['terms']['M'],.96)
        self.assertEqual(score['safe_memory_mib'],96)

    def test_boolean_claims_cannot_promote_bridge_native_or_final_eligibility(self):
        ref,rows,plan,q,res=self.inputs()
        ref.update(dispatch_contract_matches_original_sender=True,busy_transport_transfer_qualified=True,internal_work_template_qualified=True)
        result=whole_score(ref,rows,plan,q,res)
        self.assertFalse(result['main_queue_provider_eligible']);self.assertFalse(result['final_gain_eligible'])
        self.assertFalse(result['original_stability_evaluated'])

    def test_full_world_does_not_accept_prefix_or_changed_weights(self):
        ref,rows,plan,q,res=self.inputs();ref['cutoff_ns']=encode(F(100))
        with self.assertRaises(ValueError):whole_score(ref,rows,plan,q,res)
        ref,rows,plan,q,res=self.inputs();plan['weights'].update(E=.5,Q=0)
        with self.assertRaises(ValueError):whole_score(ref,rows,plan,q,res)


if __name__=='__main__':unittest.main()
