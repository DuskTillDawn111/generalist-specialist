import copy
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027 import twoFIFO_native_quality_labels_v1 as module
from tests import test_twoFIFO_native_quality_labels_v1 as author_tests


def merge_fixture():
    plan=dict(workload_rps={'ocr':1.},latency_reference_ms={'ocr':10.},weights={k:1/6 for k in 'QTLPME'})
    score=dict(terms=dict(Q=None,T=.1,L=.2,P=.3,M=.4,E=.5),
        unknown_reasons=['UNKNOWN_AGGREGATE_QUALITY:ocr'],absolute_utility_complete=False,
        per_task={'ocr':dict(mean_ms=2.,p95_ms=3.)})
    observed=dict(schema=module.labels.SCHEMA,attempt_id='attempt',case_id='case',
        measured_original_score=score,measured_resources=dict(startup_energy_joules=10.,measurement_board_energy_joules=20.))
    quality=dict(schema=module.SCHEMA,attempt_id='attempt',case_id='case',measured_Q=.6,
        unknown_reasons=[],per_task={},scoring_contract=copy.deepcopy(plan))
    return plan,observed,quality


class IndependentActualQualityTests(unittest.TestCase):
    def test_merge_rejects_normalizer_that_changes_preserved_L_P(self):
        plan,observed,quality=merge_fixture()
        plan['latency_reference_ms']['ocr']=100.
        quality['scoring_contract']=copy.deepcopy(plan)
        with self.assertRaises(ValueError):
            module.with_measured_quality(observed,quality,plan)

    def test_nonfinite_and_negative_Q_are_not_complete_utilities(self):
        for value in (float('nan'),float('inf'),-.1):
            with self.subTest(value=value):
                plan,observed,quality=merge_fixture();quality['measured_Q']=value
                with self.assertRaises(ValueError):
                    module.with_measured_quality(observed,quality,plan)

    def test_actual_CER_above_one_is_preserved_without_clipping(self):
        with tempfile.TemporaryDirectory() as temp:
            raw,plan,case,score,reference=author_tests.NativeQualityTests().fixture(Path(temp))
            raw['request_records'][0]['worker_response']['prediction']='z'*20
            raw['request_records'][1]['worker_response']['prediction']='z'*20
            quality=module.measure_case(raw,plan,case,100,score,reference)
            self.assertEqual(quality['per_task']['ocr']['numerator'],40.)
            self.assertEqual(quality['per_task']['ocr']['denominator'],10.)
            self.assertEqual(quality['measured_Q'],4.)

    def test_known_Q_does_not_fill_missing_other_term_or_recharge_resources(self):
        plan,observed,quality=merge_fixture()
        observed['measured_original_score']['terms']['M']=None
        before=copy.deepcopy(observed)
        result=module.with_measured_quality(observed,quality,plan)
        self.assertIsNone(result['terms']['M']);self.assertIsNone(result['full_U'])
        self.assertFalse(result['absolute_utility_complete'])
        self.assertEqual(observed,before)
        self.assertEqual(result['terms']['E'],.5)

    def test_missing_raw_request_fails_complete_cohort_instead_of_rescoring_subset(self):
        with tempfile.TemporaryDirectory() as temp:
            raw,plan,case,score,reference=author_tests.NativeQualityTests().fixture(Path(temp))
            raw['request_records'].pop()
            with self.assertRaises(ValueError):
                module.measure_case(raw,plan,case,100,score,reference)

    def test_truthy_success_value_is_not_native_success_evidence(self):
        for value in ('failed',1):
            with self.subTest(value=value),tempfile.TemporaryDirectory() as temp:
                raw,plan,case,score,reference=author_tests.NativeQualityTests().fixture(Path(temp))
                raw['request_records'][0]['worker_response'].update(success=value,prediction='a')
                with self.assertRaises(ValueError):
                    module.measure_case(raw,plan,case,100,score,reference)

    def test_false_success_and_missing_completion_remain_unknown(self):
        for mutation in ('false_success','missing_completion'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as temp:
                raw,plan,case,score,reference=author_tests.NativeQualityTests().fixture(Path(temp))
                if mutation=='false_success':
                    raw['request_records'][0]['worker_response'].update(success=False,prediction='a')
                else:raw['request_records'][0]['completion_time_ns']=None
                result=module.measure_case(raw,plan,case,100,score,reference)
                self.assertIsNone(result['measured_Q'])
                self.assertEqual(result['per_task']['ocr']['missing'],['r0'])
                self.assertEqual(result['offered_requests'],2)

    def test_rate_contract_changed_without_quality_recomputation_rejected(self):
        plan,observed,quality=merge_fixture()
        plan['workload_rps']['ocr']=2.
        with self.assertRaises(ValueError):
            module.with_measured_quality(observed,quality,plan)


if __name__=='__main__':unittest.main()
