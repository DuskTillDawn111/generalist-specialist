import copy
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027 import twoFIFO_actual_quality_rescore_v1 as m
from tests import test_twoFIFO_native_quality_labels_v1 as native_tests


def fixture(root):
    raw, plan, case, score, catalog = native_tests.NativeQualityTests().fixture(root)
    score.update(latency_reference_ms={'ocr':10.}, weights={k:1/6 for k in 'QTLPME'})
    observed = dict(schema=m.quality.labels.SCHEMA, attempt_id=plan['attempt_id'], case_id=case['case_id'],
        offered_requests=2, reader_labels=dict(requests=[dict(request_id='r0'),dict(request_id='r1')]),
        measured_original_score=dict(terms=dict(Q=.7,T=.1,L=.2,P=.3,M=.4,E=.5), unknown_reasons=[],
            per_task={'ocr':dict(mean_ms=2.,p95_ms=3.)}),
        measured_resources=dict(startup_energy_joules=10., measurement_board_energy_joules=20.))
    ensemble=dict(six_terms={k:dict(expected_value=v) for k,v in zip('QTLPME',(.2,.1,.2,.3,.4,.5))},
        utility=dict(expected_value=-.3))
    return raw, plan, case, score, catalog, observed, dict(ensemble=ensemble)


class ActualQualitySidecarTests(unittest.TestCase):
    def test_native_output_error_replaces_only_evaluation_Q_after_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            args=fixture(Path(directory)); before=copy.deepcopy(args)
            measured, score, errors=m.evaluate_case(*args[:3],100,*args[3:])
            self.assertEqual(measured['measured_Q'],.1)
            self.assertAlmostEqual(score['full_U'],-1.6/6)
            self.assertEqual(args,before)
            self.assertEqual(errors['terms']['Q']['measured_value'],.1)
            self.assertAlmostEqual(errors['terms']['Q']['error'],.1)
            self.assertFalse(errors['frozen_quality_used_on_both_sides'])
            for key in 'TLPME':self.assertEqual(score['terms'][key],args[5]['measured_original_score']['terms'][key])

    def test_same_count_different_reader_cohort_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args=fixture(Path(directory));args[5]['reader_labels']['requests'][0]['request_id']='foreign'
            with self.assertRaisesRegex(ValueError,'identical offered cohort'):
                m.evaluate_case(*args[:3],100,*args[3:])

    def test_missing_native_output_preserves_all_requests_and_unknown_Q(self):
        with tempfile.TemporaryDirectory() as directory:
            args=fixture(Path(directory));args[0]['request_records'][0]['status']='pending'
            measured, score, errors=m.evaluate_case(*args[:3],100,*args[3:])
            self.assertEqual(measured['offered_requests'],2)
            self.assertIsNone(score['full_U'])
            self.assertIsNone(errors['terms']['Q']['error'])
            self.assertIsNone(errors['U_error'])


if __name__=='__main__':unittest.main()
