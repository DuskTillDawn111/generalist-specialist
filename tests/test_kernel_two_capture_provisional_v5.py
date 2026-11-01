"""CPU-only provisional estimand tests with explicitly synthetic fixtures."""
import copy
import unittest

from tools.predictor_validation_20261027.kernel_two_capture_provisional_v5 import provisional_mean, ExactSourceTwoCaptureProvisionalResolver


class ProvisionalTests(unittest.TestCase):
    def test_single_capture_cannot_supply_numerical_hypothesis(self):
        rows=[dict(run_id='cpu_fixture_capture_1',duration_ns=v) for v in (10,20)]
        self.assertIsNone(provisional_mean(rows))
        self.assertIsNone(provisional_mean(rows[:1]))

    def test_two_actual_training_observations_define_explicit_mean(self):
        rows=[dict(run_id='cpu_fixture_capture_'+str(i),duration_ns=v) for i,v in enumerate((10,20))]
        self.assertEqual(provisional_mean(rows),15)

    def resolver(self,allowed):
        original=dict(observation_id='cpu_fixture_observation',known=False,strict_known=False,
            numeric_hypothesis_available=False,strict_failure_reasons=['INSUFFICIENT_OBSERVATIONS_OR_RUNS'],
            prediction=dict(duration_hypothesis_ns=None))
        r=ExactSourceTwoCaptureProvisionalResolver.__new__(ExactSourceTwoCaptureProvisionalResolver)
        r.allowed=allowed;r.source=dict(path='/cpu_fixture/report',sha256='a'*64)
        r.base=lambda kernel:copy.deepcopy(original)
        r.by_observation={'cpu_fixture_observation':dict(duration_hypothesis_ns=15,
            original_strict_failure_reasons=['INSUFFICIENT_OBSERVATIONS_OR_RUNS'])}
        return r

    def test_opt_in_required(self):
        p=self.resolver(False)({});self.assertFalse(p['numeric_hypothesis_available']);self.assertNotIn('duration_ns',p)

    def test_original_strict_rejection_and_prediction_preserved(self):
        r=self.resolver(True);p=r(dict(duration_ns=99999999))
        self.assertEqual(p['duration_ns'],15);self.assertFalse(p['strict_known']);self.assertFalse(p['known'])
        self.assertEqual(p['strict_failure_reasons'],['INSUFFICIENT_OBSERVATIONS_OR_RUNS'])
        self.assertIsNone(p['prediction']['duration_hypothesis_ns']);self.assertFalse(p['deployment_qualified'])
        p['provisional_evidence']['original_strict_failure_reasons'].clear()
        self.assertEqual(r({})['strict_failure_reasons'],['INSUFFICIENT_OBSERVATIONS_OR_RUNS'])


if __name__=='__main__':unittest.main()
