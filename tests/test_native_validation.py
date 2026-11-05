"""Independent validation must expose timing misses and prevent fit leakage."""
import copy
import unittest

from gsdeploy.forecasting.native_training import fit_native_captures
from gsdeploy.forecasting.native_validation import DEFAULT_CRITERIA,validate_native_fit
from tests.test_native_training import capture,POOL


def validation(run):
    bundle=capture(run,'overlap')
    bundle['raw']['split']=bundle['plan']['split']='development_validation'
    return bundle


class NativeValidationTests(unittest.TestCase):
    def setUp(self):
        self.fit=fit_native_captures([capture('s1','solo'),capture('s2','solo'),
                                    capture('o1','overlap'),capture('o2','overlap')],POOL,block_seconds=.5)

    def test_independent_fifo_and_exposure_energy_pass_without_utility_admission(self):
        report=validate_native_fit(self.fit,[validation('v1'),validation('v2')],DEFAULT_CRITERIA)
        self.assertTrue(report['development_accuracy_passed'],report['reasons'])
        self.assertAlmostEqual(report['latency_errors']['mean'],0.,places=5)
        self.assertFalse(report['admission_enabled']);self.assertFalse(report['formal_utility_qualified'])

    def test_unpredicted_native_slowdown_fails_without_refitting(self):
        bundles=[validation('v1'),validation('v2')]
        for bundle in bundles:
            for row in bundle['raw']['request_records']:
                response=row['worker_response'];old_end=response['observed_stages'][1]['end_ns'];delta=20_000_000
                for event in row['execution_events']:
                    if event['perf_ns']>=old_end:event['perf_ns']+=delta;event['received_time_ns']+=delta
                for span in response['observed_stages']:
                    for key in ('start_ns','end_ns'):
                        if span[key]>=old_end:span[key]+=delta
                response['end_perf_ns']+=delta;row['completion_time_ns']+=delta;row['response_reader_event_enqueue_time_ns']+=delta
                for key in ('perf_ns','received_time_ns'):row['previous_handler_next_request_ready'][key]+=delta
        original=copy.deepcopy(self.fit)
        report=validate_native_fit(self.fit,bundles,DEFAULT_CRITERIA)
        self.assertFalse(report['development_accuracy_passed'])
        self.assertIn('FIFO_E2E_ERROR_OR_UNKNOWN',report['reasons'])
        self.assertIn('NATIVE_WORK_ERROR_OR_UNKNOWN',report['reasons'])
        self.assertEqual(self.fit,original)

    def test_training_receipts_and_single_repeat_cannot_qualify(self):
        with self.assertRaisesRegex(ValueError,'nontraining'):
            validate_native_fit(self.fit,[validation('s1'),validation('v2')],DEFAULT_CRITERIA)
        report=validate_native_fit(self.fit,[validation('v1')],DEFAULT_CRITERIA)
        self.assertFalse(report['development_accuracy_passed'])
        self.assertIn('INSUFFICIENT_INDEPENDENT_VALIDATION_RUNS',report['reasons'])

    def test_fifo_energy_uses_predicted_duration_beyond_observed_stage_exposure(self):
        changed=copy.deepcopy(self.fit)
        for profile in changed['lifecycle']['profiles']:
            profile['scenarios']['mean']['parent_receipt_ms']=1000.
            profile['scenarios']['mean']['worker_next_ready_ms']=1000.
        report=validate_native_fit(changed,[validation('v1'),validation('v2')],DEFAULT_CRITERIA)
        for run in report['runs']:
            self.assertAlmostEqual(run['energy']['relative_error'],0.,places=5)
            self.assertGreater(run['fifo_energy']['relative_error'],.15)
        self.assertFalse(report['development_accuracy_passed'])
        self.assertIn('v1:FIFO_INTEGRATED_ENERGY_ERROR_OR_UNKNOWN',report['reasons'])


if __name__=='__main__':unittest.main()
