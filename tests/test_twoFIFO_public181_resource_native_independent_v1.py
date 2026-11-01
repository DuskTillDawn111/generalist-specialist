"""Independent fixed-window/censoring checks; native intake mocked only as stated."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_twoFIFO_public181_resource_native_v1 import NativeResourceTests, resources, SECOND
from tools.predictor_validation_20261027 import twoFIFO_public181_resource_execute_v1 as executor
from tools.predictor_validation_20261027 import twoFIFO_public181_resource_review_v1 as review
from tools.predictor_validation_20261027 import twoFIFO_public181_resource_intake_v1 as intake
from tools.predictor_validation_20261027 import twoFIFO_public181_partial_protocol_v1 as partial
from tools.predictor_validation_20261027.stage_response_features_v1 import checked


class IndependentResourceTests(unittest.TestCase):
    setUpClass = classmethod(NativeResourceTests.setUpClass.__func__)
    tearDownClass = classmethod(NativeResourceTests.tearDownClass.__func__)

    def test_startup_and_drain_memory_are_maxed_before_any_mean(self):
        raw = resources()
        raw['board_samples'][1]['owned_used_mib'] = 700.
        raw['board_samples'][500]['owned_used_mib'] = 600.
        raw['board_samples'][500]['watts'] = 10000.
        result = review.resource_facts(raw)
        self.assertEqual(result['joint_tuple'], dict(startup_joules=30200.,
            premeasurement_owned_sampled_peak_mib=700.,
            measurement_through_drain_owned_sampled_peak_mib=600., measurement_board_joules=9000.))
        case = result['observed_original_resources']['cases'][0]
        self.assertEqual(case['startup_through_this_drain_owned_sampled_peak_mib'], 700.)
        self.assertGreater(case['drain_energy_joules_diagnostic_only'], 12000.)
        self.assertFalse(result['safe_factor_applied'])
        self.assertFalse(result['startup_amortization_applied'])

    def test_one_rederived_ineligible_attempt_keeps_eight_rows_and_no_catalog(self):
        # Authenticate/review is deliberately mocked here to isolate denominator
        # semantics. This test cannot certify any native row.
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); refs = []; reports = {}
            joint = review.resource_facts(resources())
            for index, item in enumerate(self.manifest['attempts']):
                raw_ref = executor.campaign.save(folder/('raw'+str(index)), dict(
                    schema=executor.RAW_SCHEMA, split='training', backend='native', kind='measured'))
                eligible = index != 3
                document = dict(schema=review.SCHEMA, status=intake.PASS if eligible else 'NOT_ELIGIBLE',
                    resource_manifest=self.manifest_ref, validation_labels_read=False,
                    training_feature_extraction_eligible=eligible, issues=[] if eligible else ['INVALID_OWNERSHIP'],
                    raw=raw_ref, inputs={'plan.json': item['plan']}, sources=[],
                    execution={'path': '/EXPLICIT-CPU-MOCK'}, observed_resources=joint if eligible else None)
                report_ref = executor.campaign.save(folder/('review'+str(index)), document)
                refs.append(report_ref['path']); reports[raw_ref['path']] = report_ref
            with patch.object(intake.reviewer, 'review', side_effect=lambda raw, *a, **k: reports[str(raw)]), \
                    patch.object(intake.reviewer, 'resource_facts', return_value=joint) as measured, \
                    patch.object(intake.quality, 'measure', return_value={'CPU_TEST_ONLY': True}) as quality:
                result = intake.load_complete(self.manifest_ref['path'], refs)
            self.assertEqual((len(result['rows']), result['observed_resource_attempts'], result['unknown_resource_attempts']), (8, 7, 1))
            self.assertEqual(measured.call_count, 7); self.assertEqual(quality.call_count, 7)
            self.assertIsNone(result['paired_resource_catalog'])
            self.assertIsNone(result['rows'][3]['joint_tuple'])
            self.assertEqual(result['rows'][3]['unknown_reasons'], ['INVALID_OWNERSHIP'])
            self.assertFalse(result['unknown_attempts_removed_or_survivor_mass_renormalized'])

    def test_real_partial_phase_without_result_stays_censored(self):
        def event(**extra):
            return dict(request_id='r', perf_ns=20, received_time_ns=21,
                collector_dequeued_time_ns=22, collector_processing_time_ns=23, **extra)
        row = dict(request_id='r', model='text#0', status='incomplete', scheduled_time_ns=1,
            enqueued_time_ns=2, actual_submit_time_ns=3, submit_write_completed_time_ns=4,
            observed_worker_start_time_ns=20, execution_events=[event(pid=7)])
        phase = event(event='start', kind='gpu', segment_id='one')
        phase.update(perf_ns=24, received_time_ns=25, collector_dequeued_time_ns=26, collector_processing_time_ns=27)
        row['execution_events'].append(phase)
        result = partial.inspect_prefix(row, {'pid': 7}, shutdown_end_ns=100)
        self.assertTrue(result['missing_RESULT']); self.assertTrue(result['missing_READY'])
        self.assertEqual(result['unfinished_phase_keys'], [['gpu', 'one']])
        self.assertFalse(result['partial_device_progress_qualified'])
        changed = copy.deepcopy(row); changed['execution_events'].append(dict(phase))
        with self.assertRaises(ValueError): partial.inspect_prefix(changed, {'pid': 7}, shutdown_end_ns=100)
        changed = copy.deepcopy(row); changed['execution_events'][-1]['collector_processing_time_ns'] = 101
        with self.assertRaises(ValueError): partial.inspect_prefix(changed, {'pid': 7}, shutdown_end_ns=100)

    def test_actual_cpu02_preparation_reconstructs_all_eight(self):
        path = 'artifacts/predictor-public181-resource-owned-producer-v1-20261029-cpu02-prepared/manifest.json'
        manifest = executor.audit(path)
        self.assertEqual(len(manifest['attempts']), 8)
        self.assertEqual(sum(len(checked(item['plan'])['cases'][0]['arrivals']) for item in manifest['attempts']), 1448)
        self.assertTrue(all(checked(item['plan'])['runtime_policy']['admission'] is None for item in manifest['attempts']))


if __name__ == '__main__':
    unittest.main()
