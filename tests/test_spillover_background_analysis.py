"""Terminal evidence audit regressions; no GPU API or model execution."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('background_analysis_test', ROOT/'tools/spillover_background_20261022/analyze.py')
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


class BackgroundAnalysisTests(unittest.TestCase):
    def resource_fixture(self):
        config = {'catalog': {'safe_factor': 1.1}, 'gpu_memory_mib': 200}
        raw = {'memory_evidence': {'samples': [dict(unix_ns=900, owned_used_mib=100, foreign_pids=[]),
                                              dict(unix_ns=1250, owned_used_mib=100, foreign_pids=[])],
                                  'safe_memory_mib': 110, 'measured': True, 'resource_valid': True},
               'energy_evidence': {'samples': [dict(time_ns=t, watts=10) for t in [50, 100, 200, 250]],
                                   'startup_begin_time_ns': 50, 'run_joules': 1e-6, 'startup_joules': .5e-6,
                                   'coverage_valid': True, 'sampling_errors': []}}
        return config, raw

    def test_memory_uses_real_unix_field_and_power_uses_monotonic(self):
        config, raw = self.resource_fixture()
        result = audit.resource_summary(raw, config, 100, 200, 1000)
        self.assertTrue(result['memory_full_window_bracket'])
        self.assertTrue(result['memory_pass']); self.assertTrue(result['energy_pass'])
        self.assertAlmostEqual(result['run_joules'], 1e-6)
        raw['memory_evidence']['samples'][-1]['unix_ns'] = 1199
        result = audit.resource_summary(raw, config, 100, 200, 1000)
        self.assertFalse(result['memory_full_window_bracket']); self.assertFalse(result['memory_pass'])

    def test_resource_failure_retains_unknown_and_bad_energy_cannot_pass(self):
        config, raw = self.resource_fixture()
        raw['energy_evidence']['run_joules'] = 999
        result = audit.resource_summary(raw, config, 100, 200, 1000)
        self.assertFalse(result['energy_pass']); self.assertIsNotNone(result['energy_audit_error'])
        result = audit.resource_summary({}, config, 100, 200, 1000)
        self.assertIsNone(result['safe_memory_mib']); self.assertFalse(result['memory_pass'])
        self.assertFalse(result['energy_pass'])

    def test_bound_references_score_actual_native_outputs_input_pool_does_not(self):
        packet = json.loads((ROOT/'artifacts/static-spillover-study-20261022/text-capacity-diagnostic-v1/experiment-packet.json').read_text())
        config = json.loads(Path(packet['system_config']).read_text())
        queue = json.loads((ROOT/'artifacts/static-spillover-study-20261022/text-capacity-diagnostic-v1/runs/queue-progress.json').read_text())
        raw = json.loads(Path(queue['jobs'][0]['raw']['path']).read_text())
        refs = config['catalog']['quality']['references']
        scorer, total = audit.legacy.frozen_scorers(packet['frozen_source'])
        tasks = set()
        for row in raw['priming_request_records']:
            quality = audit.legacy.score_native(row, row['worker_response'], refs, scorer, total)
            self.assertEqual(quality['status'], 'AVAILABLE', quality)
            tasks.add(row['task'])
        self.assertEqual(tasks, {'text', 'image', 'asr', 'ocr'})
        sample = json.loads(Path(packet['samples']).read_text().splitlines()[0])
        self.assertNotIn('answers', sample['sample'])
        row = next(r for r in raw['priming_request_records'] if r['task'] == 'text')
        wrong = {'text': {row['pool_id']: sample}}
        self.assertEqual(audit.legacy.score_native(row, row['worker_response'], wrong, scorer, total)['status'], 'UNKNOWN')

    def test_native_ocr_uses_same_prediction_normalization_as_frozen_utility(self):
        source = ROOT/'artifacts/static-study-gpu-20261022/execution-source-v5'
        scorer, total = audit.legacy.frozen_scorers(source)
        row = dict(task='ocr', pool_id='p', status='completed')
        response = dict(task_type='ocr', prediction=' A\n B ', success=True)
        quality = audit.score_observed_native(row, response, {'ocr': {'p': dict(task='ocr', label='a b')}}, scorer, total)
        self.assertEqual(quality['status'], 'AVAILABLE')
        self.assertEqual(quality['loss'], 0)
        self.assertEqual(response['prediction'], ' A\n B ')

    def test_failure_before_measurement_is_preserved_as_failed_audit(self):
        raw = {'status': 'failed', 'errors': ['load failed'], 'request_records': [dict(role='primary', status='planned')]}
        result = audit.analyze_preserving_failure({}, {'job_id':'j','block':1,'seed':105}, raw, 'hash')
        self.assertEqual(result['audit_status'], 'AUDIT_INCOMPLETE_OR_FAILED')
        self.assertEqual(result['raw_errors'], ['load failed'])
        self.assertEqual(result['raw_record_counts'], {'primary:planned':1})
        self.assertFalse(result['primary_complete']); self.assertFalse(result['energy_pass'])
        self.assertIsNone(result['U'])

    def test_overlap_deduplicates_and_intersects_both_aliases(self):
        left = [(0, 8), (2, 5), (8, 9)]; right = [(3, 10)]
        self.assertEqual(audit.overlap_fraction(0, 10, left), .9)
        self.assertEqual(audit.overlap_fraction(0, 10, left, right), .6)
        with self.assertRaises(ValueError): audit.overlap_fraction(0, 0, left)
        self.assertEqual(audit.activity_context(0, 1), 'UNKNOWN_HANDLER_COVERAGE')
        self.assertEqual(audit.activity_context(0), 'IDLE')
        self.assertEqual(audit.activity_context(.95, .05), 'BUSY_GE_90PCT')

    def test_cutoff_checks_native_and_host_clock_and_unreleased_stays_unknown(self):
        row = dict(scheduled_time_ns=100, status='completed', completion_time_ns=190,
                   worker_response=dict(success=True, start_ns=1100, end_ns=1190))
        self.assertEqual(audit.classify(row, 100, 200, 1000), 'WITHIN_CUTOFF')
        late = copy.deepcopy(row); late['completion_time_ns'] = 201
        self.assertEqual(audit.classify(late, 100, 200, 1000), 'AFTER_CUTOFF')
        late = copy.deepcopy(row); late['worker_response']['end_ns'] = 1201
        self.assertEqual(audit.classify(late, 100, 200, 1000), 'AFTER_CUTOFF')
        self.assertEqual(audit.classify(dict(scheduled_time_ns=None), 100, 200, 1000), 'UNRELEASED_UNKNOWN')

    def test_serial_admission_is_verified_across_primary_support_storage_order(self):
        def row(index, time):
            return dict(admission_sequence=index, scheduled_time_ns=time, completion_time_ns=time+1,
                        status='completed', worker_response=dict(start_ns=time+1000, end_ns=time+1001))
        rows = [row(0, 10), row(2, 14), row(1, 12)]
        self.assertEqual([r['admission_sequence'] for r in audit.verify_serial(rows)], [0,1,2])
        rows[-1]['scheduled_time_ns'] = 10
        with self.assertRaises(AssertionError): audit.verify_serial(rows)


if __name__ == '__main__': unittest.main()
