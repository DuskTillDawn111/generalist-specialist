"""CPU adversarial review using immutable natural-training evidence.

Regression cases cover independently reproduced review gaps. No GPU, quality
label, or trace export is used.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027 import profile_training_evidence_review_v1 as audit
from tools.predictor_validation_20261027.device_progress_smoke_raw_verify_v4 import core
from tools.predictor_validation_20261027.native_collect import receipt


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/predictor-device-progress-paired-v6-20261028-training01'
RUN = BASE / 'training/natural'


class IndependentTrainingReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads((RUN / 'plan.json').read_text())
        cls.raw = json.loads((RUN / 'raw.json').read_text())
        execution = json.loads((BASE / 'execution-training-natural/execution.json').read_text())
        cls.collector_pid = execution['process_group']
        cls.raw_ref = receipt(RUN / 'raw.json')

    def changed(self, section):
        raw = dict(self.raw)
        raw[section] = copy.deepcopy(raw[section])
        return raw

    def test_original_full_training_cohort(self):
        result = audit.inspect_cohort(self.plan, self.raw)
        self.assertEqual((result['jobs'], result['measured_requests'], result['priming_requests']), (26, 50, 18))
        self.assertTrue(all(r['worker_response']['success'] is True
                            for r in self.raw['priming_request_records']))

    def test_original_complete_monitoring_and_memory(self):
        result = audit.inspect_samples(self.raw, self.collector_pid)
        self.assertEqual(result['total_samples'], 4762)
        self.assertEqual(result['owned_sampled_peak_mib'], 34428)
        self.assertEqual(receipt(RUN / 'raw.json'), self.raw_ref)

    def test_missing_duplicate_and_wrong_identity_requests_rejected(self):
        for mode in ('missing', 'duplicate', 'wrong_input', 'wrong_role'):
            with self.subTest(mode=mode):
                raw = self.changed('request_records')
                if mode == 'missing':
                    raw['request_records'].pop()
                elif mode == 'duplicate':
                    raw['request_records'].append(copy.deepcopy(raw['request_records'][0]))
                elif mode == 'wrong_input':
                    raw['request_records'][0]['pool_id'] = 'absent-input'
                else:
                    raw['request_records'][0]['validation_view'] = 'unseen_input'
                with self.assertRaises(ValueError):
                    audit.inspect_cohort(self.plan, raw)

    def test_batch_order_identity_and_overlap_rejected(self):
        for mode in ('order', 'identity', 'overlap'):
            with self.subTest(mode=mode):
                raw = self.changed('batches')
                if mode == 'order':
                    raw['batches'].reverse()
                elif mode == 'identity':
                    raw['batches'][0]['request_ids'].pop()
                else:
                    raw['batches'][1]['start_ns'] = raw['batches'][0]['start_ns']
                with self.assertRaises(ValueError):
                    audit.inspect_cohort(self.plan, raw)

    def test_arrival_offset_and_parent_receipt_rejected(self):
        for key in ('scheduled_time_ns', 'completion_time_ns'):
            with self.subTest(key=key):
                raw = self.changed('request_records')
                raw['request_records'][0][key] += 1
                with self.assertRaises(ValueError):
                    audit.inspect_cohort(self.plan, raw)

    def test_zero_missing_reordered_and_identity_colliding_priming_rejected(self):
        for mode in ('zero', 'missing', 'order', 'identity'):
            with self.subTest(mode=mode):
                raw = self.changed('priming_request_records')
                if mode == 'zero':
                    raw['priming_request_records'] = []
                elif mode == 'missing':
                    raw['priming_request_records'].pop()
                elif mode == 'order':
                    raw['priming_request_records'].reverse()
                else:
                    raw['priming_request_records'][0]['request_id'] = raw['request_records'][0]['request_id']
                with self.assertRaises(ValueError):
                    audit.inspect_cohort(self.plan, raw)

    def test_invalid_process_memory_and_totals_rejected(self):
        index = next(i for i, s in enumerate(self.raw['board_samples']) if s['phase'] == 'measurement')
        for mode in ('nan_process', 'negative_process', 'nan_total', 'infinity_total', 'wrong_sum', 'bool_process'):
            with self.subTest(mode=mode):
                raw = self.changed('board_samples')
                sample = raw['board_samples'][index]
                pid = next(iter(sample['owned_processes_mib']))
                if mode.endswith('_process'):
                    value = {'nan_process': float('nan'), 'negative_process': -1, 'bool_process': True}[mode]
                    sample['owned_processes_mib'][pid] = sample['processes_mib'][pid] = value
                else:
                    sample['owned_used_mib'] = {'nan_total': float('nan'), 'infinity_total': float('inf'),
                                              'wrong_sum': sample['owned_used_mib'] + 1}[mode]
                with self.assertRaises(ValueError):
                    audit.inspect_samples(raw, self.collector_pid)

    def test_foreign_pid_unresolved_ancestry_and_changed_launcher_pin_rejected(self):
        index = next(i for i, s in enumerate(self.raw['board_samples']) if s['ownership_evidence'])
        for mode in ('foreign', 'unresolved', 'launcher_pin', 'collector_pin'):
            with self.subTest(mode=mode):
                raw = self.changed('board_samples')
                sample = raw['board_samples'][index]
                if mode == 'foreign':
                    sample['foreign_pids'] = [123456789]
                elif mode == 'unresolved':
                    sample['unresolved_ownership_reasons'] = ['unresolved']
                else:
                    key = 'launcher_start_ticks' if mode == 'launcher_pin' else 'collector_start_ticks'
                    sample['ownership_evidence'][0][key] += 1
                with self.assertRaises(ValueError):
                    audit.inspect_samples(raw, self.collector_pid)

    def test_monitoring_final_drain_and_shutdown_scope_rejected(self):
        for mode in ('missing_drain', 'shutdown_sample'):
            with self.subTest(mode=mode):
                raw = self.changed('board_samples')
                if mode == 'missing_drain':
                    raw['board_samples'] = [s for s in raw['board_samples'] if s['phase'] != 'final_drain']
                else:
                    raw['board_samples'][-1]['phase'] = 'shutdown'
                with self.assertRaises(ValueError):
                    audit.inspect_samples(raw, self.collector_pid)

    def test_mutated_receipted_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'evidence.json'
            path.write_text('{"role":"training"}')
            ref = receipt(path)
            self.assertEqual(audit.checked(ref), {'role': 'training'})
            path.write_text('{"role":"validation"}')
            with self.assertRaisesRegex(ValueError, 'Immutable evidence changed'):
                audit.checked(ref)

    def test_wrong_collector_identity_rejected(self):
        with self.assertRaises(ValueError):
            audit.inspect_samples(self.raw, self.collector_pid + 1)

    def test_completed_but_unsuccessful_or_nonboolean_responses_rejected(self):
        for section in ('request_records', 'priming_request_records'):
            for value in (False, None, 1):
                with self.subTest(section=section, value=value):
                    raw = self.changed(section)
                    raw[section][0]['worker_response']['success'] = value
                    with self.assertRaises(ValueError):
                        audit.inspect_cohort(self.plan, raw)

    def test_alias_coverage_and_duplicates_rejected_in_every_phase(self):
        phases = {sample['phase'] for sample in self.raw['board_samples']}
        for phase in phases:
            index = next(i for i, sample in enumerate(self.raw['board_samples'])
                         if sample['phase'] == phase and sample['owned_aliases'])
            for mode in ('missing', 'duplicate', 'unknown'):
                with self.subTest(phase=phase, mode=mode):
                    raw = self.changed('board_samples')
                    aliases = raw['board_samples'][index]['owned_aliases']
                    if mode == 'missing':
                        aliases.pop()
                    elif mode == 'duplicate':
                        aliases.append(aliases[0])
                    else:
                        aliases[0] = 'unknown#0'
                    with self.assertRaises(ValueError):
                        audit.inspect_samples(raw, self.collector_pid)

    def test_other_collection_execution_and_raw_review_cannot_be_substituted(self):
        raw_review = ROOT / 'artifacts/predictor-device-progress-v6-natural-training-20261028-rawreview01/report.json'
        smoke_review = ROOT / 'artifacts/predictor-device-progress-v6-natural-smoke-20261028-rawreview01/report.json'
        execution = BASE / 'execution-training-natural/execution.json'
        smoke_execution = BASE / 'execution-smoke-natural/execution.json'
        for supplied_execution, supplied_review in ((smoke_execution, raw_review), (execution, smoke_review)):
            with self.subTest(execution=str(supplied_execution), review=str(supplied_review)):
                with tempfile.TemporaryDirectory() as folder:
                    out = Path(folder) / 'review'
                    with self.assertRaises(ValueError):
                        audit.review(RUN, supplied_execution, supplied_review, out)
                    self.assertFalse(out.exists())

    def test_protocol_consistent_failed_priming_must_be_rejected(self):
        raw = self.changed('priming_request_records')
        raw['priming_request_records'][0]['worker_response'].update(
            success=False, finish_reason='length', generated_tokens=128,
            terminal_token_id=0, truncated=True)
        parsed = core.reparse_outputs(raw)
        self.assertEqual(parsed['issues'], [])
        self.assertEqual(core.validate_raw(raw, parsed['policies'])['issues'], [])
        with self.assertRaises(ValueError):
            audit.inspect_cohort(self.plan, raw)

    def test_startup_aliases_must_equal_owned_ancestry_aliases(self):
        raw = self.changed('board_samples')
        raw['board_samples'][0]['owned_aliases'] = ['unknown#0']
        with self.assertRaises(ValueError):
            audit.inspect_samples(raw, self.collector_pid)

    def test_submit_after_completion_must_be_rejected(self):
        raw = self.changed('request_records')
        row = raw['request_records'][0]
        row['actual_submit_time_ns'] = row['completion_time_ns'] + 1
        with self.assertRaises(ValueError):
            audit.inspect_cohort(self.plan, raw)

    def test_all_request_chronology_edges_and_start_identity_rejected(self):
        for section in ('request_records', 'priming_request_records'):
            original = self.raw[section][0]
            changes = [
                ('enqueued_time_ns', original['scheduled_time_ns'] - 1),
                ('actual_submit_time_ns', original['enqueued_time_ns'] - 1),
                ('actual_submit_time_ns', original['worker_response']['start_perf_ns'] + 1),
                ('submit_write_completed_time_ns', original['actual_submit_time_ns'] - 1),
                ('submit_write_completed_time_ns', original['completion_time_ns'] + 1),
                ('observed_worker_start_time_ns', original['worker_response']['start_perf_ns'] + 1),
                ('response_collector_dequeued_time_ns', original['completion_time_ns'] - 1),
                ('response_collector_processing_time_ns', original['response_collector_dequeued_time_ns'] - 1),
                ('enqueued_time_ns', float(original['enqueued_time_ns'])),
            ]
            for key, value in changes:
                with self.subTest(section=section, key=key, value=value):
                    raw = self.changed(section)
                    raw[section][0][key] = value
                    with self.assertRaises(ValueError):
                        audit.inspect_cohort(self.plan, raw)

    def test_write_completion_can_follow_worker_start(self):
        raw = self.changed('request_records')
        row = raw['request_records'][0]
        row['submit_write_completed_time_ns'] = row['worker_response']['start_perf_ns'] + 1
        self.assertLess(row['submit_write_completed_time_ns'], row['completion_time_ns'])
        self.assertEqual(audit.inspect_cohort(self.plan, raw)['measured_requests'], 50)


if __name__ == '__main__':
    unittest.main()
