"""Independent CPU review of actual training extraction and known v1 limits."""
import copy
import json
from pathlib import Path
import unittest

from tools.predictor_validation_20261027 import stage_response_features_v1 as f

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/predictor-device-progress-paired-v6-20261028-training01/training/natural'


class IndependentStageFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads((RUN / 'raw.json').read_text())
        cls.plan = json.loads((RUN / 'plan.json').read_text())
        cls.extracted = f.extract(RUN / 'raw.json')
        cls.row = cls.raw['request_records'][0]
        cls.job = next(j for j in cls.plan['jobs'] if j['id'] == cls.row['profile_job_id'])
        cls.events = f.ordered_events([cls.row])

    def test_actual_training_only_counts_and_no_fit(self):
        result = self.extracted
        self.assertEqual(len(result['stages']), 464)
        self.assertEqual(sum(s['stage_kind'] == 'language_forward' for s in result['stages']), 230)
        self.assertEqual(result['priming_rows_excluded'], 18)
        for key in ('fit_performed', 'validation_labels_read', 'GPU_started', 'physical_context_used_as_service'):
            self.assertIs(result[key], False)
        for ref in result['sources']:
            self.assertEqual(f.receipt(ref['path']), ref)

    def test_future_prefix_content_cannot_change_visible_state(self):
        cutoff = self.events[2]['collector_processing_time_ns']
        baseline = f.visible_prefix(self.events, cutoff, 'peer-test-owner', 'prefill')
        changed = copy.deepcopy(self.events)
        count = 0
        for event in changed:
            if event['collector_processing_time_ns'] > cutoff:
                event.update(owner='future-mutated-owner', request_id='future-request',
                             segment_id='future-phase', semantic_phase='prefill',
                             end_event_complete=True, input_sequence_length=10**12)
                count += 1
        self.assertGreater(count, 0)
        self.assertEqual(f.visible_prefix(changed, cutoff, 'peer-test-owner', 'prefill'), baseline)

    def test_every_emitted_state_contains_only_its_available_prefix(self):
        for stage in self.extracted['stages']:
            for state in [stage['runtime_features_at_begin'], *stage['processed_covariate_path']]:
                cutoff = state['cutoff_processing_ns']
                for peer in state['peer_visible_unconfirmed_phases']:
                    self.assertLessEqual(peer['begin_available_ns'], cutoff)
                    query = peer['last_processed_query']
                    if query is not None:
                        self.assertLessEqual(query['available_ns'], cutoff)
                self.assertFalse(state['future_response_fields_used'])
                self.assertFalse(state['unconfirmed_is_actual_unfinished'])

    def test_host_end_keeps_stream_unconfirmed_until_processed_true_query(self):
        end = next(e for e in self.events if e.get('event') == 'end' and e.get('segment_id') == 'prefill_0000')
        query = next(e for e in self.events if e.get('phase_id') == 'prefill_0000'
                     and e.get('end_event_complete') is True)
        before = f.visible_prefix(self.events, end['collector_processing_time_ns'], 'other-owner', 'prefill')
        after = f.visible_prefix(self.events, query['collector_processing_time_ns'], 'other-owner', 'prefill')
        before_ids = [p['phase_id'] for p in before['peer_visible_unconfirmed_phases']]
        after_ids = [p['phase_id'] for p in after['peer_visible_unconfirmed_phases']]
        self.assertIn('prefill_0000', before_ids)
        self.assertNotIn('prefill_0000', after_ids)

    def test_host_path_is_remaining_window_and_not_full_duration_window(self):
        ended_before = 0
        for stage in self.extracted['stages']:
            label = stage['response_labels']
            begin, end = label['host_observed_interval_ns']
            cutoff = stage['BEGIN_available_ns']
            self.assertGreater(cutoff, begin)
            self.assertEqual(label['host_call_wall_ns'], end - begin)
            self.assertEqual(label['remaining_host_wall_at_BEGIN_available_ns'], max(0, end - cutoff))
            window = stage['training_only_design_exposures']
            self.assertEqual(window['window_start_ns'], cutoff)
            self.assertEqual(window['window_end_ns'], max(cutoff, end))
            ended_before += label['host_call_ended_before_BEGIN_available']
        self.assertEqual(ended_before, 137)

    def test_actual_stream_label_cannot_use_host_end_path(self):
        forwards = [s for s in self.extracted['stages'] if s['stage_kind'] == 'language_forward']
        after_end = sum(s['response_labels']['current_stream_completion_interval_ns']['upper_ns'] >
                        s['response_labels']['host_observed_interval_ns'][1] for s in forwards)
        elapsed_larger = sum(s['response_labels']['recorded_current_stream_elapsed_ns'] >
                             s['response_labels']['host_call_wall_ns'] for s in forwards)
        self.assertEqual(after_end, 230)
        self.assertEqual(elapsed_larger, 221)
        for stage in forwards:
            labels = stage['response_labels']
            self.assertFalse(labels['host_and_stream_durations_additive'])
            self.assertFalse(labels['current_stream_interval_is_service'])
            self.assertNotEqual(labels['host_label_domain'], labels['recorded_stream_label_domain'])

    def test_necessary_rank_is_not_repetition_or_jacobian_qualification(self):
        checks = [c for c in self.extracted['design_checks']
                  if c['semantic_phase'] in ('prefill', 'decode') and c['candidate'] == 'visible_phase_aware']
        self.assertEqual(len(checks), 6)
        for check in checks:
            self.assertEqual(check['rank'], 2)
            self.assertTrue(check['full_column_rank'])
            self.assertEqual(check['whole_capture_blocks'], 1)
            self.assertFalse(check['minimum_initial_repetition_design_met'])
            self.assertFalse(check['nonlinear_integrated_hazard_Jacobian_checked'])

    def test_known_v1_query_identity_gap_is_reproducible(self):
        # This test documents acceptance, not a contract endorsement. A
        # qualified raw review must precede v1 use; v1 cannot replace it.
        for key, value in (('event_id', 'wrong-event'), ('segment_id', 'wrong-phase'),
                           ('forward_index', 999), ('stream_id', 999),
                           ('device_index', 999), ('host_query_begin_ns', 0)):
            with self.subTest(key=key):
                row = copy.deepcopy(self.row)
                query = next(e for e in row['execution_events'] if e.get('end_event_complete') is True)
                query[key] = value
                stages = f.phase_rows(row, self.job, f.ordered_events([row]), 'natural', 'CPU_MUTATION_ONLY')
                prefill = next(s for s in stages if s['semantic_phase'] == 'prefill')
                original = self.row['worker_response']['semantic_forward_observations']['device_progress']['records'][0]
                self.assertEqual(prefill['response_labels']['recorded_current_stream_elapsed_ns'],
                                 original['recorded_stream_elapsed_ms'] * 1e6)


if __name__ == '__main__':
    unittest.main()
