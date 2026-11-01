"""CPU-only independent checks of reviewed host partition and aligned paths."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027 import stage_host_partition_v1 as h
from tools.predictor_validation_20261027 import stage_response_training_v2 as f
from tools.predictor_validation_20261027 import stage_response_features_v1 as old

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'artifacts/predictor-profile-training-evidence-v1-natural-20261028-review02/report.json'


class IndependentReviewedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(REVIEW.read_text())
        cls.raw = old.checked(cls.report['inputs']['raw.json'])
        cls.partition = h.extract(REVIEW)
        cls.training = f.extract(REVIEW)

    def test_actual_parent_outputs_independently_reproduced(self):
        self.assertEqual(self.partition, json.loads((REVIEW.parent / 'host-partition.json').read_text()))
        self.assertEqual(self.training, json.loads((REVIEW.parent / 'stage-training-v2.json').read_text()))
        self.assertEqual(self.partition['measured_requests'], 50)
        self.assertEqual(len(self.training['stages']), 464)

    def test_exact_handler_to_ready_coverage_by_independent_sweep(self):
        by_id = {r['request_id']: r for r in self.raw['request_records']}
        for partition in self.partition['requests']:
            raw = by_id[partition['request_id']]
            begin = raw['worker_response']['start_perf_ns']
            end = raw['previous_handler_next_request_ready']['perf_ns']
            pieces = partition['pieces']
            self.assertEqual(pieces[0]['left_ns'], begin)
            self.assertEqual(pieces[-1]['right_ns'], end)
            edges = {begin, end}
            for piece in pieces:
                edges.update((piece['left_ns'], piece['right_ns']))
                self.assertGreater(piece['duration_ns'], 0)
                self.assertEqual(piece['duration_ns'], piece['right_ns'] - piece['left_ns'])
                self.assertFalse(piece['is_device_service'])
            points = sorted(edges)
            for left, right in zip(points, points[1:]):
                count = sum(p['left_ns'] <= left and right <= p['right_ns'] for p in pieces)
                self.assertEqual(count, 1)
            self.assertEqual(sum(p['duration_ns'] for p in pieces), end - begin)

    def test_parent_native_interval_is_never_extra_additive_piece(self):
        saw_forward_external_gap = False
        for request in self.partition['requests']:
            self.assertFalse(request['independent_sync_and_device_wait_addition_permitted'])
            for piece in request['pieces']:
                self.assertNotEqual(piece['label'], 'native_inference')
                if piece['label'] == 'forward_external_gap_in_native_inference':
                    saw_forward_external_gap = True
                    self.assertIn('native_inference', piece['enclosing_phases'])
                    self.assertEqual(piece['kind'], 'host_gap')
        self.assertTrue(saw_forward_external_gap)

    def test_full_host_paths_reconstruct_full_wall_and_only_use_prefix(self):
        for stage in self.training['stages']:
            start, end = stage['response_labels']['host_observed_interval_ns']
            path = stage['full_host_wall_covariate_path']
            self.assertEqual(sum(p['duration_ns'] for p in path), end - start)
            if start < end:
                self.assertEqual(path[0]['left_ns'], start)
                self.assertEqual(path[-1]['right_ns'], end)
            for a, b in zip(path, path[1:]):
                self.assertEqual(a['right_ns'], b['left_ns'])
            for piece in path:
                self.assertEqual(piece['state']['cutoff_processing_ns'], piece['left_ns'])
                self.assertTrue(piece['each_state_uses_only_its_processed_prefix'])
                for peer in piece['state']['peer_visible_unconfirmed_phases']:
                    self.assertLessEqual(peer['begin_available_ns'], piece['left_ns'])
                    query = peer['last_processed_query']
                    if query:
                        self.assertLessEqual(query['available_ns'], piece['left_ns'])

    def test_past_and_future_path_boundary_causality(self):
        row = self.raw['request_records'][0]
        events = old.ordered_events([row])
        begin = next(e for e in events if e.get('semantic_phase') == 'prefill' and e.get('event') == 'start')
        end = next(e for e in events if e.get('segment_id') == begin['segment_id'] and e.get('event') == 'end')
        start, stop = begin['perf_ns'], end['perf_ns']
        before = f.causal_path(events, start, stop, 'other-owner', 'prefill')
        changed = copy.deepcopy(events)
        for event in changed:
            if event['collector_processing_time_ns'] >= stop:
                event.update(owner='future-owner', segment_id='future-phase', end_event_complete=True)
        after = f.causal_path(changed, start, stop, 'other-owner', 'prefill')
        self.assertEqual(before, after)
        self.assertEqual(f.causal_path(events, start, start, 'other-owner', 'prefill'), [])

    def test_late_begin_is_an_age_conditioning_requirement(self):
        count = 0
        for stage in self.training['stages']:
            start, end = stage['response_labels']['host_observed_interval_ns']
            notice = stage['BEGIN_available_ns']
            self.assertEqual(stage['elapsed_host_wall_at_BEGIN_notice_ns'], min(notice, end) - start)
            self.assertEqual(stage['full_host_label_clock_origin_ns'], start)
            self.assertFalse(stage['shape_available_to_parent_before_BEGIN_notice'])
            self.assertTrue(stage['prior_shape_template_qualification_required_for_prospective_use'])
            if end <= notice:
                count += 1
                self.assertEqual(stage['elapsed_host_wall_at_BEGIN_notice_ns'], end - start)
                self.assertFalse(stage['host_training_fit_diagnostic_eligible'])
        self.assertEqual(count, 137)

    def test_stream_fit_and_main_provider_remain_disabled(self):
        for stage in self.training['stages']:
            self.assertIsNone(stage['stream_elapsed_covariate_path'])
            self.assertFalse(stage['stream_response_fit_eligible'])
            self.assertFalse(stage['host_channel_is_effective_SM_service'])
            self.assertNotIn('processed_covariate_path', stage)
            self.assertNotIn('training_only_design_exposures', stage)
            self.assertIn('legacy_post_BEGIN_notice_path', stage)
        for key in ('fit_performed', 'model_selected', 'main_queue_provider_eligible',
                    'GPU_started', 'validation_labels_read', 'final_gain_eligible'):
            self.assertFalse(self.training[key])

    def test_review_schema_role_and_source_drift_rejected(self):
        for mode in ('schema', 'status', 'eligibility', 'source_drift'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                report = copy.deepcopy(self.report)
                if mode == 'schema':
                    report['schema'] = 'smoke-review'
                elif mode == 'status':
                    report['status'] = 'FAIL'
                elif mode == 'eligibility':
                    report['training_feature_extraction_eligible'] = False
                else:
                    path = Path(folder) / 'changed-source.py'
                    path.write_text('original')
                    report['sources'].append(old.receipt(path))
                    path.write_text('changed')
                path = Path(folder) / 'review.json'
                path.write_text(json.dumps(report))
                for extract in (h.extract, f.extract):
                    with self.assertRaises(ValueError):
                        extract(path)

    def test_partition_nested_crossings_and_identical_ranges_rejected(self):
        from tests.test_stage_host_partition_v1 import fixture, clock
        row = fixture()
        row['execution_events'][2].update(clock(155))
        row['execution_events'].sort(key=lambda x: x['perf_ns'])
        with self.assertRaisesRegex(ValueError, 'Crossing'):
            h.partition_request(row)
        row = fixture()
        extra = [dict(e, segment_id='duplicate-range') for e in row['execution_events']
                 if e['segment_id'] == 'prefill_0000']
        row['execution_events'].extend(extra)
        row['execution_events'].sort(key=lambda x: x['perf_ns'])
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            h.partition_request(row)


if __name__ == '__main__':
    unittest.main()
