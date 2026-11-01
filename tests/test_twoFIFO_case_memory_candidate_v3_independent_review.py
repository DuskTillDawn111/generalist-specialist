"""Independent CPU review: weighting, causal origin and generated evidence gates."""
import copy
import math
import unittest

import numpy as np

from tests.test_twoFIFO_case_memory_candidate_v3 import fixture, world_fixture
from tools.predictor_validation_20261027 import twoFIFO_case_memory_candidate_v3 as m


def generate(sidecar, world, requests, model, origin, domain):
    return m.generated_case_features(sidecar, world, requests, model=model,
        memory_origin=origin, observation_domain=domain)


class CaseMemoryIndependentReview(unittest.TestCase):
    def test_unequal_case_counts_and_unknown_rows_keep_original_attempt_weights(self):
        _, rows, _, _ = fixture()
        rows = rows[:32]
        extra = copy.deepcopy(rows[0]); extra['case_id'] = 'extra'
        extra['target_absolute_sampled_peak_mib'] += 123.
        rows.append(extra)
        rows[1]['features'] = None
        model = m.fit_rows(rows)
        self.assertEqual((model['total_training_rows'], model['known_training_rows']), (33, 32))
        self.assertFalse(model['full_training_denominator_supported'])
        known = [r for r in rows if m._row_known(r)]
        counts = {r['attempt_id']: sum(s['attempt_id'] == r['attempt_id'] for s in rows) for r in rows}
        weights = [1 / (8 * counts[r['attempt_id']]) for r in known]
        self.assertAlmostEqual(sum(weights), 1 - 1/40)
        x = np.array([r['features'] for r in known])
        y = np.array([(r['target_absolute_sampled_peak_mib'] - r['P0_mib'])/1024 for r in known])
        expected = np.linalg.solve(x.T @ np.diag(weights) @ x, x.T @ np.diag(weights) @ y)
        np.testing.assert_allclose(model['coefficients'], expected, rtol=1e-10, atol=1e-10)
        self.assertEqual([r['weight'] for r in model['row_weights']], weights)

    def test_unknown_training_rows_cannot_expand_public_or_observation_support(self):
        _, rows, _, _ = fixture(); rows = rows[:32]
        row = rows[0]; row['features'] = None
        row['public_keys'] = [['unseen-owner', 'unseen-task', 'unseen-input']]
        row['observation_domain'] = {'unseen_sampling_domain': True}
        model = m.fit_rows(rows)
        self.assertNotIn(tuple(row['public_keys'][0]), model['public_keys'])
        self.assertNotIn(row['observation_domain'], model['observation_domains'])
        self.assertEqual(model['training_rows'][0], row)

    def test_missing_completed_latent_mark_is_unknown_even_if_reader_and_processed_removed(self):
        sidecar, world, requests, model, origin, domain = world_fixture()
        world = m.generated.engine.decode(world)
        identity = next(r['identity'] for r in world['trace']
            if r.get('kind') == 'processed_notification' and r.get('phase') == 'prefill')
        world['trace'] = [r for r in world['trace'] if r.get('identity') != identity]
        out = generate(sidecar, world, requests, model, origin, domain)
        self.assertEqual(out['status'], 'UNKNOWN')
        self.assertIn('MISSING_COMPLETED_VECTOR_GENERATED_MARK', out['unknown_reasons'])
        self.assertIsNone(m.predict_generated_case(model, out)['prediction']['peak_memory_mib'])

    def test_processed_notice_requires_matching_reader_clock_and_latent_identity(self):
        for mode in ('missing_latent', 'missing_reader', 'reader_processing_mismatch'):
            with self.subTest(mode=mode):
                sidecar, world, requests, model, origin, domain = world_fixture()
                world = m.generated.engine.decode(world)
                notice = next(r for r in world['trace']
                    if r.get('kind') == 'processed_notification' and r.get('phase') == 'prefill')
                identity = notice['identity']
                if mode == 'reader_processing_mismatch':
                    next(r for r in world['trace'] if r.get('kind') == 'protocol_reader_available'
                        and r['identity'] == identity)['processing_ns'] += 1
                else:
                    kind = 'latent_occurrence' if mode == 'missing_latent' else 'protocol_reader_available'
                    world['trace'] = [r for r in world['trace']
                        if not (r.get('kind') == kind and r.get('identity') == identity)]
                out = generate(sidecar, world, requests, model, origin, domain)
                self.assertEqual(out['status'], 'UNKNOWN')
                self.assertIn('MISSING_GENERATED_LATENT_OR_PROCESSING_EVENT', out['unknown_reasons'])

    def test_processed_sidecar_body_and_chronology_must_agree(self):
        for mode in ('phase_body', 'future_occurrence', 'reader_after_processing', 'duplicate'):
            with self.subTest(mode=mode):
                sidecar, world, requests, model, origin, domain = world_fixture()
                world = m.generated.engine.decode(world)
                notice = next(r for r in world['trace']
                    if r.get('kind') == 'processed_notification' and r.get('phase') == 'prefill')
                if mode == 'phase_body': notice['phase'] = 'decode'
                elif mode == 'future_occurrence': notice['occurrence_ns'] = notice['available_ns'] + 1
                elif mode == 'reader_after_processing': notice['reader_receipt_ns'] = notice['available_ns'] + 1
                else: world['trace'].insert(world['trace'].index(notice) + 1, copy.deepcopy(notice))
                with self.assertRaises(ValueError): generate(sidecar, world, requests, model, origin, domain)

    def test_future_labels_are_recursively_rejected_across_all_three_inputs(self):
        for location in ('sidecar', 'world', 'public'):
            with self.subTest(location=location):
                sidecar, world, requests, model, origin, domain = world_fixture()
                target = {'sidecar': sidecar, 'world': world, 'public': requests[0]}[location]
                target['nested_metadata'] = {'nested_list': [{'owned_used_mib': 1000.}]}
                with self.assertRaisesRegex(ValueError, 'future execution/memory labels'):
                    generate(sidecar, world, requests, model, origin, domain)

    def test_donor_origin_cannot_mix_request_and_priming_blocks(self):
        sidecar, world, requests, model, origin, domain = world_fixture()
        origin['donor_attempt_id'] = 'training-block-2-shard-1'
        with self.assertRaisesRegex(ValueError, 'share whole-block latent'):
            generate(sidecar, world, requests, model, origin, domain)
        descriptor = model['startup_donors']['training-block-1-shard-1']['shape_descriptor']
        resident = dict(scope='resident_observed_before_origin', P0_mib=1000.,
            shape_descriptor=descriptor, last_observed_poll_end_ns=100)
        with self.assertRaisesRegex(ValueError, 'strictly before origin'):
            generate(sidecar, world, requests, model, resident, domain)

    def test_missing_prior_history_and_invalid_p0_never_become_known_peak(self):
        for invalid in (None, -1, True, float('nan'), float('inf')):
            with self.subTest(invalid=str(invalid)):
                sidecar, world, requests, model, origin, domain = world_fixture()
                model['startup_donors'][origin['donor_attempt_id']]['P0_mib'] = invalid
                out = generate(sidecar, world, requests, model, origin, domain)
                self.assertEqual(out['status'], 'UNKNOWN')
                self.assertIsNone(m.predict_generated_case(model, out)['prediction']['peak_memory_mib'])

    def test_actual_prefix_uses_poll_finish_strictly_before_origin(self):
        from tests.test_twoFIFO_owned_training_review_v3 import cohort_fixture
        plan, raw = cohort_fixture()
        raw['lifecycle_boundaries'] = [dict(boundary='startup_begin', time_ns=0)]
        raw['measurement_windows'] = [dict(case_id=c['case_id'], start_ns=c['epoch_ns'],
            end_ns=c['measurement_end_ns'], fixed_drain_deadline_ns=c['fixed_drain_deadline_ns'])
            for c in raw['cases']]
        plan['resident_owners'] = ['text#0', 'text#1', 'generalist_2#0', 'ocr#0']
        start = raw['measurement_windows'][0]['start_ns']
        raw['board_samples'] = [
            dict(time_ns=start-20, query_duration_ns=2, owned_used_mib=100.),
            dict(time_ns=start-1, query_duration_ns=2, owned_used_mib=900.),
            dict(time_ns=start-1, query_duration_ns=4, owned_used_mib=1000.),
            dict(time_ns=start+1, query_duration_ns=0, owned_used_mib=999999.)]
        native = dict(physical_execution_identity='0'*64, measured_case_resources=dict(cases=[
            dict(case_id=w['case_id'], startup_through_this_drain_owned_sampled_peak_mib=999999.)
            for w in raw['measurement_windows']]))
        rows = m.actual_rows_from_verified(raw, plan, {}, native)
        self.assertEqual(rows[0]['P0_mib'], 100.)
        self.assertEqual(rows[0]['target_absolute_sampled_peak_mib'], 999999.)
        self.assertTrue(rows[0]['actual_future_trajectory_used'])
        self.assertFalse(rows[0]['prospective_queue_memory_validation'])

    def test_finite_coefficients_cannot_return_nonfinite_known_prediction(self):
        for sign in (1, -1):
            with self.subTest(sign=sign):
                _, rows, _, _ = fixture(); model = m.fit_rows(rows[:32]); row = rows[32]
                model['coefficients'] = [sign*1e308, 0., 0., 0.]
                try:
                    out = m._predict(model, row['features'], row['P0_mib'], row['public_keys'],
                        row['observation_domain'], row['support_context'])
                except ValueError:
                    continue
                self.assertTrue(out['status'] == 'UNKNOWN' or
                    (math.isfinite(out['peak_memory_mib']) and math.isfinite(out['unclipped_absolute_prediction_mib'])),
                    'Arithmetic overflow must not become a known peak, even when max(P0,value) clips it')


if __name__ == '__main__': unittest.main()
