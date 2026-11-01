import copy
from fractions import Fraction as F
import unittest
from unittest.mock import patch

from tests.test_twoFIFO_generated_wholeblock_evaluation_v1 import public_case, past, law, scoring, QUALITY
from tests.test_twoFIFO_multiowner_outer_envelope_v1 import fixture, concrete
from tools.predictor_validation_20261027 import twoFIFO_queue_envelope_companion_v1 as m


def coupled_inputs():
    kw = fixture(('A', 'B'))
    seed = m.history.from_snapshot(past(), cutoff_ns=50, forecast_origin_ns=1000,
        owner_producers={'text#0': 'A:stdout', 'text#1': 'B:stdout'})
    # Retain the actual past clocks/counters; rename the fixture owner roster.
    rename = {'text#0': 'A', 'text#1': 'B'}
    seed['owner_producers'] = {rename[o]: p for o, p in seed['owner_producers'].items()}
    for key in ('initial_sender_available_ns', 'initial_worker_read_release_ns', 'phase_watermarks',
                'worker_lifetimes', 'owner_issued', 'processed_issuance_watermarks'):
        seed[key] = {rename[o]: v for o, v in seed[key].items()}
    kw['requests'] = [dict(r, arrival_ns=r['arrival_ns']+1000) for r in kw['requests']]
    kw.update(window_start_ns=1000, window_end_ns=1010, drain_end_ns=2000)
    inputs = {k: kw[k] for k in m.ensemble.INPUT_FIELDS}
    return kw, seed, [dict(draw_id='predeclared', block_id='training', weight=F(1), inputs=inputs)]


class QueueEnvelopeCompanionTests(unittest.TestCase):
    def test_original_prediction_unchanged_same_compiler_and_idle_resident(self):
        args = (public_case(), 1000, past(), law(), scoring(), QUALITY)
        original = m.generated.forecast_case(*args)
        result = m.forecast_case(*args)
        self.assertEqual(result['prediction'], original)
        self.assertEqual(result['unchanged_point_prediction_sha256'], m.generated.digest(original))
        diagnostic = m.decode(result['queue_envelope'])
        self.assertEqual(diagnostic['bounded_world_probability_mass'], 1)
        self.assertEqual(len(diagnostic['draws']), 8)
        self.assertEqual(diagnostic['offered_requests'], 2)
        self.assertEqual(m.decode(original['ensemble'])['utility']['unknown_probability_mass'], 1)
        self.assertIsNone(diagnostic['full_U_bounds'])

    def test_paired_ambiguous_multiowner_worlds_contained_in_bounds(self):
        kw, seed, draws = coupled_inputs()
        boundaries = {k: kw[k] for k in ('window_start_ns', 'window_end_ns', 'drain_end_ns')}
        out = m.decode(m.evaluate_compiled(kw['requests'], draws, seed, **boundaries))
        result = out['draws'][0]['envelope']
        stations = m._stations(seed, kw['requests'], draws[0]['inputs'], F(1000))
        kw.update(stations)
        kw['settled_initial_state'] = seed
        for coordinates in ((0, 0, 0, 0), (7, F(3, 2), 13, 4), (20, 4, 20, 4)):
            world = concrete(kw, coordinates)
            for row in result['requests']:
                rid = row['request_id']
                lo, hi = row['RESULT_reader_ns_bounds']
                self.assertLessEqual(lo, world['RESULT_receipts'][rid])
                self.assertLessEqual(world['RESULT_receipts'][rid], hi)
        self.assertEqual(out['bounded_world_probability_mass'], 1)
        self.assertFalse(out['latent_point_selected'])

    def test_missing_donor_retains_exact_half_mass_and_full_cohort(self):
        donor = law()
        donor['vectors'] = [v for v in donor['vectors'] if v['training_block_id'] == 'training-block-1']
        out = m.decode(m.forecast_case(public_case(), 1000, past(), donor, scoring(), QUALITY)['queue_envelope'])
        self.assertEqual(out['bounded_world_probability_mass'], F(1, 2))
        self.assertEqual(out['unknown_envelope_probability_mass'], F(1, 2))
        for row in out['draws']:
            self.assertEqual(row['offered_requests'], 2)
            self.assertEqual(len(row['envelope']['requests']), 2)
        for row in out['fixed_drain_risk']:
            self.assertEqual(row['fixed_drain_incomplete_probability_bounds'], [0, F(1, 2)])

    def test_unsettled_past_and_unqualified_fit_do_not_gain_bounds(self):
        for options, snapshot in (({}, None), ({}, dict(past(), queued_runtime_events=[{}])),
                                  ({'unqualified_reason': 'unqualified full/fold fit'}, past())):
            result = m.forecast_case(public_case(), 1000, snapshot, law(), scoring(), QUALITY, **options)
            out = m.decode(result['queue_envelope'])
            self.assertEqual(out['unknown_envelope_probability_mass'], 1)
            self.assertEqual(len(out['draws']), 8)
            self.assertTrue(all(row['fixed_drain_incomplete_probability_bounds'] == [0, 1]
                                for row in out['fixed_drain_risk']))

    def test_fixed_drain_never_discards_late_cohort_members(self):
        kw, seed, draws = coupled_inputs()
        out = m.decode(m.evaluate_compiled(kw['requests'], draws, seed,
            window_start_ns=1000, window_end_ns=1010, drain_end_ns=1010))
        bound = out['draws'][0]['envelope']
        self.assertEqual(out['offered_requests'], 4)
        self.assertIsNone(bound['complete_cohort_P95_ns_bounds'])
        self.assertEqual(len(out['fixed_drain_risk']), 4)
        self.assertIsNotNone(bound['untruncated_model_complete_cohort_P95_ns_bounds'])

    def test_bad_draw_contract_is_error_not_unknown(self):
        kw, seed, draws = coupled_inputs()
        for bad in ([dict(draws[0], weight=F(1, 2))], [dict(draws[0], cap=2)],
                    [dict(draws[0], inputs={})]):
            with self.assertRaises(ValueError):
                m.evaluate_compiled(kw['requests'], bad, seed,
                    window_start_ns=1000, window_end_ns=1010, drain_end_ns=2000)
        with patch.object(m.generated.joint, 'compile_draw_v4', side_effect=ValueError('programming error')):
            with self.assertRaisesRegex(ValueError, 'programming error'):
                m.forecast_case(public_case(), 1000, past(), law(), scoring(), QUALITY)

    def test_lifetime_mismatch_and_missing_rate_keep_unknown_full_denominator(self):
        for variant in ('lifetime', 'rate', 'prior_id'):
            kw, seed, draws = coupled_inputs()
            if variant == 'lifetime':
                seed['owner_producers']['A'] = 'foreign:stdout'
            elif variant == 'rate':
                draws[0]['inputs']['rates'].pop(('A', 'decode', 1, 1))
            else:
                kw['requests'][0]['request_id'] = seed['prior_request_ids'][0]
            out = m.decode(m.evaluate_compiled(kw['requests'], draws, seed,
                window_start_ns=1000, window_end_ns=1010, drain_end_ns=2000))
            self.assertEqual(out['unknown_envelope_probability_mass'], 1)
            self.assertEqual(out['offered_requests'], 4)
            self.assertEqual(len(out['draws'][0]['envelope']['requests']), 4)


if __name__ == '__main__':
    unittest.main()
