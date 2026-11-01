"""Independent contract review for the queue-envelope companion."""
import copy
from fractions import Fraction as F
import unittest

from tests.test_twoFIFO_queue_envelope_companion_v1 import coupled_inputs
from tools.predictor_validation_20261027 import twoFIFO_queue_envelope_companion_v1 as m


class QueueEnvelopeIndependentReview(unittest.TestCase):
    def test_idle_settled_producer_is_projected_but_collector_clock_survives(self):
        kw, seed, draws = coupled_inputs()
        # Add a settled, idle owner whose reader station must be removed. Its
        # global collector clock must remain in the projected protocol state.
        seed['owner_producers']['idle'] = 'idle:stdout'
        seed['initial_sender_available_ns']['idle'] = 40
        seed['initial_worker_read_release_ns']['idle'] = 40
        seed['phase_watermarks']['idle'] = 40
        seed['owner_issued']['idle'] = 1
        seed['processed_issuance_watermarks']['idle'] = 0
        seed['worker_lifetimes']['idle'] = {'pid': 99, 'spec_sha256': 'c' * 64}
        seed['producer_next']['idle:stdout'] = 1
        seed['producer_last_reader']['idle:stdout'] = 40
        seed['sequence'] += 1
        seed['initial_protocol_state']['reader_free']['idle:stdout'] = 40
        seed['prior_request_ids'].append('idle-prior')
        stations = m._stations(seed, kw['requests'], draws[0]['inputs'], F(1000))
        self.assertEqual(set(stations['initial_sender_available_ns']), {'A', 'B'})
        self.assertEqual(set(stations['initial_protocol_state']['reader_free']), {'A:stdout', 'B:stdout'})
        self.assertEqual(stations['initial_protocol_state']['collector_free'], seed['initial_protocol_state']['collector_free'])
        self.assertEqual(stations['initial_protocol_state']['last_collector_arrival'], seed['initial_protocol_state']['last_collector_arrival'])

    def test_seed_owner_lifetime_and_prior_identity_are_binding(self):
        kw, seed, draws = coupled_inputs()
        for mutation in ('pid', 'spec', 'sequence_scope', 'prior', 'origin', 'collector'):
            bad = copy.deepcopy(seed)
            if mutation == 'pid': bad['worker_lifetimes']['A']['pid'] = -1
            elif mutation == 'spec': bad['worker_lifetimes']['A']['spec_sha256'] = 'bad'
            elif mutation == 'sequence_scope': bad['sequence_scope'] = 'wrong'
            elif mutation == 'prior': bad['prior_request_ids'] = []
            elif mutation == 'origin': bad['forecast_origin_ns'] = 999
            else: bad['initial_protocol_state']['collector_free'] = 5000
            with self.subTest(mutation=mutation):
                out = m.decode(m.evaluate_compiled(kw['requests'], draws, bad,
                    window_start_ns=1000, window_end_ns=1010, drain_end_ns=2000))
                self.assertEqual(out['unknown_envelope_probability_mass'], 1)
                self.assertEqual(out['fixed_drain_risk'][0]['fixed_drain_incomplete_probability_bounds'], [0, 1])

    def test_envelope_never_upgrades_point_or_utility_and_retains_unknown_mass(self):
        kw, seed, draws = coupled_inputs()
        unknown = copy.deepcopy(draws[0]); unknown['weight'] = F(1, 2)
        unknown['draw_id'] = 'unknown'; unknown['inputs'] = m.ensemble.UnknownForecastInputs('no rate evidence')
        draws[0]['weight'] = F(1, 2); draws = [draws[0], unknown]
        out = m.decode(m.evaluate_compiled(kw['requests'], draws, seed,
            window_start_ns=1000, window_end_ns=1010, drain_end_ns=2000))
        self.assertEqual(out['bounded_world_probability_mass'], F(1, 2))
        self.assertEqual(out['unknown_envelope_probability_mass'], F(1, 2))
        self.assertIsNone(out['full_U_bounds']); self.assertIsNone(out['resource_bounds'])
        self.assertFalse(out['latent_point_selected']); self.assertFalse(out['final_gain_eligible'])
        self.assertTrue(out['all_original_draws_and_probability_mass_retained'])
        self.assertTrue(out['all_offered_requests_retained'])

    def test_finite_rate_envelope_bounds_concrete_trajectory_and_preserves_draw_identity(self):
        kw, seed, draws = coupled_inputs()
        original = m.decode(m.evaluate_compiled(kw['requests'], draws, seed,
            window_start_ns=1000, window_end_ns=1010, drain_end_ns=2000))
        row = original['draws'][0]
        self.assertEqual((row['draw_id'], row['block_id'], row['weight']), ('predeclared', 'training', F(1)))
        self.assertTrue(all(item['START_ns_bounds'][0] <= item['START_ns_bounds'][1]
                            <= item['RESULT_reader_ns_bounds'][1] for item in row['envelope']['requests']))
        self.assertTrue(all(item['fixed_drain_incomplete_indicator_bounds'] in ([0,0],[0,1],[1,1])
                            for item in row['envelope']['requests']))
        # Public identity/order is immutable and all requests remain in the risk denominator.
        self.assertEqual([r['request_id'] for r in original['public_requests']],
                         [r['request_id'] for r in kw['requests']])
        self.assertEqual(len(original['fixed_drain_risk']), len(kw['requests']))

    def test_invalid_input_is_error_while_scientific_gap_is_unknown(self):
        kw, seed, draws = coupled_inputs()
        with self.assertRaises(ValueError):
            m.evaluate_compiled(kw['requests'], draws, seed,
                window_start_ns=1000, window_end_ns=1000, drain_end_ns=2000)
        bad = copy.deepcopy(draws)
        bad[0]['inputs']['rates'] = {}
        out = m.decode(m.evaluate_compiled(kw['requests'], bad, seed,
            window_start_ns=1000, window_end_ns=1010, drain_end_ns=2000))
        self.assertEqual(out['unknown_envelope_probability_mass'], 1)


if __name__ == '__main__': unittest.main()
