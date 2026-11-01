"""Independent CPU counterexamples for serialized candidate ranking."""
import copy
from fractions import Fraction as F
import unittest
from unittest.mock import patch

from tests.test_twoFIFO_candidate_ranking_v1 import generated, supply_synthetic_resources, change_score
from tools.predictor_validation_20261027 import twoFIFO_candidate_ranking_v1 as m
from tools.predictor_validation_20261027.actual_worker_twoFIFO_reference_v5_2 import decode


class IndependentRankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = generated()

    def test_expected_utility_pairing_and_unknown_domain_override(self):
        # Deliberately crossing per-draw utilities: no stochastic dominance,
        # but the exact fixed 1/3,2/3 expectation still decides a complete pair.
        a = supply_synthetic_resources(self.document, 2.)
        b = supply_synthetic_resources(self.document, 1.)
        worse = supply_synthetic_resources(self.document, 4.)
        b['draws'][0] = worse['draws'][0]
        with patch.object(m.ensemble, 'forecast', side_effect=AssertionError('must not generate')):
            output = decode(m.rank_candidates({'a': a, 'b': b}))
        pair = output['paired_comparisons'][0]
        self.assertEqual(pair['strict_draw_relation'], 'NO_STRICT_DRAW_DOMINANCE')
        exact = sum(F(r['weight']) * F(r['delta_U']) for r in pair['comparison']['draws'])
        self.assertAlmostEqual(pair['comparison']['expected_delta_U'], float(exact))
        unknown = copy.deepcopy(a)
        unknown['draws'][0].update(status='UNKNOWN_WORLD', forecast=None,
            all_public_request_ids_retained=['r0', 'r1'], reason='missing physical support')
        all_candidates = decode(m.rank_candidates({'a': a, 'b': b, 'unresolved': unknown}))
        self.assertIsNone(all_candidates['winner_candidate_ids'])
        self.assertIsNone(all_candidates['complete_domain_ranking'])
        self.assertEqual(all_candidates['candidates'][2]['unknown_utility_probability_mass'], F(1, 3))

    def test_prephase_memory_and_startup_energy_each_have_one_application(self):
        doc = supply_synthetic_resources(self.document, 1.)
        plan = doc['score_comparison_contract']['scoring_plan']
        for draw in doc['draws']:
            prediction = draw['forecast']['scoring']['resource_prediction']
            prediction['premeasurement_owned_sampled_peak_mib'] *= 3
            prediction['startup_energy_joules'] = 3600.
            prediction['board_energy_joules'] = 2e-9 * plan['energy_reference_watts']
            expected_m = 3.
            expected_e = 1. + 1. / plan['energy_reference_watts']
            score = draw['forecast']['scoring']['original_score']
            score['terms'].update(M=expected_m, E=expected_e)
            score.update(cost=sum(score['terms'][t] / 6 for t in m.TERMS))
            score['full_U'] = -score['cost']
            draw['forecast']['scoring']['original_score_bridge']['score'] = copy.deepcopy(score)
        output = decode(m.rank_candidates({'single': doc}))
        self.assertEqual(output['winner_candidate_ids'], ['single'])
        for mutate in ('M_twice', 'startup_twice', 'missing_prephase'):
            bad = copy.deepcopy(doc)
            if mutate == 'missing_prephase':
                bad['draws'][0]['forecast']['scoring']['resource_prediction']['premeasurement_owned_sampled_peak_mib'] = None
            else:
                term = 'M' if mutate == 'M_twice' else 'E'
                def change(score):
                    score['terms'][term] *= 2
                    score['cost'] = sum(score['terms'][t] / 6 for t in m.TERMS)
                    score['full_U'] = -score['cost']
                change_score(bad, change)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                m.rank_candidates({'bad': bad})

    def test_stale_summary_never_overrides_unknown_per_draw_utilities(self):
        doc = copy.deepcopy(self.document)
        doc['utility'].update(known_probability_mass=1, unknown_probability_mass=0, expected_value=999999.)
        output = decode(m.rank_candidates({'stale_summary': doc}))
        self.assertEqual(output['winner_status'], 'UNKNOWN')
        self.assertEqual(output['candidates'][0]['unknown_utility_probability_mass'], 1)
        self.assertIsNone(output['candidates'][0]['complete_utility'])
        self.assertFalse(output['score_evidence_provenance_authenticated'])
        self.assertFalse(output['final_gain_eligible'])

    def test_incomplete_world_and_score_cannot_jointly_hide_one_offered_request(self):
        doc = supply_synthetic_resources(self.document, 1.)
        world = doc['draws'][0]['forecast']['world']
        rid = doc['public_requests'][0]['request_id']
        world['RESULT_receipts'].pop(rid)
        # A forged completion/score summary is insufficient: the whole offered
        # partition must still include the missing RESULT as unfinished.
        with self.assertRaises(ValueError):
            m.rank_candidates({'dropped': doc})


if __name__ == '__main__':
    unittest.main()
