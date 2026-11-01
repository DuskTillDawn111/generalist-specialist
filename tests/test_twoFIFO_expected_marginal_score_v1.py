"""Numerical counterexamples for unconditional expected six-term utility."""
import copy
from fractions import Fraction as F
import unittest

from tools.predictor_validation_20261027 import twoFIFO_expected_marginal_score_v1 as m
from tools.predictor_validation_20261027 import twoFIFO_forecast_ensemble_v1 as original


class ExpectedMarginalTests(unittest.TestCase):
    def fixture(self):
        domain = dict(public_routed_case='candidate-a', protocol='300+90+120', objective='original')
        contract = dict(public_arrivals_inputs='same181', protocol='300+90+120', objective='original')
        rows = [dict(status='GENERATED_WORLD', weight=F(1,2), forecast=dict(Q=1,T=2,L=3,P=4,M=5,E=6)),
                dict(status='GENERATED_WORLD', weight=F(1,2), forecast=dict(Q=3,T=4,L=5,P=6,M=7,E=8))]
        marginals = {term: dict(domain=copy.deepcopy(domain), sources=[dict(CPU_TEST_ONLY=True)],
            distribution=original.distribution(rows, lambda f, term=term: f[term])) for term in m.TERMS}
        return domain, contract, rows, marginals

    def combine(self, domain, contract, marginals, candidate_id='candidate-a'):
        return m.decode(m.combine(marginals, candidate_id=candidate_id, candidate_domain=domain,
            comparison_contract=contract, weights={t:1/6 for t in m.TERMS}))

    def test_expected_score_equals_original_joint_mean_but_infers_no_joint_probability(self):
        domain, contract, rows, marginals = self.fixture()
        result = self.combine(domain, contract, marginals)
        expected = sum(float(row['weight'])*(-sum(row['forecast'].values())/6) for row in rows)
        self.assertAlmostEqual(result['expected_U_under_supplied_marginals'], expected)
        self.assertFalse(result['independence_assumed'])
        self.assertIsNone(result['positive_delta_probability'])
        self.assertFalse(result['source_authenticated'])
        self.assertFalse(result['full_U_qualified'])

    def test_partial_resource_mass_never_uses_its_conditional_mean_as_full_utility(self):
        domain, contract, rows, marginals = self.fixture()
        rows[1].update(status='UNKNOWN_WORLD', forecast=None)
        marginals['M']['distribution'] = original.distribution(rows, lambda f:f['M'])
        self.assertEqual(marginals['M']['distribution']['conditional_on_known_mean'], 5)
        result = self.combine(domain, contract, marginals)
        self.assertIsNone(result['expected_U_under_supplied_marginals'])
        self.assertEqual(result['marginals']['M']['distribution']['unknown_probability_mass'], F(1,2))
        self.assertIsNone(result['expected_terms']['M'])

    def test_noncommuting_memory_statistic_is_already_scored_per_donor(self):
        domain, contract, _, marginals = self.fixture()
        peaks = [(100, 0), (0, 100)]
        rows = [dict(status='GENERATED_WORLD',weight=F(1,2),forecast=dict(M=max(p)/100)) for p in peaks]
        marginals['M']['distribution'] = original.distribution(rows, lambda f:f['M'])
        result = self.combine(domain, contract, marginals)
        self.assertEqual(result['expected_terms']['M'], 1)
        self.assertNotEqual(result['expected_terms']['M'], max(50,50)/100)

    def test_two_resource_blocks_and_three_queue_blocks_need_no_fabricated_pairing(self):
        domain, contract, _, marginals = self.fixture()
        for term in ('T','L','P'):
            rows = [dict(status='GENERATED_WORLD', weight=F(1,3), forecast={term:i}) for i in (1,2,6)]
            marginals[term]['distribution'] = original.distribution(rows, lambda f,term=term:f[term])
        result = self.combine(domain, contract, marginals)
        self.assertEqual(result['expected_terms']['P'], 3)
        self.assertTrue(result['all_six_marginal_expectations_complete'])
        self.assertFalse(result['joint_queue_resource_law_inferred'])

    def test_mean_comparison_reconstructs_and_keeps_stability_and_risk_unknown(self):
        domain, contract, _, marginals = self.fixture()
        left = self.combine(domain, contract, marginals)
        other = dict(domain, public_routed_case='candidate-b')
        second = copy.deepcopy(marginals)
        for marginal in second.values(): marginal['domain'] = other
        second['E']['distribution']['expected_value'] -= 1
        second['E']['distribution']['conditional_on_known_mean'] -= 1
        right = self.combine(other, contract, second, 'candidate-b')
        result = m.decode(m.compare(left, right))
        self.assertAlmostEqual(result['expected_delta_U_under_supplied_marginals'], 1/6)
        self.assertTrue(result['candidate_better_in_expected_U'])
        self.assertIsNone(result['positive_delta_probability'])
        self.assertFalse(result['original_stability_evaluated'])
        changed = copy.deepcopy(right);changed['expected_U_under_supplied_marginals'] = 999
        with self.assertRaises(ValueError):m.compare(left, changed)
        changed = copy.deepcopy(right);changed['comparison_contract']['protocol'] = 'different'
        with self.assertRaises(ValueError):m.compare(left, changed)

    def test_mixed_domain_reweighting_and_fake_full_mass_are_contract_errors(self):
        domain, contract, _, marginals = self.fixture()
        for mode in ('domain','term','mass','conditional','nonfinite','negative'):
            changed = copy.deepcopy(marginals)
            if mode=='domain':changed['M']['domain'] = {'different':True}
            elif mode=='term':changed.pop('Q')
            elif mode=='mass':changed['E']['distribution']['known_probability_mass'] = F(1,2)
            elif mode=='conditional':changed['M']['distribution']['conditional_mean_is_not_full_expectation'] = False
            elif mode=='nonfinite':changed['P']['distribution']['expected_value'] = float('inf')
            else:changed['E']['distribution']['expected_value'] = -1
            with self.subTest(mode=mode), self.assertRaises(ValueError):self.combine(domain, contract, changed)
        with self.assertRaises(ValueError):m.combine(marginals,candidate_id='a',candidate_domain=domain,
            comparison_contract=contract,weights=dict(Q=0,T=.2,L=.2,P=.2,M=.2,E=.2))


if __name__=='__main__':unittest.main()
