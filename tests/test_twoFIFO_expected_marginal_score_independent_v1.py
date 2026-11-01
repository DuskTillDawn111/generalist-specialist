"""Independent mathematical counterexamples; no native source qualification."""
import copy
from fractions import Fraction as F
import unittest
from tools.predictor_validation_20261027 import twoFIFO_expected_marginal_score_v1 as m
from tools.predictor_validation_20261027 import twoFIFO_forecast_ensemble_v1 as ensemble


class IndependentExpectedMarginals(unittest.TestCase):
    def combine(self,term_worlds):
        domain={'case':'a','protocol':'300+90+120'}
        marginals={}
        for term,values in term_worlds.items():
            rows=[dict(status='GENERATED_WORLD',weight=F(1,len(values)),forecast={'value':v}) for v in values]
            marginals[term]=dict(domain=domain,sources=[{'CPU_COUNTEREXAMPLE_ONLY':True}],
                distribution=ensemble.distribution(rows,lambda f:f['value']))
        return m.decode(m.combine(marginals,candidate_id='a',candidate_domain=domain,
            comparison_contract={'population':'181','protocol':'300+90+120'},weights={t:1/6 for t in m.TERMS}))

    def test_equal_marginals_cannot_distinguish_fixed_from_variable_full_utility(self):
        # T and E anticorrelated: every U=-1/6; correlated: U is0 or-1/3.
        anti=[(0,1),(1,0)];correlated=[(0,0),(1,1)]
        outputs=[]
        for worlds in (anti,correlated):
            terms={t:[0,0] for t in m.TERMS};terms['T']=[x for x,y in worlds];terms['E']=[y for x,y in worlds]
            outputs.append(self.combine(terms))
        self.assertEqual(outputs[0]['expected_U_under_supplied_marginals'],outputs[1]['expected_U_under_supplied_marginals'])
        self.assertNotEqual({-(x+y)/6 for x,y in anti},{-(x+y)/6 for x,y in correlated})
        self.assertTrue(all(o['positive_delta_probability'] is None and not o['joint_queue_resource_law_inferred'] for o in outputs))

    def test_expected_per_world_P95_must_not_pool_all_world_requests(self):
        # Each world contains20 identical latencies. Per-world P95s0 and100
        # average to50; pooled40-request P95 is100, a different statistic.
        terms={t:[0,0] for t in m.TERMS};terms['P']=[0,100]
        result=self.combine(terms)
        self.assertEqual(result['expected_terms']['P'],50)
        self.assertNotEqual(result['expected_terms']['P'],sorted([0]*20+[100]*20)[37])

    def test_unknown_quality_cannot_be_completed_by_other_five_marginals(self):
        terms={t:[1,1] for t in m.TERMS};complete=self.combine(terms)
        marginal=copy.deepcopy(complete['marginals']['Q'])
        marginal['distribution']=dict(known_probability_mass=0,unknown_probability_mass=1,
            expected_value=None,conditional_on_known_mean=None,conditional_mean_is_not_full_expectation=True)
        marginals=copy.deepcopy(complete['marginals']);marginals['Q']=marginal
        result=m.decode(m.combine(marginals,candidate_id='a',candidate_domain=complete['candidate_domain'],
            comparison_contract=complete['comparison_contract'],weights=complete['weights']))
        self.assertIsNone(result['expected_U_under_supplied_marginals'])
        self.assertFalse(result['UNKNOWN_mass_renormalized'])
        for term in 'TLPME':self.assertEqual(result['expected_terms'][term],1)


if __name__=='__main__':unittest.main()
