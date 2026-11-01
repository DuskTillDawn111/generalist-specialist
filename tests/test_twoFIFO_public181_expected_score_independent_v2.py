"""Independent arithmetic counterexamples; no native qualification or GPU."""
import math
import unittest
from fractions import Fraction as F

from tools.predictor_validation_20261027 import twoFIFO_public181_expected_score_v2 as m


class IndependentExpectedScoreTests(unittest.TestCase):
    ids=['a','b','c','d']

    def prediction(self,cid,primitive=None):
        terms={term:1. for term in 'QTLPME'}
        worlds=[dict(draw_id='fixture',block_id='fixture',weight=F(1),
            status='GENERATED_WORLD',forecast=terms)]
        marginals={term:dict(domain={'CPU_ONLY':True},sources=[m.receipt(__file__)],
            distribution=m.ensemble.distribution(worlds,lambda f,t=term:f[t])) for term in terms}
        return dict(evaluation_protocol=m.EVALUATION_PROTOCOL,conditional_mean_hypothesis=m.HCPU,
            expected_score=m.marginal.combine(marginals,candidate_id=cid,
                candidate_domain={'CPU_ONLY':True},comparison_contract={'CPU_ONLY':True},
                weights={term:1/6 for term in terms}),
            resource_primitive_predictions={'startup_joules':{'predicted_mean':primitive}})

    def observed(self,terms,*,utility=True,primitive=None):
        weights={term:1/6 for term in 'QTLPME'}
        return dict(terms=terms,weights=weights,
            full_U=-sum(weights[t]*v for t,v in terms.items()) if utility else None,
            primitives={'startup_joules':primitive})

    def test_complete_terms_cannot_replace_missing_original_utility_label(self):
        pred={cid:self.prediction(cid) for cid in self.ids}
        actual={cid:self.observed({t:1. for t in 'QTLPME'},utility=False) for cid in self.ids}
        result=m.decode(m.evaluate_validation(pred,actual,candidate_ids=self.ids))
        self.assertEqual(result['pair_denominator'],6)
        self.assertEqual(result['known_pair_signs'],0)
        self.assertTrue(all(row['observed_U'] is None for row in result['candidates']))
        self.assertTrue(all(row['exact_sign_agreement'] is None for row in result['pairs']))

    def test_observed_original_floating_arithmetic_is_preserved_at_large_dynamic_range(self):
        terms={t:1. for t in 'QTLPME'};terms['Q']=1e16
        actual=self.observed(terms)
        replacement=-math.fsum(actual['weights'][t]*v for t,v in terms.items())
        self.assertNotEqual(actual['full_U'],replacement)
        result=m.decode(m.evaluate_validation({'a':self.prediction('a')},{'a':actual},candidate_ids=self.ids))
        self.assertEqual(result['candidates'][0]['observed_U'],actual['full_U'])
        self.assertFalse(result['source_evidence_authenticated'])

    def test_engineering_tolerance_uses_max_and_does_not_grant_accuracy(self):
        actual=self.observed({t:1. for t in 'QTLPME'},primitive=300.)
        result=m.decode(m.evaluate_validation({'a':self.prediction('a',primitive=350.)},
            {'a':actual},candidate_ids=self.ids))
        error=result['candidates'][0]['primitive_errors']['startup_joules']
        self.assertEqual(error['engineering_tolerance'],30.)
        self.assertFalse(error['within_engineering_tolerance'])
        for key in ('empirical_transfer_qualified','independent_accuracy_qualified',
                'full_U_accuracy_qualified','final_gain_eligible'):
            self.assertFalse(result[key])


if __name__=='__main__':unittest.main()
