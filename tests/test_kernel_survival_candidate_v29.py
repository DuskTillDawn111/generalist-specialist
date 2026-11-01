import unittest

from tools.predictor_validation_20261027.kernel_survival_candidate_v29 import (
    conditional, fit, km_fit, label_at, predict, queue_bucket, score)


class KernelSurvivalTests(unittest.TestCase):
    def test_KM_censor_remains_at_risk_before_its_censor_time(self):
        model=km_fit([(2,True),(4,False),(6,True)])
        self.assertEqual(model['n'],3)
        self.assertEqual(model['censored'],1)
        self.assertAlmostEqual(model['survival'][0],2/3)
        self.assertAlmostEqual(model['survival'][1],2/3)
        self.assertEqual(model['survival'][2],0)

    def test_conditional_survival_reweights_tail_after_survived_age(self):
        model=km_fit([(2,True),(4,True),(10,True),(20,True)])
        estimate=conditional(model,4)
        self.assertEqual(estimate['quantiles_ns']['0.5'],6)
        self.assertEqual(estimate['quantiles_ns']['0.95'],16)
        self.assertEqual(conditional(model,20)['status'],'UNKNOWN_NO_SURVIVING_TRAINING_SUPPORT')

    def test_right_censored_tail_is_UNKNOWN_not_fabricated_completion(self):
        estimate=conditional(km_fit([(2,True),(10,False)]),0)
        self.assertEqual(estimate['quantiles_ns']['0.5'],2)
        self.assertIsNone(estimate['quantiles_ns']['0.95'])

    def test_signed_early_endpoint_is_state_and_late_endpoint_censors(self):
        self.assertIsNone(label_at(5,10,30))
        self.assertEqual(label_at(50,10,30),{'duration_ns':20,'event_observed':False})
        self.assertEqual(label_at(30,10,30),{'duration_ns':20,'event_observed':True})

    def test_censored_coverage_is_bounded_without_dropping_rows(self):
        rows=[{'duration_ns':2,'event_observed':True},
              {'duration_ns':4,'event_observed':False},
              {'duration_ns':20,'event_observed':False}]
        estimates=[{'quantiles_ns':{'0.95':10},'survival':{}} for _ in rows]
        actual=score(rows,estimates)['quantiles']['0.95']
        self.assertEqual(actual['supported_n'],3)
        self.assertEqual(actual['coverage_fraction_bounds'],[1/3,2/3])
        self.assertEqual(actual['uncensored_only_n'],1)

    def test_queue_is_observed_stratum_not_share_or_gate(self):
        self.assertEqual(queue_bucket(0),'0')
        self.assertEqual(queue_bucket(9),'8-31')
        with self.assertRaises(ValueError):queue_bucket(-1)
        row=dict(target='END_if_not_ended',signature=('x','decode','sig'),
                 landmark='next_observed_issue',queue_bucket='0',already_started=True,
                 duration_ns=12,event_observed=True,survived_age_ns=99)
        models=fit([row]*5,'landmark_queue_KM')
        self.assertEqual(predict(models,row,'landmark_queue_KM')['quantiles_ns']['0.5'],12)
        different=dict(row,queue_bucket='1')
        self.assertEqual(predict(models,different,'landmark_queue_KM')['status'],'UNKNOWN_UNSEEN_OR_SPARSE_GROUP')


if __name__=='__main__':unittest.main()
