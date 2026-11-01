"""Weighted objectives, frozen provider compatibility and UNKNOWN guards."""
import copy
import unittest
from unittest.mock import patch
import numpy as np

from tests import test_background_statistical as fixtures
from tools.predictor_validation_20261027 import duration_profiles as duration
from tools.predictor_validation_20261027.background_duration import fit_traces
from tools.predictor_validation_20261027.background_statistical import BackgroundStatisticalProvider


class DurationNumericsTests(unittest.TestCase):
    def test_single_context_alpha0_predicts_sample_mean_of_unequal_durations(self):
        durations=np.array([100.,200.,500.]);baseline=100.
        with patch.dict(duration.DEFINITION,alpha=0.):
            speed,_=duration.bounded_ridge((durations/baseline)[:,None],[1.])
        self.assertAlmostEqual(baseline/speed[0],np.mean(durations))
        self.assertNotAlmostEqual(baseline/speed[0],np.sum(durations**2)/np.sum(durations))

    def test_fixed_ridge_matches_closed_form_without_weight_renormalization(self):
        matrix=np.array([[1.],[2.],[5.]]);prior=.7;alpha=duration.DEFINITION['alpha']
        speed,diagnostics=duration.bounded_ridge(matrix,[prior])
        expected=(1+alpha*prior)/(np.mean(matrix)+alpha)
        self.assertAlmostEqual(speed[0],expected)
        self.assertAlmostEqual(diagnostics['row_weights']['sum'],1+1/2+1/5)
        self.assertNotAlmostEqual(100/speed[0],np.mean(matrix)*100)

    def test_multiple_context_weighted_ridge_matches_normal_equations(self):
        matrix=np.array([[.7,.3],[.1,2.],[3.,.5],[.2,.8]])
        prior=np.array([1.,.8]);weights=1/matrix.sum(axis=1);alpha=duration.DEFINITION['alpha']
        expected=np.linalg.solve(matrix.T@(weights[:,None]*matrix)/4+alpha/2*np.eye(2),
            matrix.T@weights/4+alpha/2*prior)
        actual,_=duration.bounded_ridge(matrix,prior)
        np.testing.assert_allclose(actual,expected,rtol=1e-10)

    def test_bounds_remain_active_for_extreme_observations(self):
        low,_=duration.bounded_ridge([[1000.]],[1.]);high,_=duration.bounded_ridge([[.001]],[1.])
        self.assertAlmostEqual(low[0],.05)
        self.assertLessEqual(high[0],8.)
        with patch.dict(duration.DEFINITION,alpha=0.):
            high,_=duration.bounded_ridge([[.001]],[1.])
        self.assertAlmostEqual(high[0],8.)

    def test_invalid_context_duration_weights_are_rejected(self):
        for matrix in ([[0.]],[[float('nan')]],[[float('inf')]],[[1.,-1.]]):
            with self.assertRaises(ValueError):duration.bounded_ridge(matrix,[1.]*len(matrix[0]))


class DurationProviderTests(unittest.TestCase):
    def setUp(self):
        self.traces=[]
        for repeat in range(2):
            rows=[[fixtures.request('s','s','text','text',100_000_000)],
                  [fixtures.request('g','g','generalist_2','image',100_000_000)],
                  [fixtures.request('g','g','generalist_2','asr',100_000_000)],
                  [fixtures.request('s','s','text','text',150_000_000),fixtures.request('g','g','generalist_2','image',150_000_000)]]
            for cohort in rows:self.traces.append({'id':str(len(self.traces)),'split':'training','execution_context_sha256':'v7-fixture','requests':cohort})

    def test_frozen_provider_reads_new_estimator_and_keeps_fulltask_unknown(self):
        document=fit_traces(self.traces)
        provider=BackgroundStatisticalProvider(document,document['execution_context_sha256'])
        self.assertEqual(document['definition']['base_numerics']['estimator'],'duration_weighted_relative_work_WLS.v1')
        self.assertEqual(document['coverage']['profiles'],3)
        aliases={'s':{'family':'text','supported_tasks':['text']},'g':{'family':'generalist_2','supported_tasks':['image','asr']}}
        def active(alias,task):
            profile=provider.lookup[(aliases[alias]['family'],task,'input')]
            return {'alias':alias,'task':task,'pool_id':'input','profile_id':profile['id'],'segment_id':'prefill'}
        s,g=active('s','text'),active('g','image')
        self.assertTrue(provider.service(s,{'s':s,'g':g},aliases,'mean')['known'])
        g=active('g','asr');value=provider.service(s,{'s':s,'g':g},aliases,'mean')
        self.assertFalse(value['known']);self.assertEqual(value['reason'],'UNSEEN_TARGET_FULL_FAMILY_TASK_MIXTURE')

    def test_nontraining_traces_are_rejected(self):
        traces=copy.deepcopy(self.traces);traces[0]['split']='validation'
        with self.assertRaisesRegex(ValueError,'training-only'):fit_traces(traces)


if __name__=='__main__':unittest.main()
