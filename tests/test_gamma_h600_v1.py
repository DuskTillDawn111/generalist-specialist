import math
import unittest
import copy
import tempfile
from pathlib import Path
import numpy as np
from tools.predictor_validation_20261027.gamma_profiles_h600_v1 import bounded_ridge
from tools.predictor_validation_20261027 import gamma_h600_v1 as workflow
from tools.predictor_validation_20261027.native_flow import receipt
from tools.predictor_validation_20261027.background_statistical import BackgroundStatisticalProvider
from tests.test_duration_profiles import DurationProviderTests


class GammaNumericsTests(unittest.TestCase):
    def test_single_context_alpha_zero_targets_arithmetic_duration_mean(self):
        speed,_=bounded_ridge([[1.],[2.],[5.]],[1.],alpha=0.)
        self.assertAlmostEqual(100./speed[0],800./3,places=5)

    def test_ridge_solution_matches_quadratic_stationarity(self):
        mean=8./3;alpha=.1;prior=.7
        expected=2./(mean-2*alpha*prior+math.sqrt((mean-2*alpha*prior)**2+8*alpha))
        speed,_=bounded_ridge([[1.],[2.],[5.]],[prior],alpha=alpha)
        self.assertAlmostEqual(speed[0],expected,places=7)

    def test_multiple_context_kkt_at_interior_solution(self):
        matrix=np.array([[.7,.3],[.1,2.],[3.,.5],[.2,.8]])
        prior=np.array([1.,.8]);alpha=.1
        speed,_=bounded_ridge(matrix,prior,alpha=alpha)
        gradient=matrix.T@(1.-1./(matrix@speed))/4+2*alpha*(speed-prior)/2
        np.testing.assert_allclose(gradient,0.,atol=2e-7)

    def test_original_box_and_invalid_inputs(self):
        speed,_=bounded_ridge([[1000.]],[1.],alpha=0.)
        self.assertEqual(speed[0],.05)
        for matrix in ([[0.]],[[float('nan')]],[[1.,-1.]]):
            with self.assertRaises(ValueError):bounded_ridge(matrix,[1.]*len(matrix[0]))


class GammaEvidenceTests(unittest.TestCase):
    def setUp(self):
        f=DurationProviderTests();f.setUp();self.traces=f.traces
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        path=Path(self.tmp.name)/'source.json';path.write_text('{}')
        self.ref=receipt(path)
        self.bundle=dict(protocol_receipt=self.ref,run_protocol_receipt=self.ref)
        self.group=dict(group_id='fixture',role='old_profile_training',raw=self.ref,
            direct_source=self.ref,transitive_sources=[self.ref],traces=self.traces)

    def test_provider_compatibility_and_reproducible_numerical_fit(self):
        a=workflow.fit_groups(self.bundle,[self.group],self.ref,None)
        b=workflow.fit_groups(self.bundle,[self.group],self.ref,None)
        self.assertEqual(a,b)
        provider=BackgroundStatisticalProvider(a,'v7-fixture')
        self.assertEqual(len(provider.lookup),3)
        self.assertEqual(a['definition']['estimator'],'gamma_reciprocal_mean.v1')
        self.assertEqual(a['gamma_training_evidence']['validation_sources_used'],[])

    def test_training_role_and_immutable_receipt_required(self):
        group=copy.deepcopy(self.group);group['traces'][0]['split']='validation'
        with self.assertRaisesRegex(ValueError,'Training-only'):workflow.fit_groups(self.bundle,[group],self.ref,None)
        Path(self.ref['path']).write_text('{"changed":true}')
        with self.assertRaises(ValueError):workflow.fit_groups(self.bundle,[self.group],self.ref,None)


if __name__=='__main__':unittest.main()
