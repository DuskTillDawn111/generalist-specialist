import math
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_macro_joint_response_fit_v1 import document
from tools.predictor_validation_20261027 import macro_joint_response_second_order_v1 as module


class SecondOrderTests(unittest.TestCase):
    def dynamic_rows(self):
        doc=document()
        for index,request in enumerate(doc['requests']):
            macro=request['macros'][0];piece=macro['causal_covariate_path'][0]
            left,right=macro['left_ns'],macro['right_ns'];middle=left+(right-left)//3
            piece.update(right_ns=middle,duration_ns=middle-left)
            piece['state']['peer_same_visible_macro_phase_owner_count']=index%2
            macro['causal_covariate_path'].append(dict(left_ns=middle,right_ns=right,duration_ns=right-middle,
                state=dict(piece['state'],cutoff_processing_ns=middle,
                    peer_visible_macro_owner_count=(index+1)%3,
                    peer_same_visible_macro_phase_owner_count=(index//2)%2),
                each_state_uses_only_its_processed_prefix=True))
        return module.first.prepared([doc],module.first.FEATURES)

    def test_exact_hessian_matches_numerical_gradient_and_profile(self):
        rows=self.dynamic_rows();beta=np.array([.11,-.08]);step=1e-5
        value,gradient,hessian=module.differential(beta,rows)
        expected,expected_gradient=module.first.objective(beta,rows)
        self.assertAlmostEqual(value,expected,places=12)
        np.testing.assert_allclose(gradient,expected_gradient,atol=1e-12)
        numerical=np.column_stack([(module.differential(beta+np.eye(2)[i]*step,rows)[1]
            -module.differential(beta-np.eye(2)[i]*step,rows)[1])/(2*step) for i in range(2)])
        np.testing.assert_allclose(hessian,numerical,rtol=2e-7,atol=2e-7)
        np.testing.assert_allclose(hessian,hessian.T,atol=1e-14)

    def test_constant_path_positive_negative_solutions_and_independent_starts(self):
        for truth in (.3,-.3):
            rows=module.first.prepared([document(truth)],module.first.FEATURES[:1])
            info,beta,mu,variance=module.second_order_fit(rows,'visible_count')
            self.assertAlmostEqual(beta[0],truth,places=6)
            self.assertAlmostEqual(next(iter(mu.values())),math.log(10),places=6)
            self.assertTrue(info['second_order_qualification'])
            self.assertTrue(info['all_gradient_roots_agree'])
            self.assertEqual(len(info['gradient_root_attempts']),3)
            self.assertEqual(info['optimizer_converged'],info['original_BFGS_diagnostic']['optimizer_converged'])

    def test_stationary_maximum_saddle_and_false_success_not_qualified(self):
        for gradient,hessian in [(np.zeros(1),np.array([[-1.]])),
                (np.zeros(2),np.diag([1.,-1.])),(np.array([1e-5]),np.eye(1)),
                (np.zeros(1),np.zeros((1,1)))]:
            certificate=module.local_minimum_certificate(gradient,hessian)
            self.assertFalse(certificate['stationary_strict_local_minimum'])

    def test_nuisance_absorbed_beta_is_not_a_second_order_certificate(self):
        doc=document()
        for request in doc['requests']:
            macro=request['macros'][0]
            macro['pool_id']=str(macro['causal_covariate_path'][0]['state'][module.first.FEATURES[0]])
        rows=module.first.prepared([doc],module.first.FEATURES[:1])
        with self.assertRaisesRegex(ValueError,'positive-curvature stationary root'):
            module.second_order_fit(rows,'visible_count')

    def test_far_score_can_approach_zero_without_minimum(self):
        rows=module.first.prepared([document(.3)],module.first.FEATURES[:1])
        _,gradient,hessian=module.differential(np.array([-1e12]),rows)
        self.assertLess(np.linalg.norm(gradient),module.STATIONARITY_TOLERANCE)
        self.assertFalse(module.local_minimum_certificate(gradient,hessian)['stationary_strict_local_minimum'])

    def test_full_fit_failure_excludes_selection_even_with_good_folds(self):
        docs={}
        for index in range(3):
            doc=document()
            for request in doc['requests']:request['macros'][0]['capture_sha256']=str(index)
            docs[str(index)]=doc
        def fake_fit(rows,candidate):
            info,beta,mu,variance=module.first.fit(rows,candidate)
            info['second_order_qualification']=len(rows)!=18
            return info,beta,mu,variance
        with tempfile.TemporaryDirectory() as directory:
            prior=Path(directory)/'prior.json'
            prior.write_text(json.dumps(dict(schema=module.first.SCHEMA,validation_labels_read=False)))
            with patch.object(module.first,'verified_document',side_effect=lambda path:(docs[path],[])),\
                    patch.object(module,'second_order_fit',side_effect=fake_fit):
                result=module.run(list(docs),prior)
        self.assertIsNone(result['selected_candidate'])
        self.assertTrue(all(not c['leaveout_selection_eligible'] for c in result['candidates']))


if __name__=='__main__':unittest.main()
