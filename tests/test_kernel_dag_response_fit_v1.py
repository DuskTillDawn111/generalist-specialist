"""Check low-memory certificates against the original dense small problem."""
import unittest
from unittest.mock import patch
import numpy as np

from tools.predictor_validation_20261027 import kernel_dag_response_fit_v1 as m
from tools.predictor_validation_20261027.macro_projection_contrast_training_v1 import fit as dense_fit


def rows_fixture():
    rng=np.random.default_rng(412)
    rows=[]
    for i in range(36):
        vector=rng.uniform(0,1,6);group=('a',) if i%2 else ('b',)
        response=float(np.exp((1 if i%2 else 2)+vector@np.arange(6)/10+rng.normal(0,.15)))
        rows.append(dict(group=group,capture='c'+str(i%3),durations=np.asarray([response]),
            matrix=vector[None,:],ordered_durations=np.asarray([response]),ordered_matrix=vector[None,:]))
    return rows


class KernelDagLowMemoryFitTests(unittest.TestCase):
    def test_certificate_bounds_actual_dense_singular_values(self):
        rng=np.random.default_rng(11)
        for n,g,w in [(30,3,6),(100,20,6),(9,4,6),(50,3,1)]:
            rows=[dict(group=i%g) for i in range(n)];d=rng.normal(size=(n,w))
            z=np.asarray([[int(i%g==j) for j in range(g)] for i in range(n)])
            singular=np.linalg.svd(np.column_stack((z,d)),compute_uv=False)
            cert=m.nuisance_certificate(rows,d)
            self.assertLessEqual(cert['full_jacobian_min_singular_lower_bound'],singular[-1]+1e-12)
            self.assertGreaterEqual(cert['full_jacobian_max_singular_upper_bound'],singular[0]-1e-12)
            if cert['full_column_rank']:
                self.assertEqual(np.linalg.matrix_rank(np.column_stack((z,d))),g+w)

    def test_unidentified_group_constants_never_qualify(self):
        rows=[dict(group=i%4) for i in range(40)]
        d=np.asarray([[i%4,2*(i%4),0] for i in range(40)],dtype=float)
        cert=m.nuisance_certificate(rows,d)
        self.assertFalse(cert['full_column_rank']);self.assertEqual(cert['projected_derivative_rank'],0)

    def test_large_group_mean_cannot_turn_ill_conditioned_dense_design_into_pass(self):
        rng=np.random.default_rng(1);rows=[dict(group=i%3) for i in range(36)]
        d=np.full((36,2),1e16)+rng.normal(size=(36,2))*4
        self.assertFalse(m.nuisance_certificate(rows,d)['full_column_rank'])

    def test_linear_memory_no_dense_group_dummy_allocation(self):
        rows=[dict(group=i) for i in range(1000)]
        d=np.ones((1000,6))
        with patch.object(m.np,'column_stack',side_effect=AssertionError('no dense group design')):
            cert=m.nuisance_certificate(rows,d)
        self.assertFalse(cert['dense_nuisance_matrix_allocated']);self.assertFalse(cert['full_column_rank'])

    def test_original_likelihood_solver_roots_and_coefficients_are_preserved(self):
        rows=rows_fixture();features=list(range(6))
        old,beta_old,mu_old,var_old=dense_fit(rows,features)
        new,beta_new,mu_new,var_new=m.fit(rows,features)
        np.testing.assert_allclose(beta_old,beta_new,rtol=1e-10,atol=1e-10)
        self.assertAlmostEqual(old['nll'],new['nll'],places=11)
        self.assertEqual(mu_old,mu_new);self.assertEqual(var_old,var_new)
        self.assertEqual(old['trust_exact_attempts'],new['trust_exact_attempts'])
        self.assertEqual(old['gradient_root_attempts'],new['gradient_root_attempts'])
        self.assertEqual(old['numerical_qualification'],new['numerical_qualification'])


if __name__=='__main__':unittest.main()
