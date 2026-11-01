"""Independent mathematical and half-open envelope boundary checks."""
import unittest

import numpy as np

from tools.predictor_validation_20261027.handoff_candidate_v14 import (
    baseline_predict, ridge_fit, weights,
)
from tools.predictor_validation_20261027.handoff_witness_v14 import prequeued_witness


class IndependentHandoffReviewTests(unittest.TestCase):
    def test_capture_mass_and_ridge_normal_equation(self):
        matrix = np.asarray([[1., 1.], [1., 3.], [1., 8.], [1., 12.]])
        captures = ['a', 'a', 'a', 'b']
        target = np.asarray([300., 1700., 10000., 19000.])
        fitted = ridge_fit(matrix, target, captures)
        mass = np.asarray([1/6, 1/6, 1/6, 1/2])
        np.testing.assert_allclose(weights(captures), mass)
        z = (matrix-np.asarray([0., 6.]))/np.asarray([1., np.std(matrix[:, 1])])
        beta = np.asarray(fitted['coefficients'])
        gradient = z.T@(mass*(z@beta-np.log1p(target/1000))) + beta/4
        np.testing.assert_allclose(gradient, 0., atol=1e-14)
        # Ordinary row weighting must not masquerade as equal capture mass.
        ordinary = np.linalg.solve(z.T@z/4+np.eye(2)/4, z.T@np.log1p(target/1000)/4)
        self.assertGreater(float(np.max(np.abs(beta-ordinary))), .01)

    def test_prequeued_search_combines_all_three_boundaries(self):
        def kernel(start, ready, known=True):
            return dict(observed_start_ns=start, certified_ready_bound_ns=ready,
                        recorded_dependencies_known=known)
        rows = [kernel(9, 0), kernel(10, 10, False), kernel(11, 11),
                kernel(12, None), kernel(13, 10), kernel(20, 0)]
        self.assertIs(prequeued_witness(rows, [r['observed_start_ns'] for r in rows], 10, 20), rows[4])
        self.assertIsNone(prequeued_witness(rows, [r['observed_start_ns'] for r in rows], 14, 20))

    def test_pair_baseline_preserves_capture_balance_and_full_coverage(self):
        rows = [dict(capture=c, held_owner='a', next_owner='b') for c in ['x', 'x', 'x', 'y']]
        targets = np.asarray([0., 0., 0., 3000.])
        tests = [dict(held_owner='a', next_owner='b'), dict(held_owner='a', next_owner='c')]
        prediction, fallbacks = baseline_predict(rows, targets, tests, pair=True)
        np.testing.assert_allclose(prediction, [1000., 1000.], atol=1e-12)
        self.assertEqual(fallbacks, 1)


if __name__ == '__main__':
    unittest.main()
