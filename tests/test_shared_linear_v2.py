"""CPU reference correctness; no model loading, torch import or GPU access."""
import unittest

import numpy as np

from gsdeploy.backends.shared_linear import (
    CommonRectangle, SearchBudgetExceeded, find_max_common_contiguous,
    prepare, weight_identity,
)


def brute_rectangle(left, right):
    """Independent exhaustive small-array oracle, including deterministic ties."""
    matches = []
    for ar in range(left.shape[0]):
        for ac in range(left.shape[1]):
            for br in range(right.shape[0]):
                for bc in range(right.shape[1]):
                    for h in range(1, min(left.shape[0]-ar, right.shape[0]-br)+1):
                        for w in range(1, min(left.shape[1]-ac, right.shape[1]-bc)+1):
                            if left[ar:ar+h, ac:ac+w].tobytes() == right[br:br+h, bc:bc+w].tobytes():
                                rect = CommonRectangle((ar, ac), (br, bc), (h, w))
                                matches.append(((-h*w, ar, ac, br, bc, h, w), rect))
    return min(matches, key=lambda x: x[0])[1] if matches else None


class SharedLinearV2Tests(unittest.TestCase):
    def assert_matches_independent(self, left, right, x, z):
        plan = prepare(left, right)
        (actual_left, actual_right), report = plan.execute(x, z)
        tolerance = 2e-5 if left.dtype.itemsize == 4 else 1e-12
        np.testing.assert_allclose(actual_left, x @ left, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(actual_right, z @ right, rtol=tolerance, atol=tolerance)
        reconstructed = plan.reconstruct()
        self.assertEqual(tuple(map(weight_identity, reconstructed)), tuple(map(weight_identity, (left, right))))
        self.assertEqual(report['independent_macs'], report['executed_macs'])
        expected_saved = report['common_area'] * left.dtype.itemsize
        self.assertEqual(report['weight_storage_bytes_saved'], expected_saved)
        self.assertFalse(report['GPU_started'])
        self.assertFalse(report['deployed_model_backend'])
        self.assertFalse(report['benefit_claim'])
        return plan, report

    def test_exact_full_weight_shared_once_and_concat_split_unequal_batches(self):
        weight = np.arange(20, dtype=np.float64).reshape(4, 5)
        x = np.arange(12, dtype=np.float64).reshape(3, 4)
        z = np.arange(4, dtype=np.float64).reshape(1, 4)
        plan, report = self.assert_matches_independent(weight, weight.copy(), x, z)
        self.assertEqual(plan.search.status, 'FULL_WEIGHT_MATCH')
        self.assertEqual(report['executed_gemm_calls'], 1)
        self.assertEqual(report['residual_gemm_calls'], 0)
        self.assertEqual(report['packed_weight_bytes'], weight.nbytes)
        self.assertEqual(report['concat_copy_bytes'], (3+1)*4*8)
        self.assertEqual(report['split_copy_bytes'], 0)

    def test_shifted_inner_common_block_all_four_residuals_different_widths(self):
        left = np.arange(30, dtype=np.float64).reshape(5, 6)+10
        right = np.arange(56, dtype=np.float64).reshape(7, 8)+1000
        right[2:4, 3:6] = left[1:3, 2:5]
        rng = np.random.default_rng(12)
        plan, report = self.assert_matches_independent(left, right, rng.normal(size=(2, 5)), rng.normal(size=(3, 7)))
        self.assertEqual(plan.search.rectangle, CommonRectangle((1, 2), (2, 3), (2, 3)))
        self.assertEqual(tuple(map(len, plan.residuals)), (4, 4))
        self.assertEqual(report['executed_gemm_calls'], 9)
        self.assertGreater(report['output_accumulation_elements'], 2*6+3*8)
        self.assertGreater(report['largest_residual_product_bytes'], 0)

    def test_no_common_block_falls_back_to_two_full_residuals(self):
        left = np.array([[1, 2], [3, 4]], dtype=np.float32)
        right = np.array([[9, 10, 11]], dtype=np.float32)
        _, report = self.assert_matches_independent(left, right, np.ones((2, 2), dtype=np.float32), np.ones((3, 1), dtype=np.float32))
        self.assertEqual(report['search_status'], 'NO_COMMON_BLOCK')
        self.assertEqual(report['common_area'], 0)
        self.assertEqual(report['executed_gemm_calls'], 2)
        self.assertEqual(report['concat_copy_bytes'], 0)

    def test_histogram_search_matches_independent_exhaustive_oracle(self):
        rng = np.random.default_rng(2701)
        for _ in range(35):
            a = rng.integers(0, 4, size=tuple(rng.integers(1, 5, size=2))).astype(np.float64)
            b = rng.integers(0, 4, size=tuple(rng.integers(1, 5, size=2))).astype(np.float64)
            self.assertEqual(find_max_common_contiguous(a, b).rectangle, brute_rectangle(a, b))

    def test_tie_choice_and_noncontiguous_input_layout(self):
        a = np.array([[1, 2, 9], [3, 4, 9]], dtype=np.float64)
        b = np.array([[1, 2], [3, 4], [1, 2], [3, 4]], dtype=np.float64)
        self.assertEqual(find_max_common_contiguous(a, b).rectangle, CommonRectangle((0, 0), (0, 0), (2, 2)))
        self.assertEqual(weight_identity(a), weight_identity(np.asfortranarray(a)))
        x = np.arange(20, dtype=np.float64).reshape(2, 10)[:, ::5]
        self.assertFalse(x.flags.c_contiguous)
        self.assert_matches_independent(a, b, x, np.ones((3, 4), dtype=np.float64))

    def test_hash_binds_dtype_shape_and_signed_zero_exact_bytes(self):
        a = np.array([[0.]], dtype=np.float64)
        b = np.array([[-0.]], dtype=np.float64)
        self.assertNotEqual(weight_identity(a), weight_identity(b))
        self.assertEqual(find_max_common_contiguous(a, b).status, 'NO_COMMON_BLOCK')
        self.assertNotEqual(weight_identity(a), weight_identity(a.astype(np.float32)))
        self.assertNotEqual(weight_identity(np.zeros((1, 2))), weight_identity(np.zeros((2, 1))))

    def test_budget_is_explicit_refusal_and_full_match_fast_path(self):
        a, b = np.ones((3, 3)), np.ones((3, 3))*2
        with self.assertRaisesRegex(SearchBudgetExceeded, '81.*cap=80.*no partial'):
            find_max_common_contiguous(a, b, max_search_cells=80)
        self.assertEqual(find_max_common_contiguous(a, a, max_search_cells=1).status, 'FULL_WEIGHT_MATCH')
        for bad in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                find_max_common_contiguous(a, b, max_search_cells=bad)

    def test_owned_copies_do_not_follow_caller_weight_mutation(self):
        a = np.array([[1., 2.], [3., 4.]])
        original = a.copy(); plan = prepare(a, a)
        a[:] = 99
        (one, two), _ = plan.execute(np.ones((1, 2)), np.ones((2, 2)))
        np.testing.assert_array_equal(one, np.ones((1, 2)) @ original)
        np.testing.assert_array_equal(two, np.ones((2, 2)) @ original)
        self.assertFalse(plan.common_weight.flags.writeable)

    def test_tampered_packed_storage_is_detected(self):
        plan = prepare(np.eye(2), np.eye(2))
        plan.common_weight.flags.writeable = True
        plan.common_weight[0, 0] = 12
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            plan.execute(np.ones((1, 2)), np.ones((1, 2)))

    def test_reject_nonfinite_dtype_mismatch_and_bad_dimensions(self):
        a = np.eye(2, dtype=np.float64)
        for bad in (np.ones((2, 2), dtype=np.int64), np.array([[np.nan]]), np.array([[np.inf]]),
                    np.array([1., 2.]), np.empty((0, 2))):
            with self.assertRaises(ValueError):
                prepare(a, bad)
        with self.assertRaisesRegex(ValueError, 'identical weight dtype'):
            prepare(a, a.astype(np.float32))
        plan = prepare(a, a)
        for bad in (np.ones((1, 3)), np.ones((1, 2), dtype=np.float32)):
            with self.assertRaisesRegex(ValueError, 'Input dtype/width'):
                plan.execute(bad, np.ones((1, 2)))

    def test_float32_randomized_shifted_block_outputs(self):
        rng = np.random.default_rng(483)
        for _ in range(12):
            a = rng.normal(size=(5, 6)).astype(np.float32)
            b = rng.normal(size=(6, 4)).astype(np.float32)
            b[2:5, 1:3] = a[1:4, 3:5]
            self.assert_matches_independent(a, b, rng.normal(size=(3, 5)).astype(np.float32),
                                           rng.normal(size=(2, 6)).astype(np.float32))


if __name__ == '__main__':
    unittest.main()
