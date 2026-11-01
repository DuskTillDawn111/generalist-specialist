from copy import deepcopy
import unittest

from tools.predictor_validation_20261027.operator_smoke_review_v2 import validate_operator_rows


class OperatorKernelReviewTests(unittest.TestCase):
    def observed(self):
        tensor = lambda shape: dict(shape=shape, dtype='torch.float16', element_size_bytes=2,
                                    logical_bytes=__import__('math').prod(shape)*2)
        row = dict(id='affine_0000000', module='layer.proj', operator_kind='linear_declared',
                   implementation='torch.nn.Linear', in_features=3, out_features=4,
                   input=tensor([2, 3]), output=tensor([2, 4]), direct_tensors=[tensor([4, 3])],
                   dense_gemm_flops=48, start_ns=100, end_ns=200)
        document = dict(schema='gsdeploy.affine_operator_observations.v2', request_id='r',
                        modules=1, records=[row], instrumentation_errors=[], coverage_truncated=False,
                        gpu_synchronization_added=False, tensor_values_read=False, native_outputs_modified=False)
        phase = dict(pid=123, request_id='r', segment_id=row['id'], kind='gpu', start_ns=100, end_ns=200,
                     clock_mapping_valid=True, kernels=[dict(pid=123, start_ns=150, end_ns=250)],
                     kernel_duration_sum_ns=100, kernel_union_busy_ns=100)
        return document, [phase]

    def test_real_launch_association_is_required_and_async_end_allowed(self):
        document, phases = self.observed()
        issues, pairs, summary = validate_operator_rows(document, phases, request_id='r', pid=123)
        self.assertEqual(issues, [])
        self.assertTrue(pairs[0]['kernel_target_available'])
        self.assertEqual(summary['mapped_calls'], 1)
        phases[0]['kernels'] = []
        issues, pairs, _ = validate_operator_rows(document, phases, request_id='r', pid=123)
        self.assertIn('NO_AFFINE_CALL_WITH_REAL_CUPTI_KERNEL', issues)
        self.assertFalse(pairs[0]['kernel_target_available'])

    def test_unvalidated_dimensions_or_flops_are_rejected(self):
        document, phases = self.observed()
        document['records'][0]['dense_gemm_flops'] = 24
        self.assertTrue(validate_operator_rows(document, phases, request_id='r', pid=123)[0])

    def test_missing_and_duplicate_ranges_do_not_cancel_each_other(self):
        document, phases = self.observed()
        phases.append(deepcopy(phases[0]))
        issues, _, _ = validate_operator_rows(document, phases, request_id='r', pid=123)
        self.assertIn('DUPLICATE_AFFINE_CUPTI_RANGE', issues)

    def test_truncation_and_instrumentation_errors_are_explicit_failure(self):
        document, phases = self.observed()
        document['coverage_truncated'] = True
        self.assertEqual(validate_operator_rows(document, phases, request_id='r', pid=123)[0], ['AFFINE_OBSERVATION_INCOMPLETE'])


if __name__ == '__main__':
    unittest.main()
