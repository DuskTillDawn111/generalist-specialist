"""Host-only mathematical tests; not measured new-shape validation."""
import copy
import unittest

from tools.predictor_validation_20261027.kernel_template_rules_v3 import (
    axis_candidates,evaluate_axis,fit_pattern,infer_integer_width,instantiate_model,instantiate_pattern,variant_grid)


def signature(grid,name='fixture_kernel'):
    return dict(name=name,grid=grid,block=[128,1,1],registers_per_thread=16,
        static_shared_bytes=0,dynamic_shared_bytes=0,cluster=None,
        stream_role=0,sharedMemoryExecuted=32768)


class TemplateRuleTests(unittest.TestCase):
    def test_ceil_rule_has_real_rounding_at_new_shape(self):
        candidates=axis_candidates(414,26,4514,283)
        rule=next(r for r in candidates if r==dict(kind='ceil_ratio',numerator=1,denominator=16,multiple=1))
        self.assertEqual(evaluate_axis(rule,513),33)
        self.assertNotEqual(evaluate_axis(rule,513),513//16)

    def test_swizzled_grid_is_not_linear_grid_interpolation(self):
        candidates=axis_candidates(414,56,4514,568)
        rule=dict(kind='ceil_ratio',numerator=1,denominator=64,multiple=8)
        self.assertIn(rule,candidates)
        self.assertEqual(evaluate_axis(rule,1025),136)

    def test_element_width_requires_joint_integer_constraints(self):
        self.assertEqual(infer_integer_width(414,3312,4514,36112,512),[4096])

    def test_stable_kernel_new_grid_numeric_candidate(self):
        a=signature([3312,1,1],name='at::native::vectorized_elementwise_kernel<(int)4, fixture>')
        b=copy.deepcopy(a);b['grid']=[36112,1,1]
        model=fit_pattern(a,b,414,4514,[],[])
        value,status=instantiate_pattern(model,1025)
        self.assertEqual(value['grid'],[8200,1,1]);self.assertEqual(value['name'],a['name'])
        self.assertFalse(model['measured_new_shape_validation'])
        self.assertEqual(status,'SOURCE_CONDITIONAL_GRID_CANDIDATE')

    def test_dispatch_algorithms_remain_distinct_candidates(self):
        a=signature([1,13,16],name='ampere_sgemm_32x32_sliced1x4_tn')
        b=signature([1,36,7],name='ampere_sgemm_32x128_tn');b['registers_per_thread']=64
        model=fit_pattern(a,b,414,4514,[],[])
        low,_=instantiate_pattern(model,1025,'lower_domain');high,_=instantiate_pattern(model,1025,'upper_domain')
        self.assertNotEqual(low['name'],high['name']);self.assertEqual(low['grid'],[1,33,16])
        self.assertEqual(high['grid'],[1,9,7]);self.assertFalse(model['dispatch_threshold_known'])

    def test_cutlass_tile_variant_reproduces_observed_anchor(self):
        a=signature([288,2,1],name='void cutlass::Kernel2<cutlass_80_tensorop_i16832gemm_s8_256x128_64x3_tn_align16>(params)')
        self.assertEqual(variant_grid(a,4514,4514)[0],a['grid'])
        self.assertEqual(variant_grid(a,1025,4514)[0],[72,2,1])

    def test_reduction_block_and_dynamic_shared_formula(self):
        a=signature([1,1,1],name='ReduceOp<bool, fixture>');a.update(block=[64,1,1],dynamic_shared_bytes=64)
        b=copy.deepcopy(a);b.update(block=[512,1,1],dynamic_shared_bytes=512)
        model=fit_pattern(a,b,414,4514,[],[]);value,status=instantiate_pattern(model,1025)
        self.assertEqual(value['block'],[256,1,1]);self.assertEqual(value['dynamic_shared_bytes'],256)
        self.assertIn('CARVEOUT_HYPOTHESIS',status)

    def test_bad_reduction_binding_rejected(self):
        a=signature([1,1,1],name='ReduceOp<bool, fixture>');b=copy.deepcopy(a)
        with self.assertRaisesRegex(ValueError,'reduction'):fit_pattern(a,b|dict(block=[512,1,1]),414,4514,[],[])

    def test_no_unbounded_shape_extrapolation(self):
        model=fit_pattern(signature([1,1,1]),signature([9,1,1]),414,4514,[],[])
        with self.assertRaises(ValueError):instantiate_pattern(model,4515)

    def test_all_observed_fields_preserved_at_anchors(self):
        a=signature([1,1,1]);b=signature([9,1,1]);model=fit_pattern(a,b,414,4514,[],[])
        self.assertEqual(instantiate_pattern(model,414)[0],a)
        self.assertEqual(instantiate_pattern(model,4514)[0],b)

    def test_public_shape_and_policy_checked_even_at_exact_anchor(self):
        model=fit_pattern(signature([1,1,1]),signature([9,1,1]),414,4514,[],[])
        for length in (414.0,True,0):
            with self.assertRaises(ValueError):instantiate_pattern(model,length)
        with self.assertRaises(ValueError):instantiate_pattern(model,414,'unrecognized')

    def test_complete_model_preserves_slot_order_without_mutating_observations(self):
        first=fit_pattern(signature([1,1,1]),signature([9,1,1]),414,4514,[],[])
        second=fit_pattern(signature([414,1,1]),signature([4514,1,1]),414,4514,[],[])
        report=dict(patterns=[first,second],slot_pattern_indices=[1,0,1],training_lengths=[414,4514])
        original=copy.deepcopy(report);result=instantiate_model(report,963)
        self.assertEqual([r['grid'][0] for r in result['ordered_launch_signatures']],[963,2,963])
        result['ordered_launch_signatures'][0]['grid'][0]=7
        self.assertEqual(report,original)
        self.assertFalse(result['new_shape_validated'])

    def test_even_anchors_do_not_resolve_odd_shape_basis(self):
        rules=axis_candidates(414,414,4514,4514)
        self.assertIn(963,{evaluate_axis(rule,963) for rule in rules})
        self.assertIn(964,{evaluate_axis(rule,963) for rule in rules})


if __name__=='__main__':unittest.main()
