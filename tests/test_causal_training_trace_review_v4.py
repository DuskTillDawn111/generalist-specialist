from copy import deepcopy
import unittest

from tools.predictor_validation_20261027.causal_training_trace_review_v4 import adapt,eligible_issue


class CausalLineageTests(unittest.TestCase):
    def fixture(self):
        return dict(clock_mappings={'7':dict(method='median',valid=False,max_boundary_residual_ns=190000)},
            phases=[dict(pid=7,start_ns=1000,end_ns=1100,nsys_start_ns=895,nsys_end_ns=1005,clock_mapping_valid=False,
                kernels=[dict(pid=7,nsys_start_ns=910,nsys_end_ns=930,start_ns=None,end_ns=None)]),
                dict(pid=7,start_ns=2000,end_ns=2010,nsys_start_ns=1895,nsys_end_ns=200000,
                     clock_mapping_valid=False,kernels=[])],nvtx_ranges=[],unattributed_kernels=[])

    def test_long_emission_tail_retained_and_all_brackets_satisfied(self):
        original=self.fixture();result=adapt(deepcopy(original),100,7)
        self.assertEqual(len(result['phases']),2)
        self.assertEqual(result['phases'][1]['nsys_end_ns'],200000)
        mapping=result['clock_mappings']['7']
        self.assertEqual(mapping['causal_interval']['lower_ns'],95)
        self.assertEqual(mapping['causal_interval']['upper_ns'],105)
        self.assertFalse(mapping['original_median_mapping']['valid'])
        self.assertEqual(mapping['original_median_mapping']['max_boundary_residual_ns'],190000)
        kernel=result['phases'][0]['kernels'][0]
        self.assertEqual((kernel['start_ns'],kernel['end_ns']),(1010,1030))
        self.assertEqual(kernel['end_ns']-kernel['start_ns'],20)
        self.assertFalse(mapping['cross_process_precision_qualified'])

    def test_origin_outside_any_bracket_never_accepted(self):
        with self.assertRaisesRegex(ValueError,'every source-ordered'):
            adapt(self.fixture(),106,7)
        invalid=self.fixture();invalid['phases'][1]['start_ns']=1900
        with self.assertRaisesRegex(ValueError,'every source-ordered'):
            adapt(invalid,100,7)

    def test_clock_lineage_cannot_rescue_source_or_output_failure(self):
        self.assertTrue(eligible_issue('PROCESS_CLOCK_MAPPING_FAILED:generalist_2#0'))
        self.assertTrue(eligible_issue('INVALID_AFFINE_METADATA_OR_BINDING:x:Invalid affine host/CUPTI range binding:generalist_2#0:r'))
        self.assertFalse(eligible_issue('CPU_REPARSED_OUTPUT_MISMATCH:r:prediction'))
        self.assertFalse(eligible_issue('INVALID_AFFINE_METADATA_OR_BINDING:x:Invalid logical tensor metadata:generalist_2#0:r'))


if __name__=='__main__':unittest.main()
