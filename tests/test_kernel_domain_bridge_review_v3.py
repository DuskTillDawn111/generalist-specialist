from copy import deepcopy
import unittest

from tools.predictor_validation_20261027.kernel_domain_bridge_review_v3 import compare_sequences


class KernelDomainBridgeTests(unittest.TestCase):
    def item(self, name='a', group='prefill_0000', operator='affine_1', start=10):
        return dict(group=group, operator_id=operator, signature=dict(name=name, stream_role=0),
                    kernel=dict(correlation_id=start, nsys_start_ns=start, start_ns=start,
                                end_ns=start+5))

    def test_exact_group_transfers_identity_and_measured_light_duration(self):
        left=[self.item(),self.item('b', operator=None)]
        right=deepcopy(left)
        right[0]['kernel'].update(correlation_id=90,nsys_start_ns=100,start_ns=100,end_ns=109)
        groups=compare_sequences(left,right)
        self.assertEqual(groups[0]['status'],'EXACT_OBSERVED_SIGNATURE_ORDER_MATCH')
        pairs=groups[0]['mapped_operator_kernels']
        self.assertEqual(len(pairs),1)
        self.assertEqual(pairs[0]['light_kernel']['correlation_id'],90)
        self.assertEqual(pairs[0]['light_kernel_duration_ns'],9)

    def test_one_mismatch_invalidates_entire_group(self):
        left=[self.item(),self.item('b')]
        right=deepcopy(left);right[1]['signature']['name']='changed'
        result=compare_sequences(left,right)[0]
        self.assertEqual(result['status'],'UNKNOWN_DIFFERENT_OBSERVED_KERNEL_SEQUENCE')
        self.assertEqual(result['first_mismatch_index'],1)
        self.assertEqual(result['mapped_operator_kernels'],[])

    def test_count_and_order_mismatches_do_not_transfer(self):
        left=[self.item(),self.item('b')]
        for right,first in [(left[:1],1),(list(reversed(left)),0),([],0)]:
            result=compare_sequences(left,right)[0]
            self.assertEqual(result['first_mismatch_index'],first)
            self.assertEqual(result['mapped_operator_kernels'],[])

    def test_separate_matching_group_is_retained_but_missing_group_unknown(self):
        left=[self.item(),self.item('c','decode_0001')]
        result=compare_sequences(left,left[:1])
        self.assertEqual(result[0]['status'],'EXACT_OBSERVED_SIGNATURE_ORDER_MATCH')
        self.assertEqual(result[1]['status'],'UNKNOWN_DIFFERENT_OBSERVED_KERNEL_SEQUENCE')
        self.assertEqual(result[1]['mapped_operator_kernels'],[])

    def test_empty_capture_is_never_a_successful_group(self):
        self.assertEqual(compare_sequences([],[]),[])

    def test_static_launch_attribute_change_is_unknown(self):
        left=[self.item()];right=deepcopy(left)
        right[0]['signature']['sharedMemoryExecuted']=32768
        self.assertEqual(compare_sequences(left,right)[0]['mapped_operator_kernels'],[])


if __name__=='__main__':unittest.main()
