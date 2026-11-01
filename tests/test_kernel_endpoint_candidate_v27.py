import copy
import unittest

from tools.predictor_validation_20261027.kernel_endpoint_candidate_v27 import (
    features,fit,predict,target,stats)


def row(own,remaining=10,signature='known'):
    return dict(capture='c',source_node_id='n',prefix_features=dict(cutoff_ns=100,owner='a',
        semantic_phase='prefill',launch_signature=signature,source_launch_API_entry_ns=90,
        source_launch_API_return_ns=100,already_observed_kernel_start_at_cutoff=False,
        already_observed_kernel_end_at_cutoff=False,
        observed_started_not_ended_kernel_count_by_owner={'a':0,'b':int(not own)},
        context_event_prefix_at_cutoff=dict(owner='a' if own else 'b',context_age_ns=10,
            status='UNIQUE_OPEN_CONTEXT_PREFIX_NOT_SM_SERVICE')),
        observation_labels=dict(observed_kernel_start_ns=100+remaining,
            observed_kernel_end_ns=110+remaining))


class EndpointTests(unittest.TestCase):
    def test_future_labels_cannot_change_prefix_design(self):
        value=row(False)
        changed=copy.deepcopy(value);changed['observation_labels']['observed_kernel_end_ns']=10**9
        self.assertTrue((features(value['prefix_features'])==features(changed['prefix_features'])).all())

    def test_nonheld_completion_prediction_has_no_binary_context_gate(self):
        rows=[row(True,remaining=10+i) for i in range(5)]+[row(False,remaining=30+i) for i in range(5)]
        model=fit(rows,'END_if_not_ended')
        value=predict(model,row(False)['prefix_features'])
        self.assertGreater(value['predicted_remaining_wall_ns'],0)
        self.assertFalse(value['physical_SM_service'])
        self.assertFalse(value['reference_work_rate'])
        self.assertGreater(value['predicted_remaining_wall_ns'],predict(model,row(True)['prefix_features'])['predicted_remaining_wall_ns'])

    def test_observed_started_or_ended_targets_not_fabricated_zero(self):
        value=row(True)
        value['prefix_features']['already_observed_kernel_start_at_cutoff']=True
        value['observation_labels']['observed_kernel_start_ns']=95
        self.assertIsNone(target(value,'START_if_not_started'))
        self.assertGreater(target(value,'END_if_not_ended'),0)
        value['prefix_features']['already_observed_kernel_end_at_cutoff']=True
        value['observation_labels']['observed_kernel_end_ns']=99
        self.assertIsNone(target(value,'END_if_not_ended'))

    def test_unknown_signature_preserves_unknown(self):
        model=fit([row(True),row(False)],'END_if_not_ended')
        value=predict(model,row(True,signature='unknown')['prefix_features'])
        self.assertIsNone(value['predicted_remaining_wall_ns'])
        self.assertEqual(value['status'],'UNKNOWN_LAUNCH_SIGNATURE')

    def test_error_sums_never_labelled_request_time_or_service(self):
        values=stats([10,10],[20,20])
        self.assertTrue(values['sums_overlap_across_kernels'])
        self.assertTrue(values['sums_are_not_request_latency_or_GPU_work'])


if __name__=='__main__':unittest.main()
