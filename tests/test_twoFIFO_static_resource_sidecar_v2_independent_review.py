"""Independent narrow causal/structural checks; no model or device execution."""
import copy
import unittest

from tests.test_twoFIFO_static_resource_sidecar_v2 import fixture,run


class V3StaticResourceSidecarIndependentReview(unittest.TestCase):
    def test_descriptor_availability_uses_processing_clock_at_every_piece(self):
        args=fixture();out=run(args)
        events={(e['owner'],e['request_id'],e['segment_id']):e for e in args[1]
                if e.get('event')=='start' and e.get('semantic_phase') in ('prefill','decode')}
        unseen=seen=0
        for row in out['records']:
            key=row['owner'],row['request_id'],row['macro_id']
            for piece in row['path']:
                state=piece['projection_state'];own=state['own_BEGIN_descriptor'];cut=piece['left_ns']
                if key in events:
                    event=events[key]
                    if event['collector_processing_time_ns']>cut:
                        self.assertFalse(state['own_supported']);self.assertIsNone(own);unseen+=1
                    else:
                        self.assertTrue(state['own_supported']);seen+=1
                        self.assertEqual(own['available_at_processing_ns'],event['collector_processing_time_ns'])
                        self.assertEqual(own['sequence_length'],event['input_sequence_length'])
                for pair in state['pairs']:
                    other=pair['peer_BEGIN_descriptor']
                    if other:
                        self.assertLessEqual(other['available_at_processing_ns'],cut)
                        self.assertEqual((other['owner'],other['request_id'],other['phase_id']),
                            (pair['peer_owner'],pair['peer_request_id'],pair['peer_phase_id']))
        self.assertGreater(unseen,0);self.assertGreater(seen,0)

    def test_future_END_payload_and_completed_service_labels_do_not_change_structure_records(self):
        args=fixture();before=run(args);changed=copy.deepcopy(args)
        for event in changed[1]:
            if event.get('event')!='start':
                event.update(input_sequence_length=10**9,elapsed_ms=10**12,
                    worker_response={'generated_tokens':10**9},query_seconds=10**12)
        for request in changed[0]['requests']:
            request['worker_start_to_ready_ns']=10**18
            for macro in request['macros']:
                macro['forward_shape']={'input_batch_size':999,'input_sequence_length':999}
        after=run(changed)
        self.assertEqual(before['records'],after['records'])
        self.assertEqual(before['actual_BEGIN_descriptor_catalog'],after['actual_BEGIN_descriptor_catalog'])
        # Retrospective priming history is retained literally and is separate
        # from each causal piece's structural descriptors.
        self.assertNotEqual(before['priming_processed_history'],after['priming_processed_history'])

    def test_nominal_projection_arithmetic_is_not_achieved_or_physical_resource_demand(self):
        args=fixture();out=run(args);binding=args[2]['text']
        params=sum(p['in_features']*p['out_features'] for p in binding['projections'])
        width_sum=sum(p['in_features']+p['out_features'] for p in binding['projections'])
        for descriptor in out['actual_BEGIN_descriptor_catalog']:
            f=descriptor['nominal_projection_features'];rows=f['batch_size']*f['sequence_length']
            self.assertEqual(f['base_nominal_dense_gemm_flops'],2*rows*params)
            self.assertEqual(f['base_logical_input_output_operand_bytes_sum'],rows*width_sum*binding['activation_dtype_bytes'])
            self.assertEqual(f['classification'],'unknown')
        for key,value in out['unknown_physical_resources'].items():
            self.assertEqual(value,'unknown') if key=='physical_classification' else self.assertIsNone(value)
        self.assertIsNone(out['complementary_service_multiplier'])
        self.assertFalse(out['reference_rate_modified']);self.assertFalse(out['complementarity_qualified'])

    def test_missing_bindings_preserve_entire_macro_and_pair_exposure_denominator(self):
        args=fixture();before=run(args);changed=list(args);changed[2]={};after=run(changed)
        self.assertEqual(after['requests'],before['requests']);self.assertEqual(after['macros'],before['macros'])
        self.assertEqual(after['visible_peer_exposure_ns'],before['visible_peer_exposure_ns'])
        self.assertEqual(after['known_pair_exposure_ns'],0)
        self.assertEqual(after['unknown_pair_exposure_ns'],after['visible_peer_exposure_ns'])
        self.assertEqual(after['known_BEGIN_descriptors'],0)
        self.assertEqual(after['unsupported_BEGIN_descriptors'],before['known_BEGIN_descriptors'])
        self.assertTrue(all(not p['projection_state']['own_supported'] for r in after['records'] for p in r['path']))


if __name__=='__main__':unittest.main()
