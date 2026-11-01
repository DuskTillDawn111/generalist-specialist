"""CPU feature contracts and regression tests, not measured validation."""
import unittest

from tools.predictor_validation_20261027.kernel_semantic_pooling_v5 import affine_features,semantic_features,fit_group,predict_group


class SemanticPoolingTests(unittest.TestCase):
    def test_affine_features_use_actual_named_weight_and_shapes(self):
        op=dict(implementation='torch.nn.modules.linear.Linear',in_features=4,out_features=8,
            input=dict(shape=[1,3,4],logical_bytes=24,dtype='fixture_half'),
            output=dict(shape=[1,3,8],logical_bytes=48,dtype='fixture_half'),
            direct_tensors=[dict(name='weight',logical_bytes=64,dtype='fixture_half'),dict(name='bias',logical_bytes=16,dtype='fixture_half')],dense_gemm_flops=192)
        features,config=affine_features(op)
        self.assertEqual((features['affine_M'],features['affine_N'],features['affine_K']),(3,8,4))
        self.assertEqual(features['affine_weight_logical_bytes'],64)
        self.assertEqual(config['tensor_storage_precision']['weight_tensor_name'],'weight')

    def test_cache_proxy_is_observed_positions_without_claiming_kv_tensor_length(self):
        first=dict(input_sequence_length=10,input_batch_size=1,input_token_positions=10,forward_index=0)
        second=dict(input_sequence_length=1,input_batch_size=1,input_token_positions=1,forward_index=1)
        f=semantic_features(second,{'prefill':first,'decode':second},10,300,2)
        self.assertEqual(f['cumulative_declared_input_positions'],11)
        self.assertFalse(any('kv' in k or 'flops' in k for k in f))

    def test_non_affine_unknown_features_are_absent(self):
        f=semantic_features(None,{},None,4,2)
        self.assertEqual(f,dict(request_launch_ordinal=4,phase_launch_ordinal=2))

    def test_minimum_runs_and_observations_stay_explicit(self):
        rows=[dict(run_id='cpu_fixture_1',features={'x':i},duration_ns=100+i) for i in range(3)]
        self.assertFalse(fit_group(rows,['x'])['fit_accepted'])
        rows[-1]['run_id']='cpu_fixture_2';model=fit_group(rows,['x'])
        self.assertTrue(model['fit_accepted']);self.assertAlmostEqual(predict_group(model,{'x':1})['duration_hypothesis_ns'],101)
        self.assertTrue(predict_group(model,{'x':1})['known'])
        self.assertFalse(predict_group(model,{'x':10})['known'])

    def test_missing_query_features_are_not_zero_imputed(self):
        rows=[dict(run_id=run,features={'x':x},duration_ns=100+x) for run in ('cpu_fixture_1','cpu_fixture_2') for x in (1,2)]
        with self.assertRaises(ValueError):predict_group(fit_group(rows,['x']),{})


if __name__=='__main__':unittest.main()
