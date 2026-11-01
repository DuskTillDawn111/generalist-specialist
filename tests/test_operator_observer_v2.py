"""CPU affine-observer checks against actual torch module hook semantics."""
import unittest
from types import SimpleNamespace

from tools.predictor_validation_20261027.operator_observer_v2 import observe_affine_operators

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'CPU torch unavailable')
class OperatorObserverTests(unittest.TestCase):
    def test_real_linear_metadata_and_native_output(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
        x = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
        expected = model(x).detach().clone()
        events = []
        with observe_affine_operators(model, 'req:0', sink=events.append) as obs:
            actual = model(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        doc = obs.document()
        self.assertEqual(doc['modules'], 2)
        self.assertEqual([r['dense_gemm_flops'] for r in doc['records']], [96, 64])
        self.assertEqual(doc['records'][0]['input']['logical_bytes'], 48)
        self.assertEqual(doc['records'][0]['output']['shape'], [2, 2, 4])
        self.assertEqual(len(events), 4)
        self.assertEqual(doc['instrumentation_errors'], [])
        self.assertFalse(doc['training_target_available'])
        self.assertFalse(model[0]._forward_hooks)
        self.assertFalse(model[0]._forward_pre_hooks)

    def test_packed_quantized_features_preserve_storage_dtype(self):
        class QuantLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.infeatures, self.outfeatures = 8, 4
                self.register_buffer('qweight', torch.ones(1, 4, dtype=torch.int32))
                self.register_buffer('scales', torch.ones(1, 4, dtype=torch.float16))

            def forward(self, input):
                return input[..., :4]

        model = QuantLinear()
        with observe_affine_operators(model, 'q') as obs:
            model(torch.ones(2, 8))
        row = obs.records[0]
        self.assertEqual(row['dense_gemm_flops'], 128)
        tensors = {t['name']: t for t in row['direct_tensors']}
        self.assertEqual(tensors['qweight']['logical_bytes'], 16)
        self.assertEqual(tensors['qweight']['dtype'], 'torch.int32')
        self.assertEqual(tensors['scales']['logical_bytes'], 8)
        self.assertIsNone(row['resource_measurements'])

    def test_wrapper_and_unrelated_dimensions_are_not_double_counted(self):
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.in_features, self.out_features = 3, 2
                self.base_layer = torch.nn.Linear(3, 2)

            def forward(self, x):
                return self.base_layer(x)

        model = Linear()
        with observe_affine_operators(model, 'wrapped') as obs:
            model(torch.ones(1, 3))
        self.assertEqual(len(obs.records), 1)
        self.assertEqual(obs.records[0]['module'], 'base_layer')

    def test_budget_failure_and_exception_cleanup(self):
        model = torch.nn.Linear(3, 2)
        with observe_affine_operators(model, 'budget', max_records=1) as obs:
            model(torch.ones(1, 3))
            model(torch.ones(1, 3))
            with self.assertRaises(ValueError):
                with observe_affine_operators(model, 'nested'):
                    pass
        self.assertEqual(len(obs.records), 1)
        self.assertTrue(obs.document()['coverage_truncated'])
        with self.assertRaises(RuntimeError):
            with observe_affine_operators(model, 'bad-shape') as bad:
                model(torch.ones(1, 4))
        self.assertEqual(bad.records[0]['unsupported_reason'], 'INPUT_LAST_DIMENSION_DOES_NOT_MATCH_DECLARATION')
        self.assertIsNone(bad.records[0]['output'])
        self.assertIn('end_ns', bad.records[0])
        self.assertFalse(model._forward_hooks)

    def test_optional_sink_error_does_not_change_result(self):
        def broken_sink(row):
            raise RuntimeError('observation channel failed')

        model = torch.nn.Linear(3, 2)
        x = torch.ones(1, 3)
        expected = model(x).detach().clone()
        with observe_affine_operators(model, 'sink', sink=broken_sink) as obs:
            actual = model(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(len(obs.document()['instrumentation_errors']), 2)

    def test_combined_worker_observer_preserves_forward_and_layers(self):
        from tools.predictor_validation_20261027.operator_worker_v2 import combined_observer

        class Language(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(3, 2)

            def forward(self, inputs_embeds=None, past_key_values=None):
                return self.layer(inputs_embeds)

        model = Language()
        original_forward = model.forward
        x = torch.ones(1, 4, 3)
        expected = model(inputs_embeds=x).detach().clone()
        executor = SimpleNamespace(family='chatqa', model=model)
        with combined_observer(executor, 'combo') as obs:
            actual = model(inputs_embeds=x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(model.forward, original_forward)
        doc = obs.document()
        self.assertEqual(doc['records'][0]['input_sequence_length'], 4)
        self.assertEqual(doc['affine_operator_observations']['records'][0]['dense_gemm_flops'], 48)
        self.assertFalse(model._forward_hooks)


if __name__ == '__main__':
    unittest.main()
