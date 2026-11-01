"""CPU fixtures verify provenance rules; never target-model evidence."""
import copy
import json
import unittest
import numpy as np

from tools.predictor_validation_20261027.kernel_baseline_intake_v3 import (
    capture_identity, explicit_configuration, json_ready)
from tools.predictor_validation_20261027.native_collect import context_identity


class RealIntakeContractTests(unittest.TestCase):
    def setUp(self):
        self.source=dict(path='cpu-fixture-not-real-evidence',sha256='a'*64)
        self.context=dict(execution=dict(batch_size=1),residents={'g#0':'g'},
            worker_bindings={'g':dict(execution=dict(dtype='float16',quantization=dict(bits=4)))},
            measurement_domain='light-kernel',affine_observation=False,
            execution_sources=[dict(original_path='worker.py',frozen=self.source)],sources=[self.source])
        self.record=dict(id='affine_0',module='layer.q',
            implementation='auto_gptq.nn_modules.qlinear.qlinear_exllama.QuantLinear',
            direct_tensors=[dict(name='qweight',dtype='torch.int32',logical_bytes=64),
                            dict(name='scales',dtype='torch.float16',logical_bytes=32)],
            input=dict(dtype='torch.float16'),output=dict(dtype='torch.float16'))
        self.semantic=dict(id='prefill_0',input_batch_size=1)
        self.signature=dict(name='real-observed-name-fixture',grid=[2,1,1],block=[64,1,1],registers_per_thread=32)

    def config(self):
        return explicit_configuration(self.record,self.semantic,'g',0,self.signature,
                                      self.source,self.source,self.source,self.context)

    def identity(self):
        sha=context_identity(self.context)
        return capture_identity(dict(raw=dict(sha256='b'*64),capture_run_id='raw-sha256:'+'b'*64),
                                dict(execution_context_sha256=sha),dict(execution_context_sha256=sha),self.context)

    def test_packed_weight_explicit_storage_dtype_preserved(self):
        result=self.config()
        self.assertEqual(result['weight_tensor_name'],'qweight')
        self.assertEqual(result['explicit_precision']['weight'],'torch.int32')
        self.assertEqual(result['explicit_precision']['configured_execution']['quantization']['bits'],4)

    def test_no_batch_from_tensor_shape(self):
        self.semantic.pop('input_batch_size')
        self.record['input']['shape']=[1,20,8]
        with self.assertRaisesRegex(ValueError,'batch'):self.config()

    def test_batch_requires_frozen_configuration_agreement(self):
        self.semantic['input_batch_size']=2
        with self.assertRaisesRegex(ValueError,'batch'):self.config()

    def test_unknown_weight_implementation_remains_unknown(self):
        self.record['implementation']='unknown.Linear'
        with self.assertRaisesRegex(ValueError,'Unsupported'):self.config()

    def test_ambiguous_named_weight_rejected(self):
        self.record['direct_tensors'].append(copy.deepcopy(self.record['direct_tensors'][0]))
        with self.assertRaisesRegex(ValueError,'ambiguous'):self.config()

    def test_missing_dtype_cannot_be_guessed_from_kernel_name(self):
        self.record['input'].pop('dtype')
        self.signature['name']='fp16_gemm'
        with self.assertRaisesRegex(ValueError,'dtype'):self.config()

    def test_grid_covariate_changes_allow_same_identity_but_block_does_not(self):
        original=self.config()['execution_key']
        self.signature['grid']=[4,1,1]
        self.assertEqual(original,self.config()['execution_key'])
        self.signature['block']=[128,1,1]
        self.assertNotEqual(original,self.config()['execution_key'])

    def test_local_kernel_ordinal_is_bound(self):
        first=self.config()['execution_key']
        second=explicit_configuration(self.record,self.semantic,'g',1,self.signature,
                                      self.source,self.source,self.source,self.context)['execution_key']
        self.assertNotEqual(first,second)

    def test_job_id_cannot_be_independent_capture_run(self):
        sha=context_identity(self.context)
        with self.assertRaisesRegex(ValueError,'never a job ID'):
            capture_identity(dict(raw=dict(sha256='b'*64),capture_run_id='job-1'),
                             dict(execution_context_sha256=sha),dict(execution_context_sha256=sha),self.context)

    def test_full_canonical_context_retained(self):
        run,identity=self.identity()
        self.assertEqual(run,'raw-sha256:'+'b'*64)
        changed=copy.deepcopy(self.context);changed['worker_bindings']['g']['execution']['dtype']='float32'
        with self.assertRaisesRegex(ValueError,'identity mismatch'):
            capture_identity(dict(raw=dict(sha256='b'*64),capture_run_id=run),
                             dict(execution_context_sha256=identity),dict(execution_context_sha256=identity),changed)

    def test_metadata_domain_cannot_supply_light_timing_identity(self):
        self.context['measurement_domain']='metadata';self.context['affine_observation']=True
        with self.assertRaisesRegex(ValueError,'Light timing'):self.identity()

    def test_nested_real_fit_numpy_diagnostics_serialize(self):
        model=dict(known=True,diagnostics=dict(coefficients_individually_identifiable=np.bool_(False),
                   rank=np.int64(2)),coefficients=np.array([1.,2.]))
        serialized=json.loads(json.dumps(json_ready(model),allow_nan=False))
        self.assertIs(serialized['diagnostics']['coefficients_individually_identifiable'],False)
        self.assertEqual(serialized['diagnostics']['rank'],2)
        self.assertEqual(serialized['coefficients'],[1.,2.])


if __name__=='__main__':unittest.main()
