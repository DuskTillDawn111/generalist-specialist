"""Fail-closed ExLlama branch selection without allocating or timing a GPU."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

PATH=Path(__file__).resolve().parents[1]/'tools/predictor_validation_20261027/reconstruct_observed_worker.py'
spec=importlib.util.spec_from_file_location('reconstruct_observer_test',PATH)
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)


class ReconstructObserverTests(unittest.TestCase):
    def adapter(self,**kwargs):
        layer=SimpleNamespace(QUANT_TYPE='exllama',bits=4,group_size=128,_use_act_order=False,
                              infeatures=128,outfeatures=256)
        model=SimpleNamespace(modules=lambda:[layer],device_to_buffers={0:dict(max_dq_buffer_size=128*256)})
        return SimpleNamespace(model=model,executor=SimpleNamespace(family='minicpm45',memory_audit={})),layer

    def test_threshold_one_forces_reconstruction_for_every_positive_row_count(self):
        adapter,_=self.adapter();kernel=SimpleNamespace(set_tuning_params=Mock())
        evidence=worker.configure_model(adapter,kernel)
        kernel.set_tuning_params.assert_called_once_with(1,False,False)
        for rows in [1,2,3,8,40960]:
            threshold=evidence['matmul_recons_thd']
            self.assertFalse(threshold==0 or rows<threshold)
        self.assertFalse(evidence['quantized_weights_changed'])
        self.assertFalse(evidence['determinism_qualified'])

    def test_unknown_quantization_and_insufficient_scratch_are_rejected(self):
        adapter,layer=self.adapter();layer._use_act_order=True
        with self.assertRaisesRegex(ValueError,'no-act-order'):worker.configure_model(adapter,Mock())
        adapter,_=self.adapter();adapter.model.device_to_buffers[0]['max_dq_buffer_size']=1
        with self.assertRaisesRegex(ValueError,'scratch'):worker.configure_model(adapter,Mock())

    def test_specialist_models_do_not_change_kernel_settings(self):
        kernel=SimpleNamespace(set_tuning_params=Mock())
        self.assertIsNone(worker.configure_model(SimpleNamespace(executor=SimpleNamespace(family='chatqa')),kernel))
        kernel.set_tuning_params.assert_not_called()


if __name__=='__main__':unittest.main()
