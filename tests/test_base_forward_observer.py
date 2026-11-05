"""PEFT generate delegates past outer forward; observe the actual base LM."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

from tests.test_semantic_native_observer import FakeCUDA

DIRECTORY=Path(__file__).resolve().parents[1]/'tools/predictor_validation_20261027'
spec=importlib.util.spec_from_file_location('base_forward_observer_test',DIRECTORY/'base_forward_observed_worker.py')
worker=importlib.util.module_from_spec(spec)
sys.path.insert(0,str(DIRECTORY))
try:spec.loader.exec_module(worker)
finally:sys.path.pop(0)


class BaseForwardObserverTests(unittest.TestCase):
    def tearDown(self):worker.semantic.CURRENT=None

    def test_generate_bypass_is_observed_without_output_or_sync_change(self):
        base=SimpleNamespace(forward=lambda x:x+1)
        base.generate=lambda x:base.forward(x)
        wrapper=SimpleNamespace(peft_config={'active':'lora'},get_base_model=lambda:base,
            forward=lambda x:(_ for _ in ()).throw(AssertionError('outer forward is bypassed')),
            generate=lambda x:base.generate(x))
        adapter=SimpleNamespace(model=wrapper,executor=SimpleNamespace(family='chatqa'))
        worker.install_base_forward(adapter)
        recorder=worker.semantic.DetailRecorder(SimpleNamespace(cuda=FakeCUDA()),'request')
        worker.semantic.CURRENT=recorder
        self.assertEqual(wrapper.generate(3),4)
        recorder.finish()
        self.assertEqual([row['component'] for row in recorder.records],['model_forward'])
        self.assertTrue(recorder.records[0]['device_event_complete'])

    def test_other_families_unchanged_and_unsupported_peft_fails(self):
        worker.install_base_forward(SimpleNamespace(executor=SimpleNamespace(family='siglip')))
        with self.assertRaisesRegex(ValueError,'PEFT ChatQA'):
            worker.install_base_forward(SimpleNamespace(model=SimpleNamespace(),executor=SimpleNamespace(family='chatqa')))


if __name__=='__main__':unittest.main()
