"""Strict runtime controls fail closed; no GPU or fabricated measurements."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

PATH=Path(__file__).resolve().parents[1]/'tools/predictor_validation_20261027/deterministic_observed_worker.py'
spec=importlib.util.spec_from_file_location('deterministic_native_pilot',PATH)
worker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class DeterministicNativeObserverTests(unittest.TestCase):
    def torch(self):
        return SimpleNamespace(use_deterministic_algorithms=Mock(),
            backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=True,deterministic=False)))

    def test_workspace_must_be_explicit_before_cuda_initialization(self):
        torch=self.torch()
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,'CUBLAS_WORKSPACE_CONFIG'):
                worker.configure(torch)
        torch.use_deterministic_algorithms.assert_not_called()

    def test_strict_determinism_does_not_silently_fallback(self):
        torch=self.torch()
        with patch.dict(os.environ,{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}):
            worker.configure(torch)
        torch.use_deterministic_algorithms.assert_called_once_with(True,warn_only=False)
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertTrue(torch.backends.cudnn.deterministic)


if __name__=='__main__':
    unittest.main()
