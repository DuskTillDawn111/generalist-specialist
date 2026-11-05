"""Runtime scoping must not silently relax G2 or impose strict settings on S."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

PATH = Path(__file__).resolve().parents[1]/'tools/predictor_validation_20261027/scoped_reconstruct_observed_worker.py'
spec = importlib.util.spec_from_file_location('scoped_worker', PATH)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def fake_torch():
    t = NS(cuda=NS(is_initialized=lambda: False), enabled=False,
           backends=NS(cudnn=NS(benchmark=False, deterministic=False, allow_tf32=True),
                       cuda=NS(matmul=NS(allow_tf32=False))))
    t.are_deterministic_algorithms_enabled = lambda: t.enabled
    t.is_deterministic_algorithms_warn_only_enabled = lambda: False
    t.use_deterministic_algorithms = Mock(side_effect=lambda value, **kw: setattr(t, 'enabled', value))
    return t


class ScopedRuntimeTests(unittest.TestCase):
    def test_g2_remains_strict_in_both_policies(self):
        for scope in ('all_strict', 'g2_strict'):
            t, env = fake_torch(), {}
            self.assertTrue(worker.configure_scope('generalist_2', scope, env, t))
            t.use_deterministic_algorithms.assert_called_once_with(True, warn_only=False)
            self.assertEqual(env['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')

    def test_specialist_removes_inherited_workspace_without_changing_torch_defaults(self):
        for slot in ('text', 'asr', 'image', 'ocr'):
            t, env = fake_torch(), {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
            self.assertFalse(worker.configure_scope(slot, 'g2_strict', env, t))
            self.assertNotIn('CUBLAS_WORKSPACE_CONFIG', env)
            t.use_deterministic_algorithms.assert_not_called()
            self.assertFalse(t.backends.cudnn.deterministic)

    def test_all_strict_specialist_keeps_control_policy(self):
        t, env = fake_torch(), {}
        self.assertTrue(worker.configure_scope('text', 'all_strict', env, t))
        self.assertTrue(t.backends.cudnn.deterministic)

    def test_invalid_or_late_configuration_rejected(self):
        with self.assertRaises(ValueError): worker.configure_scope('text', None, {}, fake_torch())
        t = fake_torch(); t.cuda.is_initialized = lambda: True
        with self.assertRaisesRegex(ValueError, 'before CUDA'): worker.configure_scope('text', 'all_strict', {}, t)
        t = fake_torch(); t.enabled = True
        with self.assertRaisesRegex(ValueError, 'fresh default'): worker.configure_scope('text', 'g2_strict', {}, t)

    def test_post_load_drift_cannot_pass_as_strict(self):
        t = fake_torch()
        with patch.dict(worker.os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}):
            with self.assertRaisesRegex(ValueError, 'during model loading'):
                worker.runtime_evidence(t, 'generalist_2', 'g2_strict', True)


if __name__ == '__main__': unittest.main()
