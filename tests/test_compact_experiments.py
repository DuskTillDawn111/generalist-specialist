"""CPU preparation/execution boundary and full-cohort scoring regressions."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from gsdeploy.profiling import compact_experiments as experiments
from gsdeploy.profiling.prepare import digest, write


class CompactExperimentTests(unittest.TestCase):
    def test_both_execution_entries_require_explicit_execute(self):
        for command in ['run', 'worker-profile']:
            argv = ['compact_experiments', command, '--job', 'nonexistent', '--output', 'nonexistent']
            if command == 'run':
                argv += ['--manifest', 'nonexistent']
            with self.subTest(command=command), patch.object(sys, 'argv', argv), \
                    patch.object(experiments, 'execute_job') as run, patch.object(experiments, 'profile_job') as worker:
                with self.assertRaises(SystemExit) as error:
                    experiments.main()
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()
                worker.assert_not_called()

    def test_manifest_tamper_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = {t: write(root / (t + '.jsonl'), [{'id': t, 'task': t}]) for t in experiments.TASKS}
            path = root / 'manifest.json'
            path.write_text(json.dumps({'tasks': tasks}))
            experiments.checked_manifest(path)
            (root / 'ocr.jsonl').write_text('{"id":"changed"}\n')
            with self.assertRaisesRegex(ValueError, 'Changed frozen task manifest'):
                experiments.checked_manifest(path)

    def test_import_and_help_do_not_import_torch_or_nvml(self):
        code = ('import sys; import gsdeploy.profiling.compact_experiments; '
                'assert "torch" not in sys.modules; assert "pynvml" not in sys.modules')
        subprocess.run([sys.executable, '-c', code], check=True)

class CompactOpenLoopIntegrationTests(unittest.TestCase):
    def test_measurement_uses_runtime_pool_and_fixed_window_board_energy_without_gpu(self):
        import time
        from types import SimpleNamespace
        from gsdeploy.runtime.deployment import DeploymentRuntime
        from gsdeploy.energy import integrate_power
        class MockRuntime(DeploymentRuntime):
            def __init__(self, config, registry):
                for entry in registry.values():
                    entry['command'] = [sys.executable, '-m', 'gsdeploy.backends.mock']
                    entry['env']['GS_TASKS'] = '["image"]'
                    entry['env']['CUDA_VISIBLE_DEVICES'] = ''
                super().__init__(config, registry)
        class CPUBoard:
            def __init__(self):
                import threading
                self.samples, self.errors, self.handle = [], [], None
                self.stop = threading.Event()
                self.nv = SimpleNamespace(nvmlDeviceGetUUID=lambda _: 'CPU-test-fixture',
                                          nvmlDeviceGetPowerManagementLimit=lambda _: 450000)
            def sample(self):
                value = {'time_ns': time.perf_counter_ns(), 'watts': 50, 'used_mib': 1000,
                         'counter_joules': None, 'processes': []}
                self.samples.append(value)
                return value
            def start(self):
                import threading
                self.sample()
                def poll():
                    while not self.stop.wait(.01):self.sample()
                self.thread = threading.Thread(target=poll, daemon=True);self.thread.start()
            def close(self):
                self.stop.set();self.thread.join(1);self.sample()
            def window(self, start, end):
                energy = integrate_power(self.samples, start['time_ns'], end['time_ns'])
                return {'joules': energy, 'coverage_valid': energy is not None}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = write(root / 'pool.jsonl', experiments.runtime_samples([{'id': 'i', 'task': 'image', 'label': 'x'}]))
            study = {'gpu_memory_mib': 49140, 'safe_factor': 1.110854,
                     'task_adapter_protocol': {'sha256': 'p' * 64}}
            study_path = root / 'study.json';study_path.write_text(json.dumps(study))
            worker_spec = root / 'worker.json'
            worker_spec.write_text(json.dumps({'execution': {}}))
            manifest = {'study': str(study_path), 'workers': {'image': {'python': sys.executable,
                         'spec': str(worker_spec), 'tasks': ['image']}}}
            job = {'deployment': {'image': 1}, 'routing': {'image': {'image': 10.0}}, 'rates': {'image': 10.0},
                   'duration_seconds': .3, 'seed': 42, 'sample_pool': pool['path'],
                   'sample_pool_sha256': pool['sha256'], 'input_manifest_sha256': pool['sha256'],
                   'startup_amortization_seconds': 3600.0}
            with patch('gsdeploy.runtime.deployment.DeploymentRuntime', MockRuntime), \
                    patch('gsdeploy.profiling.telemetry.BoardMonitor', CPUBoard):
                result = experiments.measure_plan(job, manifest, root)
            self.assertEqual(result['status'], 'completed', result)
            self.assertTrue(result['complete_cohort'])
            self.assertTrue(result['coverage_valid'])
            self.assertAlmostEqual(result['run_energy']['joules'], 50 * .3, places=5)
            self.assertGreater(result['effective_board_watts'], 50)
            self.assertIsNone(result['utility'])
            self.assertFalse(result['admission_granted'])
            self.assertGreater(len(result['request_records']), 0)
            self.assertIsNotNone(result['task_metrics']['image']['p99_e2e_latency_ms'])
            self.assertIsNotNone(result['p99_latency_ms'])
            idle = root / 'idle'
            idle.mkdir()
            idle_job = {**job, 'rates':{'image':0.0}, 'routing':{'image':{}}, 'idle_residency':True}
            with patch('gsdeploy.runtime.deployment.DeploymentRuntime', MockRuntime), \
                    patch('gsdeploy.profiling.telemetry.BoardMonitor', CPUBoard):
                result = experiments.measure_plan(idle_job, manifest, idle)
            self.assertEqual(result['status'],'completed',result)
            self.assertEqual(result['request_records'],[])
            self.assertEqual(result['priming_request_records'],[])
            self.assertTrue(result['coverage_valid'])
            self.assertTrue(result['complete_cohort'])
            self.assertTrue(result['stable'])
            self.assertAlmostEqual(result['run_energy']['joules'],50*.3,places=5)
            self.assertIsNone(result['mean_latency_ms'])
            self.assertIsNone(result['p99_latency_ms'])
            self.assertIsNone(result['system_metrics']['completion_ratio'])
            self.assertEqual(result['system_metrics']['within_window_completion_throughput_rps'],0)
            self.assertEqual(result['stability']['stability_state_v2'],'idle_residency')



if __name__ == '__main__':
    unittest.main()
