"""No GPU: prove phase barriers prevent invalid candidates reaching later jobs."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gsdeploy.profiling.prepare import digest
from gsdeploy.profiling.runner import save
from scripts.run_compact_gpu_queue import run_queue


class GPUQueueTests(unittest.TestCase):
    def setUp(self):
        # These minimal controller fixtures do not represent a model pool.
        # Memory arithmetic/unknown evidence are tested in test_memory_gate.
        memory = patch('gsdeploy.profiling.memory_gate.screen_pool_memory',
                       return_value={'status': 'PRELIMINARY_MEMORY_SCREEN_PASSED', 'passed': True})
        self.memory_screen = memory.start()
        self.addCleanup(memory.stop)

    def fixture(self, root):
        study = root / 'study.json'
        save(study, {'generalists': [], 'specialists': [], 'gpu_memory_mib': 49140,
                     'safe_factor': 1.110854,
                     'admission_requirements': {'specialist_generalist_memory_ratio': [.5, 2]}})
        manifest = {'sources': {}, 'code_hashes': {}, 'jobs': [], 'workers': {}, 'study': str(study)}
        phases = ['smoke', 'quality', 'profile', 'matched_cost']
        for i, phase in enumerate(phases):
            job_id = phase + '-model-n1'
            path = root / (job_id + '.json')
            save(path, {'slot': 'model', 'requires': [phases[i - 1] + '-model-n1'] if i else []})
            manifest['jobs'].append({'id': job_id, 'phase': phase, 'path': str(path), 'sha256': digest(path)})
        path = root / 'manifest.json'
        save(path, manifest)
        return path

    def subprocess(self, manifest, launched, *, fail=None, quality_passes=True):
        data = json.loads(manifest.read_text())
        def run(command, **kwargs):
            job_id = command[command.index('--job') + 1]
            launched.append(job_id)
            output = Path(command[command.index('--output') + 1])
            entry = next(e for e in data['jobs'] if e['id'] == job_id)
            save(output / 'receipt.json', {'status': 'FAILED' if job_id == fail else 'COMPLETE',
                 'manifest_sha256': digest(manifest), 'job_sha256': entry['sha256']})
            if entry['phase'] == 'quality':
                save(output / 'quality.json', {'text': {'gate': {'passed': quality_passes}}})
            return SimpleNamespace(returncode=1 if job_id == fail else 0)
        return run

    def test_failed_smoke_stops_quality_and_keeps_failed_evidence_on_resume(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            launched = []
            output = root / 'results'
            with patch('scripts.run_compact_gpu_queue.subprocess.run', self.subprocess(manifest, launched, fail='smoke-model-n1')):
                result = run_queue(manifest, output, 'matched_cost')
            self.assertEqual(launched, ['smoke-model-n1'])
            self.assertEqual(result['status'], 'NEEDS_SMOKE_REPAIR')
            before = (output / 'smoke-model-n1/receipt.json').read_bytes()
            with patch('scripts.run_compact_gpu_queue.subprocess.run') as process:
                run_queue(manifest, output, 'matched_cost')
                process.assert_not_called()
            self.assertEqual((output / 'smoke-model-n1/receipt.json').read_bytes(), before)

    def test_failed_quality_gate_stops_capacity_and_cost(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            launched = []
            with patch('scripts.run_compact_gpu_queue.subprocess.run', self.subprocess(manifest, launched, quality_passes=False)), \
                    patch('gsdeploy.profiling.compact_experiments.assess', return_value={'all_requirements_passed': False}):
                result = run_queue(manifest, root / 'results', 'matched_cost')
            self.assertEqual(launched, ['smoke-model-n1', 'quality-model-n1'])
            self.assertEqual(result['status'], 'POOL_REJECTED_BY_ABSOLUTE_QUALITY_GATES')
            self.assertFalse(result['pool_admitted'])

    def test_paired_quality_rejection_blocks_capacity_even_when_absolute_gates_pass(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            study_path = root / 'study.json'
            study = json.loads(study_path.read_text())
            study.update(generalists=[{'slot': 'g'}], specialists=[{'slot': 's'}])
            save(study_path, study)
            launched = []
            screening = {'all_requirements_passed': False, 'comparisons': [{'quality': {'passed': False}}]}
            with patch('scripts.run_compact_gpu_queue.subprocess.run', self.subprocess(manifest, launched)), \
                    patch('gsdeploy.profiling.compact_experiments.assess', return_value=screening):
                result = run_queue(manifest, root / 'results', 'matched_cost')
            self.assertEqual(launched, ['smoke-model-n1', 'quality-model-n1'])
            self.assertEqual(result['status'], 'POOL_REJECTED_BY_PAIRED_QUALITY_GATES')

    def test_failed_calibration_quality_stops_before_independent_quality(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            data = json.loads(manifest.read_text())
            path = root / 'memory-model-n1.json'
            save(path, {'slot': 'model', 'requires': ['smoke-model-n1']})
            data['jobs'].insert(1, {'id': 'memory-model-n1', 'phase': 'memory', 'path': str(path), 'sha256': digest(path)})
            data['memory_validation'] = True
            save(manifest, data)
            launched = []
            with patch('scripts.run_compact_gpu_queue.subprocess.run', self.subprocess(manifest, launched)), \
                    patch('gsdeploy.profiling.compact_experiments.verify_memory_screen'), \
                    patch('gsdeploy.profiling.compact_experiments.calibration_quality_screen',
                          return_value={'passed': False, 'status': 'POOL_REJECTED_BY_CALIBRATION_QUALITY_GATES'}):
                result = run_queue(manifest, root / 'results', 'matched_cost')
            self.assertEqual(launched, ['smoke-model-n1', 'memory-model-n1'])
            self.assertEqual(result['status'], 'POOL_REJECTED_BY_CALIBRATION_QUALITY_GATES')

    def test_memory_mismatch_stops_before_any_quality_gpu_job(self):
        self.memory_screen.return_value = {'status': 'POOL_REJECTED_BY_MEMORY_SCREEN', 'passed': False}
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            launched = []
            with patch('scripts.run_compact_gpu_queue.subprocess.run', self.subprocess(manifest, launched)):
                result = run_queue(manifest, root / 'results', 'matched_cost')
            self.assertEqual(launched, ['smoke-model-n1'])
            self.assertEqual(result['status'], 'POOL_REJECTED_BY_MEMORY_SCREEN')
            self.assertFalse(result['pool_admitted'])

    def test_successful_receipts_resume_without_repeating_gpu_work(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            launched = []
            output = root / 'results'
            with patch('scripts.run_compact_gpu_queue.subprocess.run', self.subprocess(manifest, launched)):
                run_queue(manifest, output, 'smoke')
            with patch('scripts.run_compact_gpu_queue.subprocess.run', self.subprocess(manifest, launched)), \
                    patch('gsdeploy.profiling.compact_experiments.assess', return_value={'all_requirements_passed': False}):
                result = run_queue(manifest, output, 'quality')
            self.assertEqual(launched, ['smoke-model-n1', 'quality-model-n1'])
            self.assertEqual(result['status'], 'COMPLETE_THROUGH_QUALITY')

    def test_full_calibration_memory_rejection_stops_quality_after_successful_smoke(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            data = json.loads(manifest.read_text())
            memory_job = root / 'memory-model-n1.json'
            save(memory_job, {'slot': 'model', 'requires': ['smoke-model-n1']})
            data['jobs'].insert(1, {'id': 'memory-model-n1', 'phase': 'memory',
                                   'path': str(memory_job), 'sha256': digest(memory_job)})
            data['memory_validation'] = True
            save(manifest, data)
            self.memory_screen.side_effect = [
                {'status': 'PRELIMINARY_MEMORY_SCREEN_PASSED', 'passed': True},
                {'status': 'POOL_REJECTED_BY_MEMORY_SCREEN', 'passed': False}]
            launched = []
            with patch('scripts.run_compact_gpu_queue.subprocess.run', self.subprocess(manifest, launched)):
                result = run_queue(manifest, root / 'results', 'quality')
            self.assertEqual(launched, ['smoke-model-n1', 'memory-model-n1'])
            self.assertEqual(result['status'], 'POOL_REJECTED_BY_MEMORY_SCREEN')
            self.assertFalse(result['pool_admitted'])

    def test_infeasible_optional_replica_does_not_block_valid_n1_cost(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            data = json.loads(manifest.read_text())
            path = root / 'profile-model-n2.json'
            save(path, {'slot': 'model', 'replicas': 2, 'requires': ['profile-model-n1']})
            data['jobs'].insert(3, {'id': 'profile-model-n2', 'phase': 'profile', 'path': str(path), 'sha256': digest(path)})
            save(manifest, data)
            launched = []
            with patch('scripts.run_compact_gpu_queue.subprocess.run', self.subprocess(manifest, launched, fail='profile-model-n2')), \
                    patch('gsdeploy.profiling.compact_experiments.assess', return_value={'all_requirements_passed': False}):
                result = run_queue(manifest, root / 'results', 'matched_cost')
            self.assertEqual(launched[-1], 'matched_cost-model-n1')
            self.assertNotIn('profile-model-n2', result['completed'])
            self.assertEqual(result['failed'][0]['job'], 'profile-model-n2')


if __name__ == '__main__':
    unittest.main()
