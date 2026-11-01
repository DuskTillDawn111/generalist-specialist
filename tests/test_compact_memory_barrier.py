"""CPU-only evidence checks for the complete-calibration memory barrier."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from gsdeploy.profiling.compact_experiments import execute_job, verify_memory_screen
from gsdeploy.profiling.prepare import digest
from gsdeploy.profiling.runner import save


class CalibrationMemoryBarrierTests(unittest.TestCase):
    def fixture(self, root):
        study = root / 'study.json'
        save(study, {'generalists': [{'slot': 'g'}], 'specialists': [{'slot': 's'}],
                     'gpu_memory_mib': 49140, 'safe_factor': 1.110854,
                     'admission_requirements': {'specialist_generalist_memory_ratio': [.5, 2]}})
        manifest = {'study': str(study), 'sources': {}, 'code_hashes': {},
                    'workers': {}, 'jobs': [], 'memory_validation': True}
        protocol = root / 'protocol.json'
        save(protocol, {'diagnostic_only': False})
        for slot in ('g', 's'):
            spec = {'candidate': {'slot': slot}, 'execution': {}, 'resources': {'protocol': str(protocol)}}
            spec_path = root / (slot + '.json')
            save(spec_path, spec)
            manifest['workers'][slot] = {'spec': str(spec_path), 'spec_sha256': digest(spec_path)}
            job = {'id': f'memory-{slot}-n1', 'phase': 'memory', 'exact_cohort': True,
                   'rounds': 1, 'replicas': 1, 'samples': [{'id': 'a', 'task': 'text'}], 'requires': []}
            job_path = root / (job['id'] + '.json')
            save(job_path, job)
            entry = {'id': job['id'], 'phase': 'memory', 'path': str(job_path), 'sha256': digest(job_path)}
            manifest['jobs'].append(entry)
            save(root / 'results' / job['id'] / 'profile.json', {
                'status': 'COMPLETE', 'kind': 'measured', 'scope': 'standalone_same_model_replicas',
                'model': spec, 'replicas': 1, 'telemetry_errors': [],
                'resident_after_warmup_mib': 5000, 'peak_board_used_mib': 6000,
                'manifest_sha256': hashlib.sha256(json.dumps(job['samples'], sort_keys=True).encode()).hexdigest(),
                'rounds': [{'task': 'text', 'repetition': 0, 'completed_requests': 1,
                            'successful_requests': 1, 'failed_requests': 0}]})
        manifest_path = root / 'manifest.json'
        save(manifest_path, manifest)
        for entry in manifest['jobs']:
            save(root / 'results' / entry['id'] / 'receipt.json', {
                'status': 'COMPLETE', 'manifest_sha256': digest(manifest_path), 'job_sha256': entry['sha256']})
        save(root / 'results/memory-screen.json', {'passed': True, 'manifest_sha256': digest(manifest_path),
             'profile_sha256': {slot: digest(root / 'results' / f'memory-{slot}-n1/profile.json') for slot in ('g', 's')}})
        return manifest_path, manifest

    def test_valid_complete_calibration_and_all_pairwise_memory_checks_pass(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, manifest = self.fixture(root)
            verify_memory_screen(path, manifest, root / 'results')

    def test_short_or_failed_cohort_cannot_pass_even_with_updated_profile_hash(self):
        for field, value in [('completed_requests', 0), ('successful_requests', 0), ('failed_requests', 1)]:
            with self.subTest(field=field), TemporaryDirectory() as tmp:
                root = Path(tmp)
                path, manifest = self.fixture(root)
                profile_path = root / 'results/memory-g-n1/profile.json'
                profile = json.loads(profile_path.read_text())
                profile['rounds'][0][field] = value
                save(profile_path, profile)
                screen_path = root / 'results/memory-screen.json'
                screen = json.loads(screen_path.read_text())
                screen['profile_sha256']['g'] = digest(profile_path)
                save(screen_path, screen)
                with self.assertRaisesRegex(ValueError, 'Incomplete or mismatched'):
                    verify_memory_screen(path, manifest, root / 'results')

    def test_receipt_from_another_job_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, manifest = self.fixture(root)
            receipt_path = root / 'results/memory-g-n1/receipt.json'
            receipt = json.loads(receipt_path.read_text())
            receipt['job_sha256'] = 'other'
            save(receipt_path, receipt)
            with self.assertRaisesRegex(ValueError, 'Missing or changed'):
                verify_memory_screen(path, manifest, root / 'results')

    def test_direct_quality_cli_cannot_bypass_missing_pool_screen(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, manifest = self.fixture(root)
            job_path = root / 'quality.json'
            save(job_path, {'phase': 'quality', 'requires': []})
            manifest['jobs'].append({'id': 'quality-g-n1', 'phase': 'quality',
                                     'path': str(job_path), 'sha256': digest(job_path)})
            save(path, manifest)
            (root / 'results/memory-screen.json').unlink()
            with patch('gsdeploy.profiling.compact_experiments.subprocess.run') as process:
                with self.assertRaisesRegex(ValueError, 'Missing successful full-calibration'):
                    execute_job(path, 'quality-g-n1', root / 'results/quality-g-n1')
                process.assert_not_called()
            self.assertFalse((root / 'results/quality-g-n1').exists())


if __name__ == '__main__':
    unittest.main()
