"""CPU integration checks for the actual six-job preparation code.

Checkpoint reads and the broad external binding walk are mocked. Frozen CPU
config, sample and priming logic execute unchanged; imports are restored so this
test cannot replace another test's gsdeploy package in sys.modules.
"""
from collections import Counter
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'tools/spillover_capacity_20261022'
SOURCE = ROOT / 'artifacts/static-study-gpu-20261022/execution-source-v5'


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CapacityPreparation(unittest.TestCase):
    def setUp(self):
        self.original_modules = {name: value for name, value in sys.modules.items()
                                 if name == 'gsdeploy' or name.startswith('gsdeploy.')}
        self.original_path = list(sys.path)
        for name in self.original_modules:
            del sys.modules[name]
        self.addCleanup(self.restore_imports)
        self.prepare = module_at('capacity_preparation_under_test', DIRECTORY / 'prepare.py')
        self.runner = module_at('capacity_preparation_validation', DIRECTORY / 'run.py')
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.base = self.root / 'diagnostic'
        original = json.loads((ROOT / 'artifacts/static-study-gpu-20261022/execution-packet-v2.json').read_text())
        self.original = original
        packet_path = self.root / 'artifacts/static-study-gpu-20261022/execution-packet-v2.json'
        packet_path.parent.mkdir(parents=True)
        packet_path.write_text(json.dumps(original))
        doc = self.root / 'docs/spillover_recovery_experiment_plan_20261022.md'
        doc.parent.mkdir(parents=True)
        doc.write_text('Isolated preparation test plan.\n')

        def bind(packet, job, output):
            output.mkdir(parents=True)
            config = copy.deepcopy(packet['config'])
            config['catalog']['measurement_context']['runtime_checkpoint_sha256'] = {
                family: 'test-checkpoint-' + family for family in job['plan']['deployment']}
            manifest = output / 'quality-evidence.json'
            manifest.write_text(json.dumps({'schema_version': 2, 'contract': 'quality_evidence_v1'}))
            return config, manifest, copy.deepcopy(packet['workers'])

        def bindings(inputs, source):
            paths = [inputs / 'runtime-inputs.jsonl', *(source / 'gsdeploy').rglob('*.py')]
            return [{'path': str(path), 'sha256': self.prepare.sha(path)} for path in paths]

        with mock.patch.object(self.prepare, 'ROOT', self.root), \
                mock.patch.object(self.prepare, 'BASE', self.base), \
                mock.patch.object(self.prepare, 'bind_native_job', side_effect=bind) as native, \
                mock.patch.object(self.prepare, '_bindings', side_effect=bindings), \
                contextlib.redirect_stdout(io.StringIO()):
            self.prepare.main()
        self.assertEqual(native.call_count, 1)
        self.packet = json.loads((self.base / 'experiment-packet.json').read_text())
        self.receipt = json.loads((self.base / 'preparation-receipt.json').read_text())

    def restore_imports(self):
        for name in list(sys.modules):
            if name == 'gsdeploy' or name.startswith('gsdeploy.'):
                del sys.modules[name]
        sys.modules.update(self.original_modules)
        sys.path[:] = self.original_path

    def test_six_jobs_total_cohort_and_balanced_order(self):
        jobs = self.packet['jobs']
        self.assertEqual([(j['block'], j['state']) for j in jobs],
                         [(1, 'A'), (1, 'B'), (1, 'C'), (2, 'C'), (2, 'B'), (2, 'A')])
        count = sum(len(rows) for job in jobs for rows in job['planned_requests'].values())
        self.assertEqual(count, 1152)
        self.assertEqual(self.receipt['planned_measured_requests'], count)
        self.assertFalse(self.packet['gpu_started'])
        self.assertIsNone(self.packet['U'])

    def test_each_alias_has_identical_full_input_repetitions(self):
        for job in self.packet['jobs']:
            order = job['input_pool_ids']
            self.assertEqual(len(Counter(order)), 32)
            self.assertEqual(set(Counter(order).values()), {3})
            for repetition in range(3):
                self.assertEqual(len(set(order[repetition*32:(repetition+1)*32])), 32)
            for alias, rows in job['planned_requests'].items():
                self.assertEqual([r['pool_id'] for r in rows], order)
                self.assertTrue(all(r['model'] == alias and r['task'] == 'text' for r in rows))

    def test_same_block_sequence_and_independent_blocks(self):
        jobs = self.packet['jobs']
        for block in (1, 2):
            sequences = [j['input_pool_ids'] for j in jobs if j['block'] == block]
            self.assertTrue(all(sequence == sequences[0] for sequence in sequences))
        self.assertNotEqual(jobs[0]['input_pool_ids'], jobs[3]['input_pool_ids'])

    def test_full_resident_priming_has_twenty_one_bound_requests(self):
        priming = self.packet['priming_requests']
        self.assertEqual(len(priming), 21)
        self.assertEqual(Counter((r['model'], r['task']) for r in priming), Counter({
            ('generalist_2#0', 'text'): 3, ('generalist_2#0', 'image'): 3,
            ('generalist_2#0', 'ocr'): 3, ('generalist_2#0', 'asr'): 3,
            ('ocr#0', 'ocr'): 3, ('text#0', 'text'): 3, ('text#1', 'text'): 3}))
        self.assertEqual(self.packet['resident_deployment'], {'text': 2, 'generalist_2': 1, 'ocr': 1})
        config = json.loads(Path(self.packet['system_config']).read_text())
        pool = self.prepare.SamplePool(self.packet['samples'], ['text', 'image', 'ocr', 'asr'], 101)
        expected = []
        for alias, family in sorted(self.packet['aliases'].items()):
            for task in config['catalog']['models'][family]['supported_tasks']:
                expected.extend({'model': alias, 'task': task, 'pool_id': item['pool_id']}
                                for item in self.prepare.largest_inputs(pool, task, 3))
        self.assertEqual(priming, expected)

    def test_frozen_weights_quality_and_original_pool_unchanged(self):
        config = json.loads(Path(self.packet['system_config']).read_text())
        self.assertEqual(config['weights'], self.original['config']['weights'])
        self.assertEqual(config['catalog']['quality'], self.original['config']['catalog']['quality'])
        self.assertEqual(config['catalog']['models'], self.original['config']['catalog']['models'])
        self.assertEqual(config['catalog']['safe_factor'], self.original['config']['catalog']['safe_factor'])
        self.assertEqual(self.packet['samples'], str(Path(self.original['inputs']) / 'runtime-inputs.jsonl'))

    def test_generated_packet_passes_real_runner_validation(self):
        result = self.runner.validate_packet(self.packet)
        self.assertTrue(result['valid'])
        self.assertEqual(result['jobs'], 6)

    def test_job_identity_tamper_rejected(self):
        tampered = copy.deepcopy(self.packet)
        tampered['jobs'][0]['planned_requests']['text#0'][0]['pool_id'] = 'tampered'
        with self.assertRaisesRegex(ValueError, 'identity|ordering'):
            self.runner.validate_packet(tampered)

    def test_bound_config_tamper_rejected(self):
        with Path(self.packet['system_config']).open('a') as stream:
            stream.write(' ')
        with self.assertRaisesRegex(ValueError, 'Binding changed'):
            self.runner.validate_packet(self.packet)


if __name__ == '__main__':
    unittest.main()
