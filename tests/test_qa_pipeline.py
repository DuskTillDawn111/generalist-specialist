"""The replacement must never reach GPU execution with incomplete resources."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_chatqa_replacement.py'
spec = importlib.util.spec_from_file_location('qa_pipeline_under_test', SCRIPT)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class QAPipelineTests(unittest.TestCase):
    def test_incomplete_download_stops_before_preparation_and_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = root / 'artifacts/qa-replacement-20261018/resources/resource-inventory.json'
            inventory.parent.mkdir(parents=True)
            inventory.write_text(json.dumps({'status': 'INCOMPLETE'}))
            with patch.object(pipeline, 'ROOT', root), patch.object(pipeline.subprocess, 'run') as command:
                with self.assertRaisesRegex(RuntimeError, 'hash-verified'):
                    pipeline.run(root / 'output')
                self.assertEqual(command.call_count, 1)
                self.assertIn('gsdeploy.profiling.resources', command.call_args.args[0])
            state = json.loads((root / 'output/state.json').read_text())
            self.assertEqual(state['status'], 'FAILED')
            self.assertFalse(state['gpu_started'])
            self.assertFalse(state['pool_admitted'])

    def test_source_change_during_download_prevents_gpu_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = root / 'artifacts/qa-replacement-20261018/resources/resource-inventory.json'
            inventory.parent.mkdir(parents=True)
            inventory.write_text(json.dumps({'status': 'COMPLETE'}))
            source = root / 'gsdeploy/test.py'
            source.parent.mkdir()
            source.write_text('original')
            with patch.object(pipeline, 'ROOT', root), patch.object(pipeline.subprocess, 'run',
                    side_effect=lambda *a, **k: source.write_text('changed')) as command:
                with self.assertRaisesRegex(ValueError, 'Source changed'):
                    pipeline.run(root / 'output')
                self.assertEqual(command.call_count, 1)
            self.assertFalse(json.loads((root / 'output/state.json').read_text())['gpu_started'])


if __name__ == '__main__':
    unittest.main()
