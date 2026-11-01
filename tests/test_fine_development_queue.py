"""Bounded development queue hardware/resource stop rules, without GPU calls."""
import copy
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import subprocess
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
path = ROOT/'tools/spillover_context_20261022/run_fine_development.py'
# Compile the actual two pure helpers without executing the GPU launcher's
# frozen-import bootstrap, which must not replace the test suite's gsdeploy.
tree = ast.parse(path.read_text(), filename=str(path))
helpers = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
                           and node.name in {'gpu_idle', 'resources_valid'}], type_ignores=[])
namespace = dict(subprocess=subprocess, time=time)
exec(compile(helpers, str(path), 'exec'), namespace)
queue = SimpleNamespace(**namespace)


class FineQueueTests(unittest.TestCase):
    def test_wrong_gpu_and_foreign_process_are_rejected_without_launch(self):
        with patch.object(queue.subprocess, 'run', return_value=SimpleNamespace(stdout='wrong\n')) as run:
            with self.assertRaisesRegex(RuntimeError, 'UUID'):
                queue.gpu_idle('0', 'expected')
            self.assertEqual(run.call_count, 1)
        with patch.object(queue.subprocess, 'run', side_effect=[SimpleNamespace(stdout='expected\n'), SimpleNamespace(stdout='1234\n')]):
            with self.assertRaisesRegex(RuntimeError, 'occupied'):
                queue.gpu_idle('0', 'expected')
        with patch.object(queue.subprocess, 'run', side_effect=[SimpleNamespace(stdout='expected\n'), SimpleNamespace(stdout='')]):
            self.assertEqual(queue.gpu_idle('0', 'expected')['compute_pids'], [])

    def test_ineligible_utility_can_continue_but_invalid_resources_cannot(self):
        raw = dict(status='completed', preflight=dict(gpu=dict(uuid='expected')),
                   memory_evidence=dict(measured=True, resource_valid=True), energy_evidence=dict(coverage_valid=True),
                   evaluation=dict(utility=dict(U=None, eligible=False)))
        queue.resources_valid(raw, 'expected')
        for group, key in [('memory_evidence','measured'), ('memory_evidence','resource_valid'), ('energy_evidence','coverage_valid')]:
            invalid = copy.deepcopy(raw); invalid[group][key] = False
            with self.assertRaises(RuntimeError): queue.resources_valid(invalid, 'expected')
        invalid = copy.deepcopy(raw); invalid['preflight']['gpu']['uuid'] = 'wrong'
        with self.assertRaisesRegex(RuntimeError, 'hardware'):
            queue.resources_valid(invalid, 'expected')


if __name__ == '__main__': unittest.main()
