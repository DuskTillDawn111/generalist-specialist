"""CPU-only acceptance tests and independently checked routing witnesses."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from deployment_core import (
    Deployment, Hardware, ModelProfile, Workload, generate_feasible_deployments,
    load_model_catalog, screen_deployments,
)
from deployment_core.candidate_generator import enumerate_deployments
from deployment_core.models import load_hardware
from deployment_core.routing_solver import RoutingSolverError, solve_capacity_lp
from deployment_core.workload import parse_workload

ROOT = Path(__file__).resolve().parents[2]


class FeasibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_model_catalog()
        cls.hardware = load_hardware()

    def check_report(self, report, workload, hardware=None, rho_max=0.8):
        hardware = hardware or self.hardware
        counts = report.counts
        self.assertEqual(counts['all_combinations'], 31)
        self.assertEqual(counts['coverage_feasible'] + counts['coverage_rejected'], 31)
        self.assertEqual(counts['memory_feasible'] + counts['memory_rejected'], counts['coverage_feasible'])
        self.assertEqual(counts['capacity_feasible'] + counts['capacity_rejected'], counts['memory_feasible'])
        self.assertEqual(len(report.feasible_deployments), counts['capacity_feasible'])
        all_names = [r.deployment.models for r in report.feasible_deployments + report.rejected_deployments]
        self.assertEqual(len(set(all_names)), 31)
        for row in report.feasible_deployments:
            self.assertLessEqual(row.memory_mib, hardware.gpu_memory_mib)
            self.assertEqual(row.memory_mib, sum(self.catalog[m].memory_mib for m in row.deployment.models))
            for task, rate in workload.arrival_rate.items():
                routed = sum(v for (k, _), v in row.routing.routing_rps.items() if k == task)
                self.assertAlmostEqual(routed, rate, places=7)
            for (task, model), rate in row.routing.routing_rps.items():
                self.assertIn(model, row.deployment.models)
                self.assertIn(task, self.catalog[model].supported_tasks)
                self.assertGreaterEqual(rate, 0)
                self.assertLessEqual(rate, workload.arrival_rate[task])
            for model in row.deployment.models:
                rho = sum(rate * self.catalog[m].service_time_ms[task] / 1000
                          for (task, m), rate in row.routing.routing_rps.items() if m == model)
                self.assertAlmostEqual(row.routing.utilization[model], rho, places=10)
                self.assertLessEqual(rho, rho_max + 1e-9)

    def test_all_31_nonempty_binary_combinations(self):
        deployments = list(enumerate_deployments(self.catalog))
        self.assertEqual(len(deployments), 31)
        self.assertEqual(len(set(deployments)), 31)
        self.assertEqual(deployments, sorted(deployments, key=lambda d: (len(d.models), d.models)))
        self.assertTrue(all(d.models for d in deployments))

    def test_a_text_only_allows_each_executor(self):
        workload = Workload({'text': 10})
        report = screen_deployments(self.catalog, workload, self.hardware)
        self.check_report(report, workload)
        names = {r.deployment.models for r in report.feasible_deployments}
        self.assertIn(('qwen',), names)
        self.assertIn(('distilbert',), names)
        self.assertIn(('distilbert', 'qwen'), names)
        self.assertEqual(report.counts['coverage_rejected'], 7)
        self.assertEqual(report.counts['capacity_feasible'], 24)

    def test_b_vqa_requires_qwen(self):
        workload = Workload({'visual_qa': 1})
        report = screen_deployments(self.catalog, workload, self.hardware)
        self.check_report(report, workload)
        self.assertEqual(report.counts['coverage_feasible'], 16)
        self.assertEqual(report.counts['capacity_feasible'], 16)
        self.assertTrue(all('qwen' in r.deployment.models for r in report.feasible_deployments))
        self.assertTrue(all('visual_qa' in r.reason for r in report.rejected_deployments))

    def test_c_asr_capacity_rejects_qwen_but_accepts_whisper(self):
        workload = Workload({'asr': 2})
        report = screen_deployments(self.catalog, workload, self.hardware)
        self.check_report(report, workload)
        self.assertEqual(report.counts['capacity_rejected'], 8)
        names = {r.deployment.models for r in report.feasible_deployments}
        self.assertNotIn(('qwen',), names)
        self.assertIn(('whisper',), names)
        self.assertIn(('qwen', 'whisper'), names)

    def test_asr_load_beyond_joint_capacity(self):
        workload = Workload({'asr': 100})
        report = screen_deployments(self.catalog, workload, self.hardware)
        self.check_report(report, workload)
        self.assertEqual(report.counts['capacity_feasible'], 0)
        self.assertEqual(report.counts['capacity_rejected'], 24)

    def test_d_synthetic_memory_limit(self):
        workload, hardware = Workload({'text': 10}), Hardware(750)
        report = screen_deployments(self.catalog, workload, hardware)
        self.check_report(report, workload, hardware)
        self.assertEqual([r.deployment.models for r in report.feasible_deployments], [('distilbert',)])
        self.assertEqual(report.counts['memory_rejected'], 23)
        self.assertTrue(all('memory' in r.reason for r in report.rejected_deployments if r.stage == 'memory'))

    def test_requested_six_task_example(self):
        workload = parse_workload('text=10,image=5,ocr=3,asr=2,visual_qa=1,lid=1')
        report = screen_deployments(self.catalog, workload, self.hardware)
        self.check_report(report, workload)
        self.assertTrue(report.feasible_deployments)
        self.assertTrue(report.counts['capacity_rejected'])
        self.assertEqual(generate_feasible_deployments(self.catalog, workload, self.hardware),
                         report.feasible_deployments)

    def test_split_routing_is_required(self):
        # Neither model alone can handle 1.5 req/s; the joint feasible solution must split.
        catalog = {m: ModelProfile(m, frozenset({'text'}), 1, {'text': 1000}) for m in ('a', 'b')}
        workload = Workload({'text': 1.5})
        report = screen_deployments(catalog, workload, Hardware(2))
        self.assertEqual(report.counts['capacity_rejected'], 2)
        self.assertEqual(len(report.feasible_deployments), 1)
        rates = report.feasible_deployments[0].routing.routing_rps
        self.assertAlmostEqual(sum(rates.values()), 1.5)
        self.assertTrue(all(0.7 - 1e-9 <= v <= 0.8 + 1e-9 for v in rates.values()))

    def test_tasks_have_distinct_service_demands_on_shared_model(self):
        catalog = {'m': ModelProfile('m', frozenset({'text', 'asr'}), 1,
                                     {'text': 100, 'asr': 1000})}
        deployment = Deployment(('m',))
        self.assertIsNotNone(solve_capacity_lp(deployment, catalog, Workload({'text': 7, 'asr': 0.1})))
        self.assertIsNone(solve_capacity_lp(deployment, catalog, Workload({'text': 1, 'asr': 0.8})))

    def test_rho_max_changes_feasibility(self):
        workload = Workload({'text': 10})
        self.assertIsNotNone(solve_capacity_lp(Deployment(('qwen',)), self.catalog, workload, 0.8))
        self.assertIsNone(solve_capacity_lp(Deployment(('qwen',)), self.catalog, workload, 0.4))

    def test_idle_workload_keeps_nonempty_search_space(self):
        workload = Workload({})
        report = screen_deployments(self.catalog, workload, self.hardware)
        self.check_report(report, workload)
        self.assertEqual(report.counts['capacity_feasible'], 31)
        self.assertTrue(all(not row.routing.routing_rps for row in report.feasible_deployments))

    def test_existing_measurement_provenance(self):
        self.assertEqual(self.hardware.gpu_memory_mib, 49140)
        self.assertEqual(sum(m.memory_mib for m in self.catalog.values()), 21136)
        self.assertEqual(sum(len(m.service_time_ms) for m in self.catalog.values()), 11)
        for model in self.catalog.values():
            source = model.sources['memory']
            raw = json.loads(Path(source['path']).read_text())
            self.assertEqual(model.memory_mib, raw[source['field']])
            for task, source in model.sources['service_time'].items():
                weighted, count = 0, 0
                self.assertEqual(len(source['summaries']), 3)
                for reference in source['summaries']:
                    raw = json.loads(Path(reference['path']).read_text())
                    weighted += raw['mean_worker_latency_ms'] * raw['target_count']
                    count += raw['target_count']
                self.assertEqual(count, 90)
                self.assertAlmostEqual(model.service_time_ms[task], weighted / count)

    def test_invalid_inputs(self):
        for value in ('text=-1', 'text=nan', 'text=inf', 'vqa=1', 'text=1,text=2', '', 'text'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_workload(value)
        for rho in (0, -0.1, 1.1, float('nan'), True):
            with self.assertRaises(ValueError):
                screen_deployments(self.catalog, Workload({}), self.hardware, rho)
        with self.assertRaises(ValueError):
            ModelProfile('m', frozenset({'text'}), 1, {})
        with self.assertRaises(ValueError):
            ModelProfile('m', frozenset({'text'}), 1, {'text': 0})
        with self.assertRaises(ValueError):
            Deployment(('qwen', 'qwen'))
        with self.assertRaises(ValueError):
            Hardware(-1)
        with self.assertRaises(TypeError):
            Workload({'text': 1}).arrival_rate['text'] = -1

    def test_solver_error_is_not_misreported_as_infeasibility(self):
        with patch('deployment_core.routing_solver.linprog',
                   return_value=SimpleNamespace(status=4, success=False, message='numerical failure')):
            with self.assertRaises(RoutingSolverError):
                solve_capacity_lp(Deployment(('qwen',)), self.catalog, Workload({'text': 1}))

    def test_zero_objective_and_witness_verification(self):
        with patch('deployment_core.routing_solver.linprog', return_value=SimpleNamespace(
                status=0, success=True, x=np.array([0.0]))) as lp:
            with self.assertRaisesRegex(RoutingSolverError, 'conservation'):
                solve_capacity_lp(Deployment(('qwen',)), self.catalog, Workload({'text': 1}))
            self.assertTrue(np.all(lp.call_args.kwargs['c'] == 0))
            self.assertEqual(lp.call_args.kwargs['method'], 'highs')

    def test_cli_outputs_all_witnesses_and_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result.json'
            proc = subprocess.run([sys.executable, '-m', 'deployment_core.solve', '--workload',
                                   'text=10,image=5,ocr=3,asr=2,visual_qa=1,lid=1',
                                   '--output', str(output)], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            data = json.loads(output.read_text())
            self.assertEqual(len(data['feasible_deployments']) + len(data['rejected_deployments']), 31)
            self.assertEqual(proc.stdout.count('Memory:'), len(data['feasible_deployments']))
            self.assertNotIn('best', proc.stdout.lower())
            self.assertNotIn('best', data)
            self.assertNotIn('score', data)
            self.assertTrue(all('routing_rps' in row for row in data['feasible_deployments']))


if __name__ == '__main__':
    unittest.main()
