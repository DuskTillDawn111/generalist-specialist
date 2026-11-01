"""CPU-only independent observer/supervision and real scoring regressions.

Imports the frozen preflight runner without invoking its worker or supervisor.
All child processes are fakes; no model/provider initialization is performed.
"""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.prediction_gate import _cells
from tools.predictor_validation_20261027.known_input_campaign_seal_v5 import check_complete
from tools.predictor_validation_20261027.prospective_flow_contract_v11 import (
    gate_cell, score_complete_forecast)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'artifacts/predictor-learned-availability-v16-20261028-preflight02/runner.py'


def load_runner():
    spec = importlib.util.spec_from_file_location('independent_preflight02', RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ParallelPreflightIndependentTests(unittest.TestCase):
    def test_two_atomic_progress_updates_preserve_decision_and_provenance(self):
        module = load_runner()
        decision, provenance = object(), {'unchanged': object()}
        scheduler = mock.Mock()
        scheduler.decide.return_value = decision
        scheduler.provenance.return_value = provenance
        view = SimpleNamespace(clock_ns=123, active_gpu_nodes=('a', 'b'))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with mock.patch.object(module, 'OUT', output), mock.patch.object(
                    module.time, 'monotonic', side_effect=[21., 42.]):
                observer = module.ProgressScheduler(scheduler, 'one')
                self.assertIs(observer.decide(view), decision)
                first = json.loads((output / 'progress-one.json').read_text())
                view.clock_ns = 456
                self.assertIs(observer.decide(view), decision)
                second = json.loads((output / 'progress-one.json').read_text())
                self.assertIs(observer.provenance(), provenance)
            self.assertEqual(scheduler.decide.call_count, 2)
            self.assertEqual((first['scheduler_calls'], first['clock_ns']), (1, 123))
            self.assertEqual((second['scheduler_calls'], second['clock_ns']), (2, 456))
            self.assertEqual(list(output.iterdir()), [output / 'progress-one.json'])

    def test_watchdog_and_partial_launch_write_failure_reports_after_cleanup(self):
        module = load_runner()
        for partial_launch in (False, True):
            with self.subTest(partial_launch=partial_launch), tempfile.TemporaryDirectory() as directory:
                processes = []

                class FakeProcess:
                    def __init__(self, command, **kwargs):
                        if partial_launch and processes:
                            raise OSError('independent injected launch failure')
                        self.returncode = None
                        self.command = command
                        self.terminated = False
                        self.waited = False
                        processes.append(self)

                    def poll(self):
                        return self.returncode

                    def terminate(self):
                        self.terminated = True
                        self.returncode = -15

                    def wait(self, timeout=None):
                        self.waited = True
                        return self.returncode

                output = Path(directory)
                with mock.patch.object(module, 'OUT', output), mock.patch.object(
                        module.subprocess, 'Popen', FakeProcess), mock.patch.object(
                        module.time, 'monotonic', side_effect=[0., 7201., 7202.]), \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    module.supervisor()
                report = json.loads((output / 'report.json').read_text())
                self.assertEqual(report['status'], 'PREFLIGHT_SUPERVISOR_FAILED')
                self.assertEqual(report['supervisor_failure']['exception'],
                                 'OSError' if partial_launch else 'TimeoutError')
                self.assertTrue(report['all_direct_workers_reaped'])
                self.assertTrue(all(p.terminated and p.waited for p in processes))
                self.assertEqual(len(processes), 1 if partial_launch else 4)
                self.assertEqual(len(report['candidates']), 4)
                self.assertEqual([r['full_declared_requests'] for r in report['candidates']], [57] * 4)
                self.assertFalse(report['full_four_candidate_forecast_executed'])
                self.assertEqual(report['completeness']['status'], 'INCOMPLETE_PROHIBITS_SEAL_AND_COLLECTION')
                self.assertTrue(report['core_and_public_sources_unchanged'])
                self.assertEqual(json.loads((output / 'source-before.json').read_text()),
                                 json.loads((output / 'source-after.json').read_text()))
                for field in ('GPU_started', 'validation_outcomes_read', 'protocol_sealed',
                              'predictions_sealed', 'independent_accuracy_established'):
                    self.assertFalse(report[field])


class FullScoreIndependentTests(unittest.TestCase):
    def setUp(self):
        self.plan = dict(aliases={'text#0': dict(family='text', supported_tasks=['text']),
                                  'ocr#0': dict(family='ocr', supported_tasks=['ocr'])},
            workload_rps={'text': .1, 'ocr': 2/30},
            latency_reference_ms={'text': 1000., 'ocr': 10000.},
            window_seconds=30., drain_seconds=120., memory_budget_mib=49140.,
            memory_safe_factor=1.110854, energy_reference_watts=450.,
            weights={k: 1/6 for k in 'QTLPME'})
        self.rows = [dict(request_id='synthetic:' + str(i), task=task,
                          pool_id='public:' + str(i), arrival_ns=0, alias=task+'#0')
                     for i, task in enumerate(['text']*3 + ['ocr']*2)]
        times = [100, 200, 1000, 3000, 35000]
        self.forecast = dict(cohort={r['request_id']: r for r in self.rows},
            finishes={r['request_id']: dict(r, start_ns=10_000_000,
                handler_end_ns=30_000_000, finish_ns=int(ms*1e6))
                for r, ms in zip(self.rows, times)},
            board_energy_joules=1350., peak_memory_mib=45000.)
        self.startup = dict(energy_joules=3600., peak_memory_mib=40000.,
            energy_includes_priming=True, safety_factor_applied=False,
            startup_amortization_seconds=3600., warmup_seconds=0.)
        self.quality = {'text': {'text': .2}, 'ocr': {'ocr': {'by_input': {
            'public:3': dict(basis='expected_per_request_totals', numerator=1.,
                             denominator=2., evidence='synthetic training expectation'),
            'public:4': dict(basis='expected_per_request_totals', numerator=0.,
                             denominator=8., evidence='synthetic training expectation')}}}}
        self.candidate = dict(id='synthetic', execution_context_sha256='synthetic-context')
        self.protocol = dict(candidates=[self.candidate],
            validation_domain={'synthetic-run': [{k: r[k] for k in ('request_id', 'task', 'pool_id')}
                                                for r in self.rows]},
            queue_sample_offsets_ns=[0, 10_000_000, 30_000_000_000])

    def score(self):
        return score_complete_forecast(self.forecast, self.rows, self.plan, self.quality,
                                       self.startup, phase_start_ns=0)

    def cell(self, metrics=None):
        result = dict(forecast=self.forecast)
        if metrics is not None:
            result['metrics'] = metrics
        return gate_cell(result, self.rows, self.candidate, 'synthetic-run',
                         self.protocol['queue_sample_offsets_ns'], phase_start_ns=0)

    def test_full_task_p95_parent_receipt_drain_and_single_resource_charges(self):
        metrics = self.score()
        expected = dict(Q=.16, T=.2, L=1.02, P=1.888,
                        M=1.110854*45000/49140, E=1380/(450*30))
        for term, value in expected.items():
            self.assertAlmostEqual(metrics['terms'][term], value, msg=term)
        self.assertAlmostEqual(metrics['per_task']['text']['p95_ms'], 920.)
        self.assertAlmostEqual(metrics['per_task']['ocr']['p95_ms'], 33400.)
        self.assertEqual(metrics['cohort_size'], 5)
        self.assertEqual(metrics['startup_energy_amortized_to_window_joules'], 30.)
        self.assertAlmostEqual(metrics['full_U'], -sum(expected.values())/6)
        cell = self.cell(metrics)
        self.assertEqual(cell['terms'], metrics['terms'])
        self.assertEqual(cell['requests'][-1]['service_ms'], 20.)
        self.assertEqual(cell['requests'][-1]['queue_ms'], 10.)
        self.assertEqual(cell['requests'][-1]['e2e_ms'], 35000.)
        self.assertEqual([r['depth'] for r in cell['queue_depth_samples']], [5, 0, 0])
        self.assertEqual(len(_cells([cell], self.protocol)), 1)
        self.assertEqual(check_complete(self.protocol, dict(cells=[cell]))['status'],
                         'COMPLETE_PRESEALED_PREDICTIONS')
        self.startup['peak_memory_mib'] = 47000.
        self.assertAlmostEqual(self.score()['terms']['M'], 1.110854*47000/49140)

    def test_missing_request_remains_unknown_and_original_drain_is_inclusive(self):
        rid = self.rows[-1]['request_id']
        original = copy.deepcopy(self.forecast['finishes'][rid])
        del self.forecast['finishes'][rid]
        with self.assertRaises(UnknownEvidence):
            self.score()
        cell = self.cell()
        self.assertEqual(len(cell['requests']), 5)
        self.assertIsNone(cell['requests'][-1]['e2e_ms'])
        with self.assertRaises(ValueError):
            check_complete(self.protocol, dict(cells=[cell]))
        self.forecast['finishes'][rid] = original
        self.forecast['finishes'][rid]['finish_ns'] = 150_000_000_000
        self.assertIsNotNone(self.score()['full_U'])
        self.forecast['finishes'][rid]['finish_ns'] += 1
        with self.assertRaisesRegex(UnknownEvidence, 'DRAIN_BOUNDARY'):
            self.score()


if __name__ == '__main__':
    unittest.main()
