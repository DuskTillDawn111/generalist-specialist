"""Sanity-sweep regression tests, including deliberately invalid witnesses/sets."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from deployment_core import Deployment, FeasibleDeployment, ModelProfile, RoutingResult, Workload
from deployment_core.candidate_generator import ScreeningReport
from deployment_core.experiments.stage1_5_sweep import (
    BASE, COMPOSITIONS, RHOS, SCALES, main, run_sweep, validate_monotonicity, validate_routing,
)
from deployment_core.routing_solver import LP_TOLERANCE, ROUTING_REL_TOLERANCE


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.catalog = {
            'qwen': ModelProfile('qwen', frozenset({'text', 'asr'}), 1, {'text': 100, 'asr': 1000}),
            'distilbert': ModelProfile('distilbert', frozenset({'text'}), 1, {'text': 10}),
        }
        self.workload = Workload({'text': 2})

    def witness(self, rates=None, utilization=None, models=('qwen',)):
        return FeasibleDeployment(Deployment(models), 1, RoutingResult(
            rates if rates is not None else {('text', 'qwen'): 2.0},
            utilization if utilization is not None else {'qwen': 0.2}))

    def failures(self, row, workload=None, rho=0.8):
        return validate_routing(row, self.catalog, workload or self.workload, rho, 'test_scenario')

    def test_valid_witness_and_existing_tolerance(self):
        self.assertEqual(self.failures(self.witness()), [])
        rate = 2 * (1 + ROUTING_REL_TOLERANCE / 2)
        self.assertEqual(self.failures(self.witness({('text', 'qwen'): rate}, {'qwen': rate / 10})), [])
        self.assertEqual(self.failures(self.witness(utilization={'qwen': 0.2 + LP_TOLERANCE / 2})), [])

    def test_missing_requests_and_zero_load_leak_are_detected(self):
        for workload, row in ((self.workload, self.witness({('text', 'qwen'): 1})),
                              (Workload({}), self.witness({('text', 'qwen'): 1e-12}, {'qwen': 1e-13}))):
            errors = self.failures(row, workload)
            self.assertIn('routing_conservation', {f['check'] for f in errors})
            self.assertTrue(all(f['scenario'] == 'test_scenario' and 'deployment' in f for f in errors))

    def test_unsupported_and_undeployed_routes_are_detected(self):
        cases = [self.witness({('text', 'distilbert'): 2}),
                 self.witness({('asr', 'distilbert'): 2}, {'distilbert': 0}, ('distilbert',))]
        for row in cases:
            self.assertIn('capability', {f['check'] for f in self.failures(row)})

    def test_negative_and_nonfinite_rates_are_detected(self):
        for rate in (-1e-12, float('nan'), float('inf')):
            row = self.witness({('text', 'qwen'): rate})
            self.assertIn('non_negativity', {f['check'] for f in self.failures(row)})

    def test_recompute_utilization_instead_of_trusting_stored_value(self):
        row = self.witness({('text', 'qwen'): 9}, {'qwen': 0.1})
        errors = self.failures(row, Workload({'text': 9}))
        self.assertIn('capacity', {f['check'] for f in errors})
        self.assertTrue(any(f.get('recomputed_utilization') == 0.9 for f in errors))
        self.assertIn('capacity', {f['check'] for f in self.failures(self.witness(utilization={}))})

    def test_equal_counts_do_not_hide_load_set_swap(self):
        rows = [dict(name='low', completed=True, scale=0.25, feasible_set=[['qwen']]),
                dict(name='high', completed=True, scale=0.50, feasible_set=[['distilbert']])]
        failures = validate_monotonicity(rows, 'scale', expanding=False)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]['deployment'], ['distilbert'])
        self.assertEqual(failures[0]['lower_scale'], 0.25)
        self.assertEqual(failures[0]['higher_scale'], 0.50)
        self.assertEqual(failures[0]['message'], 'MONOTONICITY VIOLATION')

    def test_rho_direction_and_all_pairs(self):
        rows = [dict(name='low', completed=True, rho_max=0.5, feasible_set=[['qwen']]),
                dict(name='mid', completed=True, rho_max=0.6, feasible_set=[]),
                dict(name='high', completed=True, rho_max=0.8, feasible_set=[])]
        failures = validate_monotonicity(rows, 'rho_max', expanding=True)
        self.assertEqual(len(failures), 2)
        self.assertEqual({f['higher_rho_max'] for f in failures}, {0.6, 0.8})
        self.assertTrue(all(f['deployment'] == ['qwen'] for f in failures))

    def test_nested_sets_and_equal_sets_pass(self):
        rows = [dict(name='low', completed=True, scale=0.25, feasible_set=[['qwen'], ['distilbert']]),
                dict(name='mid', completed=True, scale=0.5, feasible_set=[['qwen']]),
                dict(name='high', completed=True, scale=0.75, feasible_set=[['qwen']])]
        self.assertEqual(validate_monotonicity(rows, 'scale', expanding=False), [])

    def test_invalid_witness_fails_command_and_persists_evidence(self):
        # The Stage 1 call is mocked only here to inject a regression; the sweep must reject it.
        report = ScreeningReport(feasible_deployments=[self.witness({('text', 'qwen'): float('nan')})])
        with tempfile.TemporaryDirectory() as folder, patch(
                'deployment_core.experiments.stage1_5_sweep.screen_deployments', return_value=report), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            code = main(['--output-dir', folder])
            self.assertEqual(code, 1)
            summary = json.loads((Path(folder) / 'stage1_5_summary.json').read_text())
            self.assertFalse(summary['routing_validation_passed'])
            self.assertGreater(summary['num_failures'], 0)
            self.assertIn('STAGE 1.5 VALIDATION: FAIL', output.getvalue())
            self.assertIn('deployment', output.getvalue())
            # Non-finite evidence is preserved as text, rather than invalid JSON NaN.
            self.assertIn('nan', json.dumps(summary))

    def test_solver_failure_is_not_treated_as_empty_feasible_set(self):
        with tempfile.TemporaryDirectory() as folder, patch(
                'deployment_core.experiments.stage1_5_sweep.screen_deployments',
                side_effect=RuntimeError('deliberate solver failure')):
            summary = run_sweep(folder)
            self.assertFalse(summary['passed'])
            self.assertFalse(summary['load_monotonicity_passed'])
            self.assertFalse(summary['rho_monotonicity_passed'])
            self.assertEqual(summary['num_completed_scenarios'], 0)
            self.assertEqual(summary['num_failures'], 19)


class ActualSweepTests(unittest.TestCase):
    def test_all_scenarios_artifacts_and_actual_witnesses(self):
        with tempfile.TemporaryDirectory() as folder:
            summary = run_sweep(folder)
            self.assertEqual(summary['num_scenarios'], 19)
            self.assertEqual(summary['num_completed_scenarios'], 19)
            self.assertEqual(summary['num_failures'], 0)
            self.assertTrue(summary['passed'])
            for key in ('load_monotonicity_passed', 'rho_monotonicity_passed',
                        'routing_validation_passed', 'vqa_coverage_passed'):
                self.assertTrue(summary[key], key)
            groups = {}
            for filename in ('load_intensity_sweep.json', 'composition_sweep.json', 'rho_sweep.json'):
                groups[filename] = json.loads((Path(folder) / filename).read_text())['scenarios']
            self.assertEqual([s['scale'] for s in groups['load_intensity_sweep.json']], list(SCALES))
            self.assertEqual([s['rho_max'] for s in groups['rho_sweep.json']], list(RHOS))
            self.assertEqual([s['name'] for s in groups['composition_sweep.json']], list(COMPOSITIONS))
            witnesses = 0
            for rows in groups.values():
                for row in rows:
                    self.assertEqual(row['counts']['all_combinations'], 31)
                    self.assertEqual(len(row['feasible_set']), row['counts']['capacity_feasible'])
                    self.assertEqual(len(row['feasible_deployments']), len(row['feasible_set']))
                    self.assertEqual(row['failures'], [])
                    witnesses += len(row['feasible_deployments'])
            self.assertEqual(summary['num_routing_witnesses'], witnesses)
            self.assertGreater(summary['composition_observations']['distinct_feasible_sets'], 1)
            markdown = (Path(folder) / 'summary.md').read_text()
            self.assertIn('STAGE 1.5 VALIDATION: PASS', markdown)
            self.assertIn('Table 1', markdown)
            self.assertIn('Table 2', markdown)
            self.assertIn('Table 3', markdown)


if __name__ == '__main__':
    unittest.main()
