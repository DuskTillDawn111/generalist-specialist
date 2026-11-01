"""CPU fixtures for honest seed transfer and unchanged original qualification."""
import copy
import json
from pathlib import Path

from tests import test_flow_capacity_catalog as fixtures
import unittest

from tools.predictor_validation_20261027.flow_capacity_catalog import build
from tools.predictor_validation_20261027.flow_capacity_transfer import (
    GENERATION_SCHEMA, _generation, seal, verify_transfer, report)
from tools.predictor_validation_20261027.probe_validation import read, receipt
from tools.predictor_validation_20261027.profiling import digest

NS = 1_000_000_000


class CapacityTransferTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.CapacityCatalogTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.fixture = fixture; self.root = fixture.root
        contract = dict(schema=GENERATION_SCHEMA, task_order=['text'], task_phase_fractions={'text': 0.},
            rng_namespace='cpu-inputs:', request_id_prefix='r')
        # Use the source fixture's genuine training roles and clocks, adapting
        # its trace to the explicit supported CPU generator before any fit.
        for path in fixture.plans:
            plan = read(path); plan['input_generation_contract'] = contract
            plan['trace'][1]['offset_ns'] = 500_000_000
            self.write_plan(path, plan)
            for candidate in plan['candidates']:
                raw_path = self.root / (str(plan['seed']) + '-' + candidate['id']) / 'raw.json'
                raw = read(raw_path); raw['plan'] = receipt(path); raw['trace_sha256'] = digest(plan['trace'])
                row = raw['request_records'][1]
                for field in ('scheduled_time_ns', 'observed_worker_start_time_ns', 'completion_time_ns'):
                    row[field] += 400_000_000
                for field in ('start_perf_ns', 'end_perf_ns'):
                    row['worker_response'][field] += 400_000_000
                self.write(raw_path, raw)
        suite = read(fixture.suite)
        for job in suite['jobs']:
            job['plan'] = receipt(job['plan']['path'])
        self.write(fixture.suite, suite)
        build(fixture.suite, self.root / 'capacity')
        self.manifest = self.root / 'capacity/manifest.json'
        self.plan = read(fixture.plans[0]); self.plan.update(seed=3, kind='prospective_pilot', split='validation')
        self.plan.pop('phase'); self.target = self.root / 'target-plan.json'; self.write_plan(self.target, self.plan)

    def write(self, path, value):
        Path(path).write_text(json.dumps(value)); return Path(path)

    def write_plan(self, path, plan):
        plan.pop('plan_sha256', None); plan['plan_sha256'] = digest(plan); return self.write(path, plan)

    def seal(self):
        return seal(self.target, self.manifest, self.root / 'transfer.json')

    def test_seed_transfer_exact_original_arithmetic_without_catalog_changes(self):
        manifest = read(self.manifest)
        before = {row['catalog']['path']: Path(row['catalog']['path']).read_bytes() for row in manifest['conditions']}
        value = self.seal()
        self.assertEqual(len(value['candidates']), 2)
        self.assertTrue(verify_transfer(self.root / 'transfer.json'))
        for row in value['candidates']:
            original = row['original_target_resolver']; estimate = row['transfer']
            self.assertEqual(original['status'], 'UNKNOWN')
            self.assertEqual(original['performance_evidence']['reason'], 'MISSING_COLOCATION_PROFILE')
            self.assertAlmostEqual(estimate['rho_by_family']['text'], .3)
            self.assertEqual(estimate['service_time_ms']['text']['text'], 225.)
            self.assertEqual(estimate['capacity_factors']['text'], 1.5)
            self.assertTrue(row['provisional_transfer_planning_filter_passed'])
            self.assertIsNone(row['original_rho_admission_passed'])
            self.assertIsNone(row['target_applicability_validated'])
            self.assertFalse(estimate['physical_capacity_bound'])
        self.assertEqual(len({row['transfer']['condition_sha256'] for row in value['candidates']}), 2)
        for path, data in before.items():
            self.assertEqual(Path(path).read_bytes(), data)
            self.assertEqual(len(read(path)['colocation_profiles']), 2)

    def test_horizon_load_resource_and_input_generator_mismatch_rejected(self):
        for field, update in (
                ('window_seconds', 600.), ('drain_seconds', 2.), ('warmup_seconds', .1),
                ('workload_rps', {'text': 3.}), ('arrival_process', 'poisson')):
            with self.subTest(field=field):
                plan = copy.deepcopy(self.plan); plan[field] = update; self.write_plan(self.target, plan)
                with self.assertRaises(ValueError):
                    seal(self.target, self.manifest, self.root / ('bad-' + field + '.json'))
        plan = copy.deepcopy(self.plan); plan['candidates'][0]['resource_configuration'] = {'mode': 'request_admission'}
        self.write_plan(self.target, plan)
        with self.assertRaisesRegex(ValueError, 'full training condition'):
            self.seal()
        plan = copy.deepcopy(self.plan); plan['input_generation_contract']['rng_namespace'] = 'another-law:'
        self.write_plan(self.target, plan)
        with self.assertRaisesRegex(ValueError, 'full training condition'):
            self.seal()

    def test_trace_or_route_cannot_be_substituted_under_equal_domain(self):
        for field, value in [('pool_id', 'unseen'), ('offset_ns', 1)]:
            plan = copy.deepcopy(self.plan); plan['trace'][0][field] = value; self.write_plan(self.target, plan)
            with self.assertRaisesRegex(ValueError, 'input generation'):
                self.seal()
        plan = copy.deepcopy(self.plan); plan['candidates'][0]['routes']['r0'] = 's1'
        self.write_plan(self.target, plan)
        with self.assertRaisesRegex(ValueError, 'routes do not match'):
            self.seal()

    def test_training_seed_is_not_an_independent_target(self):
        self.plan['seed'] = 1; self.write_plan(self.target, self.plan)
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            self.seal()

    def test_transfer_filter_can_fail_original_threshold_without_physical_admission(self):
        for seed in (1, 2):
            for label in ('balanced', 'skewed'):
                path = self.root / (str(seed) + '-' + label) / 'raw.json'; raw = read(path)
                for row in raw['request_records']:
                    end = row['observed_worker_start_time_ns'] + 900_000_000
                    row['worker_response']['end_perf_ns'] = end; row['completion_time_ns'] = end + 1
                self.write(path, raw)
        build(self.fixture.suite, self.root / 'slow-capacity')
        value = seal(self.target, self.root / 'slow-capacity/manifest.json', self.root / 'transfer.json')
        for row in value['candidates']:
            self.assertAlmostEqual(row['transfer']['rho_by_family']['text'], .9)
            self.assertEqual(row['transfer']['threshold'], .8)
            self.assertEqual(row['transfer']['tolerance'], 1e-8)
            self.assertFalse(row['provisional_transfer_planning_filter_passed'])
            self.assertIsNone(row['original_rho_admission_passed'])
            self.assertIsNone(row['target_applicability_validated'])

    def test_transitive_native_raw_tamper_is_rejected_even_if_catalog_unchanged(self):
        self.seal()
        raw_path = self.root / '1-balanced/raw.json'; raw = read(raw_path)
        raw['request_records'][0]['worker_response']['end_perf_ns'] += 10
        self.write(raw_path, raw)
        with self.assertRaisesRegex(ValueError, 'source receipt'):
            verify_transfer(self.root / 'transfer.json')

    def test_rehashed_estimate_tamper_is_recomputed_not_trusted(self):
        value = self.seal(); value['candidates'][0]['transfer']['rho_by_family']['text'] = .01
        value.pop('transfer_sha256'); value['transfer_sha256'] = digest(value)
        self.write(self.root / 'transfer.json', value)
        with self.assertRaisesRegex(ValueError, 'recomputed'):
            verify_transfer(self.root / 'transfer.json')

    def target_raw(self, sealed):
        raw = read(self.root / '1-balanced/raw.json')
        shift = sealed['created_ns'] + NS - raw['startup_begin_ns']
        for field in ('startup_begin_ns', 'startup_end_ns', 'phase_start_ns', 'deadline_ns', 'observation_end_ns'):
            raw[field] += shift
        for sample in raw['board_samples']:
            sample['time_ns'] += shift
        for row in raw['request_records']:
            for field in ('scheduled_time_ns', 'observed_worker_start_time_ns', 'completion_time_ns'):
                row[field] += shift
            for field in ('start_perf_ns', 'end_perf_ns'):
                row['worker_response'][field] += shift
            row['actual_submit_time_ns'] = row['scheduled_time_ns'] + 1
            row['worker_queue_wait_ms'] = 0.
        raw['plan'] = receipt(self.target)
        return raw

    def test_independent_report_checks_before_startup_and_never_promotes_admission(self):
        sealed = self.seal(); raw = self.target_raw(sealed); raw_path = self.write(self.root / 'target-raw.json', raw)
        result = report(self.root / 'transfer.json', [raw_path], self.root / 'report.json')
        self.assertEqual(result['status'], 'INCOMPLETE')
        row = result['candidates'][0]
        service = row['independent_service_comparison']['text']['text']
        self.assertEqual(service['predicted_worker_service_ms'], 300.)
        self.assertEqual(service['observed_mean_worker_service_ms'], 200.)
        self.assertEqual(service['run_mean_relative_error'], .5)
        self.assertIsNone(row['original_capacity_and_stability_passed'])
        self.assertIsNone(row['target_applicability_validated'])
        self.assertFalse(row['final_gain_eligible'])
        raw['startup_begin_ns'] = sealed['created_ns']; self.write(raw_path, raw)
        with self.assertRaisesRegex(ValueError, 'before actual target startup'):
            report(self.root / 'transfer.json', [raw_path], self.root / 'late-report.json')

    def test_native_generator_rule_is_recognized_and_rebuilt_without_gpu(self):
        runner = Path(__file__).resolve().parents[1] / 'tools/predictor_validation_20261027/native_flow.py'
        plan = dict(seed=3, runner_source=receipt(runner), profile_plan={'path': 'unused', 'sha256': 'fixture'},
            context={'path': 'unused', 'sha256': 'fixture'}, catalog={'path': 'unused', 'sha256': 'fixture'},
            workload_rps=dict(text=1., image=1., asr=1., ocr=1.), window_seconds=1.,
            trace=[dict(request_id='measurement:' + str(i), offset_ns=offset, task=task, pool_id=task)
                   for i, (task, offset) in enumerate([('text', 0), ('image', 190_000_000),
                       ('asr', 470_000_000), ('ocr', 730_000_000)])])
        spec = {'known_input_domain': {task: [task] for task in plan['workload_rps']}}
        self.assertEqual(_generation(plan, spec)['contract']['schema'], GENERATION_SCHEMA)
        plan['trace'][1]['offset_ns'] += 1
        with self.assertRaisesRegex(ValueError, 'trace does not match'):
            _generation(plan, spec)


if __name__ == '__main__':
    unittest.main()
