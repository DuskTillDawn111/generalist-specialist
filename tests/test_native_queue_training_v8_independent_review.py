"""Independent CPU-only queue collector regressions; no real inputs or GPU.

Fixtures contain invented training IDs and no validation labels. Hardware is
mocked only for startup-failure cleanup and budget-wiring checks.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import native_queue_training_v8 as queue
from tools.predictor_validation_20261027.operator_profile_campaign_v2 import stat_receipt


class QueueIndependentReview(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'artifacts').mkdir()
        (self.root / 'gsdeploy').symlink_to(queue.ROOT / 'gsdeploy', target_is_directory=True)
        self.base = self.root / 'inputs'
        self.base.mkdir()
        self.profile = self.root / 'profile'
        self.profile.mkdir()
        inputs = [dict(task=task, pool_id=f'fixture-{task}-{i}', input_size=i + 1)
                  for task in ('text', 'image', 'asr', 'ocr') for i in range(2)]
        self.inputs = inputs
        (self.base / 'runtime-inputs.jsonl').write_text(''.join(
            json.dumps(dict(pool_id=r['pool_id'], task_type=r['task'],
                            sample=dict(id=r['pool_id']))) + '\n' for r in inputs))
        queue.save(self.base / 'catalog.json', dict(safe_factor=1.1, quality=dict(references={})))
        source = self.profile / 'source.py'
        source.write_text('# independent synthetic source\n')
        self.weight = self.profile / 'weight.bin'
        self.weight.write_bytes(b'original fixture weights')
        queue.save(self.profile / 'large-weight-stats.json', [stat_receipt(self.weight)])
        queue.save(self.profile / 'smoke.json', dict(status='PASS', synthetic=True))
        queue.save(self.profile / 'selection.json', dict(split='training', inputs=inputs))
        context = dict(
            residents=queue.RESIDENTS, execution=dict(physical_owners=4),
            sources=[queue.receipt(source)],
            execution_sources=[dict(original_path=str(source), frozen=queue.receipt(source))],
            large_weight_stats=queue.receipt(self.profile / 'large-weight-stats.json'),
            inherited_runtime_smoke=queue.receipt(self.profile / 'smoke.json'),
            worker_bindings={family: dict(command=['python', 'worker.py'], env={})
                             for family in set(queue.RESIDENTS.values())},
            threads={}, worker_variant='semantic_worker.py',
            collection_budget=dict(free_disk_reserve_bytes=12 * 2**30, total_timeout_seconds=2400))
        queue.save(self.profile / 'context.json', context)
        profile = dict(schema='gsdeploy.generic_full_state_resource_training_plan.v5',
                       split='training', training_only=True, domain='natural', inputs=inputs,
                       input_selection=queue.receipt(self.profile / 'selection.json'),
                       inherited_runtime_smoke=queue.receipt(self.profile / 'smoke.json'),
                       execution_context_sha256=queue.context_identity(context))
        profile['plan_sha256'] = queue.digest(profile)
        queue.save(self.profile / 'plan.json', profile)
        self.addCleanup(patch.stopall)
        patch.object(queue, 'BASE', self.base).start()
        patch.object(queue, 'ROOT', self.root).start()

    def prepare(self):
        return queue.prepare(self.profile, self.root / 'prepared', run_id='cpu_fixture')

    @staticmethod
    def reseal(plan):
        plan['plan_sha256'] = queue.digest({k: v for k, v in plan.items() if k != 'plan_sha256'})

    def test_declared_rates_must_equal_fixed_trace_counts(self):
        plan = self.prepare()
        plan['workload_rps'] = dict(text=999., image=1., asr=1., ocr=1.)
        self.reseal(plan)
        with self.assertRaises(ValueError):
            queue.verify_sources(plan)

    def test_candidate_must_bind_actual_execution_context(self):
        plan = self.prepare()
        plan['candidates'][0]['execution_context_sha256'] = '0' * 64
        self.reseal(plan)
        with self.assertRaises(ValueError):
            queue.verify_sources(plan)

    def test_parent_profile_context_mismatch_rejected_before_preparation(self):
        path = self.profile / 'plan.json'
        profile = json.loads(path.read_text())
        profile['execution_context_sha256'] = '0' * 64
        self.reseal(profile)
        path.write_text(json.dumps(profile))
        with self.assertRaises(ValueError):
            self.prepare()

    def test_large_weight_stat_drift_rejected(self):
        plan = self.prepare()
        self.weight.write_bytes(b'changed fixture weights with different size')
        with self.assertRaises(ValueError):
            queue.verify_sources(plan)

    def test_training_cannot_be_relabelled_through_validation_options(self):
        self.prepare()
        for options in (dict(run_id='validation'), dict(protocol_path='unused'),
                        dict(predictions_path='unused'),
                        dict(run_id='validation', protocol_path='unused', predictions_path='unused')):
            with self.subTest(options=options), patch.object(queue, 'DeploymentRuntime') as runtime:
                with self.assertRaisesRegex(ValueError, 'relabeled'):
                    queue.execute(self.root / 'prepared/plan.json', 'shared_declared_training',
                                  self.root / 'forbidden', **options)
                runtime.assert_not_called()
                self.assertFalse((self.root / 'forbidden').exists())

    def run_startup_failure(self, *, broken_close=False):
        self.prepare()

        class Runtime:
            workers = {}
            ready = {}
            errors = []
            shutdown_report = {}
            admission = None
            closed = False

            def start(self):
                raise RuntimeError('synthetic startup failure')

            def close(self):
                self.closed = True

        class Monitor:
            def __init__(self, runtime):
                self.samples = []
                self.errors = []
                self.collection_budget = None

            def sample(self):
                pass

            def check(self):
                pass

            def start(self):
                pass

            def close(self):
                if broken_close:
                    raise RuntimeError('synthetic monitor close failure')

        runtime = Runtime()
        monitor = Monitor(runtime)
        escaped = None
        with patch.object(queue, 'DeploymentRuntime', return_value=runtime), \
             patch.object(queue, 'BoardMonitor', return_value=monitor), \
             patch.object(queue.fcntl, 'flock'):
            try:
                queue.execute(self.root / 'prepared/plan.json', 'shared_declared_training', self.root / 'run')
            except RuntimeError as exc:
                escaped = str(exc)
        return runtime, monitor, escaped

    def test_monitor_close_failure_still_closes_workers_and_preserves_raw(self):
        runtime, _, escaped = self.run_startup_failure(broken_close=True)
        self.assertTrue(runtime.closed, f'Worker close skipped after: {escaped}')
        path = self.root / 'run/raw.json'
        self.assertTrue(path.exists(), 'Startup/cleanup failure evidence must be saved')
        raw = json.loads(path.read_text())
        self.assertEqual(raw['status'], 'failed')
        self.assertTrue(any('monitor close failure' in e for e in raw['errors']))

    def test_inherited_disk_and_time_budget_reaches_monitor(self):
        _, monitor, _ = self.run_startup_failure()
        self.assertEqual(monitor.collection_budget, dict(
            free_disk_reserve_bytes=12 * 2**30, total_timeout_seconds=2400,
            path=str(self.root / 'run')))


if __name__ == '__main__':
    unittest.main()
