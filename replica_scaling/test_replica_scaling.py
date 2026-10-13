"""CPU-only protocol/regression checks. No characterization measurements."""
from collections import Counter
from dataclasses import replace
from pathlib import Path
import json
import sys
import tempfile
import unittest
from joint_deploy.measurement.deployment_runtime import DeploymentRuntime
from joint_deploy.measurement.metrics import summarize
from replica_scaling.runtime import replica_config, execute, backlog_guard, scalar_metrics

ROOT = Path(__file__).resolve().parents[1]
PLAN = dict(warmup={'seconds': 0}, measurement_seconds=.3, drain_timeout_seconds=2,
            startup_timeout_seconds=3, shutdown_timeout_seconds=2, max_pending_per_replica=100)
MODEL = dict(model_id='fixture', task='asr', launcher=dict(python=sys.executable,
            script=str(ROOT / 'joint_deploy/measurement/tests/protocol_worker.py')))

class ReplicaProtocolTests(unittest.TestCase):
    def test_independent_processes_and_equal_rr_and_clean_release(self):
        config, registry = replica_config(MODEL, 2, 40., 7101, PLAN)
        runtime = DeploymentRuntime(config, registry)
        with tempfile.TemporaryDirectory() as directory:
            pool = Path(directory) / 'pool.jsonl'
            pool.write_text(json.dumps(dict(pool_id='test', task_type='asr', sample={'id': 'test'})) + '\n')
            records = []
            try:
                runtime.start()
                self.assertEqual(len({w.process.pid for w in runtime.workers.values()}), 2)
                window = execute(runtime, config, pool, 'measurement', .3, records, 2., None)
                self.assertEqual(Counter(r['model'] for r in records), {'fixture#0': 6, 'fixture#1': 6})
                self.assertTrue(all(r['status'] == 'completed' for r in records))
                self.assertTrue(all(r['actual_submit_time_ns'] <= r['completion_time_ns'] for r in records))
                self.assertEqual(window['duration_seconds'], .3)
            finally:
                shutdown = runtime.close()
            self.assertTrue(all(v['exit_code'] == 0 and not v['forced'] and v['bye_received'] for v in shutdown.values()))
            self.assertTrue(all(w.process.poll() is not None for w in runtime.workers.values()))

    def test_counts_are_scoped_to_one_and_two(self):
        for count in (0, 3, True, 1.5):
            with self.assertRaises(ValueError):
                replica_config(MODEL, count, 1., 1, PLAN)

    def test_backlog_guard_keeps_full_service_debt_and_ignores_completed(self):
        thresholds = dict(max_outstanding_requests=10, max_outstanding_baseline_service_seconds=1,
                          max_oldest_outstanding_age_seconds=20)
        rows = [dict(status='submitted', scheduled_time_ns=0), dict(status='queued', scheduled_time_ns=0),
                dict(status='completed', scheduled_time_ns=0)]
        evidence = backlog_guard(rows, 2_000_000_000, 600., thresholds)
        self.assertEqual(evidence['outstanding'], 2)
        self.assertEqual(evidence['reasons'], ['max_outstanding_baseline_service_seconds'])
        self.assertAlmostEqual(evidence['baseline_service_seconds'], 1.2)

    def test_primary_throughput_excludes_drain_but_latency_retains_cohort(self):
        config, _ = replica_config(MODEL, 1, 2., 1, {**PLAN, 'measurement_seconds': 1})
        rows = [dict(request_id=str(i), task='asr', model='fixture#0', scheduled_time_ns=t,
                     actual_submit_time_ns=t, completion_time_ns=finish, status='completed',
                     worker_latency_ms=10., worker_queue_wait_ms=0.)
                for i, (t, finish) in enumerate([(1, 100_000_001), (900_000_000, 1_100_000_000)])]
        window = dict(start_time_ns=0, end_time_ns=1_000_000_000, drain_end_time_ns=1_200_000_000, duration_seconds=1)
        per, system, models = summarize(rows, config, window)
        result = dict(request_records=rows, measurement_window=window, system_metrics=system,
                      worker_identities={'fixture#0': {'pid': 99}})
        metrics = scalar_metrics(result, config)
        self.assertEqual(metrics['completed_throughput_rps'], 1.)
        self.assertEqual(metrics['completion_ratio'], 1.)
        self.assertAlmostEqual(metrics['p95_e2e_ms'], 195.)
        self.assertEqual(metrics['per_replica']['fixture#0']['completed_requests'], 2)

    def test_protective_stop_preserves_shortened_window_and_records(self):
        cfg, registry = replica_config(MODEL, 1, 100., 1, {**PLAN, 'measurement_seconds': .5})
        runtime = DeploymentRuntime(cfg, registry)
        with tempfile.TemporaryDirectory() as directory:
            pool = Path(directory) / 'pool.jsonl'
            pool.write_text(json.dumps(dict(pool_id='test', task_type='asr', sample={'id': 'test', 'test_delay_seconds': .05})) + '\n')
            rows = []
            try:
                runtime.start()
                window = execute(runtime, cfg, pool, 'measurement', .5, rows, 50.,
                    dict(max_outstanding_requests=5, max_outstanding_baseline_service_seconds=10,
                         max_oldest_outstanding_age_seconds=20))
                self.assertEqual(window['stop']['kind'], 'protective_backlog_stop')
                self.assertLess(window['duration_seconds'], .5)
                self.assertGreater(len(rows), 0)
                self.assertTrue(all(r['status'] == 'completed' for r in rows))
            finally:
                runtime.close()

if __name__ == '__main__':
    unittest.main()
