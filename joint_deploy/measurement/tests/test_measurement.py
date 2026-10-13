from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

from joint_deploy.measurement.schema import MeasurementConfig
from joint_deploy.measurement.router import Router
from joint_deploy.measurement.load_generator import periodic_arrivals, phases_for, SamplePool
from joint_deploy.measurement.metrics import summarize
from joint_deploy.measurement.run_measurement import run_measurement, DEFAULT_PROFILE
from joint_deploy.measurement.deployment_runtime import DeploymentRuntime

CAPS = {'qwen': ['text', 'asr', 'lid', 'visual_qa'], 'distilbert': ['text'],
        'whisper': ['asr', 'lid'], 'other_generalist': ['text', 'asr']}


def config(**changes):
    values = dict(deployment={'qwen': 1, 'distilbert': 1}, workload_rps={'text': 4},
                  routing_rps={'text': {'qwen': 1, 'distilbert': 3}},
                  warmup_seconds=0, measurement_seconds=1, drain_timeout_seconds=2,
                  shutdown_timeout_seconds=1, startup_timeout_seconds=3)
    return MeasurementConfig(**{**values, **changes})


class UnitTests(unittest.TestCase):
    def test_schema_counts_multiple_generalists_and_runtime_limit(self):
        cfg = config(deployment={'qwen': 2, 'other_generalist': 1},
                     routing_rps={'text': {'qwen': 2, 'other_generalist': 2}})
        cfg.validate(CAPS)
        self.assertEqual(cfg.as_dict()['deployment']['qwen'], 2)
        with self.assertRaisesRegex(ValueError, 'prototype'):
            DeploymentRuntime(cfg).start()

    def test_invalid_routing_and_counts(self):
        for cfg in (config(deployment={'qwen': True}), config(deployment={'qwen': -1}),
                    config(routing_rps={'text': {'qwen': 3}}),
                    config(routing_rps={'text': {'qwen': float('nan')}}),
                    config(routing_rps={'text': {'qwen': -1, 'distilbert': 5}}),
                    config(routing_rps={'text': {'whisper': 4}}),
                    config(deployment={'distilbert': 1}), config(arrival_process='poisson')):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                cfg.validate(CAPS)

    def test_arrivals_counts_spacing_phases_and_seed(self):
        rates = {'asr': 1, 'lid': 2, 'visual_qa': 0.5}
        events = list(periodic_arrivals(rates, 20, 42))
        self.assertEqual(Counter(task for _, task in events), {'asr': 20, 'lid': 40, 'visual_qa': 10})
        self.assertEqual(events, list(periodic_arrivals(rates, 20, 42)))
        self.assertEqual(events, sorted(events))
        self.assertNotEqual(events, list(periodic_arrivals(rates, 20, 43)))
        phases = phases_for(rates, 42)
        self.assertEqual(len(set(phases.values())), 3)
        for task, rate in rates.items():
            offsets = [t for t, k in events if k == task]
            self.assertGreater(offsets[0], 0)
            self.assertLess(offsets[0], 1 / rate)
            self.assertTrue(all(abs(b - a - 1 / rate) < 1e-10 for a, b in zip(offsets, offsets[1:])))

    def test_deterministic_router_counts(self):
        router = Router({'text': {'qwen': 1, 'distilbert': 3}, 'asr': {'qwen': 0.2, 'whisper': 0.8}})
        self.assertEqual(Counter(router.choose('text') for _ in range(80)), {'qwen': 20, 'distilbert': 60})
        self.assertEqual(Counter(router.choose('asr') for _ in range(20)), {'qwen': 4, 'whisper': 16})
        for n in range(1, 30):
            r = Router({'text': {'qwen': 1, 'distilbert': 3}})
            counts = Counter(r.choose('text') for _ in range(n))
            self.assertLessEqual(abs(counts['qwen'] - n / 4), 1)

    def test_metrics_distinguish_delay_runtime_service_and_drain(self):
        rows = [dict(request_id='measurement:0', task='text', model='qwen', status='completed',
                     scheduled_time_ns=0, actual_submit_time_ns=10_000_000,
                     completion_time_ns=110_000_000, worker_latency_ms=25,
                     worker_queue_wait_ms=70, submit_write_completed_time_ns=11_000_000),
                dict(request_id='measurement:1', task='text', model='distilbert', status='completed',
                     scheduled_time_ns=900_000_000, actual_submit_time_ns=910_000_000,
                     completion_time_ns=1_110_000_000, worker_latency_ms=None),
                dict(request_id='measurement:2', task='text', model='distilbert', status='incomplete',
                     scheduled_time_ns=950_000_000, actual_submit_time_ns=None, completion_time_ns=None),
                dict(request_id='measurement:3', task='text', model='distilbert', status='error',
                     scheduled_time_ns=970_000_000, actual_submit_time_ns=None, completion_time_ns=None)]
        task, system, models = summarize(rows, config(), dict(start_time_ns=0, end_time_ns=1_000_000_000,
                                                             drain_end_time_ns=2_000_000_000))
        self.assertEqual(rows[0]['e2e_latency_ms'], 110)
        self.assertEqual(rows[0]['runtime_latency_ms'], 100)
        self.assertEqual(rows[0]['schedule_delay_ms'], 10)
        self.assertEqual(task['text']['mean_worker_latency_ms'], 25)
        self.assertEqual(task['text']['worker_latency_ms_observations'], 1)
        self.assertEqual(system['completed_requests'], 2)
        self.assertEqual(system['incomplete_requests'], 1)
        self.assertEqual(system['failed_requests'], 1)
        self.assertEqual(system['achieved_total_throughput_rps'], 2)
        self.assertEqual(system['within_window_completion_throughput_rps'], 1)
        self.assertEqual(system['completed_during_drain'], 1)
        self.assertEqual(system['completion_throughput_including_drain_time_rps'], 1)


class ProtocolIntegrationTests(unittest.TestCase):
    def execute(self, cfg, sample_overrides=None, registry_override=None):
        with tempfile.TemporaryDirectory() as directory:
            pool, output = Path(directory) / 'pool.jsonl', Path(directory) / 'result.json'
            pool.write_text(json.dumps({'pool_id': 'text:test', 'task_type': 'text',
                                       'sample': {'id': 'test', **(sample_overrides or {})}}) + '\n')
            command = {'python': sys.executable, 'script': Path(__file__).with_name('protocol_worker.py')}
            registry = {'qwen': command, 'distilbert': command} if registry_override is None else registry_override
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_measurement(cfg, output, pool, DEFAULT_PROFILE, registry)
            self.assertTrue(output.exists())
            self.assertEqual(json.loads(output.read_text())['status'], result['status'])
            return result

    def test_two_workers_open_loop_warmup_exclusion_routing_and_shutdown(self):
        cfg = config(workload_rps={'text': 20}, routing_rps={'text': {'qwen': 5, 'distilbert': 15}},
                     warmup_seconds=0.2, measurement_seconds=0.4)
        result = self.execute(cfg)
        self.assertEqual(result['status'], 'completed', result['errors'])
        self.assertEqual(result['system_metrics']['offered_requests'], 8)
        self.assertEqual(result['system_metrics']['completed_requests'], 8)
        self.assertEqual(len(result['priming_request_records']), 2)
        self.assertEqual(len(result['warmup_request_records']), 4)
        self.assertEqual(result['per_task_metrics']['text']['actual_routed_request_count'], {'distilbert': 6, 'qwen': 2})
        for record in result['request_records']:
            self.assertGreaterEqual(record['actual_submit_time_ns'], record['scheduled_time_ns'])
            self.assertGreaterEqual(record['completion_time_ns'], record['actual_submit_time_ns'])
        self.assertTrue(all(r['bye_received'] and r['exit_code'] == 0 and not r['forced']
                            for r in result['shutdown'].values()))

    def test_overload_preserves_arrivals_and_reports_timeouts(self):
        cfg = config(deployment={'qwen': 1}, workload_rps={'text': 20}, routing_rps={'text': {'qwen': 20}},
                     measurement_seconds=0.2, drain_timeout_seconds=0.01, shutdown_timeout_seconds=0.02)
        started = time.monotonic()
        result = self.execute(cfg, {'test_delay_seconds': 0.3})
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(result['system_metrics']['offered_requests'], 4)
        self.assertGreater(result['system_metrics']['incomplete_requests'], 0)
        self.assertNotEqual(result['status'], 'completed')
        self.assertTrue(result['shutdown']['qwen']['forced'])

    def test_worker_errors_are_counted_and_not_successful_latency(self):
        result = self.execute(config(), {'test_error': True})
        self.assertEqual(result['system_metrics']['offered_requests'], 4)
        self.assertEqual(result['system_metrics']['completed_requests'], 0)
        self.assertEqual(result['system_metrics']['failed_requests'], 4)
        self.assertIsNone(result['per_task_metrics']['text']['mean_e2e_latency_ms'])
        self.assertEqual(result['status'], 'completed_with_failures')

    def test_failed_startup_saves_artifact_and_cleans_up(self):
        command = {'python': '/does/not/exist/python', 'script': Path(__file__).with_name('protocol_worker.py')}
        result = self.execute(config(deployment={'qwen': 1}, routing_rps={'text': {'qwen': 4}}),
                              registry_override={'qwen': command})
        self.assertEqual(result['status'], 'failed')
        self.assertIsNone(result['system_metrics'])


if __name__ == '__main__':
    unittest.main()
