"""Independent CPU checks for final-drain telemetry and worker cleanup order."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import profile_owned_collect_v3 as v


class IndependentMonitoringBoundaryTests(unittest.TestCase):
    def monitor(self, log, *, shutdown_code=0):
        class Library:
            def nvmlShutdown(self):
                log.append('nvml_shutdown')
                return shutdown_code
        def initialize(monitor, runtime, interval):
            monitor.lib = Library()
            monitor.runtime = runtime
            monitor.interval = interval
            monitor.stop = threading.Event()
            monitor.thread = None
            monitor.samples = []
            monitor.errors = []
        with patch.object(v.BoardMonitor, '__init__', initialize):
            return v.StartupBoardMonitor(SimpleNamespace(workers={}))

    def raw(self):
        return dict(status='completed', errors=[], monitoring_scope=v.MONITORING_SCOPE,
            lifecycle_boundaries=[dict(boundary='measurement_and_drain_end', time_ns=time.perf_counter_ns())])

    def runtime(self, log, *, error=False):
        class Runtime:
            shutdown_report = {'one': dict(exit_code=0, forced=False, bye_received=True)}
            errors = ['CPU runtime diagnostic']
            def close(self):
                log.append('worker_close')
                if error:
                    raise RuntimeError('CPU worker close failed')
        return Runtime()

    def test_close_joins_inflight_query_without_holding_sample_lock(self):
        log = []
        monitor = self.monitor(log)
        entered, release = threading.Event(), threading.Event()
        count = 0
        def query():
            nonlocal count
            count += 1
            if count == 1:
                log.append('query_begin')
                entered.set()
                if not release.wait(2):
                    raise AssertionError('Independent fixture release never arrived')
                log.append('query_end')
            else:
                log.append('final_query')
        monitor._sample = query
        monitor.thread = threading.Thread(target=monitor.sample)
        monitor.thread.start()
        self.assertTrue(entered.wait(1))
        closer = threading.Thread(target=monitor.close)
        closer.start()
        try:
            self.assertTrue(monitor.stop.wait(.01))
            self.assertTrue(closer.is_alive())
            self.assertNotIn('nvml_shutdown', log)
        finally:
            release.set()
            closer.join(3)
            monitor.thread.join(3)
        self.assertFalse(closer.is_alive(), 'close deadlocked with a completed in-flight sample')
        self.assertFalse(monitor.thread.is_alive())
        self.assertEqual(log, ['query_begin', 'query_end', 'final_query', 'nvml_shutdown'])

    def test_concurrent_close_is_idempotent_and_unstarted_monitor_is_supported(self):
        log = []
        monitor = self.monitor(log)
        monitor._sample = lambda: log.append('final_query')
        closers = [threading.Thread(target=monitor.close) for _ in range(2)]
        for closer in closers:
            closer.start()
        for closer in closers:
            closer.join(2)
            self.assertFalse(closer.is_alive())
        monitor.close()
        self.assertEqual(log, ['final_query', 'nvml_shutdown'])
        self.assertTrue(monitor._closed)

    def test_final_query_finishes_before_monitoring_end_and_worker_close(self):
        log = []
        monitor = self.monitor(log)
        def query():
            self.assertEqual(monitor.phase, 'final_drain')
            self.assertNotIn('worker_close', log)
            stamp = time.perf_counter_ns()
            monitor.samples.append(dict(time_ns=stamp, phase=monitor.phase,
                foreign_pids=[], unresolved_nvml_pids=[], owned_used_mib=10.))
            log.append('final_query')
        monitor._sample = query
        raw = self.raw()
        with tempfile.TemporaryDirectory() as directory:
            v.finish_collection(self.runtime(log), monitor, raw, Path(directory))
            saved = json.loads((Path(directory)/'raw.json').read_text())
        self.assertEqual(log, ['final_query', 'nvml_shutdown', 'worker_close'])
        times = {r['boundary']: r['time_ns'] for r in saved['lifecycle_boundaries']}
        self.assertLessEqual(times['measurement_and_drain_end'], saved['board_samples'][-1]['time_ns'])
        self.assertLess(saved['board_samples'][-1]['time_ns'], times['monitoring_end'])
        self.assertLess(times['monitoring_end'], times['shutdown_begin'])
        self.assertLess(times['shutdown_begin'], times['shutdown_end'])
        self.assertEqual(saved['status'], 'completed')

    def test_earlier_unknown_error_and_all_samples_survive_successful_final_query(self):
        log = []
        monitor = self.monitor(log)
        monitor.samples = [dict(time_ns=1, phase='startup'), dict(time_ns=2, phase='measurement')]
        monitor.errors = ['UNKNOWN_NVML_PID_OWNERSHIP:[17]']
        original = copy.deepcopy(monitor.samples)
        def query():
            monitor.samples.append(dict(time_ns=time.perf_counter_ns(), phase='final_drain'))
            log.append('final_query')
        monitor._sample = query
        raw = self.raw()
        with tempfile.TemporaryDirectory() as directory:
            v.finish_collection(self.runtime(log), monitor, raw, Path(directory))
            saved = json.loads((Path(directory)/'raw.json').read_text())
        self.assertEqual(saved['board_samples'][:2], original)
        self.assertEqual(len(saved['board_samples']), 3)
        self.assertEqual(saved['telemetry_errors'], ['UNKNOWN_NVML_PID_OWNERSHIP:[17]'])
        self.assertEqual(saved['status'], 'failed')
        self.assertIn('worker_close', log)

    def test_final_sample_exception_still_shuts_NVML_closes_workers_and_saves_failure(self):
        log = []
        monitor = self.monitor(log)
        def broken():
            log.append('final_query_failed')
            raise RuntimeError('CPU final sample failed')
        monitor._sample = broken
        raw = self.raw()
        with tempfile.TemporaryDirectory() as directory:
            v.finish_collection(self.runtime(log), monitor, raw, Path(directory))
            saved = json.loads((Path(directory)/'raw.json').read_text())
        self.assertTrue(monitor._closed)
        self.assertEqual(saved['status'], 'failed')
        self.assertEqual(log, ['final_query_failed', 'nvml_shutdown', 'worker_close'])
        self.assertTrue(any('CPU final sample failed' in error for error in saved['errors']))

    def test_NVML_shutdown_failure_still_closes_workers_and_cannot_be_completed(self):
        log = []
        monitor = self.monitor(log, shutdown_code=1)
        monitor._sample = lambda: log.append('final_query')
        raw = self.raw()
        with tempfile.TemporaryDirectory() as directory:
            v.finish_collection(self.runtime(log), monitor, raw, Path(directory))
            saved = json.loads((Path(directory)/'raw.json').read_text())
        self.assertEqual(saved['status'], 'failed')
        self.assertTrue(monitor._closed)
        self.assertEqual(log, ['final_query', 'nvml_shutdown', 'worker_close'])

    def test_worker_close_failure_is_persisted_with_complete_telemetry(self):
        log = []
        monitor = self.monitor(log)
        monitor._sample = lambda: log.append('final_query')
        raw = self.raw()
        with tempfile.TemporaryDirectory() as directory:
            v.finish_collection(self.runtime(log, error=True), monitor, raw, Path(directory))
            saved = json.loads((Path(directory)/'raw.json').read_text())
        self.assertEqual(saved['status'], 'failed')
        self.assertEqual(saved['runtime_errors'], ['CPU runtime diagnostic'])
        self.assertIn('board_samples', saved)
        self.assertTrue(any('WorkerShutdownError' in error for error in saved['errors']))
        self.assertEqual(log, ['final_query', 'nvml_shutdown', 'worker_close'])

    def test_no_monitor_after_initialization_failure_still_closes_workers(self):
        log = []
        raw = self.raw()
        raw.update(status='failed', errors=['CPU monitor initialization failed'])
        with tempfile.TemporaryDirectory() as directory:
            v.finish_collection(self.runtime(log), None, raw, Path(directory))
            saved = json.loads((Path(directory)/'raw.json').read_text())
        self.assertEqual(saved['status'], 'failed')
        self.assertEqual(log, ['worker_close'])
        self.assertEqual(saved['errors'], ['CPU monitor initialization failed'])

    def test_actual_collect_calls_all_declared_drains_before_monitor_then_worker_close(self):
        from tools.predictor_validation_20261027 import paired_profile_campaign_v6 as campaign
        log = []
        class Runtime:
            def __init__(self, config, registry):
                self.workers = {alias: SimpleNamespace(process=SimpleNamespace(pid=i + 100))
                    for i, alias in enumerate(registry)}
                self.ready = {alias: dict(pid=worker.process.pid) for alias, worker in self.workers.items()}
                self.errors = []
                self.shutdown_report = {}
            def start(self):
                log.append('runtime_start')
            def enqueue(self, row, sample):
                row['status'] = 'completed'
            def drain(self, rows, timeout):
                log.append(('drain', tuple(row['request_id'] for row in rows)))
            def close(self):
                log.append('worker_close')
                self.shutdown_report = {alias: dict(exit_code=0, forced=False, bye_received=True)
                    for alias in self.workers}
        class Monitor:
            def __init__(self, runtime):
                self.samples, self.errors = [], []
                self.phase = 'startup'
                self.collector_pid, self.process_group, self.ownership = os.getpid(), os.getpgrp(), object()
            def sample(self):
                row = dict(time_ns=time.perf_counter_ns(), phase=self.phase,
                    foreign_pids=[], owned_aliases=list(campaign.RESIDENTS))
                self.samples.append(row)
                log.append('sample:' + self.phase)
            def check(self):
                pass
            def start(self):
                log.append('monitor_start')
            def set_phase(self, phase):
                self.phase = phase
            def close(self):
                log.append('monitor_close')
                self.phase = 'final_drain'
                self.sample()
        def ownership(values, runtime, *args, **kwargs):
            alias = next(name for name, worker in runtime.workers.items() if worker.process.pid in values)
            return dict(owned_aliases=[alias], unresolved_nvml_pids=[], foreign_pids=[], ownership_evidence=[])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'packet'
            campaign.prepare(root, progress_observer=True)
            run = root/'smoke/natural'
            plan = campaign.audit(run)
            with patch.object(v, 'validate_lease'), patch.object(v, 'DeploymentRuntime', Runtime), \
                 patch.object(v, 'StartupBoardMonitor', Monitor), patch.object(v, 'memory_ownership', ownership), \
                 patch.object(v.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
                raw = v.collect(run, trace=False, lease_fd=42)
        drains = [entry for entry in log if isinstance(entry, tuple) and entry[0] == 'drain']
        self.assertEqual(len(drains), len(raw['priming_request_records']) + len(plan['jobs']))
        self.assertEqual(len(raw['request_records']), sum(len(job['inputs']) for job in plan['jobs']))
        self.assertEqual(raw['status'], 'completed')
        self.assertLess(log.index(drains[-1]), log.index('monitor_close'))
        self.assertLess(log.index('monitor_close'), log.index('sample:final_drain'))
        self.assertLess(log.index('sample:final_drain'), log.index('worker_close'))
        boundaries = {row['boundary']: row['time_ns'] for row in raw['lifecycle_boundaries']}
        self.assertLessEqual(boundaries['measurement_and_drain_end'], raw['board_samples'][-1]['time_ns'])
        self.assertLess(boundaries['monitoring_end'], boundaries['shutdown_begin'])
        self.assertFalse(raw['shutdown_memory_sampled'])

    def test_stuck_NVML_join_remains_under_outer_supervisor_timeout(self):
        from tools.predictor_validation_20261027 import profile_training_supervisor_v3 as supervisor
        code = '''import threading
from types import SimpleNamespace
from tools.predictor_validation_20261027.profile_owned_collect_v3 import StartupBoardMonitor
m=object.__new__(StartupBoardMonitor)
m._sample_lock=threading.RLock()
m._close_lock=threading.Lock()
m._closed=False
m.stop=threading.Event()
m.lib=SimpleNamespace(nvmlShutdown=lambda:0)
m.phase='measurement'
entered=threading.Event()
def blocked():
    entered.set()
    threading.Event().wait()
m._sample=blocked
m.thread=threading.Thread(target=m.sample,daemon=True)
m.thread.start()
assert entered.wait(2)
m.close()
raise AssertionError('Blocked sampler detached without termination')
'''
        budget = dict(supervisor.BUDGET, total_timeout_seconds=.4,
            watchdog_interval_seconds=.01, cleanup_grace_seconds=.01,
            free_disk_reserve_bytes=0, disk_abort_margin_bytes=0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = supervisor.supervise([sys.executable, '-c', code], root/'run',
                environment={}, lock_path=root/'lock', preflight=0, budget=budget)
        self.assertEqual(result['status'], 'FAILED_NO_RETRY')
        self.assertEqual(result['watchdog']['status'], 'TIMEOUT')
        self.assertTrue(result['child_reaped'])
        self.assertTrue(result['all_owned_descendants_exited_and_reaped'])
        self.assertTrue(result['experiment_lock_held_through_group_exit'])
        self.assertEqual(result['live_group_pids'], [])


if __name__ == '__main__':
    unittest.main()
