"""CPU lifecycle regressions: final live-worker sample precedes worker close."""
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

from tools.predictor_validation_20261027 import profile_owned_collect_v3 as c


class MonitoringScopeTests(unittest.TestCase):
    def setup_runtime_monitor(self, *, telemetry_errors=None, close_error=False, foreign=False):
        order=[]
        runtime=SimpleNamespace(closed=False,errors=[],shutdown_report={})
        def worker_close():
            order.append('worker_close');runtime.closed=True
            if close_error:raise RuntimeError('worker close failed')
            runtime.shutdown_report={'text#0':dict(exit_code=0,forced=False,bye_received=True)}
        runtime.close=worker_close
        monitor=object.__new__(c.StartupBoardMonitor)
        monitor.runtime=runtime;monitor.interval=.001;monitor.stop=threading.Event()
        monitor._sample_lock=threading.RLock();monitor._close_lock=threading.Lock();monitor._closed=False
        monitor.samples=[];monitor.errors=list(telemetry_errors or []);monitor.phase='measurement'
        def query():
            self.assertFalse(runtime.closed,'sampled after worker close')
            row=dict(time_ns=time.perf_counter_ns(),phase=monitor.phase,
                owned_used_mib=10.,owned_aliases=['text#0'],foreign_pids=[123] if foreign else [])
            monitor.samples.append(row);order.append('sample:'+monitor.phase)
            return row
        monitor._sample=query
        def nvml_shutdown():order.append('nvml_shutdown');return 0
        monitor.lib=SimpleNamespace(nvmlShutdown=nvml_shutdown)
        return runtime,monitor,order

    def raw(self):
        return dict(status='completed',errors=[],monitoring_scope=c.MONITORING_SCOPE,
            shutdown_memory_sampled=False,lifecycle_boundaries=[
                dict(boundary='measurement_and_drain_end',time_ns=time.perf_counter_ns())])

    def test_actual_sampler_join_and_final_sample_before_workers_close(self):
        runtime,monitor,order=self.setup_runtime_monitor()
        monitor.start()
        with tempfile.TemporaryDirectory() as d:
            raw=self.raw();c.finish_collection(runtime,monitor,raw,Path(d))
            persisted=json.loads((Path(d)/'raw.json').read_text())
        boundary={r['boundary']:r['time_ns'] for r in raw['lifecycle_boundaries']}
        final=raw['board_samples'][-1]
        self.assertEqual(final['phase'],'final_drain')
        self.assertLessEqual(boundary['measurement_and_drain_end'],final['time_ns'])
        self.assertLessEqual(final['time_ns'],boundary['monitoring_end'])
        self.assertLess(boundary['monitoring_end'],boundary['shutdown_begin'])
        self.assertLess(order.index('sample:final_drain'),order.index('nvml_shutdown'))
        self.assertLess(order.index('nvml_shutdown'),order.index('worker_close'))
        self.assertFalse(monitor.thread.is_alive());self.assertTrue(monitor.stop.is_set())
        self.assertEqual(raw['telemetry_errors'],[]);self.assertEqual(persisted,raw)
        before=len(monitor.samples);time.sleep(.005);self.assertEqual(len(monitor.samples),before)

    def test_join_does_not_hold_sample_lock_and_waits_for_inflight_query(self):
        runtime,monitor,order=self.setup_runtime_monitor()
        query=monitor._sample;entered=threading.Event();release=threading.Event();finished=threading.Event()
        def blocking():
            if monitor.phase=='measurement':entered.set();release.wait(2)
            return query()
        monitor._sample=blocking;monitor.start();self.assertTrue(entered.wait(2))
        closer=threading.Thread(target=lambda:(monitor.close(),finished.set()))
        closer.start();self.assertFalse(finished.wait(.02));release.set();closer.join(2)
        self.assertTrue(finished.is_set());self.assertFalse(monitor.thread.is_alive())
        self.assertEqual(monitor.samples[-1]['phase'],'final_drain')

    def test_close_is_idempotent_and_never_adds_shutdown_sample(self):
        runtime,monitor,order=self.setup_runtime_monitor()
        monitor.close();runtime.close();monitor.close()
        self.assertEqual(order,['sample:final_drain','nvml_shutdown','worker_close'])

    def test_all_old_telemetry_errors_survive_and_failure_still_closes_workers(self):
        errors=['UNKNOWN_NVML_PID_OWNERSHIP:[1]','original NVML failure']
        runtime,monitor,order=self.setup_runtime_monitor(telemetry_errors=errors)
        with tempfile.TemporaryDirectory() as d:
            raw=self.raw();c.finish_collection(runtime,monitor,raw,Path(d))
        self.assertEqual(raw['status'],'failed');self.assertEqual(raw['telemetry_errors'],errors)
        self.assertTrue(runtime.closed);self.assertIn('MonitoringCloseError',raw['errors'][0])

    def test_final_foreign_pid_is_failed_without_filtering_its_sample(self):
        runtime,monitor,order=self.setup_runtime_monitor(foreign=True)
        with tempfile.TemporaryDirectory() as d:
            raw=self.raw();c.finish_collection(runtime,monitor,raw,Path(d))
        self.assertEqual(raw['status'],'failed');self.assertEqual(raw['board_samples'][-1]['foreign_pids'],[123])
        self.assertTrue(runtime.closed)

    def test_monitor_close_failure_still_persists_and_closes_workers(self):
        runtime,monitor,order=self.setup_runtime_monitor()
        def broken_close():raise RuntimeError('close failed')
        monitor.close=broken_close
        with tempfile.TemporaryDirectory() as d:
            raw=self.raw();c.finish_collection(runtime,monitor,raw,Path(d))
            self.assertTrue((Path(d)/'raw.json').exists())
        self.assertEqual(raw['status'],'failed');self.assertTrue(runtime.closed)
        self.assertIn('close failed',raw['errors'][0])

    def test_worker_close_failure_retains_monitor_samples_and_raw(self):
        runtime,monitor,order=self.setup_runtime_monitor(close_error=True)
        with tempfile.TemporaryDirectory() as d:
            raw=self.raw();c.finish_collection(runtime,monitor,raw,Path(d))
            persisted=json.loads((Path(d)/'raw.json').read_text())
        self.assertEqual(raw['status'],'failed');self.assertEqual(persisted,raw)
        self.assertEqual(raw['board_samples'][-1]['phase'],'final_drain')
        self.assertIn('WorkerShutdownError',raw['errors'][0])


if __name__=='__main__':unittest.main()
