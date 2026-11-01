"""CPU-only independent evidence, including first-version failure boundaries.

The author is fixing v3 concurrently. Counterexamples load the byte-exact frozen
review snapshot, never mutate production sources, and release all test threads.
"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import unittest

from tests.test_shared_text_batch_v2 import backend, sample
from gsdeploy.runtime.batching import BatchConfiguration
from gsdeploy.runtime.deployment import DeploymentRuntime

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT/'artifacts/batch-active-background-independent-review-v3-20261028-cpu01'
spec = importlib.util.spec_from_file_location(
    'tools.predictor_validation_20261027._independent_frozen_batch_v3',
    REVIEW/'collector_v3_reviewed_source.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
fixture_spec = importlib.util.spec_from_file_location('_independent_batch_frozen_author_tests',
    REVIEW/'author_tests_reviewed_source.py')
fixtures = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixtures)
fixtures.c = c
METRICS = {}


class IndependentBatchReview(unittest.TestCase):
    def prepare(self, root, *, unsuccessful_prime=False):
        args = fixtures.BatchProfileCollectV3Tests().fixture(root)
        if unsuccessful_prime:
            priming = c.read(args[4])
            # The real CPU backend reaches its token limit without EOS for
            # this input; no result or success field is fabricated.
            priming['training']['samples'].append(sample('prime-length', [1,4]))
            priming['training']['arrivals'][0]['sample_id'] = 'prime-length'
            args[4].write_text(json.dumps(priming))
        c.prepare(root/'plan', *args, BatchConfiguration(2,100,1_000_000),
                  measurement_ns=20_000_000, drain_timeout_seconds=.3)
        return root/'plan/plan.json'

    def test_counterexample_unsuccessful_text_priming_does_not_stop_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.prepare(Path(directory), unsuccessful_prime=True)
            runtime = fixtures.BackgroundRuntime()
            report = c.collect(path, 'training', backend_factory=backend,
                               background_runtime_factory=lambda:runtime)
            prime = c.read(path.parent/'training/collection/text-priming.json')
            self.assertFalse(prime['results'][0]['result']['success'])
            self.assertEqual(prime['results'][0]['result']['finish_reason'], 'length')
            self.assertIsNotNone(report['epoch_ns'])
            self.assertTrue(runtime.measurement_received.is_set())
            self.assertEqual(report['status'], 'CPU_TEST_ONLY_COMPLETE')
            METRICS['unsuccessful_priming_counterexample'] = dict(
                reproduced=True, prime_finish_reason='length', prime_success=False,
                measurement_started=True, final_status=report['status'],
                offered=report['offered'], all_offered_indexed=True)

    def test_counterexample_sync_text_generate_outlives_declared_drain(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.prepare(Path(directory)); runtime = fixtures.BackgroundRuntime()
            entered, release = threading.Event(), threading.Event()
            closed, holder = threading.Event(), {}
            original_close = runtime.close
            def close():
                closed.set(); return original_close()
            runtime.close = close
            def factory():
                owner = backend(); execute = owner.execute_prepared; calls = [0]
                def blocked(rows, **kw):
                    calls[0] += 1
                    if calls[0] == 2:
                        entered.set()
                        if not release.wait(3):
                            raise RuntimeError('Independent test release missing')
                    return execute(rows, **kw)
                owner.execute_prepared = blocked
                return owner
            def collect():
                try:
                    holder['report'] = c.collect(path, 'training', backend_factory=factory,
                                                background_runtime_factory=lambda:runtime)
                except BaseException as exc:
                    holder['error'] = repr(exc)
            thread = threading.Thread(target=collect, name='independent-blocked-text-test')
            thread.start()
            try:
                self.assertTrue(entered.wait(1))
                # Measurement is 20ms plus 300ms drain. A blocked synchronous
                # generate still holds the collector after that boundary.
                self.assertFalse(closed.wait(.4))
                self.assertTrue(thread.is_alive())
                active_after_boundary = thread.is_alive()
            finally:
                release.set(); thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertNotIn('error', holder)
            report = holder['report']
            self.assertFalse(report['within_declared_drain'])
            self.assertEqual(report['status'], 'INCOMPLETE_OR_FAILED_NOT_QUALIFIED')
            self.assertEqual(report['offered'], 4)
            self.assertEqual(len(c.read(c.checked(report['profile_index']))['requests']), 4)
            METRICS['synchronous_text_timeout_counterexample'] = dict(
                reproduced=True, collector_active_after_declared_boundary=active_after_boundary,
                cleanup_started_before_release=False, eventually_failed_closed=True,
                offered=report['offered'], offered_indexed=4, test_thread_joined=True)

    def test_counterexample_dispatcher_join_timeout_does_not_prove_exit(self):
        entered, release = threading.Event(), threading.Event()
        def enqueue(row, value):
            entered.set(); release.wait(3)
        runtime = SimpleNamespace(enqueue=enqueue, drain=lambda rows, timeout:None)
        arrivals = [dict(request_id='r',sample_id='s',pool_id='p',alias='generalist_2#0',
                         task='text',arrival_index=0,arrival_ns=0)]
        dispatcher = c.BackgroundDispatcher(runtime, [sample('s',[1,2])], arrivals, time.perf_counter_ns())
        dispatcher.start()
        try:
            self.assertTrue(entered.wait(.2))
            with self.assertRaises(TimeoutError):
                dispatcher.finish(time.perf_counter_ns()+10_000_000)
            self.assertTrue(dispatcher.thread.is_alive())
            alive_at_finish = dispatcher.thread.is_alive()
        finally:
            release.set(); dispatcher.thread.join(1)
        self.assertFalse(dispatcher.thread.is_alive())
        METRICS['dispatcher_exit_counterexample'] = dict(
            reproduced=True, thread_alive_after_finish_raised=alive_at_finish,
            independent_thread_eventually_joined=True,
            limitation='Injected enqueue blocks; this demonstrates absent exit proof, not an observed GPU or real DeploymentRuntime deadlock.')

    def test_real_runtime_preserves_reader_receipt_when_collector_is_delayed(self):
        runtime = DeploymentRuntime(SimpleNamespace(), {})
        runtime.ready['generalist_2#0'] = dict(pid=123)
        record = dict(request_id='r', model='generalist_2#0', task='text', status='submitted')
        runtime.pending['r'] = record
        collector = threading.Thread(target=runtime._collect, name='independent-reader-receipt-test')
        end = time.perf_counter_ns()
        try:
            with runtime.lock:
                collector.start()
                runtime.events.put(('generalist_2#0','RESULT',dict(
                    request_id='r', task_type='text', success=True, end_perf_ns=end)))
                runtime.events.put(('generalist_2#0','WORKER_READY',dict(
                    request_id='r',pid=123,clock='host_perf_counter_ns',
                    boundary='previous_handler_next_request_ready',perf_ns=end)))
                time.sleep(.02)
            c._wait_ready(runtime,[record],time.perf_counter_ns()+500_000_000,time.perf_counter_ns)
            self.assertEqual(record['completion_time_ns'], record['response_reader_event_enqueue_time_ns'])
            delay = record['response_collector_processing_time_ns']-record['completion_time_ns']
            self.assertGreater(delay, 15_000_000)
            self.assertFalse(runtime.errors)
        finally:
            runtime.collect_stop.set(); collector.join(1)
        self.assertFalse(collector.is_alive())
        METRICS['real_runtime_receipt'] = dict(reader_enqueue_is_original_completion=True,
            collector_processing_delay_ns=delay, next_request_ready_preserved=True,
            actual_DeploymentRuntime_consumer_used=True, collector_joined=True)

    def test_independent_resource_window_with_large_drain_power_and_memory(self):
        samples = [dict(time_ns=t,watts=p,owned_used_mib=m) for t,p,m in
                   [(0,10.,3),(10,20.,8),(20,40.,9),(30,1000.,600),(35,1000.,900)]]
        resources = c.resource_summary(samples,startup_ns=0,epoch_ns=10,measurement_end_ns=20,
                                       drain_end_ns=30,physical_identity='three-owner')
        self.assertAlmostEqual(resources['startup_energy_joules'], 1.5e-7)
        self.assertAlmostEqual(resources['measurement_energy_joules'], 3e-7)
        self.assertEqual(resources['owned_sampled_peak_startup_through_drain_mib'],600)
        METRICS['resource_windows'] = dict(startup_energy_joules=1.5e-7,
            measurement_energy_joules=3e-7, owned_peak_mib=600,
            high_drain_power_excluded_from_measurement=True, after_drain_memory_excluded=True,
            synthetic_CPU_samples_not_NVML_evidence=True)

    def test_startup_and_text_generation_failures_preserve_full_denominator(self):
        cases = []
        for phase in ('startup','measurement_generate'):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                path = self.prepare(Path(directory)); runtime = fixtures.BackgroundRuntime()
                if phase == 'startup':
                    def fail_start():
                        raise RuntimeError('Independent CPU startup failure')
                    runtime.start = fail_start
                def factory():
                    owner = backend(); original = owner.execute_prepared; calls = [0]
                    def execute(rows, **kw):
                        calls[0] += 1
                        if calls[0] == 2:
                            raise RuntimeError('Independent CPU measurement failure')
                        return original(rows, **kw)
                    owner.execute_prepared = execute
                    return owner
                report = c.collect(path,'training',backend_factory=factory,
                                   background_runtime_factory=lambda:runtime)
                index = c.read(c.checked(report['profile_index']))
                self.assertEqual(report['offered'],4)
                self.assertEqual(len(report['offered_cohort']),4)
                self.assertEqual(len(index['requests']),4)
                self.assertEqual(set(report['unresolved_request_ids']),{'t1','t2','t3','t4'})
                self.assertEqual(report['status'],'INCOMPLETE_OR_FAILED_NOT_QUALIFIED')
                cases.append(dict(phase=phase,offered=4,indexed=4,unresolved=4))
        METRICS['failure_denominator'] = dict(cases=cases, no_success_only_shrink=True)


if __name__ == '__main__':
    unittest.main()
