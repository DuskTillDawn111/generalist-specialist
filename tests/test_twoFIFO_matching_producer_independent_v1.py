"""Independent CPU trust-boundary checks; no supervisor/native child is run."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools.predictor_validation_20261027 import twoFIFO_matching_execute_v1 as m
from tools.predictor_validation_20261027 import twoFIFO_matching_review_v1 as review
from tools.predictor_validation_20261027 import twoFIFO_matching_intake_v1 as intake
from tools.predictor_validation_20261027 import twoFIFO_matching_macros_v1 as macros
from tests.test_twoFIFO_owned_collect_v2 import Clock, FakeRuntime, FakeMonitor
from tests.test_twoFIFO_matching_producer_v1 import fixture, completed_doc


ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT/'artifacts/predictor-twoFIFO-matching-owned-producer-v1-20261029-cpu02-prepared/manifest.json'


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class IndependentMatchingProducerTests(unittest.TestCase):
    def execution_fixture(self, root):
        plan = dict(execution_id='a'*32, attempt_id='independent-attempt')
        pp = root/'attempt'/'plan.json'
        write(pp, plan)
        item = dict(plan=m.receipt(pp), execution_id=plan['execution_id'], attempt_id=plan['attempt_id'],
                    collection=str(pp.parent/'collection'), claim=str(pp.parent/'matching-attempt-claim.json'))
        manifest = dict(attempts=[item], sources=[])
        mp = root/'manifest.json'
        write(mp, manifest)
        return plan, pp, item, manifest, mp

    def test_execute_rejects_transient_generated_plan_change_and_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, pp, item, manifest, mp = self.execution_fixture(Path(directory))
            def supervised(*args, **kwargs):
                kwargs['before_launch']()
                original = pp.read_bytes()
                pp.write_text('{"unexpected_runtime_plan": true}')
                pp.write_bytes(original)
                kwargs['after_run']()
                return dict(status='STUB_NO_PROCESS_LAUNCHED')
            with patch.object(m, 'audit', return_value=manifest), \
                 patch.object(m, 'audit_plan', return_value=plan), \
                 patch.object(m.supervisor, 'public_environment_gate'), \
                 patch.object(m.supervisor, 'supervise', side_effect=supervised), \
                 self.assertRaisesRegex(ValueError, '[Ss]ource|changed'):
                m.execute(pp, item['collection'], manifest_path=mp)

    def test_durable_claim_precedes_supervisor_and_prevents_second_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, pp, item, manifest, mp = self.execution_fixture(Path(directory))
            def supervised(*args, **kwargs):
                self.assertTrue(Path(item['claim']).is_file())
                self.assertEqual(kwargs['metadata']['claim'], m.receipt(item['claim']))
                raise RuntimeError('CPU stub prelaunch failure')
            with patch.object(m, 'audit', return_value=manifest), \
                 patch.object(m, 'audit_plan', return_value=plan), \
                 patch.object(m.supervisor, 'public_environment_gate'), \
                 patch.object(m.supervisor, 'supervise', side_effect=supervised) as supervisor:
                with self.assertRaisesRegex(RuntimeError, 'prelaunch'):
                    m.execute(pp, item['collection'], manifest_path=mp)
                claim = Path(item['claim']).read_bytes()
                with self.assertRaisesRegex(ValueError, 'claimed'):
                    m.execute(pp, item['collection'], manifest_path=mp)
                self.assertEqual(Path(item['claim']).read_bytes(), claim)
                self.assertEqual(supervisor.call_count, 1)

    def test_watch_precedes_first_full_audit_and_rejects_restore_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, pp, item, manifest, mp = self.execution_fixture(Path(directory))
            def first_audit(*args):
                original = pp.read_bytes()
                pp.write_text('{"transient_during_audit": true}')
                pp.write_bytes(original)
                return manifest
            with patch.object(m, 'audit', side_effect=first_audit), \
                 patch.object(m, 'audit_plan', return_value=plan), \
                 patch.object(m.supervisor, 'public_environment_gate'), \
                 patch.object(m.supervisor, 'supervise') as supervisor, \
                 self.assertRaisesRegex(ValueError, '[Ss]ource|changed'):
                m.execute(pp, item['collection'], manifest_path=mp)
            supervisor.assert_not_called()
            self.assertFalse(Path(item['claim']).exists())

    def test_future_plan_is_watched_even_while_another_attempt_is_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, pp, item, manifest, mp = self.execution_fixture(root)
            future = root/'future-attempt'/'plan.json'
            write(future, dict(execution_id='b'*32, attempt_id='future-attempt'))
            manifest['attempts'].append(dict(plan=m.receipt(future), execution_id='b'*32,
                attempt_id='future-attempt', collection=str(future.parent/'collection'),
                claim=str(future.parent/'claim.json')))
            write(mp, manifest)
            def supervised(*args, **kwargs):
                kwargs['before_launch']()
                original = future.read_bytes()
                future.write_text('{}')
                future.write_bytes(original)
                kwargs['after_run']()
            with patch.object(m, 'audit', return_value=manifest), \
                 patch.object(m, 'audit_plan', return_value=plan), \
                 patch.object(m.supervisor, 'public_environment_gate'), \
                 patch.object(m.supervisor, 'supervise', side_effect=supervised), \
                 self.assertRaisesRegex(ValueError, '[Ss]ource|changed'):
                m.execute(pp, item['collection'], manifest_path=mp)

    def test_invalid_lease_rejects_before_plan_or_runtime(self):
        factory = Mock()
        with patch.object(m.supervisor, 'validate_lease', side_effect=ValueError('invalid lease')), \
             patch.object(m, 'audit_plan') as audit, self.assertRaisesRegex(ValueError, 'lease'):
            m.collect('absent', 'absent', lease_fd=-1, runtime_factory=factory)
        audit.assert_not_called()
        factory.assert_not_called()

    def collect_failure(self, failure):
        manifest = json.loads(PREPARED.read_text())
        item = manifest['attempts'][0]
        plan = m.checked(item['plan'])
        clock = Clock()
        runtime = FakeRuntime(clock)
        original_drain = runtime.drain
        if failure == 'priming':
            def drain(rows, timeout):
                original_drain(rows, timeout)
                if len(runtime.enqueued) == 7:
                    rows[0]['status'] = 'incomplete'
            runtime.drain = drain
        monitor = FakeMonitor(runtime)
        if failure == 'monitor':
            def close():
                monitor.sample()
                raise RuntimeError('CPU injected monitor close failure')
            monitor.close = close
        ownership = lambda values, *a, **k: dict(owned_aliases=[m.OWNERS[next(iter(values))-1]],
            unresolved_nvml_pids=[], foreign_pids=[], ownership_evidence=[{'live': True}])
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(m.supervisor, 'validate_lease'), patch.object(m, 'audit_plan', return_value=plan):
            raw = m.collect(item['plan']['path'], directory, lease_fd=-1,
                runtime_factory=lambda *args: runtime, monitor_factory=lambda *args: monitor,
                clock_ns=clock.clock, wait_until_ns=clock.wait, ownership_check=ownership)
            self.assertEqual(raw, json.loads((Path(directory)/'raw.json').read_text()))
        return raw, runtime, plan

    def test_seventh_priming_failure_keeps_all_future_history_and_measured_rows(self):
        raw, runtime, plan = self.collect_failure('priming')
        self.assertEqual(raw['status'], 'failed')
        self.assertEqual(len(raw['priming_request_records']), 18)
        self.assertEqual([r['status'] for r in raw['priming_request_records']],
                         ['completed']*6 + ['incomplete'] + ['not_started']*11)
        self.assertEqual(len(raw['request_records']), sum(len(c['arrivals']) for c in plan['cases']))
        self.assertTrue(all(r['status'] == 'not_started' for r in raw['request_records']))
        self.assertEqual(raw['case_histories'], [])
        self.assertTrue(runtime.closed)
        self.assertEqual(raw['backend'], 'CPU_TEST_ONLY')
        self.assertFalse(raw['training_feature_extraction_eligible'])

    def test_monitor_close_failure_still_closes_workers_and_persists_failure(self):
        raw, runtime, _ = self.collect_failure('monitor')
        self.assertTrue(runtime.closed)
        self.assertEqual(raw['status'], 'failed')
        self.assertTrue(any('MonitoringCloseError' in error for error in raw['errors']))
        self.assertEqual(raw['lifecycle_boundaries'][-1]['boundary'], 'shutdown_end')

    def test_ready_receive_order_is_checked_separately_from_worker_perf_clock(self):
        plan, raw, _ = fixture()
        first = raw['priming_request_records'][0]
        following = next(r for r in raw['priming_request_records'][1:]+raw['request_records']
                         if r['model'] == first['model'])
        start = next(e for e in following['execution_events'] if e.get('event') is None)
        first['previous_handler_next_request_ready']['received_time_ns'] = start['received_time_ns']+1
        with self.assertRaisesRegex(ValueError, 'parent clock'):
            review.inspect_cohort(plan, raw)

    def test_existing_review_failure_stops_before_any_next_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, pp, item, manifest, mp = self.execution_fixture(root)
            final = Path(item['collection'])/'execution.json'
            write(final, completed_doc(item, m.receipt(mp)))
            write(final.parent/'raw.json', dict(schema=m.RAW_SCHEMA, collector_schema=m.RAW_SCHEMA,
                plan=item['plan'], execution_id=item['execution_id']))
            with patch.object(m, 'audit', return_value=manifest), patch.object(m, 'execute') as execute, \
                 patch.object(review, 'review', side_effect=ValueError('independent native-review rejection')):
                result = m.run(mp, root/'run', execute_enabled=True)
            self.assertEqual(result['status'], 'STOPPED_NO_RETRY')
            self.assertEqual(result['completed_attempts'], [])
            execute.assert_not_called()

    def test_ownership_completion_cannot_omit_a_cleanup_dimension(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, item, _, mp = self.execution_fixture(Path(directory))
            execution = completed_doc(item, m.receipt(mp))
            m.completed_execution(execution, m.receipt(mp), item)
            for name in ('child_reaped', 'live_group_pids', 'experiment_lock_held_through_group_exit',
                         'all_owned_descendants_exited_and_reaped', 'final_source_audit', 'watchdog'):
                mutant = copy.deepcopy(execution)
                del mutant[name]
                with self.subTest(name=name), self.assertRaises(ValueError):
                    m.completed_execution(mutant, m.receipt(mp), item)

    def test_cpu_raw_cannot_enter_native_review_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, pp, item, _, mp = self.execution_fixture(Path(directory))
            raw = dict(plan=m.receipt(pp), backend='CPU_TEST_ONLY', status='completed',
                       attempt_id=plan['attempt_id'], execution_id=plan['execution_id'])
            rp = Path(directory)/'raw.json'
            write(rp, raw)
            reviewed = dict(schema=review.SCHEMA,
                status='PASS_COMPLETE_MATCHING_NATIVE_TRAINING_NOT_PREDICTOR_QUALIFICATION',
                training_feature_extraction_eligible=True, issues=[], validation_labels_read=False,
                matching_manifest=m.receipt(mp), raw=m.receipt(rp))
            with self.assertRaisesRegex(ValueError, 'Native raw'):
                intake._review_identity(reviewed, plan, m.receipt(mp))

    def test_owned_peak_keeps_pre_case_history_while_energy_uses_measurement(self):
        raw = dict(lifecycle_boundaries=[dict(boundary='startup_begin', time_ns=10**9),
            dict(boundary='priming_end', time_ns=2*10**9),
            dict(boundary='monitoring_end', time_ns=73*10**9+1),
            dict(boundary='shutdown_begin', time_ns=73*10**9+2)],
            board_samples=[dict(time_ns=i*10**9, watts=100 if i <= 13 else 900,
                owned_used_mib=500 if i == 2 else 10) for i in range(1,74)],
            shutdown_memory_sampled=False, cases=[dict(case_id='c')],
            measurement_windows=[dict(case_id='c', start_ns=3*10**9, end_ns=13*10**9,
                                     fixed_drain_deadline_ns=73*10**9)])
        result = review.resource_facts(raw)
        case = result['cases'][0]
        self.assertEqual(result['startup']['occurrences'], 1)
        self.assertEqual(result['startup']['energy_joules'], 200)
        self.assertEqual(case['measurement_energy_joules'], 1000)
        self.assertGreater(case['drain_energy_joules_diagnostic_only'], 50_000)
        self.assertEqual(case['startup_through_this_drain_owned_sampled_peak_mib'], 500)
        self.assertEqual(case['local_measurement_through_drain_owned_sampled_peak_mib'], 10)
        self.assertFalse(case['memory_safety_factor_applied'])

    def test_real_macro_partition_keeps_priming_events_but_excludes_priming_labels(self):
        plan, raw, _ = fixture()
        rows, events = macros.request_rows(raw, plan, 'a'*64)
        priming = {r['request_id'] for r in raw['priming_request_records']}
        measured = [r['request_id'] for r in raw['request_records']]
        self.assertEqual([r['request_id'] for r in rows], measured)
        self.assertFalse(priming.intersection(r['request_id'] for r in rows))
        self.assertTrue(priming <= {e['request_id'] for e in events})
        for row in rows:
            self.assertTrue(all(m['training_replication_unit'] == plan['capture_id']
                                for m in row['macros']))
        raw['priming_request_records'][0]['status'] = 'incomplete'
        with self.assertRaisesRegex(ValueError, 'Incomplete cohort'):
            macros.request_rows(raw, plan, 'a'*64)


if __name__ == '__main__':
    unittest.main()
