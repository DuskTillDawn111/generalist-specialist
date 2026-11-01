"""CPU counterexamples; no signals, native workers, or live gate invocation."""
import json
from pathlib import Path
import signal
import tempfile
import types
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_public181_resource_capacity_gate_v1 as m


class CapacityGateTests(unittest.TestCase):
    def fixture(self, folder):
        root = Path(folder)
        identity = dict(pid=m.PID, start_ticks=m.START_TICKS, command=['python', '-m', 'original'], cwd=folder)
        hold = dict(resource_identity=identity)
        hold_path = root/'hold.json'; hold_path.write_text(json.dumps(hold))
        queue_root = root/'original-queue'; queue_root.mkdir()
        manifest = dict(output=str(queue_root))
        resource = dict(attempts=[dict(claim=str(root/('claim'+str(i))),
                                      collection=str(root/('collection'+str(i)))) for i in range(8)])
        final_path = root/'kernel-report.json'; final_path.write_text(json.dumps({'CPU_TEST_ONLY': True}))
        outcome = ('KERNEL_COMPLETE', m.receipt(final_path), {'CPU_TEST_ONLY': True})
        refs = [m.receipt(hold_path)]
        audited = (hold, manifest, resource, {}, {}, refs)
        return hold_path, audited, outcome

    def invoke(self, hold_path, audited, outcome, calls, **kwargs):
        defaults = dict(identity_reader=lambda identity: 'T', outcome_reader=lambda *a: outcome,
                        disk_usage=lambda path: types.SimpleNamespace(free=m.PREFLIGHT),
                        pidfd_open=lambda *a: 71, pidfd_close=lambda fd: None,
                        signal_sender=lambda pid, sig, fd: calls.append((pid, sig, fd)))
        defaults.update(kwargs)
        with patch.object(m, 'audit', return_value=audited):
            return m.run(hold_path, **defaults)

    def test_exact_original_threshold_signals_original_pid_once_and_cannot_repeat(self):
        with tempfile.TemporaryDirectory() as folder:
            path, audited, outcome = self.fixture(folder); calls = []
            result = self.invoke(path, audited, outcome, calls)
            self.assertEqual(calls, [(m.PID, signal.SIGCONT, 71)])
            self.assertEqual(result['signal_calls'], 1)
            self.assertTrue(result['first_launch_only_not_eight_attempt_completion_guarantee'])
            self.assertEqual(result['additional_arbitrary_reserve_bytes'], 0)
            self.assertEqual(result['declared_eight_attempt_incremental_scenario_bytes'], 48*2**30)
            self.assertIsNone(result['native181_measured_storage_estimate_bytes'])
            with self.assertRaises(FileExistsError): self.invoke(path, audited, outcome, calls)
            self.assertEqual(len(calls), 1)

    def test_insufficient_space_waits_and_exception_never_resumes(self):
        with tempfile.TemporaryDirectory() as folder:
            path, audited, outcome = self.fixture(folder); calls = []; waits = []
            def pause(seconds):
                waits.append(seconds)
                raise RuntimeError('CPU stop polling')
            result = self.invoke(path, audited, outcome, calls, pause=pause,
                                 disk_usage=lambda p: types.SimpleNamespace(free=m.PREFLIGHT-1))
            self.assertEqual(waits, [30]); self.assertEqual(calls, [])
            self.assertEqual(result['status'], 'HOLD_NO_AUTOMATIC_RESUME')
            self.assertFalse((path.parent/'run-v1/resume-intent.json').exists())

    def test_capacity_recovers_then_exactly_one_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            path, audited, outcome = self.fixture(folder); calls = []; waits = []; free = [m.PREFLIGHT-1]
            def pause(seconds):
                waits.append(seconds); free[0] = m.PREFLIGHT
            result = self.invoke(path, audited, outcome, calls, pause=pause,
                                 disk_usage=lambda p: types.SimpleNamespace(free=free[0]))
            self.assertEqual(waits, [30]); self.assertEqual(len(calls), 1)
            self.assertTrue(result['status'].startswith('RESUMED_'))

    def test_identity_reuse_unstopped_child_or_exception_never_signals(self):
        for reason in ('PID reused', 'not stopped', 'has children', 'unexpected exception'):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as folder:
                path, audited, outcome = self.fixture(folder); calls = []
                def bad(identity): raise ValueError(reason)
                result = self.invoke(path, audited, outcome, calls, identity_reader=bad)
                self.assertEqual(calls, []); self.assertEqual(result['status'], 'HOLD_NO_AUTOMATIC_RESUME')

    def test_kernel_missing_waits_without_resume_and_failed_kernel_releases_only_original_hold(self):
        with tempfile.TemporaryDirectory() as folder:
            path, audited, outcome = self.fixture(folder); calls = []
            def stop(seconds): raise KeyboardInterrupt('CPU stop')
            result = self.invoke(path, audited, None, calls, pause=stop)
            self.assertEqual(calls, []); self.assertEqual(result['status'], 'HOLD_NO_AUTOMATIC_RESUME')
        with tempfile.TemporaryDirectory() as folder:
            path, audited, outcome = self.fixture(folder); calls = []
            failed = ('KERNEL_FAILED_RELEASE_TO_ORIGINAL_HOLD', *outcome[1:])
            result = self.invoke(path, audited, failed, calls,
                                 disk_usage=lambda p: types.SimpleNamespace(free=0))
            self.assertEqual(len(calls), 1)
            self.assertEqual(result['status'], 'RESUMED_ORIGINAL_WAITER_FOR_KERNEL_FAILURE_HOLD')

    def test_source_mutation_started_claim_and_final_capacity_drop_never_resume(self):
        for mode in ('source', 'claim', 'queue-started', 'drop'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                path, audited, outcome = self.fixture(folder); calls = []; reads = []
                def disk(p):
                    reads.append(1)
                    if mode == 'source': path.write_text('changed')
                    if mode == 'claim': Path(audited[2]['attempts'][0]['claim']).write_text('already claimed')
                    if mode == 'queue-started': (Path(audited[1]['output'])/'native-started.json').write_text('{}')
                    return types.SimpleNamespace(free=m.PREFLIGHT if len(reads)==1 else m.PREFLIGHT-1)
                result = self.invoke(path, audited, outcome, calls, disk_usage=disk)
                self.assertEqual(calls, []); self.assertEqual(result['status'], 'HOLD_NO_AUTOMATIC_RESUME')

    def test_process_reader_checks_actual_start_ticks_command_state_and_all_thread_children(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); p = root/str(m.PID); task = p/'task'/str(m.PID); task.mkdir(parents=True)
            identity = dict(pid=m.PID, start_ticks=m.START_TICKS, command=['python', '-m', 'original'], cwd=folder)
            (p/'cwd').symlink_to(root); (task/'children').write_text('')
            (p/'cmdline').write_bytes(b'python\0-m\0original\0')
            def stat(state='T', ticks=m.START_TICKS):
                (p/'stat').write_text(str(m.PID)+' (python name) '+' '.join([state]+['0']*18+[ticks]))
            stat(); self.assertEqual(m.process_state(identity, proc_root=root), 'T')
            for mode in ('ticks', 'state', 'command', 'children'):
                stat(); (p/'cmdline').write_bytes(b'python\0-m\0original\0'); (task/'children').write_text('')
                if mode == 'ticks': stat(ticks='reused')
                if mode == 'state': stat(state='S')
                if mode == 'command': (p/'cmdline').write_bytes(b'foreign\0')
                if mode == 'children': (task/'children').write_text('123')
                with self.subTest(mode=mode), self.assertRaises(ValueError): m.process_state(identity, proc_root=root)

    def test_production_sender_uses_pidfd_sigcont_only(self):
        with patch.object(m, '_pidfd_syscall') as send, patch.object(m.os, 'kill') as kill:
            m.send_continue(m.PID, signal.SIGCONT, 71)
            self.assertEqual(send.call_count, 1)
            self.assertEqual(send.call_args.args[:3], (424, 71, signal.SIGCONT))
            kill.assert_not_called()
            with self.assertRaises(ValueError): m.send_continue(m.PID, signal.SIGTERM, 71)

    def test_kernel_complete_uses_canonical_full_validator_and_foreign_failure_is_rejected(self):
        manifest = dict(kernel_final_path='/canonical/report.json', kernel_manifest={'path': '/kernel', 'sha256': 'a'})
        final = dict(schema=m.queue.kernel.SCHEMA+'.run', manifest=manifest['kernel_manifest'],
                     status='STOPPED_NO_RETRY_PREFIX_PRESERVED', fit_started=False,
                     automatic_retry=False, validation_labels_read=False)
        ref = {'path': '/canonical/report.json', 'sha256': 'b'}
        with patch.object(m.queue.kernel, 'load_publication', return_value=(ref, final)), \
                patch.object(m.queue, 'verify'):
            self.assertEqual(m.kernel_outcome(manifest, {}, {})[0], 'KERNEL_FAILED_RELEASE_TO_ORIGINAL_HOLD')
            final['validation_labels_read'] = True
            with self.assertRaises(ValueError): m.kernel_outcome(manifest, {}, {})
        with patch.object(m.queue.kernel, 'load_publication', return_value=(ref, {'status': 'complete'})), \
                patch.object(m.queue, 'completed_kernel', side_effect=ValueError('incomplete canonical stages')):
            with self.assertRaisesRegex(ValueError, 'incomplete canonical stages'):
                m.kernel_outcome(manifest, {}, {})

    def test_final_identity_reuse_after_open_or_signal_exception_is_never_retried(self):
        for mode in ('identity', 'send'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                path, audited, outcome = self.fixture(folder); calls = []; identity_calls = []
                def identity(value):
                    identity_calls.append(1)
                    if mode == 'identity' and len(identity_calls)>1: raise ValueError('PID reused at boundary')
                def send(pid, sig, fd):
                    calls.append((pid, sig, fd))
                    raise OSError('CPU injected signal failure')
                result = self.invoke(path, audited, outcome, calls, identity_reader=identity, signal_sender=send)
                self.assertEqual(len(calls), 0 if mode == 'identity' else 1)
                self.assertEqual(result['status'], 'HOLD_NO_AUTOMATIC_RESUME')


if __name__ == '__main__': unittest.main()
