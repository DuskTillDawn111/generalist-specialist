"""Independent ownership regressions using real Linux CPU process trees.

No NVML/GPU calls: process memory amounts are explicit synthetic observations.
The process relationships, start ticks, exits, reparenting and setsid are real.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import profile_owned_collect_v2 as v
from tools.predictor_validation_20261027.profile_process_ownership_v1 import process_table, subreaper


def runtime(process=None):
    return SimpleNamespace(workers={} if process is None else {
        'text#0': SimpleNamespace(process=process)})


def classify(values, processes, tracker):
    return v.memory_ownership(values, processes, os.getpid(), os.getpgrp(), ownership=tracker)


class RealCpuOwnershipTests(unittest.TestCase):
    def test_pinned_launcher_changed_start_ticks_is_unknown(self):
        launcher = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
        try:
            tracker = v.LiveCollectorOwnership(os.getpid())
            self.assertEqual(classify({launcher.pid: 1.}, runtime(launcher), tracker)['owned_used_mib'], 1.)
            original = v._proc_identity
            # Actual live child, with a controlled changed /proc identity. This
            # is a PID-reuse rejection test, not a claim to force real PID reuse.
            def reused(pid):
                row = original(pid)
                if pid == launcher.pid:
                    row['start_ticks'] += 1
                return row
            with patch.object(v, '_proc_identity', side_effect=reused):
                result = classify({launcher.pid: 8.}, runtime(launcher), tracker)
            self.assertEqual(result['owned_used_mib'], 0.)
            self.assertIn(launcher.pid, result['unresolved_nvml_pids'])
        finally:
            launcher.kill()
            launcher.wait()

    def test_reparent_during_proc_snapshot_is_unknown(self):
        launcher = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
        try:
            tracker = v.LiveCollectorOwnership(os.getpid())
            original = v._proc_identity
            count = 0
            def unstable(pid):
                nonlocal count
                row = original(pid)
                if pid == launcher.pid:
                    count += 1
                    if count >= 2:
                        row['ppid'] += 1
                return row
            with patch.object(v, '_proc_identity', side_effect=unstable):
                result = classify({launcher.pid: 8.}, runtime(launcher), tracker)
            self.assertEqual(result['owned_used_mib'], 0.)
            self.assertIn(launcher.pid, result['unresolved_nvml_pids'])
        finally:
            launcher.kill()
            launcher.wait()

    def test_first_observation_cannot_pin_an_exited_Popen_to_a_live_PID(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
        try:
            exited_original = SimpleNamespace(pid=child.pid, poll=lambda: 0)
            result = classify({child.pid: 2.}, runtime(exited_original), v.LiveCollectorOwnership(os.getpid()))
            self.assertEqual(result['owned_used_mib'], 0.)
            self.assertIn(child.pid, result['unresolved_nvml_pids'])
        finally:
            child.kill()
            child.wait()

    def test_two_aliases_cannot_both_claim_one_launcher(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
        try:
            ambiguous = runtime(child)
            ambiguous.workers['ocr#0'] = SimpleNamespace(process=child)
            result = classify({child.pid: 2.}, ambiguous, v.LiveCollectorOwnership(os.getpid()))
            self.assertEqual(result['owned_used_mib'], 0.)
            self.assertIn(child.pid, result['foreign_pids'])
        finally:
            child.kill()
            child.wait()

    def test_two_real_setsid_levels_are_owned_via_declared_launcher(self):
        code = ('import json,subprocess,sys;'
                'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"],start_new_session=True);'
                'print(json.dumps(dict(worker=p.pid)),flush=True);p.wait()')
        launcher = subprocess.Popen([sys.executable, '-u', '-c', code], start_new_session=True,
                                    stdout=subprocess.PIPE, text=True)
        worker = None
        try:
            worker = json.loads(launcher.stdout.readline())['worker']
            table = process_table()
            self.assertEqual(table[worker]['ppid'], launcher.pid)
            self.assertEqual(table[launcher.pid]['ppid'], os.getpid())
            self.assertNotEqual(table[worker]['process_group'], os.getpgrp())
            self.assertNotEqual(table[launcher.pid]['process_group'], os.getpgrp())
            tracker = v.LiveCollectorOwnership(os.getpid())
            result = classify({worker: 12., launcher.pid: 4.}, runtime(launcher), tracker)
            self.assertEqual(result['owned_used_mib'], 16.)
            self.assertEqual(result['owned_aliases'], ['text#0'])
            self.assertEqual(result['foreign_pids'], [])
            self.assertEqual(result['unresolved_nvml_pids'], [])
            self.assertEqual(set(result['owned_processes_mib']), {worker, launcher.pid})
            self.assertEqual(len(result['ownership_evidence']), 2)
        finally:
            if worker is not None:
                try:
                    os.kill(worker, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                launcher.wait(timeout=3)
            except subprocess.TimeoutExpired:
                launcher.kill()
                launcher.wait()
            launcher.stdout.close()

    def test_live_launcher_can_change_PGID_without_changing_start_identity(self):
        code = ('import os,sys,time;print("ready",flush=True);sys.stdin.readline();'
                'os.setsid();print("changed",flush=True);time.sleep(30)')
        launcher = subprocess.Popen([sys.executable, '-u', '-c', code], stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(launcher.stdout.readline().strip(), 'ready')
            tracker = v.LiveCollectorOwnership(os.getpid())
            first = process_table()[launcher.pid]
            self.assertEqual(first['process_group'], os.getpgrp())
            before = classify({launcher.pid: 7.}, runtime(launcher), tracker)
            launcher.stdin.write('go\n')
            launcher.stdin.flush()
            self.assertEqual(launcher.stdout.readline().strip(), 'changed')
            second = process_table()[launcher.pid]
            after = classify({launcher.pid: 8.}, runtime(launcher), tracker)
            self.assertEqual(first['start_ticks'], second['start_ticks'])
            self.assertNotEqual(first['process_group'], second['process_group'])
            self.assertEqual(before['owned_used_mib'], 7.)
            self.assertEqual(after['owned_used_mib'], 8.)
        finally:
            launcher.kill()
            launcher.wait()
            launcher.stdin.close()
            launcher.stdout.close()

    def test_undeclared_collector_descendant_is_not_a_worker_owner(self):
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
        try:
            result = classify({unrelated.pid: 5.}, runtime(), v.LiveCollectorOwnership(os.getpid()))
            self.assertEqual(result['owned_used_mib'], 0.)
            self.assertIn(unrelated.pid, result['foreign_pids'])
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.kill()
            unrelated.wait()

    def test_same_PGID_external_process_is_foreign(self):
        # Probe runs as a collector. This test process is a live external
        # ancestor in its same PGID, never that collector's descendant.
        code = ('import json,os,sys;from types import SimpleNamespace;'
                'from tools.predictor_validation_20261027 import profile_owned_collect_v2 as v;'
                'pid=int(sys.argv[1]);assert os.getpgid(pid)==os.getpgrp();'
                'r=v.memory_ownership({pid:9.},SimpleNamespace(workers={}),os.getpid(),os.getpgrp(),'
                'ownership=v.LiveCollectorOwnership(os.getpid()));print(json.dumps(r))')
        probe = subprocess.run([sys.executable, '-c', code, str(os.getpid())],
                               text=True, capture_output=True, check=True)
        result = json.loads(probe.stdout)
        self.assertEqual(result['owned_used_mib'], 0.)
        self.assertIn(os.getpid(), result['foreign_pids'])

    def test_exited_NVML_PID_is_unresolved_not_owned(self):
        exited = subprocess.Popen([sys.executable, '-c', 'pass'])
        exited.wait()
        self.assertNotIn(exited.pid, process_table())
        result = classify({exited.pid: 4.}, runtime(exited), v.LiveCollectorOwnership(os.getpid()))
        self.assertEqual(result['owned_used_mib'], 0.)
        self.assertIn(exited.pid, result['unresolved_nvml_pids'])

    def test_exited_launcher_orphan_cannot_fall_back_to_generic_startup_owner(self):
        previous = subreaper(True)
        launcher = None
        child = None
        try:
            code = ('import subprocess,sys;'
                    'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"],'
                    'start_new_session=True);print(p.pid,flush=True)')
            launcher = subprocess.Popen([sys.executable, '-u', '-c', code],
                                        stdout=subprocess.PIPE, text=True)
            child = int(launcher.stdout.readline())
            launcher.wait(timeout=3)
            until = time.monotonic() + 2
            while process_table()[child]['ppid'] != os.getpid() and time.monotonic() < until:
                time.sleep(.01)
            self.assertEqual(process_table()[child]['ppid'], os.getpid())
            result = classify({child: 11.}, runtime(launcher), v.LiveCollectorOwnership(os.getpid()))
            self.assertEqual(result['owned_used_mib'], 0.)
            self.assertIn(child, result['foreign_pids'] + result['unresolved_nvml_pids'])
        finally:
            if child is not None:
                try:
                    os.kill(child, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    os.waitpid(child, 0)
                except ChildProcessError:
                    pass
            if launcher is not None:
                if launcher.poll() is None:
                    launcher.kill()
                    launcher.wait()
                launcher.stdout.close()
            subreaper(previous)


if __name__ == '__main__':
    unittest.main()
