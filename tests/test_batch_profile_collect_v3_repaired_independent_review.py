"""Independent CPU regression and integrated watchdog tests of repaired v3."""
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from tests import test_batch_profile_collect_v3_independent_review as old_review
from tests.test_shared_text_batch_v2 import backend, sample
from gsdeploy.runtime.batching import BatchConfiguration
from tools.predictor_validation_20261027 import batch_profile_collect_v3 as c

ROOT = Path(__file__).resolve().parents[1]
METRICS = {}


def prepare(root, *, unsuccessful_prime=False, active_tail=False):
    args = old_review.fixtures.BatchProfileCollectV3Tests().fixture(root)
    if unsuccessful_prime:
        priming = c.read(args[4])
        priming['training']['samples'].append(sample('prime-length',[1,4]))
        priming['training']['arrivals'][0]['sample_id'] = 'prime-length'
        args[4].write_text(json.dumps(priming))
    if active_tail:
        arrivals = c.read(args[3])
        arrivals['training'][2]['arrival_ns'] = 10_000_000
        arrivals['training'][3]['arrival_ns'] = 20_000_000
        arrivals['training'].append(dict(request_id='t5',sample_id='ta',alias='text#1',arrival_ns=15_000_000))
        args[3].write_text(json.dumps(arrivals))
    c.prepare(root/'plan',*args,BatchConfiguration(2,100,1_000_000),measurement_ns=40_000_000,
              drain_timeout_seconds=.3,supervisor_timeout_seconds=5.)
    return root/'plan/plan.json'


def blocked_child(plan_path):
    runtime = old_review.fixtures.BackgroundRuntime()
    def factory():
        owner = backend(); execute = owner.execute_prepared; calls = [0]
        def blocked(rows, **kw):
            calls[0] += 1
            if calls[0] == 2:
                child = subprocess.Popen([sys.executable,'-c',
                    'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)'])
                Path(plan_path).with_name('blocked.json').write_text(json.dumps(dict(grandchild=child.pid)))
                threading.Event().wait(30)
            return execute(rows,**kw)
        owner.execute_prepared = blocked
        return owner
    c.collect(plan_path,'training',backend_factory=factory,background_runtime_factory=lambda:runtime)


class RepairedIndependentBatchReview(unittest.TestCase):
    def test_unsuccessful_priming_blocks_epoch_and_preserves_full_plans(self):
        with tempfile.TemporaryDirectory() as directory:
            path = prepare(Path(directory),unsuccessful_prime=True)
            runtime = old_review.fixtures.BackgroundRuntime()
            report = c.collect(path,'training',backend_factory=backend,background_runtime_factory=lambda:runtime)
            self.assertIsNone(report['epoch_ns'])
            self.assertFalse(runtime.measurement_received.is_set())
            self.assertEqual(report['status'],'INCOMPLETE_OR_FAILED_NOT_QUALIFIED')
            self.assertEqual(set(report['unresolved_request_ids']),{'t1','t2','t3','t4'})
            self.assertEqual(len(c.read(c.checked(report['profile_index']))['requests']),4)
            checkpoint = c.read(c.checked(report['checkpoint']))
            self.assertEqual(len(checkpoint['offered_cohort']),4)
            self.assertEqual(len(checkpoint['priming_cohort']),3)
            self.assertIsNone(checkpoint['epoch_ns'])
            METRICS['unsuccessful_priming_repaired'] = dict(epoch_blocked=True,
                offered_and_indexed=4,priming_planned=3,missing_requests_not_removed=True)

    def test_both_background_owners_run_during_text_block_and_tail_arrival_survives(self):
        with tempfile.TemporaryDirectory() as directory:
            path = prepare(Path(directory),active_tail=True)
            runtime = old_review.fixtures.BackgroundRuntime()
            both, timing = threading.Event(), {}
            original_enqueue = runtime.enqueue
            def enqueue(row, value):
                original_enqueue(row,value)
                if {r['model'] for r in runtime.records if r['phase']=='measurement'} == set(c.BACKGROUND):
                    both.set()
            runtime.enqueue = enqueue
            def factory():
                owner = backend(); execute = owner.execute_prepared; calls = [0]
                def blocked(rows, **kw):
                    calls[0] += 1
                    if calls[0] == 2:
                        timing['text_start_ns'] = time.perf_counter_ns()
                        if not both.wait(.25):
                            raise AssertionError('Both declared background owners did not arrive while text was blocked')
                        timing['text_unblocked_ns'] = time.perf_counter_ns()
                    return execute(rows,**kw)
                owner.execute_prepared = blocked
                return owner
            report = c.collect(path,'training',backend_factory=factory,background_runtime_factory=lambda:runtime)
            self.assertEqual(report['status'],'CPU_TEST_ONLY_COMPLETE')
            self.assertEqual(report['offered'],5)
            backgrounds = [r for r in runtime.records if r['phase']=='measurement']
            self.assertEqual({r['model'] for r in backgrounds},set(c.BACKGROUND))
            for row in backgrounds:
                self.assertGreaterEqual(row['actual_submit_time_ns'],timing['text_start_ns'])
                self.assertLess(row['completion_time_ns'],timing['text_unblocked_ns'])
                self.assertEqual(row['scheduled_time_ns'],report['epoch_ns']+row['scheduled_offset_ns'])
            tail = next(r for r in report['results'] if r['request_id']=='t5')
            self.assertEqual(tail['scheduled_arrival_ns'],15_000_000)
            self.assertGreaterEqual(tail['original_text_result']['prepare_start_ns'],
                                    timing['text_unblocked_ns']-report['epoch_ns'])
            METRICS['both_backgrounds_during_text_block'] = dict(
                offered_and_indexed=5,background_owners=sorted(c.BACKGROUND),
                both_receipts_before_text_unblocks=True,tail_scheduled_arrival_ns=15_000_000,
                tail_prepare_wait_ns=tail['original_text_result']['pre_prepare_wait_ns'],
                CPU_fake_request_execution=True)

    def test_integrated_blocked_collector_checkpoints_and_entire_group_are_supervised(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = prepare(root)
            command = [sys.executable,'-m',__name__,'--blocked-child',str(path)]
            # The complete collector uses injected CPU owners. The supervisor
            # and actual periodic checkpoint writer are production functions.
            outcome = c.supervise_process(command,root,.8,grace_seconds=.1)
            self.assertTrue(outcome['timed_out'])
            self.assertTrue(outcome['all_process_group_members_exited'])
            self.assertEqual(c._live_group_pids(outcome['process_group']),[])
            self.assertTrue(path.with_name('blocked.json').exists())
            checkpoint = c.read(path.parent/'training/collection/progress.json')
            self.assertEqual(len(checkpoint['offered_cohort']),4)
            self.assertEqual(len(checkpoint['priming_cohort']),3)
            self.assertIsNotNone(checkpoint['epoch_ns'])
            self.assertEqual(len(checkpoint['background_records']),2)
            self.assertTrue(all(r['status']=='completed' for r in checkpoint['background_records']))
            self.assertEqual({e['event'] for e in checkpoint['text_events']} & {'BATCH_START','BATCH_COMPLETE'}, {'BATCH_START'})
            self.assertEqual(checkpoint['board_samples'],[])
            self.assertTrue(checkpoint['incomplete_checkpoint_not_final_report'])
            METRICS['integrated_watchdog'] = dict(timed_out=True,
                all_process_group_members_exited=True,live_group_pids_after=[],
                offered_persisted=4,priming_persisted=3,background_receipt_prefix_persisted=2,
                actual_text_batch_start_persisted=True,missing_text_completion_not_imputed=True,
                missing_board_samples_not_imputed=True,CPU_injected_owners_only=True,
                signals=outcome['signals'],elapsed_seconds=outcome['elapsed_seconds'])


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--blocked-child':
        blocked_child(sys.argv[2])
    else:
        unittest.main()
