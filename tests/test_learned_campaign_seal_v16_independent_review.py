"""Independent supervision and parent-gate checks using synthetic CPU fixtures.

Shared setup supplies tiny valid protocol documents, not forecast execution.
"""
import json
from pathlib import Path
import sys
import unittest

from tests import test_learned_campaign_seal_v16 as author_fixtures
from tools.predictor_validation_20261027 import learned_campaign_parallel_v16 as parallel


class ParallelCampaignIndependentTests(unittest.TestCase):
    def setUp(self):
        self.fixture = author_fixtures.ParallelCampaignTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_real_four_workers_reach_rendezvous_before_any_can_finish(self):
        fixture = self.fixture
        specs, protocol, _ = fixture.supervisor_fixture()
        barrier = fixture.root/'rendezvous'
        barrier.mkdir()
        script = fixture.root/'independent_barrier_worker.py'
        script.write_text('''import json,os,pathlib,sys,time
spec=json.loads(pathlib.Path(sys.argv[1]).read_text())
barrier=pathlib.Path(sys.argv[2])
(barrier/('started-'+spec['candidate_id'])).write_text(str(os.getpid()))
began=time.monotonic()
while len(list(barrier.glob('started-*')))!=4:
    if time.monotonic()-began>3: raise SystemExit(9)
    time.sleep(.01)
out=pathlib.Path(spec['output_directory'])
(out/'worker-receipt.json').write_text(json.dumps(dict(pid=os.getpid(),all_four_seen=True)))
''')
        report = parallel.run_parallel(specs, protocol, timeout_seconds=5,
            _command_factory=lambda spec, unused:
                [sys.executable, str(script), spec['path'], str(barrier)])
        self.assertEqual(report['status'], 'ALL_FOUR_CPU_WORKERS_EXITED_ZERO')
        self.assertTrue(report['all_workers_reaped'])
        self.assertEqual(len({w['pid'] for w in report['workers']}), 4)
        for worker in report['workers']:
            receipt = json.loads(Path(worker['receipt']['path']).read_text())
            self.assertEqual(receipt['pid'], worker['pid'])
            self.assertTrue(receipt['all_four_seen'])

    def modify_only_fresh_result(self, modify):
        original = self.fixture.result

        def result(candidate, *, shift, term):
            result = original(candidate, shift=shift, term=term)
            if shift == 50:
                modify(result)
            return result

        self.fixture.result = result

    def test_parent_completeness_rejects_missing_finish_despite_complete_worker_receipts(self):
        self.modify_only_fresh_result(lambda result:
            result['forecast']['finishes'].pop(next(iter(result['forecast']['finishes']))))
        with self.assertRaisesRegex(ValueError, 'Incomplete predictions'):
            self.fixture.run_producer()
        self.assertFalse((self.fixture.out/'predictions.json').exists())
        failure = json.loads((self.fixture.out/'failed-production.json').read_text())
        self.assertEqual(len(failure['worker_audits']), 4)
        self.assertTrue(all(w['status'] == parallel.COMPLETE for w in failure['worker_audits']))
        self.assertTrue(all(len(json.loads(Path(w['result']['path']).read_text())['full_public_cohort']) == 57
                            for w in failure['worker_audits']))

    def test_parent_cell_validation_rejects_negative_term_despite_complete_workers(self):
        self.modify_only_fresh_result(lambda result: result['metrics']['terms'].__setitem__('E', -.1))
        with self.assertRaises(ValueError):
            self.fixture.run_producer()
        self.assertFalse((self.fixture.out/'predictions.json').exists())
        failure = json.loads((self.fixture.out/'failed-production.json').read_text())
        self.assertEqual(len(failure['worker_audits']), 4)
        self.assertTrue(all(w['status'] == parallel.COMPLETE for w in failure['worker_audits']))

    def test_nonfinite_fresh_result_fails_JSON_and_cannot_borrow_preflight_cells(self):
        self.modify_only_fresh_result(lambda result: result['metrics']['terms'].__setitem__('E', float('nan')))
        with self.assertRaises(ValueError):
            self.fixture.run_producer()
        self.assertFalse((self.fixture.out/'predictions.json').exists())
        self.assertTrue((self.fixture.out/'failed-production.json').exists())
        receipts = list((self.fixture.out/'workers').glob('*/worker-receipt.json'))
        self.assertEqual(len(receipts), 4)
        for path in receipts:
            worker = json.loads(path.read_text())
            self.assertEqual(worker['status'], 'WORKER_FAILED_PROHIBITS_SEAL')
            self.assertIsNone(worker['result'])


if __name__ == '__main__':
    unittest.main()
