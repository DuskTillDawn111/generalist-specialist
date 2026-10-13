"""CPU checks for measured accepted-path accounting and final-state metrics."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from deployment_search.dynamic_real_pilot import LiveAcceptedPath, gap, saving, stamp
from deployment_search.dynamic_adaptation import StructuralReconfigurationCost
from deployment_search.tests.fixtures import model


class FakeLivePath(LiveAcceptedPath):
    """Fake transport only; retains the actual orchestration/event accounting."""
    def load(self, name):
        self.workers[name] = SimpleNamespace(process=SimpleNamespace(
            pid=100+len(self.workers), poll=lambda: None))
        self.ready[name] = dict(stamp(), pid=self.workers[name].process.pid)

    def unload(self, name):
        self.workers.pop(name)
        self.ready.pop(name)
        return dict(forced=False, exit_code=0, exited_timestamp=stamp())


class DynamicPilotAccountingTests(unittest.TestCase):
    def test_accepted_path_keeps_retained_workers_and_excludes_setup(self):
        catalog = {'base':model('base', 3, text=100), 'extra':model('extra', 2, text=50)}
        old = dict(base=1, extra=0)
        new = dict(base=1, extra=1)
        costs = StructuralReconfigurationCost(catalog)
        path = [dict(mode='normal', iteration=0, old_deployment=old,
                     new_deployment=new, cost=costs.estimate(old,new))]
        with TemporaryDirectory() as temp:
            live = FakeLivePath(catalog, Path(temp))
            events = live.execute(old, path)
            self.assertEqual(len(events),1)
            event = events[0]
            self.assertEqual(event['models_to_load'],['extra'])
            self.assertEqual(event['models_to_unload'],[])
            self.assertEqual(event['retained_worker_pids']['base'],event['workers_ready_timestamp']['base']['pid'])
            self.assertGreaterEqual(event['measured_switch_time_ms'],0)
            self.assertLess(event['workers_ready_timestamp']['base']['monotonic_ns'],
                            event['switch_start_timestamp']['monotonic_ns'])
            self.assertEqual(event['cost']['models_added'],1)
            self.assertEqual(event['cost']['added_vram_mib'],2)
            self.assertFalse(live.workers)  # Teardown not reported as an accepted switch.

    def test_no_switch_does_not_fabricate_latency(self):
        catalog = {'base':model('base', 3, text=100)}
        with TemporaryDirectory() as temp:
            live = FakeLivePath(catalog, Path(temp))
            self.assertEqual(live.execute(dict(base=1), []), [])
            self.assertTrue((Path(temp)/'final_residency_verified.json').exists())

    def test_resume_does_not_reapply_completed_switch(self):
        catalog = {'base':model('base', 3, text=100), 'extra':model('extra', 2, text=50)}
        old,new=dict(base=1,extra=0),dict(base=1,extra=1)
        path=[dict(mode='normal',old_deployment=old,new_deployment=new,
                   cost=StructuralReconfigurationCost(catalog).estimate(old,new))]
        with TemporaryDirectory() as temp:
            first=FakeLivePath(catalog,Path(temp)).execute(old,path)
            second=FakeLivePath(catalog,Path(temp)).execute(old,path)
            self.assertEqual(first,second)

    def test_undefined_ratios_are_null(self):
        self.assertIsNone(saving(0,0))
        self.assertIsNone(gap(1,0))
        self.assertAlmostEqual(saving(3,6),.5)
        self.assertAlmostEqual(gap(105,100),.05)

    def test_final_acceptance_counts_both_top_b_evaluations(self):
        from module5b_finalize import acceptance_evaluation_index
        record = dict(final_deployment=dict(base=1,extra=1),
            runtime_accesses=[dict(deployment=dict(base=1,extra=0))],
            trace=dict(iterations=[dict(num_runtime_candidates=2,accepted=['base','extra']),
                                   dict(num_runtime_candidates=2,accepted=None)]))
        self.assertEqual(acceptance_evaluation_index(record),3)
        record['final_deployment']=dict(base=1,extra=0)
        self.assertEqual(acceptance_evaluation_index(record),1)


if __name__=='__main__':
    unittest.main()
