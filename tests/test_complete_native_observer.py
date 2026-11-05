"""Boundary bookkeeping only; these fixtures do not establish CUDA semantics."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import complete_observed_worker as worker


class CompleteObserverTests(unittest.TestCase):
    def observe(self, fail=False):
        events=[];stack=[]
        nvtx=SimpleNamespace(range_push=stack.append,range_pop=stack.pop)
        torch=SimpleNamespace(cuda=SimpleNamespace(nvtx=nvtx))
        with patch.object(worker,'emit',side_effect=lambda kind,row:events.append((kind,row))), \
             patch.object(worker.time,'perf_counter_ns',side_effect=[110,120,130]):
            observer=worker.CompleteObserver(torch,'r',100)
            def operation():
                if fail:raise ValueError('native failure')
                return 7
            if fail:
                with self.assertRaisesRegex(ValueError,'native failure'):
                    observer.call('inference','gpu',operation)
            else:self.assertEqual(observer.call('inference','gpu',operation),7)
            end=observer.finish()
        self.assertEqual(end,130);self.assertEqual(stack,[])
        self.assertEqual([(s['start_ns'],s['end_ns']) for s in observer.stages],
                         [(100,110),(110,120),(120,130)])
        self.assertEqual([s['kind'] for s in observer.stages],['host','gpu','host'])
        self.assertEqual([r['event'] for _,r in events],['start','end']*3)
        self.assertTrue(all(r['request_id']=='r' for _,r in events))

    def test_shared_observed_boundaries_cover_handler(self):self.observe()

    def test_native_exception_closes_call_and_retains_host_tail(self):self.observe(fail=True)


if __name__=='__main__':unittest.main()
