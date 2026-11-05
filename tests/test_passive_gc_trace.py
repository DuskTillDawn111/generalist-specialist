"""Passive GC observations must not alter policy or hide incomplete coverage."""
import gc
import importlib.util
from pathlib import Path
import unittest

PATH=Path(__file__).resolve().parents[1]/'tools/predictor_validation_20261027/gc_trace_scoped_worker.py'
spec=importlib.util.spec_from_file_location('passive_gc_test_worker',PATH)
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)


class PassiveGCTests(unittest.TestCase):
    def test_collection_is_retained_without_clipping_or_policy_changes(self):
        before=(gc.isenabled(),gc.get_threshold(),list(gc.callbacks))
        clock=iter([100,160]);trace=worker.GCTrace(clock=lambda:next(clock))
        trace.callback('start',dict(generation=2));trace.callback('stop',dict(generation=2,collected=12,uncollectable=0))
        result=trace.snapshot(120,150)
        self.assertTrue(result['coverage_complete']);self.assertEqual(result['events'][0]['start_ns'],100)
        self.assertEqual(result['events'][0]['end_ns'],160)
        self.assertEqual(result['events'][0]['collected'],12)
        self.assertEqual((gc.isenabled(),gc.get_threshold(),list(gc.callbacks)),before)

    def test_dropped_events_only_allow_covered_later_windows(self):
        clock=iter([0,10,20,30,40,50]);trace=worker.GCTrace(clock=lambda:next(clock),capacity=2)
        for i in range(3):
            trace.callback('start',dict(generation=0));trace.callback('stop',dict(generation=0,collected=0,uncollectable=0))
        self.assertFalse(trace.snapshot(0,60)['coverage_complete'])
        self.assertTrue(trace.snapshot(25,60)['coverage_complete'])
        self.assertEqual(trace.snapshot(25,60)['dropped_process_events'],1)

    def test_incomplete_or_mismatched_collection_cannot_claim_coverage(self):
        clock=iter([100,160]);trace=worker.GCTrace(clock=lambda:next(clock))
        trace.callback('start',dict(generation=0));self.assertFalse(trace.snapshot(90,150)['coverage_complete'])
        trace.callback('stop',dict(generation=1,collected=0,uncollectable=0))
        self.assertFalse(trace.snapshot(90,180)['coverage_complete'])


if __name__=='__main__':unittest.main()
