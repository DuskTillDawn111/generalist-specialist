"""GC correlation uses host-clock unions and rejects incomplete observations."""
import unittest
from experiments.analyze_activation_gc_diagnostic import overlap_ms,validate_trace


class GCAnalysisTests(unittest.TestCase):
    def test_host_intersection_union_clips_without_double_counting(self):
        events=[dict(start_ns=0,end_ns=50_000_000),dict(start_ns=40_000_000,end_ns=90_000_000)]
        self.assertEqual(overlap_ms(events,20_000_000,70_000_000),50)
        self.assertEqual(overlap_ms(events,90_000_000,100_000_000),0)

    def test_trace_rejects_missing_coverage_and_policy_change(self):
        policy=dict(gc_enabled=True,thresholds=[700,10,10])
        trace=dict(schema='gsdeploy.passive_gc_trace.v1',enabled=True,clock='host_perf_counter_ns',
            window_start_ns=10,window_end_ns=20,coverage_complete=True,errors=[],gc_policy_unchanged=True,
            gc_enabled=True,gc_thresholds=[700,10,10],events=[dict(start_ns=5,end_ns=25,generation=2)])
        validate_trace(trace,10,20,policy)
        for replacement in [dict(coverage_complete=False),dict(gc_thresholds=[0,10,10]),dict(errors=['drop']),
                            dict(window_end_ns=21),dict(events=[dict(start_ns=20,end_ns=25,generation=2)])]:
            with self.subTest(replacement=replacement),self.assertRaises(ValueError):
                validate_trace(dict(trace,**replacement),10,20,policy)


if __name__=='__main__':unittest.main()
