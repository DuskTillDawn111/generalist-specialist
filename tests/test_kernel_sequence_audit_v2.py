import unittest

from tools.predictor_validation_20261027.kernel_sequence_audit_v2 import analyze_sequence, interval_statistics


def kernel(start, end, stream=7, correlation=1):
    return dict(nsys_start_ns=start, nsys_end_ns=end, pid=42, device=0,
                context_id=1, stream=stream, correlation_id=correlation)


class KernelSequenceAuditTests(unittest.TestCase):
    def test_union_does_not_sum_overlaps_or_count_touching_as_overlap(self):
        result = interval_statistics([(0, 10), (5, 15), (15, 20), (7, 9)])
        self.assertEqual(result['duration_sum_ns'], 27)
        self.assertEqual(result['union_busy_ns'], 20)
        self.assertEqual(result['overlap_excess_ns'], 7)
        self.assertEqual(result['simultaneous_overlap_wall_ns'], 5)
        self.assertEqual(result['maximum_simultaneous_kernels'], 3)

    def test_serial_stream_and_unclassified_gaps(self):
        result = analyze_sequence([kernel(20, 30, correlation=2), kernel(0, 10)])
        self.assertTrue(result['single_serial_stream_expressible'])
        self.assertEqual(result['inter_kernel_unclassified_gaps']['sum_ns'], 10)
        self.assertEqual(result['stream_dag']['proven_fifo_edge_count'], 1)
        self.assertFalse(result['natural_timing_validated'])

    def test_multistream_even_nonoverlapping_is_not_flattened(self):
        for start in (5, 15):
            result = analyze_sequence([kernel(0, 10), kernel(start, start+10, stream=13)])
            self.assertFalse(result['single_serial_stream_expressible'])
            self.assertEqual(result['stream_dag']['proven_fifo_edge_count'], 0)

    def test_same_stream_overlap_and_inverted_runtime_launch_rejected(self):
        self.assertFalse(analyze_sequence([kernel(0, 10), kernel(5, 15, correlation=2)])['single_serial_stream_expressible'])
        result = analyze_sequence([kernel(0, 10), kernel(20, 30, correlation=2)],
            runtime_launches={(42, 1): dict(start=-5), (42, 2): dict(start=-10)})
        self.assertEqual(result['launch_order_violations'], 1)
        self.assertFalse(result['single_serial_stream_expressible'])

    def test_async_host_anchor_is_not_added_as_cpu_duration(self):
        row = dict(worker_response=dict(start_perf_ns=90, end_perf_ns=140,
            observed_stages=[dict(id='inference', kind='gpu', start_ns=95, end_ns=135)]))
        result = analyze_sequence([kernel(0, 10)], row, clock_mapping=dict(valid=True, offset_ns=100))
        boundaries = result['host_boundaries']
        self.assertEqual(boundaries['handler_wall_ns'], 50)
        self.assertEqual(boundaries['prefix_unclassified_ns'], 10)
        self.assertEqual(boundaries['suffix_unclassified_ns'], 30)
        self.assertNotIn('cpu_duration_ns', result)
        self.assertEqual(boundaries['anchors'][0]['end_ns'], 135)

    def test_missing_mapping_and_launch_evidence_stay_unknown(self):
        row = dict(worker_response=dict(start_perf_ns=0, end_perf_ns=20))
        result = analyze_sequence([kernel(1, 10)], row)
        self.assertEqual(result['runtime_launch_count'], 0)
        self.assertIsNone(result['host_boundaries']['prefix_unclassified_ns'])
        self.assertFalse(result['training_observations_created'])

    def test_nonpositive_intervals_rejected(self):
        with self.assertRaises(ValueError): interval_statistics([(3, 3)])


if __name__ == '__main__': unittest.main()
