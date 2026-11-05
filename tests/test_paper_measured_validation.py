"""Prevent a diagnostic from silently dropping or inventing native evidence."""

import copy
import unittest

from experiments.validate_paper_measured import match_observations, source_coverage


class MeasuredCohortTests(unittest.TestCase):
    def setUp(self):
        self.expected = [dict(arrival_index=0, task="text", pool_id="input-a",
            alias="g#0", arrival_seconds=.1)]
        self.raw = dict(measurement_window=dict(offered_requests=1), request_records=[
            dict(arrival_index=0, task="text", pool_id="input-a", model="g#0",
                scheduled_offset_seconds=.1, request_id="measurement:0", status="completed",
                worker_response=dict(success=True, start_ns=100, end_ns=200),
                worker_latency_ms=1., worker_queue_wait_ms=2., e2e_latency_ms=3.1,
                schedule_delay_ms=.1, completion_time_ns=300)])

    def test_missing_and_duplicate_requests_are_rejected(self):
        for records in ([], self.raw["request_records"] * 2):
            with self.subTest(size=len(records)):
                raw = {**self.raw, "request_records": records}
                with self.assertRaisesRegex(ValueError, "cohort"):
                    match_observations(self.expected, raw)

    def test_changed_input_route_and_trace_are_rejected(self):
        for field, value in (("pool_id", "other"), ("model", "s#0"),
                             ("scheduled_offset_seconds", .2), ("arrival_index", 1)):
            raw = copy.deepcopy(self.raw)
            raw["request_records"][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "identity"):
                match_observations(self.expected, raw)

    def test_failed_outcome_is_not_dropped(self):
        self.raw["request_records"][0]["worker_response"]["success"] = False
        with self.assertRaisesRegex(ValueError, "failed"):
            match_observations(self.expected, self.raw)

    def test_receipt_and_handler_are_not_ready_or_segment_evidence(self):
        coverage = source_coverage(self.raw)
        self.assertEqual(coverage["parent_receipt"], 1)
        self.assertEqual(coverage["handler_start_end"], 1)
        self.assertEqual(coverage["explicit_worker_ready"], 0)
        self.assertEqual(coverage["ordered_segments"], 0)
        self.assertFalse(coverage["ready_inferred_from_receipt"])
        rows = match_observations(self.expected, self.raw)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["observed_e2e_ms"], 3.1)


if __name__ == "__main__":
    unittest.main()
