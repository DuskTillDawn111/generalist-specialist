import threading
import unittest

from gsdeploy.profiling.report import edit_distance, qa_score, quality
from gsdeploy.profiling.telemetry import BoardMonitor


class ProfilingEvidenceTests(unittest.TestCase):
    def test_native_t5_no_answer_marker_and_failed_request(self):
        self.assertEqual(qa_score("no answer>", []), (1.0, 1.0))
        self.assertEqual(qa_score("", [], success=False), (0.0, 0.0))
        self.assertEqual(qa_score("the Pacific Ocean", ["Pacific Ocean"]), (1.0, 1.0))

    def test_repeated_load_requests_do_not_inflate_quality_sample_count(self):
        good = {"sample_id": "a", "prediction": "en", "reference": "en", "success": True}
        bad = {"sample_id": "b", "prediction": "en", "reference": "fr", "success": True}
        result = quality("lid", [good] * 100 + [bad])
        self.assertEqual(result["unique_samples"], 2)
        self.assertEqual(result["macro_accuracy"], .5)

    def test_gpu_counter_does_not_replace_missing_board_power_coverage(self):
        monitor = BoardMonitor.__new__(BoardMonitor)
        monitor.lock = threading.Lock()
        monitor.errors = []
        monitor.samples = [
            {"time_ns": 0, "watts": 100, "counter_joules": 1000},
            {"time_ns": 10_000_000_000, "watts": 100, "counter_joules": 2000},
        ]
        result = monitor.window(*monitor.samples)
        self.assertIsNone(result["power_integral_joules"])
        self.assertEqual(result["counter_joules"], 1000)
        self.assertIsNone(result["joules"])
        self.assertFalse(result["coverage_valid"])

    def test_character_and_word_error_numerators(self):
        self.assertEqual(edit_distance("kitten", "sitting"), 3)
        self.assertEqual(edit_distance(["a", "cat"], ["the", "cat", "sat"]), 2)


if __name__ == "__main__":
    unittest.main()
