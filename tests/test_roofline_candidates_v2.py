import unittest
from gsdeploy.forecasting.roofline import classify_operator_work, complementary_operator_candidates


class RooflineCandidatesTests(unittest.TestCase):
    def setUp(self):
        self.hardware = dict(hardware_id="fixture-gpu", precision_signature="fp16-matched", peak_evidence="cpu-fixture",
                             peak_flops_per_second=1000., peak_dram_bytes_per_second=100.)

    def config(self, key, flops, traffic):
        return dict(configuration_sha256=key, hardware_id="fixture-gpu", precision_signature="fp16-matched",
                    operation_evidence="cpu-operator-fixture", flops=flops, declared_traffic_bytes=traffic,
                    traffic_assumption="Declared one-pass tensor traffic; no measured DRAM/cache claim")

    def test_batch_arithmetic_intensity_changes_candidate_label(self):
        # Same resident weight, different batch changes work/traffic balance.
        self.assertEqual(classify_operator_work(self.config("batch1", 100, 100), self.hardware)["classification"], "bandwidth_proxy")
        result = classify_operator_work(self.config("batch64", 6400, 200), self.hardware)
        self.assertEqual(result["classification"], "compute_proxy")
        self.assertFalse(result["throughput_prediction_qualified"])

    def test_matched_precision_required_and_capacity_does_not_substitute_traffic(self):
        wrong = self.config("wrong", 100, 100)
        wrong["precision_signature"] = "int8-other-pipeline"
        self.assertEqual(classify_operator_work(wrong, self.hardware)["classification"], "unknown")
        wrong.pop("declared_traffic_bytes")
        wrong["resident_bytes"] = 100
        self.assertEqual(classify_operator_work(wrong, self.hardware)["classification"], "unknown")

    def test_complementarity_orders_all_pairs_without_pruning(self):
        rows = [self.config("a", 100, 100), self.config("b", 6400, 200), self.config("c", 1000, 100)]
        pairs = complementary_operator_candidates(rows, self.hardware)
        self.assertTrue(pairs[0]["complementary_proxy"])
        self.assertEqual(len(pairs), 3)
        self.assertFalse(any(p["pruning_allowed"] for p in pairs))


if __name__ == "__main__": unittest.main()
