"""Metadata compatibility must never rewrite or refit sealed predictions."""
import json
from pathlib import Path
import unittest

from tests.test_probe_validation import ProbeValidationTests
from tools.predictor_validation_20261027.probe_validation import receipt
from tools.predictor_validation_20261027.probe_report_compat import report


class ProbeReportCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.fixture = ProbeValidationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.sealing = self.fixture.sealed()
        self.prediction_path = Path(self.sealing["prediction"]["path"])
        prediction = json.loads(self.prediction_path.read_text())
        for row in prediction["requests"]:
            row.pop("input_known_to_model")
            row.pop("timing_pattern")
        self.fixture.write(self.prediction_path, prediction)
        self.sealing["prediction"] = receipt(self.prediction_path)
        self.sealing.pop("sealer_source", None)
        self.fixture.write(self.fixture.root / "sealed" / "sealing.json", self.sealing)

    def test_old_metadata_reconstructed_without_changing_sealed_bytes(self):
        original = self.prediction_path.read_bytes()
        assessed = report(self.fixture.root / "sealed", self.fixture.raw(self.sealing), self.fixture.root / "report")
        self.assertEqual(self.prediction_path.read_bytes(), original)
        self.assertEqual(assessed["status"], "PASS")
        self.assertEqual(assessed["probe_groups"]["input_known_to_model:True"]["requested"], 1)
        compat = assessed["metadata_compatibility"]
        self.assertEqual(compat["reconstructed_group_fields"], {"input_known_to_model": 1, "timing_pattern": 1})
        self.assertFalse(compat["forecast_values_changed"])
        self.assertFalse(compat["seal_has_original_sealer_source_receipt"])
        self.assertTrue((self.fixture.root / "report" / "compatibility-receipt.json").is_file())
        self.assertEqual(assessed["validation_case_count"], 1)

    def test_all_unknown_remain_in_denominator(self):
        prediction = json.loads(self.prediction_path.read_text())
        for row in prediction["requests"]:
            for key in ("service_ms", "wait_ms", "e2e_ms"):
                row[key] = None
            row["unknown_reason"] = "UNKNOWN_FROZEN_CONTEXT"
        prediction["p95_by_task"] = {"text": None}
        self.fixture.write(self.prediction_path, prediction)
        self.sealing["prediction"] = receipt(self.prediction_path)
        self.fixture.write(self.fixture.root / "sealed" / "sealing.json", self.sealing)
        assessed = report(self.fixture.root / "sealed", self.fixture.raw(self.sealing), self.fixture.root / "report")
        self.assertEqual(assessed["status"], "INCOMPLETE")
        self.assertEqual(assessed["point_coverage"]["service"], {"requested": 1, "known": 0, "physical_online": 0})
        self.assertEqual(assessed["unknown_reason_counts"], {"UNKNOWN_FROZEN_CONTEXT": 1})
        self.assertEqual(assessed["pending_categories"]["delta_U"], "NOT_RUN")

    def test_failed_output_history_is_preserved_and_receipted(self):
        failed = self.fixture.root / "report-v1"
        failed.mkdir()
        failure = failed / "failure.txt"
        failure.write_text("KeyError: input_known_to_model")
        assessed = report(self.fixture.root / "sealed", self.fixture.raw(self.sealing), self.fixture.root / "report-v2", failed_output=failed)
        self.assertTrue(failure.is_file())
        history = assessed["metadata_compatibility"]["historical_failed_report"]
        self.assertTrue(history["exists"])
        self.assertEqual(history["files"], [receipt(failure)])


if __name__ == "__main__":
    unittest.main()
