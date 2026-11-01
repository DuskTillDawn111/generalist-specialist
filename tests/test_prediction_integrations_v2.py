"""Independent CLI/roofline/preparation evidence-contract checks, CPU only."""
import copy
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.__main__ import forecast_request
from gsdeploy.forecasting.recovery_risk import fit_recovery_risk
from gsdeploy.forecasting.roofline import classify_operator_work, complementary_operator_candidates
from gsdeploy.forecasting.simulator import FullHorizonModel
from gsdeploy.runtime.batching import BatchConfiguration
from tests.test_initial_handoff_forecast_v2 import cli_request
from tests.test_recovery_risk_v2 import training, features, HORIZON
from tests.test_batched_forecast_v2 import make_provider, METADATA, row, MS


class PredictionIntegrationTests(unittest.TestCase):
    def test_optional_cli_risk_is_independent_of_forecast_and_all_six_metrics(self):
        request = cli_request()
        request["admission_state"] = {"active": {"old": "s"}, "waiting": [("g", "next")]}
        baseline = forecast_request(request, allow_cpu_fixture=True)
        rows = training()
        for record in rows:
            record["execution_context_sha256"] = "fixture"
            record["features"]["execution_context_sha256"] = "fixture"
        frozen = fit_recovery_risk(rows, "fixture", horizon_ns=HORIZON, recovery_threshold_ms=50.)
        current = features(2., now=105 * MS)
        current["execution_context_sha256"] = "fixture"
        request["risk_request"] = {"model": frozen, "features": current}
        with_risk = forecast_request(request, allow_cpu_fixture=True)
        self.assertEqual(with_risk["metrics"], baseline["metrics"])
        self.assertEqual(with_risk["forecast"], baseline["forecast"])
        self.assertTrue(with_risk["recovery_risk"]["known"])
        self.assertFalse(with_risk["recovery_risk"]["calibrated_probability"])
        current["queue_work_ms"] = 1000.
        unsupported = forecast_request(request, allow_cpu_fixture=True)
        self.assertEqual(unsupported["recovery_risk"]["status"], "UNKNOWN")
        self.assertEqual(unsupported["metrics"], baseline["metrics"])

    def test_serial_preparation_needs_separate_power_and_memory_evidence(self):
        metadata = copy.deepcopy(METADATA)
        metadata["text"]["a"]["input_ready_delay_ns"] = 4 * MS
        aliases, provider, _ = make_provider({"owner": [(("a",), 3.)]}, metadata)
        pending = row("a")
        pending.pop("batch_ready_ns")
        pending.pop("batch_ready_evidence")
        model = FullHorizonModel(aliases, provider,
            batch_configurations={"owner": BatchConfiguration(1, 30, 0)})
        with self.assertRaisesRegex(UnknownEvidence, "UNKNOWN_WHOLE_BOARD_POWER_CONTEXT"):
            model.forecast({"owner": [pending]}, 0, 20 * MS)
        with self.assertRaisesRegex(UnknownEvidence, "UNKNOWN_WHOLE_BOARD_PEAK_MEMORY_CONTEXT"):
            model.forecast({"owner": [pending]}, 0, 20 * MS, require_energy=False, require_memory=True)
        result = model.forecast({"owner": [pending]}, 0, 20 * MS, require_energy=False)
        self.assertIsNone(result["board_energy_joules"])
        self.assertIsNone(result["peak_memory_mib"])
        self.assertEqual(result["finishes"]["a"]["finish_ns"], 7 * MS)

    def test_new_input_exact_batch_config_cannot_reuse_solo_profile(self):
        aliases, provider, _ = make_provider({"owner": [(("a",), 3.)]})
        # A measured solo pool c exists, but no batch-executor pool c profile.
        with self.assertRaisesRegex(UnknownEvidence, "UNKNOWN_INPUT_EXECUTION_PROFILE"):
            FullHorizonModel(aliases, provider,
                batch_configurations={"owner": BatchConfiguration(1, 30, 0)}).forecast(
                    {"owner": [row("new", "c")]}, 0, 20 * MS)

    def test_roofline_changes_by_work_and_precision_but_never_prunes_pairs(self):
        hardware = {"hardware_id": "cpu", "precision_signature": "fp16", "peak_evidence": "cpu-fixture",
                    "peak_flops_per_second": 1000., "peak_dram_bytes_per_second": 100.}
        base = {"configuration_sha256": "batch1", "hardware_id": "cpu", "precision_signature": "fp16",
                "operation_evidence": "cpu-work", "flops": 100., "declared_traffic_bytes": 100.,
                "traffic_assumption": "one-pass tensor traffic; cache unknown"}
        larger = {**base, "configuration_sha256": "batch64", "flops": 6400., "declared_traffic_bytes": 200.}
        self.assertEqual(classify_operator_work(base, hardware)["classification"], "bandwidth_proxy")
        classified = classify_operator_work(larger, hardware)
        self.assertEqual(classified["classification"], "compute_proxy")
        self.assertFalse(classified["pruning_allowed"])
        self.assertFalse(classified["throughput_prediction_qualified"])
        self.assertEqual(classify_operator_work({**base, "precision_signature": "int8"}, hardware)["classification"], "unknown")
        capacity = {**base, "resident_bytes": 100.}
        del capacity["declared_traffic_bytes"]
        self.assertEqual(classify_operator_work(capacity, hardware)["classification"], "unknown")
        pairs = complementary_operator_candidates([base, larger, {**base, "configuration_sha256": "third"}], hardware)
        self.assertEqual(len(pairs), 3)
        self.assertFalse(any(p["pruning_allowed"] for p in pairs))


if __name__ == "__main__":
    unittest.main()
