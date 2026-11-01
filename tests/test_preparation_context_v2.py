import copy
import unittest
from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.batch_profiles import BatchedProfileProvider
from gsdeploy.forecasting.profiles import ProfileProvider, preparation_context_key
from gsdeploy.forecasting.simulator import FullHorizonModel
from gsdeploy.runtime.batching import BatchConfiguration
from tests.test_batched_forecast_v2 import CONTEXT, METADATA, MS, make_provider, row


class PreparationContextTests(unittest.TestCase):
    def setup_model(self, with_evidence=True):
        metadata = copy.deepcopy(METADATA)
        metadata["text"]["a"]["input_ready_delay_ns"] = 4 * MS
        metadata["text"]["b"]["input_ready_delay_ns"] = 6 * MS
        aliases, provider, _ = make_provider({"owner": [(("a", "b"), 12.)]}, metadata)
        doc = copy.deepcopy(provider.provider.document)
        if with_evidence:
            doc["preparation_contexts"] = {}
            for pool in ("a", "b"):
                preparing = {"owner": dict(family="family-owner", task="text", pool_id=pool)}
                key = preparation_context_key({a: {"row": r} for a, r in preparing.items()}, {})
                doc["preparation_contexts"][key] = dict(active={}, preparing=preparing,
                    scenarios={"mean": dict(board_watts=15., peak_memory_mib=110.)}, evidence="cpu-preparation-context")
        p = BatchedProfileProvider(ProfileProvider(doc, CONTEXT, allow_cpu_fixture=True), metadata,
                                   execution_context_sha256=CONTEXT)
        return FullHorizonModel(aliases, p, batch_configurations={"owner": BatchConfiguration(2, 30, 20 * MS)})

    def rows(self):
        result = [row("a"), row("b", "b")]
        for r in result:
            r.pop("batch_ready_ns", None)
            r.pop("batch_ready_evidence", None)
        return {"owner": result}

    def test_preparation_and_generate_each_charge_board_once(self):
        result = self.setup_model().forecast(self.rows(), 0, 30 * MS, require_memory=True)
        self.assertEqual(result["finishes"]["a"]["finish_ns"], 22 * MS)
        self.assertAlmostEqual(result["board_energy_joules"], .47)
        self.assertEqual(result["peak_memory_mib"], 110.)
        self.assertEqual(len(result["batch_preparation_events"]), 4)

    def test_missing_preparation_evidence_never_uses_idle_context(self):
        with self.assertRaisesRegex(UnknownEvidence, "POWER_CONTEXT"):
            self.setup_model(False).forecast(self.rows(), 0, 30 * MS)
        result = self.setup_model(False).forecast(self.rows(), 0, 30 * MS, require_energy=False)
        self.assertIsNone(result["board_energy_joules"])
        self.assertIsNone(result["peak_memory_mib"])

    def test_bad_preparation_identity_is_rejected(self):
        m = self.setup_model()
        doc = copy.deepcopy(m.service_provider.provider.document)
        key = next(iter(doc["preparation_contexts"]))
        doc["preparation_contexts"][key]["preparing"]["owner"]["pool_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "identity"):
            ProfileProvider(doc, CONTEXT, allow_cpu_fixture=True)


if __name__ == "__main__": unittest.main()
