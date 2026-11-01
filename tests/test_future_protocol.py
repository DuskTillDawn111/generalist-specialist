"""CPU preregistration checks; no launcher, CUDA import, or GPU subprocess."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027.future_protocol import (
    CONFIRM_SEEDS, DEV_SEEDS, candidate_domain, prepare, seed_audit,
)


class FutureProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("configs", "artifacts", "tools", "gsdeploy/runtime"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def write(self, path, value):
        target = self.root / path
        target.write_text(json.dumps(value))
        return target

    def inputs(self):
        self.write("configs/resource_spillover_objective_20261024.json", {
            "weights": {k: 1 / 6 for k in "QTLPME"}, "absolute_quality_gate": False})
        self.write("configs/resource_spillover_study_20261024.json", {
            "system_config": "../artifacts/system.json", "seeds": [501, 502],
            "workload_rps": {"text": 1.85, "image": .01, "ocr": .01, "asr": .01},
            "latency_reference_ms": {"text": 1000, "image": 250, "ocr": 10000, "asr": 3000},
            "background_routing": {"ocr": {"ocr": .01}, "image": {"generalist_2": .01}, "asr": {"generalist_2": .01}}})
        self.write("artifacts/system.json", {"catalog": "catalog.json", "gpu_memory_mib": 49140,
                                             "startup_amortization_seconds": 3600, "energy_reference_watts": 450})
        self.write("artifacts/catalog.json", {"safe_factor": 1.110854})
        (self.root / "gsdeploy/stability.py").write_text("frozen source")
        (self.root / "gsdeploy/runtime/runner.py").write_text("frozen runner")

    def test_seed_keys_not_latency_or_request_text(self):
        self.write("artifacts/run.json", {"service_ms": 627101, "request_id": "measurement:627201", "label": "501"})
        audit = seed_audit(self.root)
        self.assertEqual(audit["proposed_conflicts"], [])
        self.assertEqual(audit["records"], [])

    def test_multiline_seed_list_and_scalar_are_detected(self):
        (self.root / "configs/design.json").write_text('{"seeds": [\n 501,\n 502\n]}')
        self.write("artifacts/manifest.json", {"random_seed": DEV_SEEDS[0]})
        audit = seed_audit(self.root)
        self.assertEqual(audit["proposed_conflicts"], [DEV_SEEDS[0]])
        self.assertTrue(audit["historical_501_502"]["source_field_found"]["501"])
        self.assertFalse(audit["historical_501_502"]["reuse"])

    def test_phase_seed_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            seed_audit(self.root, development=[1], confirmation=[1])

    def test_finite_domain_has_fair_same_execution_points(self):
        domain = candidate_domain()
        self.assertEqual(len(domain), 21)
        self.assertEqual(len({c["id"] for c in domain}), 21)
        for allow in (False, True):
            choices = {c["resource"]["id"] for c in domain if c["routing"]["allow_spill"] == allow}
            self.assertEqual(choices, {"shared", "admission_max1", "admission_max2"})
        self.assertTrue(all(c["batch_size"] == 1 and not c["weight_reuse"] for c in domain))

    def test_prepare_never_creates_executable_jobs(self):
        self.inputs()
        report_path = self.write("artifacts/prediction-report.json", {"status": "INCOMPLETE"})
        protocol = prepare(self.root / "future", root=self.root, predictor_report=report_path)
        self.assertEqual(protocol["status"], "NOT_RUN")
        self.assertFalse(protocol["GPU_started"])
        self.assertEqual(protocol["executable_jobs"], [])
        self.assertEqual(protocol["gates"]["service_and_probe_P95"]["status"], "INCOMPLETE")
        self.assertEqual(protocol["gates"]["full_queue_and_full_window_P95"]["status"], "NOT_RUN")
        self.assertEqual({k: len(v) for k, v in protocol["group_candidate_ids"].items()},
                         {"original_no_spill": 4, "original_allow_spill": 3, "optimized_no_spill": 12, "optimized_allow_spill": 9})
        self.assertEqual(protocol["development"]["seeds"], list(DEV_SEEDS))
        self.assertEqual(protocol["confirmation"]["seeds"], list(CONFIRM_SEEDS))
        self.assertEqual(protocol["resource_and_stability"]["catalog_safe_factor"], 1.110854)
        descriptors = json.loads((self.root / "future/four-group-descriptors.json").read_text())
        self.assertEqual(len(descriptors["descriptors"]), 32)
        self.assertTrue(all(d["status"] == "NOT_RUN" and d["U"] is None for d in descriptors["descriptors"]))

    def test_reservation_conflict_prevents_new_protocol(self):
        self.inputs()
        self.write("artifacts/reservation.json", {"development_seeds": list(DEV_SEEDS)})
        with self.assertRaisesRegex(ValueError, "already used/reserved"):
            prepare(self.root / "future", root=self.root)
        self.assertFalse((self.root / "future").exists())

    def test_existing_protocol_never_overwritten(self):
        self.inputs()
        (self.root / "future").mkdir()
        with self.assertRaises(FileExistsError):
            prepare(self.root / "future", root=self.root)


if __name__ == "__main__":
    unittest.main()
