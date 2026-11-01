"""CPU-only tests for prospective regularized model validation contracts."""
import json
from pathlib import Path
import tempfile
import time
import unittest

from tools.predictor_validation_20261027.probe_validation import receipt
from tools.predictor_validation_20261027.profiling import PLAN_SCHEMA, digest
from tools.predictor_validation_20261027.statistical_profiles import SCHEMA, full_count, phase_count
from tools.predictor_validation_20261027.statistical_validation import seal, report


class StatisticalValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.training_dir = self.root / "training"
        self.training_dir.mkdir()
        self.independent = self.root / "independent"
        self.independent.mkdir()
        self.training_path = self.training_dir / "raw.json"
        self.training = {"schema": "gsdeploy.native_phase_collection.v1", "status": "completed",
                         "execution_context_sha256": "exec", "request_records": [{"profile_job_id": "train-job", "split": "training"}]}
        self.write(self.training_path, self.training)
        self.target = ["text", "text", "known"]
        fk, fm = full_count([{"family": "text"}])
        pk, pm = phase_count([({"family": "text"}, {"phase": "native"})])
        self.model = {"schema": SCHEMA, "execution_context_sha256": "exec",
                      "source_receipts": [receipt(self.training_path)],
                      "profiles": [{"id": "p", "family": "text", "task": "text", "pool_id": "known",
                                    "execution": {"batch_size": 1, "weight_reuse": False, "backend": "statistical"},
                                    "segments": [{"id": "native", "kind": "mixed", "duration_ms": 100,
                                                  "demand": dict(sm=None, dram=None, l2=None)}]}],
                      "handler_models": [{"target": self.target, "progress": {fk: 1}}],
                      "semantic_models": [{"target": self.target + ["native"], "progress": {pk: 1}}],
                      "family_contexts": {fk: fm}, "phase_contexts": {pk: pm}}
        self.model_path = self.root / "model.json"
        self.freeze_model()
        self.plan = {"schema": PLAN_SCHEMA, "execution_context_sha256": "exec",
                     "inputs": [{"task": "text", "pool_id": "known", "input_size": 100}],
                     "jobs": [{"id": "validation-job", "context": "S0", "split": "validation",
                               "inputs": {"text#0": "known"}, "start_offsets_ms": {"text#0": 0},
                               "validation_view": "independent_trial"}]}
        self.plan_path = self.independent / "plan.json"
        self.write(self.plan_path, self.plan)

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def freeze_model(self):
        self.model.pop("model_sha256", None)
        self.model["model_sha256"] = digest(self.model)
        self.write(self.model_path, self.model)

    def sealed(self):
        return seal(self.model_path, self.training_path, self.plan_path, self.root / "sealed")

    def raw(self, sealing):
        origin = max(time.perf_counter_ns(), sealing["created_ns"] + 1000)
        rows, batches = [], []
        for i, job in enumerate(self.plan["jobs"]):
            now = origin + i * 1_000_000_000
            rid = "native:" + str(i)
            rows.append({"request_id": rid, "model": "text#0", "task": "text", "pool_id": job["inputs"]["text#0"],
                         "profile_job_id": job["id"], "split": "validation", "status": "completed",
                         "scheduled_time_ns": now, "observed_worker_start_time_ns": now + 1_000_000,
                         "worker_response": {"start_perf_ns": now + 1_000_000, "end_perf_ns": now + 101_000_000}})
            batches.append({"job_id": job["id"], "start_ns": now, "request_ids": [rid]})
        raw = {"schema": "gsdeploy.native_phase_collection.v1", "kind": "measured", "backend": "native",
               "status": "completed", "performance_scope": "unprofiled_with_phase_markers",
               "execution_context_sha256": "exec", "observation_start_ns": origin,
               "plan": receipt(self.plan_path), "request_records": rows, "batches": batches,
               "errors": [], "runtime_errors": [], "telemetry_errors": []}
        path = self.independent / "raw.json"
        self.write(path, raw)
        return path

    def test_known_input_gate_one_run_and_regularization_semantics(self):
        sealing = self.sealed()
        prediction = json.loads(Path(sealing["prediction"]["path"]).read_text())
        row = prediction["requests"][0]
        self.assertEqual(row["service_ms"], 100)
        self.assertFalse(row["physically_identified"])
        self.assertTrue(row["regularization_dependent"])
        assessed = report(self.root / "sealed", self.raw(sealing), self.root / "report")
        self.assertEqual(assessed["validation_case_count"], 1)
        self.assertEqual(assessed["known_input_gate"]["status"], "PASS")
        self.assertEqual(assessed["known_input_gate"]["coverage"], 1)
        self.assertEqual(assessed["pending_categories"]["delta_U"], "NOT_RUN")

    def test_unseen_keeps_full_domain_incomplete_without_poisoning_known_subdomain(self):
        self.plan["inputs"].append({"task": "text", "pool_id": "unseen", "input_size": 200})
        job = dict(self.plan["jobs"][0], id="unseen-job", inputs={"text#0": "unseen"}, validation_view="unseen_input")
        self.plan["jobs"].append(job)
        self.write(self.plan_path, self.plan)
        sealing = self.sealed()
        assessed = report(self.root / "sealed", self.raw(sealing), self.root / "report")
        self.assertEqual(assessed["status"], "INCOMPLETE")
        self.assertEqual(assessed["point_coverage"]["service"], {"requested": 2, "known": 1, "physical_online": 1})
        self.assertEqual(assessed["known_input_gate"]["status"], "PASS")
        self.assertEqual(assessed["unseen_input_diagnostics"]["known"], 0)

    def test_unobserved_phase_context_declares_interpolation(self):
        self.model["semantic_models"][0]["progress"] = {}
        self.freeze_model()
        sealing = self.sealed()
        row = json.loads(Path(sealing["prediction"]["path"]).read_text())["requests"][0]
        self.assertTrue(row["interpolated"])
        assessed = report(self.root / "sealed", self.raw(sealing), self.root / "report")
        self.assertEqual(assessed["statistical_prediction_semantics"]["interpolated_known_requests"], 1)

    def test_seal_after_observation_refused(self):
        self.write(self.independent / "raw.json", {})
        with self.assertRaisesRegex(ValueError, "before independent collection"):
            self.sealed()

    def test_training_validation_mix_refused(self):
        self.training["request_records"].append({"profile_job_id": "old-val", "split": "validation"})
        self.write(self.training_path, self.training)
        with self.assertRaisesRegex(ValueError, "Training-only"):
            self.sealed()

    def test_source_receipt_mutation_refused(self):
        self.training["note"] = "changed"
        self.write(self.training_path, self.training)
        with self.assertRaisesRegex(ValueError, "source receipt"):
            self.sealed()

    def test_unseen_full_family_context_does_not_extrapolate(self):
        self.model["handler_models"][0]["progress"] = {}
        self.freeze_model()
        sealing = self.sealed()
        assessed = report(self.root / "sealed", self.raw(sealing), self.root / "report")
        self.assertEqual(assessed["known_input_gate"]["status"], "INCOMPLETE")
        self.assertEqual(assessed["point_coverage"]["service"]["known"], 0)


if __name__ == "__main__":
    unittest.main()
