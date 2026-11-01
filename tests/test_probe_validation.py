"""CPU contract fixtures; these fabricated observations are never GPU evidence."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

from gsdeploy.forecasting.profiles import PROFILE_SCHEMA, context_key
from tools.predictor_validation_20261027.profiling import PLAN_SCHEMA
from tools.predictor_validation_20261027.probe_validation import receipt, report, seal


class ProbeValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.training_dir = self.root / "training"
        self.training_dir.mkdir()
        self.independent_dir = self.root / "independent"
        self.independent_dir.mkdir()
        self.training = {"schema": "gsdeploy.native_phase_collection.v1", "status": "completed",
                         "execution_context_sha256": "exec", "request_records": [{"profile_job_id": "train-job", "split": "training"}]}
        self.training_path = self.training_dir / "raw.json"
        self.write(self.training_path, self.training)
        members = {"text#0": {"profile_id": "text-profile", "segment_id": "native"}}
        self.document = {"schema": PROFILE_SCHEMA, "execution_context_sha256": "exec",
                         "provenance": {"kind": "measured", "split": "training", "sources": ["training"]},
                         "profiles": [{"id": "text-profile", "family": "text", "task": "text", "pool_id": "known",
                                       "execution": {"batch_size": 1, "backend": "native", "weight_reuse": False},
                                       "segments": [{"id": "native", "kind": "gpu", "duration_ms": {"mean": 100},
                                                     "demand": dict(sm=.2, dram=.8, l2=.5)}]}],
                         "contexts": {context_key(members): {"members": members, "evidence": "training",
                                       "scenarios": {"mean": {"progress": {"text#0": 1}, "board_watts": None,
                                                               "peak_memory_mib": None}}}}}
        self.model_path = self.root / "fit.json"
        self.write(self.model_path, {"fitted_document": self.document})
        self.plan = {"schema": PLAN_SCHEMA, "execution_context_sha256": "exec",
                     "inputs": [{"task": "text", "pool_id": "known", "input_size": 100}],
                     "jobs": [{"id": "validation-job", "context": "S0", "split": "validation", "repeat": 2,
                               "inputs": {"text#0": "known"}, "start_offsets_ms": {"text#0": 0},
                               "validation_view": "independent_trial"}]}
        self.plan_path = self.independent_dir / "plan.json"
        self.write(self.plan_path, self.plan)

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def sealed(self):
        return seal(self.model_path, self.training_path, self.plan_path, self.root / "sealed")

    def raw(self, sealing):
        start = max(time.perf_counter_ns(), sealing["created_ns"] + 1000)
        rows, batches = [], []
        for index, job in enumerate(self.plan["jobs"]):
            base = start + index * 1_000_000_000
            rid = "profile:" + str(index)
            rows.append({"request_id": rid, "model": "text#0", "task": "text", "pool_id": job["inputs"]["text#0"],
                         "profile_job_id": job["id"], "split": "validation", "status": "completed",
                         "scheduled_time_ns": base, "observed_worker_start_time_ns": base + 1_000_000,
                         "worker_response": {"start_perf_ns": base + 1_000_000, "end_perf_ns": base + 101_000_000}})
            batches.append({"job_id": job["id"], "start_ns": base, "request_ids": [rid]})
        raw = {"schema": "gsdeploy.native_phase_collection.v1", "kind": "measured", "status": "completed",
               "backend": "native", "performance_scope": "unprofiled_with_phase_markers",
               "execution_context_sha256": "exec", "observation_start_ns": start, "plan": receipt(self.plan_path),
               "request_records": rows, "batches": batches, "errors": [], "runtime_errors": [], "telemetry_errors": []}
        path = self.independent_dir / "raw.json"
        self.write(path, raw)
        return path

    def test_seal_uses_simulator_and_preserves_training_reference(self):
        sealing = self.sealed()
        prediction = json.loads(Path(sealing["prediction"]["path"]).read_text())
        row = prediction["requests"][0]
        self.assertEqual(row["service_ms"], 100)
        self.assertEqual(row["solo_ms"], 100)
        self.assertEqual(row["e2e_ms"], 100)
        self.assertFalse(prediction["actual_future_trace_used"])
        self.assertIn("not independent validation solo", prediction["slowdown_reference"])

    def test_report_single_run_one_case_and_pending_gates(self):
        sealing = self.sealed()
        assessed = report(self.root / "sealed", self.raw(sealing), self.root / "assessed")
        self.assertEqual(assessed["validation_case_count"], 1)
        self.assertEqual(assessed["status"], "PASS")
        self.assertEqual(assessed["pending_categories"]["full_queue"], "NOT_RUN")
        self.assertEqual(assessed["pending_categories"]["delta_U"], "NOT_RUN")
        self.assertEqual(assessed["physical_online_metrics"]["service"]["mae"], 0)
        self.assertEqual(assessed["physical_online_metrics"]["wait"]["mae"], 1)
        self.assertEqual(assessed["probe_groups"]["context:S0"]["known"], 1)

    def test_unseen_input_retains_unknown_and_blocks_tail_pass(self):
        self.plan["inputs"].append({"task": "text", "pool_id": "unseen", "input_size": 200})
        job = copy.deepcopy(self.plan["jobs"][0])
        job.update(id="unseen-job", inputs={"text#0": "unseen"}, validation_view="unseen_input")
        self.plan["jobs"].append(job)
        self.write(self.plan_path, self.plan)
        sealing = self.sealed()
        assessed = report(self.root / "sealed", self.raw(sealing), self.root / "assessed")
        self.assertEqual(assessed["status"], "INCOMPLETE")
        self.assertEqual(assessed["point_coverage"]["service"]["requested"], 2)
        self.assertEqual(assessed["point_coverage"]["service"]["known"], 1)
        self.assertEqual(assessed["point_coverage"]["p95"]["known"], 0)
        self.assertEqual(assessed["probe_groups"]["view:unseen_input"]["known"], 0)

    def test_empty_training_model_keeps_all_requests_unknown(self):
        self.write(self.model_path, {"known": False, "reason": "rank deficient"})
        sealing = self.sealed()
        assessed = report(self.root / "sealed", self.raw(sealing), self.root / "assessed")
        self.assertEqual(assessed["status"], "INCOMPLETE")
        self.assertEqual(assessed["probe_groups"]["all"]["known"], 0)

    def test_nonvalidation_plan_rejected(self):
        self.plan["jobs"][0]["split"] = "training"
        self.write(self.plan_path, self.plan)
        with self.assertRaisesRegex(ValueError, "validation-only"):
            self.sealed()

    def test_training_job_overlap_rejected(self):
        self.plan["jobs"][0]["id"] = "train-job"
        self.write(self.plan_path, self.plan)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            self.sealed()

    def test_early_actual_observation_rejected(self):
        sealing = self.sealed()
        path = self.raw(sealing)
        raw = json.loads(path.read_text())
        raw["observation_start_ns"] = sealing["created_ns"]
        self.write(path, raw)
        with self.assertRaisesRegex(ValueError, "sealed before"):
            report(self.root / "sealed", path, self.root / "assessed")

    def test_tampered_model_cannot_change_after_seal(self):
        sealing = self.sealed()
        path = self.raw(sealing)
        self.write(self.model_path, {})
        with self.assertRaisesRegex(ValueError, "Changed sealed model_source"):
            report(self.root / "sealed", path, self.root / "assessed")

    def test_report_rejects_omitted_job(self):
        sealing = self.sealed()
        path = self.raw(sealing)
        raw = json.loads(path.read_text())
        raw["batches"] = []
        self.write(path, raw)
        with self.assertRaisesRegex(ValueError, "missing, duplicate"):
            report(self.root / "sealed", path, self.root / "assessed")

    def test_seal_no_overwrite(self):
        self.sealed()
        with self.assertRaises(FileExistsError):
            self.sealed()


if __name__ == "__main__":
    unittest.main()
