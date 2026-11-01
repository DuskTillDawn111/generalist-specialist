"""CPU fixtures verify validation contracts; they are not GPU evidence."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027.validation import (
    OBSERVATION_SCHEMA, PREDICTION_SCHEMA, SCHEMA, assess, main,
)


class PredictorValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = {"model": "frozen-test"}
        self.protocol = {
            "weights": {key: 1 / 6 for key in "QTLPME"},
            "utility_tie_tolerance": 1e-8,
            "required_categories": ["service", "slowdown", "queue", "p95", "delta_U", "ranking"],
            "planned_cases": [{"id": "case-1", "categories": ["service", "slowdown", "queue", "p95", "delta_U", "ranking"],
                               "expected_tasks": ["text"], "expected_candidate_ids": ["no-spill", "spill"]}],
            "acceptance": {"three_way_sign_accuracy": {"operator": ">=", "value": .9},
                           "delta_U_mae": {"operator": "<=", "value": .1}},
        }
        self.manifest = {
            "schema": SCHEMA,
            "model": {"artifact": self.write("model.json", self.model), "frozen_ns": 10,
                      "training_sources": [{"run_id": "train-1", "artifact": self.write("train.json", {"raw": "training"})}]},
            "protocol": {"artifact": self.write("protocol.json", self.protocol), "frozen_ns": 11},
            "cases": [],
        }
        terms = lambda x: {key: x for key in "QTLPME"}
        self.prediction = {
            "schema": PREDICTION_SCHEMA,
            "model_sha256": self.manifest["model"]["artifact"]["sha256"],
            "protocol_sha256": self.manifest["protocol"]["artifact"]["sha256"],
            "execution_context_sha256": "execution-1", "offered_cohort_sha256": "cohort-1",
            "created_ns": 21, "origin_ns": 20, "information_cutoff_ns": 20,
            "mode": "online_distribution", "actual_future_trace_used": False,
            "requests": [{"request_id": "r1", "task": "text", "pool_id": "input-1",
                          "service_ms": 200, "wait_ms": 100, "e2e_ms": 300, "solo_ms": 100},
                         {"request_id": "r2", "task": "text", "pool_id": "input-2",
                          "service_ms": 300, "wait_ms": 200, "e2e_ms": 500, "solo_ms": 100}],
            "p95_by_task": {"text": 490},
            "queue_samples": [{"offset_ns": 0, "depth": 2}, {"offset_ns": 100, "depth": 1}],
            "baseline_candidate_id": "no-spill",
            "candidates": [{"candidate_id": "no-spill", "terms": terms(.5)},
                           {"candidate_id": "spill", "terms": terms(.4)}],
        }
        self.observation = {
            "schema": OBSERVATION_SCHEMA, "evidence_kind": "cpu_fixture", "run_id": "validation-1",
            "execution_context_sha256": "execution-1", "offered_cohort_sha256": "cohort-1", "available_ns": 30,
            "requests": [{"request_id": "r1", "task": "text", "pool_id": "input-1",
                          "service_ms": 250, "wait_ms": 100, "e2e_ms": 350, "solo_ms": 100},
                         {"request_id": "r2", "task": "text", "pool_id": "input-2",
                          "service_ms": 350, "wait_ms": 200, "e2e_ms": 550, "solo_ms": 100}],
            "queue_samples": [{"offset_ns": 0, "depth": 3}, {"offset_ns": 100, "depth": 2}],
            "candidates": [{"candidate_id": "no-spill", "terms": terms(.5), "offered_cohort_sha256": "cohort-1"},
                           {"candidate_id": "spill", "terms": terms(.6), "offered_cohort_sha256": "cohort-1"}],
        }

    def write(self, name, value):
        raw = json.dumps(value, sort_keys=True).encode()
        (self.root / name).write_bytes(raw)
        return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}

    def seal(self):
        self.manifest["cases"] = [{"id": "case-1", "split": "validation", "run_id": "validation-1",
                                  "prediction": self.write("prediction.json", self.prediction),
                                  "observation": self.write("observation.json", self.observation)}]
        return self.manifest

    def refresh_protocol(self):
        self.manifest["protocol"]["artifact"] = self.write("protocol.json", self.protocol)
        self.prediction["protocol_sha256"] = self.manifest["protocol"]["artifact"]["sha256"]

    def report(self):
        return assess(self.seal(), artifact_root=self.root)

    def test_empty_report_is_not_run(self):
        report = assess(self.manifest, artifact_root=self.root)
        self.assertEqual(report["status"], "NOT_RUN")
        self.assertFalse(report["GPU_started"])

    def test_cpu_arithmetic_never_passes_physical_validation(self):
        report = self.report()
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(report["all_evidence_diagnostics"]["service"]["mae"], 50)
        self.assertAlmostEqual(report["all_evidence_diagnostics"]["slowdown"]["mae"], .5)
        self.assertEqual(report["all_evidence_diagnostics"]["queue"]["mae"], 1)
        self.assertAlmostEqual(report["all_evidence_diagnostics"]["p95"]["mae"], 50)
        self.assertEqual(report["physical_online_metrics"]["service"]["count"], 0)
        self.assertEqual(report["point_coverage"]["service"]["requested"], 2)
        self.assertFalse(report["physical_improvement_certified"])

    def test_measured_flag_exercises_sign_ranking_and_regret(self):
        # Synthetic bytes marked measured exercise the branch only; no artifact
        # or claim from this test is a physical accuracy result.
        self.observation["evidence_kind"] = "measured"
        report = self.report()
        self.assertEqual(report["status"], "FAIL")
        metrics = report["physical_online_decisions"]
        self.assertEqual(metrics["three_way_sign_accuracy"], 0)
        self.assertEqual(metrics["false_positive_gain_count"], 1)
        self.assertEqual(metrics["pairwise_rank_accuracy"], 0)
        self.assertAlmostEqual(metrics["mean_selection_regret"], .1)
        self.assertAlmostEqual(report["physical_online_metrics"]["delta_U"]["mae"], .2)

    def test_correct_ranking_and_delta_can_pass_frozen_gates(self):
        self.observation["evidence_kind"] = "measured"
        self.observation["candidates"][1]["terms"] = {key: .4 for key in "QTLPME"}
        self.assertEqual(self.report()["status"], "PASS")

    def test_unknown_predictions_remain_in_denominator(self):
        self.observation["evidence_kind"] = "measured"
        self.prediction["requests"].pop()
        report = self.report()
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(report["point_coverage"]["service"], {"requested": 2, "known": 1, "physical_online": 1})
        self.assertEqual(report["cases"][0]["unknown_requests"], ["r2"])

    def test_missing_candidate_does_not_inflate_rank_coverage(self):
        self.observation["evidence_kind"] = "measured"
        self.protocol["planned_cases"][0]["expected_candidate_ids"].append("third")
        self.refresh_protocol()
        self.observation["candidates"].append({"candidate_id": "third", "terms": {k: .3 for k in "QTLPME"},
                                                "offered_cohort_sha256": "cohort-1"})
        report = self.report()
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(report["point_coverage"]["delta_U"]["requested"], 2)
        self.assertEqual(report["point_coverage"]["delta_U"]["known"], 1)

    def test_train_run_overlap_rejected(self):
        manifest = self.seal()
        manifest["cases"][0]["run_id"] = "train-1"
        with self.assertRaisesRegex(ValueError, "overlaps training"):
            assess(manifest, artifact_root=self.root)

    def test_training_hash_overlap_rejected(self):
        manifest = self.seal()
        manifest["model"]["training_sources"].append({"run_id": "other-training", "artifact": manifest["cases"][0]["observation"]})
        with self.assertRaisesRegex(ValueError, "overlaps training evidence"):
            assess(manifest, artifact_root=self.root)

    def test_native_profile_training_sources_must_match(self):
        self.model.update(schema="gsdeploy.segment_resource_profiles.v1", provenance={"sources": ["hidden-train.json"]})
        self.manifest["model"]["artifact"] = self.write("model.json", self.model)
        with self.assertRaisesRegex(ValueError, "training provenance"):
            assess(self.manifest, artifact_root=self.root)

    def test_tampered_artifact_rejected(self):
        manifest = self.seal()
        (self.root / "prediction.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            assess(manifest, artifact_root=self.root)

    def test_prediction_sealed_after_observation_rejected(self):
        self.prediction["created_ns"] = 31
        with self.assertRaisesRegex(ValueError, "unsealed"):
            self.report()

    def test_future_information_rejected(self):
        self.prediction["information_cutoff_ns"] = 21
        with self.assertRaisesRegex(ValueError, "future information"):
            self.report()

    def test_actual_trace_only_allowed_as_diagnostic_replay(self):
        self.prediction["actual_future_trace_used"] = True
        with self.assertRaisesRegex(ValueError, "future leakage"):
            self.report()
        self.prediction["mode"] = "trace_replay"
        self.observation["evidence_kind"] = "measured"
        report = self.report()
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(report["physical_online_metrics"]["p95"]["count"], 0)

    def test_undeclared_schedule_rejected(self):
        self.prediction["mode"] = "declared_plan"
        with self.assertRaisesRegex(ValueError, "declared before"):
            self.report()

    def test_execution_context_mismatch_rejected(self):
        self.observation["execution_context_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "execution contexts differ"):
            self.report()

    def test_unpaired_candidate_cohorts_rejected(self):
        self.observation["candidates"][1]["offered_cohort_sha256"] = "different-inputs"
        with self.assertRaisesRegex(ValueError, "not paired"):
            self.report()

    def test_missing_six_term_evidence_stays_unknown(self):
        self.observation["evidence_kind"] = "measured"
        self.prediction["candidates"][1]["terms"]["M"] = None
        report = self.report()
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(report["physical_online_decisions"]["count"], 0)
        self.assertEqual(report["point_coverage"]["delta_U"]["requested"], 1)

    def test_observed_tail_comes_from_full_latency_set(self):
        report = self.report()
        self.assertEqual(report["cases"][0]["per_task"]["text"]["observed_p95_ms"], 540)

    def test_duplicate_validation_runs_rejected(self):
        manifest = self.seal()
        duplicate = copy.deepcopy(manifest["cases"][0])
        manifest["cases"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "duplicated"):
            assess(manifest, artifact_root=self.root)

    def test_omitting_planned_case_cannot_pass(self):
        self.observation["evidence_kind"] = "measured"
        planned = copy.deepcopy(self.protocol["planned_cases"][0])
        planned["id"] = "missing-case"
        self.protocol["planned_cases"].append(planned)
        self.refresh_protocol()
        report = self.report()
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(report["missing_case_ids"], ["missing-case"])
        self.assertEqual(report["coverage"]["service"]["requested_cases"], 2)

    def test_omitting_observed_task_cannot_pass(self):
        self.observation["evidence_kind"] = "measured"
        self.protocol["planned_cases"][0]["expected_tasks"].append("ocr")
        self.refresh_protocol()
        report = self.report()
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(report["cases"][0]["missing_observed_tasks"], ["ocr"])
        self.assertEqual(report["point_coverage"]["p95"]["requested"], 2)

    def test_zero_observed_gain_is_distinct_from_negative(self):
        self.observation["evidence_kind"] = "measured"
        self.observation["candidates"][1]["terms"] = {key: .5 for key in "QTLPME"}
        self.prediction["candidates"][1]["terms"] = {key: .6 for key in "QTLPME"}
        self.assertEqual(self.report()["physical_online_decisions"]["three_way_sign_accuracy"], 0)

    def test_cli_refuses_overwrite(self):
        manifest = self.seal()
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest))
        output = self.root / "report.json"
        main(["--input", str(path), "--output", str(output)])
        self.assertEqual(json.loads(output.read_text())["status"], "INCOMPLETE")
        with self.assertRaises(SystemExit):
            main(["--input", str(path), "--output", str(output)])


if __name__ == "__main__":
    unittest.main()
