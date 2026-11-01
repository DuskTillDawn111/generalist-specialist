"""Six-term arithmetic and quality provenance regression tests, CPU only."""
import copy
import random
import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from gsdeploy.quality import predict_quality, task_quality, measure_quality, quality_digest
from gsdeploy.prediction import components


def quality_fixture(mode="per_sample_replay"):
    paths = {m: {"text": {"metric": "em", "unit": "fraction", "value": value,
                         "checkpoint_sha256": "CPU fixture", "source": "CPU fixture",
                         "samples": {"a": {"correct": m == "specialist"},
                                     "b": {"correct": m == "generalist"}}}}
             for m, value in [("generalist", 0.5), ("specialist", 0.9)]}
    return {"quality": {"schema_version": 1, "mode": mode,
                        "identity": {k: "CPU fixture" for k in
                                     ("input_pool_sha256", "reference_sha256", "protocol_sha256", "source")},
                        "paths": paths, "input_order": {"text": ["a", "b"]}}}


class QualityUtilityTests(unittest.TestCase):
    def test_missing_quality_does_not_get_zero_or_renormalized(self):
        weights = {k: 1 / 6 for k in "QT LPME".replace(" ", "")}
        result = components({"text": {"mean_ms": 1, "p95_ms": 1}},
                            {"text": 1}, {"text": 1}, 0, 0, weights, E=0)
        self.assertIsNone(result["Q_norm"])
        self.assertIsNone(result["Q_contribution"])
        self.assertIsNone(result["U"])
        self.assertFalse(result["utility_complete"])
        self.assertAlmostEqual(result["diagnostic_partial_U"], -2 / 6)

    def test_legacy_quality_zero_semantics_preserved(self):
        result = components({"text": {"mean_ms": 1, "p95_ms": 1}},
                            {"text": 1}, {"text": 1}, 0, 0,
                            {"Q": 0, "T": .25, "L": .25, "P": .25, "M": .25})
        self.assertEqual(result["utility_schema"], "legacy_four_term")
        self.assertEqual(result["Q_contribution"], 0)
        self.assertEqual(result["U"], -.5)

    def test_replay_uses_input_sequence_and_actual_replica_router(self):
        data = quality_fixture()
        trace = tuple((i, "text") for i in range(8))
        x = {"generalist": 2, "specialist": 1}
        row = {"generalist": .5, "specialist": .5}
        from gsdeploy.routing import Router
        from gsdeploy.trace import expand_routes
        aliases, routes = expand_routes(x, {"text": row})
        router = Router(routes)
        order = ["a", "b"]
        random.Random("17:samples:text").shuffle(order)
        expected = sum(not data["quality"]["paths"][aliases[router.choose("text")]]["text"]
                       ["samples"][order[i % 2]]["correct"] for i in range(8)) / 8
        loss, evidence = task_quality(data, "text", x, row, trace, seed=17)
        self.assertEqual(loss, expected)
        self.assertEqual(evidence["kind"], "offline_per_sample_replay")
        self.assertFalse(evidence["runtime_output_reproduction_verified"])
        data["quality"]["paths"]["generalist"]["text"]["samples"].pop("a")
        self.assertIsNone(task_quality(data, "text", x, row, trace, seed=17)[0])

    def test_aggregate_proxy_is_explicit_and_low_accuracy_is_not_rejected(self):
        data = quality_fixture("aggregate_proxy")
        data["quality"]["paths"]["generalist"]["text"]["value"] = 0
        loss, evidence = predict_quality(data, {"generalist": 1, "specialist": 1},
                                        {"text": {"generalist": .5, "specialist": .5}},
                                        {"text": 1}, ((0, "text"),))
        self.assertAlmostEqual(loss, .55)
        self.assertEqual(evidence["kind"], "aggregate_proxy")
        data["quality"]["paths"]["generalist"]["text"].pop("checkpoint_sha256")
        self.assertIsNone(predict_quality(data, {"generalist": 1},
                                         {"text": {"generalist": 1}}, {"text": 1},
                                         ((0, "text"),))[0])

    def test_edit_error_rates_are_micro_aggregated_and_can_exceed_one(self):
        data = quality_fixture()
        q = data["quality"]
        q["paths"] = {"m": {"ocr": {"metric": "cer", "unit": "fraction", "source": "CPU fixture",
                                    "checkpoint_sha256": "CPU fixture", "samples": {
                                        "short": {"errors": 3, "reference_units": 1},
                                        "long": {"errors": 1, "reference_units": 9}}}}}
        q["input_order"] = {"ocr": ["short", "long"]}
        loss, evidence = task_quality(data, "ocr", {"m": 1}, {"m": 1}, ((0, "ocr"), (1, "ocr")))
        self.assertEqual(loss, .4)
        self.assertEqual(evidence["reference_denominator"], 10)
        q["paths"]["m"]["ocr"]["samples"]["long"]["errors"] = 30
        self.assertEqual(task_quality(data, "ocr", {"m": 1}, {"m": 1},
                                      ((0, "ocr"), (1, "ocr")))[0], 3.3)

    def test_measured_quality_uses_actual_output_not_offline_correctness(self):
        data = quality_fixture()
        data["quality"]["references"] = {"text": {"a": {"task": "text", "answers": ["gold"]}}}
        data["quality"]["input_order"] = {"text": ["a"]}
        data["quality"]["identity"]["reference_sha256"] = quality_digest(data["quality"]["references"])
        raw = {"preflight": {"sample_pool_sha256": "CPU fixture"},
               "transport_config": {"random_seed": 42},
               "replica_to_family": {"generalist#0": "generalist"},
               "request_records": [{"task": "text", "model": "generalist#0", "pool_id": "a",
                                    "status": "completed", "worker_response": {
                                        "task": "text", "prediction": "gold", "success": True}}]}
        self.assertEqual(measure_quality(data, raw, {"text": 1})[0], 0)
        raw["request_records"][0]["worker_response"]["prediction"] = "wrong"
        self.assertEqual(measure_quality(data, raw, {"text": 1})[0], 1)
        raw["request_records"][0].pop("worker_response")
        self.assertIsNone(measure_quality(data, raw, {"text": 1})[0])

    def test_native_evidence_has_no_absolute_quality_threshold(self):
        from gsdeploy.evidence import verify_admission, checkpoint, digest, quality_covers
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "weights.bin"
            weights.write_bytes(b"CPU fixture")
            actual = checkpoint(weights)
            data = quality_fixture("aggregate_proxy")
            record = data["quality"]["paths"]["generalist"]["text"]
            record.update(value=0, checkpoint_sha256=actual["sha256"])
            report = root / "quality.json"
            report.write_text(json.dumps(record))
            manifest = {"schema_version": 2, "contract": "quality_evidence_v1",
                        "quality_sha256": quality_digest(data["quality"]),
                        "models": {"generalist": {"checkpoint": actual,
                            "quality_reports": [{"path": str(report), "sha256": digest(report)}]}}}
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            winner = {"deployment": {"generalist": 1}, "routing": {"text": {"generalist": 1}}}
            config = SimpleNamespace(weights={"Q": 1 / 6}, catalog=data)
            evidence = verify_admission(config, winner,
                                        {"generalist": {"env": {"GS_MODEL_PATH": str(weights)}}}, path)
            self.assertTrue(evidence["verified"])
            self.assertTrue(quality_covers(config, {"plan": winner, "admission_evidence": evidence}))
            record.pop("checkpoint_sha256")
            record["checkpoint_identity"] = {"kind": "configuration_manifest", "source": "CPU fixture", "sha256": "CPU receipt"}
            manifest["quality_sha256"] = quality_digest(data["quality"])
            manifest["models"]["generalist"].update(
                historical_quality_checkpoint_identities={"text": copy.deepcopy(record["checkpoint_identity"])},
                runtime_checkpoint_scope="current_execution_only_not_retroactive_offline_hash",
                receipt_checks=[{"current_sha256": actual["sha256"], "historical_sha256": None}])
            path.write_text(json.dumps(manifest))
            evidence = verify_admission(config, winner,
                                        {"generalist": {"env": {"GS_MODEL_PATH": str(weights)}}}, path)
            self.assertFalse(evidence["pairs"][0]["offline_full_checkpoint_equivalence_verified"])
            self.assertTrue(quality_covers(config, {"plan": winner, "admission_evidence": evidence}))


if __name__ == "__main__":
    unittest.main()
