import json
from pathlib import Path
import struct
import tempfile
import unittest

from gsdeploy.profiling.static_inputs import checkpoint_inventory, score_sample, shared_protocol


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts/static-study-v6-20261021/inputs"


class StaticInputsUnitTests(unittest.TestCase):
    def test_native_qa_and_ocr_quality_without_absolute_gate(self):
        sample = {"task": "text", "answers": ["the Bactrian camel"]}
        output = {"task": "text", "prediction": "Bactrian camel", "success": True}
        self.assertTrue(score_sample(sample, output)["correct"])
        # CTC text is normalized using the same page view as the audited release.
        sample = {"task": "ocr", "label": "TOTAL 10.00"}
        score = score_sample(sample, {"task": "ocr", "prediction": "total\n10.00", "success": True})
        self.assertEqual(score["errors"], 0)
        self.assertEqual(score["reference_units"], 11)
        # An inaccurate executable path still produces a finite loss, not rejection.
        self.assertGreater(score_sample(sample, {"task": "ocr", "prediction": "x" * 30, "success": True})["errors"], 11)

    def test_zero_placeholder_is_not_a_ready_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            path.write_bytes(b"")
            checkpoint = {"path": directory, "files": [{"name": path.name, "size": 0}]}
            self.assertFalse(checkpoint_inventory(checkpoint)["ready"])
            header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
            checkpoint["files"][0]["size"] = path.stat().st_size
            self.assertTrue(checkpoint_inventory(checkpoint)["ready"])

    def test_merge_keeps_each_tasks_native_preset(self):
        with tempfile.TemporaryDirectory() as directory:
            specs = {}
            for task, preset in (("text", "greedy"), ("image", "beam")):
                policy = {"tasks": {"text": {"dataset": "qa"}, "image": {"dataset": "images"}},
                          "native_decoding_presets": {"model": {"text": preset, "image": preset}}}
                path = Path(directory) / (task + ".json")
                path.write_text(json.dumps(policy))
                specs[task] = {"resources": {"protocol": str(path)}}
            self.assertEqual(shared_protocol(specs)["native_decoding_presets"]["model"],
                             {"text": "greedy", "image": "beam"})


@unittest.skipUnless((BUNDLE / "manifest.json").exists(), "real retained bundle is not prepared")
class RetainedBundleIntegrationTests(unittest.TestCase):
    def test_quality_matches_runtime_pool_and_references(self):
        quality = json.loads((BUNDLE / "quality-index.json").read_text())
        pool = [json.loads(line) for line in (BUNDLE / "runtime-inputs.jsonl").read_text().splitlines()]
        self.assertEqual(len(pool), 160)
        self.assertEqual(sum(len(v) for v in quality["paths"].values()), 16)
        for row in pool:
            self.assertNotIn("native_answers", row["sample"])
            self.assertNotIn("label", row["sample"])
            task, sid = row["task_type"], row["pool_id"]
            self.assertIn(sid, quality["references"][task])
            for paths in quality["paths"].values():
                if task in paths:
                    self.assertIn(sid, paths[task]["samples"])

    def test_resource_import_preserves_failed_and_task_specific_states(self):
        catalog = json.loads((BUNDLE / "catalog.json").read_text())
        index = json.loads((BUNDLE / "profile-index.json").read_text())
        self.assertEqual(sum(r["status"] == "COMPLETE" for r in index["rows"]), 61)
        self.assertEqual(sum(r["status"] != "COMPLETE" for r in index["rows"]), 6)
        curves = catalog["task_replica_profiles"]["generalist_2"]
        self.assertLess(curves["text"]["2"]["capacity_ratio_vs_n1"], 1)
        self.assertGreater(curves["asr"]["2"]["capacity_ratio_vs_n1"], 1.5)
        # n4 ASR remains a candidate even though n4 QA was not measured.
        self.assertIn("4", catalog["capacity_factors"]["generalist_1"])
        self.assertNotIn("4", catalog["task_replica_profiles"]["generalist_1"]["text"])
        self.assertFalse(catalog["energy_profiles"])


if __name__ == "__main__":
    unittest.main()
