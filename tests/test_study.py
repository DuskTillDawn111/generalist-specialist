"""Study preparation must not recycle old protocols or grant admission."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from gsdeploy.profiling.study import ACTIVE_TASKS, prepare_study


ROOT = Path(__file__).resolve().parents[1]


class FourTaskStudyTests(unittest.TestCase):
    def test_qwen_loader_speaker_dictionary_is_a_verified_resource(self):
        from gsdeploy.profiling.resources import selected_files
        names = ['model.safetensors', 'config.json', 'spk_dict.pt', 'unused.pt']
        selected = selected_files({'siblings': [{'rfilename': n} for n in names]})
        self.assertEqual({f['rfilename'] for f in selected}, set(names) - {'unused.pt'})

    def fixture(self, root, old_text=False):
        study = json.loads((ROOT / "configs/study.json").read_text())
        config = root / "config.json"
        config.write_text(json.dumps(study))
        source = root / "source"
        source.mkdir()
        asset = source / "asset"
        asset.write_bytes(b"cpu integrity fixture, not an inference sample")
        declared = {}
        for task in ACTIVE_TASKS:
            row = {"id": task, "task": task, "dataset": study["tasks"][task]["dataset"], "label": "fixture"}
            if task == "text":
                if old_text:
                    row.update(dataset="SST-2", input="old sentiment text")
                else:
                    row.update(context="context", question="question", answers=["answer"])
            else:
                row.update(input=str(asset), input_sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
            path = source / (task + ".jsonl")
            path.write_text(json.dumps(row) + "\n")
            declared[task] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        (source / "sample-manifests.json").write_text(json.dumps({"tasks": declared}))
        # Removed-task artifacts must neither be parsed nor copied.
        (source / "lid.jsonl").write_text("invalid old data")
        (source / "visual_qa.jsonl").write_text("invalid old data")
        return config, source, asset

    def test_removed_tasks_are_absent_and_candidates_stay_unadmitted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, source, _ = self.fixture(root)
            output = root / "out"
            report = prepare_study(config, source, output)
            self.assertFalse(report["gpu_started"])
            self.assertFalse(report["native_execution_ready"])
            self.assertIsNone(report["utility"])
            self.assertFalse((output / "lid.jsonl").exists())
            self.assertFalse((output / "visual_qa.jsonl").exists())
            self.assertEqual({json.loads(s)["task"] for s in (output / "samples.jsonl").read_text().splitlines()}, set(ACTIVE_TASKS))
            self.assertEqual({json.loads(s)["task"] for s in (output / "speech.jsonl").read_text().splitlines()}, {"asr"})
            before = (output / "study-status.json").read_bytes()
            with self.assertRaises(FileExistsError):
                prepare_study(config, source, output)
            self.assertEqual(before, (output / "study-status.json").read_bytes())

    def test_old_sentiment_samples_cannot_be_relabelled_as_new_qa(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, source, _ = self.fixture(root, old_text=True)
            with self.assertRaisesRegex(ValueError, "dataset protocol"):
                prepare_study(config, source, root / "out")
            self.assertFalse((root / "out").exists())

    def test_changed_input_cannot_reuse_a_frozen_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, source, asset = self.fixture(root)
            asset.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "Input changed"):
                prepare_study(config, source, root / "out")
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
