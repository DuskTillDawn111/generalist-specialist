"""Admission integrity and launch prevention, entirely on CPU fixtures."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from gsdeploy import DeploymentSystem, SystemConfig
from gsdeploy.evidence import admission_covers, checkpoint, digest, verify_admission
from gsdeploy.evaluation import evaluate_run
from gsdeploy.runtime.preflight import preflight
from gsdeploy.runtime.runner import registry_for
from gsdeploy.energy import energy_key
from gsdeploy.trace import make_trace, trace_digest

ROOT = Path(__file__).resolve().parents[1]


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.system = DeploymentSystem(SystemConfig.load(ROOT / "configs/demo.json"))
        self.winner = self.system.plan(deployment={"generalist": 1})["winner"]
        weights = self.root / "weights.bin"
        weights.write_bytes(b"fixture weights")
        self.weights = weights
        self.workers = {
            "generalist": {
                "command": [sys.executable, "-m", "gsdeploy.backends.mock"],
                "env": {
                    "GS_MODEL_PATH": str(weights),
                    "GS_PROMPT_FILE": str(ROOT / "configs/prompts.json"),
                },
            }
        }
        identity = checkpoint(weights)
        report = {
            "model": "generalist",
            "checkpoint_sha256": identity["sha256"],
            "tasks": {
                task: {
                    "metric": "accuracy",
                    "value": 0.9,
                    "dataset": "CPU fixture only",
                    "protocol": "CPU fixture only",
                }
                for task in self.system.config.workload
            },
        }
        self.report = self.root / "quality.json"
        self.report.write_text(json.dumps(report))
        self.manifest = self.root / "admission.json"
        self.data = {
            "schema_version": 1,
            "models": {
                "generalist": {
                    "checkpoint": identity,
                    "quality_report": "quality.json",
                    "quality_report_sha256": digest(self.report),
                    "admission_rules": {
                        task: {
                            "metric": "accuracy",
                            "threshold": 0.8,
                            "direction": "min",
                        }
                        for task in report["tasks"]
                    },
                }
            },
        }
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.data))

    def test_verified_records_cover_exact_routed_pairs(self):
        evidence = verify_admission(
            self.system.config, self.winner, self.workers, self.manifest
        )
        raw = {"plan": self.winner, "admission_evidence": evidence}
        self.assertTrue(admission_covers(raw))
        incomplete = copy.deepcopy(raw)
        incomplete["admission_evidence"]["pairs"].pop()
        self.assertFalse(admission_covers(incomplete))
        self.assertFalse(
            admission_covers({"plan": self.winner, "quality_admission_confirmed": True})
        )
        raw["admission_evidence"]["pairs"][0]["measurement"]["value"] = float("nan")
        self.assertFalse(admission_covers(raw))

    def test_changed_checkpoint_report_and_failed_quality_are_rejected(self):
        self.weights.write_bytes(b"other weights")
        with self.assertRaisesRegex(ValueError, "Checkpoint fingerprint mismatch"):
            verify_admission(
                self.system.config, self.winner, self.workers, self.manifest
            )
        self.weights.write_bytes(b"fixture weights")
        self.report.write_text(self.report.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Quality report fingerprint mismatch"):
            verify_admission(
                self.system.config, self.winner, self.workers, self.manifest
            )
        entry = self.data["models"]["generalist"]
        entry["quality_report_sha256"] = digest(self.report)
        entry["admission_rules"]["text"]["threshold"] = 0.95
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Quality admission failed"):
            verify_admission(
                self.system.config, self.winner, self.workers, self.manifest
            )

    def test_formal_utility_requires_success_and_verified_admission(self):
        raw = {
            "plan": self.winner,
            "backend": "native",
            "status": "completed",
            "replica_to_family": {"generalist#0": "generalist"},
            "per_task_metrics": {
                task: {"mean_e2e_latency_ms": 2, "p95_e2e_latency_ms": 3}
                for task in self.system.config.workload
            },
            "system_metrics": {"completion_ratio": 1, "completed_within_window": 300},
            "measurement_window": {"duration_seconds": 1},
            "memory_evidence": {
                "measured": True,
                "resource_valid": True,
                "safe_memory_mib": 64,
                "scope": "CPU test fixture",
            },
            "admission_evidence": verify_admission(
                self.system.config, self.winner, self.workers, self.manifest
            ),
        }
        trace = make_trace(self.system.config.workload, 1, 42)
        raw["measurement_window"].update(
            start_time_ns=0,
            end_time_ns=10**9,
            drain_end_time_ns=10**9,
            offered_requests=len(trace),
            trace_sha256=trace_digest(trace),
        )
        raw["transport_config"] = {
            "random_seed": 42,
            "arrival_process": "deterministic_periodic",
        }
        raw["request_records"] = [
            {
                "request_id": f"measurement:{i}",
                "task": t,
                "model": "generalist#0",
                "scheduled_time_ns": round(a * 1e9),
            }
            for i, (a, t) in enumerate(trace)
        ]
        for record in raw["request_records"]:
            record.update(
                status="completed",
                actual_submit_time_ns=record["scheduled_time_ns"],
                completion_time_ns=record["scheduled_time_ns"] + 1000000,
            )
        raw["system_metrics"]["completed_within_window"] = len(trace)
        raw["energy_evidence"] = {
            "kind": "measured",
            "scope": "whole_gpu_board",
            "coverage_valid": True,
            "run_joules": 80.0,
            "startup_joules": 8.0,
            "profile_key": energy_key(
                self.winner["deployment"],
                self.winner["routing"],
                self.system.config.workload,
                1,
                42,
                "deterministic_periodic",
                catalog=self.system.config.catalog,
            ),
        }
        with (
            patch("gsdeploy.evaluation.reconstruct", return_value={}),
            patch(
                "gsdeploy.evaluation.classify",
                return_value={"stability_state_v2": "steady"},
            ),
        ):
            self.assertTrue(
                evaluate_run(self.system.config, raw)["utility"]["eligible"]
            )
            self.assertAlmostEqual(
                evaluate_run(self.system.config, raw)["utility"]["T_norm"], 0
            )
            saved = raw.pop("energy_evidence")
            self.assertIsNone(evaluate_run(self.system.config, raw)["utility"]["U"])
            raw["energy_evidence"] = saved
            saved["kind"] = "synthetic_fixture"
            self.assertIsNone(evaluate_run(self.system.config, raw)["utility"]["U"])
            saved["kind"] = "measured"
            raw["request_records"][0]["scheduled_time_ns"] += 1
            self.assertIsNone(evaluate_run(self.system.config, raw)["utility"]["U"])
            raw["request_records"][0]["scheduled_time_ns"] -= 1
            raw["status"] = "failed"
            self.assertIsNone(evaluate_run(self.system.config, raw)["utility"]["U"])
            raw["status"] = "completed"
            raw.pop("admission_evidence")
            raw["quality_admission_confirmed"] = True
            self.assertIsNone(evaluate_run(self.system.config, raw)["utility"]["U"])

    def test_directory_fingerprint_detects_added_files(self):
        directory = self.root / "checkpoint"
        directory.mkdir()
        (directory / "config.json").write_text("{}")
        before = checkpoint(directory)
        (directory / "weights.bin").write_bytes(b"weights")
        self.assertNotEqual(before["sha256"], checkpoint(directory)["sha256"])

    def test_missing_gpu_prevents_worker_start(self):
        # Text-only fixture avoids pretending the mock media samples are native assets.
        data = self.system.config.as_dict()
        data["workload"] = {"text": 50}
        system = DeploymentSystem(SystemConfig(**data))
        plan = system.plan(deployment={"generalist": 1})
        with (
            patch(
                "gsdeploy.runtime.preflight.subprocess.run",
                side_effect=FileNotFoundError("no GPU"),
            ),
            patch("gsdeploy.runtime.runner.DeploymentRuntime.start") as start,
        ):
            checks = preflight(
                system.config,
                plan["winner"],
                ROOT / "examples/demo_samples.jsonl",
                self.workers,
                require_gpu=True,
            )
            self.assertFalse(checks["ready"])
            self.assertIn("GPU unavailable", checks["issues"][0])
            with self.assertRaisesRegex(ValueError, "Native preflight failed"):
                system.run(
                    plan,
                    ROOT / "examples/demo_samples.jsonl",
                    backend="native",
                    workers=self.workers,
                )
            start.assert_not_called()

    def test_worker_external_pythonpath_is_preserved(self):
        self.workers["generalist"]["env"]["PYTHONPATH"] = "/external/strhub"
        registry = registry_for(
            self.system.config, {"generalist#0": "generalist"}, "native", self.workers
        )
        self.assertTrue(
            registry["generalist#0"]["env"]["PYTHONPATH"].endswith("/external/strhub")
        )


if __name__ == "__main__":
    unittest.main()
