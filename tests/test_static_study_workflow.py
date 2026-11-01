"""Static experiment preparation stays CPU-only and preserves physical identities."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gsdeploy.config import SystemConfig
from gsdeploy.static_study import _job, execute_job, prepare_experiments, validate_packet, measurement_queue, freeze_evaluator, queue_packet, bind_native_job
from tests.test_quality_utility import quality_fixture


def config_fixture():
    catalog = {"safe_factor": 1, "capability_contract": "verified_executable_v1",
               "models": {m: {"role": role, "supported_tasks": ["text"], "executable_tasks": ["text"],
                              "service_time_ms": {"text": 1}}
                          for m, role in [("generalist", "generalist"), ("specialist", "specialist")]},
               "capacity_factors": {m: {"0": 0, "1": 1} for m in ("generalist", "specialist")},
               "memory_table_mib": {m: {"0": 0, "1": 1} for m in ("generalist", "specialist")},
               **quality_fixture()}
    return SystemConfig(catalog=catalog, workload={"text": 1}, gpu_memory_mib=32,
                        latency_reference_ms={"text": 1}, weights={k: 1 / 6 for k in "QTLPME"},
                        utility_schema="six_term_quality_v1", horizon_seconds=1800)


class StaticStudyWorkflowTests(unittest.TestCase):
    def test_native_binding_accepts_inventory_symlink_and_preserves_historical_quality(self):
        from gsdeploy.evidence import digest
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "snapshot"
            model.mkdir()
            blob = root / "shared-blob"
            blob.write_bytes(b"CPU fixture weights")
            link = model / "weights.bin"
            link.symlink_to(blob)
            report = root / "quality-report.json"
            report.write_text("CPU quality report fixture")
            config = config_fixture()
            record = config.catalog["quality"]["paths"]["generalist"]["text"]
            for score in record["samples"].values():
                score["source"] = str(report)
            plan = {"deployment": {"generalist": 1}, "routing": {"text": {"generalist": 1}}}
            workers = {"generalist": {"env": {"GS_MODEL_PATH": str(model)}, "checkpoint": {
                "path": str(model), "ready": True, "files": [{"path": str(link), "size": blob.stat().st_size,
                                                             "historical_sha256": digest(blob)}]}, "execution": {}}}
            packet = {"config": config.as_dict(), "workers": workers}
            job = {"plan": plan, "identity": {"horizon_seconds": 600}, "seed": 42}
            derived, manifest_path, _ = bind_native_job(packet, job, root / "binding")
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["models"]["generalist"]["checkpoint"]["files"], {"weights.bin": digest(blob)})
            self.assertEqual(derived["catalog"]["quality"], config.catalog["quality"])

    def test_cross_pool_identity_ignores_unused_candidates_but_not_traffic(self):
        config = config_fixture()
        workers = {"generalist": {"command": ["python"], "env": {}}}
        plan = {"deployment": {"generalist": 1}, "routing": {"text": {"generalist": 1}}}
        first = _job(config, workers, plan, "first", "calibration", 42, [], 30)
        data = config.as_dict()
        for key in ("models", "capacity_factors", "memory_table_mib"):
            data["catalog"][key].pop("specialist")
        reduced = _job(SystemConfig(**data), workers, plan, "second", "calibration", 42, [], 30)
        self.assertEqual(first["id"], reduced["id"])
        self.assertNotEqual(first["provenance"], reduced["provenance"])
        changed = _job(config, workers, plan, "third", "calibration", 43, [], 30)
        self.assertNotEqual(first["id"], changed["id"])
        routing_source = Path(__file__).resolve().parents[1] / "gsdeploy/routing.py"
        code_a = _job(config, workers, plan, "source_a", "calibration", 42,
                      [{"path": str(routing_source), "sha256": "fixture revision a"}], 30)
        code_b = _job(config, workers, plan, "source_b", "calibration", 42,
                      [{"path": str(routing_source), "sha256": "fixture revision b"}], 30)
        self.assertNotEqual(code_a["id"], code_b["id"])

    def test_prepare_validate_tamper_and_missing_energy_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            inputs.mkdir()
            config = config_fixture()
            contract = {"calibration_horizon_seconds": 600, "seeds": {"development": [42], "confirmation": [43, 44, 45]},
                        "warmup": {"seconds": 30}}
            for name, value in [("system-config.json", config.as_dict()), ("study-contract.json", contract),
                                ("worker-registry.json", {m: {"command": ["python"], "env": {}}
                                                          for m in config.catalog["models"]}),
                                ("source-bindings.json", {})]:
                (inputs / name).write_text(json.dumps(value))
            packet = prepare_experiments(inputs, root / "out", root=root)
            self.assertFalse(packet["gpu_started"])
            self.assertEqual(packet["confirmation"]["status"], "WAITING_GPU_CALIBRATION")
            self.assertTrue(validate_packet(packet)["valid"])
            altered = copy.deepcopy(packet)
            altered["jobs"][0]["plan"]["routing"]["text"]["generalist"] = 2
            self.assertFalse(validate_packet(altered)["valid"])
            with self.assertRaisesRegex(ValueError, "explicit"):
                execute_job(packet, packet["jobs"][0]["id"], root / "run")
            self.assertFalse((root / "run").exists())
            with self.assertRaises(FileExistsError):
                prepare_experiments(inputs, root / "out", root=root)

    def test_profile_memory_is_exact_condition_and_unknown_is_queued_once(self):
        from gsdeploy.performance import profile_key
        from gsdeploy.prediction import predict
        config = config_fixture()
        catalog = config.catalog
        catalog.update(performance_mode="profiled", memory_mode="profiled")
        plan = {"deployment": {"generalist": 1}, "routing": {"text": {"generalist": 1}}}
        args = (catalog, plan["deployment"], plan["routing"], config.workload,
                config.gpu_memory_mib, config.latency_reference_ms, config.weights)
        key = profile_key(catalog, plan["deployment"], plan["routing"], config.workload, 1800, 42, "deterministic_periodic")
        catalog["colocation_profiles"] = {key: {"kind": "synthetic_fixture", "scope": "whole_gpu_colocation",
            "coverage_valid": True, "service_time_ms": {"generalist": {"text": 1}}, "safe_memory_mib": 3}}
        self.assertEqual(predict(*args, horizon=1800)["safe_memory_mib"], 3)
        self.assertEqual(config.validate_plan(plan["deployment"], plan["routing"])["safe_memory_mib"], 3)
        self.assertTrue(predict(*args, horizon=600)["performance_unknown"])
        catalog["colocation_profiles"][key].pop("safe_memory_mib")
        self.assertTrue(predict(*args, horizon=1800)["memory_unknown"])
        queued = measurement_queue(config, [plan, plan], {"generalist": {"command": ["python"], "env": {}}}, [])
        self.assertEqual(len(queued["jobs"]), 1)
        self.assertEqual(queued["evaluated"][0]["state"], "needs_measurement")
        packet = {"config": config.as_dict(), "workers": {"generalist": {"command": ["python"], "env": {}}},
                  "jobs": [], "bindings": [], "frozen_files_sha256": __import__("gsdeploy.quality", fromlist=["quality_digest"]).quality_digest([])}
        derived = queue_packet(packet, config, [plan, plan])
        self.assertEqual(len(derived["jobs"]), 1)
        self.assertTrue(validate_packet(derived)["valid"])
        self.assertFalse(derived["gpu_started"])

    def test_freeze_exact_conditions_replicates_and_reject_confirmation(self):
        # Deliberately synthetic raw transport fixture; evaluate_run validity is
        # isolated here, with measured-output validation covered in quality tests.
        from gsdeploy.prediction import predict
        from gsdeploy.quality import quality_digest
        config = config_fixture()
        workers = {"generalist": {"command": ["python"], "env": {}}}
        plan = {"deployment": {"generalist": 1}, "routing": {"text": {"generalist": 1}}}
        job = _job(config, workers, plan, "fixture calibration", "calibration", 42, [], 30)
        packet = {"config": config.as_dict(), "workers": workers, "jobs": [job],
                  "bindings": [], "frozen_files_sha256": quality_digest([])}
        actual_identity = copy.deepcopy(job["identity"])
        actual_identity["hardware_contract"]["device_uuid"] = "CPU fixture GPU"
        actual_identity["runtime_checkpoint_sha256"] = {"generalist": "CPU fixture weights"}
        raw_config = config.as_dict()
        raw_config["catalog"]["measurement_context"] = {"gpu_uuid": "CPU fixture GPU",
            "runtime_checkpoint_sha256": {"generalist": "CPU fixture weights"}}
        raw = {"experiment_job_id": job["id"], "backend": "native", "status": "completed",
               "config": raw_config, "plan": plan, "executed_measurement_identity": actual_identity,
               "executed_measurement_id": quality_digest(actual_identity),
               "transport_config": {"random_seed": 42, "arrival_process": "deterministic_periodic",
                                    "warmup_seconds": 30, "measurement_seconds": 1800, "drain_timeout_seconds": 60},
               "priming_policy": job["identity"]["warmup_policy"],
               "priming_size_metric": job["identity"]["warmup_size_metric"],
               "measurement_window": {"trace_sha256": job["identity"]["trace_sha256"]},
               "preflight": {"gpu": {"uuid": "CPU fixture GPU"}},
               "replica_to_family": {"generalist#0": "generalist"},
               "request_records": [{"model": "generalist#0", "task": "text", "worker_latency_ms": 2}],
               "energy_evidence": {"kind": "measured", "coverage_valid": True, "run_joules": 18, "startup_joules": 2},
               "memory_evidence": {"safe_memory_mib": 3}}
        with tempfile.TemporaryDirectory() as temporary, patch("gsdeploy.evaluation.evaluate_run", return_value={"utility": {"eligible": True}}):
            root = Path(temporary)
            first, second = root / "first.json", root / "second.json"
            first.write_text(json.dumps(raw))
            another = copy.deepcopy(raw)
            another["request_records"][0]["worker_latency_ms"] = 4
            another["memory_evidence"]["safe_memory_mib"] = 5
            second.write_text(json.dumps(another))
            report = freeze_evaluator(packet, [first, second], root / "frozen")
            self.assertEqual(report["conditions"], 1)
            frozen = SystemConfig.load(root / "frozen/system-config.json")
            profile = next(iter(frozen.catalog["colocation_profiles"].values()))
            self.assertEqual(profile["service_time_ms"]["generalist"]["text"], 3)
            self.assertEqual(profile["safe_memory_mib"], 5)
            self.assertEqual(profile["replicate_count"], 2)
            args = (frozen.catalog, plan["deployment"], plan["routing"], frozen.workload,
                    frozen.gpu_memory_mib, frozen.latency_reference_ms, frozen.weights)
            self.assertEqual(predict(*args, horizon=1800)["safe_memory_mib"], 5)
            self.assertTrue(predict(*args, horizon=600)["performance_unknown"])
            self.assertTrue(predict(*args, horizon=1800, seed=43)["performance_unknown"])
            packet["jobs"][0]["phase"] = "confirmation"
            with self.assertRaisesRegex(ValueError, "confirmation"):
                freeze_evaluator(packet, [first], root / "refit")


if __name__ == "__main__":
    unittest.main()
