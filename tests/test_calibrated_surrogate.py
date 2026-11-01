"""Native data is represented by explicit CPU fixtures; no GPU/model imports."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gsdeploy.calibration import fit_calibrated_surrogate, surrogate_service, surrogate_auxiliary, _import_source, declare_protocol
from gsdeploy.config import SystemConfig
from gsdeploy.prediction import components, predict
from gsdeploy.trace import make_trace
from tests.test_static_study_workflow import config_fixture


def calibration_fixture():
    config = config_fixture()
    config.catalog["quality"]["mode"] = "aggregate_proxy"
    config.catalog["task_replica_profiles"] = {m: {"text": {"1": {"mean_service_ms": 1.0}}}
                                               for m in config.catalog["models"]}
    imported = {}
    sources = []
    for i, share in enumerate([0.0, 1.0, .1, .2, .3, .4, .6, .8]):
        deployment = {m: 1 for m, fraction in [("generalist", share), ("specialist", 1 - share)] if fraction > 0}
        routing = {"text": {m: fraction for m, fraction in [("generalist", share), ("specialist", 1 - share)] if fraction > 0}}
        count = sum(deployment.values())
        watts = 100 + 10 * count
        duration = 12
        e = (watts + 20 * count / duration) / config.energy_reference_watts
        mem = 1.1 * count
        quality = share * .5 + (1 - share) * .1
        measured = components({"text": {"mean_ms": 1.2, "p95_ms": 1.2}}, config.workload,
                              config.latency_reference_ms, 0, mem / config.gpu_memory_mib,
                              config.weights, e, quality)
        from gsdeploy.trace import expand_routes
        from gsdeploy.routing import Router
        aliases, routes = expand_routes(deployment, routing)
        router = Router(routes)
        records = []
        for offset, task in make_trace(config.workload, duration, seed=42):
            records.append({"model": router.choose(task), "task": task, "worker_latency_ms": 1.2,
                            "scheduled_time_ns": round(offset * 1e9),
                            "completion_time_ns": round(offset * 1e9) + 1200000})
        raw_config = config.as_dict()
        raw_config["catalog"]["measurement_context"] = {"gpu_uuid": "CPU fixture GPU",
            "runtime_checkpoint_sha256": {m: "CPU fixture weights " + m for m in deployment}}
        raw = {"config": raw_config, "executed_measurement_id": "CPU fixture condition " + str(i),
               "plan": {"deployment": deployment, "routing": routing}, "replica_to_family": aliases,
               "request_records": records, "measurement_window": {"duration_seconds": duration},
               "transport_config": {"random_seed": 42}, "preflight": {"gpu": {"uuid": "CPU fixture GPU"}},
               "energy_evidence": {"run_joules": watts * duration, "startup_joules": 20 * count},
               "memory_evidence": {"safe_memory_mib": mem}}
        key = str(i)
        imported[key] = raw, measured, {"id": key, "kind": "CPU_ONLY_TEST_FIXTURE"}
        sources.append({"id": key})
    return config, imported, sources


class CalibratedSurrogateTests(unittest.TestCase):
    def fit(self, config, imported, sources):
        with patch("gsdeploy.calibration._import_source", side_effect=lambda source, accepted, phases=None: imported[source["id"]]):
            return fit_calibrated_surrogate(config, sources, [str(i) for i in range(6)], ["6", "7"], ["CPU fixture version"])

    def test_fit_holdout_transfer_new_window_seed_and_nonnegative_board_model(self):
        config, imported, sources = calibration_fixture()
        result, report = self.fit(config, imported, sources)
        fitted = SystemConfig(**result)
        spec = fitted.catalog["calibrated_surrogate"]
        self.assertEqual(report["status"], "VALIDATED_ON_HELDOUT_CONDITIONS")
        self.assertAlmostEqual(spec["service_multiplier"], 1.2)
        self.assertTrue(all(v >= 0 for v in spec["power_coefficients"]))
        self.assertLessEqual(len(spec["selected_power_columns"]), 3)
        self.assertEqual(spec["training_ids"], [str(i) for i in range(6)])
        self.assertIn("per_task_latency_ms", report["validation"]["conditions"][0])
        x, routing = imported["6"][0]["plan"].values()
        p = predict(fitted.catalog, x, routing, fitted.workload, fitted.gpu_memory_mib,
                    fitted.latency_reference_ms, fitted.weights, horizon=1800, seed=43)
        self.assertTrue(p["eligible"])
        self.assertEqual(p["energy_evidence"]["kind"], "calibrated_prediction")
        self.assertFalse(p["energy_evidence"]["measured"])
        self.assertEqual(p["performance_evidence"]["optimality_scope"], "frozen predictor finite domain only")
        self.assertGreaterEqual(p["safe_memory_mib"], sum(fitted.catalog["memory_table_mib"][m]["1"] for m in x))
        self.assertTrue(fitted.validate_plan(x, routing)["constraints_valid"])

    def test_bad_holdout_changes_status_not_fitted_coefficients(self):
        config, imported, sources = calibration_fixture()
        first, _ = self.fit(config, imported, sources)
        altered = copy.deepcopy(imported)
        altered["6"][1]["P_norm"] *= 10
        altered["6"][1]["U"] -= 1
        second, report = self.fit(config, altered, sources)
        for key in ["service_multiplier", "power_coefficients", "startup_joules_per_worker", "memory_multiplier"]:
            self.assertEqual(first["catalog"]["calibrated_surrogate"][key], second["catalog"]["calibrated_surrogate"][key])
        self.assertEqual(report["status"], "VALIDATION_FAILED")
        self.assertNotEqual(first["catalog"]["calibrated_surrogate"]["version_sha256"], report["version_sha256"])
        x, routing = altered["6"][0]["plan"].values()
        refused = predict(second["catalog"], x, routing, config.workload, config.gpu_memory_mib,
                          config.latency_reference_ms, config.weights, horizon=12)
        self.assertIsNone(refused["U"])
        self.assertFalse(refused["eligible"])
        self.assertIn("NOT_VALIDATED", refused["reason"])

    def test_unknown_paths_gpu_or_workload_do_not_become_measured(self):
        config, imported, sources = calibration_fixture()
        result, _ = self.fit(config, imported, sources)
        data = result["catalog"]
        x, routing = imported["6"][0]["plan"].values()
        changed = copy.deepcopy(data)
        changed["measurement_context"]["gpu_uuid"] = "different GPU"
        self.assertIsNone(surrogate_service(changed, x, routing, config.workload)[0])
        changed = copy.deepcopy(data)
        changed["measurement_context"]["runtime_checkpoint_sha256"]["generalist"] = "changed execution weights"
        self.assertIsNone(surrogate_service(changed, x, routing, config.workload)[0])
        changed = copy.deepcopy(data)
        changed["quality"]["paths"]["generalist"]["text"]["value"] = .99
        self.assertIsNone(surrogate_service(changed, x, routing, config.workload)[0])
        self.assertIsNone(surrogate_service(data, x, routing, {"text": 2})[0])
        data["task_replica_profiles"]["generalist"]["text"].pop("1")
        self.assertIsNone(surrogate_service(data, x, routing, config.workload)[0])

    def test_split_shortcuts_and_repeated_condition_are_rejected(self):
        config, imported, sources = calibration_fixture()
        with self.assertRaisesRegex(ValueError, "disjoint"):
            fit_calibrated_surrogate(config, sources, ["0", "1", "2", "3"], ["3", "4"], [])
        with self.assertRaisesRegex(ValueError, "disjoint"):
            fit_calibrated_surrogate(config, sources, ["0"] * 4, ["6", "7"], [])
        with self.assertRaisesRegex(ValueError, "disjoint"):
            fit_calibrated_surrogate(config, sources, ["0", "1", "2", "3"], ["6", "6"], [])
        imported["6"][0]["executed_measurement_id"] = imported["0"][0]["executed_measurement_id"]
        with self.assertRaisesRegex(ValueError, "Repeated"):
            self.fit(config, imported, sources)

    def test_native_import_checks_packet_revision_role_and_runtime_acceptance(self):
        from gsdeploy.evidence import digest
        from gsdeploy.quality import quality_digest
        from gsdeploy.static_study import _job
        config, imported, _ = calibration_fixture()
        config = replace(config, horizon_seconds=12)
        raw, measured, _ = copy.deepcopy(imported["0"])
        workers = {m: {"command": ["python"], "env": {}} for m in config.catalog["models"]}
        job = _job(config, workers, raw["plan"], "CPU fixture", "calibration", 42, [], 30)
        packet = {"config": config.as_dict(), "workers": workers, "jobs": [job], "bindings": [],
                  "frozen_files_sha256": quality_digest([])}
        identity = copy.deepcopy(job["identity"])
        identity["hardware_contract"]["device_uuid"] = "CPU fixture GPU"
        identity["runtime_checkpoint_sha256"] = raw["config"]["catalog"]["measurement_context"]["runtime_checkpoint_sha256"]
        raw.update(status="completed", backend="native", experiment_job_id=job["id"],
                   executed_measurement_identity=identity, executed_measurement_id=quality_digest(identity),
                   priming_policy=identity["warmup_policy"], priming_size_metric=identity["warmup_size_metric"])
        raw["config"]["horizon_seconds"] = 12
        raw["energy_evidence"]["kind"] = "measured"
        raw["measurement_window"]["trace_sha256"] = identity["trace_sha256"]
        raw["transport_config"].update(measurement_seconds=12, warmup_seconds=30, drain_timeout_seconds=60,
                                      arrival_process="deterministic_periodic")
        raw["preflight"]["gpu"]["memory_mib"] = 32
        raw["preflight"]["device"] = "0"
        raw["preflight"].update(ready=True, gpu_checked=True)
        measured["eligible"] = True
        with tempfile.TemporaryDirectory() as temporary, patch("gsdeploy.evaluation.evaluate_run", return_value={"utility": measured}):
            root = Path(temporary)
            pp, rp = root / "packet.json", root / "run.json"
            pp.write_text(json.dumps(packet))
            rp.write_text(json.dumps(raw))
            source = {"id": "CPU fixture", "path": str(rp), "sha256": digest(rp),
                      "packet_path": str(pp), "packet_sha256": digest(pp)}
            version = identity["runtime_code_sha256"]
            self.assertEqual(_import_source(source, {version})[1], measured)
            with self.assertRaisesRegex(ValueError, "explicitly accepted"):
                _import_source(source, set())
            packet["jobs"][0]["phase"] = "confirmation"
            pp.write_text(json.dumps(packet))
            source["packet_sha256"] = digest(pp)
            with self.assertRaisesRegex(ValueError, "development"):
                _import_source(source, {version})
            packet["jobs"][0]["phase"] = "adaptive"
            pp.write_text(json.dumps(packet))
            source["packet_sha256"] = digest(pp)
            with self.assertRaisesRegex(ValueError, "purpose"):
                _import_source(source, {version}, {"calibration", "adaptive"})
            source["purpose"] = "adaptive_development"
            self.assertEqual(_import_source(source, {version}, {"calibration", "adaptive"})[1], measured)
            with self.assertRaisesRegex(ValueError, "development"):
                _import_source(source, {version})

            # Recompute the actual condition digest: a self-consistent digest
            # does not authorize changing the prepared hardware contract.
            packet["jobs"][0]["phase"] = "calibration"
            pp.write_text(json.dumps(packet))
            source["packet_sha256"] = digest(pp)
            for field, value in [("memory_budget_mib", 100), ("single_gpu", False)]:
                altered = copy.deepcopy(raw)
                altered["executed_measurement_identity"]["hardware_contract"][field] = value
                altered["executed_measurement_id"] = quality_digest(altered["executed_measurement_identity"])
                rp.write_text(json.dumps(altered))
                source["sha256"] = digest(rp)
                with self.assertRaisesRegex(ValueError, "hardware contract"):
                    _import_source(source, {version})
            for scope, field, value in [("gpu", "memory_mib", .1), (None, "device", "0,1"),
                                         (None, "device", "1"), (None, "gpu_checked", False)]:
                altered = copy.deepcopy(raw)
                preflight = altered["preflight"] if scope is None else altered["preflight"][scope]
                preflight[field] = value
                rp.write_text(json.dumps(altered))
                source["sha256"] = digest(rp)
                with self.assertRaisesRegex(ValueError, "configuration/preflight"):
                    _import_source(source, {version})

            rp.write_text(json.dumps(raw))
            source["sha256"] = digest(rp)
            original = root / "original.json"
            original.write_text(json.dumps(packet))
            relocated = copy.deepcopy(packet)
            relocated["source_relocation"] = {"source_packet": str(original),
                "source_packet_sha256": digest(original), "mapping": []}
            # Storage provenance alone is allowed to change on relocation.
            relocated["jobs"][0]["provenance"]["frozen_files_sha256"] = "relocated binding digest"
            pp.write_text(json.dumps(relocated))
            source["packet_sha256"] = digest(pp)
            self.assertEqual(_import_source(source, {version})[1], measured)
            for field, value in [("phase", "confirmation"), ("purpose", "final confirmation")]:
                original_packet = copy.deepcopy(packet)
                original_packet["jobs"][0][field] = value
                original.write_text(json.dumps(original_packet))
                relocated["source_relocation"]["source_packet_sha256"] = digest(original)
                pp.write_text(json.dumps(relocated))
                source["packet_sha256"] = digest(pp)
                with self.assertRaisesRegex(ValueError, "semantics/phase/purpose"):
                    _import_source(source, {version})

            # A relocation cannot silently discard even unchanged source files.
            original_code, snapshot_code = root / "code.py", root / "snapshot.py"
            original_code.write_text("# CPU-only frozen execution source\n")
            snapshot_code.write_bytes(original_code.read_bytes())
            packet["bindings"] = [{"path": str(original_code), "sha256": digest(original_code)}]
            packet["frozen_files_sha256"] = quality_digest(packet["bindings"])
            original.write_text(json.dumps(packet))
            relocated = copy.deepcopy(packet)
            relocated["bindings"] = [{"path": str(snapshot_code), "sha256": digest(snapshot_code)}]
            relocated["frozen_files_sha256"] = quality_digest(relocated["bindings"])
            relocated["source_relocation"] = {"source_packet": str(original),
                "source_packet_sha256": digest(original), "mapping": [{"original": str(original_code),
                    "snapshot": str(snapshot_code), "sha256": digest(snapshot_code)}]}
            pp.write_text(json.dumps(relocated))
            source["packet_sha256"] = digest(pp)
            self.assertEqual(_import_source(source, {version})[1], measured)
            relocated["bindings"] = []
            relocated["frozen_files_sha256"] = quality_digest([])
            pp.write_text(json.dumps(relocated))
            source["packet_sha256"] = digest(pp)
            with self.assertRaisesRegex(ValueError, "dropped, added"):
                _import_source(source, {version})

    def test_protocol_prevents_posthoc_threshold_changes(self):
        config, imported, sources = calibration_fixture()
        protocol = {"base_problem_sha256": __import__("gsdeploy.quality", fromlist=["quality_digest"]).quality_digest(config.as_dict()),
                    "training_ids": [str(i) for i in range(6)], "validation_ids": ["6", "7"],
                    "thresholds": {"invalid": 100}, "sha256": "not a frozen protocol"}
        with self.assertRaisesRegex(ValueError, "declared"):
            fit_calibrated_surrogate(config, sources, [str(i) for i in range(6)], ["6", "7"], ["fixture"], protocol=protocol)


if __name__ == "__main__":
    unittest.main()
