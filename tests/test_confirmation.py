"""Independent confirmation packet/transport guards; no native or GPU execution."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gsdeploy.calibration import fit_calibrated_surrogate
from gsdeploy.config import SystemConfig
from gsdeploy.confirmation import (build_confirmation_packet, validate_confirmation_packet,
    verify_confirmation_run, run_confirmation_job, CANDIDATE_LABELS, _seal)
from gsdeploy.quality import quality_digest
from tests.test_calibrated_surrogate import calibration_fixture


def prepared(root, *, config=None):
    if config is None:
        base, imported, sources = calibration_fixture()
        with patch("gsdeploy.calibration._import_source", side_effect=lambda source, versions, phases=None: imported[source["id"]]):
            data, report = fit_calibrated_surrogate(base, sources, [str(i) for i in range(6)], ["6", "7"], ["fixture"])
        assert report["status"] == "VALIDATED_ON_HELDOUT_CONDITIONS"
        config = SystemConfig(**data)
    inputs = root / "inputs"
    inputs.mkdir()
    contract = {"formal_horizon_seconds": 1800, "workload": config.workload,
                "runtime_input_pool": str(inputs / "runtime-inputs.jsonl"),
                "seeds": {"development": [42], "confirmation": [43, 44, 45]}, "warmup": {"seconds": 30}}
    (inputs / "study-contract.json").write_text(json.dumps(contract))
    (inputs / "source-bindings.json").write_text("{}")
    workers = {m: {"command": ["python"], "env": {}} for m in config.catalog["models"]}
    source = {"inputs": str(inputs), "config": config.as_dict(), "workers": workers,
              "jobs": [], "bindings": [], "frozen_files_sha256": quality_digest([])}
    full = config.as_dict()
    def plan(share):
        fractions = {m: f for m, f in [("generalist", share), ("specialist", 1 - share)] if f > 0}
        return {"deployment": {m: 1 for m in fractions},
                "routing": {t: {m: q * f for m, f in fractions.items()} for t, q in config.workload.items()}}
    plans = [plan(share) for share in (.6, 1., .2, 1., 1., 0.)]
    results = []
    for label, plan in zip(CANDIDATE_LABELS, plans):
        problem = copy.deepcopy(full)
        if label in {"best_generalist_only", "best_specialist_only"}:
            role = "generalist" if label == "best_generalist_only" else "specialist"
            chosen = {m for m, s in config.catalog["models"].items() if s["role"] == role}
            for field in ("models", "capacity_factors", "memory_table_mib"):
                problem["catalog"][field] = {m: v for m, v in problem["catalog"][field].items() if m in chosen}
        results.append({"label": label, "search": {"problem": problem, "winner": plan,
                       "status": "OPTIMAL", "optimality_certified": True}})
    paths = [root / n for n in ("source.json", "config.json", "search.json")]
    for path, value in zip(paths, (source, full, {"config_sha256": quality_digest(full), "candidates": results})):
        path.write_text(json.dumps(value))
    packet = build_confirmation_packet(*paths, root=root)
    return packet, paths


def native_fixture(packet, job):
    config = replace(SystemConfig(**packet["config"]), random_seed=job["seed"]).as_dict()
    config["catalog"]["performance_mode"] = "task_replica_proxy"
    config["catalog"]["memory_mode"] = "standalone_proxy"
    hashes = {m: packet["execution_checkpoint_contract"]["runtime_checkpoint_sha256"][m]
              for m in job["plan"]["deployment"]}
    config["catalog"]["measurement_context"]["runtime_checkpoint_sha256"] = hashes
    identity = copy.deepcopy(job["identity"])
    identity["runtime_checkpoint_sha256"] = hashes
    return {"status": "completed", "backend": "native", "experiment_job_id": job["id"],
            "experiment_phase": "confirmation", "confirmation_packet_sha256": packet["confirmation_sha256"],
            "executed_measurement_identity": identity, "executed_measurement_id": quality_digest(identity),
            "config": config, "plan": job["plan"], "priming_policy": identity["warmup_policy"],
            "priming_size_metric": identity["warmup_size_metric"],
            "preflight": {"gpu": {"uuid": "CPU fixture GPU", "memory_mib": 32},
                          "ready": True, "gpu_checked": True, "device": "0"},
            "transport_config": {"random_seed": job["seed"], "measurement_seconds": 1800,
                "arrival_process": "deterministic_periodic", "warmup_seconds": 30, "drain_timeout_seconds": 60},
            "measurement_window": {"trace_sha256": identity["trace_sha256"], "duration_seconds": 1800}}


class ConfirmationTests(unittest.TestCase):
    def test_runtime_hash_matches_job_with_real_top_level_runtime_and_worker_sources(self):
        from gsdeploy.confirmation import _prepared_runtime_hash
        from gsdeploy.static_study import _job, _file
        from tests.test_static_study_workflow import config_fixture
        root = Path(__file__).resolve().parents[1]
        config = config_fixture()
        workers = {"generalist": {"command": ["python"], "env": {}}}
        plan = {"deployment": {"generalist": 1}, "routing": {"text": {"generalist": 1}}}
        # Include actual files from all three categories rather than an empty
        # temporary source root, which cannot reveal a missing top-level file.
        bindings = [_file(root / path) for path in ("gsdeploy/energy.py", "gsdeploy/runtime/protocol.py",
                                                    "scripts/static_study_worker.py")]
        job = _job(config, workers, plan, "runtime coverage", "confirmation", 43, bindings, 30)
        packet = {"identity_runtime_root": str(root), "bindings": bindings, "workers": workers}
        self.assertEqual(_prepared_runtime_hash(packet, plan["deployment"]), job["identity"]["runtime_code_sha256"])
        packet["bindings"] = bindings[1:]
        self.assertNotEqual(_prepared_runtime_hash(packet, plan["deployment"]), job["identity"]["runtime_code_sha256"])

    def _execute_entry_collects_with_proxy_then_verifies(self, config=None):
        # Exercise real execute_job serialization; only model binding, GPU UUID
        # query and transport inference are replaced with explicit CPU fixtures.
        with tempfile.TemporaryDirectory() as temporary, patch("gsdeploy.evaluation.evaluate_run", return_value={
                "utility": {"eligible": True, "energy_verified": True, "U": -.5}}):
            root = Path(temporary)
            packet, _ = prepared(root, config=config)
            job = packet["jobs"][0]
            bound = replace(SystemConfig(**packet["config"]), random_seed=job["seed"]).as_dict()
            bound["catalog"]["measurement_context"]["runtime_checkpoint_sha256"] = {
                m: packet["execution_checkpoint_contract"]["runtime_checkpoint_sha256"][m]
                for m in job["plan"]["deployment"]}
            calls = []
            def transport(config, plan, *args, priming_policy=None, **options):
                calls.append((config.catalog["performance_mode"], config.catalog["memory_mode"],
                              options["duration"], options["seed"], priming_policy))
                raw = native_fixture(packet, job)
                raw["config"] = config.as_dict()
                # The actual execution function must add these receipts itself.
                raw.pop("confirmation_packet_sha256")
                raw.pop("experiment_phase")
                return raw
            with patch("gsdeploy.static_study.bind_native_job", return_value=(bound, root / "mock-manifest.json", packet["workers"])), \
                    patch("subprocess.check_output", return_value="CPU fixture GPU"), \
                    patch("gsdeploy.runtime.runner.run", side_effect=transport) as mocked:
                # inspect.signature must see the real protocol boundary.
                mocked.__signature__ = __import__("inspect").signature(transport)
                result = run_confirmation_job(packet, job["id"], root / "runs", execute=True)
            self.assertFalse(result["resumed"])
            self.assertEqual(calls, [("task_replica_proxy", "standalone_proxy", 1800, 43, "largest_three_per_task")])
            raw, utility = verify_confirmation_run(packet, job, result["path"])
            self.assertEqual(raw["experiment_phase"], "confirmation")
            self.assertEqual(raw["confirmation_packet_sha256"], packet["confirmation_sha256"])
            self.assertTrue(utility["eligible"])

    def test_execute_entry_collects_with_proxy_then_verifies_the_independent_receipt(self):
        self._execute_entry_collects_with_proxy_then_verifies()

    def test_v2_prepare_execute_cpu_mock_and_verify_preserve_frozen_replay(self):
        from tests.test_calibrated_replay_v2 import fitted_replay_fixture
        config = fitted_replay_fixture()
        self.assertEqual(config.catalog["performance_mode"], "calibrated_replay_v2")
        self._execute_entry_collects_with_proxy_then_verifies(config)

    def test_v2_failed_validation_wrong_snapshot_and_version_cannot_confirm(self):
        from gsdeploy.calibration_v2 import MODE, _sealed
        from tests.test_calibrated_replay_v2 import fitted_replay_fixture
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet, paths = prepared(root, config=fitted_replay_fixture())
            self.assertTrue(validate_confirmation_packet(packet)["valid"])
            for field, value in (("identity_runtime_root", str(root / "other-runtime")),
                                 ("surrogate_version_sha256", "different predictor version")):
                changed = copy.deepcopy(packet)
                changed[field] = value
                changed["confirmation_sha256"] = _seal(changed)
                self.assertFalse(validate_confirmation_packet(changed)["valid"], field)
            config = json.loads(paths[1].read_text())
            spec = config["catalog"][MODE]
            spec["status"] = "VALIDATION_FAILED"
            spec["version_sha256"] = _sealed(spec, "version_sha256")
            paths[1].write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "V2_REPLAY_NOT_VALIDATED"):
                build_confirmation_packet(*paths, root=root)

    def test_additional_checkpoint_receipt_binds_extrapolated_model_without_refitting(self):
        with tempfile.TemporaryDirectory() as temporary, patch("gsdeploy.evaluation.evaluate_run", return_value={
                "utility": {"eligible": True, "energy_verified": True, "U": -.5}}):
            root = Path(temporary)
            _, paths = prepared(root)
            config = json.loads(paths[1].read_text())
            # Synthetic representation of a pool member absent from calibration.
            spec = config["catalog"]["calibrated_surrogate"]
            config["catalog"]["measurement_context"]["runtime_checkpoint_sha256"].pop("generalist")
            spec["measurement_context"]["runtime_checkpoint_sha256"].pop("generalist")
            spec["version_sha256"] = quality_digest({k: v for k, v in spec.items() if k != "version_sha256"})
            search = json.loads(paths[2].read_text())
            search["config_sha256"] = quality_digest(config)
            for record in search["candidates"]:
                record["search"]["problem"]["catalog"]["measurement_context"] = copy.deepcopy(config["catalog"]["measurement_context"])
                record["search"]["problem"]["catalog"]["calibrated_surrogate"] = copy.deepcopy(spec)
            paths[1].write_text(json.dumps(config))
            paths[2].write_text(json.dumps(search))
            with self.assertRaisesRegex(ValueError, "WAITING_CONFIRMATION_CHECKPOINT_BINDING"):
                build_confirmation_packet(*paths, root=root)
            from gsdeploy.evidence import digest
            manifest = root / "additional-checkpoint-manifest.json"
            manifest.write_text(json.dumps({"models": {"generalist": {"checkpoint": {"sha256": "new CPU weights"}}}}))
            report = root / "checkpoint-report.json"
            report.write_text(json.dumps({"models": {"generalist": {"checkpoint": {"sha256": "new CPU weights"},
                "manifest": str(manifest), "manifest_sha256": digest(manifest)}}}))
            packet = build_confirmation_packet(*paths, root=root, checkpoint_bindings_path=report)
            self.assertEqual(packet["config"], config)
            self.assertEqual(packet["surrogate_version_sha256"], spec["version_sha256"])
            self.assertEqual(packet["execution_checkpoint_contract"]["additional_cpu_bound_models"], ["generalist"])
            self.assertTrue(validate_confirmation_packet(packet)["valid"])
            job = packet["jobs"][0]
            run_path = root / "additional-run.json"
            run_path.write_text(json.dumps(native_fixture(packet, job)))
            self.assertTrue(verify_confirmation_run(packet, job, run_path)[1]["eligible"])
            manifest.write_text("{}")
            self.assertFalse(validate_confirmation_packet(packet)["valid"])

    def test_preparation_deduplicates_roles_but_preserves_independent_seed_blocks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet, _ = prepared(root)
            self.assertTrue(validate_confirmation_packet(packet)["valid"])
            self.assertEqual(len(packet["candidates"]), 4)
            self.assertEqual(len(packet["jobs"]), 12)
            self.assertEqual([j["seed"] for j in packet["jobs"]], [43] * 4 + [44] * 4 + [45] * 4)
            self.assertEqual(len({j["id"] for j in packet["jobs"]}), 12)
            for seed in (43, 44, 45):
                self.assertEqual(len({j["identity"]["trace_sha256"] for j in packet["jobs"] if j["seed"] == seed}), 1)
            self.assertFalse(packet["gpu_started"])
            with self.assertRaisesRegex(ValueError, "explicit"):
                run_confirmation_job(packet, packet["jobs"][0]["id"], root / "runs")
            self.assertFalse((root / "runs").exists())

    def test_phase_and_seed_tamper_rejected_even_if_manifest_is_resealed(self):
        with tempfile.TemporaryDirectory() as temporary:
            packet, _ = prepared(Path(temporary))
            changed = copy.deepcopy(packet)
            changed["jobs"][0]["phase"] = "calibration"
            changed["confirmation_sha256"] = _seal(changed)
            self.assertFalse(validate_confirmation_packet(changed)["valid"])
            for field, value in {"warmup_seconds": 0, "drain_seconds": 999, "warmup_policy": "none",
                                 "runtime_code_sha256": "other runtime", "trace_sha256": "other trace"}.items():
                changed = copy.deepcopy(packet)
                job = changed["jobs"][0]
                job["identity"][field] = value
                job["id"] = quality_digest(job["identity"])
                changed["confirmation_sha256"] = _seal(changed)
                self.assertFalse(validate_confirmation_packet(changed)["valid"], field)
            changed = copy.deepcopy(packet)
            changed["jobs"][0]["seed"] = 42
            changed["confirmation_sha256"] = _seal(changed)
            self.assertFalse(validate_confirmation_packet(changed)["valid"])

    def test_source_search_file_mutation_invalidates_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary:
            packet, paths = prepared(Path(temporary))
            paths[2].write_text("{}")
            self.assertFalse(validate_confirmation_packet(packet)["valid"])

    def test_failed_model_or_missing_comparison_role_cannot_prepare(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet, paths = prepared(root)
            search = json.loads(paths[2].read_text())
            search["candidates"].pop()
            paths[2].write_text(json.dumps(search))
            with self.assertRaisesRegex(ValueError, "six comparison"):
                build_confirmation_packet(*paths, root=root)
            config = json.loads(paths[1].read_text())
            spec = config["catalog"]["calibrated_surrogate"]
            spec["status"] = "VALIDATION_FAILED"
            spec["version_sha256"] = quality_digest({k: v for k, v in spec.items() if k != "version_sha256"})
            paths[1].write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "NOT_VALIDATED"):
                build_confirmation_packet(*paths, root=root)

    def test_native_verification_and_resume_reject_wrong_seed_gpu_and_reused_run(self):
        with tempfile.TemporaryDirectory() as temporary, patch("gsdeploy.evaluation.evaluate_run", return_value={
                "utility": {"eligible": True, "energy_verified": True, "U": -.5}}):
            root = Path(temporary)
            packet, _ = prepared(root)
            job = packet["jobs"][0]
            raw = native_fixture(packet, job)
            path = root / "run.json"
            path.write_text(json.dumps(raw))
            self.assertEqual(verify_confirmation_run(packet, job, path)[1]["U"], -.5)
            for field in ("seed", "gpu", "reused", "weights", "phase", "packet", "mode", "nan_memory"):
                changed = copy.deepcopy(raw)
                if field == "seed":
                    changed["transport_config"]["random_seed"] = 42
                elif field == "gpu":
                    changed["preflight"]["gpu"]["uuid"] = "another GPU"
                elif field == "weights":
                    changed["config"]["catalog"]["measurement_context"]["runtime_checkpoint_sha256"]["generalist"] = "changed"
                elif field == "phase":
                    changed["experiment_phase"] = "calibration"
                elif field == "packet":
                    changed["confirmation_packet_sha256"] = "another packet"
                elif field == "mode":
                    changed["config"]["catalog"]["performance_mode"] = "independent"
                elif field == "nan_memory":
                    changed["preflight"]["gpu"]["memory_mib"] = float("nan")
                else:
                    changed["experiment_job_id"] = packet["jobs"][4]["id"]
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    verify_confirmation_run(packet, job, path)
            with patch("gsdeploy.confirmation.execute_job") as execute:
                for job in (packet["jobs"][0], packet["jobs"][4]):
                    directory = root / "runs" / (job["labels"][0] + "-seed" + str(job["seed"]) + "-" + job["id"][:12])
                    directory.mkdir(parents=True)
                    (directory / "run.json").write_text(json.dumps(native_fixture(packet, job)))
                    result = run_confirmation_job(packet, job["id"], root / "runs", execute=True)
                    self.assertTrue(result["resumed"])
                    self.assertIn("seed" + str(job["seed"]), result["path"])
                execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
