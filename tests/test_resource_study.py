"""Prospective six-term resource packets and causal CPU policy integration."""

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from gsdeploy import DeploymentSystem, SystemConfig
from gsdeploy.cli import main as cli_main
from gsdeploy.resource_study import build_packet, validate_packet, main
from gsdeploy.resources import identity
from gsdeploy.runtime.runner import run
from tests.test_resource_control import resources
from tools.resource_control_20261024.policy import Dispatcher, ResourceBoundProvider


ROOT = Path(__file__).resolve().parents[1]


class StudyFixture(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        capabilities = {"text": ["text"], "generalist_2": ["text", "image", "asr"], "ocr": ["ocr"]}
        catalog = {"capability_contract": "verified_executable_v1", "safe_factor": 1,
                   "models": {m: {"role": "generalist" if m == "generalist_2" else "specialist",
                                   "supported_tasks": ts, "executable_tasks": ts,
                                   "service_time_ms": {t: 1 for t in ts}} for m, ts in capabilities.items()},
                   "capacity_factors": {m: {"0": 0, "1": 1, "2": 2} for m in capabilities},
                   "memory_table_mib": {m: {"0": 0, "1": 1, "2": 2} for m in capabilities}}
        self.system = {"catalog": catalog, "workload": {t: 10 for t in ("text", "image", "asr", "ocr")},
                       "gpu_memory_mib": 100, "latency_reference_ms": {t: 100 for t in ("text", "image", "asr", "ocr")},
                       "utility_schema": "six_term_quality_v1", "weights": {k: 1/6 for k in "QTLPME"},
                       "horizon_seconds": 1}
        self.write("system.json", self.system)
        workers = {m: {"command": [sys.executable, "-m", "gsdeploy.backends.mock"],
                       "env": {"GS_TASKS": json.dumps(ts)}} for m, ts in capabilities.items()}
        self.write("workers.json", workers)
        self.samples = self.path / "samples.jsonl"
        self.samples.write_text("".join(json.dumps({"task_type": task, "pool_id": f"{task}-{i}",
                            "sample": {"id": f"{task}-{i}", "input": "cpu-only", "label": "fixture"}}) + "\n"
                            for task in self.system["workload"] for i in range(2)))
        self.write("objective.json", json.loads((ROOT / "configs/resource_spillover_objective_20261024.json").read_text()))
        self.design = json.loads((ROOT / "configs/resource_spillover_study_20261024.json").read_text())
        self.design.update(system_config="system.json", workers="workers.json", samples="samples.jsonl",
                           objective="objective.json", workload_rps=self.system["workload"],
                           latency_reference_ms=self.system["latency_reference_ms"], duration_seconds=1,
                           warmup_seconds=0, drain_seconds=1, arrival_process="deterministic_periodic",
                           background_routing={"image": {"generalist_2": 10}, "asr": {"generalist_2": 10}, "ocr": {"ocr": 10}})
        self.design_path = self.path / "design.json"
        self.write("design.json", self.design)

    def write(self, name, value):
        (self.path / name).write_text(json.dumps(value))

    def packet(self):
        return build_packet(self.design_path)


class PacketTests(StudyFixture):
    def test_prepare_and_independent_reconstruction_never_launch(self):
        with patch("subprocess.Popen", side_effect=AssertionError("No workers")), \
                patch("subprocess.run", side_effect=AssertionError("No GPU checks")):
            packet = self.packet()
            self.assertTrue(validate_packet(packet)["valid"])
            self.assertEqual(packet, self.packet())
        self.assertEqual(len(packet["jobs"]), 16)
        self.assertFalse(packet["GPU_started"])
        self.assertFalse(packet["native_execution_enabled"])
        self.assertTrue(all(job["U"] is None for job in packet["jobs"]))
        self.assertTrue(all(job["resource_prediction"]["service"] is None for job in packet["jobs"]))

    def test_pairs_share_exact_cohort_resources_and_weights(self):
        packet = self.packet(); pairs = {}
        for job in packet["jobs"]:
            pairs.setdefault(job["pair_id"], []).append(job)
        self.assertEqual(len(pairs), 8)
        for off, on in pairs.values():
            self.assertEqual(off["cohort"], on["cohort"])
            self.assertEqual(off["execution_context_sha256"], on["execution_context_sha256"])
            self.assertEqual(off["weights"], {k: 1/6 for k in "QTLPME"})
            self.assertEqual(off["candidate"]["resource_configuration"], on["candidate"]["resource_configuration"])
            self.assertNotEqual(off["candidate_sha256"], on["candidate_sha256"])
            self.assertEqual({off["candidate"]["routing_policy"]["allow_spill"],
                              on["candidate"]["routing_policy"]["allow_spill"]}, {True, False})
            self.assertEqual(set(off["nominal_no_spill_plan"]["routing"]), set(self.system["workload"]))

    def test_second_seed_reverses_order_without_dropping_controls(self):
        jobs = self.packet()["jobs"]
        middle = len(jobs) // 2
        first = [(j["variant"], j["candidate"]["routing_policy"]["allow_spill"]) for j in jobs[:middle]]
        second = [(j["variant"], j["candidate"]["routing_policy"]["allow_spill"]) for j in jobs[middle:]]
        self.assertEqual(first, list(reversed(second)))

    def test_self_rehash_cannot_hide_added_result_or_changed_control(self):
        for field, value in (("U", 0), ("execution_status", "completed")):
            packet = self.packet(); packet["jobs"][0][field] = value
            packet["packet_sha256"] = identity({k: v for k, v in packet.items() if k != "packet_sha256"})
            self.assertFalse(validate_packet(packet)["valid"])

    def test_modified_input_bytes_are_rejected(self):
        packet = self.packet()
        self.samples.write_text(self.samples.read_text() + "\n")
        result = validate_packet(packet)
        self.assertFalse(result["valid"])
        self.assertIn("SHA256 mismatch", result["issues"][0])

    def test_modified_source_receipt_is_rejected(self):
        packet = self.packet(); packet["sources"][0]["sha256"] = "0" * 64
        self.assertFalse(validate_packet(packet)["valid"])

    def test_p95_zero_weight_and_stability_change_are_rejected(self):
        objective = json.loads((self.path / "objective.json").read_text())
        for update in ({"weights": {k: (0 if k == "P" else .2) for k in "QTLPME"}},
                       {"stability_policy": "relaxed"}, {"absolute_quality_gate": True}):
            changed = {**objective, **update}; self.write("objective.json", changed)
            with self.assertRaisesRegex(ValueError, "six equal terms"):
                self.packet()

    def test_missing_background_task_is_rejected(self):
        self.design["background_routing"].pop("asr"); self.write("design.json", self.design)
        with self.assertRaisesRegex(ValueError, "Every background"):
            self.packet()

    def test_wrong_or_nonconserving_background_route_is_rejected(self):
        for row in ({"not_deployed": 10}, {"generalist_2": 9}):
            self.design["background_routing"]["asr"] = row; self.write("design.json", self.design)
            with self.assertRaises(ValueError):
                self.packet()

    def test_duplicate_resources_seeds_and_missing_shared_baseline_rejected(self):
        original = copy.deepcopy(self.design)
        for mutate in (lambda d: d["resource_variants"].update(duplicate=d["resource_variants"]["shared"]),
                       lambda d: d.update(seeds=[501, 501]),
                       lambda d: d["resource_variants"].pop("shared")):
            changed = copy.deepcopy(original); mutate(changed); self.write("design.json", changed)
            with self.assertRaises(ValueError):
                self.packet()

    def test_shared_baseline_not_mislabeled_as_verified_isolation(self):
        packet = self.packet()
        for variant in packet["resource_variants"].values():
            for worker in variant["prepared_worker_registry"].values():
                self.assertFalse(worker["resource_binding"]["effective_isolation_verified"])
        self.assertFalse(packet["comparison_contract"]["physical_optimality_certified"])

    def test_sparse_cohort_does_not_fill_missing_tasks_with_zero(self):
        self.design["duration_seconds"] = .00001; self.write("design.json", self.design)
        with self.assertRaisesRegex(ValueError, "observations"):
            self.packet()

    def test_cli_prepares_validates_and_refuses_overwrite(self):
        output = str(self.path / "packet.json")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli_main(["resource-study", "prepare", "--design", str(self.design_path), "--output", output]), 0)
            self.assertEqual(cli_main(["resource-study", "validate", "--packet", output]), 0)
            before = Path(output).read_bytes()
            self.assertEqual(main(["prepare", "--design", str(self.design_path), "--output", output]), 2)
            self.assertEqual(Path(output).read_bytes(), before)


class PolicyTests(StudyFixture):
    def dispatcher(self, packet, allow_spill=True, variant="shared", **kwargs):
        job = next(j for j in packet["jobs"] if j["variant"] == variant and
                   j["candidate"]["routing_policy"]["allow_spill"] is allow_spill)
        return Dispatcher(packet, job["id"], **kwargs)

    def test_wrong_resource_provider_not_called(self):
        class Wrong:
            execution_context_sha256 = "old-uncontrolled-model"
            def service(self, *args):
                raise AssertionError("Old provider must not be used")
        provider = ResourceBoundProvider(Wrong(), "new-resource-context")
        self.assertFalse(provider.service(None)["known"])
        self.assertIsNone(provider.power(None)["watts"])

    def test_p95_changes_policy_score_even_when_mean_latency_stays_equal(self):
        packet = self.packet()
        quality = {m: {t: (.1 if t in {"text", "image"} else
                          {"basis": "expected_per_request_totals", "numerator": 1,
                           "denominator": 10, "evidence": "CPU fixture"})
                      for t in model["supported_tasks"]}
                   for m, model in packet["effective_system"]["catalog"]["models"].items()}
        dispatcher = self.dispatcher(packet, quality_loss=quality)
        policy = dispatcher.policy; policy.start_phase(0, 1_000_000_000)
        policy.forecast_state = {"observed_peak_memory_mib": 0.}
        def score(text_latencies):
            rows = [{"request_id": str(i), "alias": "text#0", "task": "text", "pool_id": "text-0",
                     "arrival_ns": 0, "finish_ns": round(ms * 1e6)}
                    for i, ms in enumerate(text_latencies)]
            for task, alias in (("image", "generalist_2#0"), ("asr", "generalist_2#0"), ("ocr", "ocr#0")):
                rows.append({"request_id": task, "alias": alias, "task": task, "pool_id": task+"-0",
                             "arrival_ns": 0, "finish_ns": 100_000_000})
            cohort = {a: [r for r in rows if r["alias"] == a] for a in dispatcher.aliases}
            forecast = {"finishes": {r["request_id"]: r for r in rows},
                        "cohort": {r["request_id"]: r for r in rows},
                        "board_energy_joules": 0, "peak_memory_mib": 100.,
                        "future_scenario": {"arrivals": [], "cohort_sha256": "empty", "source": "cpu-fixture"},
                        "forecast_now_ns": 0, "future_arrival_count": 0,
                        "transition_count": 0, "trace_scope": "cpu-fixture",
                        "continuation_scope": "fixed-cpu-fixture",
                        "queue_diagnostics": {}, "lifecycle_scope": "cpu-fixture"}
            return policy._terms(forecast, cohort)
        even, tail = score([100, 100, 100]), score([0, 0, 300])
        self.assertAlmostEqual(even["terms"]["L"], tail["terms"]["L"])
        self.assertGreater(tail["terms"]["P"], even["terms"]["P"])
        self.assertAlmostEqual(tail["cost"] - even["cost"],
                               (tail["terms"]["P"] - even["terms"]["P"]) / 6)
        self.assertIsNone(tail["full_U"])

    def test_matching_context_delegates_without_scaling_service(self):
        class Provider:
            execution_context_sha256 = "new"
            def service(self, *args):
                return {"known": True, "service_ms": 123}
        self.assertEqual(ResourceBoundProvider(Provider(), "new").service(None)["service_ms"], 123)

    def test_unknown_predictor_falls_back_and_no_spill_control_is_strict(self):
        packet = self.packet()
        for allow in (True, False):
            dispatcher = self.dispatcher(packet, allow)
            dispatcher.start_phase(0, 1_000_000_000)
            dispatcher.set_arrival("current", 0)
            snapshot = {a: {"outstanding": []} for a in dispatcher.aliases}
            result = dispatcher.choose("text", "text-0", 0, snapshot)
            self.assertEqual(dispatcher.aliases[result["alias"]]["family"], "text")
            self.assertFalse(result["actual_spill_selected"])
            self.assertFalse(result["physical_six_term_improvement_certified"])
            self.assertEqual(dispatcher.policy.weights["P"], 1/6)

    def test_host_admission_uses_new_model_and_requires_resource_evidence(self):
        packet = self.packet()
        dispatcher = self.dispatcher(packet, variant="admit_two_generates")
        dispatcher.start_phase(0, 1_000_000_000); dispatcher.set_arrival("current", 0)
        result = dispatcher.choose("text", "text-0", 0,
                                   {a: {"outstanding": []} for a in dispatcher.aliases})
        self.assertFalse(result["actual_spill_selected"])
        reasons = [reason for row in result["audit"] for reason in row["unknown_reasons"]]
        from gsdeploy.forecasting import FullHorizonModel
        self.assertIsInstance(dispatcher.policy.model, FullHorizonModel)
        self.assertTrue(dispatcher.policy.model.gated)
        self.assertTrue(any("UNKNOWN_SHARED_SEGMENT_SERVICE" in reason for reason in reasons))

    def test_fixed_background_obligations_retained_in_both_arms(self):
        packet = self.packet()
        for allow in (False, True):
            dispatcher = self.dispatcher(packet, allow)
            dispatcher.start_phase(0, 1_000_000_000)
            snapshot = {a: {"outstanding": []} for a in dispatcher.aliases}
            for i, (task, family) in enumerate((("image", "generalist_2"), ("asr", "generalist_2"), ("ocr", "ocr"))):
                dispatcher.set_arrival(str(i), i)
                result = dispatcher.choose(task, task+"-0", i, snapshot)
                self.assertEqual(dispatcher.aliases[result["alias"]]["family"], family)
                self.assertIn(str(i), dispatcher.policy.admitted)

    def test_no_arrival_or_future_arrival_rejected(self):
        dispatcher = self.dispatcher(self.packet())
        with self.assertRaises(ValueError):
            dispatcher.choose("text", "text-0", 0, {})
        dispatcher.set_arrival("future", 1)
        with self.assertRaises(ValueError):
            dispatcher.choose("text", "text-0", 0, {})


class RuntimeResourceTests(unittest.TestCase):
    def test_mock_roundtrip_records_resources_but_never_transfers_certificate(self):
        config = SystemConfig.load(ROOT / "configs/demo.json")
        system = DeploymentSystem(config); plan = system.plan(deployment={"generalist": 1})
        shared = resources(aliases=("generalist#0",))
        raw = system.run(plan, ROOT / "examples/demo_samples.jsonl", duration=.1,
                         warmup=0, resource_configuration=shared)
        self.assertEqual(raw["status"], "completed", raw["errors"])
        self.assertEqual(raw["resource_configuration"], shared)
        self.assertFalse(raw["planning_contract"]["optimality_transferred"])
        self.assertIsNone(raw["planning_contract"]["matched_prediction"]["U"])
        self.assertIsNone(raw["evaluation"]["utility"]["U"])
        self.assertFalse(raw["evaluation"]["utility"]["resource_contract_verified"])
        self.assertIn("resource_binding", raw["worker_registry"]["generalist#0"])

    def test_native_resource_run_rejected_before_any_gpu_probe(self):
        config = SystemConfig.load(ROOT / "configs/demo.json")
        winner = {"deployment": {"generalist": 1}, "routing": {t: {"generalist": v} for t, v in config.workload.items()}}
        with patch("gsdeploy.runtime.runner.preflight", side_effect=AssertionError("No GPU probe")), \
                patch("subprocess.Popen", side_effect=AssertionError("No worker launch")):
            with self.assertRaisesRegex(ValueError, "not enabled"):
                run(config, winner, ROOT / "examples/demo_samples.jsonl", backend="native",
                    resource_configuration=resources(aliases=("generalist#0",)))


if __name__ == "__main__":
    unittest.main()
