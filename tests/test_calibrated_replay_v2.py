"""Independent v2 per-input CPU fixtures; no GPU or real calibration fitting."""
import copy
from contextlib import contextmanager
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gsdeploy.calibration_v2 import (MODE, _sealed, declare_replay_protocol, fit_calibrated_replay,
    freeze_input_baselines, input_service_ms, replay_input_ids, replay_path_statistics,
    service_lower_ms, validated_replay_spec)
from gsdeploy.config import SystemConfig
from gsdeploy.evidence import digest
from gsdeploy.metrics import mean, percentile
from gsdeploy.prediction import components, predict
from gsdeploy.quality import predict_quality, quality_digest
from gsdeploy.routing import Router
from gsdeploy.static_study import _job
from gsdeploy.trace import expand_routes, make_trace
from tests.test_static_study_workflow import config_fixture


@contextmanager
def replay_fixture(*, quality_mode="aggregate_proxy"):
    """Yield (base_config, frozen, sources, protocol, imported); 2 tasks, n1/n2."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        base = config_fixture()
        data = copy.deepcopy(base.catalog)
        for model in data["models"]:
            data["models"][model].update(supported_tasks=["text", "image"], executable_tasks=["text", "image"],
                                         service_time_ms={"text": 150, "image": 150})
            data["capacity_factors"][model]["2"] = 2
            data["memory_table_mib"][model]["2"] = 1.8
            data["quality"]["paths"][model]["image"] = copy.deepcopy(data["quality"]["paths"][model]["text"])
            data["quality"]["paths"][model]["image"]["metric"] = "accuracy"
        data["quality"]["mode"] = quality_mode
        data["quality"]["input_order"]["image"] = ["a", "b"]
        data["task_replica_profiles"] = {}
        for model in data["models"]:
            for task in ("text", "image"):
                for n in (1, 2):
                    scale = (1.2 if n == 2 else 1) * (1.1 if model == "specialist" else 1)
                    path = root / f"{model}-{task}-{n}.jsonl"
                    records = [{"task": task, "success": True, "sample_id": sample,
                        "protocol_sha256": "CPU fixture", "request_sha256": f"request-{model}-{task}-{sample}",
                        "service_ms": value * scale, "replica": replica}
                        for sample, value in [("a", 60.), ("b", 240.)] for replica in range(n)]
                    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
                    data["task_replica_profiles"].setdefault(model, {}).setdefault(task, {})[str(n)] = {
                        "mean_service_ms": 150 * scale, "raw_service_samples": [str(path)]}
        base = replace(base, catalog=data, workload={"text": .4, "image": .3},
                       latency_reference_ms={"text": 1000, "image": 1000})
        frozen = freeze_input_baselines(base)
        workers = {m: {"command": ["python"], "env": {}} for m in data["models"]}
        imported, sources, declarations = {}, [], []
        for i, share in enumerate([0., 1., .2, .4, .6, .8, .25, .75, .3, .7]):
            seed = 42 if i < 8 else 46 + i - 8
            x = {m: (2 if i in {4, 5} else 1) for m, f in [("generalist", share), ("specialist", 1 - share)] if f > 0}
            routing = {task: {m: q * f for m, f in [("generalist", share), ("specialist", 1 - share)] if f > 0}
                       for task, q in base.workload.items()}
            config = replace(base, horizon_seconds=600, random_seed=seed)
            plan = {"deployment": x, "routing": routing}
            job = _job(config, workers, plan, f"CPU v2 {i}", "calibration", seed, [], 30)
            if i >= 8:
                job["purpose"] = "v2_predeclared_heldout_validation"
            packet = {"config": config.as_dict(), "workers": workers, "jobs": [job], "bindings": [],
                      "frozen_files_sha256": quality_digest([])}
            packet_path = root / f"packet-{i}.json"
            packet_path.write_text(json.dumps(packet))
            purpose = "legacy_v1_development" if i < 8 else "v2_independent_holdout"
            declaration = {"id": job["id"], "packet_path": str(packet_path),
                           "packet_sha256": digest(packet_path), "purpose": purpose}
            declarations.append(declaration)
            aliases, routes = expand_routes(x, routing)
            router, trace = Router(routes), make_trace(base.workload, 600, seed, base.arrival_process)
            samples = replay_input_ids(data, trace, seed)
            clocks, stats, records, inwindow = {alias: 0. for alias in aliases}, {t: [] for t in base.workload}, [], 0
            for index, ((offset, task), sample) in enumerate(zip(trace, samples)):
                alias = router.choose(task)
                model = aliases[alias]
                baseline = frozen["input_baselines"][model][task][str(x[model])]["samples"][sample]["median_service_ms"]
                service = baseline * 1.1
                finish = max(offset, clocks[alias]) + service / 1000
                clocks[alias] = finish
                stats[task].append((finish - offset) * 1000)
                inwindow += finish <= 600
                records.append({"model": alias, "task": task, "pool_id": sample, "worker_latency_ms": service,
                    "scheduled_time_ns": round(offset * 1e9), "completion_time_ns": round(finish * 1e9)})
            count = sum(x.values())
            watts, memory = 100 + 10 * count, 1.1 * sum(data["memory_table_mib"][m][str(n)] for m, n in x.items())
            energy = (watts + 20 * count / (config.startup_amortization_seconds or 600)) / config.energy_reference_watts
            q, _ = predict_quality(data, x, routing, base.workload, trace, seed)
            measured = components({task: {"mean_ms": mean(v), "p95_ms": percentile(v, 95)} for task, v in stats.items()},
                base.workload, base.latency_reference_ms, 1 - inwindow / len(trace), memory / base.gpu_memory_mib,
                base.weights, energy, q)
            measured["eligible"] = True
            native = config.as_dict()
            native["catalog"]["measurement_context"] = {"gpu_uuid": "CPU fixture GPU",
                "runtime_checkpoint_sha256": {m: "CPU fixture weights " + m for m in x}}
            raw = {"config": native, "executed_measurement_id": "CPU v2 condition " + str(i),
                "plan": plan, "replica_to_family": aliases, "request_records": records,
                "measurement_window": {"duration_seconds": 600}, "transport_config": {"random_seed": seed},
                "preflight": {"gpu": {"uuid": "CPU fixture GPU"}},
                "energy_evidence": {"run_joules": watts * 600, "startup_joules": 20 * count},
                "memory_evidence": {"safe_memory_mib": memory}}
            imported[job["id"]] = (raw, measured, {"id": job["id"], "kind": "CPU_ONLY_TEST_FIXTURE"})
            sources.append({**declaration, "path": str(root / f"run-{i}.json"), "sha256": "mocked CPU native import"})
        legacy = {"training_ids": [d["id"] for d in declarations[:6]], "validation_ids": [d["id"] for d in declarations[6:8]]}
        legacy["sha256"] = _sealed(legacy)
        protocol = declare_replay_protocol(base, frozen, declarations[:8], declarations[8:], legacy)
        yield base, frozen, sources, protocol, imported


def fitted_replay_fixture(*, quality_mode="aggregate_proxy"):
    """Return validated SystemConfig from actual v2 fit on explicit CPU fixtures."""
    with replay_fixture(quality_mode=quality_mode) as (base, frozen, sources, protocol, imported):
        with patch("gsdeploy.calibration_v2._import_source", side_effect=lambda source, versions, phases: imported[source["id"]]):
            config, report = fit_calibrated_replay(base, frozen, sources, protocol)
        assert report["status"] == "VALIDATED_ON_HELDOUT_CONDITIONS", report
        return SystemConfig(**config)


class CalibratedReplayV2Tests(unittest.TestCase):
    def test_real_cpu_fit_replays_variable_inputs_and_has_safe_bound(self):
        config = fitted_replay_fixture()
        spec = validated_replay_spec(config.catalog, config.workload)
        self.assertAlmostEqual(spec["service_multiplier"], 1.1)
        stats = replay_path_statistics(spec, "generalist", "text", 1)
        self.assertAlmostEqual(stats["minimum_ms"], 66.)
        self.assertAlmostEqual(stats["mean_ms"], 165.)
        self.assertEqual(service_lower_ms(config.catalog, config.workload, "generalist", "text", 1), stats["minimum_ms"])
        x = {"generalist": 2, "specialist": 1}
        routing = {task: {"generalist": q * .5, "specialist": q * .5} for task, q in config.workload.items()}
        pred = predict(config.catalog, x, routing, config.workload, config.gpu_memory_mib,
                       config.latency_reference_ms, config.weights, horizon=20, seed=48)
        self.assertTrue(pred["eligible"])
        self.assertGreater(pred["per_task"]["text"]["p95_ms"], stats["mean_ms"])
        self.assertTrue(config.validate_plan(x, routing)["constraints_valid"])

    def test_runtime_sample_order_is_route_independent_and_cycles(self):
        config = fitted_replay_fixture()
        from gsdeploy.runtime.workload import SamplePool
        trace = make_trace(config.workload, 50, 46)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "inputs.jsonl"
            path.write_text("\n".join(json.dumps({"task_type": task, "pool_id": sample})
                for task, samples in config.catalog["quality"]["input_order"].items() for sample in samples))
            pool = SamplePool(path, list(config.workload), 46)
            self.assertEqual(replay_input_ids(config.catalog, trace, 46), [pool.next(task)["pool_id"] for _, task in trace])

    def test_changed_holdout_workers_never_change_fit_or_prediction(self):
        with replay_fixture() as (base, frozen, sources, protocol, imported):
            def fit(values):
                with patch("gsdeploy.calibration_v2._import_source", side_effect=lambda source, versions, phases: values[source["id"]]):
                    return fit_calibrated_replay(base, frozen, sources, protocol)
            first, _ = fit(imported)
            changed = copy.deepcopy(imported)
            for key in protocol["validation_ids"]:
                for record in changed[key][0]["request_records"]:
                    record["worker_latency_ms"] *= 100
            second, report = fit(changed)
            self.assertEqual(report["status"], "VALIDATION_FAILED")
            for field in ["service_multiplier", "power_coefficients", "memory_multiplier", "startup_joules_per_worker"]:
                self.assertEqual(first["catalog"][MODE][field], second["catalog"][MODE][field])
            a, b = first["catalog"][MODE]["validation"]["conditions"], report["validation"]["conditions"]
            self.assertEqual([row["prediction"] for row in a], [row["prediction"] for row in b])
            raw = imported[protocol["validation_ids"][0]][0]
            p = predict(second["catalog"], raw["plan"]["deployment"], raw["plan"]["routing"], rates=base.workload, budget=base.gpu_memory_mib,
                        refs=base.latency_reference_ms, weights=base.weights)
            self.assertIsNone(p["U"])
            self.assertFalse(p["eligible"])

    def test_missing_input_path_and_changed_context_fail_closed(self):
        c = fitted_replay_fixture()
        for mutation in ("input", "context", "quality", "schema", "safe_factor", "input_value"):
            data = copy.deepcopy(c.catalog)
            spec = data[MODE]
            if mutation == "input":
                spec["input_baselines"]["generalist"]["text"]["1"]["samples"].pop("a")
                spec["version_sha256"] = _sealed(spec, "version_sha256")
            elif mutation == "context":
                data["measurement_context"]["gpu_uuid"] = "other GPU"
            elif mutation == "quality":
                data["quality"]["paths"]["generalist"]["text"]["value"] = .99
            elif mutation == "safe_factor":
                data["safe_factor"] *= .5
            elif mutation == "input_value":
                spec["input_baselines"]["generalist"]["text"]["1"]["samples"]["a"]["median_service_ms"] *= .1
                spec["version_sha256"] = _sealed(spec, "version_sha256")
            else:
                spec["schema"] = "v1 pretending to be v2"
            p = predict(data, {"generalist": 1}, {t: {"generalist": q} for t, q in c.workload.items()},
                        c.workload, c.gpu_memory_mib, c.latency_reference_ms, c.weights)
            self.assertIsNone(p["U"])
            self.assertFalse(p["eligible"])

    def test_protocol_binding_and_frozen_raw_files_are_required(self):
        with replay_fixture() as (base, frozen, sources, protocol, imported):
            altered = copy.deepcopy(protocol)
            altered["thresholds"]["utility_absolute_error"] = 999
            altered["sha256"] = _sealed(altered)
            with self.assertRaisesRegex(ValueError, "predeclared"):
                fit_calibrated_replay(base, frozen, sources, altered)
            Path(frozen["sources"][0]["path"]).write_text("changed")
            with self.assertRaisesRegex(ValueError, "file changed"):
                fit_calibrated_replay(base, frozen, sources, protocol)

    def test_missing_raw_hashes_need_profile_job_and_identical_frozen_media(self):
        with replay_fixture() as (base, frozen, sources, protocol, imported):
            data = copy.deepcopy(base.catalog)
            raw_path = Path(data["task_replica_profiles"]["generalist"]["image"]["1"]["raw_service_samples"][0])
            root = raw_path.parent
            inputs, image_samples = [], []
            for task, ids in data["quality"]["input_order"].items():
                for sample in ids:
                    media = root / (task + sample + ".png")
                    media.write_bytes((task + sample).encode())
                    value = {"id": sample, "task": task, "input": str(media), "input_sha256": digest(media)}
                    inputs.append({"task_type": task, "pool_id": sample, "sample": value})
                    if task == "image":
                        image_samples.append(value)
            pool = root / "runtime-inputs.jsonl"
            pool.write_text("\n".join(json.dumps(value) for value in inputs))
            data["quality"]["identity"]["input_pool_sha256"] = digest(pool)
            worker, job, profile = root / "worker.json", root / "job.json", root / "profile.json"
            worker_value = {"id": "CPU historical worker", "resources": {}}
            worker.write_text(json.dumps(worker_value))
            job.write_text(json.dumps({"spec": str(worker), "samples": image_samples}))
            profile.write_text(json.dumps({"status": "COMPLETE", "model": worker_value}))
            entry = data["task_replica_profiles"]["generalist"]["image"]["1"]
            entry["source"] = str(profile)
            data["quality"]["paths"]["generalist"]["image"]["source"] = str(worker)
            records = [json.loads(line) for line in raw_path.read_text().splitlines()]
            for record in records:
                record.pop("request_sha256")
                record.pop("protocol_sha256")
            raw_path.write_text("\n".join(json.dumps(record) for record in records))
            revised = replace(base, catalog=data)
            unknown = freeze_input_baselines(revised)
            self.assertFalse(unknown["input_baselines"]["generalist"]["image"]["1"]["samples"])
            linked = freeze_input_baselines(revised, pool)
            sample = linked["input_baselines"]["generalist"]["image"]["1"]["samples"]["a"]
            self.assertEqual(sample["identity_kind"], "historical_profile_input_binding")
            self.assertIsNone(sample["request_sha256"])
            self.assertFalse(sample["request_protocol_equivalence_to_current_execution_verified"])
            self.assertEqual(sample["source_binding"]["input"]["sha256"], image_samples[0]["input_sha256"])
            Path(image_samples[0]["input"]).write_bytes(b"changed media")
            with self.assertRaisesRegex(ValueError, "media changed"):
                freeze_input_baselines(revised, pool)

    def test_v2_cli_waits_without_fitting_incomplete_conditions(self):
        from gsdeploy.io import save
        from scripts.fit_static_replay_v2 import fit_from_completed
        with replay_fixture() as (base, frozen, sources, protocol, imported):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for name, value in [("base-config.json", base.as_dict()), ("protocol.json", protocol),
                                    ("frozen-input-baselines.json", frozen)]:
                    save(root / name, value)
                save(root / "manifest.json", {"files": {path.name: digest(path) for path in root.iterdir()}})
                with patch("scripts.fit_static_replay_v2.fit_calibrated_replay") as fitter:
                    result = fit_from_completed(root, root / "runs", root / "fitted")
                self.assertEqual(result["status"], "WAITING_FOR_V2_DECLARED_CONDITIONS")
                self.assertEqual(result["required_conditions"], 10)
                self.assertFalse((root / "fitted").exists())
                fitter.assert_not_called()

    def test_raw_scoring_and_baseline_contract_changes_are_rejected(self):
        with replay_fixture() as (base, frozen, sources, protocol, imported):
            changes = [("catalog", "safe_factor"), ("catalog", "models"), (None, "gpu_memory_mib"),
                       (None, "energy_reference_watts"), (None, "weights"),
                       (None, "latency_reference_ms"), (None, "startup_amortization_seconds"),
                       ("catalog", "quality")]
            for section, field in changes:
                altered = copy.deepcopy(imported)
                native = altered[protocol["development_ids"][0]][0]["config"]
                target = native[section] if section else native
                if field == "models":
                    target[field]["generalist"]["service_time_ms"]["text"] += 1
                elif field == "quality":
                    target[field]["paths"]["generalist"]["text"]["value"] = .99
                elif field in {"weights", "latency_reference_ms"}:
                    target[field][next(iter(target[field]))] += 1
                else:
                    target[field] = (target[field] or 0) + 1
                with patch("gsdeploy.calibration_v2._import_source", side_effect=lambda source, versions, phases: altered[source["id"]]):
                    with self.subTest(field=field), self.assertRaisesRegex(ValueError, "changed"):
                        fit_calibrated_replay(base, frozen, sources, protocol)


if __name__ == "__main__":
    unittest.main()
