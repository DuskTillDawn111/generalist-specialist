"""CPU-only transfer identity/activation tests; native importer explicitly mocked."""
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gsdeploy.config import SystemConfig
from gsdeploy.evidence import digest
from gsdeploy.prediction import predict
from gsdeploy.replay_transfer import (MODE, PARAMETERS, activate_transfer, declare_validation,
    predict_validation, prepare_transfer, seal, validate_transfer, validated_transfer_spec)
from gsdeploy.static_study import _job
from gsdeploy.surrogate_bounds import surrogate_bounds
from gsdeploy.trace import expand_routes, make_trace
from gsdeploy.routing import Router
from gsdeploy.quality import quality_digest
from gsdeploy.calibration_v2 import input_service_ms, replay_input_ids
from tests.test_calibrated_replay_v2 import fitted_replay_fixture
from tests.test_calibrated_search_bounds_v6 import context


class ReplayTransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = fitted_replay_fixture(quality_mode="per_sample_replay")
        self.source = self.root / "source.json"
        self.source.write_text(json.dumps(self.original.as_dict()))
        self.config = SystemConfig(**prepare_transfer(self.source, {"text": .5, "image": .35}))
        self.x = {"generalist": 1, "specialist": 2}
        self.rows = {t: {"generalist": q / 3, "specialist": q * 2 / 3} for t, q in self.config.workload.items()}

    def fixture(self, *, fail=False):
        c = self.config
        workers = {m: {"command": ["python"], "env": {}} for m in c.catalog["models"]}
        jobs = []
        for seed in (61, 62):
            job = _job(replace(c, horizon_seconds=600, random_seed=seed), workers,
                {"deployment": self.x, "routing": self.rows}, "synthetic_transfer_holdout", "calibration", seed, [], 30)
            job["purpose"] = "replay_transfer_independent_validation"
            jobs.append(job)
        packet_path = self.root / "packet.json"
        packet_path.write_text(json.dumps({"config": c.as_dict(), "workers": workers, "jobs": jobs,
            "bindings": [], "frozen_files_sha256": quality_digest([])}))
        protocol = declare_validation(c, packet_path)
        imported, sources = {}, []
        for job in jobs:
            seed = job["seed"]
            p = predict_validation(c, self.x, self.rows, seed=seed)
            self.assertTrue(p["constraints_eligible"])
            measured = {"U": p["diagnostic_partial_U"], "eligible": True,
                        **{t+"_norm": p[t+"_norm"] for t in "QTLPME"}}
            if fail:
                measured["U"] -= .2
            aliases, routes = expand_routes(self.x, self.rows)
            router, records = Router(routes), []
            trace = make_trace(c.workload, 600, seed, c.arrival_process)
            for (_, task), sample in zip(trace, replay_input_ids(c.catalog, trace, seed)):
                m = aliases[router.choose(task)]
                records.append({"worker_latency_ms": input_service_ms(c.catalog[MODE], m, task, self.x[m], sample)})
            native = c.as_dict()
            raw = {"config": native, "experiment_job_id": job["id"], "experiment_phase": "calibration",
                "executed_measurement_id": f"synthetic{seed}", "plan": job["plan"],
                "measurement_window": {"duration_seconds": 600}, "transport_config": {"random_seed": seed},
                "request_records": records}
            path = self.root / f"raw{seed}.json"
            path.write_text(json.dumps(raw))
            source = {"id": job["id"], "path": str(path), "sha256": digest(path),
                "packet_path": str(packet_path), "packet_sha256": digest(packet_path)}
            sources.append(source)
            imported[job["id"]] = (raw, measured, source)
        return protocol, sources, imported

    def report(self, *, fail=False):
        protocol, sources, imported = self.fixture(fail=fail)
        with patch("gsdeploy.replay_transfer._import_source", side_effect=lambda source, versions: imported[source["id"]]):
            report = validate_transfer(self.config, protocol, sources)
        p, r = self.root / "protocol.json", self.root / "report.json"
        p.write_text(json.dumps(protocol)); r.write_text(json.dumps(report))
        return SystemConfig(**activate_transfer(self.config, p, r)), report, r

    def test_unvalidated_refuses_formal_but_diagnostic_replays_exact_input_math(self):
        spec = self.config.catalog[MODE]
        self.assertEqual(self.config.catalog["calibrated_replay_v2"], self.original.catalog["calibrated_replay_v2"])
        for key in PARAMETERS:
            self.assertEqual(spec[key], self.original.catalog["calibrated_replay_v2"][key])
        self.assertEqual(spec["status"], "TRANSFER_UNVALIDATED")
        with self.assertRaisesRegex(ValueError, "TRANSFER_NOT_VALIDATED"):
            validated_transfer_spec(self.config.catalog, self.config.workload)
        self.assertIsNone(surrogate_bounds(self.config.catalog, self.config.workload, **context(self.config)))
        p = predict_validation(self.config, self.x, self.rows)
        self.assertIsNone(p["U"]); self.assertFalse(p["eligible"])
        self.assertIsNotNone(p["diagnostic_partial_U"])

    def test_independent_validation_activates_without_refitting_and_bounds_are_valid(self):
        c, report, _ = self.report()
        self.assertEqual(report["status"], "VALIDATED_ON_NEW_WORKLOAD")
        self.assertFalse(report["parameters_fitted_on_new_runs"])
        spec = validated_transfer_spec(c.catalog, c.workload)
        self.assertEqual(spec["power_coefficients"], self.original.catalog["calibrated_replay_v2"]["power_coefficients"])
        bounds = surrogate_bounds(c.catalog, c.workload, **context(c))
        self.assertIsNotNone(bounds)
        p = predict(c.catalog, self.x, self.rows, c.workload, c.gpu_memory_mib, c.latency_reference_ms, c.weights, **{k:v for k,v in context(c).items() if k not in {'search_policy','milp_capacity_witness'}})
        self.assertTrue(p["eligible"])
        diagnostic = c.validate_plan(self.x,self.rows)
        self.assertTrue(diagnostic["constraints_valid"])
        self.assertAlmostEqual(diagnostic["safe_memory_mib"],p["safe_memory_mib"],places=12)
        from gsdeploy.optimizer import UtilityOptimizer
        from gsdeploy.milp import JointMilpOptimizer
        from tests.test_calibrated_search_bounds_v6 import optimizer
        dfs = optimizer(UtilityOptimizer, c)
        milp = optimizer(JointMilpOptimizer, c)
        memory = sum(milp.state_memory(m,n) for m,n in self.x.items())
        for partial in ({}, {"text":self.rows["text"]}, self.rows):
            self.assertGreaterEqual(dfs.upper(self.x, partial, memory)+1e-9,p["U"])

    def test_failed_validation_is_preserved_and_rejected(self):
        c, report, _ = self.report(fail=True)
        self.assertEqual(report["status"], "TRANSFER_VALIDATION_FAILED")
        self.assertIsNone(surrogate_bounds(c.catalog,c.workload,**context(c)))
        with self.assertRaisesRegex(ValueError,"TRANSFER_NOT_VALIDATED"):
            validated_transfer_spec(c.catalog,c.workload)

    def test_parameter_workload_quality_report_and_source_tampering_rejected(self):
        c, _, report_path = self.report()
        for name in ("parameter", "quality", "workload", "claim_without_receipts", "code"):
            data, rates = copy.deepcopy(c.catalog), dict(c.workload)
            spec = data[MODE]
            if name == "parameter":
                spec["service_multiplier"] *= 1.01
            elif name == "quality":
                data["quality"]["unbound"] = True
            elif name == "workload":
                rates["text"] += .01
            elif name == "claim_without_receipts":
                del spec["validation_report"]
            else:
                spec["code_bindings"][0]["sha256"] = "changed"
            spec["version_sha256"] = seal(spec,"version_sha256")
            with self.assertRaises((ValueError,KeyError)):
                validated_transfer_spec(data,rates)
        content = report_path.read_text();report_path.write_text(content + " ")
        with self.assertRaisesRegex(ValueError,"RECEIPT_CHANGED"):
            validated_transfer_spec(c.catalog,c.workload)
        report_path.write_text(content)
        self.source.write_text(self.source.read_text()+" ")
        with self.assertRaisesRegex(ValueError,"RECEIPT_CHANGED"):
            validated_transfer_spec(c.catalog,c.workload)

    def test_validation_purpose_and_seed_roles_cannot_be_relabelled(self):
        protocol, sources, _ = self.fixture()
        packet = json.loads(Path(protocol["packet"]["path"]).read_text())
        packet["jobs"][0]["purpose"] = "new_load_exploratory_development_not_final_confirmation"
        path = self.root / "wrong-purpose.json";path.write_text(json.dumps(packet))
        with self.assertRaisesRegex(ValueError,"JOB_CONTRACT_CHANGED"):
            declare_validation(self.config,path)
        with self.assertRaisesRegex(ValueError,"SEEDS_MUST_BE_DISJOINT"):
            prepare_transfer(self.source,self.config.workload,validation_seeds=(51,62))

    def test_private_search_session_requires_new_validation_and_preserves_scores(self):
        from gsdeploy.search_session import SearchSession
        root = Path(__file__).resolve().parents[1]
        with self.assertRaisesRegex(ValueError, "TRANSFER_NOT_VALIDATED"):
            SearchSession(root,self.config.as_dict())
        c, _, _ = self.report()
        with SearchSession(root,c.as_dict()) as session:
            p = session.predict(self.x,self.rows)
            self.assertTrue(p["eligible"])
            self.assertIsNotNone(session.optimizer("dfs").surrogate)


if __name__ == "__main__":
    unittest.main()
