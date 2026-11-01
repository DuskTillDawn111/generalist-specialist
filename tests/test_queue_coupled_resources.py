"""CPU checks for shared-board accounting, provenance, and supported residuals."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("coupled_resources", ROOT/"tools/queue_coupled_utility_20261023/resources.py")
r = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(r)


class ResourceContracts(unittest.TestCase):
    def fixture(self, directory):
        ns = lambda x: int(x*1e9)
        aliases = {"text#0": {"family": "text"}, "text#1": {"family": "text"}, "g#0": {"family": "generalist_2"}}
        rows = []
        for index,(alias,a,b,service) in enumerate([
            ("text#0",1.1,2.1,1000.), ("text#0",3.,5.,2000.), ("g#0",6.,7.,1000.)
        ]):
            rows.append({"status": "completed", "model": alias, "task": "text", "pool_id": "input-a",
                "request_id": str(index), "enqueued_time_ns": ns(a), "completion_time_ns": ns(b),
                "worker_response": {"service_ms": service},
                "routing_decision": {"snapshot": {x: {"outstanding": []} for x in aliases}}})
        raw = {"backend": "native", "status": "completed", "experiment_phase": "development",
            "transport_config": {"random_seed": 203},
            "policy_contract": {"aliases": aliases,"deployment":{"text":2,"generalist_2":1}},
            "energy_evidence": {"kind":"measured","scope":"whole_gpu_board","coverage_valid":True,
                "samples":[{"time_ns":ns(t),"watts":100.} for t in range(11)],
                "run_joules":800.,"startup_joules":100.,"startup_begin_time_ns":0},
            "measurement_window":{"start_time_ns":ns(1),"end_time_ns":ns(9)},
            "request_records":rows, "memory_evidence":{"measured":True,"resource_valid":True,"safe_memory_mib":1000.}}
        path = Path(directory)/"run.json"; path.write_text(json.dumps(raw))
        manifest = r.make_manifest([{"path":path,"seed":203,"phase":"development"}])
        return raw,manifest

    def model(self):
        with tempfile.TemporaryDirectory() as directory:
            raw,manifest = self.fixture(directory)
            return r.ResourceModel(r.fit_manifest(manifest))

    def test_partition_reproduces_board_energy_once(self):
        model = self.model(); deployment={"text":2,"generalist_2":1}
        result=model.energy_for_timeline(deployment,[{"duration_ms":1000,"active_counts":{"text":1}}])
        self.assertEqual(result["joules"],100.)
        self.assertEqual(model.document["run_diagnostics"][0]["window_joules"],800.)
        self.assertTrue(model.document["run_diagnostics"][0]["window_energy_partition_reproduced"])

    def test_reserved_declaration_rejected_before_open(self):
        manifest={"schema":r.MANIFEST_SCHEMA,"reserved_seeds_used":False,
            "sources":[{"path":"/does-not-exist","sha256":"abc","seed":205,"phase":"development"}]}
        with patch.object(Path,"read_bytes",side_effect=AssertionError("Must not open reserved input")):
            with self.assertRaisesRegex(ValueError,"Reserved"):
                r.fit_manifest(manifest)

    def test_sha_and_raw_seed_are_both_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            raw,manifest=self.fixture(directory); raw["transport_config"]["random_seed"]=209
            Path(manifest["sources"][0]["path"]).write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError,"SHA"):
                r.fit_manifest(manifest)
            manifest["sources"][0].update(r.sha(manifest["sources"][0]["path"]))
            with self.assertRaisesRegex(ValueError,"Reserved"):
                r.fit_manifest(manifest)

    def test_unseen_power_context_and_deployment_remain_unknown(self):
        model=self.model(); result=model.predict_power({"text":2,"generalist_2":1},{"text":2,"generalist_2":1})
        self.assertIsNone(result["watts"])
        self.assertIsNone(model.predict_power({"text":1},{"text":1})["watts"])
        result=model.energy_for_timeline({"text":2,"generalist_2":1},[
            {"duration_ms":1000,"active_counts":{"text":1}},
            {"duration_ms":1000,"active_counts":{"text":2,"generalist_2":1}}])
        self.assertIsNone(result["joules"])

    def test_context_service_and_residual_never_extrapolate(self):
        model=self.model(); args=({"text":2,"generalist_2":1},"text","text","input-a",{})
        self.assertAlmostEqual(model.predict_service(*args)["mean_ms"],1500.)
        self.assertAlmostEqual(model.predict_residual(*args,500)["mean_ms"],1000.)
        self.assertIsNone(model.predict_residual(*args,1500)["mean_ms"])
        self.assertIsNone(model.predict_residual(*args,3000)["mean_ms"])

    def test_externality_is_association_and_unseen_is_unknown(self):
        model=self.model(); d={"text":2,"generalist_2":1}
        result=model.service_externality(d,"text","text","input-a",{}, {"generalist_2":1})
        self.assertIsNone(result["service_ratio"]); self.assertFalse(result["causal"])
        self.assertFalse(result["counterfactual_certified"])

    def test_memory_range_is_not_physical_bound_or_zero_delta(self):
        prediction=self.model().predict_memory({"text":2,"generalist_2":1})
        self.assertEqual(prediction["observed_max_mib"],1000.)
        self.assertIsNone(prediction["physical_upper_bound_mib"])
        self.assertIsNone(prediction["same_residency_route_delta_mib"])
        self.assertFalse(prediction["route_invariant"])

    def test_power_gap_has_no_imputation(self):
        integral=r.PowerIntegral([{"time_ns":0,"watts":100.},{"time_ns":3_000_000_000,"watts":100.}])
        self.assertIsNone(integral.integrate(0,3_000_000_000))

    def test_future_outcomes_do_not_enter_visible_context(self):
        aliases={"a":{"family":"text"},"b":{"family":"generalist_2"}}
        snapshot={"a":{"outstanding":[{"hidden_service_ms":100000}]},"b":{"outstanding":[]}}
        self.assertEqual(r.active_counts_from_snapshot(aliases,snapshot,"a"),{"text":0,"generalist_2":0})


class ControlledProgressContracts(unittest.TestCase):
    def fixture(self,directory):
        deployment={"text":2,"generalist_2":1,"ocr":1}
        aliases={"text#0":{"family":"text","supported_tasks":["text"]},
                 "text#1":{"family":"text","supported_tasks":["text"]},
                 "g#0":{"family":"generalist_2","supported_tasks":["text","image"]},
                 "o#0":{"family":"ocr","supported_tasks":["ocr"]}}
        requests=[]; batches=[]; clock=1_000_000_000
        for seed in (301,302):
            for context,active,ms in [("S",["text#0"],100),("G",["g#0"],100),
                ("SS",["text#0","text#1"],200),("SG",["text#0","g#0"],200),
                ("SSG",["text#0","text#1","g#0"],400)]:
                bid=f"{seed}-{context}"; ids=[]
                for alias in active:
                    rid=f"{bid}-{alias}"; ids.append(rid)
                    requests.append({"request_id":rid,"model":alias,"task":"text","pool_id":"i0","status":"completed",
                        "worker_response":{"service_ms":float(ms)},
                        "actual_service_interval":{"start_time_ns":clock,"end_time_ns":clock+ms*1_000_000,
                            "start_unix_ns":clock+10**15,"end_unix_ns":clock+10**15+ms*1_000_000}})
                batches.append({"id":bid,"seed":seed,"pool_id":"i0","context":context,"aliases":active,
                    "request_ids":ids,"start_time_ns":clock,"end_time_ns":clock+ms*1_000_000,"all_completed":True})
                clock+=1_000_000_000
        raw={"schema":"gsdeploy.coupled_controlled_calibration.v1","status":"completed","errors":[],
             "backend":"native","calibration_usable":True,
             "deployment":deployment,"aliases":aliases,"request_records":requests,"batches":batches,
             "energy_evidence":{"scope":"whole_gpu_board","kind":"measured","coverage_valid":True,"samples":[{"time_ns":t*100_000_000,"watts":100.} for t in range(130)]},
             "memory_evidence":{"measured":True,"safe_memory_mib":1000.}}
        path=Path(directory)/"calibration.json";path.write_text(json.dumps(raw))
        return path,raw

    def fit(self):
        with tempfile.TemporaryDirectory() as directory:
            path,raw=self.fixture(directory)
            return r.fit_controlled(path),raw

    def test_actual_overlap_recovers_progress_and_full_rank(self):
        document,raw=self.fit(); text=document["family_models"]["text"]
        self.assertEqual(text["fit"]["design"]["rank"],4)
        self.assertEqual(document["family_models"]["generalist_2"]["fit"]["design"]["rank"],3)
        for fold in text["leave_seed_out"]:
            self.assertEqual(fold["unknown_rows"],0)
            self.assertLess(fold["held_work_max_absolute_relative_error"],1e-10)
        self.assertTrue(document["power_partition"]["reproduced"])

    def test_dynamic_provider_reprices_when_other_active_set_changes(self):
        document,raw=self.fit();provider=r.load_provider(document,raw["deployment"]);aliases=raw["aliases"]
        s={"alias":"text#0","task":"text","pool_id":"i0"}
        g={"alias":"g#0","task":"text","pool_id":"i0"}
        s1={"alias":"text#1","task":"text","pool_id":"i0"}
        self.assertAlmostEqual(provider.service(s,{"text#0":s},aliases)["service_ms"],100.)
        self.assertAlmostEqual(provider.service(s,{"text#0":s,"g#0":g},aliases)["service_ms"],200.)
        self.assertAlmostEqual(provider.service(s,{"text#0":s,"text#1":s1,"g#0":g},aliases)["service_ms"],400.)

    def test_residual_support_and_background_unknown(self):
        document,raw=self.fit();provider=r.load_provider(document,raw["deployment"]);aliases=raw["aliases"]
        s={"alias":"text#0","task":"text","pool_id":"i0"}
        self.assertAlmostEqual(provider.residual(s,50,{"text#0":s},aliases)["remaining_ms"],50.)
        self.assertFalse(provider.residual(s,200,{"text#0":s},aliases)["known"])
        bg={"alias":"g#0","task":"image","pool_id":"i0"}
        self.assertFalse(provider.service(s,{"text#0":s,"g#0":bg},aliases)["known"])
        self.assertFalse(provider.power({"g#0":bg},aliases)["known"])

    def test_whole_board_power_not_sum_by_active_alias(self):
        document,raw=self.fit();provider=r.load_provider(document,raw["deployment"])
        active={alias:{"alias":alias,"task":"text","pool_id":"i0"} for alias in ("text#0","text#1","g#0")}
        self.assertAlmostEqual(provider.power(active,raw["aliases"])["watts"],100.)

    def test_reserved_or_wrong_calibration_seed_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path,raw=self.fixture(directory);raw["batches"][0]["seed"]=205;path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError,"seeds301/302"):
                r.fit_controlled(path)

    def test_rank_deficiency_remains_unknown(self):
        zero=r.key({"text":0});unseen=r.key({"text":1})
        rows=[{"seed":301,"pool_id":"i0","service_ms":100.,"durations_ms":{zero:100.}}]
        # Direct solver never fills a column absent from training data.
        result=r._fit_progress(rows,[zero,unseen],{301})
        self.assertEqual(result["status"],"UNKNOWN_RANK_DEFICIENT")
        self.assertIsNone(result["progress"])


if __name__=="__main__":
    unittest.main()
