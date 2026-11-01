import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from tests.test_background_statistical import fixture
from tools.predictor_validation_20261027.background_probe_validation import seal,report,_cohort
from tools.predictor_validation_20261027.probe_validation import receipt
from tools.predictor_validation_20261027.profiling import PLAN_SCHEMA,digest


class BackgroundProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name);self.validation = self.root/"validation";self.validation.mkdir()
        self.training_path = self.root/"training.json"
        self.training = {"schema":"gsdeploy.native_phase_collection.v1","status":"completed",
            "execution_context_sha256":"v7-fixture","request_records":[{"profile_job_id":"train","split":"training"}]}
        self.training_path.write_text(json.dumps(self.training))
        self.model = fixture();self.model["source_receipts"] = [receipt(self.training_path)]
        self.model.pop("model_sha256");self.model["model_sha256"] = digest(self.model)
        self.model_path = self.root/"model.json";self.model_path.write_text(json.dumps(self.model))
        self.plan = {"schema":PLAN_SCHEMA,"execution_context_sha256":"v7-fixture",
            "inputs":[{"task":t,"pool_id":"input","input_size":100} for t in ("image","asr")],
            "jobs":[{"id":"val-"+task,"context":"G_"+task,"split":"validation","inputs":{"generalist_2#0":"input"},
                "tasks":{"generalist_2#0":task},"start_offsets_ms":{"generalist_2#0":0},"validation_view":"independent_trial"}
                for task in ("image","asr")]}
        self.plan_path = self.validation/"plan.json";self.plan_path.write_text(json.dumps(self.plan))

    def seal(self):
        # Strict raw mapper itself is independently tested with real snapshots.
        with patch('tools.predictor_validation_20261027.background_probe_validation.map_background_collection',return_value={}):
            return seal(self.model_path,self.training_path,self.plan_path,self.root/"sealed")

    def raw(self,sealing):
        origin = max(time.perf_counter_ns(),sealing["created_ns"]+1)
        rows,batches = [],[]
        for i,job in enumerate(self.plan["jobs"]):
            start = origin+i*1_000_000_000;rid = "observed:"+str(i);task = job["tasks"]["generalist_2#0"]
            rows.append({"request_id":rid,"model":"generalist_2#0","task":task,"pool_id":job["inputs"]["generalist_2#0"],
                "profile_job_id":job["id"],"split":"validation","status":"completed","scheduled_time_ns":start,
                "observed_worker_start_time_ns":start,"worker_response":{"start_perf_ns":start,"end_perf_ns":start+100_000_000}})
            batches.append({"job_id":job["id"],"start_ns":start,"request_ids":[rid]})
        raw = {"schema":"gsdeploy.native_phase_collection.v1","status":"completed","kind":"measured","backend":"native",
            "performance_scope":"unprofiled_with_phase_markers","execution_context_sha256":"v7-fixture",
            "observation_start_ns":origin,"plan":receipt(self.plan_path),"request_records":rows,"batches":batches,
            "errors":[],"runtime_errors":[],"telemetry_errors":[]}
        path = self.validation/"raw.json";path.write_text(json.dumps(raw));return path

    def test_seal_preserves_actual_image_asr_and_report_gate(self):
        sealing = self.seal();pred = json.loads(Path(sealing["prediction"]["path"]).read_text())
        self.assertEqual({r["task"] for r in pred["requests"]},{"image","asr"})
        self.assertTrue(all(r["service_ms"] is not None for r in pred["requests"]))
        with patch('tools.predictor_validation_20261027.background_probe_validation.map_background_collection',return_value={}):
            result = report(self.root/"sealed",self.raw(sealing),self.root/"report")
        self.assertEqual(result["known_input_gate"]["status"],"PASS")
        self.assertEqual(set(result["task_groups"]),{"image","asr"})
        self.assertFalse(result["statistical_prediction_semantics"]["competitor_task_pooling"])

    def test_cohort_binds_task_not_just_family_and_pool(self):
        before = _cohort(self.plan);changed = copy.deepcopy(self.plan)
        changed["jobs"][0]["tasks"]["generalist_2#0"] = "asr"
        self.assertNotEqual(_cohort(changed),before)

    def test_unseen_image_stays_in_full_domain_denominator(self):
        self.plan["inputs"].append({"task":"image","pool_id":"unseen","input_size":200})
        self.plan["jobs"].append({**self.plan["jobs"][0],"id":"val-unseen","inputs":{"generalist_2#0":"unseen"}})
        self.plan_path.write_text(json.dumps(self.plan));sealing = self.seal()
        with patch('tools.predictor_validation_20261027.background_probe_validation.map_background_collection',return_value={}):
            result = report(self.root/"sealed",self.raw(sealing),self.root/"report")
        self.assertEqual(result["point_coverage"]["service"]["requested"],3)
        self.assertEqual(result["point_coverage"]["service"]["known"],2)
        self.assertEqual(result["known_input_gate"]["status"],"PASS")
        self.assertEqual(result["unseen_input_diagnostics"]["known"],0)

    def test_seal_refuses_after_raw_collection(self):
        (self.validation/"raw.json").write_text("{}")
        with self.assertRaisesRegex(ValueError,"before independent collection"):self.seal()


if __name__=="__main__":unittest.main()
