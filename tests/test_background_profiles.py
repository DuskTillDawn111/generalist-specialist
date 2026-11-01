import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027.background_profiles import (
    RESIDENTS, background_semantic_traces, map_background_collection, receipt,
)
from tools.predictor_validation_20261027.profiling_analysis import digest
from tools.predictor_validation_20261027.statistical_profiles import fit_traces


class BackgroundMapperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        source = self.root/"worker.py";source.write_text("# frozen\n")
        ready = {a:{"pid":i+100,"worker":f,"spec_sha256":"spec-"+f,
                    "tasks":["text","image","asr","ocr"] if f=="generalist_2" else ["ocr" if f=="ocr" else "text"]}
                 for i,(a,f) in enumerate(RESIDENTS.items())}
        context = {"residents":RESIDENTS,"sources":[receipt(source)],
            "execution_sources":[{"original_path":"/frozen/original/worker.py","frozen":receipt(source)}],
            "worker_bindings":{f:{"spec_sha256":"spec-"+f} for f in RESIDENTS.values()},
            "execution":{"batch_size":1,"backend":"native-semantic","weight_reuse":False}}
        value = {k:v for k,v in context.items() if k not in {"sources","execution_sources"}}
        value["source_content"] = [{"original_path":"/frozen/original/worker.py","sha256":receipt(source)["sha256"]}]
        self.identity = digest(value)
        plan = {"resident_aliases":RESIDENTS,"execution_context_sha256":self.identity,
                "inputs":[{"task":t,"pool_id":"pool-"+t,"input_size":10} for t in ("image","asr")],"jobs":[]}
        records,batches = [],[]
        for i,task in enumerate(("image","image","asr","asr")):
            job = {"id":"job-"+str(i),"inputs":{"generalist_2#0":"pool-"+task},
                   "tasks":{"generalist_2#0":task},"start_offsets_ms":{"generalist_2#0":0},"split":"training"}
            plan["jobs"].append(job);rid = "r:"+str(i);offset=i*200_000_000
            stages = [{"id":"native_inference","kind":"gpu","start_ns":offset+20_000_000,"end_ns":offset+80_000_000},
                      {"id":"post_sync","kind":"sync","start_ns":offset+80_000_000,"end_ns":offset+90_000_000}]
            forwards = [{"id":"prefill_0000","kind":"gpu","start_ns":offset+30_000_000,"end_ns":offset+40_000_000,"forward_index":0},
                        {"id":"decode_0001","kind":"gpu","start_ns":offset+60_000_000,"end_ns":offset+70_000_000,"forward_index":1}]
            response = {"request_id":rid,"task_type":task,"task":task,"start_perf_ns":offset+10_000_000,
                "end_perf_ns":offset+100_000_000,"observed_stages":stages,"semantic_forward_observations":{
                    "schema":"gsdeploy.semantic_forward_observation.v1","request_id":rid,"clock":"time.perf_counter_ns",
                    "records":forwards,"instrumentation_errors":[],"gpu_synchronization_added":False}}
            events = [{"request_id":rid,"pid":ready["generalist_2#0"]["pid"],"perf_ns":response["start_perf_ns"]}]
            for s in stages+forwards:
                for edge in ("start","end"):
                    events.append({"request_id":rid,"event":edge,"segment_id":s["id"],"kind":s["kind"],
                        "perf_ns":s[edge+"_ns"],"nvtx_name":f"GS_STAGE:{rid}:{s['kind']}:{s['id']}"})
            records.append({"request_id":rid,"model":"generalist_2#0","task":task,"pool_id":"pool-"+task,
                "profile_job_id":job["id"],"split":"training","status":"completed","worker_response":response,
                "observed_worker_start_time_ns":response["start_perf_ns"],"execution_events":events})
            batches.append({"job_id":job["id"],"start_ns":offset+1_000_000,"end_ns":offset+110_000_000,"request_ids":[rid]})
        plan["plan_sha256"] = digest(plan)
        cp,pp,ep = self.root/"context.json",self.root/"plan.json",self.root/"executed.json"
        cp.write_text(json.dumps(context));pp.write_text(json.dumps(plan));ep.write_text(json.dumps({"jobs":plan["jobs"]}))
        samples = [{"time_ns":i*1_000_000,"watts":50.,"processes_mib":{str(r["pid"]):100. for r in ready.values()},
                    "owned_used_mib":400.,"owned_aliases":list(RESIDENTS),"foreign_pids":[]} for i in range(0,801,10)]
        self.raw = {"schema":"gsdeploy.native_phase_collection.v1","kind":"measured","backend":"native","status":"completed",
            "execution_context_sha256":self.identity,"context":receipt(cp),"plan":receipt(pp),"executed_jobs":receipt(ep),
            "ready":ready,"request_records":records,"batches":batches,"board_samples":samples,
            "observation_start_ns":0,"observation_end_ns":800_000_000,"errors":[],"runtime_errors":[],"telemetry_errors":[]}
        self.path = self.root/"raw.json"

    def save(self):
        self.path.write_text(json.dumps(self.raw))

    def test_image_asr_observations_map_and_train_without_text_only_audit(self):
        self.save();mapped = map_background_collection(self.path)
        self.assertEqual(mapped["mapping_report"]["tasks"],["asr","image"])
        traces,_ = background_semantic_traces(self.path)
        self.assertEqual([s["phase"] for s in traces[0]["requests"][0]["segments"]],["prepare","prefill","decode","finalize"])
        model = fit_traces(traces)
        self.assertEqual(model["coverage"]["profiles"],2)
        self.assertEqual({p["task"] for p in model["profiles"]},{"image","asr"})

    def test_task_cannot_change_without_frozen_job_binding(self):
        self.raw["request_records"][0]["task"] = "text";self.save()
        with self.assertRaisesRegex(ValueError,"Task differs"):map_background_collection(self.path)

    def test_phase_or_START_mismatch_rejected(self):
        self.raw["request_records"][0]["execution_events"][1]["perf_ns"] += 1;self.save()
        with self.assertRaisesRegex(ValueError,"phase does not match"):map_background_collection(self.path)

    def test_foreign_or_missing_resident_telemetry_rejected(self):
        self.raw["board_samples"][0]["foreign_pids"] = [999];self.save()
        with self.assertRaisesRegex(ValueError,"owned residents"):map_background_collection(self.path)

    def test_snapshot_tampering_rejected(self):
        (self.root/"worker.py").write_text("# changed\n");self.save()
        with self.assertRaisesRegex(ValueError,"receipt"):map_background_collection(self.path)

    def test_semantic_forward_missing_rejected(self):
        self.raw["request_records"][0]["worker_response"]["semantic_forward_observations"]["records"] = [];self.save()
        with self.assertRaisesRegex(ValueError,"No observed forward"):map_background_collection(self.path)

    def test_mixed_execution_training_sources_rejected(self):
        self.save();traces,_ = background_semantic_traces(self.path)
        traces[-1]["execution_context_sha256"] = "other"
        with self.assertRaisesRegex(ValueError,"Mixed execution"):fit_traces(traces)


if __name__=="__main__":unittest.main()
