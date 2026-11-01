"""Independent CPU checks; temporary synthetic seals never authorize GPU work."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import main_prediction_gate_supervisor_v1 as m
from tools.predictor_validation_20261027 import known_input_gate_v5 as gate
from tools.predictor_validation_20261027 import projected_handler_campaign_seal_v48 as campaign

ROOT=m.ROOT


class IndependentSupervisorTests(unittest.TestCase):
    def fixture(self,root,*,bind_review=True,changed_threshold=False):
        plan_path=ROOT/'artifacts/predictor-gpu-20261027/known-input-availability-v11-20261028-prepared01/plan.json'
        plan=m.read(plan_path);context=campaign.checked(plan['context'])
        catalog=next(s['frozen']['path'] for s in context['execution_sources'] if s['original_path'].endswith('/runtime-inputs.jsonl'))
        m.save(root/'model.json',dict(synthetic_CPU_fixture_only=True))
        m.save(root/'preparation.json',dict(schema=campaign.SCHEMA,
            status='CPU_PREPARED_REQUIRES_INDEPENDENT_ARTIFACT_REVIEW_NO_PROTOCOL_OR_SEAL',plan=m.receipt(plan_path),
            all_four_results_independently_reprojected=True))
        prep=m.receipt(root/'preparation.json')
        m.save(root/'review.json',dict(schema='gsdeploy.projected_handler_artifact_binding_independent_review.v48',
            status='PASS',issues=[],preparation=prep,complete_four_forecasts_reprojected=True))
        review=m.receipt(root/'review.json')
        source_paths=[r['path'] for r in m.implementation_sources(m.__file__)]+[str(plan_path),catalog]
        if bind_review:source_paths.extend([prep['path'],review['path']])
        thresholds=copy.deepcopy(plan['validation_thresholds'])
        if changed_threshold:thresholds['service_mare_max']=999.
        protocol=gate.freeze_gate_protocol(model_paths={'CPU_fixture':str(root/'model.json')},source_paths=source_paths,
            training_data_paths=[str(ROOT/'artifacts/predictor-complete-training-index-v12-20261028-built02/training-index.json')],
            candidates=[dict(id=c['id'],model_id='CPU_fixture',execution_context_sha256=c['execution_context_sha256']) for c in plan['candidates']],
            baseline_candidate_id=plan['baseline_candidate_id'],validation_domain=[dict(run_id=plan['run_id'],requests=[{k:r[k] for k in ('request_id','pool_id','task')} for r in plan['trace']])],
            thresholds=thresholds,created_unix_ns=1,weights=plan['weights'],queue_sample_offsets_ns=plan['queue_sample_offsets_ns'],public_input_catalog_path=catalog)
        original=m.read(ROOT/'artifacts/predictor-projected-handler-v44-20261028-preflight01/shared_balanced-unsealed-cell.json')
        cells=[]
        for candidate in plan['candidates']:
            cell=copy.deepcopy(original);cell.update(candidate_id=candidate['id'],execution_context_sha256=candidate['execution_context_sha256']);cells.append(cell)
        predictions=gate.seal_predictions(protocol,cells,created_unix_ns=2)
        m.save(root/'protocol.json',protocol);m.save(root/'predictions.json',predictions)
        m.save(root/'production.json',dict(schema=campaign.SCHEMA,
            status='ARTIFACT_BOUND_PREDICTIONS_SEALED_REQUIRES_POSTSEAL_GPU_VALIDATION',protocol=m.receipt(root/'protocol.json'),
            predictions=m.receipt(root/'predictions.json'),preparation=prep,independent_review=review,
            original_v5_precollection_bindings_passed=True,all_GPU_must_start_after_seal=True,GPU_started=False))
        return [plan_path,root/'protocol.json',root/'predictions.json',root/'production.json'],protocol,plan

    def test_real_v5_binding_and_original_v11_plan_accept_compatible_temporary_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            args,protocol,plan=self.fixture(Path(d))
            actual,sources=m.prerequisites(*args)
            self.assertEqual(actual,plan);self.assertTrue(all(ref in protocol['sources'] for ref in sources))

    def test_reproduces_unbound_preparation_and_review_acceptance_gap(self):
        with tempfile.TemporaryDirectory() as d:
            args,protocol,_=self.fixture(Path(d),bind_review=False)
            production=m.read(args[3])
            self.assertNotIn(production['preparation'],protocol['sources'])
            self.assertNotIn(production['independent_review'],protocol['sources'])
            # Reproduces the missing check; this does not qualify production.
            m.prerequisites(*args)

    def test_reproduces_threshold_mismatch_acceptance_gap(self):
        with tempfile.TemporaryDirectory() as d:
            args,protocol,plan=self.fixture(Path(d),changed_threshold=True)
            self.assertNotEqual(protocol['thresholds'],plan['validation_thresholds'])
            m.prerequisites(*args)

    def test_actual_gate_rejects_forged_request_pool_and_context(self):
        with tempfile.TemporaryDirectory() as d:
            args,protocol,_=self.fixture(Path(d));predictions=m.read(args[2])
            for field in ('pool','context'):
                changed=copy.deepcopy(predictions)
                if field=='pool':changed['cells'][0]['requests'][0]['pool_id']='foreign'
                else:changed['cells'][0]['execution_context_sha256']='foreign'
                from gsdeploy.resources import identity
                changed['prediction_sha256']=identity({k:v for k,v in changed.items() if k!='prediction_sha256'})
                args[2].write_text(json.dumps(changed))
                with self.subTest(field=field),self.assertRaises(ValueError):m.prerequisites(*args)

    def test_timeout_kills_owned_descendant_and_preserves_unrelated_session(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pidfile=root/'child.pid'
            code='import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]); open(sys.argv[1],"w").write(str(p.pid));time.sleep(30)'
            own=subprocess.Popen([sys.executable,'-c',code,str(pidfile)],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            other=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            try:
                deadline=time.monotonic()+3
                while not pidfile.exists() and time.monotonic()<deadline:time.sleep(.01)
                self.assertTrue(pidfile.exists());descendant=int(pidfile.read_text())
                self.assertIn(descendant,m.group_members(own.pid))
                with patch.object(m.shutil,'disk_usage',return_value=type('Disk',(),dict(free=20*2**30))()):
                    result=m.wait_bounded(own,root,dict(m.BUDGET,total_timeout_seconds=.02,cleanup_grace_seconds=.05,watchdog_interval_seconds=.01))
                self.assertEqual(result['status'],'TIMEOUT');self.assertEqual(result['cleanup']['live_survivors'],[])
                self.assertIsNone(other.poll());self.assertIn(other.pid,m.group_members(other.pid))
                self.assertTrue(all(e['owned_process_group']==own.pid for e in result['cleanup']['events']))
            finally:m.cleanup_group(own,.05);m.cleanup_group(other,.05)

    def test_exact_13GiB_floor_and_6GiB_increment_boundaries(self):
        # An exited owned process permits a single exact-boundary sample.
        for initial,current,status in [(20*2**30,14*2**30,'EXITED'),(20*2**30,14*2**30-1,'STORAGE_SCENARIO_EXCEEDED'),
                                       (18*2**30,13*2**30,'EXITED'),(18*2**30,13*2**30-1,'DISK_MARGIN')]:
            process=subprocess.Popen([sys.executable,'-c','pass'],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);process.wait()
            try:
                samples=[type('Disk',(),dict(free=x))() for x in (initial,current)]
                with patch.object(m.shutil,'disk_usage',side_effect=samples):result=m.wait_bounded(process,'/tmp',m.BUDGET)
                self.assertEqual(result['status'],status)
            finally:m.cleanup_group(process,.05)


if __name__=='__main__':unittest.main()
