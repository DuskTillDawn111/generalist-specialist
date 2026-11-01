"""Independent CPU regressions for starts, producer closure and seal identity."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from gsdeploy.forecast_errors import UnknownEvidence

from tools.predictor_validation_20261027 import projected_handler_campaign_seal_v48 as old
from tools.predictor_validation_20261027 import projected_handler_campaign_seal_v55 as new


def fixture():
    plan=dict(aliases={'text#0':dict(family='text',supported_tasks=['text'])},workload_rps={'text':1.},
        latency_reference_ms={'text':1000.},window_seconds=30.,drain_seconds=120.,weights={k:1/6 for k in 'QTLPME'},
        memory_budget_mib=100.,energy_reference_watts=10.,memory_safe_factor=1.1,
        execution_context_sha256='ctx',run_id='fixture',queue_sample_offsets_ns=[0,10_000_000])
    rows=[dict(request_id='cpu:'+str(i),task='text',pool_id='p',alias='text#0',arrival_ns=i*1_000_000) for i in range(57)]
    forecast=dict(cohort={r['request_id']:r for r in rows},starts={r['request_id']:r['arrival_ns']+1_000_000 for r in rows},
        finishes={r['request_id']:dict(r,start_ns=r['arrival_ns']+1_000_000,
            handler_end_ns=r['arrival_ns']+2_000_000,finish_ns=r['arrival_ns']+3_000_000) for r in rows},
        peak_memory_mib=20.,board_energy_joules=30.)
    startup=dict(energy_joules=10.,peak_memory_mib=10.,energy_includes_priming=True,
        safety_factor_applied=False,startup_amortization_seconds=3600.,warmup_seconds=0.)
    config={'schema':new.MODEL_SCHEMA};plan_ref={'path':'fixture','sha256':'fixture'}
    candidate=dict(id='shared_balanced',execution_context_sha256='ctx')
    quality={'text':{'text':.1}}
    result=dict(schema=new.MODEL_SCHEMA,status=new.COMPLETE,configuration=config,plan=plan_ref,
        candidate_id=candidate['id'],execution_context_sha256='ctx',full_public_cohort=rows,
        CPU_representation_version=16,lifecycle_scheduler_version=44,GPU_started=False,
        global_request_concurrency_cap_used=False,physical_exclusive_gate_claimed=False,
        static_provider_compilation=dict(nodes_or_edges_dropped=False,fitted_values_changed=False),
        forecast=forecast,startup_prediction=startup)
    result['metrics']=new.score_complete_forecast(forecast,rows,plan,quality,startup,phase_start_ns=0)
    cell=new.gate_cell(result,rows,candidate,plan['run_id'],plan['queue_sample_offsets_ns'],phase_start_ns=0)
    return result,(dict(metrics=copy.deepcopy(result['metrics'])),cell,plan,plan_ref,config,rows,candidate,0,quality)


class IndependentSealTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)

    def test_frozen_v48_start_value_defect_and_v55_rejection(self):
        result,args=fixture()
        self.assertEqual(new.verify_projection(result,*args),args[1])
        for value in ('not-numeric',None,True,1.5,999999):
            with self.subTest(value=value):
                changed=copy.deepcopy(result);changed['forecast']['starts']['cpu:0']=value
                self.assertEqual(old.verify_projection(changed,*args),args[1])
                with self.assertRaisesRegex(ValueError,'numeric START'):
                    new.verify_projection(changed,*args)

    def test_same_complete_metrics_do_not_hide_missing_or_changed_receipt_identity(self):
        result,args=fixture()
        for mutate in ('missing_start','missing_finish','foreign_alias','drain','metric','cell'):
            changed=copy.deepcopy(result);local=list(copy.deepcopy(args))
            if mutate=='missing_start':changed['forecast']['starts'].pop('cpu:0')
            elif mutate=='missing_finish':changed['forecast']['finishes'].pop('cpu:0')
            elif mutate=='foreign_alias':changed['forecast']['finishes']['cpu:0']['alias']='ocr#0'
            elif mutate=='drain':changed['forecast']['finishes']['cpu:0']['finish_ns']=151_000_000_000
            elif mutate=='metric':changed['metrics']['terms']['E']+=1
            else:local[1]['queue_depth_samples'][0]['depth']+=1
            with self.subTest(mutate=mutate),self.assertRaises((ValueError,UnknownEvidence)):
                new.verify_projection(changed,*local)

    def test_full_dynamic_producer_closure_must_preexist_prediction_and_retain_bytes(self):
        contract=new.forecast_contract()
        complete=new.implementation_sources(contract['factory_source']['path'],contract['preflight_source']['path'])
        self.assertEqual(new.verify_producer_sources(complete,copy.deepcopy(complete),contract),complete)
        entry_only=new.strict_references([contract['factory_source'],contract['preflight_source']])
        with self.assertRaisesRegex(ValueError,'complete static import closure'):
            new.verify_producer_sources(entry_only,entry_only,contract)
        with self.assertRaisesRegex(ValueError,'source-before/after'):
            new.verify_producer_sources(complete,complete[:-1],contract)
        changed=copy.deepcopy(complete);changed[0]['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'bytes changed'):
            new.verify_producer_sources(changed,changed,contract)

    def test_gzip_exact_hash_and_duplicate_keys_are_checked_beyond_selected_projection(self):
        parser,_=new.parser_runtime('/tmp/v16-preflight-readonly-ijson')
        for data,sha,size in [(b'{"status":"UNKNOWN","opaque":{"x":1,"x":2}}',None,None),
                (b'{"status":"UNKNOWN"}','0'*64,None),(b'{"status":"UNKNOWN"}',None,1)]:
            path=self.root/'fixture.gz';path.write_bytes(gzip.compress(data))
            summary=dict(result=dict(artifact=new.receipt(path),uncompressed_sha256=sha or hashlib.sha256(data).hexdigest(),
                uncompressed_bytes=len(data) if size is None else size,complete_lossless_roundtrip_verified=True))
            with self.assertRaises(ValueError):new.forecast_projection(summary,parser)

    def test_v55_contract_and_review_do_not_silently_accept_v48_identity(self):
        contract=old.forecast_contract();path=self.root/'contract.json';path.write_text(json.dumps(contract))
        with self.assertRaisesRegex(ValueError,'contract required'):new.forecast_contract(path)
        preparation=self.root/'prep.json'
        preparation.write_text(json.dumps(dict(schema=new.SCHEMA,
            status='CPU_PREPARED_REQUIRES_INDEPENDENT_ARTIFACT_REVIEW_NO_PROTOCOL_OR_SEAL',
            all_four_results_independently_reprojected=True,preflight_cells_reused=True,
            provider_constructed_after_freeze=False,exact_numeric_START_completion_consistency_verified=True,
            producer_static_import_closure_frozen_before_predictions=True,protocol_sealed=False,
            GPU_started=False,validation_outcomes_read=False)))
        review=self.root/'review.json';review.write_text(json.dumps(dict(
            schema='gsdeploy.projected_handler_artifact_binding_independent_review.v48',status='PASS',
            preparation=new.receipt(preparation),complete_four_forecasts_reprojected=True,issues=[])))
        with self.assertRaisesRegex(ValueError,'independent artifact review'):
            new.seal(preparation,review,self.root/'not-created')
        self.assertFalse((self.root/'not-created').exists())

    def test_actual_UNKNOWN_report_refuses_before_parser_training_or_output(self):
        base=Path(new.__file__).resolve().parents[2]
        campaign=base/'artifacts/predictor-projected-handler-v44-20261028-preflight01'
        report=json.loads((campaign/'report.json').read_text())
        with gzip.open(campaign/'shared_spill_high-forecast.json.gz','rt') as stream:
            diagnostic=json.load(stream)
        config=self.root/'configuration.json';config.write_text(json.dumps(diagnostic['configuration']))
        with mock.patch.object(new,'parser_runtime',side_effect=AssertionError('parser must stay unopened')), \
                mock.patch.object(new.legacy,'_complete_index',side_effect=AssertionError('training must stay unopened')):
            with self.assertRaises(ValueError):
                new.prepare(diagnostic['plan']['path'],config,campaign/'report.json','not-read',self.root/'not-created','not-read')
        self.assertFalse((self.root/'not-created').exists())


if __name__=='__main__':unittest.main()
