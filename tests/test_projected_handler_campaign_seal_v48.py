"""Tiny CPU artifact-binding contracts; no production seal or GPU work."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027 import projected_handler_campaign_seal_v48 as v


class ArtifactBindingTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)

    def report(self):
        return dict(schema=v.PREFLIGHT_SCHEMA,status='FOUR_FULL_COHORT_CPU_PREFLIGHTS_EXECUTED',
            source_bytes_unchanged=True,all_original_six_term_metrics_available=True,GPU_started=False,
            validation_labels_read=False,collection_eligible=False,protocol_sealed=False,
            cells=[dict(candidate_id=cid,status=v.COMPLETE,cohort_size=57,starts=57,receipts=57,metrics={'tiny':1})
                for cid in sorted(v.legacy.CANDIDATES)])

    def test_unknown_missing_candidate_or_source_drift_refuses_numeric_prerequisite(self):
        v.numeric_summary(self.report())
        for field,value in [('status','FAILED'),('source_bytes_unchanged',False),('GPU_started',True)]:
            changed=self.report();changed[field]=value
            with self.assertRaises(ValueError):v.numeric_summary(changed)
        changed=self.report();changed['cells'].pop()
        with self.assertRaises(ValueError):v.numeric_summary(changed)
        for field,value in [('status','UNKNOWN'),('starts',56),('receipts',None),('metrics',None)]:
            changed=self.report();changed['cells'][1][field]=value
            with self.assertRaises(ValueError):v.numeric_summary(changed)

    def test_streaming_gzip_projection_roundtrip_and_duplicate_or_bad_crc_refusal(self):
        parser,_=v.parser_runtime('/tmp/v16-preflight-readonly-ijson')
        def summary(data):
            p=self.root/'forecast.gz';p.write_bytes(gzip.compress(data))
            return dict(result=dict(artifact=v.receipt(p),uncompressed_sha256=hashlib.sha256(data).hexdigest(),
                uncompressed_bytes=len(data),complete_lossless_roundtrip_verified=True))
        value=dict(status=v.COMPLETE,forecast=dict(cohort={'r':{}},starts={'r':1},finishes={'r':{'finish_ns':2}},
            board_energy_joules=3.,peak_memory_mib=4.),scheduler_diagnostic={'queries':['opaque']*100},metrics={'terms':{}})
        s=summary(json.dumps(value).encode());result=v.forecast_projection(s,parser)
        self.assertEqual(result['forecast'],value['forecast']);self.assertNotIn('scheduler_diagnostic',result)
        duplicate=summary(b'{"status":"UNKNOWN","status":"COMPLETE"}')
        with self.assertRaisesRegex(ValueError,'Duplicate'):v.forecast_projection(duplicate,parser)
        s=summary(b'{"status":"UNKNOWN"}');p=Path(s['result']['artifact']['path']);data=bytearray(p.read_bytes());data[-8]^=1;p.write_bytes(data)
        s['result']['artifact']=v.receipt(p)
        with self.assertRaises((OSError,EOFError,ValueError)):v.forecast_projection(s,parser)

    def test_independent_six_term_and_gate_projection_mismatch_is_rejected(self):
        plan=dict(aliases={'text#0':dict(family='text',supported_tasks=['text'])},workload_rps={'text':1.},
            latency_reference_ms={'text':1000.},window_seconds=30.,drain_seconds=120.,weights={k:1/6 for k in 'QTLPME'},
            memory_budget_mib=100.,energy_reference_watts=10.,memory_safe_factor=1.1,
            execution_context_sha256='ctx',run_id='heldout',queue_sample_offsets_ns=[0,10_000_000])
        rows=[dict(request_id='r'+str(i),task='text',pool_id='p',alias='text#0',arrival_ns=i*1_000_000) for i in range(57)]
        forecast=dict(cohort={r['request_id']:r for r in rows},starts={r['request_id']:r['arrival_ns']+5_000_000 for r in rows},
            finishes={r['request_id']:dict(r,start_ns=r['arrival_ns']+5_000_000,handler_end_ns=r['arrival_ns']+6_000_000,finish_ns=r['arrival_ns']+7_000_000) for r in rows},
            peak_memory_mib=20.,board_energy_joules=30.)
        startup=dict(energy_joules=10.,peak_memory_mib=10.,energy_includes_priming=True,safety_factor_applied=False,startup_amortization_seconds=3600.,warmup_seconds=0.)
        config={'schema':v.MODEL_SCHEMA};plan_ref={'path':'fixture','sha256':'fixture'};candidate=dict(id='shared_balanced',execution_context_sha256='ctx')
        result=dict(schema=v.MODEL_SCHEMA,status=v.COMPLETE,configuration=config,plan=plan_ref,candidate_id=candidate['id'],
            execution_context_sha256='ctx',full_public_cohort=rows,CPU_representation_version=16,lifecycle_scheduler_version=44,
            GPU_started=False,global_request_concurrency_cap_used=False,physical_exclusive_gate_claimed=False,
            static_provider_compilation=dict(nodes_or_edges_dropped=False,fitted_values_changed=False),forecast=forecast,startup_prediction=startup)
        quality={'text':{'text':.1}};result['metrics']=v.score_complete_forecast(forecast,rows,plan,quality,startup,phase_start_ns=0)
        summary=dict(metrics=copy.deepcopy(result['metrics']));cell=v.gate_cell(result,rows,candidate,'heldout',plan['queue_sample_offsets_ns'],phase_start_ns=0)
        args=(summary,cell,plan,plan_ref,config,rows,candidate,0,quality)
        self.assertEqual(v.verify_projection(result,*args),cell)
        changed=copy.deepcopy(result);changed['metrics']['terms']['E']+=1
        with self.assertRaisesRegex(ValueError,'six-term'):v.verify_projection(changed,*args)
        changed=copy.deepcopy(cell);changed['queue_depth_samples'][0]['depth']+=1
        with self.assertRaisesRegex(ValueError,'gate-cell'):v.verify_projection(result,summary,changed,*args[2:])

    def test_new_factory_contract_is_explicit_and_source_bound(self):
        factory=self.root/'factory.py';factory.write_text('pass\n');preflight=self.root/'preflight.py';preflight.write_text('pass\n')
        contract=dict(schema='gsdeploy.projected_handler_artifact_contract.v48',forecast_schema='gsdeploy.full_cohort_projected_worker_handler_context_forecast.v50',
            preflight_schema='gsdeploy.projected_worker_handler_context_full_cohort_CPU_preflight.v50',lifecycle_scheduler_version=50,
            factory_source=v.receipt(factory),preflight_source=v.receipt(preflight))
        p=self.root/'contract.json';p.write_text(json.dumps(contract));self.assertEqual(v.forecast_contract(p),contract)
        report=self.report();report['schema']=contract['preflight_schema'];v.numeric_summary(report,contract)
        with self.assertRaises(ValueError):v.numeric_summary(report)
        factory.write_text('changed\n')
        with self.assertRaises(ValueError):v.forecast_contract(p)

    def test_availability_raw_must_be_covered_by_strict_complete_index(self):
        from unittest.mock import patch
        base=dict(component_bundle={'path':'bundle','sha256':'b'},availability_model={'path':'availability','sha256':'a'})
        config=dict(base_v15_configuration=dict(base_model_configuration=base))
        training=dict(artifacts={'source-manifest.json':{'path':'manifest','sha256':'m'}},raw_sha256=['included'])
        values={'manifest':dict(configuration=dict(model_lineage_roots=[base['component_bundle']])),
            'availability':dict(source_DAG_reports=[{'path':'report','sha256':'r'}])}
        audit=[dict(raw={'path':'raw','sha256':'included'})]
        with patch.object(v,'checked',side_effect=lambda ref:values[ref['path']]),patch.object(v,'availability_training_sources',return_value=audit):
            self.assertEqual(v.availability_training(config,training)[0],audit)
            with self.assertRaisesRegex(ValueError,'omitted'):
                v.availability_training(config,dict(training,raw_sha256=['unrelated']))
            values['manifest']['configuration']['model_lineage_roots']=[]
            with self.assertRaisesRegex(ValueError,'bundle'):
                v.availability_training(config,training)

    def test_seal_without_exact_independent_pass_refuses_before_protocol_creation(self):
        preparation=self.root/'preparation.json';preparation.write_text(json.dumps(dict(schema=v.SCHEMA,
            status='CPU_PREPARED_REQUIRES_INDEPENDENT_ARTIFACT_REVIEW_NO_PROTOCOL_OR_SEAL',all_four_results_independently_reprojected=True)))
        review=self.root/'review.json';review.write_text(json.dumps(dict(schema='gsdeploy.projected_handler_artifact_binding_independent_review.v48',
            status='FAIL',issues=['UNKNOWN'],preparation=v.receipt(preparation),complete_four_forecasts_reprojected=False)))
        output=self.root/'not-created'
        with self.assertRaises(ValueError):v.seal(preparation,review,output)
        self.assertFalse(output.exists())

    def test_additional_supervisor_freezes_import_closure_and_detects_drift(self):
        source=Path(v.__file__).with_name('unified_corun_collect_v8_1.py')
        manifest=self.root/'execution.json';manifest.write_text(json.dumps(dict(
            schema='gsdeploy.additional_execution_sources.v48',sources=[v.receipt(source)])))
        refs=v.additional_execution_sources(manifest)
        self.assertIn(v.receipt(source),refs);self.assertIn(v.receipt(manifest),refs)
        self.assertGreater(len(refs),2)
        data=json.loads(manifest.read_text());data['sources'][0]['sha256']='bad';manifest.write_text(json.dumps(data))
        with self.assertRaises(ValueError):v.additional_execution_sources(manifest)


if __name__=='__main__':unittest.main()
