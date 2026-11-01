"""CPU fixtures for fresh workers and actual subprocess failure supervision."""
import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from tests import test_learned_campaign_seal_v15 as previous_fixture
from tools.predictor_validation_20261027 import learned_campaign_parallel_v16 as parallel
from tools.predictor_validation_20261027 import learned_campaign_seal_v16 as producer
from tools.predictor_validation_20261027.known_input_gate_v5 import freeze_gate_protocol
from tools.predictor_validation_20261027.known_input_observed_v5 import verify_before_collection
from tools.predictor_validation_20261027.native_collect import receipt


class ParallelCampaignTests(unittest.TestCase):
    write=previous_fixture.LearnedCampaignSealTests.write
    result=previous_fixture.LearnedCampaignSealTests.result

    def setUp(self):
        previous_fixture.LearnedCampaignSealTests.setUp(self)
        report=json.loads(self.preflight_path.read_text());report['CPU_representation_version']=16
        for summary in report['candidates']:
            path=Path(summary['result']['path']);result=json.loads(path.read_text())
            result.update(CPU_representation_version=16,
                static_provider_compilation=dict(nodes_or_edges_dropped=False,fitted_values_changed=False))
            path.write_text(json.dumps(result));summary['result']=receipt(path)
        self.preflight_path.write_text(json.dumps(report))
        self.duplicate_pids=False;self.missing_raw=False;self.bad_worker=False
        self.bad_compile=False;self.delete_source=False;self.wrong_pid=False

    def run_producer(self):
        case=self
        class FakeProvider:
            def __init__(self,configuration,**kwargs):
                case.assertTrue((case.out/'protocol.json').exists());case.calls.append('construct_after_freeze')
                self.bundle=json.loads(Path(configuration['base_model_configuration']['component_bundle']['path']).read_text())
                self.availability_training_audit=[dict(raw=r) for r in case.raw_refs]
                if case.missing_raw:self.availability_training_audit=[dict(raw=dict(path='missing',sha256='omitted'))]
            def execute(self,plan,cid,public,routes,startup,*,max_events):
                case.calls.append(cid);case.assertEqual(max_events,5_000_000)
                result=case.result(cid,shift=50,term=.25)
                result.update(CPU_representation_version=16,
                    static_provider_compilation=dict(nodes_or_edges_dropped=False,fitted_values_changed=False))
                if case.incomplete:result['status']='UNKNOWN'
                if case.bad_compile:result['static_provider_compilation']['nodes_or_edges_dropped']=True
                if case.mutate:case.source.write_text('changed during fresh worker')
                if case.delete_source and case.source.exists():case.source.unlink()
                return result
        def run_fake_workers(specs,protocol,**kwargs):
            workers=[]
            for i,spec in enumerate(specs):
                with patch.object(parallel.os,'getpid',return_value=1001 if case.duplicate_pids else 1001+i):
                    summary=parallel.run_worker(spec,protocol)
                path=Path(json.loads(Path(spec['path']).read_text())['output_directory'])/'worker-receipt.json'
                workers.append(dict(candidate_id=summary['candidate_id'],spec=spec,pid=summary['pid']+int(case.wrong_pid),
                    exit_code=0 if summary['status']==parallel.COMPLETE else 1,receipt=receipt(path)))
            return dict(status='FAILED_PROHIBITS_SEAL' if case.bad_worker else 'ALL_FOUR_CPU_WORKERS_EXITED_ZERO',
                workers=workers,all_workers_reaped=True,error=None)
        with patch.object(parallel,'LearnedAvailabilityForecast',FakeProvider),\
             patch.object(parallel,'run_parallel',side_effect=run_fake_workers),\
             patch.object(producer.previous,'prediction_source_paths',return_value=[self.source]):
            return producer.prepare_and_seal(self.plan_path,self.index_path,self.config_path,self.preflight_path,self.out)

    def test_four_fresh_worker_constructions_after_freeze_and_original_collector_compatibility(self):
        report=self.run_producer();self.assertEqual(len(self.calls),8)
        self.assertEqual(self.calls[::2],['construct_after_freeze']*4)
        prediction=json.loads(Path(report['predictions']['path']).read_text())
        self.assertEqual([len(c['requests']) for c in prediction['cells']],[57]*4)
        self.assertTrue(all(c['terms']['E']==.25 for c in prediction['cells']))
        protocol=json.loads(Path(report['protocol']['path']).read_text())
        self.assertEqual(protocol['weights'],self.plan['weights']);self.assertEqual(protocol['thresholds'],self.plan['validation_thresholds'])
        for helper in (producer.__file__,parallel.__file__):self.assertIn(receipt(helper),protocol['sources'])
        check=verify_before_collection(report['protocol']['path'],report['predictions']['path'],self.plan_path,
            run_id='fresh-run',candidate_id='shared_balanced',output_root=self.root/'future-collection')
        self.assertFalse(check['GPU_started']);self.assertFalse((self.root/'future-collection').exists())

    def test_incomplete_preflight_never_freezes_or_constructs_workers(self):
        report=json.loads(self.preflight_path.read_text());report['completeness']['status']='INCOMPLETE'
        self.preflight_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,'numeric'):self.run_producer()
        self.assertFalse(self.out.exists());self.assertEqual(self.calls,[])

    def test_v15_representation_preflight_cannot_qualify_v16_workers(self):
        report=json.loads(self.preflight_path.read_text());report['CPU_representation_version']=15
        self.preflight_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,'v16 compiled-provider'):self.run_producer()
        self.assertFalse(self.out.exists())

    def test_incomplete_fresh_worker_cannot_borrow_preflight_cells(self):
        self.incomplete=True
        with self.assertRaises(ValueError):self.run_producer()
        self.assertFalse((self.out/'predictions.json').exists());self.assertTrue((self.out/'failed-production.json').exists())

    def test_missing_actual_availability_training_raw_prohibits_fresh_forecast_and_seal(self):
        self.missing_raw=True
        with self.assertRaises(ValueError):self.run_producer()
        self.assertEqual(self.calls,['construct_after_freeze']*4)
        self.assertFalse((self.out/'predictions.json').exists())

    def test_source_mutation_during_worker_prohibits_seal(self):
        self.mutate=True
        with self.assertRaises(ValueError):self.run_producer()
        self.assertFalse((self.out/'predictions.json').exists())

    def test_duplicate_worker_process_identity_prohibits_seal(self):
        self.duplicate_pids=True
        with self.assertRaisesRegex(ValueError,'independent source-verified worker'):self.run_producer()
        self.assertFalse((self.out/'predictions.json').exists())

    def test_supervisor_and_worker_pid_must_agree(self):
        self.wrong_pid=True
        with self.assertRaisesRegex(ValueError,'independent source-verified worker'):self.run_producer()
        self.assertFalse((self.out/'predictions.json').exists())

    def test_changed_graph_compilation_prohibits_seal(self):
        self.bad_compile=True
        with self.assertRaises(ValueError):self.run_producer()
        self.assertFalse((self.out/'predictions.json').exists())

    def test_missing_source_still_records_failed_production(self):
        self.delete_source=True
        with self.assertRaises(ValueError):self.run_producer()
        self.assertFalse((self.out/'predictions.json').exists())
        self.assertTrue((self.out/'failed-production.json').exists())
        after=json.loads((self.out/'source-after.json').read_text())
        self.assertTrue(any('verification_error' in r for r in after))

    def test_last_source_verification_failure_quarantines_written_seal(self):
        original=producer._verify_protocol
        def verify(protocol):
            if (self.out/'predictions.json').exists():self.source.write_text('changed after seal write')
            return original(protocol)
        with patch.object(producer,'_verify_protocol',side_effect=verify):
            with self.assertRaisesRegex(ValueError,'Frozen evidence identity changed'):self.run_producer()
        self.assertFalse((self.out/'predictions.json').exists())
        self.assertTrue((self.out/'rejected-predictions.json').exists())
        self.assertTrue(json.loads((self.out/'failed-production.json').read_text())['prediction_seal_written'])

    def test_existing_worker_receipts_are_not_overwritten_or_reused(self):
        self.run_producer();spec=receipt(self.out/'worker-spec-shared_balanced.json')
        old=receipt(self.out/'workers'/'shared_balanced'/'worker-receipt.json')
        with self.assertRaisesRegex(ValueError,'must not already exist'):
            parallel.run_worker(spec,receipt(self.out/'protocol.json'))
        self.assertEqual(old,receipt(old['path']))

    def test_supervisor_failure_prohibits_seal_even_if_all_result_files_exist(self):
        self.bad_worker=True
        with self.assertRaisesRegex(ValueError,'successful reaped'):self.run_producer()
        self.assertFalse((self.out/'predictions.json').exists())

    def supervisor_fixture(self):
        protocol_path=self.root/'parallel-protocol.json';spec_refs=[]
        for candidate in self.plan['candidates']:
            cid=candidate['id'];spec=dict(schema=parallel.SPEC_SCHEMA,protocol_path=str(protocol_path),
                plan=receipt(self.plan_path),configuration=receipt(self.config_path),candidate_id=cid,
                public_inputs=receipt(self.plan_path.with_name('public-'+cid+'.json')),
                routes=receipt(self.plan_path.with_name('routes-'+cid+'.json')),
                training_index=receipt(self.index_path),output_directory=str(self.root/'processes'/cid),max_events=5_000_000)
            spec_refs.append(receipt(self.write('spec-'+cid+'.json',spec)))
        catalog=self.root/'runtime-inputs.jsonl'
        sources=[*spec_refs,receipt(self.plan_path),receipt(self.config_path),receipt(catalog)]
        protocol=freeze_gate_protocol(model_paths={'model':str(self.config_path)},source_paths=[r['path'] for r in sources],
            training_data_paths=[str(self.index_path)],candidates=[dict(id=c['id'],model_id='model',execution_context_sha256='ctx') for c in self.plan['candidates']],
            baseline_candidate_id='shared_balanced',validation_domain=[dict(run_id='fresh-run',requests=[{k:r[k] for k in ('request_id','task','pool_id')} for r in self.plan['trace']])],
            thresholds=self.plan['validation_thresholds'],created_unix_ns=1,weights=self.plan['weights'],
            utility_tie_tolerance=0.,queue_sample_offsets_ns=self.plan['queue_sample_offsets_ns'],public_input_catalog_path=str(catalog))
        self.write(protocol_path.name,protocol)
        script=self.root/'fixture_worker.py';script.write_text('''import json,os,pathlib,sys,time
spec=json.loads(pathlib.Path(sys.argv[1]).read_text())
mode=sys.argv[2]
if mode=='hang': time.sleep(30)
if mode=='exit': raise SystemExit(7)
out=pathlib.Path(spec['output_directory'])
(out/'worker-receipt.json').write_text(json.dumps(dict(candidate_id=spec['candidate_id'],pid=os.getpid())))
''')
        return spec_refs,receipt(protocol_path),script

    def test_supervisor_spawns_four_real_independent_cpu_processes(self):
        specs,protocol,script=self.supervisor_fixture();launched=[]
        def command(spec,unused):launched.append(spec);return [sys.executable,str(script),spec['path'],'ok']
        result=parallel.run_parallel(specs,protocol,timeout_seconds=5,_command_factory=command)
        self.assertEqual(result['status'],'ALL_FOUR_CPU_WORKERS_EXITED_ZERO');self.assertEqual(len(launched),4)
        pids=[json.loads(Path(w['receipt']['path']).read_text())['pid'] for w in result['workers']]
        self.assertEqual(len(set(pids)),4);self.assertNotIn(os.getpid(),pids);self.assertTrue(result['all_workers_reaped'])

    def test_supervisor_timeout_reaps_all_workers_and_cannot_claim_success(self):
        specs,protocol,script=self.supervisor_fixture()
        result=parallel.run_parallel(specs,protocol,timeout_seconds=.15,
            _command_factory=lambda spec,unused:[sys.executable,str(script),spec['path'],'hang'])
        self.assertEqual(result['status'],'FAILED_PROHIBITS_SEAL');self.assertEqual(result['error']['type'],'TimeoutError')
        self.assertTrue(result['all_workers_reaped']);self.assertTrue(all(w['exit_code'] is not None for w in result['workers']))

    def test_supervisor_nonzero_exit_reaps_all_workers(self):
        specs,protocol,script=self.supervisor_fixture()
        result=parallel.run_parallel(specs,protocol,timeout_seconds=5,
            _command_factory=lambda spec,unused:[sys.executable,str(script),spec['path'],'exit'])
        self.assertEqual(result['status'],'FAILED_PROHIBITS_SEAL');self.assertTrue(result['all_workers_reaped'])
        self.assertTrue(all(w['exit_code'] is not None for w in result['workers']))

    def test_supervisor_launch_failure_reaps_already_started_worker(self):
        specs,protocol,script=self.supervisor_fixture();launched=[]
        def command(spec,unused):
            launched.append(spec)
            if len(launched)==2:raise RuntimeError('fixture second launch failure')
            return [sys.executable,str(script),spec['path'],'hang']
        result=parallel.run_parallel(specs,protocol,timeout_seconds=5,_command_factory=command)
        self.assertEqual(result['status'],'FAILED_PROHIBITS_SEAL');self.assertTrue(result['all_workers_reaped'])
        self.assertEqual(len(result['workers']),1);self.assertIsNotNone(result['workers'][0]['exit_code'])

    def test_cleanup_exception_does_not_skip_other_worker_reaping(self):
        specs,protocol,script=self.supervisor_fixture();original=parallel._terminate;cleaned=[]
        def cleanup(process):
            original(process);cleaned.append(process.pid)
            if len(cleaned)==1:raise RuntimeError('fixture post-reap cleanup failure')
        with patch.object(parallel,'_terminate',side_effect=cleanup):
            result=parallel.run_parallel(specs,protocol,timeout_seconds=.15,
                _command_factory=lambda spec,unused:[sys.executable,str(script),spec['path'],'hang'])
        self.assertEqual(result['status'],'FAILED_PROHIBITS_SEAL');self.assertEqual(len(cleaned),4)
        self.assertEqual(result['error']['type'],'WorkerCleanupFailure');self.assertTrue(result['all_workers_reaped'])


if __name__=='__main__':unittest.main()
