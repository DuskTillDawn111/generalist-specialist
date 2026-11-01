"""CPU synthetic contracts only: these fixtures are not GPU smoke evidence."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from tests.test_device_progress_observer_v1 import Cuda
from tests.test_semantic_observer import FakeModule,FakeNvtx,TensorShape
from tools.predictor_validation_20261027.device_progress_observer_v1 import observe_language_forwards
from tools.predictor_validation_20261027 import device_progress_smoke_raw_verify_v1 as v


def fixture():
    raw=dict(status='completed',split='smoke',ready={},request_records=[],shutdown={},errors=[],runtime_errors=[],telemetry_errors=[])
    for index,owner in enumerate(v.OWNERS):
        pid=100+index;rid='fixture:'+owner;begin=1000;tick=1100;emitted=[]
        def clock():
            nonlocal tick
            tick+=10;return tick
        raw['ready'][owner]=dict(pid=pid,torch='CPU_FAKE_NOT_REAL_GPU',clock='time.perf_counter_ns')
        if owner!='ocr#0':
            module=FakeModule();cuda=Cuda();cuda.complete=True
            with observe_language_forwards(SimpleNamespace(family='chatqa',model=module),rid,
                    torch=SimpleNamespace(cuda=cuda),nvtx=FakeNvtx(),sink=emitted.append,clock=clock) as observer:
                module(TensorShape(1,3))
            document=observer.document()
        else:document=None
        end=clock()+10
        response=dict(request_id=rid,success=True,raw_output='fixture-output',prediction='fixture-prediction',start_perf_ns=begin,end_perf_ns=end)
        if document is not None:response['semantic_forward_observations']=document
        events=[dict(request_id=rid,pid=pid,perf_ns=begin)]+emitted
        for event in events:
            at=event['perf_ns'];event.update(received_time_ns=at+100,collector_dequeued_time_ns=at+110,collector_processing_time_ns=at+120)
        raw['request_records'].append(dict(request_id=rid,model=owner,status='completed',worker_response=response,execution_events=events,
            response_reader_event_enqueue_time_ns=end+100,response_collector_dequeued_time_ns=end+110,response_collector_processing_time_ns=end+120,
            completion_time_ns=end+100,previous_handler_next_request_ready=dict(request_id=rid,pid=pid,clock='host_perf_counter_ns',
                boundary='previous_handler_next_request_ready',perf_ns=end+20,received_time_ns=end+130,collector_dequeued_time_ns=end+140,collector_processing_time_ns=end+150)))
        raw['shutdown'][owner]=dict(forced=False,exit_code=0,bye_received=True)
    return raw


def language(raw):return raw['request_records'][0]
def progress(raw):return language(raw)['worker_response']['semantic_forward_observations']['device_progress']


class RawVerifierTests(unittest.TestCase):
    def packet(self,directory,*,split='smoke',mode='natural'):
        root=Path(directory);raw=fixture();raw.pop('split')
        view='correctness_clock_only_not_fit' if split=='smoke' else 'training'
        plan=dict(schema='gsdeploy.paired_profile_campaign.v4',split=split,mode=mode,
            progress_observer=True,mode_switch_at_execute=False,final_gain_eligible=False,execution_context_sha256='fixture-context',
            jobs=[dict(id='fixture-job',split=split,validation_view=view)])
        context=dict(paired_profile_mode=mode,protocol='paired_profile_v4_'+mode,worker_variant='semantic_progress_worker_v1.py')
        raw.update(schema='gsdeploy.native_phase_collection.v1',kind='measured',backend='native',execution_context_sha256='fixture-context',
            performance_scope='profiler_perturbed' if mode=='kernel' else 'unprofiled_with_phase_markers')
        for row in raw['request_records']:row.update(split=split,profile_job_id='fixture-job',validation_view=view)
        for name,value in [('plan',plan),('context',context)]:
            path=root/(name+'.json');path.write_text(json.dumps(value));raw[name]=v.receipt(path)
        return raw,plan,context

    def test_run_accepts_actual_collector_shape_without_top_level_split(self):
        with tempfile.TemporaryDirectory() as directory:
            for split,mode in [('smoke','natural'),('training','kernel')]:
                raw,_,_=self.packet(directory,split=split,mode=mode)
                path=Path(directory)/'synthetic-raw.json';path.write_text(json.dumps(raw))
                report=json.loads(Path(v.run(path,Path(directory)/(split+'-cpu-fixture-only'))['path']).read_text())
                self.assertFalse(report['qualified_role']['raw_top_level_split_present'])
                self.assertEqual(report['qualified_role']['split'],split)
                self.assertEqual(report['status'],'RAW_STRUCTURAL_CONTRACT_ACCEPTED_DEVICE_API_NOT_AUTHENTICATED')

    def test_actual_shape_rejects_plan_role_schema_mode_and_row_role_conflicts(self):
        for change in [('schema','wrong'),('split','validation'),('mode','unknown'),('progress_observer',False)]:
            with tempfile.TemporaryDirectory() as directory:
                raw,plan,_=self.packet(directory);plan[change[0]]=change[1]
                path=Path(directory)/'plan.json';path.write_text(json.dumps(plan));raw['plan']=v.receipt(path)
                with self.assertRaises(ValueError):v.qualify_roles(raw)
        for change in [('split','validation'),('split',None),('profile_job_id','undeclared'),('validation_view','validation')]:
            with tempfile.TemporaryDirectory() as directory:
                raw,_,_=self.packet(directory);raw['request_records'][0][change[0]]=change[1]
                with self.assertRaises(ValueError):v.qualify_roles(raw)

    def test_role_documents_need_exact_receipts_and_matching_context_and_raw_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            raw,_,_=self.packet(directory)
            (Path(directory)/'plan.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Exact raw plan/context'):v.qualify_roles(raw)
        for mutate in [lambda raw:raw.update(split='validation'),lambda raw:raw.update(performance_scope='profiler_perturbed'),
                lambda raw:raw.update(validation_labels_used=True)]:
            with tempfile.TemporaryDirectory() as directory:
                raw,_,_=self.packet(directory);mutate(raw)
                with self.assertRaises(ValueError):v.qualify_roles(raw)
        with tempfile.TemporaryDirectory() as directory:
            raw,_,context=self.packet(directory);context['paired_profile_mode']='kernel'
            path=Path(directory)/'context.json';path.write_text(json.dumps(context));raw['context']=v.receipt(path)
            with self.assertRaisesRegex(ValueError,'context observer/mode'):v.qualify_roles(raw)

    def test_complete_fixture_accepts_structure_without_claiming_real_GPU(self):
        result=v.validate_raw(fixture())
        self.assertEqual(result['issues'],[])
        self.assertEqual(result['status'],'RAW_STRUCTURAL_CONTRACT_ACCEPTED_DEVICE_API_NOT_AUTHENTICATED')
        self.assertEqual(result['uninstrumented_owner']['status'],'UNKNOWN_NOT_INSTRUMENTED')

    def test_language_probe_missing_or_errors_fail_and_preserve_error(self):
        raw=fixture();language(raw)['worker_response']['semantic_forward_observations']['instrumentation_errors']=[dict(where='device_record_begin',type='RuntimeError')]
        result=v.validate_raw(raw);self.assertEqual(result['status'],'FAIL')
        self.assertEqual(result['requests'][0]['errors_preserved'],[dict(where='device_record_begin',type='RuntimeError')])
        raw=fixture();progress(raw)['records']=[]
        self.assertEqual(v.validate_raw(raw)['status'],'FAIL')

    def test_query_stream_device_binding_and_global_claims_rejected(self):
        for field,value in [('device_index',1),('stream_id',999),('all_request_GPU_work_complete',True),('in_flight_remaining_fraction',.2),('physical_SM_service_ns',1)]:
            raw=fixture();progress(raw)['queries'][0][field]=value
            self.assertEqual(v.validate_raw(raw)['status'],'FAIL',field)

    def test_record_query_NVTX_or_parent_order_tamper_rejected(self):
        for mutate in [lambda raw:progress(raw)['records'][0].update(begin_record_host_bracket_ns=[0,1]),
                lambda raw:progress(raw)['queries'][0].update(host_query_begin_ns=0),
                lambda raw:progress(raw)['nvtx_host_brackets'][0].update(pop_after_ns=0),
                lambda raw:language(raw)['execution_events'][-1].update(received_time_ns=0)]:
            raw=fixture();mutate(raw);self.assertEqual(v.validate_raw(raw)['status'],'FAIL')

    def test_transport_omission_query_duplicates_and_pending_probe_rejected(self):
        raw=fixture();language(raw)['execution_events']=[e for e in language(raw)['execution_events'] if e.get('event')!='observation']
        self.assertEqual(v.validate_raw(raw)['status'],'FAIL')
        raw=fixture();progress(raw)['queries'].append(copy.deepcopy(progress(raw)['queries'][0]))
        self.assertEqual(v.validate_raw(raw)['status'],'FAIL')
        raw=fixture();progress(raw)['pending_event_ids']=['not-finished']
        self.assertEqual(v.validate_raw(raw)['status'],'FAIL')

    def test_four_owner_READY_exit_and_output_required_including_OCR(self):
        for mutate in [lambda raw:raw['ready'].pop('ocr#0'),lambda raw:raw['shutdown']['ocr#0'].update(exit_code=1),
                lambda raw:raw['request_records'][1]['worker_response'].update(raw_output='')]:
            raw=fixture();mutate(raw);self.assertEqual(v.validate_raw(raw)['status'],'FAIL')

    def test_exception_record_preserved_and_rejected_even_with_completion_claim(self):
        raw=fixture();progress(raw)['records'][0]['observation_error']='RuntimeError'
        result=v.validate_raw(raw)
        self.assertEqual(result['status'],'FAIL')
        self.assertEqual(result['requests'][0]['record_errors_preserved'][0]['observation_error'],'RuntimeError')


if __name__=='__main__':unittest.main()
