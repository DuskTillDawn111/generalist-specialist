"""CPU contract tests for completed-service accounting and no-raw gates."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SPEC=importlib.util.spec_from_file_location('diag',Path(__file__).with_name('diagnose_text26_service.py'))
diag=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diag)


class DiagnosticContracts(unittest.TestCase):
    def fixture(self):
        mono=100_000_000
        unix=1_700_000_000_000_000_000
        identity=dict(seed=51)
        job=dict(id='job',labels=['synthetic'],seed=51,identity=identity,
            plan=dict(deployment={'text':1},routing={'text':{'text':1}}))
        records=[]
        for k,(submit,start,end,service) in enumerate([(0,.5,3.5,2.5),(1,4,6,1.5)]):
            ns=lambda x:round(x*1e6)
            w=dict(request_id=str(k),pool_id=str(k),submit_ns=unix+ns(submit),
                start_ns=unix+ns(start),end_ns=unix+ns(end),service_ms=service)
            records.append(dict(request_id=str(k),arrival_index=k,phase='measurement',task='text',
                model='text#0',pool_id=str(k),status='completed',scheduled_time_ns=mono+ns(submit),
                enqueued_time_ns=mono+ns(submit),actual_submit_time_ns=mono+ns(submit),
                actual_submit_unix_ns=unix+ns(submit),submit_write_completed_time_ns=mono+ns(submit+.1),
                completion_time_ns=mono+ns(end+.25),completion_unix_ns=unix+ns(end+.25),
                worker_latency_ms=end-start,worker_queue_wait_ms=start-submit,worker_response=w))
        raw=dict(status='completed',experiment_phase='calibration',experiment_job_id='job',plan=job['plan'],
            transport_config={'random_seed':51},executed_measurement_identity=identity,
            executed_measurement_id=diag.quality_digest(identity),errors=[],
            evaluation={'utility':{'eligible':True}},shutdown={'text#0':dict(forced=False,exit_code=0,bye_received=True)},
            request_records=records,measurement_window=dict(offered_requests=2,drain_end_time_ns=123),
            replica_to_family={'text#0':'text'})
        data=dict(quality={'input_order':{'text':['0','1']}},calibrated_replay_v2={'input_baselines':
            {'text':{'text':{'1':{'samples':{str(k):{'median_service_ms':2} for k in range(2)}}}}}})
        return job,raw,data

    def test_fifo_uses_observed_submit_and_reports_unobserved_gaps(self):
        job,raw,data=self.fixture()
        result=diag.analyze_run(job,raw,{},data)
        rows=result['requests']
        self.assertEqual([r['fifo_diagnostic']['observed_start_minus_max_submit_previous_observed_end_ms'] for r in rows],[.5,.5])
        self.assertEqual([r['fifo_diagnostic']['handler_fifo_observed_minus_replay_end_ms'] for r in rows],[.5,1.])
        self.assertEqual([r['fifo_diagnostic']['service_fifo_observed_minus_replay_end_ms'] for r in rows],[1.,2.])
        self.assertEqual([r['ledger']['duration_decomposition_residual_ms'] for r in rows],[0.,0.])
        self.assertEqual([r['ledger']['handler_field_residual_ms'] for r in rows],[0.,0.])

    def test_incomplete_request_is_preserved_and_does_not_bridge_fifo(self):
        job,raw,data=self.fixture();raw['request_records'][0]['status']='submitted'
        result=diag.analyze_run(job,raw,{},data)
        self.assertEqual(result['cohort']['unknown_requests'],1)
        self.assertEqual(result['requests'][0]['diagnostic_status'],'UNKNOWN')
        self.assertEqual(result['requests'][0]['original_record'],raw['request_records'][0])
        self.assertIsNone(result['requests'][0]['service_ms'])
        self.assertEqual(result['requests'][1]['fifo_diagnostic']['status'],'UNKNOWN_CENSORED')
        self.assertEqual(result['fifo_by_alias']['text#0']['handler_fifo_end_residual_ms']['count'],0)
        self.assertEqual(result['task_sequences']['text']['sample_sequence'],['0','1'])

    def test_terminal_failed_ineligible_missing_drain_still_has_completed_ledgers(self):
        job,raw,data=self.fixture()
        raw.update(status='failed',errors=['timeout'],shutdown={})
        raw['evaluation']['utility']=dict(eligible=False,U=None)
        raw['measurement_window'].pop('drain_end_time_ns')
        result=diag.analyze_run(job,raw,{},data)
        self.assertEqual(result['cohort']['analyzed_completed_requests'],2)
        self.assertEqual(result['terminal_status'],'failed')
        self.assertIsNone(result['diagnostic_U'])
        self.assertFalse(result['ranking_performed'])
        self.assertEqual(result['fifo_by_alias']['text#0']['status'],'UNKNOWN_CENSORED')
        self.assertGreater(len(result['censoring_limitations']),1)

    def test_completed_resource_invalid_never_implies_eligible(self):
        job,raw,data=self.fixture()
        raw['evaluation']['utility'].update(eligible=False,M_norm=1.0219675952787954)
        raw['evaluation']['stability']={'stability_state_v2':'recoverable_congested'}
        raw['memory_evidence']=dict(resource_valid=False,safe_memory_mib=50219.487632)
        result=diag.analyze_run(job,raw,{},data,gpu_memory_mib=49140)
        self.assertEqual(result['terminal_status'],'completed')
        self.assertIs(result['original_eligible'],False)
        self.assertIs(result['resource_evidence']['resource_valid'],False)
        self.assertEqual(result['resource_evidence']['gpu_memory_mib'],49140)
        self.assertEqual(result['original_stability_state'],'recoverable_congested')
        self.assertEqual(result['cohort']['analyzed_completed_requests'],2)
        self.assertIsNone(result['diagnostic_U'])

    def test_no_completed_requests_and_malformed_completed_request(self):
        job,raw,data=self.fixture()
        raw['request_records'][0]['status']='incomplete'
        raw['request_records'][1]['worker_response'].pop('start_ns')
        result=diag.analyze_run(job,raw,{},data)
        self.assertEqual(result['cohort']['unknown_requests'],2)
        self.assertEqual(result['ledger_summary'],{})
        self.assertEqual(result['paths'],[])
        self.assertEqual(result['fifo_by_alias']['text#0']['analyzed_completed_requests'],0)
        self.assertTrue(all(r['diagnostic_status']=='UNKNOWN' for r in result['requests']))

    def test_unknown_warmup_does_not_claim_known_fifo_initial_state(self):
        job,raw,data=self.fixture()
        raw['warmup_request_records']=[dict(model='text#0',status='incomplete',worker_response=None)]
        result=diag.analyze_run(job,raw,{},data)
        self.assertEqual(result['cohort']['analyzed_completed_requests'],2)
        self.assertFalse(result['fifo_by_alias']['text#0']['full_trace_valid'])

    def test_aborted_window_drain_marker_is_not_successful_drain(self):
        job,raw,data=self.fixture();raw['measurement_window']['aborted']=True
        result=diag.analyze_run(job,raw,{},data)
        self.assertFalse(result['fifo_by_alias']['text#0']['full_trace_valid'])
        self.assertTrue(any('aborted' in v for v in result['censoring_limitations']))

    def test_missing_records_are_disclosed_without_inventing_services(self):
        job,raw,data=self.fixture();raw['measurement_window']['offered_requests']=4
        result=diag.analyze_run(job,raw,{},data)
        self.assertEqual(result['cohort']['unrecorded_requests'],2)
        self.assertEqual(len(result['requests']),2)
        self.assertFalse(result['fifo_by_alias']['text#0']['full_trace_valid'])

    def test_identity_mismatch_and_nonterminal_raw_rejected(self):
        job,raw,data=self.fixture();raw['executed_measurement_id']='wrong'
        with self.assertRaisesRegex(ValueError,'identity'):
            diag.analyze_run(job,raw,{},data)
        job,raw,data=self.fixture();raw['status']='running'
        with self.assertRaisesRegex(ValueError,'terminal declared'):
            diag.analyze_run(job,raw,{},data)

    def test_wall_clock_span_difference_is_reported_not_forced_zero(self):
        job,raw,data=self.fixture()
        raw['request_records'][0]['completion_unix_ns']+=100_000
        result=diag.analyze_run(job,raw,{},data)
        self.assertAlmostEqual(result['requests'][0]['ledger']['duration_decomposition_residual_ms'],-.1)

    def test_check_only_never_invokes_progress_or_raw_reader(self):
        metadata={'raw_files_read':0}
        with mock.patch.object(diag,'inspect_metadata',return_value=({}, {},metadata)), \
             mock.patch.object(diag,'completed_paths',side_effect=AssertionError('progress read')), \
             mock.patch.object(diag,'load_final_json',side_effect=AssertionError('raw read')), \
             mock.patch('sys.argv',['diag','--check-only']),mock.patch('builtins.print'):
            diag.main()

    def test_waiting_progress_does_not_open_raw(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'progress.json'
            path.write_text(json.dumps({'jobs':[dict(job_id='a',status='RUNNING',output='/do/not/read/run.json')]}))
            with mock.patch.object(diag,'load_final_json',side_effect=AssertionError('raw read')):
                waiting,paths=diag.completed_paths({'jobs':[dict(id='a',labels=['a'])]},path)
            self.assertEqual(len(waiting),1);self.assertEqual(paths,[])

    def test_merge_explicit_receipts_accepts_failed_and_deduplicates_consistently(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);final=root/'run.json';final.write_text('{}\n')
            row=dict(job_id='a',status='failed',eligible=False,exit_code=1,output=str(final))
            first=root/'first.json';second=root/'second.json'
            first.write_text(json.dumps({'status':'STOPPED_FOR_REVIEW','jobs':[row]}))
            preserved=dict(row);preserved.pop('output');preserved['raw']=diag.receipt(final);preserved['attempt']=1
            second.write_text(json.dumps({'status':'ALL_PREDECLARED_ATTEMPTS_COLLECTED','preserved_attempts':[preserved],'jobs':[]}))
            waiting,paths=diag.completed_paths({'jobs':[dict(id='a',labels=['a'])]},[first,second])
            self.assertEqual(waiting,[]);self.assertEqual(len(paths),1)
            self.assertEqual(paths[0][2]['_sha256'],diag.sha(final))
            preserved['raw']['path']=str(root/'another-attempt'/'run.json')
            second.write_text(json.dumps({'jobs':[preserved]}))
            with self.assertRaisesRegex(ValueError,'Conflicting registration'):
                diag.completed_paths({'jobs':[dict(id='a',labels=['a'])]},[first,second])

    def test_registered_sha_is_checked_before_json_parse(self):
        with tempfile.TemporaryDirectory() as td:
            final=Path(td)/'run.json';final.write_text('not JSON\n')
            with self.assertRaisesRegex(ValueError,'SHA256 differs'):
                diag.load_final_json(final,'wrong')

    def test_wrapper_execution_error_merges_with_terminal_raw_receipt_in_either_order(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);output=root/'condition-attempt1';output.mkdir()
            final=output/'run.json';final.write_text('{}\n')
            wrapper=root/'original.json';continuation=root/'remaining.json'
            wrapper.write_text(json.dumps({'jobs':[dict(job_id='a',status='EXECUTION_ERROR',
                output=str(output),exit_code=1)]}))
            preserved=dict(job_id='a',status='failed',eligible=False,attempt=1,raw=diag.receipt(final))
            continuation.write_text(json.dumps({'preserved_attempts':[preserved],'jobs':[]}))
            packet={'jobs':[dict(id='a',labels=['a'])]}
            for receipts in [[wrapper,continuation],[continuation,wrapper]]:
                with self.subTest(receipts=receipts):
                    waiting,paths=diag.completed_paths(packet,receipts)
                    self.assertEqual(waiting,[]);self.assertEqual(len(paths),1)
                    self.assertEqual(paths[0][1],final)
                    registration=paths[0][2]
                    self.assertEqual(registration['status'],'failed')
                    self.assertEqual(registration['wrapper_status'],'EXECUTION_ERROR')
                    self.assertEqual(registration['exit_code'],1)
                    self.assertEqual(registration['_sha256'],diag.sha(final))
            with mock.patch.object(diag,'load_final_json',side_effect=AssertionError('raw read')):
                waiting,paths=diag.completed_paths(packet,[wrapper])
            self.assertEqual(paths,[])
            self.assertEqual(waiting[0]['metadata_status'],'EXECUTION_ERROR')
            # A mere terminal label without the explicit raw hash receipt must
            # not promote the wrapper error to an analyzable raw run.
            preserved['raw'].pop('sha256')
            continuation.write_text(json.dumps({'jobs':[preserved]}))
            waiting,paths=diag.completed_paths(packet,[wrapper,continuation])
            self.assertEqual(paths,[]);self.assertEqual(len(waiting),1)

    def test_partial_main_opens_only_registered_terminal_and_published_jobs(self):
        job,raw,data=self.fixture()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);final=root/'run.json';final.write_text(json.dumps(raw)+'\n')
            progress=root/'progress.json'
            progress.write_text(json.dumps({'status':'RUNNING','jobs':[
                dict(job_id=job['id'],status='completed',eligible=True,output=str(final)),
                dict(job_id='running',status='RUNNING',output='/never/read/run.json')]}))
            packet=dict(config={'gpu_memory_mib':49140},jobs=[job,dict(id='running',labels=['running']),dict(id='unregistered',labels=['unregistered'])])
            marker=root/'metadata';marker.write_text('immutable')
            metadata=dict(packet=diag.receipt(marker),baseline=diag.receipt(marker),
                tool=diag.receipt(Path(diag.__file__)),semantic_source_receipts=[],actual_worker_semantic_sources=[])
            with mock.patch.object(diag,'inspect_metadata',return_value=(packet,data,copy.deepcopy(metadata))), \
                 mock.patch.object(diag,'load_final_json',wraps=diag.load_final_json) as reader, \
                 mock.patch('sys.argv',['diag','--queue-progress',str(progress),'--output',str(root/'out')]), \
                 mock.patch('builtins.print'):
                with self.assertRaises(SystemExit) as stopped:diag.main()
                self.assertEqual(stopped.exception.code,3);reader.assert_not_called()
            with mock.patch.object(diag,'inspect_metadata',return_value=(packet,data,copy.deepcopy(metadata))), \
                 mock.patch.object(diag,'load_final_json',wraps=diag.load_final_json) as reader, \
                 mock.patch('sys.argv',['diag','--queue-progress',str(progress),'--output',str(root/'out'),'--partial']), \
                 mock.patch('builtins.print'):
                diag.main()
                reader.assert_called_once_with(final,None)
            report=json.loads((root/'out/report.json').read_text())
            self.assertEqual(report['raw_files_read'],1)
            self.assertEqual(len(report['waiting']),2)
            self.assertIsNone(report['diagnostic_U'])

    def test_reject_partial_filename_before_open(self):
        with self.assertRaisesRegex(ValueError,'exact final run.json'):
            diag.load_final_json('/do/not/read/run.json.partial')


if __name__=='__main__':
    unittest.main()
