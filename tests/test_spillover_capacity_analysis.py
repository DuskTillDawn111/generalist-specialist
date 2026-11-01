"""CPU tests for exact, censored descriptive handler exposure accounting."""
import importlib.util
from pathlib import Path
import unittest

SPEC=importlib.util.spec_from_file_location('capacity_analysis',Path(__file__).resolve().parents[1]/'tools/spillover_capacity_20261022/analyze.py')
analysis=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(analysis)


class ActivityAccounting(unittest.TestCase):
    def test_exact_three_alias_overlap_and_request_exposure(self):
        intervals=[dict(alias='a',start_ns=0,end_ns=10),dict(alias='b',start_ns=5,end_ns=15),dict(alias='c',start_ns=7,end_ns=9)]
        result=analysis.common_activity(intervals,['a','b','c'])
        self.assertEqual(result['observed_all_active_aliases_busy_ns'],2)
        self.assertEqual(result['observed_busy_ns_by_alias'],dict(a=10,b=10,c=2))
        self.assertEqual(result['observed_concurrency_duration_ns'],{'1':10,'2':3,'3':2})
        self.assertEqual(result['common_observed_lifetime_span'],dict(start_ns=7,end_ns=9,duration_ns=2))
        exposure=analysis.interval_exposure(0,10,result['segments'],['a','b','c'],'a')
        self.assertEqual(exposure['any_other_alias_busy_fraction'],.5)
        self.assertEqual(exposure['all_active_aliases_busy_fraction'],.2)

    def test_half_open_boundary_does_not_manufacture_overlap(self):
        result=analysis.common_activity([dict(alias='a',start_ns=0,end_ns=5),dict(alias='b',start_ns=5,end_ns=10)],['a','b'])
        self.assertEqual(result['observed_all_active_aliases_busy_ns'],0)
        self.assertEqual(result['observed_concurrency_duration_ns'],{'1':10})

    def test_overlapping_same_alias_is_flagged_and_union_counted_once(self):
        result=analysis.common_activity([dict(alias='a',start_ns=0,end_ns=10),dict(alias='a',start_ns=5,end_ns=15)],['a'])
        self.assertEqual(result['observed_busy_ns_by_alias']['a'],15)
        self.assertEqual(len(result['serial_overlap_flags']['a']),1)
        self.assertNotIn('2',result['observed_concurrency_duration_ns'])

    def test_unknown_and_missing_alias_do_not_claim_complete_activity(self):
        result=analysis.common_activity([dict(alias='a',start_ns=0,end_ns=10)],['a','b'],unknown_requests=1)
        self.assertFalse(result['observation_complete'])
        self.assertIsNone(result['common_observed_lifetime_span'])
        self.assertEqual(result['observed_all_active_aliases_busy_ns'],0)
        self.assertIsNone(result['U']);self.assertFalse(result['fitted_parameters'])

    def test_common_lifetime_is_not_common_busy(self):
        intervals=[dict(alias='a',start_ns=0,end_ns=2),dict(alias='a',start_ns=8,end_ns=10),dict(alias='b',start_ns=3,end_ns=7)]
        result=analysis.common_activity(intervals,['a','b'])
        self.assertEqual(result['common_observed_lifetime_span']['duration_ns'],4)
        self.assertEqual(result['observed_all_active_aliases_busy_ns'],0)
        self.assertEqual(result['observed_concurrency_duration_ns']['0'],2)

    def test_invalid_interval_or_unexpected_alias_rejected(self):
        for interval in [dict(alias='a',start_ns=5,end_ns=5),dict(alias='a',start_ns=None,end_ns=10),dict(alias='b',start_ns=0,end_ns=1)]:
            with self.subTest(interval=interval),self.assertRaises(ValueError):analysis.common_activity([interval],['a'])




class FrozenCohortAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        cls.root=Path(__file__).resolve().parents[1]
        cls.packet_path=cls.root/'artifacts/static-spillover-study-20261022/text-capacity-diagnostic-v1/experiment-packet.json'
        cls.packet,cls.config,cls.workers,cls.references,scorer,total,cls.metadata=analysis.inspect_packet(cls.packet_path)
        cls.scorer=staticmethod(scorer);cls.total=staticmethod(total)

    def fixture(self,state='A',block=1):
        import copy
        packet=copy.deepcopy(self.packet);job=next(j for j in packet['jobs'] if j['block']==block and j['state']==state)
        start=40_000_000_000;rows=[]
        for alias_index,alias in enumerate(job['active_aliases']):
            for plan in job['planned_requests'][alias]:
                i=plan['sequence'];submit=start+i*20_000_000+alias_index*1_000_000
                reference=self.references['text'][plan['pool_id']]
                answer=(reference['answers'] or [''])[0] if i%2==0 else 'deliberately_wrong_zebra_999'
                response=dict(request_id=plan['request_id'],pool_id=plan['pool_id'],task_type='text',
                    submit_ns=submit,start_ns=submit+1_000_000,end_ns=submit+11_000_000,
                    service_ms=9.0,prediction=answer,success=True,input_tokens=20,generated_tokens=3)
                rows.append(dict(plan,phase='measurement',status='completed',scheduled_time_ns=submit,
                    enqueued_time_ns=submit,actual_submit_time_ns=submit,actual_submit_unix_ns=submit,
                    submit_write_completed_time_ns=submit+10_000,completion_time_ns=submit+12_000_000,
                    completion_unix_ns=submit+12_000_000,worker_response=response))
        raw=dict(schema='gsdeploy.balanced_closed_loop_capacity.raw.v1',protocol=packet['protocol'],status='completed',
            packet_sha256=self.metadata['packet']['sha256'],job=job,job_identity_sha256=job['job_identity_sha256'],
            replica_to_family=packet['aliases'],plan={'deployment':packet['resident_deployment']},request_records=rows,
            measurement_window=dict(start_time_ns=start,end_time_ns=start+2_000_000_000),
            memory_evidence=dict(resource_valid=True),errors=[],shutdown={})
        return packet,job,raw

    def analyze(self,fixture):
        packet,job,raw=fixture
        return analysis.analyze_job(packet,job,raw,self.metadata['packet']['sha256'],self.references,self.scorer,self.total,{'verified':True})

    def test_real_frozen_scorer_balanced_repeated_cohort(self):
        result=self.analyze(self.fixture())
        self.assertEqual(result['completion']['service_available_requests'],192)
        self.assertEqual(result['completion']['diagnostic_native_accuracy'],.5)
        self.assertEqual(result['completion']['quality_available_requests'],192)
        self.assertEqual(len(result['per_input']),64)
        self.assertTrue(all(r['quality_available_requests']==3 for r in result['per_input']))
        self.assertTrue(all(r['service_available_requests']==32 for r in result['per_repetition']))
        self.assertEqual(result['finite_cohort_timing']['completions_per_common_measurement_second'],96)
        self.assertTrue(result['finite_cohort_timing']['all_alias_cohorts_complete'])
        self.assertEqual(result['finite_cohort_timing']['post_first_alias_completion_tail_seconds'],.001)
        self.assertEqual(result['repetition_drift'][0]['repetitions'][2]['median_service_over_repetition0'],1)
        self.assertIsNone(result['U']);self.assertFalse(result['ranking_performed']);self.assertFalse(result['fitted_parameters'])
        self.assertEqual(result['request_diagnostics'][0]['ledger']['terms']['clock_span_residual_ms'],0)

    def test_missing_and_incomplete_stay_unknown_resource_failure_keeps_completed(self):
        fixture=self.fixture();raw=fixture[2];raw['status']='failed';raw['memory_evidence']['resource_valid']=False
        raw['request_records'].pop();raw['request_records'][0].update(status='incomplete',worker_response=None)
        result=self.analyze(fixture)
        self.assertEqual(result['completion']['service_available_requests'],190)
        self.assertEqual(result['completion']['service_unknown_requests'],2)
        self.assertEqual(result['completion']['recorded_requests'],191)
        self.assertIsNone(result['request_diagnostics'][-1]['service_ms'])
        self.assertFalse(result['common_activity']['observation_complete'])
        self.assertFalse(result['resource_valid'])
        self.assertIsNone(result['finite_cohort_timing']['post_first_alias_completion_tail_seconds'])

    def test_native_failure_has_service_but_unknown_quality_and_bad_identity_has_neither(self):
        fixture=self.fixture();rows=fixture[2]['request_records'];rows[0]['worker_response']['success']=False
        rows[1]['worker_response']['pool_id']='wrong_input'
        result=self.analyze(fixture)
        self.assertEqual(result['completion']['service_available_requests'],191)
        self.assertEqual(result['completion']['quality_available_requests'],190)
        self.assertEqual(result['request_diagnostics'][0]['service_status'],'AVAILABLE')
        self.assertEqual(result['request_diagnostics'][0]['quality']['status'],'UNKNOWN')
        self.assertEqual(result['request_diagnostics'][1]['service_status'],'UNKNOWN')

    def test_tampered_raw_identity_duplicate_or_plan_is_fatal(self):
        for case in ['packet','job','duplicate','plan']:
            fixture=self.fixture();raw=fixture[2]
            if case=='packet':raw['packet_sha256']='bad'
            elif case=='job':raw['job_identity_sha256']='bad'
            elif case=='duplicate':raw['request_records'].append(raw['request_records'][0])
            else:raw['request_records'][0]['repetition']=2
            with self.subTest(case=case),self.assertRaises(ValueError):self.analyze(fixture)

    def test_completed_requests_require_independent_runtime_identity(self):
        packet,job,raw=self.fixture()
        with self.assertRaises(ValueError):analysis.analyze_job(packet,job,raw,self.metadata['packet']['sha256'],self.references,self.scorer,self.total)

    def test_imbalanced_or_reordered_plan_is_rejected_even_with_recomputed_digest(self):
        for field,value in [('pool_id','nonexistent'),('sequence',99),('arrival_index',5),('repetition',2)]:
            packet,job,raw=self.fixture();job['planned_requests']['text#0'][0][field]=value
            job['job_identity_sha256']=analysis.job_digest(job)
            with self.subTest(field=field),self.assertRaises(ValueError):analysis.validate_job(job,packet['aliases'])

    def runtime_fixture(self,state='A',block=1):
        import copy,json,os
        packet,job,raw=self.fixture(state,block);registry={};ready={}
        for alias,family in packet['aliases'].items():
            entry=copy.deepcopy(self.workers[family]);env=entry.setdefault('env',{})
            env['PYTHONPATH']=packet['frozen_source']+os.pathsep+env.get('PYTHONPATH','inherited_path')
            env.setdefault('CUDA_VISIBLE_DEVICES','0');entry.setdefault('cwd',packet['frozen_source']);registry[alias]=entry
            specpath=Path(entry.get('spec') or entry['command'][entry['command'].index('--spec')+1]);spec=json.loads(specpath.read_text())
            ready[alias]=dict(worker=spec['candidate']['slot'],model=spec['candidate']['model'],
                spec_sha256=analysis.file_receipt(specpath)['sha256'],tasks=self.config['catalog']['models'][family]['supported_tasks'])
        transport=dict(deployment={a:1 for a in packet['aliases']},workload_rps={'text':1},
            routing_rps={'text':{a:1/len(job['active_aliases']) for a in job['active_aliases']}},
            warmup_seconds=30,measurement_seconds=600,arrival_process='deterministic_periodic',random_seed=job['seed'],
            routing_source=packet['protocol'],startup_timeout_seconds=300,drain_timeout_seconds=300,shutdown_timeout_seconds=10,
            max_pending_per_model=1,idle_residency=False)
        def phase_row(plan,phase,index,complete):
            row=dict(plan,request_id=f'{phase}:{index}',phase=phase,status='completed',completion_time_ns=complete)
            row['worker_response']=dict(request_id=row['request_id'],pool_id=row['pool_id'],task_type=row['task'],success=True)
            return row
        raw.update(config=copy.deepcopy(self.config),worker_registry=registry,worker_ready=ready,transport_config=transport,
            preflight=dict(sample_pool_sha256=packet['bindings'][packet['samples']],gpu={'uuid':packet['gpu_uuid']},ready=True,gpu_checked=True),
            gpu_boundary_preflight=dict(uuid=packet['gpu_uuid'],compute_pids=[]),priming_path_identity={'verified':True},
            gpu_before_worker_start=dict(uuid=packet['gpu_uuid'],compute_pids=[]),
            priming_request_records=[phase_row(p,'priming',i,100_000_000) for i,p in enumerate(packet['priming_requests'])],
            warmup_request_records=[phase_row(dict(model=a,task='text',pool_id=job['input_pool_ids'][0]),'warmup',i,31_000_000_000) for i,a in enumerate(job['active_aliases'])],
            warmup_window=dict(start_time_ns=1_000_000_000,end_time_ns=32_000_000_000))
        raw['cpu_launcher_checks']={family:dict(command=['--check' if arg=='--execute' else arg for arg in worker['command']]+['--pool',packet['samples']],
            exit_code=0,stdout=json.dumps({'gpu_started':False})) for family,worker in self.workers.items() if family in packet['resident_deployment']}
        admission=json.loads(Path(packet['admission']).read_text());quality=self.config['catalog']['quality']
        raw['priming_path_identity'].update(contract='quality_evidence_v1',manifest_sha256=packet['bindings'][packet['admission']],
            quality_sha256=analysis.digest(quality),checkpoints={family:admission['models'][family]['checkpoint']['sha256'] for family in packet['resident_deployment']},
            reports=[dict(model=family,**r) for family in packet['resident_deployment'] for r in admission['models'][family]['quality_reports']],
            pairs=[dict(model=family,task=task,measurement=quality['paths'][family][task],
                runtime_checkpoint_sha256=admission['models'][family]['checkpoint']['sha256'],offline_full_checkpoint_equivalence_verified=False)
                for family in packet['resident_deployment'] for task in self.config['catalog']['models'][family]['supported_tasks']])
        return packet,raw

    def test_runtime_checks_accept_exact_bound_identity(self):
        packet,raw=self.runtime_fixture()
        result=analysis.runtime_identity_checks(packet,raw,self.config,self.workers)
        self.assertTrue(result['verified'],result['checks'])

    def test_runtime_identity_rejects_ready_transport_priming_or_warmup_tamper(self):
        for case in ['ready','transport','registry','priming','warmup','config','launcher','priming_binding']:
            packet,raw=self.runtime_fixture()
            if case=='ready':raw['worker_ready']['text#0']['spec_sha256']='bad'
            elif case=='transport':raw['transport_config']['max_pending_per_model']=2
            elif case=='registry':raw['worker_registry']['text#0']['env']['PYTHONPATH']='wrong'
            elif case=='priming':raw['priming_request_records'].pop()
            elif case=='warmup':raw['warmup_request_records'][0]['completion_time_ns']=50_000_000_000
            elif case=='launcher':raw['cpu_launcher_checks']['text']['stdout']='{"gpu_started":true}'
            elif case=='priming_binding':raw['priming_path_identity']['pairs'].pop()
            else:raw['config']['gpu_memory_mib']=1
            result=analysis.runtime_identity_checks(packet,raw,self.config,self.workers)
            with self.subTest(case=case):self.assertFalse(result['verified'],result['checks'])

    def test_same_alias_same_input_pairing_reports_block_comparison_without_pooling(self):
        a=self.analyze(self.fixture('A'));c=self.analyze(self.fixture('C'))
        result=analysis.pair_job_results(a,c,'block1_A_to_C')
        self.assertEqual(len(result),64)
        self.assertTrue(all(r['right_over_left']==1 for r in result))
        self.assertTrue(all(r['left_completed']==3 and r['right_completed']==3 for r in result))

    def test_check_only_never_opens_progress_or_raw(self):
        import contextlib,io
        from unittest import mock
        with mock.patch('sys.argv',['analyze','--packet',str(self.packet_path),'--check-only','--queue-progress','/never/read/me']),mock.patch.object(analysis,'registered_paths',side_effect=AssertionError('must not read progress')),mock.patch.object(analysis,'read_final',side_effect=AssertionError('must not read raw')),contextlib.redirect_stdout(io.StringIO()) as output:
            analysis.main()
        import json
        result=json.loads(output.getvalue());self.assertEqual(result['raw_files_read'],0);self.assertFalse(result['progress_read'])


class ReceiptAndCLISafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FrozenCohortAnalysis.setUpClass()

    def test_final_sha_is_checked_before_json_parse_and_newline_required(self):
        import hashlib,tempfile
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'run.json';path.write_bytes(b'not-json\n')
            with self.assertRaisesRegex(ValueError,'SHA'):analysis.read_final(path,'wrong')
            path.write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError,'incomplete'):analysis.read_final(path,hashlib.sha256(b'{}').hexdigest())
            path.write_bytes(b'{}\n')
            raw,receipt=analysis.read_final(path,hashlib.sha256(b'{}\n').hexdigest())
            self.assertEqual(raw,{});self.assertEqual(receipt['bytes'],3)

    def test_nonterminal_unregistered_and_missing_receipts_never_read(self):
        import json,tempfile
        from unittest import mock
        f=FrozenCohortAnalysis();packet,job,raw=f.fixture()
        with tempfile.TemporaryDirectory() as temp:
            progress=Path(temp)/'queue-progress.json'
            progress.write_text(json.dumps(dict(schema='gsdeploy.capacity_queue.v1',status='RUNNING',packet_sha256=f.metadata['packet']['sha256'],
                jobs=[dict(job_id=job['job_id'],status='measurement',raw={'path':str(Path(temp)/'run.json'),'sha256':'fake'})])))
            with mock.patch.object(analysis,'read_final',side_effect=AssertionError('must not read')):
                waiting,ready=analysis.registered_paths(packet,[progress],f.metadata['packet']['sha256'])
            self.assertEqual(len(waiting),6);self.assertEqual(ready,[])

    def cli_fixture(self,temp):
        import json
        f=FrozenCohortAnalysis();packet,raw=f.runtime_fixture();raw['shutdown']={}
        raw_path=Path(temp)/'attempt1'/'run.json';raw_path.parent.mkdir();raw_path.write_text(json.dumps(raw)+'\n')
        progress=Path(temp)/'queue-progress.json';progress.write_text(json.dumps(dict(schema='gsdeploy.capacity_queue.v1',status='RUNNING',
            packet_sha256=f.metadata['packet']['sha256'],jobs=[dict(job_id=raw['job']['job_id'],status='completed',raw=analysis.file_receipt(raw_path))]))+'\n')
        return f,progress

    def test_default_gate_reads_zero_raw_when_other_jobs_are_unfinished(self):
        import contextlib,io,json,tempfile
        from unittest import mock
        with tempfile.TemporaryDirectory() as temp:
            f,progress=self.cli_fixture(temp)
            with mock.patch('sys.argv',['analyze','--packet',str(f.packet_path),'--queue-progress',str(progress),'--output',str(Path(temp)/'output')]),mock.patch.object(analysis,'read_final',side_effect=AssertionError('must not open terminal subset by default')),contextlib.redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(SystemExit) as error:analysis.main()
            self.assertEqual(error.exception.code,3);self.assertEqual(json.loads(output.getvalue())['raw_files_read'],0)
            self.assertFalse((Path(temp)/'output').exists())

    def test_explicit_partial_end_to_end_is_incomplete_and_preserves_receipts(self):
        import contextlib,io,json,tempfile
        from unittest import mock
        with tempfile.TemporaryDirectory() as temp:
            f,progress=self.cli_fixture(temp);destination=Path(temp)/'output'
            argv=['analyze','--packet',str(f.packet_path),'--queue-progress',str(progress),'--output',str(destination),'--partial']
            with mock.patch('sys.argv',argv),contextlib.redirect_stdout(io.StringIO()):analysis.main()
            report=json.loads((destination/'report.json').read_text());manifest=json.loads((destination/'manifest.json').read_text())
            self.assertEqual(report['status'],'INCOMPLETE');self.assertEqual(report['raw_files_read'],1);self.assertEqual(len(report['waiting']),5)
            self.assertIsNone(report['U']);self.assertFalse(report['ranking_performed']);self.assertFalse(report['independent_formal_validation'])
            for filename,sha in manifest['files'].items():self.assertEqual(analysis.file_receipt(destination/filename)['sha256'],sha)
            self.assertIn('不判断最终重复性',(destination/'report.md').read_text())
            with mock.patch('sys.argv',argv),self.assertRaises(FileExistsError):analysis.main()


    def test_all_six_default_gate_then_complete_analysis(self):
        import contextlib,io,json,tempfile
        from unittest import mock
        f=FrozenCohortAnalysis()
        with tempfile.TemporaryDirectory() as temp:
            registrations=[]
            for job in f.packet['jobs']:
                packet,raw=f.runtime_fixture(job['state'],job['block'])
                path=Path(temp)/job['job_id']/'run.json';path.parent.mkdir();path.write_text(json.dumps(raw)+'\n')
                registrations.append(dict(job_id=job['job_id'],status='completed',raw=analysis.file_receipt(path)))
            queue=dict(schema='gsdeploy.capacity_queue.v1',status='RUNNING',packet_sha256=f.metadata['packet']['sha256'],jobs=registrations)
            progress=Path(temp)/'queue-progress.json';progress.write_text(json.dumps(queue)+'\n')
            destination=Path(temp)/'output'
            argv=['analyze','--packet',str(f.packet_path),'--queue-progress',str(progress),'--output',str(destination)]
            with mock.patch('sys.argv',argv),mock.patch.object(analysis,'read_final',side_effect=AssertionError('final queue publication required')),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as error:analysis.main()
            self.assertEqual(error.exception.code,3)
            queue['status']='ALL_SIX_DIAGNOSTIC_ATTEMPTS_COMPLETE';progress.write_text(json.dumps(queue)+'\n')
            with mock.patch('sys.argv',argv),contextlib.redirect_stdout(io.StringIO()):analysis.main()
            report=json.loads((destination/'report.json').read_text())
            self.assertEqual(report['status'],'ALL_SIX_DEVELOPMENT_DIAGNOSTICS_COMPLETE')
            self.assertEqual(report['raw_files_read'],6);self.assertEqual(report['waiting'],[])
            self.assertEqual(len(report['common_input_comparisons']),384)
            self.assertFalse(report['independent_formal_validation']);self.assertIsNone(report['U'])


if __name__=='__main__':unittest.main()
