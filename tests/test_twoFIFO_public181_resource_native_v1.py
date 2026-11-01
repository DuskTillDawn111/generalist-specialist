import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027 import twoFIFO_public181_resource_execute_v1 as m
from tools.predictor_validation_20261027 import twoFIFO_public181_resource_review_v1 as review
from tools.predictor_validation_20261027 import twoFIFO_public181_resource_intake_v1 as intake
from tools.predictor_validation_20261027 import twoFIFO_public181_quality_labels_v1 as quality
from tools.predictor_validation_20261027.stage_response_features_v1 import checked,receipt
from tests.test_twoFIFO_fresh181_execute_v1 import Clock,Runtime
from tests.test_twoFIFO_owned_collect_v2 import FakeMonitor

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/'artifacts/predictor-public181-resource-campaign-v1-20261029-cpu02-prepared/manifest.json'
SECOND=10**9


def resources():
    return dict(shutdown_memory_sampled=False,
        lifecycle_boundaries=[dict(boundary=k,time_ns=t*SECOND) for k,t in
            [('startup_begin',0),('priming_end',2),('monitoring_end',513),('shutdown_begin',514)]],
        board_samples=[dict(time_ns=t*SECOND,watts=100.,owned_used_mib=
            200. if t==500 else 9999. if t==513 else 100.) for t in range(514)],
        cases=[dict(case_id='original181')],measurement_windows=[dict(case_id='original181',
            start_ns=302*SECOND,end_ns=392*SECOND,fixed_drain_deadline_ns=512*SECOND)])


class NativeResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory();cls.folder=Path(cls.temporary.name)/'prepared'
        cls.manifest_ref=m.prepare(CAMPAIGN,cls.folder);cls.manifest=checked(cls.manifest_ref)
        cls.plan=checked(cls.manifest['attempts'][0]['plan'])
    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    def collect(self,*,incomplete=False,unsettled=False,wrong_spec=False,checkpoint_late=False,
                late_dispatch=False,enqueue_error=False,drain_error=False,monitor_error=False):
        clock=Clock();holder={}
        class TestRuntime(Runtime):
            def enqueue(self,row,sample):
                if row['phase']!='priming' and enqueue_error:raise RuntimeError('CPU enqueue failed')
                super().enqueue(row,sample)
                if row['phase']!='priming' and late_dispatch and not hasattr(self,'was_late'):
                    self.was_late=True;clock.now+=m.MEASUREMENT_NS+m.DRAIN_NS
            def drain(self,rows,timeout):
                if rows and rows[0]['phase']!='priming' and drain_error:raise RuntimeError('CPU drain failed')
                result=super().drain(rows,timeout)
                for row in rows:
                    if row['phase']=='priming' and row.get('worker_response'):
                        row['worker_response'].update(request_id=row['request_id'],task=row['task'],task_type=row['task'])
                        row['execution_events'][0]['request_id']=row['request_id']
                return result
        class Monitor(FakeMonitor):
            def check(self):
                if monitor_error and self.phase=='measurement':raise RuntimeError('CPU monitor failed')
        with tempfile.TemporaryDirectory() as directory:
            plan=copy.deepcopy(self.plan);path=Path(directory)/'attempt'/'plan.json'
            path.parent.mkdir();path.write_text(json.dumps(plan));output=path.parent/'collection'
            def factory(config,registry):
                holder['config']=config;holder['registry']=registry
                holder['runtime']=TestRuntime(clock,registry,incomplete=incomplete,unsettled=unsettled,wrong_spec=wrong_spec)
                return holder['runtime']
            def ownership(values,*args,**kwargs):
                pid=next(iter(values));return dict(owned_aliases=[m.OWNERS[pid-1]],unresolved_nvml_pids=[],
                    foreign_pids=[],ownership_evidence=[dict(CPU_TEST_ONLY=True,live=True)])
            original=m.atomic_save
            def save(target,payload):
                if checkpoint_late and payload.get('checkpoint_context',{}).get('kind')=='precase':
                    clock.now+=m.LEAD_NS
                return original(target,payload)
            with patch.object(m,'audit_plan',return_value=plan),patch.object(m.supervisor,'validate_lease'),\
                    patch.object(m,'atomic_save',side_effect=save):
                raw=m.collect(path,output,lease_fd=-1,runtime_factory=factory,monitor_factory=Monitor,
                    clock_ns=clock.clock,wait_until_ns=clock.wait,ownership_check=ownership)
            self.assertEqual(raw,checked(receipt(output/'raw.json')))
            ledger=checked(receipt(output/m.ACTIVITY_FILE))
            return copy.deepcopy(raw),holder['runtime'],clock,ledger

    def test_new_eight_plans_preserve_public181_and_bind_complete_source_and_quality(self):
        manifest=m.audit(self.manifest_ref['path']);parent=m.campaign.audit(CAMPAIGN)
        old={a['execution_id'] for a in parent['training_attempts']+parent['independent_validation_references']}
        self.assertEqual(len(manifest['attempts']),8)
        for item,original in zip(manifest['attempts'],parent['training_attempts']):
            plan=m.audit_plan(item['plan']['path']);base=checked(original['plan'])
            self.assertNotIn(plan['execution_id'],old);self.assertEqual(plan['public_case'],base['public_case'])
            self.assertEqual(plan['runtime_config'],base['runtime_config']);self.assertEqual(plan['cases'],base['cases'])
            self.assertEqual(plan['priming_semantics'],base['priming_semantics'])
            self.assertEqual((plan['measurement_ns'],plan['drain_ns'],plan['precohort_lead_seconds']),
                (90*SECOND,120*SECOND,300))
            self.assertEqual(len(m.prepared_references(plan,plan['quality_catalog'])),22)
            self.assertIsNone(plan['runtime_policy']['admission'])
        self.assertFalse(manifest['GPU_started']);self.assertFalse(manifest['final_gain_eligible'])

    def test_actual_control_flow_full_independent_enqueue_and_protected_window(self):
        raw,runtime,clock,ledger=self.collect()
        self.assertEqual(raw['status'],'completed',raw['errors']);self.assertTrue(runtime.closed)
        self.assertEqual(len(runtime.enqueued),199);self.assertEqual(runtime.drain_counts[-1],199)
        self.assertEqual(raw['backend'],'CPU_TEST_ONLY');self.assertFalse(raw['training_feature_extraction_eligible'])
        prefix=review.inspect_precohort_prefix(self.plan,raw);self.assertEqual(prefix['actual_prior_priming'],18)
        window=raw['measurement_windows'][0];epoch=window['start_ns'];deadline=window['fixed_drain_deadline_ns']
        self.assertEqual(epoch-raw['case_histories'][0]['prior_history_snapshot']['snapshot_time_ns'],300*SECOND)
        self.assertEqual((window['end_ns']-epoch,deadline-window['end_ns']),(90*SECOND,120*SECOND))
        self.assertEqual([r['actual_submit_time_ns']-epoch for r in raw['request_records']],
            [a['arrival_ns'] for a in self.plan['cases'][0]['arrivals']])
        self.assertEqual(clock.now,deadline+1);self.assertEqual(raw['board_samples'][-1]['phase'],'final_drain')
        for span in ledger['checkpoint_spans']:
            self.assertTrue(span['end_ns']<epoch or span['start_ns']>deadline,span)
        self.assertTrue(raw['no_mutating_checkpoint_during_case'])
        activity=review.inspect_checkpoint_activity(self.plan,raw,ledger)
        self.assertEqual(activity['critical_interval_overlaps'],0)
        self.assertEqual(review.inspect_histories(self.plan,raw)['full_prior_histories'],1)

    def test_missing_ready_wrong_worker_and_checkpoint_overrun_send_no_target(self):
        for options in (dict(unsettled=True),dict(wrong_spec=True),dict(checkpoint_late=True)):
            with self.subTest(options=options):
                raw,runtime,_,_=self.collect(**options)
                self.assertEqual(raw['status'],'failed');self.assertEqual(len(runtime.enqueued),18)
                self.assertEqual(len(raw['request_records']),181);self.assertTrue(runtime.closed)
                self.assertTrue(all(r['status']=='not_started' for r in raw['request_records']))

    def test_incomplete_and_late_dispatch_keep_all_offered_and_fixed_deadline(self):
        for options in (dict(incomplete=True),dict(late_dispatch=True)):
            raw,runtime,_,_=self.collect(**options)
            self.assertEqual(raw['status'],'completed',raw['errors']);self.assertTrue(runtime.closed)
            self.assertFalse(raw['all_requests_completed']);self.assertTrue(raw['fixed_resource_observation_complete'])
            self.assertEqual(len(raw['request_records']),181)
            self.assertIsNone(raw['U']);self.assertFalse(raw['training_feature_extraction_eligible'])
            if options.get('incomplete'):self.assertEqual(len(runtime.enqueued),199)
            else:self.assertEqual(len(runtime.enqueued),19)
            self.assertEqual(raw['measurement_windows'][0]['fixed_drain_deadline_ns']-
                raw['measurement_windows'][0]['start_ns'],210*SECOND)

    def test_enqueue_drain_monitor_failure_still_final_samples_before_close(self):
        for options in (dict(enqueue_error=True),dict(drain_error=True),dict(monitor_error=True)):
            raw,runtime,_,_=self.collect(**options)
            self.assertEqual(raw['status'],'completed' if options.get('enqueue_error') else 'failed');self.assertTrue(runtime.closed)
            self.assertEqual(raw['board_samples'][-1]['phase'],'final_drain')
            self.assertEqual([b['boundary'] for b in raw['lifecycle_boundaries']][-2:],['shutdown_begin','shutdown_end'])

    def test_prefix_rederives_exact_priming_and_rejects_future_or_forged_seed(self):
        raw,_,_,_=self.collect();source=raw['case_histories'][0]['prior_history_snapshot']
        epoch=raw['measurement_windows'][0]['start_ns']
        changes=[lambda s:s['request_records'].pop(),lambda s:s.update(queued_runtime_events=[{}]),
            lambda s:s['request_records'][0].update(submit_write_completed_time_ns=epoch+1),
            lambda s:s['ready'][m.OWNERS[0]].update(spec_sha256='0'*64)]
        for change in changes:
            value=copy.deepcopy(source);change(value)
            with self.assertRaises((ValueError,UnknownEvidence)):m.settled_priming(self.plan,value,epoch)
        raw['precohort_training_prefix']['settled_seed']['sequence']+=1
        with self.assertRaises(ValueError):review.inspect_precohort_prefix(self.plan,raw)

    def test_resource_tuple_original_energy_and_memory_scopes_analytic(self):
        out=review.resource_facts(resources());joint=out['joint_tuple']
        self.assertEqual(joint,dict(startup_joules=30200.,premeasurement_owned_sampled_peak_mib=100.,
            measurement_through_drain_owned_sampled_peak_mib=200.,measurement_board_joules=9000.))
        self.assertEqual(out['observed_original_resources']['cases'][0]['drain_energy_joules_diagnostic_only'],12000.)
        self.assertFalse(out['startup_amortization_applied']);self.assertFalse(out['safe_factor_applied'])
        changed=resources();changed['board_samples'][500]['watts']=10000.
        self.assertEqual(review.resource_facts(changed)['joint_tuple'],joint)

    def test_resource_missing_brackets_wrong_horizon_and_shutdown_rejected(self):
        for mutate in (lambda r:r['measurement_windows'][0].update(end_ns=312*SECOND),
            lambda r:r.update(shutdown_memory_sampled=True),
            lambda r:r.update(board_samples=r['board_samples'][1:])):
            raw=resources();mutate(raw)
            with self.assertRaises(ValueError):review.resource_facts(raw)

    def test_cpu_role_rejected_before_plan_or_output_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'raw.json';path.write_text(json.dumps(dict(schema=m.RAW_SCHEMA,
                split='training',backend='CPU_TEST_ONLY',kind='CPU_TEST_ONLY')))
            with self.assertRaisesRegex(ValueError,'Actual native source'):
                review.review(path,Path(directory)/'review',manifest_path=self.manifest_ref['path'])
        with self.assertRaisesRegex(ValueError,'training roles'):quality.measure(dict(schema=m.RAW_SCHEMA),{}, {})

    def test_run_without_execute_claimed_unfinished_and_failed_receipts_never_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=copy.deepcopy(self.manifest);item=manifest['attempts'][0]
            item['collection']=str(root/'collection');item['claim']=str(root/'claim.json')
            def run(name):
                with patch.object(m,'audit',return_value=manifest),patch.object(m,'execute') as launch:
                    result=m.run(self.manifest_ref['path'],root/name,execute_enabled=True)
                    launch.assert_not_called();return result
            Path(item['claim']).write_text('{}');self.assertEqual(run('claimed')['status'],'HOLD_CLAIMED_ATTEMPT_NO_RETRY')
            Path(item['claim']).unlink();Path(item['collection']).mkdir()
            self.assertEqual(run('unfinished')['status'],'WAITING_EXISTING_ATTEMPT_FINAL_RECEIPT')
            (Path(item['collection'])/'execution.json').write_text('{}')
            self.assertEqual(run('failed')['status'],'STOPPED_NO_RETRY')
        with tempfile.TemporaryDirectory() as directory,patch.object(m,'execute') as launch:
            result=m.run(self.manifest_ref['path'],Path(directory)/'dry')
            launch.assert_not_called();self.assertEqual(result['status'],'READY_EXPLICIT_EXECUTE_REQUIRED')

    def test_intake_requires_all_roles_before_reading_any_raw_or_numeric_tuple(self):
        with tempfile.TemporaryDirectory() as directory:
            paths=[]
            for i in range(8):
                document=dict(schema=review.SCHEMA,status=intake.PASS,issues=[],resource_manifest=self.manifest_ref,
                    validation_labels_read=False,training_feature_extraction_eligible=True)
                if i==7:document['validation_labels_read']=True
                path=Path(directory)/str(i);path.write_text(json.dumps(document));paths.append(path)
            with self.assertRaisesRegex(ValueError,'before tuple labels'):intake.load_complete(self.manifest_ref['path'],paths)

    def test_intake_forged_pass_reaches_native_reviewer_not_caller_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            paths=[];root=Path(directory)
            for index,item in enumerate(self.manifest['attempts']):
                raw_ref=m.campaign.save(root/('raw'+str(index)),dict(schema=m.RAW_SCHEMA,split='training',backend='native',kind='measured'))
                document=dict(schema=review.SCHEMA,status=intake.PASS,issues=[],resource_manifest=self.manifest_ref,
                    validation_labels_read=False,training_feature_extraction_eligible=True,raw=raw_ref,
                    inputs={'plan.json':item['plan']},sources=[],execution={'path':'/forged'})
                path=root/('review'+str(index));path.write_text(json.dumps(document));paths.append(path)
            with self.assertRaises(KeyError),patch.object(intake.quality,'measure') as numeric:
                intake.load_complete(self.manifest_ref['path'],paths)
            numeric.assert_not_called()

    def quality_fixture(self,root):
        from gsdeploy.tasks import asr_normalize
        plan=copy.deepcopy(self.plan);plan_ref=m.campaign.save(root/'plan.json',plan)
        references=m.prepared_references(plan,plan['quality_catalog']);epoch=100
        rows=[]
        for a in plan['cases'][0]['arrivals']:
            task=a['task'];reference=references[task,a['pool_id']]
            prediction=((reference['answers'] or [''])[0] if task=='text' else reference['label'])
            if task=='asr':prediction=asr_normalize(prediction)
            rows.append(dict(request_id=a['request_id'],model=a['owner'],task=task,pool_id=a['pool_id'],
                sample_id=reference['id'],scheduled_time_ns=epoch+a['arrival_ns'],
                completion_time_ns=epoch+a['arrival_ns']+1,status='completed',
                worker_response=dict(request_id=a['request_id'],task=task,success=True,prediction=prediction)))
        raw=dict(schema=m.RAW_SCHEMA,split='training',backend='native',kind='measured',status='completed',
            **{k:plan[k] for k in ('attempt_id','capture_id','execution_id')},plan=plan_ref,
            request_records=rows,cases=[dict(case_id='original181')],measurement_windows=[dict(
                case_id='original181',start_ns=epoch,end_ns=epoch+m.MEASUREMENT_NS,
                fixed_drain_deadline_ns=epoch+m.MEASUREMENT_NS+m.DRAIN_NS)])
        return plan,raw

    def test_quality_original_totals_rate_weighting_and_exact_output_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan,raw=self.quality_fixture(root)
            ocr=[r for r in raw['request_records'] if r['task']=='ocr'];ocr[0]['worker_response']['prediction']=''
            raw_ref=m.campaign.save(root/'raw.json',raw);out=quality.measure(raw,plan,raw_ref)
            refs=m.prepared_references(plan,plan['quality_catalog'])
            numerator=len(quality.page_ocr_text(refs['ocr',ocr[0]['pool_id']]['label']))
            denominator=sum(len(quality.page_ocr_text(refs['ocr',r['pool_id']]['label'])) for r in ocr)
            self.assertEqual(out['per_task']['ocr']['numerator'],numerator)
            self.assertEqual(out['per_task']['ocr']['denominator'],denominator)
            self.assertAlmostEqual(out['measured_Q'],.05/2*numerator/denominator)
            self.assertEqual(sum(v['offered'] for v in out['per_task'].values()),181)
            self.assertEqual(out['requests'][0]['native_output_source'],dict(raw=raw_ref,
                json_pointer='/request_records/0/worker_response'))
            self.assertFalse(out['native_authentication_by_pure_helper']);self.assertFalse(out['quality_model_fitted'])

    def test_quality_missing_late_failed_native_output_preserves_unknown_not_frozenQ(self):
        for failure in ('missing','late','failed'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);plan,raw=self.quality_fixture(root);row=raw['request_records'][0]
                if failure=='missing':row['status']='incomplete'
                elif failure=='late':row['completion_time_ns']=raw['measurement_windows'][0]['fixed_drain_deadline_ns']+1
                else:row['worker_response']['success']=False
                ref=m.campaign.save(root/'raw.json',raw);out=quality.measure(raw,plan,ref)
                self.assertIsNone(out['measured_Q']);self.assertEqual(out['offered_requests'],181)
                self.assertIn(row['request_id'],out['per_task'][row['task']]['missing'])
                self.assertFalse(out['training_expectation_fallback_used'])

    def test_quality_rebound_bytes_cannot_change_route_input_or_response_identity(self):
        for field in ('model','sample_id','response'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);plan,raw=self.quality_fixture(root);row=raw['request_records'][0]
                if field=='response':row['worker_response']['request_id']='foreign'
                else:row[field]='foreign'
                ref=m.campaign.save(root/'raw.json',raw)
                with self.assertRaises(ValueError):quality.measure(raw,plan,ref)

    def test_completed_execution_requires_exact_command_and_live_ownership_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            item=copy.deepcopy(self.manifest['attempts'][0]);item['claim']=str(Path(directory)/'claim.json')
            claim=m.campaign.save(item['claim'],dict(schema=m.SCHEMA+'.claim',manifest=self.manifest_ref,
                plan=item['plan'],execution_id=item['execution_id'],attempt_id=item['attempt_id'],no_retries=True))
            command=[sys.executable,'-m','tools.predictor_validation_20261027.twoFIFO_public181_resource_execute_v1',
                '_child','--manifest',self.manifest_ref['path'],'--plan',item['plan']['path'],
                '--output',item['collection'],'--execute']
            execution=dict(command=command,profile_execution=dict(schema=m.SCHEMA+'.execution_binding',
                manifest=self.manifest_ref,plan=item['plan'],execution_id=item['execution_id'],
                attempt_id=item['attempt_id'],raw_schema=m.RAW_SCHEMA,claim=claim,
                downstream_adapter_required=True,automatic_fit=False),budget=m.supervisor.BUDGET,
                status='CHILD_EXITED_REQUIRES_PROFILE_REVIEW',collection_subprocess_launched=True,
                process_group=123,child_reaped=True,live_group_pids=[],experiment_lock_held_through_group_exit=True,
                all_owned_descendants_exited_and_reaped=True,final_source_audit='PASS',watchdog=dict(status='EXITED',return_code=0))
            m.completed_execution(execution,self.manifest_ref,item)
            changes=[lambda e:e['command'].__setitem__(2,'foreign'),lambda e:e.update(live_group_pids=[999]),
                lambda e:e['watchdog'].update(return_code=1),lambda e:e.update(final_source_audit='FAIL'),
                lambda e:e['profile_execution'].update(raw_schema='old')]
            for mutate in changes:
                value=copy.deepcopy(execution);mutate(value)
                with self.assertRaises(ValueError):m.completed_execution(value,self.manifest_ref,item)

    def test_intake_complete_interface_preserves_quality_separately_from_closed_resource_tuple(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);paths=[];reports={}
            joint=review.resource_facts(resources())
            for index,item in enumerate(self.manifest['attempts']):
                raw=m.campaign.save(root/('raw'+str(index)),dict(schema=m.RAW_SCHEMA,split='training',backend='native',kind='measured'))
                doc=dict(schema=review.SCHEMA,status=intake.PASS,issues=[],resource_manifest=self.manifest_ref,
                    validation_labels_read=False,training_feature_extraction_eligible=True,raw=raw,
                    inputs={'plan.json':item['plan']},sources=[],execution={'path':'/CPU-MOCK-ONLY'},observed_resources=joint)
                path=root/('review'+str(index));path.write_text(json.dumps(doc));paths.append(path);reports[raw['path']]=receipt(path)
            # CPU interface check only: real reviewer rejects these incomplete
            # native-shaped raws (previous test), and is deliberately mocked.
            with patch.object(intake.reviewer,'review',side_effect=lambda raw,*a,**k:reports[str(raw)]),\
                    patch.object(intake.reviewer,'resource_facts',return_value=joint),\
                    patch.object(intake.quality,'measure',return_value={'CPU_TEST_ONLY':True}):
                out=intake.load_complete(self.manifest_ref['path'],paths)
            self.assertEqual(len(out['rows']),8);self.assertEqual(len(out['paired_resource_catalog']['draws']),2)
            self.assertEqual(out['rows'][0]['actual_native_quality'],{'CPU_TEST_ONLY':True})
            self.assertFalse(out['queue_resource_probability_coupling_inferred'])

    def test_rehashed_plan_cannot_change_sources_runtime_or_bound_quality_catalog(self):
        path=Path(self.manifest['attempts'][0]['plan']['path']);original=path.read_bytes()
        mutations=[lambda d:d['runtime_config'].update(max_pending_requests=2),
            lambda d:d['cases'][0]['arrivals'][0].update(arrival_ns=12),
            lambda d:d.update(quality_catalog={'path':'/foreign','sha256':'0'*64}),
            lambda d:d.update(source_freeze=[])]
        try:
            for mutate in mutations:
                value=copy.deepcopy(self.plan);mutate(value);value.pop('execution_context_sha256')
                value['execution_context_sha256']=m.digest(value);path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):m.audit(self.manifest_ref['path'])
        finally:path.write_bytes(original)

    def test_change_restore_unselected_plan_is_detected_before_claim_or_native_launch(self):
        selected=self.manifest['attempts'][0];other=Path(self.manifest['attempts'][1]['plan']['path'])
        original=other.read_bytes();original_audit=m.audit
        def tamper(path):
            other.write_bytes(original+b' ');other.write_bytes(original);return original_audit(path)
        with patch.object(m,'audit',side_effect=tamper),patch.object(m.supervisor,'public_environment_gate'),\
                patch.object(m.supervisor,'supervise') as launch:
            with self.assertRaises(ValueError):m.execute(selected['plan']['path'],selected['collection'],manifest_path=self.manifest_ref['path'])
            launch.assert_not_called()
        self.assertFalse(Path(selected['claim']).exists());self.assertEqual(receipt(other),self.manifest['attempts'][1]['plan'])

    def test_incomplete_actual_collector_enters_explicit_partial_review_not_survivor_filter(self):
        from tools.predictor_validation_20261027 import twoFIFO_public181_partial_protocol_v1 as partial
        raw,_,_,_=self.collect(incomplete=True)
        cohort=review.inspect_cohort(self.plan,raw)
        self.assertEqual(cohort['issues'],[]);self.assertEqual(len(cohort['outcome_diagnostics']),181)
        self.assertEqual(cohort['offered_requests'],181)
        # Actual CPU collector prefix checker runs on all199 rows. Full-output
        # parsers are mocked only for the18 CPU priming outputs, which do not
        # pretend to contain original model responses or native device probes.
        with patch.object(partial,'reparse_outputs',return_value=dict(issues=[],sources=[],policies={})),\
                patch.object(partial,'validate_request',return_value=([],{'CPU_PRIMING_ONLY':True})) as complete:
            protocol=partial.inspect_protocol(raw)
        self.assertEqual(protocol['all_offered_and_priming_prefixes'],199)
        self.assertEqual(protocol['partial_observations'],181);self.assertEqual(complete.call_count,18)
        self.assertTrue(all(r['prefix']['missing_RESULT'] for r in protocol['requests'][18:]))
        self.assertFalse(protocol['absent_outcomes_are_successes'])
        self.assertEqual(raw['backend'],'CPU_TEST_ONLY')

    def test_partial_prefix_rejects_forged_start_phase_and_ready_without_result(self):
        from tools.predictor_validation_20261027 import twoFIFO_public181_partial_protocol_v1 as partial
        raw,_,_,_=self.collect(incomplete=True);row=raw['request_records'][0]
        end=raw['lifecycle_boundaries'][-1]['time_ns'];ready=raw['ready'][row['model']]
        mutations=[lambda r:r.update(observed_worker_start_time_ns=end+1),
            lambda r:r.update(previous_handler_next_request_ready={'perf_ns':0}),
            lambda r:r.update(response_reader_event_enqueue_time_ns=end),
            lambda r:r.update(execution_events=[dict(request_id=r['request_id'],event='end',kind='gpu',
                segment_id='fake',perf_ns=0,received_time_ns=0,collector_dequeued_time_ns=0,collector_processing_time_ns=0)])]
        for mutate in mutations:
            value=copy.deepcopy(row);mutate(value)
            with self.assertRaises(ValueError):partial.inspect_prefix(value,ready,shutdown_end_ns=end)

    def test_intake_retains_all_eight_unknown_when_no_native_observation_exists(self):
        out=intake.load_complete(self.manifest_ref['path'],[])
        self.assertEqual(len(out['rows']),8);self.assertEqual(out['unknown_resource_attempts'],8)
        self.assertIsNone(out['paired_resource_catalog'])
        self.assertTrue(all(r['joint_tuple'] is None and r['actual_native_quality'] is None for r in out['rows']))
        self.assertFalse(out['unknown_attempts_removed_or_survivor_mass_renormalized'])


if __name__=='__main__':unittest.main()
