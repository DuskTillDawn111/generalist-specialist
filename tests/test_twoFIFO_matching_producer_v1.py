"""Matching producer CPU contracts; fake transport is never native evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_matching_execute_v1 as m
from tools.predictor_validation_20261027 import twoFIFO_matching_review_v1 as review
from tools.predictor_validation_20261027 import twoFIFO_matching_macros_v1 as macros
from tools.predictor_validation_20261027 import twoFIFO_matching_intake_v1 as intake
from tools.predictor_validation_20261027 import twoFIFO_source_rebound_execute_v1 as old
from tests.test_twoFIFO_owned_collect_v2 import Clock, FakeRuntime, FakeMonitor
from tests.test_twoFIFO_owned_training_review_v3 import fixture as base_fixture


ROOT=Path(__file__).resolve().parents[1]
SUPPLEMENT=ROOT/'artifacts/predictor-twoFIFO-matching-macro-supplement-v1-20261029-cpu02-prepared/manifest.json'


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def fixture():
    plan,raw,ledger=base_fixture()
    plan['schema']=m.SCHEMA+'.attempt';raw['schema']=m.RAW_SCHEMA
    for planned,observed in zip(plan['priming'],raw['priming_request_records']):
        planned['request_id']=observed['request_id']
    raw['priming_public_calendar']=copy.deepcopy(plan['priming'])
    return plan,raw,ledger


def resource_raw():
    start=3_000_000_000;end=start+10_000_000_000;deadline=end+60_000_000_000
    return dict(lifecycle_boundaries=[dict(boundary='startup_begin',time_ns=1_000_000_000),
        dict(boundary='priming_end',time_ns=2_000_000_000),
        dict(boundary='monitoring_end',time_ns=deadline+2),
        dict(boundary='shutdown_begin',time_ns=deadline+3)],
        board_samples=[dict(time_ns=i*1_000_000_000,watts=100,owned_used_mib=10)
                       for i in range(1,74)],shutdown_memory_sampled=False,
        measurement_windows=[dict(case_id='case-1',start_ns=start,end_ns=end,fixed_drain_deadline_ns=deadline)],
        cases=[dict(case_id='case-1')])


def completed_doc(item,manifest_ref):
    claim=dict(schema=m.SCHEMA+'.claim',manifest=manifest_ref,plan=item['plan'],
        execution_id=item['execution_id'],attempt_id=item['attempt_id'],no_retries=True)
    write(item['claim'],claim)
    return dict(profile_execution=dict(schema=m.SCHEMA+'.execution_binding',manifest=manifest_ref,
        plan=item['plan'],execution_id=item['execution_id'],attempt_id=item['attempt_id'],
        raw_schema=m.RAW_SCHEMA,claim=m.receipt(item['claim']),downstream_adapter_required=True,automatic_fit=False),
        budget=m.supervisor.BUDGET,status='CHILD_EXITED_REQUIRES_PROFILE_REVIEW',
        collection_subprocess_launched=True,process_group=42,child_reaped=True,live_group_pids=[],
        experiment_lock_held_through_group_exit=True,all_owned_descendants_exited_and_reaped=True,
        final_source_audit='PASS',watchdog=dict(status='EXITED',return_code=0))


def intake_fixture(root):
    """Real receipts/arithmetic, mocked upstream native/protocol authentication."""
    prepared=json.loads(SUPPLEMENT.read_text());items=[];plans=[]
    for block in prepared['blocks']:
        for seedref in block['attempts']:
            plan=m.checked(seedref);aid=plan['attempt_id']
            plan.update(schema=m.SCHEMA+'.attempt',context=prepared['artifacts']['candidate_context'])
            plan['priming']=[dict(row,request_id=aid+':priming:'+str(i)) for i,row in enumerate(plan['priming'])]
            pp=root/'producer'/aid/'plan.json';write(pp,plan);plans.append(plan)
            items.append(dict(plan=m.receipt(pp),attempt_id=aid,execution_id=plan['execution_id'],
                claim=str(pp.parent/'matching-attempt-claim.json'),collection=str(pp.parent/'collection')))
    manifest=dict(attempts=items,sources=[]);mp=root/'producer'/'manifest.json';write(mp,manifest);mf=m.receipt(mp)
    completed=[];reviews={};macrodocs={};paths={}
    for item,plan in zip(items,plans):
        aid=plan['attempt_id'];windows=[]
        for i,case in enumerate(plan['cases']):
            start=(3+71*i)*1_000_000_000
            windows.append(dict(case_id=case['case_id'],start_ns=start,end_ns=start+10_000_000_000,
                fixed_drain_deadline_ns=start+70_000_000_000))
        last=windows[-1]['fixed_drain_deadline_ns'];sample_end=last//1_000_000_000
        raw=resource_raw();raw.update(schema=m.RAW_SCHEMA,backend='native',status='completed',
            attempt_id=aid,execution_id=plan['execution_id'],plan=item['plan'],
            measurement_windows=windows,cases=[dict(case_id=c['case_id']) for c in plan['cases']],
            request_records=[dict(request_id=r['request_id']) for c in plan['cases'] for r in c['arrivals']],
            board_samples=[dict(time_ns=i*1_000_000_000,watts=100,owned_used_mib=10)
                           for i in range(1,sample_end+1)])
        raw['lifecycle_boundaries'][-2]['time_ns']=last+2
        raw['lifecycle_boundaries'][-1]['time_ns']=last+3
        rp=Path(item['collection'])/'raw.json';write(rp,raw)
        execution=completed_doc(item,mf);ep=Path(item['collection'])/'execution.json';write(ep,execution)
        resources=review.resource_facts(raw)
        semantic=m.planned_priming_semantics(plan,m.checked(plan['context']))
        rd=dict(schema=review.SCHEMA,status='PASS_COMPLETE_MATCHING_NATIVE_TRAINING_NOT_PREDICTOR_QUALIFICATION',
            training_feature_extraction_eligible=True,issues=[],validation_labels_read=False,
            matching_manifest=mf,raw=m.receipt(rp),execution=m.receipt(ep),sources=[m.receipt(item['claim'])],
            observed_resources=resources,priming_semantics=dict(planned=semantic,
                actual_native_identity=semantic['sha256'],actual_native_history_qualified=True,
                provenance_request_ids=[r['request_id'] for r in plan['priming']]))
        rdpath=root/'run'/aid/'native-review'/'report.json';write(rdpath,rd)
        reviews[str(rp)]=m.receipt(rdpath)
        macro=dict(schema='gsdeploy.twoFIFO_matching_composable_response_macro_training.v1',
            status='COMPLETE_ATTEMPT_DISJOINT_TRAINING_MACROS_NOT_FITTED_PROVIDER',mode='natural',
            validation_labels_read=False,matching_manifest=mf,whole_attempt_history_used_for_visible_prefix=True,
            case_history_reset=False,capture_id=plan['capture_id'],attempt_id=aid,execution_id=plan['execution_id'],
            requests=raw['request_records'],measured_requests=len(raw['request_records']),
            priming_rows_excluded_as_labels=18,fitted_model_qualified=False,sources=[])
        macro_path=root/'run'/aid/'macros-v1.json';write(macro_path,macro);macrodocs[str(rdpath)]=macro
        resource_path=root/'run'/aid/'observed-resources.json';write(resource_path,resources)
        entry=dict(attempt_id=aid,execution=m.receipt(ep),training_review=m.receipt(rdpath),
            macros=m.receipt(macro_path),observed_resources=m.receipt(resource_path));completed.append(entry)
        paths[aid]=dict(raw=rp,review=rdpath,macro=macro_path,resource=resource_path)
    final=dict(schema=m.CONT_SCHEMA+'.run',status=intake.FINAL_STATUS,no_retries=True,
        fit_eligible=False,automatic_fit=False,numerical_fit_run=False,validation_labels_read=False,
        manifest=mf,completed_attempts=completed)
    fp=root/'run'/'report.json';write(fp,final)
    return mp,fp,manifest,reviews,macrodocs,paths


class MatchingProducerContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temp.name)
        cls.manifest_ref=m.prepare(SUPPLEMENT,cls.root/'producer')
        cls.path=Path(cls.manifest_ref['path'])
        cls.manifest=m.audit(cls.path)
        cls.plans=[m.checked(a['plan']) for a in cls.manifest['attempts']]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_real_prepare_audit_15_attempts_3_blocks_and_all660_unique_ids(self):
        self.assertEqual(len(self.plans),15)
        blocks=intake._attempt_layout(self.plans)
        self.assertEqual(len(blocks),3)
        measured=[r['request_id'] for p in self.plans for c in p['cases'] for r in c['arrivals']]
        prime=[r['request_id'] for p in self.plans for r in p['priming']]
        self.assertEqual((len(measured),len(prime),len(set(measured+prime))),(390,270,660))
        self.assertEqual(sum(len(p['cases']) for p in self.plans),171)
        for p in self.plans:
            self.assertEqual(m.audit_plan(next(a['plan']['path'] for a in self.manifest['attempts']
                if a['attempt_id']==p['attempt_id'])),p)
            self.assertEqual(p['source_freeze'],self.plans[0]['source_freeze'])
            self.assertNotEqual(p['schema'],old.original.SCHEMA+'.attempt')
            self.assertEqual(p['execution_context_sha256'],m.digest({k:v for k,v in p.items()
                if k!='execution_context_sha256'}))

    def test_no_global_mutation_and_shared_lowlevel_identity(self):
        self.assertIs(m.dispatch_case,old.dispatch_case)
        self.assertIs(m.finish_collection,old.finish_collection)
        self.assertIs(m.ActivityLedger,old.ActivityLedger)
        self.assertIs(m.SourceChangeGuard,old.SourceChangeGuard)
        self.assertIsNot(m.audit_plan,old.audit_plan)
        self.assertNotEqual(m.RAW_SCHEMA,old.RAW_SCHEMA)
        for p in self.plans:
            self.assertIsNone(p['runtime_policy']['admission'])
            self.assertFalse(p['runtime_policy']['batching'])
            self.assertFalse(p['runtime_policy']['weight_reuse'])

    def test_cross_attempt_priming_semantics_equal_while_provenance_ids_disjoint(self):
        context=m.checked(self.plans[0]['context'])
        semantic=[m.planned_priming_semantics(p,context) for p in self.plans]
        self.assertEqual(len({r['sha256'] for r in semantic}),1)
        self.assertEqual(len({p['priming'][0]['request_id'] for p in self.plans}),15)
        self.assertTrue(all(r['request_ids_are_provenance_not_physical_axes'] for r in semantic))
        self.assertTrue(all(not r['actual_native_history_qualified'] for r in semantic))
        from tools.predictor_validation_20261027.natural_macro_resources_candidate_v1 import priming_identity
        synthetic=dict(priming_request_records=[dict(model=r['owner'],task=r['task'],pool_id=r['pool_id'],
            request_id=r['request_id'],status='completed') for r in self.plans[0]['priming']])
        self.assertEqual(priming_identity(synthetic,context),semantic[0]['sha256'])
        changed=copy.deepcopy(self.plans[0]);changed['priming'].reverse()
        self.assertNotEqual(m.planned_priming_semantics(changed,context)['sha256'],semantic[0]['sha256'])
        changed=copy.deepcopy(self.plans[0]);changed['priming'][0]['exact_input_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'exact context declaration'):
            m.planned_priming_semantics(changed,context)
        changed_context=copy.deepcopy(context)
        chosen=self.plans[0]['priming'][0]['pool_id']
        for row in changed_context['declared_inputs']:
            if row['pool_id']==chosen:row['exact_input_sha256']='0'*64
        changed=copy.deepcopy(self.plans[0])
        for row in changed['priming']:
            if row['pool_id']==chosen:row['exact_input_sha256']='0'*64
        self.assertNotEqual(m.planned_priming_semantics(changed,changed_context)['sha256'],semantic[0]['sha256'])

    def test_plan_calendar_priming_budget_schema_and_context_tamper_rejected(self):
        item=self.manifest['attempts'][0];path=Path(item['plan']['path']);original=path.read_bytes()
        for kind in ('primeid','primedrop','arrival','schema','budget','context'):
            changed=copy.deepcopy(self.plans[0])
            if kind=='primeid':changed['priming'][0]['request_id']='foreign'
            elif kind=='primedrop':changed['priming'].pop()
            elif kind=='arrival':changed['cases'][0]['arrivals'][0]['arrival_ns']+=1
            elif kind=='schema':changed['schema']=old.original.SCHEMA+'.attempt'
            elif kind=='budget':changed['supervisor_budget']['total_timeout_seconds']=1
            else:changed['context']['sha256']='0'*64
            changed['execution_context_sha256']=m.digest({k:v for k,v in changed.items()
                if k!='execution_context_sha256'})
            try:
                write(path,changed)
                with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,'Exact matching plan'):
                    m.audit(self.path)
            finally:path.write_bytes(original)

    def test_manifest_order_source_and_fit_flag_tamper_rejected(self):
        original=self.path.read_bytes()
        for kind in ('order','source','fit','blockcount'):
            changed=copy.deepcopy(self.manifest)
            if kind=='order':changed['attempts'].reverse()
            elif kind=='source':changed['sources'].pop()
            elif kind=='fit':changed['automatic_fit']=True
            else:changed['training_blocks']=15
            try:
                write(self.path,changed)
                with self.subTest(kind=kind),self.assertRaises(ValueError):m.audit(self.path)
            finally:self.path.write_bytes(original)

    def fake_collect(self,fail_start=False,fail_case=False):
        item=self.manifest['attempts'][0];clock=Clock();runtime=FakeRuntime(clock,fail_case)
        if fail_start:
            def fail():raise RuntimeError('injected startup failure')
            runtime.start=fail
        ownership=lambda values,*a,**k:dict(owned_aliases=[m.OWNERS[next(iter(values))-1]],
            unresolved_nvml_pids=[],foreign_pids=[],ownership_evidence=[{'live':True}])
        with tempfile.TemporaryDirectory() as temp,patch.object(m.supervisor,'validate_lease'):
            raw=m.collect(item['plan']['path'],temp,lease_fd=-1,runtime_factory=lambda *a:runtime,
                monitor_factory=FakeMonitor,clock_ns=clock.clock,wait_until_ns=clock.wait,
                ownership_check=ownership)
            saved=json.loads((Path(temp)/'raw.json').read_text())
            ledger=json.loads((Path(temp)/m.ACTIVITY_FILE).read_text())
        self.assertEqual(raw,saved)
        return raw,ledger,runtime

    def test_fake_full_attempt_retains_priming_history_and_fixed_allcase_windows(self):
        raw,ledger,runtime=self.fake_collect()
        plan=self.plans[0]
        self.assertEqual(raw['status'],'completed')
        self.assertEqual((raw['kind'],raw['backend']),('CPU_TEST_ONLY','CPU_TEST_ONLY'))
        self.assertEqual(len(raw['priming_request_records']),18)
        self.assertEqual([r['request_id'] for r in raw['priming_request_records']],
            [r['request_id'] for r in plan['priming']])
        self.assertEqual(len(raw['request_records']),sum(c['offered'] for c in plan['cases']))
        self.assertEqual(len(raw['case_histories'][0]['prior_history_snapshot']['request_records']),18)
        self.assertEqual(len(raw['measurement_windows']),13)
        for window in raw['measurement_windows']:
            self.assertEqual(window['end_ns']-window['start_ns'],10_000_000_000)
            self.assertEqual(window['fixed_drain_deadline_ns']-window['end_ns'],60_000_000_000)
        self.assertEqual(review.inspect_checkpoint_activity(plan,raw,ledger)['critical_interval_overlaps'],0)
        self.assertTrue(runtime.closed)
        self.assertFalse(raw['training_feature_extraction_eligible'])

    def test_startup_failure_retains_all18_unstarted_priming_and_all_measured(self):
        raw,_,runtime=self.fake_collect(fail_start=True)
        self.assertEqual(raw['status'],'failed');self.assertTrue(runtime.closed)
        self.assertEqual(len(raw['priming_request_records']),18)
        self.assertTrue(all(r['status']=='not_started' and r['scheduled_time_ns'] is None
                            for r in raw['priming_request_records']))
        self.assertTrue(all(r['status']=='not_started' for r in raw['request_records']))

    def test_failed_case_keeps_future_offered_rows_and_cleanup(self):
        raw,_,runtime=self.fake_collect(fail_case=True)
        self.assertEqual(raw['status'],'failed');self.assertTrue(runtime.closed)
        self.assertEqual(len(raw['cases']),13)
        self.assertTrue(all(c['status']=='NOT_STARTED' for c in raw['cases'][1:]))
        self.assertEqual(len(raw['request_records']),sum(c['offered'] for c in self.plans[0]['cases']))

    def test_default_run_does_not_execute_and_claim_or_unfinished_never_retry(self):
        item=self.manifest['attempts'][0]
        for kind,status in [('new','READY_EXPLICIT_EXECUTE_REQUIRED'),
                            ('claim','HOLD_CLAIMED_ATTEMPT_NO_RETRY'),
                            ('unfinished','WAITING_EXISTING_ATTEMPT_FINAL_RECEIPT')]:
            target=Path(item['claim'] if kind=='claim' else item['collection'])
            if kind=='claim':write(target,{})
            elif kind=='unfinished':target.mkdir()
            try:
                with tempfile.TemporaryDirectory() as temp,patch.object(m,'execute') as execute:
                    result=m.run(self.path,Path(temp)/'run',execute_enabled=kind!='new')
                    self.assertEqual(result['status'],status);execute.assert_not_called()
            finally:
                if kind=='claim':target.unlink()
                elif kind=='unfinished':target.rmdir()


class MatchingReviewContracts(unittest.TestCase):
    def test_execute_guard_covers_unselected_plan_during_first_audit_and_supervisor(self):
        for stage in ('audit','supervised'):
            with self.subTest(stage=stage),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);items=[];plans=[]
                for i in range(2):
                    pp=root/('attempt-'+str(i))/'plan.json'
                    plan=dict(attempt_id='attempt-'+str(i),execution_id=str(i)*32)
                    write(pp,plan);plans.append(plan)
                    items.append(dict(plan=m.receipt(pp),attempt_id=plan['attempt_id'],
                        execution_id=plan['execution_id'],claim=str(pp.parent/'claim.json'),
                        collection=str(pp.parent/'collection')))
                manifest=dict(attempts=items,sources=[]);mp=root/'manifest.json';write(mp,manifest)
                other=Path(items[1]['plan']['path'])
                def change_restore():
                    original=other.read_bytes();other.write_text('changed');other.write_bytes(original)
                audits=[]
                def audit(*args):
                    audits.append(True)
                    if stage=='audit' and len(audits)==1:change_restore()
                    return manifest
                def supervised(*args,**kwargs):
                    kwargs['before_launch']();change_restore();kwargs['after_run']()
                    return dict(status='CPU_STUB_NO_PROCESS')
                with patch.object(m,'audit',side_effect=audit), \
                     patch.object(m,'audit_plan',return_value=plans[0]), \
                     patch.object(m.supervisor,'public_environment_gate'), \
                     patch.object(m.supervisor,'supervise',side_effect=supervised) as supervisor, \
                     self.assertRaises(ValueError):
                    m.execute(items[0]['plan']['path'],items[0]['collection'],manifest_path=mp)
                if stage=='audit':supervisor.assert_not_called()

    def test_completed_execution_authenticates_durable_claim_and_owned_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);pp=root/'plan.json';write(pp,{})
            item=dict(plan=m.receipt(pp),execution_id='a'*32,attempt_id='matching-attempt',
                claim=str(root/'claim.json'))
            mf=dict(path='manifest.json',sha256='b'*64)
            execution=completed_doc(item,mf);m.completed_execution(execution,mf,item)
            for kind in ('old','claim','cleanup','guard','budget','watchdog','launch'):
                bad=copy.deepcopy(execution)
                if kind=='old':bad['profile_execution']['schema']=old.SCHEMA+'.execution_binding'
                elif kind=='claim':bad['profile_execution']['claim']['sha256']='0'*64
                elif kind=='cleanup':bad['all_owned_descendants_exited_and_reaped']=False
                elif kind=='guard':bad['final_source_audit']='FAIL'
                elif kind=='budget':bad['budget']=dict(m.supervisor.BUDGET,total_timeout_seconds=1)
                elif kind=='watchdog':bad['watchdog']['return_code']=1
                else:bad['collection_subprocess_launched']=False
                with self.subTest(kind=kind),self.assertRaises(ValueError):m.completed_execution(bad,mf,item)
            claim=json.loads(Path(item['claim']).read_text());claim['no_retries']=False;write(item['claim'],claim)
            with self.assertRaisesRegex(ValueError,'durable no-retry'):m.completed_execution(execution,mf,item)

    def test_new_cohort_keeps_complete18_priming_and_offered_denominator(self):
        plan,raw,_=fixture()
        result=review.inspect_cohort(plan,raw)
        self.assertEqual(result['issues'],[]);self.assertEqual(result['priming_requests'],18)
        self.assertTrue(result['complete_offered_denominator_preserved'])
        self.assertFalse(result['global_admission_cap_added'])

    def test_omitted_or_reidentified_priming_and_measured_requests_rejected(self):
        for kind in ('primeomit','primeid','collision','measuredomit','oldschema'):
            plan,raw,_=fixture()
            if kind=='primeomit':raw['priming_request_records'].pop()
            elif kind=='primeid':raw['priming_request_records'][0]['request_id']='other'
            elif kind=='collision':
                plan['priming'][0]['request_id']=raw['request_records'][0]['request_id']
                raw['priming_request_records'][0]['request_id']=raw['request_records'][0]['request_id']
            elif kind=='measuredomit':raw['request_records'].pop()
            else:raw['schema']=old.RAW_SCHEMA
            with self.subTest(kind=kind),self.assertRaises(ValueError):review.inspect_cohort(plan,raw)

    def test_clock_and_checkpoint_protected_interval_checks_still_apply(self):
        plan,raw,ledger=fixture()
        raw['request_records'][0]['completion_time_ns']=raw['request_records'][0]['scheduled_time_ns']-1
        with self.assertRaises(ValueError):review.inspect_cohort(plan,raw)
        plan,raw,ledger=fixture()
        ledger['checkpoint_spans'][1].update(start_ns=raw['measurement_windows'][0]['start_ns'],
            end_ns=raw['measurement_windows'][0]['start_ns'])
        with self.assertRaisesRegex(ValueError,'intersect'):
            review.inspect_checkpoint_activity(plan,raw,ledger)

    def test_old_raw_is_rejected_before_authentication_or_label_inspection(self):
        with tempfile.TemporaryDirectory() as temp:
            raw=Path(temp)/'raw.json';write(raw,dict(schema=old.RAW_SCHEMA,split='training'))
            with patch.object(m,'audit') as audit, self.assertRaisesRegex(ValueError,'Training-only raw'):
                review.review(raw,Path(temp)/'review',manifest_path='absent.json')
            audit.assert_not_called()

    def test_resource_windows_separate_measurement_drain_and_startup(self):
        raw=resource_raw();facts=review.resource_facts(raw)
        self.assertEqual(facts['cases'][0]['measurement_energy_joules'],1000)
        self.assertEqual(facts['cases'][0]['drain_energy_joules_diagnostic_only'],6000)
        self.assertEqual(facts['startup']['energy_joules'],200)
        self.assertEqual(facts['startup']['occurrences'],1)
        self.assertFalse(facts['cases'][0]['memory_safety_factor_applied'])
        self.assertFalse(facts['resource_forecast_qualified'])

    def test_missing_power_fixed_window_or_shutdown_memory_is_rejected(self):
        for kind in ('gap','window','shutdown','nonfinite'):
            raw=resource_raw()
            if kind=='gap':raw['board_samples']=[s for s in raw['board_samples']
                                               if not 5_000_000_000<=s['time_ns']<=9_000_000_000]
            elif kind=='window':raw['measurement_windows'][0]['end_ns']+=1
            elif kind=='shutdown':raw['shutdown_memory_sampled']=True
            else:raw['board_samples'][2]['owned_used_mib']=float('inf')
            with self.subTest(kind=kind),self.assertRaises(ValueError):review.resource_facts(raw)

    def test_source_guard_change_restore_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'source.py';source.write_text('original')
            with m.SourceChangeGuard([m.receipt(source)]) as guard:
                source.write_text('changed');source.write_text('original')
                with self.assertRaises(ValueError):guard.check()


class MatchingIntakeContracts(unittest.TestCase):
    def call_intake(self,mp,fp,manifest,reviews,macrodocs):
        # Explicit trust-boundary mocks: no CPU fixture qualifies native GPU data.
        # File receipts, execution claims, plan/cohort counts, identity projections,
        # resource arithmetic and full intake iteration remain real.
        with patch.object(m,'audit',return_value=manifest), \
             patch.object(review,'review',side_effect=lambda raw,*a,**k:reviews[str(raw)]), \
             patch.object(macros,'extract',side_effect=lambda rr:macrodocs[str(rr)]):
            return intake.intake(mp,fp)

    def test_complete_intake_390_plus270_and_15_startups_171_resource_windows(self):
        with tempfile.TemporaryDirectory() as temp:
            args=intake_fixture(Path(temp));mp,fp,manifest,reviews,macrodocs,_=args
            result=self.call_intake(mp,fp,manifest,reviews,macrodocs)
            self.assertEqual(result['status'],'COMPLETE_MATCHING_EXACT15_TRAINING_INTAKE_NOT_FIT')
            self.assertEqual((result['measured_requests'],result['owned_history_requests']),(390,270))
            self.assertEqual((result['startup_occurrences'],result['observed_resource_windows']),(15,171))
            self.assertEqual(result['training_replication_units'],3)
            self.assertTrue(result['priming_request_identity_is_provenance'])
            self.assertFalse(result['kernel_resource_fit']);self.assertFalse(result['main_queue_provider_eligible'])

    def test_rehashed_resource_or_native_semantic_forgery_rejected(self):
        for kind in ('resource','semantic','missingattempt'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temp:
                mp,fp,manifest,reviews,macrodocs,paths=intake_fixture(Path(temp))
                final=json.loads(fp.read_text());entry=final['completed_attempts'][0];aid=entry['attempt_id']
                if kind=='resource':
                    path=paths[aid]['resource'];doc=json.loads(path.read_text())
                    doc['cases'][0]['measurement_energy_joules']+=1;write(path,doc)
                    entry['observed_resources']=m.receipt(path)
                elif kind=='semantic':
                    path=paths[aid]['review'];doc=json.loads(path.read_text())
                    doc['priming_semantics']['actual_native_identity']='0'*64;write(path,doc)
                    entry['training_review']=m.receipt(path);reviews[str(paths[aid]['raw'])]=m.receipt(path)
                else:final['completed_attempts'].pop()
                write(fp,final)
                with self.assertRaises(ValueError):self.call_intake(mp,fp,manifest,reviews,macrodocs)

    def test_15_layout_rejects_missing_duplicate_or_four_shard_cohorts(self):
        plans=[dict(schema=m.SCHEMA+'.attempt',split='training',no_retries=True,
            capture_id='block-'+str(i//5),attempt_id='attempt-'+str(i),shard_index=i%5+1,
            shards_in_block=5) for i in range(15)]
        self.assertEqual(len(intake._attempt_layout(plans)),3)
        for changed in (plans[:-1],plans+[plans[0]],plans[:-1]+[plans[0]],
                        [dict(p,shards_in_block=4) for p in plans]):
            with self.assertRaises(ValueError):intake._attempt_layout(changed)

    def test_macro_complete_cohort_schema_and_no_fit_gates(self):
        plan,raw,_=fixture();ref=dict(path='manifest',sha256='a'*64)
        doc=dict(schema='gsdeploy.twoFIFO_matching_composable_response_macro_training.v1',
            status='COMPLETE_ATTEMPT_DISJOINT_TRAINING_MACROS_NOT_FITTED_PROVIDER',mode='natural',
            validation_labels_read=False,matching_manifest=ref,whole_attempt_history_used_for_visible_prefix=True,
            case_history_reset=False,capture_id=plan['capture_id'],attempt_id=plan['attempt_id'],
            execution_id=plan['execution_id'],requests=copy.deepcopy(raw['request_records']),
            measured_requests=len(raw['request_records']),priming_rows_excluded_as_labels=18,
            fitted_model_qualified=False)
        intake._macro_identity(doc,plan,ref)
        for kind in ('request','priming','fit','schema','history'):
            changed=copy.deepcopy(doc)
            if kind=='request':changed['requests'].pop();changed['measured_requests']-=1
            elif kind=='priming':changed['priming_rows_excluded_as_labels']=17
            elif kind=='fit':changed['fitted_model_qualified']=True
            elif kind=='schema':changed['schema']='gsdeploy.twoFIFO_source_rebound_composable_response_macro_training.v1'
            else:changed['case_history_reset']=True
            with self.subTest(kind=kind),self.assertRaises(ValueError):intake._macro_identity(changed,plan,ref)


if __name__=='__main__':unittest.main()
