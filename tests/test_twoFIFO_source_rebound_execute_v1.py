import copy
import json
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_source_rebound_execute_v1 as m
from tools.predictor_validation_20261027 import twoFIFO_source_rebound_review_v1 as review
from tools.predictor_validation_20261027 import twoFIFO_source_rebound_macros_v1 as macros
from tests.test_twoFIFO_owned_collect_v2 import Clock, FakeRuntime, FakeMonitor
from tests.test_twoFIFO_owned_training_review_v3 import fixture as cohort_fixture


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))


class SourceGuardTests(unittest.TestCase):
    def test_unchanged_source_and_unrelated_files_are_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source.py';p.write_text('original')
            with m.SourceChangeGuard([m.receipt(p)]) as guard:
                (Path(d)/'unrelated.log').write_text('log');guard.check()
    def test_change_restore_is_rejected_even_with_original_final_sha(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source.py';p.write_text('original');ref=m.receipt(p)
            with m.SourceChangeGuard([ref]) as guard:
                p.write_text('changed');p.write_text('original')
                self.assertEqual(m.receipt(p),ref)
                with self.assertRaisesRegex(ValueError,'changed during run'):guard.check()
    def test_preexisting_baseline_hardlink_is_allowed_when_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'source.py';alias=root/'baseline-alias.py';p.write_text('original');alias.hardlink_to(p);ref=m.receipt(p)
            with m.SourceChangeGuard([ref]) as guard: guard.check()
    def test_hardlink_write_restore_is_rejected_by_inode_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'source.py';alias=root/'alias.py';p.write_text('original');ref=m.receipt(p)
            with m.SourceChangeGuard([ref]) as guard:
                alias.hardlink_to(p);alias.write_text('changed');alias.write_text('original')
                self.assertEqual(m.receipt(p),ref)
                with self.assertRaises(ValueError):guard.check()
    def test_ancestor_directory_swap_restore_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'ancestor'/'nested'/'source.py'
            source.parent.mkdir(parents=True);source.write_text('original');ref=m.receipt(source)
            with m.SourceChangeGuard([ref]) as guard:
                (root/'ancestor').rename(root/'away')
                source.parent.mkdir(parents=True);source.write_text('changed')
                shutil.rmtree(root/'ancestor');(root/'away').rename(root/'ancestor')
                self.assertEqual(m.receipt(source),ref)
                with self.assertRaises(ValueError):guard.check()
    def test_atomic_replace_same_bytes_and_delete_are_rejected(self):
        for change in ('replace','delete'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                p=Path(d)/'source.py';p.write_text('original')
                with m.SourceChangeGuard([m.receipt(p)]) as guard:
                    if change=='replace':
                        q=Path(d)/'temp';q.write_text('original');q.replace(p)
                    else:p.unlink()
                    with self.assertRaises(ValueError):guard.check()


class ContractTests(unittest.TestCase):
    def test_real_cpu02_full_independent_audit(self):
        campaign=m.audit_campaign('artifacts/predictor-twoFIFO-source-rebound-v1-20261028-cpu09/campaign.json')
        self.assertEqual(len(campaign['attempts']),12)
        self.assertEqual(m.receipt(m.CAMPAIGN)['sha256'],m.CAMPAIGN_SHA)
    def test_shared_helpers_preserve_original_dispatch_and_cleanup_without_globals_patch(self):
        self.assertIs(m.dispatch_case,m.original.dispatch_case)
        self.assertIs(m.finish_collection,m.original.finish_collection)
        self.assertIs(m.ActivityLedger,m.original.ActivityLedger)
        self.assertIsNot(m.audit_plan,m.original.audit_plan)
        self.assertNotEqual(m.RAW_SCHEMA,m.original.RAW_SCHEMA)
    def test_old_raw_schema_is_rejected_and_new_cohort_keeps_numeric_gates(self):
        plan,raw,ledger=cohort_fixture()
        with self.assertRaises(ValueError):review.inspect_cohort(plan,raw)
        raw['schema']=m.RAW_SCHEMA
        self.assertEqual(review.inspect_cohort(plan,raw)['issues'],[])
        raw['request_records'].pop()
        with self.assertRaises(ValueError):review.inspect_cohort(plan,raw)
    def test_new_review_explicitly_rejects_old_raw_before_labels_or_audit(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'raw.json';write(p,dict(schema=m.original.RAW_SCHEMA,split='training'))
            with patch.object(m,'audit') as audit:
                with self.assertRaisesRegex(ValueError,'Training-only raw'):
                    review.review(p,Path(d)/'review',manifest_path=Path(d)/'missing.json')
            audit.assert_not_called()
    def test_new_schema_activity_still_rejects_protected_window_overlap(self):
        plan,raw,ledger=cohort_fixture();raw['schema']=m.RAW_SCHEMA
        ledger['checkpoint_spans'][1].update(start_ns=raw['measurement_windows'][0]['start_ns'],end_ns=raw['measurement_windows'][0]['start_ns'])
        with self.assertRaisesRegex(ValueError,'intersect'):review.inspect_checkpoint_activity(plan,raw,ledger)
    def test_cpu_collect_new_raw_has_all_offered_rows_and_original_timing(self):
        campaign=json.loads(m.CAMPAIGN.read_text());plan=json.loads(Path(campaign['attempts'][0]['path']).read_text())
        plan['cases']=copy.deepcopy(plan['cases'][:2]);clock=Clock();runtime=FakeRuntime(clock)
        ownership=lambda values,*a,**k:dict(owned_aliases=[m.OWNERS[next(iter(values))-1]],
            unresolved_nvml_pids=[],foreign_pids=[],ownership_evidence=[{'live':True}])
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);p=d/'plan.json';write(p,plan)
            old_audit=m.original.audit_plan;old_schema=m.original.RAW_SCHEMA
            with patch.object(m.supervisor,'validate_lease'),patch.object(m,'audit_plan',return_value=plan):
                raw=m.collect(p,d,lease_fd=-1,runtime_factory=lambda *a:runtime,monitor_factory=FakeMonitor,
                    clock_ns=clock.clock,wait_until_ns=clock.wait,ownership_check=ownership)
            self.assertEqual(m.original.audit_plan,old_audit);self.assertEqual(m.original.RAW_SCHEMA,old_schema)
            self.assertEqual(raw['status'],'completed');self.assertEqual(raw['schema'],m.RAW_SCHEMA)
            self.assertEqual(raw['kind'],'CPU_TEST_ONLY');self.assertEqual(raw['backend'],'CPU_TEST_ONLY')
            self.assertEqual(len(raw['priming_request_records']),18)
            self.assertEqual(len(raw['request_records']),sum(c['offered'] for c in plan['cases']))
            self.assertTrue(runtime.closed)
            for w in raw['measurement_windows']:
                self.assertEqual(w['end_ns']-w['start_ns'],10_000_000_000)
                self.assertEqual(w['fixed_drain_deadline_ns']-w['end_ns'],60_000_000_000)
            self.assertEqual(raw['source_rebound_executor_sources'],m.implementation_sources(m.__file__))


class ContinuationTests(unittest.TestCase):
    def setup_manifest(self,root):
        campaign=root/'campaign.json';write(campaign,{})
        attempts=[]
        for i in range(12):
            p=root/('attempt-'+str(i))/'plan.json';write(p,dict(execution_id='id-'+str(i),attempt_id='a-'+str(i)))
            attempts.append(dict(plan=m.receipt(p),attempt_id='a-'+str(i),execution_id='id-'+str(i),
                collection=str(p.parent/'collection'),claim=str(p.parent/'source-rebound-attempt-claim.json')))
        doc=dict(campaign=m.receipt(campaign),attempts=attempts,sources=[]);p=root/'manifest.json';write(p,doc)
        return p,doc
    def execution(self,item,manifest_ref,status='CHILD_EXITED_REQUIRES_PROFILE_REVIEW'):
        return dict(profile_execution=dict(schema=m.SCHEMA+'.execution_binding',manifest=manifest_ref,
            plan=item['plan'],execution_id=item['execution_id'],attempt_id=item['attempt_id'],raw_schema=m.RAW_SCHEMA,
            downstream_adapter_required=True,automatic_fit=False),budget=m.supervisor.BUDGET,status=status,
            child_reaped=True,live_group_pids=[],experiment_lock_held_through_group_exit=True,
            all_owned_descendants_exited_and_reaped=True,final_source_audit='PASS',watchdog=dict(status='EXITED',return_code=0))
    def test_default_off_and_existing_unfinished_and_claim_no_retry(self):
        for condition,status in [('new','READY_EXPLICIT_EXECUTE_REQUIRED'),('unfinished','WAITING_EXISTING_ATTEMPT_FINAL_RECEIPT'),('claim','HOLD_CLAIMED_ATTEMPT_NO_RETRY')]:
            with self.subTest(condition=condition),tempfile.TemporaryDirectory() as d:
                root=Path(d);p,doc=self.setup_manifest(root);item=doc['attempts'][0]
                if condition=='unfinished':Path(item['collection']).mkdir()
                if condition=='claim':write(Path(item['claim']),{})
                with patch.object(m,'audit',return_value=doc),patch.object(m,'execute') as execute:
                    out=m.run(p,root/'run',execute_enabled=condition!='new')
                self.assertEqual(out['status'],status);execute.assert_not_called()
    def test_failed_old_or_wrong_manifest_execution_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p,doc=self.setup_manifest(root);item=doc['attempts'][0]
            base=self.execution(item,m.receipt(p))
            for mutation in ('failed','old','foreign','cleanup','source'):
                x=copy.deepcopy(base)
                if mutation=='failed':x['status']='FAILED_NO_RETRY'
                if mutation=='old':x['profile_execution']['schema']=m.original.SCHEMA+'.execution_binding'
                if mutation=='foreign':x['profile_execution']['manifest']['sha256']='f'*64
                if mutation=='cleanup':x['all_owned_descendants_exited_and_reaped']=False
                if mutation=='source':x['final_source_audit']='FAIL'
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):m.completed_execution(x,m.receipt(p),item)
    def test_exact12_review_resource_macro_order_and_no_auto_fit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p,doc=self.setup_manifest(root);calls=[]
            def execute(plan_path,output,**kwargs):
                item=next(a for a in doc['attempts'] if a['plan']['path']==plan_path);calls.append(('execute',item['attempt_id']))
                write(Path(output)/'execution.json',self.execution(item,m.receipt(p)))
                write(Path(output)/'raw.json',dict(schema=m.RAW_SCHEMA,collector_schema=m.RAW_SCHEMA,plan=item['plan'],execution_id=item['execution_id']))
            def native(raw_path,output,execution_path,**kwargs):
                item=next(a for a in doc['attempts'] if a['collection']==str(Path(raw_path).parent));calls.append(('review',item['attempt_id']))
                q=Path(output)/'report.json';write(q,dict(schema=review.SCHEMA,source_rebound_manifest=m.receipt(p),
                    status='PASS_COMPLETE_SOURCE_REBOUND_NATIVE_TRAINING_NOT_PREDICTOR_QUALIFICATION',training_feature_extraction_eligible=True,issues=[]));return m.receipt(q)
            def extract(path):
                calls.append(('macro',Path(path).parent.parent.name));return dict(source_rebound_manifest=m.receipt(p))
            with patch.object(m,'audit',return_value=doc),patch.object(m,'execute',side_effect=execute),patch.object(review,'review',side_effect=native),patch.object(macros,'extract',side_effect=extract),patch('tools.predictor_validation_20261027.twoFIFO_observed_resources_v2.observed',return_value=dict(startup=dict(energy_joules=1),cases=[dict(measurement_energy_joules=1)])):
                out=m.run(p,root/'run',execute_enabled=True)
            self.assertEqual(out['status'],'COMPLETE_ALL_TWELVE_REBOUND_NATIVE_ATTEMPTS_REVIEWED_NO_FIT')
            self.assertEqual([x[0] for x in calls],['execute','review','macro']*12)
            self.assertEqual(len(out['completed_attempts']),12);self.assertFalse(out['automatic_fit'])
            self.assertFalse(out['numerical_fit_run']);self.assertFalse(out['fit_eligible'])
    def test_ineligible_native_review_stops_before_next_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p,doc=self.setup_manifest(root);item=doc['attempts'][0]
            write(Path(item['collection'])/'execution.json',self.execution(item,m.receipt(p)))
            write(Path(item['collection'])/'raw.json',dict(schema=m.RAW_SCHEMA,collector_schema=m.RAW_SCHEMA,plan=item['plan'],execution_id=item['execution_id']))
            q=root/'ineligible.json';write(q,dict(schema=review.SCHEMA,source_rebound_manifest=m.receipt(p),status='NOT_ELIGIBLE',training_feature_extraction_eligible=False,issues=['gap']))
            with patch.object(m,'audit',return_value=doc),patch.object(m,'execute') as execute,patch.object(review,'review',return_value=m.receipt(q)),patch.object(macros,'extract') as extract:
                out=m.run(p,root/'run',execute_enabled=True)
            self.assertEqual(out['status'],'STOPPED_NO_RETRY');execute.assert_not_called();extract.assert_not_called()


if __name__=='__main__':unittest.main()
