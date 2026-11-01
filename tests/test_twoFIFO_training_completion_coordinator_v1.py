"""CPU fixtures for exact metadata gates and one-shot stage boundaries."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_training_completion_coordinator_v1 as m


class CoordinatorFixture:
    def __init__(self,root):
        self.root=Path(root);self.reviews=[];items=[]
        self.guard=self.write('guard.json',{'frozen':'CPU-only fixture'})
        for block in range(1,4):
            for shard in range(1,5):
                name=f'training-block-{block}-shard-{shard}'
                plan=dict(attempt_id=name,execution_id=name+'-execution',capture_id=f'training-block-{block}',
                    split='training',shards_in_block=4,shard_index=shard,cases=[dict(offered=1)],priming=[{}]*18)
                plan_ref=self.write(name+'/plan.json',plan)
                collection=self.root/name/'collection'
                item=dict(plan=plan_ref,attempt_id=name,execution_id=plan['execution_id'],collection=str(collection))
                items.append(item)
                execution=dict(profile_execution=dict(schema=m.continuation.collector.SCHEMA+'.execution_binding',
                    plan=plan_ref,execution_id=plan['execution_id'],attempt_id=name),budget=m.continuation.collector.supervisor.BUDGET,
                    status='CHILD_EXITED_REQUIRES_PROFILE_REVIEW',child_reaped=True,live_group_pids=[],
                    experiment_lock_held_through_group_exit=True,all_owned_descendants_exited_and_reaped=True,
                    final_source_audit='PASS',watchdog=dict(status='EXITED',return_code=0))
                execution_ref=self.write(name+'/collection/execution.json',execution)
                review=dict(schema=m.resources.intake.REVIEW_SCHEMA,status=m.resources.intake.COMPLETE,
                    validation_labels_read=False,training_feature_extraction_eligible=True,issues=[],
                    inputs={'plan.json':plan_ref},raw={'path':'RAW_MUST_NOT_BE_OPENED_BY_COORDINATOR_METADATA_GATE','sha256':'x'})
                review_ref=self.write('continuation/'+name+'/native-review/report.json',review)
                macros_ref=self.write('continuation/'+name+'/macros-v3.json',dict(
                    schema='gsdeploy.twoFIFO_composable_response_macro_training.v3',
                    status='COMPLETE_ATTEMPT_DISJOINT_TRAINING_MACROS_NOT_FITTED_PROVIDER',mode='natural',
                    validation_labels_read=False,attempt_id=name,execution_id=plan['execution_id'],capture_id=plan['capture_id'],
                    whole_attempt_history_used_for_visible_prefix=True,case_history_reset=False,
                    measured_requests=1,priming_rows_excluded_as_labels=18,CPU_FIXTURE_ONLY=True))
                self.reviews.append(dict(attempt_id=name,execution=execution_ref,training_review=review_ref,macros=macros_ref))
        campaign=self.write('campaign.json',{'attempts':[item['plan'] for item in items]})
        self.driver=dict(campaign=campaign,attempts=items,sources=[self.guard])
        self.driver_ref=self.write('driver/manifest.json',self.driver)
        self.design=dict(campaign=campaign,sources=[self.guard])
        self.design_ref=self.write('design.json',self.design)
        self.accepted=self.write('native-accepted.json',dict(schema='gsdeploy.twoFIFO_native_v3_independent_review.v1',
            status='PASS_NATIVE_CONTRACTS_AND_RESOURCE_COVERAGE',training_feature_extraction_eligible=True,
            validation_labels_read=False,plan=items[0]['plan']))
        self.final=self.root/'continuation/report.json';self.run_output=self.root/'fixed-run'
        self.context=ExitStack()
        self.context.enter_context(patch.object(m.continuation,'audit',return_value=self.driver))
        self.context.enter_context(patch.object(m.generated,'audit_design',return_value=(self.design_ref,self.design)))
        self.manifest_ref=m.prepare(self.driver_ref['path'],self.final,self.design_ref['path'],self.accepted['path'],
            self.root/'prepared',self.run_output)

    def write(self,path,value):
        path=self.root/path;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));return m.receipt(path)

    def report(self,**changes):
        value=dict(schema=m.continuation.SCHEMA+'.run',status=m.COMPLETE,manifest=self.driver_ref,
            no_retries=True,validation_labels_read=False,completed_attempts=copy.deepcopy(self.reviews))
        value.update(changes);self.write('continuation/report.json',value);return value

    def close(self):self.context.close()


class CompletionCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.fixture=CoordinatorFixture(self.tmp.name)
        self.addCleanup(self.tmp.cleanup);self.addCleanup(self.fixture.close)

    def no_fit(self):
        stack=ExitStack();resource=stack.enter_context(patch.object(m.resources,'run'))
        generated=stack.enter_context(patch.object(m.generated,'run'));self.addCleanup(stack.close)
        return resource,generated

    def test_absent_final_report_is_single_waiting_check_without_reviews_or_outputs(self):
        resource,generated=self.no_fit()
        with patch.object(m,'complete_reviews',side_effect=AssertionError('reviews accessed')):
            result=m.run(self.fixture.manifest_ref['path'])
        self.assertEqual(result['status'],'WAITING_FOR_FINAL_CONTINUATION_REPORT')
        self.assertFalse(result['reviews_opened']);self.assertFalse(self.fixture.run_output.exists())
        resource.assert_not_called();generated.assert_not_called()

    def test_every_noncomplete_final_status_stops_without_review_or_fit(self):
        resource,generated=self.no_fit()
        for status in ['STOPPED_NO_RETRY','WAITING_EXISTING_ATTEMPT_FINAL_RECEIPT','READY_EXPLICIT_EXECUTE_REQUIRED',None]:
            self.fixture.report(status=status)
            with patch.object(m,'complete_reviews',side_effect=AssertionError('reviews accessed')):
                result=m.run(self.fixture.manifest_ref['path'])
            self.assertEqual(result['status'],'STOPPED_NONCOMPLETE_CONTINUATION_REPORT')
            self.assertFalse(self.fixture.run_output.exists())
        resource.assert_not_called();generated.assert_not_called()

    def test_exact_twelve_order_duplicate_and_receipt_drift_rejected_before_fit(self):
        resource,generated=self.no_fit();rows=self.fixture.reviews
        for bad in [rows[:-1],rows+[rows[-1]],rows[:-1]+[rows[0]],list(reversed(rows))]:
            self.fixture.report(completed_attempts=bad)
            with self.assertRaises(ValueError):m.run(self.fixture.manifest_ref['path'])
        self.fixture.report();Path(rows[0]['training_review']['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'receipt drift'):m.run(self.fixture.manifest_ref['path'])
        self.assertFalse(self.fixture.run_output.exists());resource.assert_not_called();generated.assert_not_called()

    def test_ineligible_twelfth_review_and_wrong_native_execution_rejected(self):
        resource,generated=self.no_fit();rows=copy.deepcopy(self.fixture.reviews)
        review=m.checked(rows[-1]['training_review']);review['training_feature_extraction_eligible']=False
        rows[-1]['training_review']=self.fixture.write('bad-review.json',review)
        self.fixture.report(completed_attempts=rows)
        with self.assertRaisesRegex(ValueError,'twelve eligible'):m.run(self.fixture.manifest_ref['path'])
        rows=copy.deepcopy(self.fixture.reviews);rows[-1]['execution']=rows[0]['execution']
        self.fixture.report(completed_attempts=rows)
        with self.assertRaisesRegex(ValueError,'another native attempt'):m.run(self.fixture.manifest_ref['path'])
        resource.assert_not_called();generated.assert_not_called()

    def test_source_drift_and_stage_path_tampering_rejected_before_final_check(self):
        resource,generated=self.no_fit();path=Path(self.fixture.manifest_ref['path'])
        original=json.loads(path.read_text());bad=copy.deepcopy(original)
        bad['stages']['resource_provider_v3']=str(self.fixture.root/'arbitrary-output')
        path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError,'Fixed stage'):m.run(path)
        path.write_text(json.dumps(original));Path(self.fixture.guard['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'source drift'):m.run(path)
        resource.assert_not_called();generated.assert_not_called()

    def test_macro_receipt_for_another_attempt_cannot_complete_exact_twelve_binding(self):
        resource,generated=self.no_fit();rows=copy.deepcopy(self.fixture.reviews)
        rows[-1]['macros']=rows[0]['macros'];self.fixture.report(completed_attempts=rows)
        with self.assertRaisesRegex(ValueError,'exact attempt'):m.run(self.fixture.manifest_ref['path'])
        resource.assert_not_called();generated.assert_not_called();self.assertFalse(self.fixture.run_output.exists())

    def test_complete_receipts_persist_before_resource_then_generated_once_at_fixed_outputs(self):
        self.fixture.report();order=[];expected=[row['training_review']['path'] for row in self.fixture.reviews]
        manifest=m.checked(self.fixture.manifest_ref)
        def resource(campaign,reviews,output):
            self.assertTrue((self.fixture.run_output/'started.json').is_file())
            self.assertEqual(reviews,expected);self.assertEqual(output,manifest['stages']['resource_provider_v3'])
            started=json.loads((self.fixture.run_output/'started.json').read_text())
            self.assertEqual(len(started['training_reviews']),12);self.assertFalse(started['numerical_fit_run'])
            order.append('resource');return self.fixture.write(Path(output)/'model.json',dict(CPU_FIXTURE_ONLY=True))
        def generated(design,reviews,output):
            self.assertEqual(order,['resource']);self.assertTrue((self.fixture.run_output/'resource-stage-complete.json').is_file())
            self.assertEqual(design,self.fixture.design_ref['path']);self.assertEqual(reviews,expected)
            self.assertEqual(output,manifest['stages']['generated_wholeblock_evaluation_v1']);order.append('generated')
            return self.fixture.write(Path(output)/'report.json',dict(numerical_fit_run=True,
                status='COMPLETE_NATIVE_TRAINING_FIT_UNQUALIFIED_ALL_GENERATED_WORLDS_UNKNOWN'))
        with patch.object(m.resources,'run',side_effect=resource),patch.object(m.generated,'run',side_effect=generated):
            ref=m.run(self.fixture.manifest_ref['path'])
        result=m.checked(ref);self.assertEqual(order,['resource','generated'])
        self.assertTrue(result['status'].startswith('COMPLETE_CPU_TRAINING_DIAGNOSTICS'))
        self.assertFalse(result['final_gain_eligible']);self.assertFalse(result['actual_native_quality_validated'])
        self.assertFalse(result['full_U_resource_qualification']);self.assertFalse(result['GPU_started'])
        with self.assertRaisesRegex(ValueError,'no retry'):m.run(self.fixture.manifest_ref['path'])

    def test_partial_resource_output_stops_permanently_without_generated_or_retry(self):
        self.fixture.report()
        def failed(*args):
            Path(args[2]).mkdir();raise RuntimeError('injected interrupted resource fit')
        with patch.object(m.resources,'run',side_effect=failed) as resource,patch.object(m.generated,'run') as generated:
            result=m.checked(m.run(self.fixture.manifest_ref['path']))
            self.assertEqual(result['status'],'STOPPED_NO_RETRY');self.assertFalse(result['generated_evaluation_run'])
            with self.assertRaisesRegex(ValueError,'no retry'):m.run(self.fixture.manifest_ref['path'])
        self.assertEqual(resource.call_count,1);generated.assert_not_called()

    def test_source_change_after_resource_prevents_generated_stage(self):
        self.fixture.report()
        def changed(*args):
            ref=self.fixture.write(Path(args[2])/'model.json',{'CPU_FIXTURE_ONLY':True})
            Path(self.fixture.guard['path']).write_text('{"drift": true}');return ref
        with patch.object(m.resources,'run',side_effect=changed),patch.object(m.generated,'run') as generated:
            result=m.checked(m.run(self.fixture.manifest_ref['path']))
        self.assertEqual(result['status'],'STOPPED_NO_RETRY');self.assertIn('source drift',result['error'])
        generated.assert_not_called()

    def test_nested_prepare_and_run_outputs_are_rejected_before_creating_either(self):
        for prepared,execution in [(self.fixture.root/'new-run'/'prepared',self.fixture.root/'new-run'),
                (self.fixture.root/'new-prepare',self.fixture.root/'new-prepare'/'run')]:
            with self.assertRaisesRegex(ValueError,'without nesting'):
                m.prepare(self.fixture.driver_ref['path'],self.fixture.final,self.fixture.design_ref['path'],
                    self.fixture.accepted['path'],prepared,execution)
            self.assertFalse(prepared.exists());self.assertFalse(execution.exists())

    def test_resource_output_mutation_during_generated_stage_cannot_complete(self):
        self.fixture.report();saved=[]
        def resource(*args):
            ref=self.fixture.write(Path(args[2])/'model.json',{'CPU_FIXTURE_ONLY':True});saved.append(ref);return ref
        def generated(*args):
            Path(saved[0]['path']).write_text('{"unexpected_mutation":true}')
            return self.fixture.write(Path(args[2])/'report.json',dict(numerical_fit_run=True,
                status='COMPLETE_CONDITIONAL_GENERATED_WHOLEBLOCK_TRAINING_EVALUATION'))
        with patch.object(m.resources,'run',side_effect=resource),patch.object(m.generated,'run',side_effect=generated):
            result=m.checked(m.run(self.fixture.manifest_ref['path']))
        self.assertEqual(result['status'],'STOPPED_NO_RETRY');self.assertIn('Resource output changed',result['error'])


if __name__=='__main__':unittest.main()
