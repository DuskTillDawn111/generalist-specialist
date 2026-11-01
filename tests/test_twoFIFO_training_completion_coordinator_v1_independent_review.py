"""Independent CPU checks of coordinator gates and permanent stage consumption.

These tests do not rerun forecast internals. Native collection and numerical
stage entry are guarded or replaced with explicitly synthetic CPU artifacts.
"""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_twoFIFO_training_completion_coordinator_v1 import CoordinatorFixture
from tools.predictor_validation_20261027 import twoFIFO_training_completion_coordinator_v1 as m
from tools.predictor_validation_20261027 import twoFIFO_owned_collect_v3 as collector


class CompletionCoordinatorIndependentReview(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.f=CoordinatorFixture(self.tmp.name)
        self.addCleanup(self.tmp.cleanup);self.addCleanup(self.f.close)
        self.addCleanup(patch.stopall)
        patch.object(collector,'execute',side_effect=AssertionError('GPU launch forbidden')).start()
        patch.object(m.continuation,'run',side_effect=AssertionError('native continuation forbidden')).start()

    def resource(self,*args):
        return self.f.write(Path(args[2])/'model.json',{'CPU_FIXTURE_ONLY':True})

    def generated(self,*args):
        return self.f.write(Path(args[2])/'report.json',dict(numerical_fit_run=True,
            status='COMPLETE_NATIVE_TRAINING_FIT_UNQUALIFIED_ALL_GENERATED_WORLDS_UNKNOWN'))

    def test_absent_and_noncomplete_final_do_not_open_any_attempt_receipts_or_consume_output(self):
        paths={row[key]['path'] for row in self.f.reviews for key in ('execution','training_review','macros')}
        original=m.checked
        def read(ref):
            if ref['path'] in paths:raise AssertionError('attempt receipt opened before final COMPLETE')
            return original(ref)
        with patch.object(m,'checked',side_effect=read),patch.object(m.resources,'run') as resource,\
                patch.object(m.generated,'run') as generated:
            out=m.run(self.f.manifest_ref['path'])
            self.assertEqual(out['status'],'WAITING_FOR_FINAL_CONTINUATION_REPORT')
            for status in ('COMPLETE','complete_all_twelve_native_attempts_reviewed',False,1,{}):
                self.f.report(status=status,completed_attempts='not a receipt list')
                out=m.run(self.f.manifest_ref['path'])
                self.assertEqual(out['status'],'STOPPED_NONCOMPLETE_CONTINUATION_REPORT')
            self.assertFalse(self.f.run_output.exists())
            resource.assert_not_called();generated.assert_not_called()

    def test_new_valid_hash_cannot_bypass_original_supervisor_or_training_role_gates(self):
        original=copy.deepcopy(self.f.reviews)
        for mutation in ('watchdog','live_pid','role','review_plan'):
            rows=copy.deepcopy(original)
            if mutation in ('watchdog','live_pid'):
                row=m.checked(rows[-1]['execution'])
                if mutation=='watchdog':row['watchdog']['return_code']=1
                else:row['live_group_pids']=[123]
                rows[-1]['execution']=self.f.write(Path(rows[-1]['execution']['path']),row)
            else:
                row=m.checked(rows[-1]['training_review'])
                if mutation=='role':row['validation_labels_read']=True
                else:row['inputs']['plan.json']=self.f.driver['attempts'][0]['plan']
                rows[-1]['training_review']=self.f.write('alternate-review-'+mutation+'.json',row)
            self.f.report(completed_attempts=rows)
            with patch.object(m.resources,'run') as resource,patch.object(m.generated,'run') as generated:
                with self.assertRaises(ValueError):m.run(self.f.manifest_ref['path'])
                resource.assert_not_called();generated.assert_not_called()
            # Restore the fixture execution body and its original durable hash.
            if mutation in ('watchdog','live_pid'):
                execution=m.checked(self.f.reviews[0]['execution'])
                last=self.f.driver['attempts'][-1]
                execution['profile_execution'].update(plan=last['plan'],execution_id=last['execution_id'],attempt_id=last['attempt_id'])
                self.assertEqual(self.f.write(Path(original[-1]['execution']['path']),execution),original[-1]['execution'])
        self.assertFalse(self.f.run_output.exists())

    def test_actual_resource_entry_rechecks_native_evidence_before_any_numeric_fit(self):
        self.f.report()
        with patch.object(m.resources,'fit_tables',side_effect=AssertionError('metadata must not enable fit')) as fit,\
                patch.object(m.generated,'run') as generated:
            out=m.checked(m.run(self.f.manifest_ref['path']))
        self.assertEqual(out['status'],'STOPPED_NO_RETRY')
        self.assertTrue((self.f.run_output/'started.json').is_file())
        self.assertFalse(out['numerical_fit_run']);fit.assert_not_called();generated.assert_not_called()
        with self.assertRaisesRegex(ValueError,'no retry'):m.run(self.f.manifest_ref['path'])

    def test_cross_attempt_or_reidentified_macro_receipts_do_not_pass_exact_twelve_gate(self):
        for mutation in ('cross_attempt','capture','schema','training_role'):
            rows=copy.deepcopy(self.f.reviews)
            if mutation=='cross_attempt':rows[-1]['macros']=rows[0]['macros']
            else:
                macro=m.checked(rows[-1]['macros'])
                if mutation=='capture':macro['capture_id']='training-block-1'
                elif mutation=='schema':macro['schema']='gsdeploy.twoFIFO_composable_response_macro_training.v2'
                else:macro['validation_labels_read']=True
                rows[-1]['macros']=self.f.write('alternate-macros-'+mutation+'.json',macro)
            self.f.report(completed_attempts=rows)
            with patch.object(m.resources,'run') as resource,patch.object(m.generated,'run') as generated:
                with self.assertRaisesRegex(ValueError,'[Mm]acro'):m.run(self.f.manifest_ref['path'])
                resource.assert_not_called();generated.assert_not_called()
            self.assertFalse(self.f.run_output.exists())

    def test_actual_started_marker_fsync_precedes_first_stage_and_systemexit_consumes_output(self):
        self.f.report();synced=[];original=os.fsync
        def sync(fd):
            synced.append(Path(os.readlink('/proc/self/fd/'+str(fd))).name)
            return original(fd)
        def resource(*args):
            self.assertIn('.started.json.tmp',synced)
            marker=json.loads((self.f.run_output/'started.json').read_text())
            self.assertEqual(len(marker['training_reviews']),12)
            self.assertEqual(marker['continuation_report'],m.receipt(self.f.final))
            raise SystemExit('synthetic interrupted CPU stage')
        with patch.object(collector.os,'fsync',side_effect=sync),patch.object(m.resources,'run',side_effect=resource),\
                patch.object(m.generated,'run') as generated:
            out=m.checked(m.run(self.f.manifest_ref['path']))
        self.assertEqual(out['status'],'STOPPED_NO_RETRY');self.assertIn('SystemExit',out['error'])
        generated.assert_not_called()
        with self.assertRaisesRegex(ValueError,'no retry'):m.run(self.f.manifest_ref['path'])

    def test_final_receipt_drift_between_stages_stops_and_preserves_consumption(self):
        self.f.report()
        def resource(*args):
            ref=self.resource(*args);self.f.report(extra_field='changed after first stage began');return ref
        with patch.object(m.resources,'run',side_effect=resource),patch.object(m.generated,'run') as generated:
            out=m.checked(m.run(self.f.manifest_ref['path']))
        self.assertEqual(out['status'],'STOPPED_NO_RETRY');self.assertIn('source drift',out['error'])
        self.assertFalse(out['generated_evaluation_run']);generated.assert_not_called()

    def test_later_stage_mutation_of_resource_output_cannot_produce_complete_with_stale_receipt(self):
        self.f.report();resource_path=self.f.run_output/'resource-provider-v3/model.json'
        def generated(*args):
            ref=self.generated(*args);resource_path.write_text('{"changed_by_later_stage": true}');return ref
        with patch.object(m.resources,'run',side_effect=self.resource),patch.object(m.generated,'run',side_effect=generated):
            out=m.checked(m.run(self.f.manifest_ref['path']))
        self.assertEqual(out['status'],'STOPPED_NO_RETRY')
        self.assertIn('Resource output',out['error'])
        with self.assertRaisesRegex(ValueError,'no retry'):m.run(self.f.manifest_ref['path'])

    def test_nested_preparation_and_execution_directories_are_rejected_before_any_creation(self):
        for prepared,destination in ((self.f.root/'new-run/prepared',self.f.root/'new-run'),
                                     (self.f.root/'new-prepared',self.f.root/'new-prepared/run')):
            with self.assertRaisesRegex(ValueError,'distinct|overlap|ancestor|nested|nesting'):
                m.prepare(self.f.driver_ref['path'],self.f.final,self.f.design_ref['path'],self.f.accepted['path'],
                    prepared,destination)
            self.assertFalse(prepared.exists());self.assertFalse(destination.exists())

    def test_unrecognized_generated_result_is_stopped_and_never_promotes_resource_or_utility(self):
        self.f.report()
        def generated(*args):return self.f.write(Path(args[2])/'report.json',dict(numerical_fit_run=True,status='PASS'))
        with patch.object(m.resources,'run',side_effect=self.resource),patch.object(m.generated,'run',side_effect=generated):
            out=m.checked(m.run(self.f.manifest_ref['path']))
        self.assertEqual(out['status'],'STOPPED_NO_RETRY')
        for key in ('GPU_started','validation_labels_read','external_validation_labels_read',
                'main_queue_provider_eligible','full_U_resource_qualification','actual_native_quality_validated','final_gain_eligible'):
            self.assertIs(out[key],False)
        with self.assertRaisesRegex(ValueError,'no retry'):m.run(self.f.manifest_ref['path'])


if __name__=='__main__':unittest.main()
