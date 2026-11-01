"""Independent recovery gates; supervisor is mocked, no child/GPU runs."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_prelaunch_recovery_v1 as m
from tools.predictor_validation_20261027 import twoFIFO_prelaunch_recovery_intake_v1 as intake
from tests.test_twoFIFO_source_rebound_intake_filesystem_v1 import fixture as intake_fixture


def put(path, doc):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(doc))
    return m.receipt(path)


def launch_fixture(root):
    original_ref=put(root/'old-manifest.json',{})
    tail=[]
    for index in range(8):
        plan=put(root/str(index)/'plan.json',{'CPU_FIXTURE':True,'index':index})
        tail.append(dict(plan=plan,attempt_id='attempt-'+str(index),execution_id=str(index),
            claim=str(Path(plan['path']).parent/'old-claim.json'),
            collection=str(Path(plan['path']).parent/'collection')))
    item=tail[0]
    put(Path(item['claim']),m.expected_claim(original_ref,item))
    proven=dict(original_manifest=original_ref,prelaunch_claim=m.receipt(item['claim']),tail=tail)
    recovery=dict(proof=proven)
    path=root/'recovery.json';put(path,recovery)
    return path,recovery,dict(attempts=[{}]*4+tail),item


class IndependentRecoveryLaunchTests(unittest.TestCase):
    def test_insufficient_space_never_consumes_authorization_or_invokes_supervisor(self):
        with tempfile.TemporaryDirectory() as work:
            path,doc,original,item=launch_fixture(Path(work))
            old_claim=m.receipt(item['claim'])
            with patch.object(m,'audit',return_value=doc),patch.object(m.old,'audit',return_value=original), \
                 patch.object(m.shutil,'disk_usage',return_value=shutil._ntuple_diskusage(100,90,m.old.supervisor.PREFLIGHT-1)), \
                 patch.object(m.old.supervisor,'supervise') as supervise:
                with self.assertRaisesRegex(ValueError,'preflight insufficient'):
                    m.launch(path,item,0,guard=unittest.mock.Mock())
            supervise.assert_not_called()
            self.assertFalse((Path(item['plan']['path']).parent/m.RECOVERY_CLAIM).exists())
            self.assertFalse(Path(item['collection']).exists())
            self.assertEqual(m.receipt(item['claim']),old_claim)

    def run_launch(self,root,*,mutate_restore=False,index=0):
        path,doc,original,_=launch_fixture(root)
        item=doc['proof']['tail'][index]
        old_fifth=doc['proof']['prelaunch_claim']
        authorization=Path(item['plan']['path']).parent/m.RECOVERY_CLAIM
        calls=[]
        def supervise(command,collection,*,before_launch,after_run,metadata,**kwargs):
            self.assertEqual(command[3],'_child')
            self.assertIn('--execute',command)
            self.assertEqual(metadata['manifest'],doc['proof']['original_manifest'])
            self.assertTrue(authorization.is_file())
            Path(collection).mkdir()
            before_launch();calls.append('before')
            if mutate_restore:
                saved=authorization.read_bytes();authorization.write_text('{}');authorization.write_bytes(saved)
            execution=dict(CPU_FIXTURE_NOT_NATIVE=True)
            put(Path(collection)/'execution.json',execution)
            after_run();calls.append('after')
            return execution
        with patch.object(m,'audit',return_value=doc),patch.object(m.old,'audit',return_value=original), \
             patch.object(m.old,'audit_plan'),patch.object(m.old.supervisor,'public_environment_gate'), \
             patch.object(m.shutil,'disk_usage',return_value=shutil._ntuple_diskusage(100,0,m.old.supervisor.PREFLIGHT)), \
             patch.object(m.old.supervisor,'supervise',side_effect=supervise):
            with m.old.SourceChangeGuard([m.receipt(path),item['plan'],old_fifth]) as guard:
                result=m.launch(path,item,index,guard=guard)
            with self.assertRaisesRegex(ValueError,'existing/started|already consumed'):
                m.launch(path,item,index,guard=unittest.mock.Mock())
        self.assertEqual(m.receipt(old_fifth['path']),old_fifth)
        return result,calls,item,doc

    def test_true_source_guards_before_after_and_one_shot_envelope(self):
        for index in (0,1):
            with self.subTest(index=index),tempfile.TemporaryDirectory() as work:
                (execution,envelope_ref),calls,item,doc=self.run_launch(Path(work),index=index)
                self.assertEqual(calls,['before','after'])
                envelope=m.checked(envelope_ref)
                auth=m.checked(envelope['authorization'])
                self.assertEqual(envelope['execution'],m.receipt(Path(item['collection'])/'execution.json'))
                self.assertEqual(auth['preserved_claim'],doc['proof']['prelaunch_claim'] if index==0 else None)
                self.assertFalse(auth['already_started_capture_retry'])
                self.assertEqual(m.checked(m.receipt(item['claim'])),m.expected_claim(doc['proof']['original_manifest'],item))

    def test_authorization_write_restore_is_detected_after_child_boundary(self):
        with tempfile.TemporaryDirectory() as work:
            root=Path(work)
            with self.assertRaisesRegex(ValueError,'changed'):
                self.run_launch(root,mutate_restore=True)
            self.assertTrue((root/'0'/m.RECOVERY_CLAIM).exists())
            self.assertFalse((root/'0'/m.RECOVERY_ENVELOPE).exists())


def recovery_intake_fixture(root):
    manref,manifest,oldfinal,reviews,macros,nested=intake_fixture(root)
    rows=m.checked(oldfinal)['completed_attempts']
    prefix=copy.deepcopy(rows[:4])
    old_claim=put(root/'preserved-fifth-claim.json',{'CPU_FIXTURE':True})
    stopped=put(root/'stopped.json',{'CPU_FIXTURE_STOPPED_NO_RETRY':True})
    recovery_doc=dict(proof=dict(prefix=prefix,tail=manifest['attempts'][4:],
        stopped_report=stopped,prelaunch_claim=old_claim),run_output=str(root/'new-run'))
    recovery_ref=put(root/'recovery.json',recovery_doc)
    tail=copy.deepcopy(rows[4:])
    for index,(item,row) in enumerate(zip(recovery_doc['proof']['tail'],tail)):
        base=Path(item['plan']['path']).parent
        auth=put(base/m.RECOVERY_CLAIM,dict(schema=m.SCHEMA+'.one_shot_authorization',
            recovery_manifest=recovery_ref,policy=m.POLICY,original_manifest=manref,item=item,
            preserved_claim=old_claim if index==0 else None,already_started_capture_retry=False))
        row['recovery_envelope']=put(base/m.RECOVERY_ENVELOPE,dict(schema=m.SCHEMA+'.execution_envelope',
            policy=m.POLICY,recovery_manifest=recovery_ref,authorization=auth,execution=row['execution'],
            item=item,original_manifest=manref,source_guard='KERNEL_INOTIFY_AND_FULL_CLOSURE_BEFORE_AFTER',
            no_started_capture_retry=True))
    final=dict(schema=m.SCHEMA+'.run',status=m.FINAL,no_started_capture_retries=True,
        fit_eligible=False,numerical_fit_run=False,validation_labels_read=False,
        automatic_fit=False,execute_invocations=8,
        original_manifest=manref,recovery_manifest=recovery_ref,policy=m.POLICY,stopped_report=stopped,
        prefix_attempts=prefix,completed_tail=tail,completed_attempts=prefix+tail)
    finalref=put(Path(recovery_doc['run_output'])/'report.json',final)
    return manref,manifest,finalref,recovery_ref,recovery_doc,reviews,macros


class IndependentRecoveryIntakeTests(unittest.TestCase):
    def invoke(self,root,*,mutation=None):
        manref,manifest,finalref,recovery_ref,recovery_doc,reviews,macros=recovery_intake_fixture(root)
        calls=[]
        if mutation=='envelope':
            final=m.checked(finalref);row=final['completed_tail'][0]
            envelope=m.checked(row['recovery_envelope']);envelope['policy']='OLD_POLICY'
            row['recovery_envelope']=put(Path(row['recovery_envelope']['path']),envelope)
            final['completed_attempts'][4]=row
            put(Path(finalref['path']),final)
        def review(raw_path,output,execution,*,manifest_path):
            output=Path(output);output.mkdir()
            calls.append(('review',raw_path))
            return put(output/'report.json',reviews[raw_path])
        def extract(path):
            calls.append(('macro',path))
            if mutation=='late-final' and len(calls)==24:
                Path(finalref['path']).write_text('{}')
            return copy.deepcopy(macros[path])
        with patch.object(intake.executor,'audit',return_value=manifest), \
             patch.object(intake.executor,'completed_execution'), \
             patch.object(intake.recovery,'audit',return_value=recovery_doc), \
             patch.object(intake.native_review,'review',side_effect=review), \
             patch.object(intake.macros,'extract',side_effect=extract):
            if mutation:
                with self.assertRaises(ValueError):
                    intake.intake(manref['path'],finalref['path'],recovery_manifest_path=recovery_ref['path'])
            else:
                out=intake.intake(manref['path'],finalref['path'],recovery_manifest_path=recovery_ref['path'])
                self.assertEqual((out['attempt_count'],out['measured_requests'],out['priming_requests']),(12,546,216))
                self.assertEqual((out['preserved_prefix_attempts'],out['new_policy_tail_attempts']),(4,8))
                self.assertFalse(out['downstream_old_fit_compatible'])
                self.assertTrue(all(m.receipt(r['path'])==r for r in out['sources']))
        return calls

    def test_true_twelve_filesystem_rederivations_and_new_provenance(self):
        with tempfile.TemporaryDirectory() as work:
            self.assertEqual(len(self.invoke(Path(work))),24)

    def test_foreign_envelope_stops_before_native_or_macro_rederivation(self):
        with tempfile.TemporaryDirectory() as work:
            self.assertEqual(self.invoke(Path(work),mutation='envelope'),[])

    def test_final_source_drift_on_last_macro_rederivation_fails_closed(self):
        with tempfile.TemporaryDirectory() as work:
            self.assertEqual(len(self.invoke(Path(work),mutation='late-final')),24)


if __name__ == '__main__':
    unittest.main()
