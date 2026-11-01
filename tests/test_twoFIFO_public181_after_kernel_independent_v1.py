"""Independent CPU counterexamples for pinned kernel→resource continuation."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tools.predictor_validation_20261027 import twoFIFO_public181_after_kernel_v1 as m


class IndependentResourceContinuation(unittest.TestCase):
    def proc_fixture(self,root):
        parent=root/'kernel-manifest.json';parent.write_text(json.dumps({'output':str(root/'parent'),'queue_identity':{'pid':123}}))
        manifest_ref=m.receipt(parent);folder=root/'parent';folder.mkdir()
        claim=folder/'claim.json';claim.write_text(json.dumps(dict(manifest=manifest_ref,queue_identity={'pid':123},no_retries=True)))
        proc=root/'proc'/'123';proc.mkdir(parents=True)
        command=[sys.executable,'-m',m.MODULE,'run','--manifest',str(parent),'--execute']
        (proc/'cmdline').write_bytes(('\0'.join(command)+'\0').encode())
        fields=['S']+['0']*18+['777']+['0']*5
        (proc/'stat').write_text('123 (odd name) '+' '.join(fields))
        return manifest_ref,m.receipt(claim),proc

    def test_kernel_identity_binds_exact_claim_command_and_startticks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest,claim,proc=self.proc_fixture(root)
            actual=m.kernel_identity(123,manifest,claim,proc_root=root/'proc')
            self.assertEqual(actual['start_ticks'],'777')
            text=(proc/'stat').read_text();(proc/'stat').write_text(text.replace('777','778'))
            changed=m.kernel_identity(123,manifest,claim,proc_root=root/'proc')
            self.assertNotEqual(actual,changed)
            command=actual['command'];command[5]='/foreign';(proc/'cmdline').write_bytes(('\0'.join(command)+'\0').encode())
            with self.assertRaises(ValueError):m.kernel_identity(123,manifest,claim,proc_root=root/'proc')
            (proc/'cmdline').write_bytes(('\0'.join(actual['command'])+'\0').encode())
            document=m.checked(claim);document['no_retries']=False;Path(claim['path']).write_text(json.dumps(document))
            with self.assertRaises(ValueError):m.kernel_identity(123,manifest,m.receipt(claim['path']),proc_root=root/'proc')

    def test_changed_writer_cannot_release_even_complete_final_and_dead_without_final_fails(self):
        manifest=dict(kernel_identity={'pid':123,'start_ticks':'777'},kernel_manifest={},kernel_claim={})
        with patch.object(m,'completed_kernel',return_value=({'path':'CPU_ONLY'},{})) as completed:
            with self.assertRaisesRegex(ValueError,'start ticks changed'):
                m.wait_kernel(manifest,{}, {},lambda:None,identity_reader=lambda *a:{'pid':123,'start_ticks':'778'})
            completed.assert_not_called()
        with patch.object(m,'completed_kernel',return_value=None),self.assertRaisesRegex(ValueError,'without complete final'):
            m.wait_kernel(manifest,{}, {},lambda:None,identity_reader=lambda *a:(_ for _ in ()).throw(FileNotFoundError()))

    def test_kernel_final_success_flag_cannot_hide_missing_capture_or_cpu_stage(self):
        manifest=dict(kernel_final_path='/CPU_ONLY',kernel_manifest={'path':'CPU_ONLY'})
        result=dict(schema=m.kernel.SCHEMA+'.run',manifest=manifest['kernel_manifest'],
            status='COMPLETE_THREE_NATIVE_CAPTURES_AND_CPU_DIAGNOSTICS_NO_FIT',fit_started=False,
            automatic_retry=False,validation_labels_read=False,completed_captures=[{'capture_id':'a'}],completed_cpu_stages=[])
        with patch.object(m.kernel,'load_publication',return_value=({},result)),\
                patch.object(m.kernel,'cpu_stages',return_value=[{'name':'stage'}]),self.assertRaises(ValueError):
            m.completed_kernel(manifest,{'device_attributes':{'path':'CPU_ONLY'},'output':'CPU_ONLY'},
                {'runs':[{'capture_id':'a'},{'capture_id':'b'},{'capture_id':'c'}]})

    def run_fixture(self,root,*,missing_attempt=False,unknown_intake=False,wait_failure=False,mutate_review=False):
        resource_path=root/'resource.json';resource_path.write_text('{}');resource_ref=m.receipt(resource_path)
        source_path=root/'kernel-final.json';source_path.write_text('{}');final=m.receipt(source_path)
        manifest=dict(schema=m.SCHEMA,output=str(root/'run'),kernel_identity={'pid':123,'start_ticks':'777'},
            resource_manifest=resource_ref)
        path=root/'manifest.json';path.write_text(json.dumps(manifest))
        resource={'attempts':[dict(attempt_id='a'+str(i),claim=str(root/('claim'+str(i))),
            collection=str(root/('collection'+str(i)))) for i in range(8)]};calls=[]
        def command(args,**kwargs):
            calls.append(args);out=Path(args[args.index('--output')+1]);out.mkdir(parents=True)
            if len(calls)==1:
                attempts=[]
                for a in resource['attempts']:
                    review_path=root/('review-'+a['attempt_id']+'.json');review_path.write_text(json.dumps({'CPU_TEST_ONLY':True}))
                    attempts.append(dict(attempt_id=a['attempt_id'],training_review=m.receipt(review_path)))
                if missing_attempt:attempts.pop()
                doc=dict(schema=m.executor.CONT_SCHEMA,manifest=resource_ref,
                    status='COMPLETE_EIGHT_NATIVE_RESOURCE_ATTEMPTS_REVIEWED_NO_FIT',fit_started=False,
                    validation_labels_read=False,completed_attempts=attempts)
            else:
                if mutate_review:
                    path=root/'review-a0.json';original=path.read_bytes();path.write_bytes(original+b' ');path.write_bytes(original)
                doc=dict(schema=m.intake.SCHEMA,manifest=resource_ref,
                    status='COMPLETE_EIGHT_NATIVE_RESOURCE_TUPLES_REDERIVED_NO_FIT',
                    observed_resource_attempts=7 if unknown_intake else 8,
                    unknown_resource_attempts=1 if unknown_intake else 0)
            (out/'report.json').write_text(json.dumps(doc));return SimpleNamespace(returncode=0)
        waiter=patch.object(m,'wait_kernel',side_effect=ValueError('incomplete pinned parent')) if wait_failure else \
            patch.object(m,'wait_kernel',return_value=(final,{}))
        with patch.object(m,'audit',return_value=(manifest,resource,{},{})),waiter,\
                patch.object(m.executor,'audit',return_value=resource):
            result_ref=m.run(path,run_command=command)
            with self.assertRaises(FileExistsError):m.run(path,run_command=command)
        return m.checked(result_ref),calls

    def test_no_parent_success_never_launches_and_no_retry_even_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            result,calls=self.run_fixture(Path(directory),wait_failure=True)
            self.assertEqual(result['status'],'STOPPED_NO_RETRY_PREFIX_PRESERVED');self.assertEqual(calls,[])
            self.assertFalse(result['GPU_started'])

    def test_all8_attempts_and_all8_native_intake_are_required_no_survivor_success(self):
        for missing,unknown in ((True,False),(False,True),(False,False)):
            with self.subTest(missing=missing,unknown=unknown),tempfile.TemporaryDirectory() as directory:
                result,calls=self.run_fixture(Path(directory),missing_attempt=missing,unknown_intake=unknown)
                self.assertEqual(len(calls),1 if missing else 2)
                if missing or unknown:self.assertEqual(result['status'],'STOPPED_NO_RETRY_PREFIX_PRESERVED')
                else:
                    self.assertEqual(result['status'],'COMPLETE_EIGHT_NATIVE_RESOURCE_TRAINING_AND_INTAKE_NOT_VALIDATION')
                    self.assertEqual(calls[1].count('--review'),8)
                self.assertFalse(result['automatic_retry']);self.assertFalse(result['fit_started'])

    def test_review_changed_then_restored_during_intake_never_authenticates_success(self):
        with tempfile.TemporaryDirectory() as directory:
            result,calls=self.run_fixture(Path(directory),mutate_review=True)
            self.assertEqual(len(calls),2)
            self.assertEqual(result['status'],'STOPPED_NO_RETRY_PREFIX_PRESERVED')
            self.assertIn('source metadata changed during run',result['error'])


if __name__=='__main__':unittest.main()
