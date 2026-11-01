"""CPU-only orchestration counterexamples; no real benchmark, fit or GPU."""
from contextlib import ExitStack,contextmanager
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_public181_full_score_continuation_v1 as m


class TrainingContinuationTests(unittest.TestCase):
    def write(self,root,name,value):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return m.receipt(path)

    def proc(self,root,pid,command):
        path=root/'proc'/str(pid);path.mkdir(parents=True)
        (path/'cmdline').write_bytes(('\0'.join(command)+'\0').encode());(path/'cwd').symlink_to(root,target_is_directory=True)
        (path/'stat').write_text(str(pid)+' (with spaces) '+' '.join(['S']+['0']*18+['777']+['0']*5));return path

    def test_exact_matching_cpu_command_claim_actions_and_start_ticks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);inputs={key:self.write(root,key+'.json',{}) for key in ('fit_manifest','generated_design','resource_design','queue_claim')}
            output=root/'writer';output.mkdir();args=[]
            for key,ref in inputs.items():args+=['--'+key.replace('_','-'),ref['path']]
            args+=['--output',str(output),'--writer-pid','12']
            command=['python','-m',m.MATCHING_MODULE,*args]
            claim=self.write(output,'claim.json',dict(schema=m.matching_driver.SCHEMA,inputs=inputs,GPU_started=False,no_retries=True,writer_pid=12,
                actions=m.matching_driver.stages(inputs['fit_manifest']['path'],inputs['generated_design']['path'],inputs['resource_design']['path'],output)))
            folder=self.proc(root,123,command);identity=m.matching_identity(123,claim,proc_root=root/'proc')
            self.assertEqual(identity['start_ticks'],'777')
            (folder/'stat').write_text((folder/'stat').read_text().replace('777','778'))
            self.assertNotEqual(m.matching_identity(123,claim,proc_root=root/'proc'),identity)
            command[3]='--foreign';(folder/'cmdline').write_bytes(('\0'.join(command)+'\0').encode())
            with self.assertRaises(ValueError):m.matching_identity(123,claim,proc_root=root/'proc')

    def test_exact_resource_queue_command_manifest_and_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'resource';output.mkdir()
            manifest=self.write(root,'manifest.json',dict(output=str(output),kernel_identity={'pid':17}))
            claim=self.write(output,'claim.json',dict(schema=m.resource_queue.SCHEMA+'.claim',manifest=manifest,kernel_identity={'pid':17},no_retries=True))
            command=['python','-m',m.RESOURCE_MODULE,'run','--manifest',manifest['path'],'--execute']
            folder=self.proc(root,124,command);identity=m.resource_identity(124,manifest,claim,proc_root=root/'proc')
            self.assertEqual(identity['manifest'],manifest);command[5]='/foreign'
            (folder/'cmdline').write_bytes(('\0'.join(command)+'\0').encode())
            with self.assertRaises(ValueError):m.resource_identity(124,manifest,claim,proc_root=root/'proc')

    def test_reused_writer_checked_before_final_and_dead_without_final_stops(self):
        with patch.object(m,'matching_publication',return_value={'artifacts':[],'documents':[]}) as final:
            with self.assertRaisesRegex(ValueError,'reused'):
                m.wait_inputs({},lambda:None,identity_reader=lambda *args:(_ for _ in ()).throw(ValueError('reused writer')))
            final.assert_not_called()
        with patch.object(m,'matching_publication',return_value=None),self.assertRaisesRegex(ValueError,'without a complete'):
            m.wait_inputs({},lambda:None,identity_reader=lambda *args:None)

    def test_partial_publication_waits_only_original_writer_then_accepts_canonical_dead_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ref=self.write(root,'final.json',{'CPU_ONLY':True});value=dict(artifacts=[ref],documents=[{'CPU_ONLY':True}])
            pauses=[]
            with patch.object(m,'matching_publication',side_effect=[None,value]),patch.object(m,'resource_publication',return_value=value):
                result=m.wait_inputs({},lambda:None,identity_reader=lambda *args:{'pid':1},pause=lambda duration:pauses.append(duration))
            self.assertEqual(set(result),{'matching','resource'});self.assertEqual(pauses,[30])
            with patch.object(m,'matching_publication',return_value=value),patch.object(m,'resource_publication',return_value=value):
                self.assertEqual(len(m.wait_inputs({},lambda:None,identity_reader=lambda *args:None)),2)

    def test_first_completed_publication_guard_survives_wait_until_second_and_catches_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ref=self.write(root,'final.json',{'CPU_ONLY':True});value=dict(artifacts=[ref],documents=[{'CPU_ONLY':True}])
            def edit_restore(duration):
                path=Path(ref['path']);data=path.read_bytes();path.write_bytes(data+b' ');path.write_bytes(data)
            with patch.object(m,'matching_publication',return_value=value),patch.object(m,'resource_publication',side_effect=[None,value]),\
                    self.assertRaisesRegex(ValueError,'metadata changed|changed'):
                m.wait_inputs({},lambda:None,identity_reader=lambda *args:{'pid':1},pause=edit_restore)

    def test_fit_marker_requires_exact_fit_manifest_and_exclusive_original_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest_ref=self.write(root,'source.json',{})
            fit=self.write(root,'fit.json',dict(schema=m.response.SCHEMA,manifest=manifest_ref,status='COMPLETE_MATCHING_SOURCE_BOUND_PHASE12_TRAINING_DIAGNOSTICS',numerical_fit_run=True,validation_labels_read=False,GPU_started=False,continuation={'path':'CPU_ONLY'}))
            marker=self.write(root,'fit-complete.json',dict(name='fit',result=fit,status='COMPLETE_MATCHING_SOURCE_BOUND_PHASE12_TRAINING_DIAGNOSTICS'))
            claim=self.write(root,'fitclaim.json',dict(manifest=manifest_ref,continuation={'path':'CPU_ONLY'},no_retries=True))
            manifest=dict(matching_fit_complete_path=marker['path'],matching_driver_final_path=str(root/'none'),matching_fit_path=fit['path'],matching_fit_manifest=manifest_ref,matching_fit_claim_path=claim['path'])
            result=m.matching_publication(manifest);self.assertEqual(result['fit'],fit)
            doc=m.checked(claim);doc['no_retries']=False;Path(claim['path']).write_text(json.dumps(doc))
            with self.assertRaises(ValueError):m.matching_publication(manifest)

    def qualification_fixture(self,root,*,rank=False,q=True):
        gates={key:True for key in ('fitted_root_numerical_qualification','all_leaveout_nuisance_groups_supported','conditional_response_fit_qualified','OCR_entry_support_qualified','response_law_compilation_eligible')}
        gates['fitted_root_numerical_qualification']=rank
        manifest_ref=self.write(root,'fitmanifest.json',{});fit=self.write(root,'fit.json',{'CPU_ONLY':True})
        rows=[dict(attempt_id=str(i),joint_tuple={'CPU_ONLY':True},actual_native_quality={'measured_Q':.1} if q else None) for i in range(8)]
        intake=self.write(root,'intake.json',{'rows':rows});contract=self.write(root,'contract.json',{})
        inputs=dict(matching={'fit':fit},resource={'intake':intake,'resource_manifest':{'path':'/CPU_ONLY'},'reviews':[]})
        manifest=dict(matching_fit_manifest=manifest_ref,public_contract=contract)
        return manifest,inputs,dict(report={'manifest':manifest_ref,'fit':gates})

    def test_rank_failure_and_missing_Q_are_truthful_holds_without_law_or_benchmark(self):
        for rank,q in ((False,True),(True,False)):
            with tempfile.TemporaryDirectory() as directory:
                manifest,inputs,fit=self.qualification_fixture(Path(directory),rank=rank,q=q)
                with patch.object(m.response,'audit_fit',return_value=fit),patch.object(m.law,'open_law_session') as law:
                    result=m.qualification(manifest,inputs)
                self.assertFalse(result['eligible']);self.assertTrue(result['reasons']);law.assert_not_called()

    def test_unknown_public_key_support_holds_before_resource_reauthentication_or_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest,inputs,fit=self.qualification_fixture(Path(directory),rank=True,q=True)
            @contextmanager
            def session(path):yield SimpleNamespace(checked_payload=lambda:{'law':{'CPU_ONLY':True}})
            with patch.object(m.response,'audit_fit',return_value=fit),patch.object(m.law,'open_law_session',side_effect=session),\
                    patch.object(m.law,'support',return_value={'complete_all_keys_in_every_donor_block':False}),\
                    patch.object(m.expected,'open_resource_session') as resources:
                result=m.qualification(manifest,inputs)
            self.assertFalse(result['eligible']);self.assertIn('PUBLIC181_LAW_KEY_SUPPORT_UNKNOWN',result['reasons']);resources.assert_not_called()

    def test_resource_final_cannot_hide_missing_kernel_or_an_eighth_native_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rmanifest=self.write(root,'resource-manifest.json',{})
            kernel_campaign=self.write(root,'kernel-campaign.json',{})
            kernel=self.write(root,'kernel.json',{'campaign':kernel_campaign})
            rq=self.write(root,'queue.json',{'kernel_manifest':kernel,'resource_manifest':rmanifest})
            kref=self.write(root,'kernel-final.json',{})
            ids=[str(i) for i in range(8)];reviews=[self.write(root,'review'+i+'.json',{}) for i in ids]
            native=self.write(root,'native.json',dict(schema=m.resource.CONT_SCHEMA,manifest=rmanifest,
                status='COMPLETE_EIGHT_NATIVE_RESOURCE_ATTEMPTS_REVIEWED_NO_FIT',fit_started=False,validation_labels_read=False,
                completed_attempts=[dict(attempt_id=i,training_review=review) for i,review in zip(ids,reviews)]))
            intake=self.write(root,'intake.json',dict(schema=m.intake.SCHEMA,manifest=rmanifest,
                status='COMPLETE_EIGHT_NATIVE_RESOURCE_TUPLES_REDERIVED_NO_FIT',observed_resource_attempts=8,unknown_resource_attempts=0,
                rows=[dict(attempt_id=i,native_review=review) for i,review in zip(ids,reviews)],validation_labels_read=False,fit_started=False))
            final=self.write(root,'final.json',dict(schema=m.resource_queue.SCHEMA+'.run',manifest=rq,
                status='COMPLETE_EIGHT_NATIVE_RESOURCE_TRAINING_AND_INTAKE_NOT_VALIDATION',kernel_final=kref,native_report=native,intake=intake))
            manifest=dict(resource_final_path=final['path'],resource_queue_manifest=rq,resource_native_report_path=native['path'],
                resource_intake_path=intake['path'],expected_resource_attempt_ids=ids)
            with patch.object(m.resource_queue,'completed_kernel',return_value=(kref,{})):
                self.assertEqual(len(m.resource_publication(manifest)['reviews']),8)
                doc=m.checked(native);doc['completed_attempts'].pop();Path(native['path']).write_text(json.dumps(doc))
                final_doc=m.checked(final);final_doc['native_report']=m.receipt(native['path']);Path(final['path']).write_text(json.dumps(final_doc))
                with self.assertRaisesRegex(ValueError,'denominators'):
                    m.resource_publication(manifest)
            with patch.object(m.resource_queue,'completed_kernel',return_value=None),self.assertRaisesRegex(ValueError,'three-kernel'):
                m.resource_publication(manifest)

    def run_fixture(self,root,*,eligible=True,fits=True,failure=False,mutate=False):
        source=self.write(root,'source.json',{'CPU_ONLY':True});public=self.write(root,'public.json',{'CPU_ONLY':True})
        manifest=dict(output=str(root/'run'),public_preparation=public)
        path=self.write(root,'manifest.json',manifest)
        inputs=dict(matching={'artifacts':[source],'documents':[{'CPU_ONLY':True}],'fit':source},
            resource={'artifacts':[source],'documents':[{'CPU_ONLY':True}],'resource_manifest':source,'reviews':[source]*8})
        calls=[]
        def execute(command,**kwargs):
            calls.append(command);self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
            output=Path(command[command.index('--output')+1]);output.mkdir();self.write(output,'report.json',{'CPU_ONLY':True})
            if mutate:
                p=Path(source['path']);data=p.read_bytes();p.write_bytes(data+b' ');p.write_bytes(data)
            return SimpleNamespace(returncode=1 if failure else 0)
        with patch.object(m,'audit',return_value=manifest),patch.object(m,'wait_inputs',return_value=inputs),\
                patch.object(m,'qualification',return_value=dict(eligible=eligible,reasons=[] if eligible else ['RANK_OR_SUPPORT_UNKNOWN'])),\
                patch.object(m.budget,'audit',return_value=dict(fits_predeclared_budget=fits,public_preparation=public)):
            result=m.run(path['path'],run_command=execute)
            with self.assertRaises(FileExistsError):m.run(path['path'],run_command=execute)
        return m.checked(result),calls

    def test_supported_branch_runs_exactly_one_CPU_budget_command_and_never_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            result,calls=self.run_fixture(Path(directory))
            self.assertEqual(result['benchmark_invocations'],1);self.assertEqual(len(calls),1)
            command=calls[0];self.assertEqual(command[2],'tools.predictor_validation_20261027.twoFIFO_public181_full_score_budget_v1')
            self.assertEqual(command.count('--resource-review'),8);self.assertNotIn('--fit',command);self.assertNotIn('--execute',command)
            self.assertTrue(result['status'].startswith('COMPLETE_FULL_SIX_TERM_CPU_BUDGET'))
            self.assertFalse(result['GPU_started']);self.assertFalse(result['numerical_fit_invoked'])

    def test_unsupported_and_failed_budget_and_excess_budget_are_complete_holds(self):
        for arguments in ({'eligible':False},{'failure':True},{'fits':False}):
            with tempfile.TemporaryDirectory() as directory:
                result,calls=self.run_fixture(Path(directory),**arguments)
                self.assertTrue(result['status'].startswith('COMPLETE_HOLD'))
                self.assertEqual(len(calls),0 if arguments.get('eligible') is False else 1)
                self.assertFalse(result['automatic_retry']);self.assertFalse(result['final_gain_eligible'])

    def test_real_source_edit_restore_during_injected_cpu_stage_rejects_final_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            result,calls=self.run_fixture(Path(directory),mutate=True)
            self.assertTrue(result['status'].startswith('COMPLETE_HOLD_SOURCE'))
            self.assertEqual(result['benchmark_invocations'],1);self.assertNotIn('budget',result)


if __name__=='__main__':unittest.main()
