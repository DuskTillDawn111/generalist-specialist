"""CPU-only writer, seal/fit order, HOLD and source restoration counterexamples."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import kernel_dag_response_numerical_continuation_v1 as m


class NumericalContinuationTests(unittest.TestCase):
    def test_wait_accepts_partial_json_only_while_exact_kernel_writer_lives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);native=m.write(root/'manifest.json',{})
            plan=dict(continuation=native);final=root/'report.json'
            identity=dict(pid=77,start_ticks='exact',manifest=native)
            manifest=dict(kernel_identity=identity,kernel_final_path=str(final));waits=[]
            def pause(seconds):
                waits.append(seconds)
                if len(waits)==1:final.write_text('{')
                else:final.write_text(json.dumps(dict(schema=m.kernel.SCHEMA+'.run',manifest=native,
                    status='COMPLETE_THREE_NATIVE_CAPTURES_AND_CPU_DIAGNOSTICS_NO_FIT')))
            with patch.object(m.numerical,'native_publication',return_value=('final','report','refs')):
                result=m.wait_kernel(manifest,plan,lambda:None,poll_seconds=1,pause=pause,
                    identity_reader=lambda *args:identity)
            self.assertEqual(result,('final','report','refs'));self.assertEqual(waits,[1,1])

    def test_wait_rejects_reused_pid_and_dead_writer_without_complete_final(self):
        identity=dict(pid=77,start_ticks='exact',manifest={})
        manifest=dict(kernel_identity=identity,kernel_final_path='/missing-numerical-final')
        with self.assertRaisesRegex(ValueError,'PID/start ticks'):
            m.wait_kernel(manifest,dict(continuation={'path':'manifest'}),lambda:None,
                identity_reader=lambda *args:dict(identity,start_ticks='reused'))
        def absent(*args):raise FileNotFoundError()
        with self.assertRaisesRegex(ValueError,'exited before complete'):
            m.wait_kernel(manifest,dict(continuation={'path':'manifest'}),lambda:None,identity_reader=absent)

    def test_stopped_kernel_final_is_not_a_retry_or_cpu_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);native=m.write(root/'manifest.json',{});final=root/'final.json'
            m.write(final,dict(schema=m.kernel.SCHEMA+'.run',manifest=native,status='STOPPED_NO_RETRY_PREFIX_PRESERVED'))
            identity=dict(pid=77,start_ticks='exact',manifest=native)
            with self.assertRaisesRegex(ValueError,'stopped/incomplete'):
                m.wait_kernel(dict(kernel_identity=identity,kernel_final_path=str(final)),dict(continuation=native),
                    lambda:None,identity_reader=lambda *args:identity)

    def fixture(self,root,*,qualified=False,fail_fit=False,restore_seal=False,restore_after_result=False):
        root=Path(root);manifest_ref=m.write(root/'manifest.json',{});plan_ref=m.write(root/'plan.json',{})
        final_ref=m.write(root/'kernel-final.json',{})
        manifest=dict(numerical_plan=plan_ref,output=str(root/'run'),sources=[])
        plan={};review_ref=m.write(root/'review.json',dict(schema=m.SCHEMA+'.independent_review',
            status='PASS_CPU_WAIT_SEAL_FIT_INTERFACES_NOT_NUMERICAL_QUALIFICATION',manifest=manifest_ref,
            numerical_plan=plan_ref,unresolved_blocking_findings=[],test_summary=dict(status='PASS')))
        calls=[]
        def child(command,**kwargs):
            calls.append((command,kwargs));op=command[3];out=Path(command[command.index('--output')+1]);out.mkdir()
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
            self.assertEqual(kwargs['env']['OPENBLAS_NUM_THREADS'],'1')
            result=dict(plan=plan_ref,GPU_started=False,fit_started=False,validation_labels_read=False)
            if op=='seal':
                result.update(schema=m.numerical.SCHEMA+'.seal',status='SEALED_COMPLETE_CPU_NUMERICAL_DATA_NO_FIT')
                m.write(out/'seal.json',result)
            else:
                if fail_fit:return SimpleNamespace(returncode=1)
                seal=root/'run/seal/seal.json';ref=m.receipt(seal)
                if restore_seal:
                    original=seal.read_bytes();seal.write_bytes(original+b' ');seal.write_bytes(original)
                result.update(schema=m.numerical.SCHEMA+'.fit',status='COMPLETE_NUMERICAL_TRAINING_DIAGNOSTIC' if qualified else 'UNKNOWN_NUMERICAL_QUALIFICATION_FAILED',
                    seal=ref,numerical_qualification=qualified,prospective_prediction_qualified=False,fit_started=True)
                m.write(out/'report.json',result)
            return SimpleNamespace(returncode=0)
        original_write=m.write
        def writing(path,value):
            ref=original_write(path,value)
            if restore_after_result and str(path).endswith('/fit-complete.json'):
                seal=root/'run/seal/seal.json';original=seal.read_bytes()
                seal.write_bytes(original+b' ');seal.write_bytes(original)
            return ref
        with patch.object(m,'audit',return_value=(manifest,plan)),patch.object(m,'wait_kernel',return_value=(final_ref,{},[final_ref])),\
             patch.object(m,'temporary_requirement',return_value=1),patch.object(m.kernel,'wait_capacity',return_value={}),\
             patch.object(m,'write',side_effect=writing):
            result=m.run(manifest_ref['path'],review_ref['path'],run_command=child)
        return result,calls

    def test_cpu_only_seal_then_fit_once_qualified_remains_training_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            result,calls=self.fixture(tmp,qualified=True)
            self.assertEqual([c[0][3] for c in calls],['seal','fit'])
            self.assertEqual(result['status'],'COMPLETE_CPU_NUMERICAL_TRAINING_DIAGNOSTIC')
            self.assertTrue(result['numerical_qualification']);self.assertFalse(result['prospective_prediction_qualified'])
            self.assertFalse(result['GPU_started']);self.assertFalse(result['resource_chain_dependency'])

    def test_full_cohort_numerical_hold_is_completed_and_unqualified(self):
        with tempfile.TemporaryDirectory() as tmp:
            result,calls=self.fixture(tmp)
            self.assertEqual(result['status'],'COMPLETE_CPU_NUMERICAL_HOLD')
            self.assertFalse(result['numerical_qualification']);self.assertEqual(len(calls),2)

    def test_failed_fit_preserves_seal_prefix_and_never_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            result,calls=self.fixture(tmp,fail_fit=True)
            self.assertEqual(result['status'],'STOPPED_NO_RETRY_PREFIX_PRESERVED')
            self.assertEqual(len(calls),2);self.assertTrue(result['fit_command_started'])
            self.assertTrue((Path(tmp)/'run/seal-complete.json').is_file())
            self.assertFalse(result['numerical_qualification'])

    def test_seal_change_and_restore_during_fit_is_rejected_by_real_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            result,calls=self.fixture(tmp,qualified=True,restore_seal=True)
            self.assertEqual(result['status'],'STOPPED_NO_RETRY_PREFIX_PRESERVED')
            self.assertFalse(result['numerical_qualification']);self.assertEqual(len(calls),2)

    def test_final_guard_failure_after_qualified_return_revokes_qualification(self):
        with tempfile.TemporaryDirectory() as tmp:
            result,calls=self.fixture(tmp,qualified=True,restore_after_result=True)
            self.assertEqual(result['status'],'STOPPED_NO_RETRY_PREFIX_PRESERVED')
            self.assertFalse(result['numerical_qualification']);self.assertFalse(result['prospective_prediction_qualified'])
            self.assertNotIn('numerical_result',result)
            self.assertIn('unauthenticated_numerical_result_diagnostic',result)

    def test_root_review_must_bind_exact_manifest_and_plan(self):
        ref={};manifest={'path':'manifest'};plan={'path':'plan'}
        good=dict(schema=m.SCHEMA+'.independent_review',status='PASS_CPU_WAIT_SEAL_FIT_INTERFACES_NOT_NUMERICAL_QUALIFICATION',
            manifest=manifest,numerical_plan=plan,unresolved_blocking_findings=[],test_summary=dict(status='PASS'))
        for field in good:
            bad=dict(good);bad[field]='wrong'
            with self.subTest(field=field),patch.object(m,'read',return_value=bad),self.assertRaises(ValueError):
                m.checked_review(ref,manifest,plan)


if __name__=='__main__':unittest.main()
