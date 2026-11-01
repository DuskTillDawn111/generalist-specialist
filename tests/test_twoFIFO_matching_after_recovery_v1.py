"""CPU control-flow checks; subprocess and native auditors are explicit stubs."""
import contextlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_matching_after_recovery_v1 as m
from tools.predictor_validation_20261027.native_collect import receipt


class MatchingAfterRecoveryTests(unittest.TestCase):
    def setup_case(self, root):
        def put(name,value):
            path=root/name;path.write_text(json.dumps(value));return path
        manifest=put('matching.json',{})
        attempts=[]
        for i in range(15):
            plan=put('plan'+str(i)+'.json',{})
            attempts.append(dict(plan=receipt(plan),claim=str(root/('claim'+str(i))),
                                 collection=str(root/('collection'+str(i)))))
        reviewed=put('review.json',dict(schema='gsdeploy.twoFIFO_matching_owned_producer_independent_review.v1',
            status='PASS_CPU_CONTRACTS_GUARD_FIX_VERIFIED_NOT_NATIVE_QUALIFICATION',manifest=receipt(manifest),
            unresolved_blocking_findings=[],test_summary={'status':'PASS'}))
        fit=put('fit.json',{});final=put('baseline-final.json',{'status':'synthetic-final'})
        fit_value=dict(expected_continuation_report_path=str(final),recovery_manifest={'path':'synthetic-recovery'})
        return manifest,reviewed,fit,final,dict(attempts=attempts,sources=[]),fit_value

    def run_case(self,root,*,complete=True,exit_code=0,matching_status=None,mutate_plan=False):
        manifest,reviewed,fit,final,matching,baseline=self.setup_case(root)
        target=root/'matching-run';calls=[]
        def process(command,**kwargs):
            calls.append(command)
            if mutate_plan:
                path=Path(matching['attempts'][4]['plan']['path']);old=path.read_bytes()
                path.write_bytes(b'changed');path.write_bytes(old)
            target.mkdir()
            report=dict(schema=m.matching.CONT_SCHEMA+'.run',manifest=receipt(manifest),
                status=matching_status or 'COMPLETE_ALL_FIFTEEN_MATCHING_NATIVE_ATTEMPTS_REVIEWED_NO_FIT',
                completed_attempts=[{}]*15,numerical_fit_run=False)
            (target/'report.json').write_text(json.dumps(report))
            return SimpleNamespace(returncode=exit_code)
        with patch.object(m.matching,'audit',return_value=matching), \
             patch.object(m.baseline,'audit',return_value=baseline), \
             patch.object(m.baseline,'complete_final',return_value=complete):
            ref=m.run(fit,manifest,reviewed,root/'driver',target,controller_pid=1,run_command=process)
        return json.loads(Path(ref['path']).read_text()),calls,matching

    def test_complete_baseline_launches_once_with_original_GPU_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            result,calls,_=self.run_case(Path(temp))
        self.assertEqual(len(calls),1)
        self.assertEqual(result['status'],'COMPLETE_MATCHING_CAPTURE_AFTER_EXACT12_NOT_PREDICTOR_QUALIFICATION')
        self.assertTrue(result['capture_launch_invoked']);self.assertFalse(result['numerical_fit_run'])
        self.assertFalse(result['independent_prediction_accuracy_validated'])
        self.assertIn('--execute',calls[0])

    def test_incomplete_baseline_never_creates_matching_capture_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            result,calls,matching=self.run_case(Path(temp),complete=False)
            self.assertTrue(all(not Path(a['claim']).exists() for a in matching['attempts']))
        self.assertEqual(calls,[]);self.assertFalse(result['capture_launch_invoked'])
        self.assertEqual(result['status'],'STOPPED_NO_RETRY')

    def test_zero_exit_with_hold_status_and_nonzero_exit_are_not_success(self):
        for code,status in [(0,'HOLD_CLAIMED_ATTEMPT_NO_RETRY'),(1,None)]:
            with tempfile.TemporaryDirectory() as temp:
                result,calls,_=self.run_case(Path(temp),exit_code=code,matching_status=status)
            self.assertEqual(len(calls),1);self.assertEqual(result['status'],'STOPPED_NO_RETRY')

    def test_generated_plan_change_and_restore_stops_driver(self):
        with tempfile.TemporaryDirectory() as temp:
            result,calls,_=self.run_case(Path(temp),mutate_plan=True)
        self.assertEqual(len(calls),1);self.assertEqual(result['status'],'STOPPED_NO_RETRY')
        self.assertIn('source',result['error'].lower())

    def test_review_for_other_manifest_or_unresolved_finding_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);manifest,reviewed,_,_,_,_=self.setup_case(root)
            original=json.loads(reviewed.read_text())
            for mode in ('manifest','finding','failed_tests'):
                value=json.loads(json.dumps(original))
                if mode=='manifest':value['manifest']['sha256']='other'
                elif mode=='finding':value['unresolved_blocking_findings']=['open']
                else:value['test_summary']['status']='FAIL'
                reviewed.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError,'independent'):
                    m.checked_review(reviewed,receipt(manifest))

    def test_existing_capture_is_never_queued(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);manifest,reviewed,fit,_,matching,baseline=self.setup_case(root)
            Path(matching['attempts'][3]['claim']).write_text('already claimed')
            with patch.object(m.matching,'audit',return_value=matching), \
                 patch.object(m.baseline,'audit',return_value=baseline), \
                 patch.object(m.subprocess,'run') as launch:
                with self.assertRaisesRegex(ValueError,'unstarted'):
                    m.run(fit,manifest,reviewed,root/'driver',root/'run',controller_pid=1)
            launch.assert_not_called()


if __name__=='__main__':unittest.main()
