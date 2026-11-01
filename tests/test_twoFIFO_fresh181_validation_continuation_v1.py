"""One-shot orchestration counterexamples with no GPU or numerical fit."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_fresh181_validation_continuation_v1 as m


class ContinuationTests(unittest.TestCase):
    def write(self,root,name,value):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));return m.receipt(path)

    def test_exact_fullscore_writer_claim_cwd_and_start_ticks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);parent=self.write(root,'parent.json',{'output':str(root/'upstream')})
            claim=self.write(root,'upstream/claim.json',dict(schema=m.cpu.SCHEMA+'.claim',manifest=parent,
                no_retries=True,CPU_only=True,benchmark_invocations_maximum=1))
            command=['python','-m',m.MODULE,'run','--manifest','parent.json','--execute']
            folder=root/'proc/123';folder.mkdir(parents=True)
            (folder/'cmdline').write_bytes(('\0'.join(command)+'\0').encode())
            (folder/'cwd').symlink_to(root,target_is_directory=True)
            (folder/'stat').write_text('123 (name with spaces) '+' '.join(['S']+['0']*18+['777']+['0']*5))
            identity=m.writer_identity(123,parent,claim,proc_root=root/'proc')
            self.assertEqual(identity['start_ticks'],'777')
            (folder/'stat').write_text((folder/'stat').read_text().replace('777','778'))
            self.assertNotEqual(m.writer_identity(123,parent,claim,proc_root=root/'proc'),identity)
            command[5]='foreign.json';(folder/'cmdline').write_bytes(('\0'.join(command)+'\0').encode())
            with self.assertRaises(ValueError):m.writer_identity(123,parent,claim,proc_root=root/'proc')

    def upstream(self,root,status=m.READY):
        manifest=self.write(root,'upstream-manifest.json',{})
        source=self.write(root,'source.json',{'CPU_TEST':True})
        final=dict(schema=m.cpu.SCHEMA+'.run',manifest=manifest,status=status,CPU_only=True,
            GPU_started=False,numerical_fit_invoked=False,validation_labels_read=False,CPU_TEST_ONLY=False,
            budget=source,eligibility=source,inputs=dict(matching={'fit':source},resource={'resource_manifest':source,'reviews':[source]*8}),
            benchmark_invocations=1,budget_qualified=True,fits_predeclared_budget=True,full_score_budget_ready=True)
        final_ref=self.write(root,'upstream/report.json',final)
        manifest_doc=dict(fullscore_manifest=manifest,fullscore_final_path=final_ref['path'],independent_budget_review_path=str(root/'independent/report.json'),
            independent_source_reviews=[],evaluation_protocol=m.native.expected.EVALUATION_PROTOCOL)
        return manifest_doc,final_ref,final

    def test_pid_reuse_is_checked_before_complete_final(self):
        with patch.object(m.cpu,'publication') as publish,ExitStack() as stack:
            with self.assertRaisesRegex(ValueError,'reused'):
                m.wait_fullscore({},stack,lambda:None,identity_reader=lambda *args:(_ for _ in ()).throw(ValueError('reused PID')))
            publish.assert_not_called()

    def test_dead_original_only_accepted_with_canonical_final_and_test_fake_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest,ref,doc=self.upstream(root)
            with ExitStack() as stack:
                self.assertEqual(m.wait_fullscore(manifest,stack,lambda:None,identity_reader=lambda *args:None),(ref,doc))
            doc['CPU_TEST_ONLY']=True;Path(ref['path']).write_text(json.dumps(doc))
            with ExitStack() as stack,self.assertRaisesRegex(ValueError,'Exact real'):
                m.wait_fullscore(manifest,stack,lambda:None,identity_reader=lambda *args:None)
            Path(ref['path']).unlink()
            with ExitStack() as stack,self.assertRaisesRegex(ValueError,'without a complete'):
                m.wait_fullscore(manifest,stack,lambda:None,identity_reader=lambda *args:None)

    def budget_review(self,manifest,ref,doc):
        return dict(schema=m.BUDGET_REVIEW_SCHEMA,status=m.BUDGET_REVIEW_STATUS,independent_review=True,
            fullscore_manifest=manifest['fullscore_manifest'],fullscore_final=ref,budget=doc['budget'],
            fit=doc['inputs']['matching']['fit'],resource_manifest=doc['inputs']['resource']['resource_manifest'],
            resource_reviews=doc['inputs']['resource']['reviews'],evaluation_protocol=manifest['evaluation_protocol'],
            test_summary={'status':'PASS'},unresolved_blocking_findings=[],GPU_started=False,
            validation_labels_read=False,numerical_fit_invoked=False,final_gain_eligible=False,
            source_bindings=m.review_required_refs(doc))

    def test_budget_review_exact_budget_identity_and_independent_role_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest,ref,doc=self.upstream(root)
            value=self.budget_review(manifest,ref,doc)
            review=self.write(root,'independent/report.json',value)
            self.assertEqual(m.accept_budget_review(manifest,ref,doc,review),[review])
            for key,new in [('budget',ref),('independent_review',False),('validation_labels_read',True),('unresolved_blocking_findings',['blocker']),('resource_reviews',[]),('status','PASS_CPU_TEST')]:
                bad=dict(value);bad[key]=new;badref=self.write(root,'independent/report.json',bad)
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Separate real independent'):
                    m.accept_budget_review(manifest,ref,doc,badref)

    def test_generic_pass_review_cannot_omit_required_current_source_and_exact_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest,ref,doc=self.upstream(root)
            value=self.budget_review(manifest,ref,doc);value['source_bindings']=[]
            review=self.write(root,'independent/report.json',value)
            with self.assertRaisesRegex(ValueError,'must cover'):
                m.accept_budget_review(manifest,ref,doc,review)

    def test_budget_wait_keeps_upstream_mutation_restore_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest,ref,doc=self.upstream(root)
            def restore(_):
                path=Path(doc['budget']['path']);data=path.read_bytes();path.write_bytes(data+b' ');path.write_bytes(data)
            with self.assertRaisesRegex(ValueError,'changed'),ExitStack() as stack:
                guard=m._watch(stack,[ref,doc])
                m.wait_budget_review(manifest,ref,doc,stack,guard.check,pause=restore)

    def test_new_budget_review_mutation_restore_during_acceptance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest,ref,doc=self.upstream(root)
            review=self.write(root,'independent/report.json',self.budget_review(manifest,ref,doc))
            def restore(*args):
                path=Path(review['path']);data=path.read_bytes();path.write_bytes(data+b' ');path.write_bytes(data)
                return [review]
            with self.assertRaisesRegex(ValueError,'changed'),ExitStack() as stack,\
                    patch.object(m,'accept_budget_review',side_effect=restore):
                m.wait_budget_review(manifest,ref,doc,stack,lambda:None)

    def test_false_fullscore_readiness_never_audits_fit_or_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest,ref,doc=self.upstream(root);doc['budget_qualified']=False
            with patch.object(m.cpu,'audit') as audit,self.assertRaisesRegex(ValueError,'Only complete qualified'):
                m.qualify_fullscore(manifest,ref,doc)
            audit.assert_not_called()

    def test_fullscore_cannot_bind_different_training_publication_even_with_ready_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest,ref,doc=self.upstream(root)
            with patch.object(m.cpu,'audit',return_value={}),\
                    patch.object(m.cpu,'matching_publication',return_value={'fit':ref,'documents':[]}),\
                    patch.object(m.cpu,'resource_publication',return_value={'resource_manifest':ref,'documents':[]}),\
                    patch.object(m.cpu,'qualification') as qualify,patch.object(m.native.timing,'audit') as budget,\
                    self.assertRaisesRegex(ValueError,'exact canonical complete'):
                m.qualify_fullscore(manifest,ref,doc)
            qualify.assert_not_called();budget.assert_not_called()

    def fixture(self,root,*,hold=False,review_status='PASS_CONDITIONAL_NATIVE_VALIDATION_EVIDENCE_AND_DESCRIPTIVE_ERRORS',
                fail_capture=None,restore_after_final=False):
        manifest,final_ref,final=self.upstream(root,status='COMPLETE_HOLD_TEST' if hold else m.READY)
        manifest.update(output=str(root/'run'),native_preparation_path=str(root/'run/native4'),namespace='1'*32,sources=[final['budget']])
        own=self.write(root,'manifest.json',manifest)
        matching=self.write(root,'matching.json',{})
        fitmanifest=self.write(root,'fitmanifest.json',{'executor_manifest':matching})
        parent={'public_preparation':matching,'matching_fit_manifest':fitmanifest}
        inputs=final['inputs'];review=self.write(root,'independent/report.json',self.budget_review(manifest,final_ref,final))
        calls=[];native_prepares=[]
        def prepare(public,producer,fit,namespace,out,**kwargs):
            native_prepares.append((public,producer,fit,namespace,kwargs));out=Path(out)
            attempts=[]
            for i,cid in enumerate(m.CANDIDATES):
                plan=self.write(out,cid+'/plan.json',{'candidate_id':cid})
                attempts.append(dict(candidate_id=cid,plan=plan,collection=str(out/cid/'collection'),claim=str(out/cid/'claim.json')))
            return self.write(out,'manifest.json',{'native_launch_enabled':True,'attempts':attempts})
        def runner(command,**kwargs):
            module=command[2];calls.append((module,command,kwargs['env']))
            arg=lambda key:command[command.index('--'+key)+1]
            if module.endswith('native_v4'):
                target=m.checked(m.receipt(arg('manifest')));item=next(a for a in target['attempts'] if a['plan']['path']==arg('plan'))
                if fail_capture==item['candidate_id']:return SimpleNamespace(returncode=1)
                self.write(root,Path(item['claim']).relative_to(root),{'CPU_TEST':True})
                self.write(root,Path(item['collection']).relative_to(root)/'execution.json',{'candidate_id':item['candidate_id']})
                self.write(root,Path(item['collection']).relative_to(root)/'raw.json',{'CPU_TEST':True})
            elif module.endswith('validation_review_v1'):
                target=m.checked(m.receipt(arg('manifest')));item=next(a for a in target['attempts'] if str(Path(a['collection'])/'raw.json')==arg('raw'))
                self.write(Path(arg('output')),'report.json',dict(schema=m.validation.SCHEMA,status=review_status,
                    manifest=m.receipt(arg('manifest')),plan=item['plan'],execution=m.receipt(arg('execution')),sources=[]))
            elif module.endswith('validation_evaluation_v1'):
                reviews=command[command.index('--reviews')+1:command.index('--output')]
                self.write(Path(arg('output')),'report.json',dict(schema=m.evaluation.SCHEMA,
                    manifest=m.receipt(arg('manifest')),reviews=[m.receipt(p) for p in reviews],candidate_denominator=4,
                    pair_denominator=6,offered_per_candidate=181,worlds_per_candidate=12,
                    final_gain_eligible=False,all_predeclared_pilot_thresholds_pass=False))
            else:raise AssertionError('Unplanned command '+repr(command))
            return SimpleNamespace(returncode=0)
        atomic=m.cpu.atomic_save
        def save_final(path,value):
            atomic(path,value)
            if restore_after_final and value.get('native_validation_complete'):
                path=Path(final['budget']['path']);data=path.read_bytes();path.write_bytes(data+b' ');path.write_bytes(data)
        with ExitStack() as patches:
            patches.enter_context(patch.object(m,'audit',return_value=manifest))
            patches.enter_context(patch.object(m,'wait_fullscore',return_value=(final_ref,final)))
            patches.enter_context(patch.object(m,'qualify_fullscore',return_value=(parent,inputs,{'CPU_TEST':True})))
            wait=patches.enter_context(patch.object(m,'wait_budget_review',return_value=[review]))
            patches.enter_context(patch.object(m.native,'prepare',side_effect=prepare))
            patches.enter_context(patch.object(m.native,'audit',side_effect=lambda p:m.checked(m.receipt(p))))
            patches.enter_context(patch.object(m.native,'audit_plan',side_effect=lambda p:m.checked(m.receipt(p))))
            patches.enter_context(patch.object(m.native,'require_native_ready',side_effect=lambda p:p))
            patches.enter_context(patch.object(m.native,'completed_execution',return_value=None))
            patches.enter_context(patch.object(m,'wait_capacity',return_value={'CPU_TEST':True}))
            patches.enter_context(patch.object(m.cpu,'atomic_save',side_effect=save_final))
            result=m.checked(m.run(own['path'],run_command=runner))
            return result,calls,native_prepares,wait.call_count

    def test_upstream_hold_never_waits_independent_review_or_prepares_or_launches(self):
        with tempfile.TemporaryDirectory() as directory:
            result,calls,prepares,wait=self.fixture(Path(directory),hold=True)
        self.assertEqual(calls,[]);self.assertEqual(prepares,[]);self.assertEqual(wait,0)
        self.assertIn('UPSTREAM_FULLSCORE_NOT_READY',result['status']);self.assertFalse(result['GPU_started'])

    def test_exact_serial_four_native_review_then_aggregate_preserves_negative_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            result,calls,prepares,wait=self.fixture(Path(directory))
        self.assertEqual(len(prepares),1);self.assertEqual(wait,1);self.assertEqual(len(calls),9)
        self.assertEqual(result['completed_candidates'],m.CANDIDATES)
        self.assertTrue(result['native_validation_complete']);self.assertFalse(result['pilot_thresholds_pass'])
        self.assertFalse(result['independent_accuracy_qualified']);self.assertFalse(result['final_gain_eligible'])
        for i in range(4):
            self.assertTrue(calls[2*i][0].endswith('native_v4'))
            self.assertTrue(calls[2*i+1][0].endswith('validation_review_v1'))
            self.assertEqual(calls[2*i+1][2]['CUDA_VISIBLE_DEVICES'],'')
        self.assertTrue(calls[-1][0].endswith('validation_evaluation_v1'))

    def test_failed_capture_stops_later_candidates_preserving_partial_aggregate_and_fixed_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            result,calls,prepares,_=self.fixture(Path(directory),fail_capture=m.CANDIDATES[1])
        self.assertEqual(len(prepares),1);self.assertEqual(len(calls),4)
        self.assertEqual(result['native_capture_invocations'],2);self.assertFalse(result['readiness_qualified'])
        self.assertFalse(result['native_validation_complete']);self.assertIn('native_manifest',result)
        self.assertTrue(result['aggregate_source_authenticated'])
        self.assertEqual(result['candidate_denominator'],4);self.assertEqual(result['pair_denominator'],6)
        self.assertEqual([r['status'] for r in result['candidate_outcomes']],
            ['REVIEWED_CONDITIONAL_EVIDENCE','FAILED_NO_RETRY','NOT_STARTED','NOT_STARTED'])
        self.assertTrue(all(r['offered_requests']==181 and r['planned_request_ledger'] for r in result['candidate_outcomes']))
        self.assertTrue(result['candidate_outcomes'][0]['actual_request_ledger'])
        self.assertIsNone(result['candidate_outcomes'][1]['actual_request_ledger'])

    def test_unqualified_review_stops_next_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            result,calls,_,_=self.fixture(Path(directory),review_status='NOT_ELIGIBLE_NATIVE_VALIDATION')
        self.assertEqual(len(calls),3);self.assertEqual(result['native_capture_invocations'],1)
        self.assertFalse(result['native_validation_complete']);self.assertEqual(len(result['reviews']),1)
        self.assertEqual([r['status'] for r in result['candidate_outcomes']],
            ['REVIEW_NOT_ELIGIBLE','NOT_STARTED','NOT_STARTED','NOT_STARTED'])
        self.assertEqual(result['candidate_denominator'],4);self.assertEqual(result['pair_denominator'],6)
        self.assertTrue(result['candidate_outcomes'][0]['actual_request_ledger'])

    def test_mutation_restore_after_success_write_revokes_qualification_without_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            result,calls,_,_=self.fixture(Path(directory),restore_after_final=True)
        self.assertEqual(len(calls),9);self.assertFalse(result['native_validation_complete'])
        self.assertFalse(result['readiness_qualified']);self.assertNotIn('aggregate',result)
        self.assertIn('unauthenticated_diagnostics',result)

    def test_one_controller_attempt_cannot_be_reentered(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);root.joinpath('run').mkdir()
            ref=self.write(root,'manifest.json',{'output':str(root/'run')})
            with patch.object(m,'audit',return_value=m.checked(ref)),patch.object(m,'wait_fullscore') as wait:
                with self.assertRaises(FileExistsError):m.run(ref['path'])
                wait.assert_not_called()

    def test_capacity_wait_uses_unchanged_supervisor_floor_and_checks_sources_each_poll(self):
        free=[m.native.supervisor.PREFLIGHT-1,m.native.supervisor.PREFLIGHT]
        checks=[];pauses=[]
        result=m.wait_capacity('/tmp',lambda:checks.append(True),
            disk_usage=lambda path:SimpleNamespace(free=free.pop(0)),pause=lambda t:pauses.append(t))
        self.assertEqual(pauses,[30]);self.assertEqual(len(checks),4)
        self.assertEqual(result['required_free_bytes'],m.native.supervisor.PREFLIGHT)
        self.assertEqual(result['actual_free_bytes'],m.native.supervisor.PREFLIGHT)


if __name__=='__main__':unittest.main()
