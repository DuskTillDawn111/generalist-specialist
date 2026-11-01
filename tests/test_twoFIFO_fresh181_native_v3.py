"""CPU supervisor/claim/fit-budget gates; every native launch is mocked."""
import copy
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027 import twoFIFO_fresh181_native_v3 as m
from tools.predictor_validation_20261027 import twoFIFO_matching_prospective_bridge_v3 as typed
from tools.predictor_validation_20261027.stage_response_features_v1 import receipt,checked
from tests.test_twoFIFO_fresh181_execute_v1 import PUBLIC,MATCHING


class NativeV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        cls.fit_manifest=m.save(cls.root/'fit-manifest.json',dict(executor_manifest=receipt(MATCHING)))
        cls.fit_ref=m.save(cls.root/'fit.json',dict(CPU_SYNTHETIC_ONLY=True))
        cls.fit=dict(manifest=cls.fit_manifest,sources=[])
        cls.budget_ref=m.save(cls.root/'budget.json',dict(CPU_SYNTHETIC_ONLY=True))
        cls.budget=dict(public_preparation=receipt(PUBLIC),fits_predeclared_budget=True,sources=[])
        cls.review_ref=m.save(cls.root/'independent.json',dict(schema='CPU_SYNTHETIC_ONLY.independent_review',status='PASS_CPU_FIXTURE',
            sources=[receipt(m.__file__),receipt(typed.__file__),receipt(m.timing.__file__),cls.budget_ref]))
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def prepared(self,name,*,reviewed=True,lead=300):
        return m.prepare(PUBLIC,MATCHING,self.fit_ref['path'],('1' if reviewed else '2')*32,self.root/name,
            budget_report_path=self.budget_ref['path'],independent_review_paths=[self.review_ref['path']] if reviewed else [],lead_seconds=lead)
    def guards(self):return patch.object(m.response,'audit_fit',return_value=dict(report=self.fit)),patch.object(m.timing,'audit',return_value=self.budget)

    def test_without_real_typed_fit_preparation_fails_before_creating_output(self):
        out=self.root/'no-fit'
        with self.assertRaisesRegex(ValueError,'Saved new-schema complete training fit'):
            self.prepared('no-fit')
        self.assertFalse(out.exists())

    def test_no_independent_source_budget_review_cannot_claim_or_launch(self):
        pa,pb=self.guards()
        with pa,pb:
            ref=self.prepared('unreviewed',reviewed=False);doc=m.audit(ref['path']);item=doc['attempts'][0]
            self.assertFalse(doc['native_launch_enabled'])
            with patch.object(m.supervisor,'public_environment_gate') as env,patch.object(m.supervisor,'supervise') as launch:
                with self.assertRaises(UnknownEvidence):m.execute(item['plan']['path'],item['collection'],manifest_path=ref['path'])
                env.assert_not_called();launch.assert_not_called()
            self.assertFalse(Path(item['claim']).exists());self.assertFalse(Path(item['collection']).exists())

    def test_explicit_review_fit_budget_enables_exact_supervisor_command_and_durable_no_retry_claim(self):
        pa,pb=self.guards();calls=[]
        def supervise(command,output,**kwargs):
            kwargs['before_launch']();calls.append(dict(command=command,output=output,metadata=kwargs['metadata']))
            kwargs['after_run']();return dict(CPU_TEST_ONLY=True,status='MOCK_SUPERVISED')
        with pa,pb:
            ref=self.prepared('reviewed');doc=m.audit(ref['path']);item=doc['attempts'][0]
            self.assertTrue(doc['native_launch_enabled'])
            with patch.object(m.supervisor,'public_environment_gate'),patch.object(m.supervisor,'supervise',side_effect=supervise):
                result=m.execute(item['plan']['path'],item['collection'],manifest_path=ref['path'])
                with self.assertRaisesRegex(ValueError,'No retry'):m.execute(item['plan']['path'],item['collection'],manifest_path=ref['path'])
            self.assertEqual(result['status'],'MOCK_SUPERVISED');self.assertEqual(len(calls),1)
            command=calls[0]['command'];self.assertEqual(command[2],'tools.predictor_validation_20261027.twoFIFO_fresh181_native_v3')
            self.assertEqual(command[-1],'--execute');self.assertEqual(calls[0]['metadata']['replay_budget'],self.budget_ref)
            plan=m.verify_child_claim(ref['path'],item['plan']['path'],item['collection'])
            self.assertEqual(plan['candidate_id'],item['candidate_id'])
            execution=dict(profile_execution=calls[0]['metadata'],command=calls[0]['command'],budget=m.supervisor.BUDGET,
                status='CHILD_EXITED_REQUIRES_PROFILE_REVIEW',collection_subprocess_launched=True,process_group=123,
                child_reaped=True,live_group_pids=[],experiment_lock_held_through_group_exit=True,
                all_owned_descendants_exited_and_reaped=True,final_source_audit='PASS',watchdog=dict(status='EXITED',return_code=0))
            m.completed_execution(execution,ref,item)
            for change in [lambda d:d['profile_execution'].update(replay_budget=self.fit_ref),
                    lambda d:d['profile_execution'].update(typed_producer_schema='old'),
                    lambda d:d.update(command=['foreign-child']),lambda d:d['command'].__setitem__(2,'old.module'),
                    lambda d:d.update(live_group_pids=[456]),lambda d:d['watchdog'].update(return_code=1)]:
                bad=copy.deepcopy(execution);change(bad)
                with self.assertRaises(ValueError):m.completed_execution(bad,ref,item)
            claim=Path(item['claim']);original=claim.read_bytes();value=checked(receipt(claim));value['candidate_id']='foreign'
            claim.write_text(m.json.dumps(value))
            try:
                with self.assertRaisesRegex(ValueError,'exclusive fit/budget/candidate'):m.verify_child_claim(ref['path'],item['plan']['path'],item['collection'])
            finally:claim.write_bytes(original)

    def test_insufficient_budget_remains_hold_even_when_independently_reviewed(self):
        pa,pb=self.guards()
        with pa,pb:
            original=self.budget;self.budget=dict(original,fits_predeclared_budget=False)
            # patch.object captured the original object; update its one field
            # temporarily to exercise actual deterministic prepare/audit.
            original['fits_predeclared_budget']=False
            try:
                ref=self.prepared('insufficient');doc=m.audit(ref['path']);plan=checked(doc['attempts'][0]['plan'])
                self.assertFalse(doc['native_launch_enabled'])
                with self.assertRaises(UnknownEvidence):m.require_native_ready(plan)
            finally:original['fits_predeclared_budget']=True;self.budget=original

    def test_new_budget_requires_new_prepare_and_preserves_original90plus120(self):
        pa,pb=self.guards()
        with pa,pb:
            ref=self.prepared('lead600',lead=600);doc=m.audit(ref['path']);plan=checked(doc['attempts'][0]['plan'])
            self.assertEqual(doc['lead_seconds'],600);self.assertEqual(plan['forecast_barrier']['lead_seconds'],600)
            self.assertEqual((plan['measurement_ns'],plan['drain_ns']),(90_000_000_000,120_000_000_000))
            self.assertEqual(len(plan['cases'][0]['arrivals']),181)
            self.assertTrue(plan['no_retries'])

    def test_other_candidate_change_restore_before_claim_prevents_launch(self):
        pa,pb=self.guards()
        with pa,pb:
            ref=self.prepared('all-plan-watch');doc=m.audit(ref['path']);item=doc['attempts'][0]
            def environment():
                path=Path(doc['attempts'][3]['plan']['path']);original=path.read_bytes()
                path.write_bytes(original+b' ');path.write_bytes(original)
            with patch.object(m.supervisor,'public_environment_gate',side_effect=environment),patch.object(m.supervisor,'supervise') as launch:
                with self.assertRaises(ValueError):m.execute(item['plan']['path'],item['collection'],manifest_path=ref['path'])
                launch.assert_not_called()
            self.assertFalse(Path(item['claim']).exists());self.assertFalse(Path(item['collection']).exists())

    def test_supervisor_failure_consumes_claim_and_cannot_retry(self):
        pa,pb=self.guards()
        with pa,pb:
            ref=self.prepared('failed-after-claim');item=checked(ref)['attempts'][0]
            with patch.object(m.supervisor,'public_environment_gate'),patch.object(m.supervisor,'supervise',side_effect=RuntimeError('CPU_MOCK_FAILURE')) as launch:
                with self.assertRaisesRegex(RuntimeError,'CPU_MOCK_FAILURE'):
                    m.execute(item['plan']['path'],item['collection'],manifest_path=ref['path'])
                self.assertTrue(Path(item['claim']).exists())
                with self.assertRaisesRegex(ValueError,'No retry'):
                    m.execute(item['plan']['path'],item['collection'],manifest_path=ref['path'])
                self.assertEqual(launch.call_count,1)

    def test_postrun_audit_report_change_restore_before_seal_is_rejected(self):
        # This focused race test mocks only previously tested semantic parsers;
        # the post-run outer filesystem watch and fixed receipts remain real.
        root=self.root/'postrun';collection=root/'attempt'/'collection';collection.mkdir(parents=True)
        report=m.save(collection/'typed-report.json',dict(CPU_TEST_ONLY=True))
        pref=m.save(collection/'prefix.json',dict(CPU_TEST_ONLY=True))
        gate=m.save(collection/'gate.json',dict(prefix=pref))
        seal=m.save(collection/'seal.json',dict(typed_report=report,seal_begin_ns=1))
        execution=m.save(collection/'execution.json',dict(process_group=123))
        claim=m.save(root/'claim.json',dict(CPU_TEST_ONLY=True))
        plan=dict(saved_matching_fit=self.fit_ref,sources=[],execution_id='cpu',candidate_id='cpu',
            forecast_barrier=dict(gate_request_path=gate['path'],seal_path=seal['path'],prefix_path=pref['path']))
        plan_ref=m.save(root/'attempt'/'plan.json',plan);manifest=m.save(root/'manifest.json',dict(CPU_TEST_ONLY=True))
        item=dict(plan=plan_ref,collection=str(collection),claim=claim['path'])
        @contextmanager
        def session(*args):yield object()
        def changed(*args,**kwargs):
            path=Path(report['path']);original=path.read_bytes();path.write_bytes(original+b' ');path.write_bytes(original)
            return dict(CPU_TEST_ONLY=True)
        with patch.object(m,'audit',return_value=dict(attempts=[item])),patch.object(m,'audit_plan',return_value=plan),\
                patch.object(m,'completed_execution'),patch.object(typed,'open_target_session',side_effect=session),\
                patch.object(typed,'_prefix_and_gate',return_value=({},plan,dict(prefix_snapshot={},ready_ownership_evidence={}),dict(prefix=pref))),\
                patch.object(m.prefix,'inspect_native_ownership'),patch.object(typed,'audit_report',side_effect=changed),\
                patch.object(typed,'audit_seal_structure') as accept:
            with self.assertRaises(ValueError):
                m.audit_completed_prefix(manifest['path'],plan_ref['path'],execution['path'],root/'review.json')
            accept.assert_not_called()
        self.assertFalse((root/'review.json').exists())


if __name__=='__main__':unittest.main()
