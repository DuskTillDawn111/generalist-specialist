"""Execution control fixtures; GPU/native collector is never called here."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import h600_four_group_run_v1 as m


class FinalRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.plan_path=self.root/'plan.json'
        m.save(self.plan_path,{'role_reservation':m.receipt(m.final.RESERVATION)})
        self.plan=m.read(self.plan_path)
        self.plan.update(created_ns=1,candidates=[dict(id=cid) for cid in m.final.IDS])
        self.model_path=self.root/'model.json';m.save(self.model_path,{'model':'fixture','status':'FAIL'})
        self.role={'service_hypothesis':{'model':m.receipt(self.model_path)}}
        self.role_path=self.root/'roles.json';m.save(self.role_path,self.role)
        self.plan['role_reservation']=m.receipt(self.role_path)
        self.plan_path.unlink();m.save(self.plan_path,self.plan)
        self.predictions=[dict(candidate_id=cid,execution_precheck='PREDICTED_COMPLETE',known=1200)
                          for cid in m.final.IDS]
        self.sealed_path=self.root/'seal.json'
        self.sealed=dict(schema=m.SCHEMA+'.seal',entry_source=m.receipt(m.__file__),
            plan=m.receipt(self.plan_path),validation_report=m.receipt(self.model_path),
            training_completed=m.receipt(self.model_path),independent_gate=m.receipt(self.model_path),
            independent_gate_status='FAIL',independent_report_verification={'verified':True},information_cutoff_ns=2,created_ns=3,
            service_predictions=self.predictions,candidate_order=list(m.final.ORDER),
            locked_policies=m.final.LOCKS,source_receipts=[m.receipt(self.model_path),
                m.receipt(self.role_path),m.receipt(m.__file__)],
            requests=9600,diagnostic=True,not_for_training=True,final_gain_eligible=False,
            no_retries=True,no_seed_replacement=True)
        self.sealed['seal_sha256']=m.digest(self.sealed);m.save(self.sealed_path,self.sealed)
        self.prerequisite=dict(plan=self.plan,independent_gate=self.sealed['independent_gate'],
            independent_gate_status='FAIL',independent_report_verification={'verified':True},
            source_receipts=list(self.sealed['source_receipts']))

    def controlled(self):
        patches=[patch.object(m.final,'audit_plan',return_value=self.plan),
                 patch.object(m,'sources',return_value=[m.receipt(m.__file__)]),
                 patch.object(m,'_prerequisites',return_value=self.prerequisite),
                 patch.object(m.service,'predict',side_effect=lambda p,c,model:
                    copy.deepcopy(next(x for x in self.predictions if x['candidate_id']==c['id'])))]
        for p in patches:p.start();self.addCleanup(p.stop)

    def collector(self,plan,cid,output):
        Path(output).mkdir(parents=True)
        m.save(Path(output)/'raw.json',dict(candidate_id=cid))

    def audited(self,plan,raw,seal_ns):
        cid=m.read(raw)['candidate_id']
        return dict(candidate_id=cid,source_receipts=[m.receipt(raw)])

    def report(self,plan,paths,output,seal_ns):
        Path(output).mkdir(parents=True)
        value=dict(status='DESCRIPTIVE_COMPLETE' if len(paths)==8 else 'DESCRIPTIVE_INCOMPLETE',
            known=1200*len(paths),source_receipts=[m.receipt(p) for p in paths])
        m.save(Path(output)/'report.json',value);return value

    def test_explicit_diagnostic_required_before_launch(self):
        with patch.object(m.final.native_flow,'execute') as execute:
            with self.assertRaises(ValueError):m.run(self.sealed_path,self.root/'execution')
            execute.assert_not_called()

    def test_fixed_eight_order_and_failed_gate_preserved(self):
        self.controlled()
        with patch.object(m.final.native_flow,'execute',side_effect=self.collector) as execute, \
             patch.object(m.final,'audit_actual',side_effect=self.audited), \
             patch.object(m.final,'report_actual',side_effect=self.report):
            result=m.run(self.sealed_path,self.root/'execution',diagnostic=True)
        self.assertEqual([call.args[1] for call in execute.call_args_list],list(m.final.ORDER))
        self.assertEqual(result['known'],9600)
        self.assertEqual(result['independent_gate_status'],'FAIL')
        self.assertFalse(result['final_gain_eligible'])
        self.assertTrue((self.root/'F8-attempt.json').exists())

    def test_bad_first_native_stops_without_retry_and_retains_failure(self):
        self.controlled()
        def bad(*args):raise RuntimeError('native fixture failure')
        with patch.object(m.final.native_flow,'execute',side_effect=bad) as execute, \
             patch.object(m.final,'report_actual',side_effect=self.report):
            result=m.run(self.sealed_path,self.root/'execution',diagnostic=True)
        self.assertEqual(execute.call_count,1)
        self.assertEqual(result['requested'],9600);self.assertEqual(result['known'],0)
        self.assertIn('native fixture failure',result['execution_error'])
        self.assertEqual(m.read(self.root/'execution/run-state.json')['status'],'INCOMPLETE_NO_RETRY')

    def test_different_output_cannot_retry_same_final_plan(self):
        self.controlled()
        with patch.object(m.final.native_flow,'execute',side_effect=self.collector) as execute, \
             patch.object(m.final,'audit_actual',side_effect=self.audited), \
             patch.object(m.final,'report_actual',side_effect=self.report):
            m.run(self.sealed_path,self.root/'execution',diagnostic=True)
            count=execute.call_count
            with self.assertRaises(FileExistsError):m.run(self.sealed_path,self.root/'different-output',diagnostic=True)
            self.assertEqual(execute.call_count,count)

    def test_source_change_blocks_next_final_candidate(self):
        self.controlled()
        def collect(plan,cid,output):
            self.collector(plan,cid,output)
            self.model_path.write_text('{"changed":true}')
        with patch.object(m.final.native_flow,'execute',side_effect=collect) as execute, \
             patch.object(m.final,'audit_actual',side_effect=self.audited):
            with self.assertRaises(ValueError):m.run(self.sealed_path,self.root/'execution',diagnostic=True)
        self.assertEqual(execute.call_count,1)
        self.assertTrue((self.root/'F8-attempt.json').exists())

    def test_rehashed_seal_cannot_change_order_or_refit_predictions(self):
        self.controlled()
        bad=copy.deepcopy(self.sealed);bad['candidate_order'].reverse()
        bad['seal_sha256']=m.digest({k:v for k,v in bad.items() if k!='seal_sha256'})
        self.sealed_path.unlink();m.save(self.sealed_path,bad)
        with self.assertRaises(ValueError):m.check_sealed(self.sealed_path)
        bad=copy.deepcopy(self.sealed);bad['service_predictions'][0]['invented']=1
        bad['seal_sha256']=m.digest({k:v for k,v in bad.items() if k!='seal_sha256'})
        self.sealed_path.unlink();m.save(self.sealed_path,bad)
        with patch.object(m.final.native_flow,'execute') as execute:
            with self.assertRaisesRegex(ValueError,'fresh numerical'):m.run(self.sealed_path,self.root/'execution',diagnostic=True)
            execute.assert_not_called()


if __name__=='__main__':unittest.main()
