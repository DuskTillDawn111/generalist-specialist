"""CPU control failures; all numerical/GPU stage operations mocked."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import h600_four_group_continuation_v1 as m


class FinalContinuationTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.base=self.root/'artifacts';self.base.mkdir();self.out=self.root/'continuation'
        self.patches=[]
        def patcher(*a,**kw):
            p=patch.object(*a,**kw);self.patches.append(p);self.addCleanup(p.stop);return p.start()
        self.patch=patcher;patcher(m,'BASE',self.base)
        paths=m._paths()
        for key,path in paths.items():
            if 'completed' in key:continue
            path.parent.mkdir(parents=True,exist_ok=True)
            m.save(path,dict(cpu_fixture=key))
        self.role=m.receipt(paths['role_reservation']);self.plan=dict(role_reservation=self.role)
        self.tprotocol=dict(role_reservation=self.role)
        self.vprotocol=dict(training_protocol=m.receipt(paths['training_protocol']))
        self.v=SimpleNamespace(__file__=__file__,audit_protocol=lambda p:copy.deepcopy(self.vprotocol),
            audit_root_review=lambda p,r:(copy.deepcopy(self.vprotocol),{}))
        patcher(m,'independent',return_value=self.v)
        patcher(m.execution.final,'audit_plan',return_value=self.plan)
        patcher(m.training,'audit_protocol',return_value=self.tprotocol)
        patcher(m,'sources',side_effect=lambda:[m.receipt(m.__file__)])
        self.active={pid:dict(pid=pid,start_ticks=pid,argv=['python','-u','-m',module,'run',
            '--protocol',str(paths[name+'_protocol']),'--review',str(paths[name+'_protocol'].with_name('root-review.json'))],
            cwd=str(m.execution.final.ROOT))
            for name,pid,module in (('training',m.T_PID,m.T_MODULE),('validation',90001,m.V_MODULE))}
        self.vprotocol['training_process_identity']=copy.deepcopy(self.active[m.T_PID])
        patcher(m.training,'process_identity',side_effect=lambda pid:copy.deepcopy(self.active.get(pid)))
        patcher(m,'matching_processes',side_effect=lambda module:[copy.deepcopy(p) for p in self.active.values() if m.runs_module(p,module)])
        self.protocol=m.prepare(self.out,90001);self.protocol_path=self.out/'protocol.json'
        self.review_path=self.out/'root-review.json'
        m.save(self.review_path,dict(schema=m.SCHEMA+'.main_thread_review',
            decision='APPROVED_FOR_CONDITIONAL_F8_DIAGNOSTIC_ONLY',reviewer_role='main_thread',
            protocol=m.receipt(self.protocol_path),created_ns=self.protocol['created_ns']+1,
            **{k:self.protocol[k] for k in ('source','plan','training_protocol','validation_protocol','role_reservation')},
            preserve_prediction_failures_and_unknown=True,statistical_significance_claim=False,final_gain_eligible=False))
        self.no_native=patcher(m.execution.final.native_flow,'execute',side_effect=AssertionError('Never GPU'))

    def completed_upstream(self,gate='FAIL'):
        paths=m._paths();vfolder=self.base/'admission-validation-v1/report';vfolder.mkdir(parents=True)
        for filename in ('report.json','gate.json','validation-audit.json','report-verification.json'):
            m.save(vfolder/filename,dict(cpu_fixture=filename))
        verification=dict(verified=True,not_for_training=True)
        (vfolder/'report-verification.json').unlink();m.save(vfolder/'report-verification.json',verification)
        t=dict(schema=m.training.SCHEMA+'.completed',source=self.protocol['upstream'][0]['source'],
            protocol=self.protocol['training_protocol'],final_gain_eligible=False)
        m.save(paths['training_completed'],t)
        self.validation=dict(schema=m.V_SCHEMA+'.completed',source=self.protocol['upstream'][1]['source'],
            protocol=self.protocol['validation_protocol'],training_completed=m.receipt(paths['training_completed']),
            not_for_training=True,independent_runs_launched=4,final_launched=False,GPU_started=True,
            native_requested=4800,native_known=4800,report=m.receipt(vfolder/'report.json'),gate=m.receipt(vfolder/'gate.json'),
            validation_audit=m.receipt(vfolder/'validation-audit.json'),report_verification=m.receipt(vfolder/'report-verification.json'),
            status='DIAGNOSTIC_COLLECTED',collection_error=None,independent_gate_status=gate,final_gain_eligible=False)
        review_path=paths['validation_protocol'].with_name('root-review.json');m.save(review_path,dict(cpu_fixture=True))
        self.validation['root_review']=m.receipt(review_path)
        m.save(paths['validation_completed'],self.validation)
        self.prerequisite=dict(independent_gate_status=gate,independent_gate=self.validation['gate'],
            independent_report_verification=verification,source_receipts=[])
        self.prereq=self.patch(m.execution,'_prerequisites',return_value=self.prerequisite)

    def test_prepare_binding_exact_modules_canonical_paths_and_no_GPU(self):
        self.assertEqual(m.audit_protocol(self.protocol_path),self.protocol)
        self.assertEqual(self.protocol['upstream'][0]['pid'],85446)
        self.assertEqual(self.protocol['upstream'][1]['pid'],90001)
        self.no_native.assert_not_called()
        with self.assertRaises(ValueError):m.prepare(self.root/'invalid',None)
        self.active[90001]['argv']=['python','different.py']
        with self.assertRaisesRegex(ValueError,'live exact original'):m.prepare(self.root/'wrong',90001)

    def test_original_process_dies_or_recycles_without_completion_final_blocked(self):
        self.active.pop(90001)
        with self.assertRaisesRegex(RuntimeError,'exited without completed'):m.wait_upstream(self.protocol_path,self.review_path,poll_seconds=.001)
        self.active[90001]=dict(pid=90001,start_ticks=123,argv=['python','-m',m.V_MODULE],cwd=str(self.root))
        with self.assertRaisesRegex(ValueError,'changed/recycled'):m.process_status(self.protocol['upstream'][1])

    def test_duplicate_module_cannot_start_second_final_continuation(self):
        self.active[99999]={**self.active[90001],'pid':99999}
        with self.assertRaisesRegex(ValueError,'Duplicate'):m.process_status(self.protocol['upstream'][1])

    def test_completed_originals_can_exit_failure_and_UNKNOWN_remain(self):
        self.completed_upstream('INCOMPLETE');self.active.clear()
        m.wait_upstream(self.protocol_path,self.review_path,poll_seconds=.001)
        result=m.audit_upstream(self.protocol)
        self.assertEqual(result['independent_gate_status'],'INCOMPLETE')
        self.prereq.assert_called_once();self.no_native.assert_not_called()

    def test_partial_actual_V4_or_changed_original_status_cannot_launch_final(self):
        self.completed_upstream();path=m._paths()['validation_completed']
        self.validation['native_known']=4799;path.unlink();m.save(path,self.validation)
        with self.assertRaisesRegex(ValueError,'Complete fixed independent'):m.audit_upstream(self.protocol)
        self.validation['native_known']=4800;self.validation['independent_gate_status']='PASS';path.unlink();m.save(path,self.validation)
        with self.assertRaisesRegex(ValueError,'changed original gate'):m.audit_upstream(self.protocol)

    def test_root_review_or_source_change_rejected_before_seal(self):
        review=m.read(self.review_path);review['statistical_significance_claim']=True
        self.review_path.unlink();m.save(self.review_path,review)
        with self.assertRaisesRegex(ValueError,'root source-bound'):m.run(self.protocol_path,self.review_path)
        self.assertFalse((self.out/'started.json').exists());self.no_native.assert_not_called()

    def seal(self,plan,report,training,out):
        out.mkdir();m.save(out/'seal.json',dict(cpu_fixture=True))

    def execute(self,sealed,out,diagnostic):
        self.assertTrue(diagnostic);out.mkdir();folder=out/'report';folder.mkdir()
        report=dict(requested=9600,known=9600,independent_blocks=1,statistical_significance_claim=False,
            original_capacity_qualification='UNKNOWN_NOT_ESTABLISHED_BY_THIS_REPORT',
            descriptive_comparison={'QTLPME':dict(Q=.1,T=.2,L=.3,P=.4,M=.5,E=.6),
                'strong_N_baseline':-1.,'qualified_envelope':'UNKNOWN'},
            actual_audits=[],invalid_attempts=[],primary_value=-.1)
        m.save(folder/'report.json',report)
        result=dict(requested=9600,known=9600,report=m.receipt(folder/'report.json'),not_for_training=True,
            final_gain_eligible=False,status='DESCRIPTIVE_COMPLETE',execution_error=None)
        m.save(out/'completed.json',result);return result

    def test_one_seal_and_run_exact_diagnostic_no_training_or_validation_then_full_completion(self):
        self.completed_upstream('FAIL')
        with patch.object(m.execution,'seal',side_effect=self.seal) as seal,\
                patch.object(m.execution,'run',side_effect=self.execute) as run:
            result=m.run(self.protocol_path,self.review_path)
            seal.assert_called_once();run.assert_called_once()
            with self.assertRaises(FileExistsError):m.run(self.protocol_path,self.review_path)
        self.assertEqual(result['independent_gate_status'],'FAIL')
        self.assertFalse(result['final_gain_eligible']);self.assertEqual(result['independent_blocks'],1)
        self.assertTrue(result['no_training_launched']);self.assertTrue(result['no_independent_launched'])
        self.assertEqual(len(result['descriptive_comparison']['QTLPME']),6);self.no_native.assert_not_called()

    def test_seal_failure_has_permanent_claim_no_run_no_retry(self):
        self.completed_upstream()
        with patch.object(m.execution,'seal',side_effect=ValueError('bounded drain precheck failed')),\
                patch.object(m.execution,'run') as run:
            with self.assertRaisesRegex(ValueError,'bounded drain'):m.run(self.protocol_path,self.review_path)
            run.assert_not_called()
        self.assertTrue((Path(self.protocol['plan']['path']).parent/'F8-continuation-attempt.json').exists())
        self.assertFalse((self.out/'completed.json').exists())
        with self.assertRaises(FileExistsError):m.run(self.protocol_path,self.review_path)

    def test_upstream_file_mutation_after_seal_prevents_actual_final_execution(self):
        self.completed_upstream()
        def seal(*args):
            self.seal(*args)
            m._paths()['validation_completed'].write_text('{"changed":true}')
        with patch.object(m.execution,'seal',side_effect=seal),patch.object(m.execution,'run') as run:
            with self.assertRaises(ValueError):m.run(self.protocol_path,self.review_path)
            run.assert_not_called()

    def test_final_qualification_or_significance_upgrade_rejected(self):
        self.completed_upstream()
        def execute(*args,**kwargs):
            result=self.execute(*args,**kwargs);p=Path(result['report']['path']);report=m.read(p)
            report['statistical_significance_claim']=True;p.unlink();m.save(p,report)
            result['report']=m.receipt(p);return result
        with patch.object(m.execution,'seal',side_effect=self.seal),patch.object(m.execution,'run',side_effect=execute):
            with self.assertRaisesRegex(ValueError,'preserve original finite scope'):m.run(self.protocol_path,self.review_path)
        self.assertFalse((self.out/'completed.json').exists())

    def test_live_same_module_with_other_protocol_or_cwd_is_rejected(self):
        self.active[90001]['argv'][-1]='/foreign/root-review.json'
        with self.assertRaisesRegex(ValueError,'live exact original'):m.prepare(self.root/'foreign-review',90001)
        self.active[90001]['argv'][-1]=str(m._paths()['validation_protocol'].with_name('root-review.json'))
        self.active[90001]['cwd']='/foreign'
        with self.assertRaisesRegex(ValueError,'live exact original'):m.prepare(self.root/'foreign-cwd',90001)

    def test_V4_completed_needs_its_own_source_bound_root_authorization(self):
        self.completed_upstream()
        with patch.object(self.v,'audit_root_review',side_effect=ValueError('V4 root review rejected')):
            with self.assertRaisesRegex(ValueError,'V4 root review'):m.audit_upstream(self.protocol)


if __name__=='__main__':unittest.main()
