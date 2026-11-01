"""CPU failure injection for finite continuation; collector is always mocked."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import admission_training_continuation_v1 as m


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.base=self.root/'artifacts';self.base.mkdir();self.out=self.root/'continuation'
        original_base=m.BASE
        self.addCleanup(patch.stopall)
        patch.object(m,'BASE',self.base).start()
        self.campaign_dir=self.base/'admission-training-v1'
        self.campaign=m.training.prepare(self.campaign_dir)
        for key,folder in m.PROTOCOL_NAMES.items():
            target=self.base/folder;target.mkdir()
            m.MODULES[key].prepare(self.campaign_dir/'campaign.json',target/'protocol.json')
        m.save(self.base/'admission-next-stage-roles-v1.json',m.read(original_base/'admission-next-stage-roles-v1.json'))
        self.active={}
        for name,pid,folder,script in m.UPSTREAM:
            target=self.base/folder;target.mkdir();entry=target/script;entry.write_text('# CPU fixture upstream\n')
            self.active[pid]=dict(pid=pid,start_ticks=pid,argv=['python','-u',str(entry)],cwd=str(self.root))
        patch.object(m,'process_identity',side_effect=lambda pid:copy.deepcopy(self.active.get(pid))).start()
        patch.object(m,'matching_processes',side_effect=lambda s:[copy.deepcopy(p) for p in self.active.values() if m._runs_script(p,s)]).start()
        self.no_gpu=patch.object(m.training.native_flow,'execute',side_effect=AssertionError('No real collector')).start()
        self.protocol=m.prepare(self.out);self.protocol_path=self.out/'protocol.json'
        self.review_path=self.out/'root-review.json'
        m.save(self.review_path,dict(schema=m.SCHEMA+'.main_thread_review',
            decision='APPROVED_FOR_CONDITIONAL_TRAINING_ONLY',reviewer_role='main_thread',
            protocol=m.receipt(self.protocol_path),created_ns=self.protocol['created_ns']+1,
            **{k:self.protocol[k] for k in ('campaign','capacity_protocol','resource_protocol','quality_protocol','role_reservation','source')},
            stage3_incomplete_training_reason='Finite training only; keep failed gate and every activation insufficiency.',final_gain_eligible=False))

    def complete_upstream(self):
        left=self.base/'flow-validation-h600-selected-v16';right=self.base/'h600-admission-pilot-v1'
        (left/'report').mkdir();(right/'report').mkdir();(right/'sealed').mkdir();(right/'runs').mkdir()
        self.report2=dict(candidate_coverage={'requested':4,'observed':4},
            candidates=[dict(candidate_id='shared_'+r,known=1200) for r in ('balanced','skewed','spill_low','spill_high')])
        m.save(left/'report/report.json',self.report2)
        self.gate=dict(status='FAIL',report=m.receipt(left/'report/report.json'),failed_checks=['queue'],unknown_checks=[])
        m.save(left/'gate.json',self.gate)
        self.report3=dict(schema=m.pilot.SCHEMA+'.report',status='DIAGNOSTIC_INCOMPLETE',created_ns=10,
            complete_three_mode_mechanism_conclusion='INCOMPLETE_OR_INSUFFICIENT_ACTIVATION',
            candidates=[dict(candidate_id=cid,complete=False,evidence_error='CPU fixture missing support',
                activation=None,original_stability=None) for cid in m.pilot.IDS])
        m.save(right/'report/report.json',self.report3);m.save(right/'sealed/predictions.json',{'cpu_fixture':True})
        m.save(right/'runs/run-state.json',dict(candidate_order=list(m.pilot.IDS),status='INCOMPLETE_DECLARED_MODES_SKIPPED',
            seal=m.receipt(right/'sealed/predictions.json'),entry_source=m.receipt(m.pilot.__file__),
            final_gain_eligible=False,completed=[]))
        m.save(left/'completed.json',dict(source=self.protocol['upstream'][0]['script'],
            report=m.receipt(left/'report/report.json'),gate=m.receipt(left/'gate.json'),final_gain_eligible=False))
        m.save(right/'completed.json',dict(source=self.protocol['upstream'][1]['script'],
            seal=m.receipt(right/'sealed/predictions.json'),run_state=m.receipt(right/'runs/run-state.json'),
            report=m.receipt(right/'report/report.json'),final_gain_eligible=False))
        self.selected_gate=patch.object(m.selected,'gate',return_value=copy.deepcopy(self.gate)).start()
        patch.object(m.pilot,'check_sealed',return_value={}).start()
        self.pilot_report=patch.object(m.pilot,'report',return_value={**copy.deepcopy(self.report3),'created_ns':11}).start()

    def test_prepare_audits_three_protocols_original_pids_and_never_gpu(self):
        self.assertEqual(m.audit_protocol(self.protocol_path),self.protocol)
        self.assertEqual([s['pid'] for s in self.protocol['upstream']],[79375,79980])
        self.no_gpu.assert_not_called()
        self.assertTrue(m._runs_script(dict(argv=['python','a/script.py'],cwd='/tmp'),'/tmp/a/script.py'))

    def test_upstream_died_missing_completed_never_replaced_or_waited_forever(self):
        self.active.pop(79375)
        with self.assertRaisesRegex(RuntimeError,'exited without completed'):m.wait_upstream(self.protocol_path,poll_seconds=.001)
        self.no_gpu.assert_not_called()

    def test_recycled_or_duplicate_upstream_pid_rejected(self):
        self.active[79375]['start_ticks']+=1
        with self.assertRaisesRegex(ValueError,'recycled'):m.upstream_status(self.protocol['upstream'][0])
        self.active[79375]['start_ticks']-=1
        self.active[99999]={**self.active[79375],'pid':99999}
        with self.assertRaisesRegex(ValueError,'Duplicate'):m.upstream_status(self.protocol['upstream'][0])

    def test_completed_dependencies_can_exit_and_failure_is_preserved(self):
        self.complete_upstream();self.active.clear()
        m.wait_upstream(self.protocol_path,poll_seconds=.001)
        result=m.audit_upstream(self.protocol)
        self.assertEqual(result['stage2_gate_status'],'FAIL');self.assertEqual(result['stage3_status'],'DIAGNOSTIC_INCOMPLETE')
        self.assertEqual(len(result['insufficiencies']),3)
        self.selected_gate.assert_called_once();self.pilot_report.assert_called_once();self.no_gpu.assert_not_called()

    def test_recomputed_gate_or_pilot_cannot_be_laundered(self):
        self.complete_upstream();self.selected_gate.return_value={**self.gate,'status':'PASS'}
        with self.assertRaisesRegex(ValueError,'gate differs'):m.audit_upstream(self.protocol)
        self.selected_gate.return_value=self.gate
        self.pilot_report.return_value={**self.report3,'status':'DIAGNOSTIC_COMPLETE'}
        with self.assertRaisesRegex(ValueError,'Stage3 report differs'):m.audit_upstream(self.protocol)

    def test_source_review_and_mode_protocol_bindings_fail_closed(self):
        review=m.read(self.review_path);review['quality_protocol']=review['capacity_protocol']
        self.review_path.unlink();m.save(self.review_path,review)
        with self.assertRaisesRegex(ValueError,'Root source-bound'):m.audit_root_review(self.protocol_path,self.review_path)
        source=Path(self.protocol['upstream'][0]['script']['path']);source.write_text('# changed\n')
        with self.assertRaises(ValueError):m.audit_protocol(self.protocol_path)

    def test_transitive_json_closure_catches_changes_beyond_direct_report(self):
        leaf=self.root/'leaf.json';middle=self.root/'middle.json';m.save(leaf,dict(value=1))
        m.save(middle,dict(nested=m.receipt(leaf)))
        closure=m.json_source_closure([m.receipt(middle)])
        self.assertEqual(len(closure),2)
        leaf.unlink();m.save(leaf,dict(value=2))
        with self.assertRaises(ValueError):m.capacity.unique_refs(closure)

    def fake_execute(self,campaign,job_id,review):
        job=next(j for j in self.campaign['jobs'] if j['job_id']==job_id)
        folder=Path(job['output']);folder.mkdir(parents=True,exist_ok=False)
        m.save(folder/'training-complete.json',dict(job=job,campaign=m.receipt(campaign),cpu_fixture=True))

    def test_eight_jobs_fixed_order_upstream_once_fits_then_completion_no_rerun(self):
        self.complete_upstream();executed=[]
        def execute(*args):executed.append(args[1]);return self.fake_execute(*args)
        with patch.object(m.training,'execute',side_effect=execute),patch.object(m,'_fits',return_value={}) as fits:
            result=m.run(self.protocol_path,self.review_path)
            fits.assert_called_once()
            with self.assertRaises(FileExistsError):m.run(self.protocol_path,self.review_path)
        self.assertEqual(executed,[j['job_id'] for j in self.campaign['jobs']])
        self.assertEqual(result['job_count'],8);self.assertFalse(result['service_refitted'])
        self.assertEqual(result['upstream_statuses']['stage2_gate_status'],'FAIL')
        self.selected_gate.assert_called_once();self.pilot_report.assert_called_once();self.no_gpu.assert_not_called()

    def test_failed_attempt_stops_no_next_job_no_fit_no_completed(self):
        self.complete_upstream();count=[]
        def execute(*args):
            count.append(args[1])
            if len(count)==2:raise RuntimeError('native fixture failed')
            self.fake_execute(*args)
        with patch.object(m.training,'execute',side_effect=execute),patch.object(m,'_fits') as fits:
            with self.assertRaisesRegex(RuntimeError,'native fixture failed'):m.run(self.protocol_path,self.review_path)
            fits.assert_not_called()
        self.assertEqual(len(count),2);self.assertFalse((self.out/'completed.json').exists())
        self.assertTrue((self.out/'started.json').exists());self.no_gpu.assert_not_called()

    def test_three_fits_cpu_only_verify_each_and_restore_environment(self):
        import os
        calls=[]
        def fit(path,folder):
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'')
            folder.mkdir();m.save(folder/'manifest.json',dict(cpu_fixture=True));calls.append(path)
        original=os.environ.get('CUDA_VISIBLE_DEVICES')
        with patch.object(m.capacity,'fit',side_effect=fit),patch.object(m.resources,'fit',side_effect=fit),\
                patch.object(m.quality,'fit',side_effect=fit),patch.object(m.capacity,'verify_manifest') as a,\
                patch.object(m.resources,'verify_manifest') as b,patch.object(m.quality,'verify_manifest') as c:
            result=m._fits(self.protocol,self.out)
        self.assertEqual(len(calls),3);self.assertEqual(len(result),3)
        a.assert_called_once();b.assert_called_once();c.assert_called_once()
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'),original)

    def test_materialized_training_review_drift_stops_before_next_job(self):
        self.complete_upstream();executed=[]
        def execute(*args):
            executed.append(args[1]);self.fake_execute(*args)
            review_path=Path(args[2]);review=m.read(review_path)
            review['root_authorization']={'path':'/changed','sha256':'changed'}
            review_path.unlink();m.save(review_path,review)
        with patch.object(m.training,'execute',side_effect=execute),patch.object(m,'_fits') as fits:
            with self.assertRaises(ValueError):m.run(self.protocol_path,self.review_path)
            fits.assert_not_called()
        self.assertEqual(len(executed),1);self.assertFalse((self.out/'completed.json').exists())


if __name__=='__main__':unittest.main()
