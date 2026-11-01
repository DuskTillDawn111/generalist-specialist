import copy
import json
from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.predictor_validation_20261027 import twoFIFO_prelaunch_recovery_v1 as m


def write(path,doc):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(doc))


class ProofTests(unittest.TestCase):
    def test_actual_exact_boundary_is_prefix4_unlaunched8(self):
        p=m.proof(m.ORIGINAL_MANIFEST,m.ORIGINAL_REPORT);m.no_started_tail(p)
        self.assertEqual(len(p['prefix']),4);self.assertEqual(len(p['tail']),8)
        self.assertEqual(p['stopped_report']['sha256'],m.STOPPED_SHA)
        self.assertEqual(p['prelaunch_claim']['sha256'],m.FIFTH_CLAIM_SHA)
    def test_other_report_or_manifest_not_recoverable(self):
        for p,r in [('/other/manifest.json',m.ORIGINAL_REPORT),(m.ORIGINAL_MANIFEST,'/other/report.json')]:
            with self.assertRaises(ValueError):m.proof(p,r)
    def fixture(self,root):
        tail=[]
        for i in range(8):
            p=root/str(i)/'plan.json';write(p,{})
            tail.append(dict(plan=m.receipt(p),attempt_id='a'+str(i),execution_id='e'+str(i),
                claim=str(p.parent/'old-claim.json'),collection=str(p.parent/'collection')))
        write(Path(tail[0]['claim']),{});return dict(tail=tail,prelaunch_claim=m.receipt(tail[0]['claim']))
    def test_started_collection_oldlaterclaim_and_recovery_claim_all_block(self):
        for mode in ['collection','later-claim','recovery-claim','envelope','oldclaim-drift']:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                p=self.fixture(Path(d));item=p['tail'][0]
                if mode=='collection':Path(item['collection']).mkdir()
                if mode=='later-claim':write(Path(p['tail'][1]['claim']),{})
                if mode=='recovery-claim':write(Path(item['plan']['path']).parent/m.RECOVERY_CLAIM,{})
                if mode=='envelope':write(Path(item['plan']['path']).parent/m.RECOVERY_ENVELOPE,{})
                if mode=='oldclaim-drift':write(Path(item['claim']),{'changed':True})
                with self.assertRaises(ValueError):m.no_started_tail(p)
    def test_canonical_new_policy_authorization_is_exclusive(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/m.RECOVERY_CLAIM;m.write(p,{'manifest':'one'})
            with self.assertRaises(FileExistsError):m.write(p,{'manifest':'two'})
            self.assertEqual(json.loads(p.read_text()),{'manifest':'one'})


class RunTests(unittest.TestCase):
    def fixture(self,root):
        original_path=root/'original.json';write(original_path,{})
        campaign=root/'campaign.json';write(campaign,{})
        prefix=[dict(attempt_id='prefix'+str(i)) for i in range(4)]
        tail=[]
        for i in range(8):
            p=root/('tail'+str(i))/'plan.json';write(p,{})
            tail.append(dict(attempt_id='tail'+str(i),execution_id='id'+str(i),plan=m.receipt(p),collection=str(p.parent/'collection'),claim=str(p.parent/'old-claim.json')))
        proof=dict(prefix=prefix,tail=tail,original_manifest=m.receipt(original_path),stopped_report={'path':'stopped','sha256':'s'})
        doc=dict(proof=proof,run_output=str(root/'run'),sources=[],recovery_claims=str(root/'auth'))
        path=root/'manifest.json';write(path,doc)
        original=dict(campaign=m.receipt(campaign),attempts=prefix+tail)
        return path,doc,original
    def patches(self,doc,original):
        return [patch.object(m,'audit',return_value=doc),patch.object(m.old,'audit',return_value=original),
                patch.object(m,'no_started_tail'),patch.object(m.old,'SourceChangeGuard',return_value=unittest.mock.MagicMock())]
    def test_default_off_does_not_launch_or_recheck(self):
        with tempfile.TemporaryDirectory() as d:
            p,doc,o=self.fixture(Path(d));a,b,c,e=self.patches(doc,o)
            with a,b,c,e,patch.object(m,'launch') as launch,patch.object(m,'recheck') as recheck:
                out=m.run(p)
            self.assertEqual(out['status'],'READY_NEW_POLICY_EXPLICIT_EXECUTE_REQUIRED');launch.assert_not_called();recheck.assert_not_called()
    def test_recheck_then_eight_review_power_macro_in_order(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p,doc,o=self.fixture(root);calls=[];a,b,c,e=self.patches(doc,o)
            def prefix(row,path,directory):calls.append('prefix')
            def launch(path,item,index,**kwargs):
                calls.append('launch');out=Path(item['collection']);out.mkdir();write(out/'execution.json',{});return {},{'path':'envelope','sha256':'e'}
            def review(raw,out,execution,**kwargs):
                calls.append('review');r=Path(out)/'report.json';raw=Path(raw);write(raw,{})
                write(r,dict(raw=m.receipt(raw),issues=[],training_feature_extraction_eligible=True,status='PASS_COMPLETE_SOURCE_REBOUND_NATIVE_TRAINING_NOT_PREDICTOR_QUALIFICATION'));return m.receipt(r)
            def power(raw):calls.append('power');return dict(startup=dict(energy_joules=1),cases=[dict(measurement_energy_joules=1)])
            def macro(path):calls.append('macro');return {}
            with a,b,c,e,patch.object(m,'recheck',side_effect=prefix),patch.object(m,'launch',side_effect=launch),patch.object(m.old,'completed_execution'),patch.object(m.native,'review',side_effect=review),patch.object(m,'observed',side_effect=power),patch.object(m.macros,'extract',side_effect=macro):
                out=m.run(p,execute=True)
            self.assertEqual(out['status'],m.FINAL);self.assertEqual(calls,['prefix']*4+['launch','review','power','macro']*8)
            self.assertEqual(len(out['completed_attempts']),12);self.assertFalse(out['automatic_fit'])
            self.assertEqual(out['prefix_attempts'],doc['proof']['prefix'])
    def test_prefix_failure_stops_before_any_tail(self):
        with tempfile.TemporaryDirectory() as d:
            p,doc,o=self.fixture(Path(d));a,b,c,e=self.patches(doc,o)
            with a,b,c,e,patch.object(m,'recheck',side_effect=ValueError('changed prefix')),patch.object(m,'launch') as launch:
                out=m.run(p,execute=True)
            self.assertEqual(out['status'],'STOPPED_NEW_POLICY_NO_RETRY');launch.assert_not_called()
    def test_failed_execution_stops_before_review_and_later_tail(self):
        with tempfile.TemporaryDirectory() as d:
            p,doc,o=self.fixture(Path(d));a,b,c,e=self.patches(doc,o)
            with a,b,c,e,patch.object(m,'recheck'),patch.object(m,'launch',return_value=({},{})) as launch,patch.object(m.old,'completed_execution',side_effect=ValueError('failed')),patch.object(m.native,'review') as review:
                out=m.run(p,execute=True)
            self.assertEqual(out['status'],'STOPPED_NEW_POLICY_NO_RETRY');self.assertEqual(launch.call_count,1);review.assert_not_called()
    def test_old_child_and_supervisor_frozen_sources_unchanged(self):
        self.assertEqual(m.receipt(m.old.__file__)['sha256'],m.ORIGINAL_EXECUTOR_SHA)
        self.assertEqual(m.receipt(m.old.supervisor.__file__)['sha256'],m.ORIGINAL_SUPERVISOR_SHA)


if __name__=='__main__':unittest.main()

class LaunchTests(unittest.TestCase):
    def fixture(self,root,index=0):
        p=root/'attempt'/'plan.json';write(p,{})
        item=dict(plan=m.receipt(p),attempt_id='a',execution_id='e',claim=str(p.parent/'oldclaim.json'),collection=str(p.parent/'collection'))
        mr=write(root/'original.json',{}) if False else None
        write(root/'original.json',{});mr=m.receipt(root/'original.json')
        if index==0:write(Path(item['claim']),m.expected_claim(mr,item))
        proven=dict(original_manifest=mr,tail=[item],prelaunch_claim=m.receipt(item['claim']) if index==0 else None)
        doc=dict(proof=proven,recovery_claims=[]);mp=root/'manifest.json';write(mp,doc)
        return mp,doc,item
    def test_same_frozen_child_budget_and_cross_manifest_canonical_claim(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);mp,doc,item=self.fixture(root);original=dict(attempts=[{}]*4+[item]);before=Path(item['claim']).read_bytes();guard=unittest.mock.MagicMock();seen=[]
            def supervise(command,output,**kwargs):
                seen.append((command,kwargs));kwargs['before_launch']();Path(output).mkdir();write(Path(output)/'execution.json',{});kwargs['after_run']();return {}
            with patch.object(m,'audit',return_value=doc),patch.object(m.old,'audit',return_value=original),patch.object(m.old,'audit_plan'),patch.object(m.old.supervisor,'public_environment_gate'),patch.object(m.shutil,'disk_usage',return_value=type('D',(),dict(free=m.old.supervisor.PREFLIGHT))()),patch.object(m.old.supervisor,'supervise',side_effect=supervise):
                _,er=m.launch(mp,item,0,guard=guard)
            self.assertEqual(Path(item['claim']).read_bytes(),before)
            argv,kwargs=seen[0];self.assertEqual(argv[2],'tools.predictor_validation_20261027.twoFIFO_source_rebound_execute_v1');self.assertIn('_child',argv)
            self.assertNotIn('budget',kwargs);self.assertNotIn('preflight',kwargs)
            envelope=m.checked(er);self.assertEqual(envelope['policy'],m.POLICY)
            self.assertEqual(envelope['authorization']['path'],str(Path(item['plan']['path']).parent/m.RECOVERY_CLAIM))
            # Even without an existing collection (e.g. a prelaunch exception),
            # a second manifest cannot consume the same canonical authorization.
            other=root/'other-manifest.json';write(other,doc)
            with patch.object(m,'audit',return_value=doc),patch.object(m.old,'audit',return_value=original),patch.object(m.old.supervisor,'supervise') as sup:
                with self.assertRaises(ValueError):m.launch(other,item,0,guard=guard)
            sup.assert_not_called()
    def test_preflight_stops_before_new_authorization_or_old_claim_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);mp,doc,item=self.fixture(root);before=Path(item['claim']).read_bytes()
            with patch.object(m,'audit',return_value=doc),patch.object(m.old,'audit',return_value=dict(attempts=[{}]*4+[item])),patch.object(m.shutil,'disk_usage',return_value=type('D',(),dict(free=0))()),patch.object(m.old.supervisor,'supervise') as sup:
                with self.assertRaises(ValueError):m.launch(mp,item,0,guard=unittest.mock.MagicMock())
            self.assertFalse((Path(item['plan']['path']).parent/m.RECOVERY_CLAIM).exists());self.assertEqual(Path(item['claim']).read_bytes(),before);sup.assert_not_called()
