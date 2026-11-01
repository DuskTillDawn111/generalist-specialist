"""Endpoint selection and reserved isolation, without GPU or real raw reads."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from tools.queue_refined_20261023 import iteration as m

class Selection(unittest.TestCase):
    def packet(self,rate):
        jobs=[]
        for seed in (203,204):
            for role,strategy in [('calibrated_spill','queue_quality'),('risk_matched_no_spill','queue_specialist_only'),('legacy_latency','queue_latency'),('strong_other_deployment','queue_specialist_only')]:
                c=dict(deployment=m.PRIMARY if role!='strong_other_deployment' else {'text':2,'image':1,'asr':1,'ocr':1},strategy=strategy,text_rate=rate)
                job=dict(seed=seed,role=role,strategy=strategy,policy_contract=c,identity=dict(seed=seed,phase='development',policy_contract_sha256=m.protocol.digest(c)),label=role+'-seed'+str(seed),phase='development')
                job['id']=m.protocol.digest(job['identity']);jobs.append(job)
        return dict(jobs=jobs,seeds=[203,204])
    def report(self,packet,spill,control,other=-.5,eligible=True):
        rows=[]
        for job in packet['jobs'][:4]:
            value=spill if job['role']=='calibrated_spill' else control if job['role']=='risk_matched_no_spill' else other
            rows.append(dict(candidate_id=m.cid(job),candidate=m.symbolic(job),mean_U=value,complete_eligible_all_declared_seeds=eligible))
        rows.sort(key=lambda r:-r['mean_U']);return dict(ranking=rows,winner=rows[0])
    def test_workload_uses_pair_delta_not_crossload_absolute_U(self):
        packets=[self.packet(1.85),self.packet(1.89)]
        reports=[self.report(packets[0],-.9,-1.),self.report(packets[1],-1.1,-1.3)]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);design=m.supplemental_design(root/'design.json')
            result,bounds=m.freeze_selection(packets,reports,root,design)
            self.assertEqual(result['selected']['load'],1.89)
            self.assertAlmostEqual(result['selected']['mean_delta_U'],.2)
            self.assertEqual(len(bounds),2)
            first=json.loads(Path(bounds[0]['path']).read_text());second=json.loads(Path(bounds[1]['path']).read_text())
            self.assertEqual(first['seeds'],[205,206]);self.assertEqual(second['seeds'],[207,208,209])
            jobs=first['jobs']+second['jobs'];self.assertEqual(len(jobs),10)
            self.assertTrue(all(j['role'] in ('calibrated_spill','risk_matched_no_spill') for j in jobs))
            self.assertTrue(all(j['policy_contract']['deployment']==m.PRIMARY for j in jobs))
            self.assertEqual([j['role'] for j in jobs[:4]],['calibrated_spill','risk_matched_no_spill','risk_matched_no_spill','calibrated_spill'])
            self.assertTrue(all(m.protocol.digest(j['identity'])==j['id'] for j in jobs))
    def test_unknown_pair_does_not_trigger_replacement_or_reserved(self):
        packet=self.packet(1.85)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);design=m.supplemental_design(root/'design.json')
            result,bounds=m.freeze_selection([packet],[self.report(packet,-.9,-1.,eligible=False)],root,design)
            self.assertEqual(result['status'],'UNRESOLVED_NO_ELIGIBLE_PRIMARY_PAIR');self.assertEqual(bounds,[])
            self.assertFalse(list(root.glob('*packet.json')))
    def test_tie_anchor_uses_lower_load_deterministically(self):
        packets=[self.packet(1.89),self.packet(1.85)]
        reports=[self.report(p,-.9,-1.) for p in packets]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);design=m.supplemental_design(root/'design.json')
            selected,_=m.freeze_selection(packets,reports,root,design)
            self.assertEqual(selected['selected']['load'],1.85)
if __name__=='__main__':unittest.main()
