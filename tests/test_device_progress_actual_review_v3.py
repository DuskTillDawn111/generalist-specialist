"""Retained live ancestry evidence allows nsys session/process-group changes."""
import copy
import unittest

from tools.predictor_validation_20261027.device_progress_actual_review_v3 import valid_live_ownership


def evidence():
    chain=[dict(pid=300,ppid=200,start_ticks=30,process_group=300,session=300,state='S'),
        dict(pid=200,ppid=100,start_ticks=20,process_group=200,session=200,state='S'),
        dict(pid=100,ppid=1,start_ticks=10,process_group=100,session=100,state='R')]
    return dict(pid=300,start_ticks=30,process_group=300,collector_pid=100,collector_start_ticks=10,collector_process_group=100,
        alias='text#0',launcher_pid=200,launcher_start_ticks=20,ancestry=chain,process_group_equality_required=False,
        owner_scope='live_declared_launcher_descendant_with_stable_pid_start_ticks')


class OwnershipEvidenceTests(unittest.TestCase):
    def test_cross_session_and_group_live_ancestry_accepted(self):
        item=evidence();before=copy.deepcopy(item);self.assertTrue(valid_live_ownership(item,100,100));self.assertEqual(item,before)
    def test_old_v1_scope_and_group_only_evidence_rejected(self):
        item=evidence();item['owner_scope']='launched_worker_descendant';self.assertFalse(valid_live_ownership(item,100,100))
        self.assertFalse(valid_live_ownership(dict(pid=300,collector_pid=100,process_group=100),100,100))
    def test_reused_collector_worker_or_launcher_PID_rejected(self):
        for key in ('start_ticks','collector_start_ticks','launcher_start_ticks'):
            item=evidence();item[key]+=1
            with self.subTest(key=key):self.assertFalse(valid_live_ownership(item,100,100))
    def test_broken_chain_zombie_and_duplicate_identity_rejected(self):
        for mutation in ('reparent','zombie','duplicate','missing_collector','missing_launcher'):
            item=evidence()
            if mutation=='reparent':item['ancestry'][0]['ppid']=99
            elif mutation=='zombie':item['ancestry'][1]['state']='Z'
            elif mutation=='duplicate':item['ancestry'].append(copy.deepcopy(item['ancestry'][0]))
            elif mutation=='missing_collector':item['ancestry'].pop()
            else:item['launcher_pid']=99
            with self.subTest(mutation=mutation):self.assertFalse(valid_live_ownership(item,100,100))
    def test_unknown_alias_fabricated_scope_and_malformed_identity_rejected(self):
        for key,value in [('alias','other'),('process_group_equality_required',True),('collector_pid',200),('collector_process_group',200),('pid',False),('ancestry',None)]:
            item=evidence();item[key]=value
            with self.subTest(key=key):self.assertFalse(valid_live_ownership(item,100,100))


if __name__=='__main__':unittest.main()
