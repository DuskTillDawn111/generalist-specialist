import copy
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027.joint_host_scenarios_v13 import (
    SCHEMA, complete_members, kernel_prefix, natural_prefix, public_workload, select_scenarios)
from tools.predictor_validation_20261027.profiling import digest


class JointHostScenarioTests(unittest.TestCase):
    def setUp(self):
        self.job = dict(id='j', inputs={'a':'x','b':'y'}, tasks={'a':'text','b':'text'},
                        start_offsets_ms={'a':0,'b':2}, split='training')
        self.rows = [dict(model=p, task='text', pool_id=pool, request_id=p,
            scheduled_time_ns=100+off, actual_submit_time_ns=101+off,
            profile_job_id='j', split='training', status='completed')
            for p,pool,off in [('a','x',0),('b','y',2_000_000)]]

    def test_complete_joint_members_keep_public_offsets(self):
        workload, rows, origin = complete_members(self.job,self.rows)
        self.assertEqual(origin,100)
        self.assertEqual([r['arrival_ns'] for r in workload],[0,2_000_000])
        self.assertEqual(len(rows),2)

    def test_missing_duplicate_changed_member_and_shifted_clock_rejected(self):
        alternatives = [self.rows[:1], [self.rows[0],self.rows[0]]]
        changed = copy.deepcopy(self.rows); changed[1]['pool_id']='foreign'; alternatives.append(changed)
        changed = copy.deepcopy(self.rows); changed[1]['scheduled_time_ns']+=1; alternatives.append(changed)
        for rows in alternatives:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                complete_members(self.job,rows)

    def test_validation_member_cannot_enter_training_bank(self):
        rows = copy.deepcopy(self.rows); rows[0]['split']='validation'
        with self.assertRaises(ValueError): complete_members(self.job,rows)

    def test_natural_prefix_retains_cross_owner_order_and_blocking_wait(self):
        row = dict(self.rows[0], worker_response=dict(start_perf_ns=120,
            semantic_forward_observations={'records':[dict(start_ns=180,semantic_phase='prefill')]},
            observed_stages=[dict(id='pre_sync',start_ns=130,end_ns=170)]))
        result = natural_prefix(row,100)
        self.assertEqual(result['first_semantic_start_offset_ns'],80)
        self.assertEqual(result['pre_sync_end_offset_ns']-result['pre_sync_start_offset_ns'],40)
        self.assertFalse(result['synchronization_wait_removed'])
        self.assertFalse(result['first_semantic_is_first_CUDA_release'])

    def test_selection_keeps_whole_scenarios_and_domains_separate(self):
        workload = public_workload(self.job)
        common = dict(public_workload=workload,workload_sha256=digest(workload))
        first = dict(common,domain='natural',prefixes=[{'alias':'b'},{'alias':'a'}])
        second = dict(common,domain='natural',prefixes=[{'alias':'a'},{'alias':'b'}])
        light = dict(common,domain='unified-light-kernel',prefixes=[])
        bank = dict(schema=SCHEMA,training_only=True,scenarios=[first,second,light])
        self.assertEqual(select_scenarios(bank,'natural',workload),[first,second])
        other = copy.deepcopy(workload); other[0]['arrival_ns']+=1
        with self.assertRaises(UnknownEvidence): select_scenarios(bank,'natural',other)

    def test_missing_OCR_sync_and_semantic_are_retained_as_unknown(self):
        row = dict(self.rows[0],worker_response=dict(start_perf_ns=120,observed_stages=[]))
        result = natural_prefix(row,100)
        self.assertIsNone(result['first_semantic_start_offset_ns'])
        self.assertEqual(result['pre_sync_missing_reason'],'NO_EXPLICIT_PRE_SYNC_STAGE')
        self.assertEqual(result['alias'],'a')

    def test_multistream_launch_and_activity_order_are_distinguished(self):
        apis = [dict(correlationId=1,start=1,end=2),dict(correlationId=2,start=3,end=4)]
        result,first = kernel_prefix(apis,{(1,100,110),(2,10,20)})
        self.assertEqual(result['first_kernel_launch_API_entry_ns'],1)
        self.assertEqual(result['first_kernel_launch_activity_start_ns'],100)
        self.assertEqual(result['first_kernel_activity_API_entry_ns'],3)
        self.assertEqual(result['first_kernel_activity_start_ns'],10)
        self.assertFalse(result['launch_and_activity_first_are_same_kernel'])

    def test_missing_or_ambiguous_first_API_cannot_select_a_later_kernel(self):
        api = dict(correlationId=2,start=3,end=4)
        for apis in ([api],[api,dict(api)]):
            with self.assertRaises(ValueError): kernel_prefix(apis,{(1,1,2),(2,10,20)})


if __name__ == '__main__': unittest.main()
