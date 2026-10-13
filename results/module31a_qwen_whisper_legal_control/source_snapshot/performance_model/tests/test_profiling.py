import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from performance_model.profiling.plan import make_plan
from performance_model.profiling.run import assess
from performance_model.profiling.analyze import execution_coverage,measured


class PlanTests(unittest.TestCase):
    def test_small_feature_plan_invariants_and_no_enumeration(self):
        with tempfile.TemporaryDirectory() as folder,patch('deployment_core.candidate_generator.enumerate_deployments',side_effect=AssertionError('enumeration forbidden')):
            plan=make_plan(folder)
        self.assertEqual(plan['unique_configurations'],14)
        self.assertEqual(plan['maximum_repetitions'],42)
        self.assertEqual(len(plan['contender_selection']['selected']),3)
        groups={f:[c for c in plan['configs'] if c['experiment_family']==f] for f in ('load_axis','task_mix_axis','colocation_axis')}
        a=groups['load_axis']; self.assertEqual(len(a),7)
        self.assertTrue(all(c['deployment']['qwen']==1 and sum(c['deployment'].values())==1 for c in a))
        for i,c in enumerate(a,1):
            self.assertAlmostEqual(c['expected_u'],i/10)
            for k,v in c['expected_task_mix'].items(): self.assertAlmostEqual(v,a[0]['expected_task_mix'][k])
        for c in groups['task_mix_axis']+groups['colocation_axis']:
            self.assertAlmostEqual(c['expected_u'],.4)
        cgroup=groups['colocation_axis']
        self.assertTrue(all(c['deployment']==cgroup[0]['deployment'] for c in cgroup))
        self.assertTrue(all(c['expected_task_mix']==cgroup[0]['expected_task_mix'] for c in cgroup))
        for c in plan['configs']:
            self.assertEqual(len(c['runs']),3)
            for r in c['runs']: self.assertGreaterEqual(min(r['expected_task_counts'].values()),30)
            for task,rate in c['workload'].items(): self.assertAlmostEqual(sum(c['routing'][task].values()),rate)

    def test_incomplete_error_and_drain_stop(self):
        policy={'completion_ratio_threshold':.98,'drain_overload_seconds':30}
        data=dict(request_records=[dict(task='text',model='qwen',status='incomplete',scheduled_time_ns=0)],status='failed',errors=['worker error'],
            system_metrics={'completion_ratio':0},measurement_window={'start_time_ns':0,'end_time_ns':10000000000,'drain_seconds':60})
        result=assess(data,policy)
        self.assertTrue(result['overloaded_or_unstable'])
        self.assertIn('incomplete',result['reasons'])
        self.assertIn('drain_exceeds_threshold',result['reasons'])

    def test_deferred_anchor_is_required_after_load_stop(self):
        plan={'configs':[{'name':name,'runs':[{'run_id':f'{name}_{rep}'} for rep in range(3)]}
                         for name in ('A_u_0.30','A_u_0.40','A_u_0.50','B_text','C_control')]}
        state={'status':'measurement_complete','stopped_load_axis_at':.3,
               'skipped_configurations':['A_u_0.40','A_u_0.50']}
        runs=[r for c in plan['configs'] if c['name']!='A_u_0.50' for r in c['runs']]
        coverage=execution_coverage(plan,runs,state)
        self.assertTrue(coverage['complete'])
        self.assertEqual(coverage['expected_runs_after_adaptive_stop'],12)
        self.assertEqual(coverage['skipped_configurations'],['A_u_0.50'])
        without_anchor=[r for r in runs if not r['run_id'].startswith('A_u_0.40')]
        self.assertFalse(execution_coverage(plan,without_anchor,state)['complete'])

    def test_censored_requests_remain_in_counts_and_latency_uses_successes(self):
        records=[{'status':'completed','completion_time_ns':10,'e2e_latency_ms':7,
                  'runtime_latency_ms':5,'schedule_delay_ms':2,'worker_latency_ms':3,'worker_queue_wait_ms':1},
                 {'status':'incomplete','completion_time_ns':None,'schedule_delay_ms':900}]
        result=measured(records,10,9)
        self.assertEqual(result['offered'],2)
        self.assertEqual(result['incomplete'],1)
        self.assertEqual(result['mean_e2e_latency_ms'],7)
        self.assertEqual(result['mean_schedule_delay_ms'],2)
        self.assertEqual(result['throughput_rps'],.1)
        self.assertEqual(result['within_window_throughput_rps'],0)

if __name__=='__main__': unittest.main()
