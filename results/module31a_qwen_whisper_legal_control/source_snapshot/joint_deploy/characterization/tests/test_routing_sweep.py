import copy
import unittest
from collections import Counter

from joint_deploy.characterization.routing_sweep import (
    DEPLOYMENT, WORKLOAD, SHARES, SEEDS, make_config, audit, stats,
)
from joint_deploy.measurement.load_generator import periodic_arrivals
from joint_deploy.measurement.router import Router


def fixture(share=0.5, seed=101):
    cfg = make_config(share, seed).as_dict()
    router = Router(cfg['routing_rps'])
    records = []
    for i, (offset, task) in enumerate(periodic_arrivals(WORKLOAD, 100, seed)):
        when = 1000000000 + round(offset * 1e9)
        records.append(dict(request_id=str(i), task=task, model=router.choose(task),
                            scheduled_time_ns=when, actual_submit_time_ns=when+100,
                            completion_time_ns=when+1000, status='completed'))
    counts = Counter(r['task'] for r in records)
    return cfg, {**copy.deepcopy(cfg), 'status':'completed', 'errors':[],
                 'measurement_window':{'start_time_ns':1000000000}, 'request_records':records,
                 'per_task_metrics':{task:dict(offered_requests=n, completed_requests=n) for task,n in counts.items()},
                 'shutdown':{m:dict(exit_code=0, bye_received=True, forced=False) for m in DEPLOYMENT}}


class CharacterizationTests(unittest.TestCase):
    def test_fifteen_configs_fixed_x_lambda_and_request_minimum(self):
        for share in SHARES:
            for seed in SEEDS:
                cfg = make_config(share, seed)
                self.assertEqual(cfg.deployment, DEPLOYMENT)
                self.assertEqual(cfg.workload_rps, WORKLOAD)
                self.assertEqual(cfg.measurement_seconds, 100)
                self.assertEqual(set(cfg.routing_rps), set(WORKLOAD))
                for task, route in cfg.routing_rps.items():
                    self.assertEqual(set(route), set(DEPLOYMENT))
                    self.assertAlmostEqual(sum(route.values()), WORKLOAD[task])
                counts=Counter(t for _,t in periodic_arrivals(WORKLOAD,100,seed))
                self.assertEqual(min(counts.values()), 30)
                self.assertEqual(sum(counts.values()), 580)

    def test_same_seed_reuses_arrivals_different_repetitions_change_them(self):
        events = [list(periodic_arrivals(WORKLOAD, 100, seed)) for seed in SEEDS]
        self.assertNotEqual(events[0], events[1])
        self.assertNotEqual(events[1], events[2])

    def test_actual_router_rounding_all_settings(self):
        for share in SHARES:
            for seed in SEEDS:
                cfg, data=fixture(share,seed)
                checked=audit(data,cfg)
                self.assertTrue(checked['integrity_passed'], checked['issues'])
                self.assertFalse(checked['overloaded_or_unstable'])

    def test_incomplete_point_is_kept_and_marked(self):
        cfg,data=fixture()
        row=data['request_records'][0]
        row['status']='incomplete'
        data['per_task_metrics'][row['task']]['completed_requests']-=1
        checked=audit(data,cfg)
        self.assertTrue(checked['integrity_passed'])
        self.assertTrue(checked['overloaded_or_unstable'])
        self.assertEqual(checked['incomplete_request_ids'],[row['request_id']])
        self.assertEqual(checked['actual_completed_requests'],579)

    def test_bad_route_or_config_is_flagged(self):
        cfg,data=fixture()
        data['request_records'][0]['model']='not_deployed'
        data['random_seed']=999
        checked=audit(data,cfg)
        self.assertFalse(checked['integrity_passed'])
        self.assertTrue(any('routing mismatch' in item for item in checked['issues']))
        self.assertIn('config mismatch: random_seed',checked['issues'])

    def test_failed_shutdown_and_worker_error_are_retained(self):
        cfg,data=fixture()
        data['shutdown']['qwen']['forced']=True
        data['errors']=['worker failure']
        self.assertTrue(audit(data,cfg)['overloaded_or_unstable'])

    def test_sample_std_minmax_and_missing(self):
        s=stats([1,2,3,None])
        self.assertEqual(s['mean'],2)
        self.assertEqual(s['std'],1)
        self.assertEqual(s['min'],1)
        self.assertEqual(s['max'],3)
        self.assertEqual(s['missing'],1)
        self.assertEqual(s['values'],[1,2,3])


if __name__=='__main__':
    unittest.main()
