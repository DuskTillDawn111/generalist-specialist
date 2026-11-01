"""Monitoring startup-through-drain contract; all fixtures are CPU synthetic."""
import copy
import unittest

from tools.predictor_validation_20261027.device_progress_review_contract_v4 import V6,monitoring_boundaries


def fixture():
    names=['startup_begin','startup_ready','priming_end','measurement_and_drain_end','monitoring_end','shutdown_begin','shutdown_end']
    return dict(monitoring_scope=V6.monitoring_scope,owned_peak_scope=V6.owned_peak_scope,shutdown_memory_sampled=False,
        lifecycle_boundaries=[dict(boundary=n,time_ns=t) for n,t in zip(names,[10,20,30,50,70,80,90])],observation_start_ns=35,observation_end_ns=50,
        board_samples=[dict(phase='startup',time_ns=5),dict(phase='measurement',time_ns=40),dict(phase='final_drain',time_ns=60)])


class MonitorContractTests(unittest.TestCase):
    def test_exact_startup_drain_monitor_close_chain(self):
        raw=fixture();before=copy.deepcopy(raw);r=monitoring_boundaries(raw);self.assertEqual(r['issues'],[]);self.assertEqual(raw,before)
    def test_final_sample_must_follow_drain_and_precede_monitor_end(self):
        for time in (49,71,81):
            raw=fixture();raw['board_samples'][-1]['time_ns']=time;self.assertTrue(monitoring_boundaries(raw)['issues'])
    def test_shutdown_sample_and_late_monitor_stop_rejected(self):
        raw=fixture();raw['board_samples'].append(dict(phase='shutdown',time_ns=85));self.assertTrue(monitoring_boundaries(raw)['issues'])
        raw=fixture();raw['lifecycle_boundaries'][4]['time_ns']=80;self.assertTrue(monitoring_boundaries(raw)['issues'])
    def test_missing_startup_duplicate_final_and_old_scope_rejected(self):
        for mutation in ('startup','final','scope','sampled'):
            raw=fixture()
            if mutation=='startup':raw['board_samples'][0]['time_ns']=11
            elif mutation=='final':raw['board_samples'].insert(-1,dict(phase='final_drain',time_ns=55))
            elif mutation=='scope':raw['monitoring_scope']='startup_through_shutdown'
            else:raw['shutdown_memory_sampled']=True
            with self.subTest(mutation=mutation):self.assertTrue(monitoring_boundaries(raw)['issues'])


if __name__=='__main__':unittest.main()
