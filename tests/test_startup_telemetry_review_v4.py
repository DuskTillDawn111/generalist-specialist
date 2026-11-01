from copy import deepcopy
import unittest

from tools.predictor_validation_20261027.profiling import RESIDENTS
from tools.predictor_validation_20261027.startup_telemetry_review_v4 import BOUNDARIES,validate_startup_telemetry


class StartupTelemetryTests(unittest.TestCase):
    def fixture(self):
        aliases=list(RESIDENTS);pids=list(range(101,105));ready={a:{'pid':p} for a,p in zip(aliases,pids)}
        def sample(t,loaded):
            values={p:10. for p in pids} if loaded else {}
            owners=dict(zip(pids,aliases)) if loaded else {}
            return dict(time_ns=t,query_duration_ns=1,watts=50.,board_used_mib=60.,board_total_mib=100.,board_free_mib=40.,
                owned_used_mib=sum(values.values()),processes_mib=values,owned_pid_aliases=owners,
                owned_aliases=aliases if loaded else [],foreign_pids=[])
        return dict(lifecycle_boundaries=[dict(boundary=b,time_ns=t,clock='host_perf_counter_ns') for b,t in zip(BOUNDARIES,[10,20,21,30,31,50,51,60])],
            board_samples=[sample(9,False),sample(11,False),sample(20,True),sample(22,True),sample(32,True),sample(50,True),sample(61,False)],
            observation_start_ns=33,observation_end_ns=49,ready=ready)

    def test_loading_absence_is_valid_and_active_requires_all_pids(self):
        raw=self.fixture();self.assertEqual(validate_startup_telemetry(raw),[])
        raw['board_samples'][4]['owned_aliases']=[]
        self.assertIn('INVALID_STARTUP_OWNED_BOARD_SAMPLE',validate_startup_telemetry(raw))

    def test_missing_start_envelope_and_lifecycle_do_not_pass(self):
        raw=self.fixture();raw['board_samples'].pop(0)
        self.assertEqual(validate_startup_telemetry(raw),['TELEMETRY_DOES_NOT_ENVELOPE_STARTUP_SHUTDOWN'])
        raw=self.fixture();raw['lifecycle_boundaries'].pop(0)
        self.assertEqual(validate_startup_telemetry(raw),['EXACT_STARTUP_LIFECYCLE_BOUNDARIES_REQUIRED'])

    def test_foreign_loading_pid_not_hidden_by_clean_active_window(self):
        raw=self.fixture();raw['board_samples'][0]['foreign_pids']=[999]
        self.assertIn('INVALID_STARTUP_OWNED_BOARD_SAMPLE',validate_startup_telemetry(raw))

    def test_bad_owned_sum_or_nonfinite_board_sample_rejected(self):
        for field,value in [('owned_used_mib',1.),('board_used_mib',float('nan')),('board_total_mib',0.)]:
            raw=self.fixture();raw['board_samples'][2][field]=value
            self.assertIn('INVALID_STARTUP_OWNED_BOARD_SAMPLE',validate_startup_telemetry(raw))


if __name__=='__main__':unittest.main()
