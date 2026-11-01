"""CPU qualification arithmetic; no physical evidence is created by fixtures."""
import copy
import unittest
from tools.predictor_validation_20261027.flow_capacity_qualification import assess_rows, POLICY


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.plan = {'aliases': {'s': {}}, 'trace': [{'request_id': 'r'}], 'candidates': [{'id': 'c'}]}
        self.protocol = {'created_ns': 1, 'policy': POLICY}
        raw = dict(startup_begin_ns=2, startup_end_ns=3, phase_start_ns=4, deadline_ns=5,
            request_records=[{'status': 'completed'}], board_samples=[dict(time_ns=t, owned_used_mib=100.,
            watts=10., foreign_pids=[], owned_aliases=[] if t < 3 else ['s']) for t in (2, 3, 4, 5)])
        self.raws = {'c': raw}
        prediction = dict(transferred_rho_threshold_satisfied=True, original_target_resolver={'status': 'UNKNOWN'},
            transfer={'routing_rps_by_family': {'text': {'text': 1.}}, 'rho_by_family': {'text': .6}})
        self.desc = {'candidates': [dict(candidate_id='c', transfer_prediction=prediction,
            original_stability_passed=True, independent_service_comparison={'text': {'text': dict(
                request_count=1, predicted_worker_service_ms=100., observed_mean_worker_service_ms=120.)}})]}

    def row(self):
        return assess_rows(self.protocol, self.plan, self.desc, self.raws)[0]

    def test_twenty_percent_boundary_qualifies_statistics_not_physical_or_old_resolver(self):
        row = self.row(); self.assertTrue(row['statistical_profile_applicable'])
        self.assertTrue(row['statistical_original_rho_planning_admission_passed'])
        self.assertIsNone(row['old_target_resolver_admission'])
        self.assertFalse(row['physical_capacity_bound_certified']); self.assertFalse(row['final_gain_eligible'])

    def test_error_rho_stability_or_telemetry_failures_block_statistical_admission(self):
        base = copy.deepcopy(self.desc)
        self.desc['candidates'][0]['independent_service_comparison']['text']['text']['observed_mean_worker_service_ms'] = 120.1
        self.assertFalse(self.row()['statistical_original_rho_planning_admission_passed'])
        self.desc = copy.deepcopy(base); self.desc['candidates'][0]['transfer_prediction']['transferred_rho_threshold_satisfied'] = False
        self.assertFalse(self.row()['statistical_original_rho_planning_admission_passed'])
        self.desc = copy.deepcopy(base); self.desc['candidates'][0]['original_stability_passed'] = False
        self.assertFalse(self.row()['statistical_original_rho_planning_admission_passed'])
        self.desc = copy.deepcopy(base); self.raws['c']['board_samples'][-1]['foreign_pids'] = [999]
        self.assertFalse(self.row()['statistical_original_rho_planning_admission_passed'])

    def test_missing_active_task_or_late_protocol_never_passes(self):
        self.desc['candidates'][0]['independent_service_comparison'] = {}
        self.assertFalse(self.row()['statistical_profile_applicable'])
        self.protocol['created_ns'] = 2
        with self.assertRaisesRegex(ValueError, 'precede'):
            self.row()


if __name__ == '__main__': unittest.main()
