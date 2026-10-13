"""CPU guards for W2 reuse, unchanged budgets, and evidence-based conclusions."""
import copy
import unittest
from unittest.mock import patch
from .prepare import imported_protocol, PILOT, read
from .campaign import run_workload
from .report import decision_gate, category
from candidate_pool_service_demand.dry_run import load_completed_catalog


class MultiworkloadTests(unittest.TestCase):
    def test_W2_is_rejected_before_any_runtime_or_freeze_call(self):
        with patch('integer_replica_multiworkload.campaign.ReplicaRuntimeEvaluator') as evaluator, patch('integer_replica_multiworkload.campaign.verify_freeze') as freeze:
            with self.assertRaisesRegex(ValueError, 'W2 is immutable reference'):
                run_workload('w2', 'W2_generalist_heavy')
            evaluator.assert_not_called(); freeze.assert_not_called()

    def test_imported_protocol_retains_all_measurement_and_budget_fields(self):
        base = read(PILOT / 'pilot_protocol.json'); original = copy.deepcopy(base)
        _, _, _, records, _ = load_completed_catalog()
        workload = copy.deepcopy(base['workload']); workload['name'] = 'W1_balanced'
        workload['arrival_rates_rps'] = {k: v / 2 for k, v in workload['arrival_rates_rps'].items()}
        protocol = imported_protocol(base, workload, records)
        for key in ('measurement_settings', 'search_config', 'search_seeds', 'confirmation_seeds', 'confirmation_order',
                    'objective', 'DQA', 'search_physical_attempt_cap', 'repetitions_per_state', 'confirmation_physical_attempt_cap'):
            self.assertEqual(base[key], protocol[key])
        self.assertEqual(base, original)
        self.assertEqual(protocol['smoke_physical_attempt_cap'], 0)
        bad = copy.deepcopy(workload); bad['warmup_seconds'] = 11
        with self.assertRaises(RuntimeError):
            imported_protocol(base, bad, records)

    def test_gate_does_not_force_deployment_diversity(self):
        row = {'complete': True, 'mean_improvement_percent': 20., 'runtime_objective_improved': True,
            'confirmation_objective_direction_maintained': True, 'runtime_evaluations': 5, 'enumeration_calls': 0,
            'final_stability': 'steady', 'final': {'qwen': 1, 'whisper-small': 1}}
        rows = [copy.deepcopy(row) for _ in range(3)]
        self.assertEqual(decision_gate(rows, [])['status'], 'SUPPORT')
        rows[2]['final']['resnet50'] = 1
        self.assertEqual(decision_gate(rows, [])['status'], 'STRONG SUPPORT')
        rows[2]['confirmation_objective_direction_maintained'] = False
        self.assertEqual(decision_gate(rows, [])['status'], 'PARTIAL')

    def test_gate_requires_three_complete_steady_audited_workloads(self):
        self.assertEqual(decision_gate([{'complete': False}] * 3, [])['status'], 'PARTIAL')
        row = {'complete': True, 'mean_improvement_percent': 20., 'runtime_objective_improved': True,
            'confirmation_objective_direction_maintained': True, 'runtime_evaluations': 7, 'enumeration_calls': 0,
            'final_stability': 'steady', 'final': {'qwen': 1}}
        self.assertEqual(decision_gate([row] * 3, [])['status'], 'PARTIAL')
        row['runtime_evaluations'] = 5
        self.assertEqual(decision_gate([row] * 3, ['source drift'])['status'], 'PARTIAL')

    def test_all_candidate_categories_distinguish_types_from_replicas(self):
        self.assertEqual(category('generalist', 1, 2), 'generalist_replica_up')
        self.assertEqual(category('specialist', 1, 2), 'specialist_replica_up')
        self.assertEqual(category('generalist', 0, 1), 'new_generalist_type')
        self.assertEqual(category('specialist', 0, 1), 'new_specialist_type')
        self.assertEqual(category('generalist', 2, 1), 'scale_down')


if __name__ == '__main__':
    unittest.main()
