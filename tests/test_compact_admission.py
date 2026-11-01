import copy
import unittest
from tests.test_compact_protocol import fixtures
from gsdeploy.profiling.admission import quality_gate, compare_quality, compare_costs


class CompactAdmissionTests(unittest.TestCase):
    def test_failure_and_each_absolute_quality_stratum_are_checked(self):
        adapter, _, _ = fixtures()
        good = {'failed_samples': 0, 'EM': .6, 'F1': .7,
                'answerable': {'EM': .6}, 'unanswerable': {'EM': .6}}
        self.assertTrue(quality_gate(adapter.policy, 'text', good)['passed'])
        for changed in [{'failed_samples': 1}, {'EM': .4}, {'F1': .4},
                        {'answerable': {'EM': .4}}, {'unanswerable': {'EM': None}}]:
            self.assertFalse(quality_gate(adapter.policy, 'text', {**good, **changed})['passed'])
        self.assertTrue(quality_gate(adapter.policy, 'asr', {'failed_samples': 0, 'WER': .15})['passed'])
        self.assertFalse(quality_gate(adapter.policy, 'asr', {'failed_samples': 0, 'WER': .151})['passed'])

    def test_cluster_paired_advantage_requires_strict_positive_confidence_bound(self):
        adapter, study, samples = fixtures()
        cohort, generalist, specialist = [], [], []
        for i in range(12):
            sample = {**samples['text'], 'id': str(i), 'article': 'topic-' + str(i // 2)}
            cohort.append(sample)
            for candidate, output, target in [(study['generalists'][0], 'wrong', generalist),
                                               (study['specialists'][0], 'Paris', specialist)]:
                request = adapter.build(candidate, sample)
                target.append(adapter.finish(request, sample, output, finish_reason='eos', generated_tokens=3,
                                              terminal_token_id=2, eos_token_ids=[2]))
        policy = copy.deepcopy(adapter.policy)
        policy['quality_gate']['specialist_comparison']['paired_cluster_bootstrap']['resamples'] = 1000
        result = compare_quality(policy, cohort, generalist, specialist)
        self.assertTrue(result['passed'])
        self.assertEqual(result['source_clusters'], 6)
        self.assertEqual(result['one_sided_lower_bound'], 1)
        tie = compare_quality(policy, cohort, specialist, specialist)
        self.assertFalse(tie['passed'])
        specialist[0]['success'] = False
        self.assertFalse(compare_quality(policy, cohort, generalist, specialist)['passed'])

    def test_energy_and_cost_gate_rejects_closed_loop_or_mismatched_observations(self):
        baseline = {'task': 'asr', 'input_manifest_sha256': 'a' * 64, 'trace_sha256': 'b' * 64,
                    'gpu_uuid': 'GPU-fixture', 'duration_seconds': 600.0,
                    'startup_amortization_seconds': 3600.0, 'energy_method': 'nvml_board_power_integral',
                    'protocol_sha256': 'c' * 64, 'power_limit_watts': 450.0,
                    'kind': 'measured', 'scope': 'whole_gpu_board', 'workload': 'matched_open_loop',
                    'coverage_valid': True, 'stable': True, 'complete_cohort': True,
                    'resident_mib': 6000.0, 'mean_latency_ms': 100.0, 'effective_board_watts': 50.0}
        specialist = {**baseline, 'resident_mib': 5000.0, 'mean_latency_ms': 80.0, 'effective_board_watts': 60.0}
        self.assertTrue(compare_costs(baseline, specialist)['passed'])
        for change in [{'workload': 'closed_loop'}, {'coverage_valid': False}, {'trace_sha256': 'wrong'},
                       {'resident_mib': 2999.0}, {'mean_latency_ms': 201.0}, {'effective_board_watts': 49.0},
                       {'stable': False}, {'kind': 'synthetic_fixture'}, {'power_limit_watts': None}]:
            self.assertFalse(compare_costs(baseline, {**specialist, **change})['passed'], change)


if __name__ == '__main__':
    unittest.main()
