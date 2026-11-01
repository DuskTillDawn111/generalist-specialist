"""Counterexamples for resource dimensions, event alignment and evidence gates."""
import copy
import unittest

from gsdeploy.forecasting.kernel_resources import (
    static_occupancy, check_resident_blocks, predict_sequence_alignment,
    complementarity_candidate,
)


def hardware():
    return dict(hardware_id='synthetic-device', evidence='synthetic-limits', warp_size=32,
                sm_count=2, max_threads_per_block=1024, max_threads_per_sm=1024,
                max_warps_per_sm=32, max_blocks_per_sm=8, registers_per_sm=32768,
                shared_bytes_per_sm=65536)


def kernel(duration=100, shared=0):
    return dict(hardware_id='synthetic-device', static_evidence='synthetic-launch',
                block=[128, 1, 1], grid=[128, 1, 1], registers_per_thread=32,
                static_shared_bytes=shared, dynamic_shared_bytes=0,
                allocated_registers_per_block=4096, allocated_shared_bytes=shared,
                allocation_evidence='synthetic-rounded-allocation',
                solo_duration_ns=duration, solo_timing_basis='synthetic', timing_evidence='synthetic-timing')


def sequence(name, kernels, offset=0):
    return dict(id=name, offset_ns=offset, kernels=kernels)


def rollout(sequences, **options):
    return predict_sequence_alignment(sequences, hardware(), allow_inverse_occupancy_assumption=True,
                                      concurrency_evidence='synthetic-same-context-capability', **options)


def config(name, compute, dram, size=1):
    return dict(config_id=name, counter_config_id=name, hardware_id='synthetic-device',
                counter_evidence='synthetic-counters', compute_fraction=compute, dram_fraction=dram,
                counter_metric_signature='synthetic-compatible-compute-dram-definition',
                compute_basis='measured_compute_throughput', dram_basis='measured_dram_throughput',
                resident_bytes=size)


class StaticConstraintTests(unittest.TestCase):
    def test_warp_rounding_and_explicit_allocation(self):
        k = kernel(); k.update(block=[33, 1, 1], registers_per_thread=1,
                                allocated_registers_per_block=4096)
        result = static_occupancy(k, hardware())
        self.assertEqual(result['warps_per_block'], 2)
        self.assertEqual(result['blocks_per_sm'], 8)
        self.assertEqual(result['theoretical_occupancy'], .5)

    def test_shared_and_register_limits_are_distinct(self):
        k = kernel(shared=32768)
        result = static_occupancy(k, hardware())
        self.assertEqual(result['blocks_per_sm'], 2)
        self.assertEqual(result['limiting_resources'], ['shared'])
        self.assertFalse(result['physically_validated'])

    def test_naive_register_product_is_not_an_exact_allocation(self):
        k = kernel(); k.pop('allocated_registers_per_block')
        result = static_occupancy(k, hardware())
        self.assertFalse(result['known'])
        self.assertEqual(result['arithmetic_block_upper_bound'], 8)
        self.assertIsNone(result['theoretical_occupancy'])

    def test_missing_geometry_and_hardware_identity_fail_closed(self):
        k = kernel(); k['block'] = None
        self.assertFalse(static_occupancy(k, hardware())['known'])
        k = kernel(); k['hardware_id'] = 'another-device'
        self.assertEqual(static_occupancy(k, hardware())['reasons'], ['HARDWARE_IDENTITY_MISMATCH'])

    def test_integer_block_packing_can_fail_even_if_individual_kernels_fit(self):
        k = kernel(shared=40000)
        result = check_resident_blocks({'a': k, 'b': k}, {'a': 1, 'b': 1}, hardware())
        self.assertFalse(result['feasible_under_declared_constraints'])
        self.assertEqual(result['exceeded'], ['shared'])
        self.assertFalse(result['actual_concurrent_residency_proven'])

    def test_oversize_block_or_allocation_has_no_residency(self):
        k = kernel(); k['block'] = [2048, 1, 1]; k['allocated_registers_per_block'] = 65536
        self.assertEqual(static_occupancy(k, hardware())['status'], 'STATICALLY_INFEASIBLE')
        k = kernel(); k['allocated_registers_per_block'] = 1
        with self.assertRaises(ValueError):
            static_occupancy(k, hardware())

    def test_cluster_and_bad_geometry_are_not_silently_flattened(self):
        k = kernel(); k['cluster'] = [2, 1, 1]
        self.assertEqual(static_occupancy(k, hardware())['reasons'], ['CLUSTER_RESIDENCY_NOT_MODELED'])
        k = kernel(); k['block'] = [0, 1, 1]
        with self.assertRaises(ValueError):
            static_occupancy(k, hardware())


class SequenceAlignmentTests(unittest.TestCase):
    def test_default_never_converts_occupancy_to_throughput(self):
        result = predict_sequence_alignment([sequence('a', [kernel()])], hardware())
        self.assertFalse(result['known'])
        self.assertIn('OPT_IN', result['reasons'][0])

    def test_completion_event_restores_progress_and_conserves_work(self):
        result = rollout([sequence('a', [kernel(100)]), sequence('b', [kernel(200)])])
        self.assertEqual(result['sequences']['a']['finish_ns'], 200)
        self.assertEqual(result['sequences']['b']['finish_ns'], 300)
        # b receives100 solo-ns while sharing, then100 at full occupancy.
        self.assertEqual(result['transitions'][1]['assumed_relative_progress']['b'], 1)
        self.assertEqual(result['sequences']['b']['relative_increase_over_solo'], .5)
        self.assertAlmostEqual(result['sequences']['b']['fraction_of_colocated_latency'], 1/3)
        self.assertFalse(result['roomie_algorithm_1_reproduction'])

    def test_start_offset_avoids_overlap(self):
        result = rollout([sequence('a', [kernel(100)]), sequence('b', [kernel(100)], 100)])
        self.assertEqual(result['sequences']['b']['latency_ns'], 100)
        self.assertEqual(result['sequences']['a']['slowdown_ratio'], 1)

    def test_unequal_kernel_lengths_advance_on_completion_not_lockstep(self):
        result = rollout([sequence('a', [kernel(100), kernel(100)]), sequence('b', [kernel(300)])])
        self.assertEqual(result['sequences']['a']['finish_ns'], 400)
        self.assertEqual(result['sequences']['b']['finish_ns'], 500)
        self.assertEqual(result['transitions'][1]['kernel_indices'], {'a': 1, 'b': 0})

    def test_host_gap_does_not_consume_gpu_residency(self):
        k = kernel(100); k['gap_before_ns'] = 100
        result = rollout([sequence('a', [k]), sequence('b', [kernel(100)])])
        self.assertEqual(result['sequences']['a']['finish_ns'], 200)
        self.assertEqual(result['sequences']['a']['solo_latency_ns'], 200)
        self.assertEqual(result['sequences']['b']['finish_ns'], 100)

    def test_incompatible_shared_memory_serializes_without_zero_occupancy_division(self):
        result = rollout([sequence('a', [kernel(100, 40000)]), sequence('b', [kernel(100, 40000)])])
        self.assertEqual(result['sequences']['a']['finish_ns'], 100)
        self.assertEqual(result['sequences']['b']['finish_ns'], 200)
        tiny = rollout([sequence('a', [kernel(1e-12, 40000)]), sequence('b', [kernel(1e-12, 40000)])])
        self.assertEqual(tiny['sequences']['b']['finish_ns'], 2e-12)

    def test_profiled_duration_and_small_grid_stay_unknown(self):
        k = kernel(); k['solo_timing_basis'] = 'profiled_kernel'
        self.assertIn('NOT_ISOLATED', rollout([sequence('a', [k])])['reasons'][0])
        k = kernel(); k['grid'] = [1, 1, 1]
        self.assertIn('SMALL_GRID', rollout([sequence('a', [k])])['reasons'][0])

    def test_policy_choice_is_explicit_and_inputs_unchanged(self):
        seqs = [sequence('a', [kernel(100)]), sequence('b', [kernel(100)])]
        saved = copy.deepcopy(seqs)
        result = rollout(seqs, policy='fcfs_blocks')
        self.assertEqual(result['sequences']['a']['finish_ns'], 100)
        self.assertEqual(result['sequences']['b']['finish_ns'], 200)
        self.assertEqual(seqs, saved)

    def test_budget_exhaustion_does_not_return_partial_success(self):
        result = rollout([sequence('a', [kernel(), kernel()])], max_events=1)
        self.assertFalse(result['known'])

    def test_exact_event_budget_completion_is_accepted(self):
        result = rollout([sequence('a', [kernel()])], max_events=1)
        self.assertTrue(result['known'])


class ComplementarityTests(unittest.TestCase):
    def test_configuration_specific_opposite_demands_propose_only_a_candidate(self):
        result = complementarity_candidate(config('small', .7, .2), config('large', .2, .7), memory_budget_bytes=4)
        self.assertEqual(result['status'], 'HEURISTIC_CANDIDATE')
        self.assertEqual(result['labels'], ['COMPUTE', 'DRAM'])
        self.assertFalse(result['candidate_domain_pruning_allowed'])
        self.assertFalse(result['physical_interference_avoided'])

    def test_vram_or_occupancy_cannot_replace_counters(self):
        a = config('a', .8, .2); b = config('b', .2, .8)
        b.pop('dram_fraction'); b['resident_bytes'] = 1
        self.assertFalse(complementarity_candidate(a, b)['known'])
        b = config('b', .2, .8); b['compute_basis'] = 'theoretical_occupancy'
        self.assertFalse(complementarity_candidate(a, b)['known'])

    def test_counter_configuration_and_hardware_are_bound(self):
        a = config('a', .8, .2); b = config('b', .2, .8)
        b['counter_config_id'] = 'different-batch'
        self.assertFalse(complementarity_candidate(a, b)['known'])
        b = config('b', .2, .8); b['hardware_id'] = 'other'
        self.assertFalse(complementarity_candidate(a, b)['known'])

    def test_bandwidth_screen_and_memory_capacity_are_separate(self):
        a = config('a', .8, .3, 10); b = config('b', .3, .8, 10)
        result = complementarity_candidate(a, b, memory_budget_bytes=100)
        self.assertTrue(result['complementary'])
        self.assertFalse(result['additive_screen_fits'])
        result = complementarity_candidate(config('a', .7, .2, 10), config('b', .2, .7, 10), memory_budget_bytes=15)
        self.assertEqual(result['memory_capacity']['status'], 'EXCEEDS_DECLARED_BYTES')
        self.assertEqual(result['status'], 'NOT_PREFERRED_BY_HEURISTIC')

    def test_metric_normalizations_must_match(self):
        a = config('a', .7, .2); b = config('b', .2, .7)
        b['counter_metric_signature'] = 'different-compute-peak-or-collection-window'
        self.assertFalse(complementarity_candidate(a, b)['known'])

    def test_missing_memory_budget_does_not_fabricate_peak(self):
        result = complementarity_candidate(config('a', .7, .2), config('b', .2, .7))
        self.assertEqual(result['memory_capacity']['status'], 'UNKNOWN')
        self.assertFalse(result['deployment_qualified'])


if __name__ == '__main__':
    unittest.main()
