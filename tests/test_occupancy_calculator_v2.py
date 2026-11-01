"""Host g++/NVIDIA header tests; no CUDA library, context, or device code."""
import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from gsdeploy.forecasting.kernel_resources import static_occupancy
from tools.predictor_validation_20261027.occupancy_calculator_v2 import (
    DEFAULT_HEADER, build_calculator, calculate_occupancy,
)


def fixture():
    hardware = dict(hardware_id='synthetic-cc89', evidence='synthetic-device-properties',
        compute_capability_major=8, compute_capability_minor=9, max_threads_per_block=1024,
        max_threads_per_sm=1536, max_warps_per_sm=48, registers_per_block=65536,
        registers_per_sm=65536, warp_size=32, shared_bytes_per_block=49152,
        shared_bytes_per_sm=102400, sm_count=128, shared_bytes_per_block_optin=101376,
        reserved_shared_bytes_per_block=1024, max_blocks_per_sm=24)
    kernel = dict(hardware_id='synthetic-cc89', static_evidence='synthetic-launch',
        block=[128, 1, 1], grid=[4096, 1, 1], registers_per_thread=32,
        static_shared_bytes=0, dynamic_shared_bytes=0)
    function = dict(max_threads_per_block=1024, partitioned_gc_config='OFF',
        shmem_limit_config='DEFAULT', max_dynamic_shared_bytes=0, num_block_barriers=1,
        evidence='synthetic-function-declaration')
    state = dict(cache_config='NONE', carveout_config=-1, evidence='synthetic-state-declaration')
    return kernel, hardware, function, state


@unittest.skipUnless(DEFAULT_HEADER.is_file() and shutil.which('g++'), 'Host compiler/header required')
class HostOccupancyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.calculator = build_calculator(Path(cls.temp.name) / 'build')

    def calculate(self, values=None, **options):
        return calculate_occupancy(*(fixture() if values is None else values), calculator=self.calculator, **options)

    def test_cc89_supported_and_real_allocation_is_header_rounded(self):
        values = fixture(); values[0].update(block=[33, 1, 1], registers_per_thread=33, static_shared_bytes=1)
        result = self.calculate(values)
        self.assertTrue(result['known'])
        calculated = result['calculation']
        self.assertEqual(calculated['allocated_registers_per_block'], 2560)
        self.assertEqual(calculated['allocated_shared_bytes'], 1152)
        self.assertEqual(calculated['active_blocks_per_sm'], 24)
        self.assertEqual(calculated['block_limits']['blocks'], 24)
        self.assertFalse(result['physically_validated'])

    def test_register_subpartitions_restrict_more_than_rounded_aggregate_total(self):
        values = fixture(); values[0].update(block=[96, 1, 1], registers_per_thread=100)
        result = self.calculate(values)
        calc = result['calculation']
        self.assertEqual(calc['active_blocks_per_sm'], 5)
        self.assertEqual(values[1]['registers_per_sm'] // calc['allocated_registers_per_block'], 6)
        integrated = static_occupancy({**values[0], **result['kernel_patch']}, values[1])
        self.assertEqual(integrated['blocks_per_sm'], 5)
        self.assertIn('architecture_calculator', integrated['limiting_resources'])

    def test_driver_reserved_shared_memory_is_not_silently_zero(self):
        result = self.calculate()
        self.assertEqual(result['calculation']['allocated_shared_bytes'], 1024)
        self.assertEqual(result['calculation']['active_blocks_per_sm'], 12)

    def test_optin_and_carveout_change_static_residency_under_explicit_settings(self):
        values = fixture(); values[0]['dynamic_shared_bytes'] = 60000
        default = self.calculate(values)
        self.assertEqual(default['calculation']['active_blocks_per_sm'], 0)
        values[2].update(shmem_limit_config='OPTIN', max_dynamic_shared_bytes=60000)
        optin = self.calculate(values)
        self.assertEqual(optin['calculation']['active_blocks_per_sm'], 1)
        values = fixture(); values[0]['static_shared_bytes'] = 8192
        values[3]['carveout_config'] = 0
        low = self.calculate(values)['calculation']['active_blocks_per_sm']
        values[3]['carveout_config'] = 100
        high = self.calculate(values)['calculation']['active_blocks_per_sm']
        self.assertLess(low, high)

    def test_missing_function_optin_is_unknown_without_running_binary(self):
        values = fixture(); values[2].pop('shmem_limit_config')
        with patch('subprocess.run', side_effect=AssertionError('Missing evidence must stop before execution')):
            result = self.calculate(values)
        self.assertFalse(result['known'])
        self.assertIn('function.shmem_limit_config', result['missing_fields'])

    def test_explicit_assumption_needs_optin_and_never_emits_normal_kernel_patch(self):
        assumptions = ['device_state.carveout_config']
        self.assertFalse(self.calculate(assumed_fields=assumptions)['known'])
        result = self.calculate(assumed_fields=assumptions, allow_assumptions=True,
                                assumption_evidence='hypothetical default state only')
        self.assertEqual(result['status'], 'ASSUMED_STATIC_HYPOTHESIS')
        self.assertIsNone(result['kernel_patch'])

    def test_unknown_architecture_or_header_block_mismatch_is_unknown(self):
        values = fixture(); values[1]['compute_capability_major'] = 10
        self.assertFalse(self.calculate(values)['known'])
        values = fixture(); values[1]['max_blocks_per_sm'] = 16
        self.assertEqual(self.calculate(values)['reason'], 'HEADER_ARCHITECTURE_BLOCK_LIMIT_DISAGREES_WITH_DEVICE')

    def test_build_and_binary_hashes_are_explicit_and_tampering_is_rejected(self):
        self.assertFalse(self.calculator['links_cuda'])
        self.assertEqual(self.calculator['returncode'], 0)
        self.assertTrue(self.calculator['header']['sha256'])
        changed = copy.deepcopy(self.calculator); changed['binary']['sha256'] = 'bad'
        result = calculate_occupancy(*fixture(), calculator=changed)
        self.assertEqual(result['reason'], 'HOST_CALCULATOR_BINARY_EVIDENCE_MISMATCH')


if __name__ == '__main__': unittest.main()
