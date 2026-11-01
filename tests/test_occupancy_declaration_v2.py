"""Pinned-header branch proofs on CPU fixtures; no model accuracy evidence."""
import copy
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest

from tools.predictor_validation_20261027.occupancy_calculator_v2 import (
    DEFAULT_HEADER, build_calculator, calculate_occupancy,
)
from tools.predictor_validation_20261027.occupancy_declaration_v2 import (
    effective_carveout_representatives, read_sqlite_launches, resolve_declaration,
)
from tests.test_occupancy_calculator_v2 import fixture


class SqliteDeclarationMetadataTests(unittest.TestCase):
    def test_sqlite_enum_ids_are_decoded_and_device_binding_requires_exact_uuid(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'synthetic.sqlite'
            kernel,hardware,_,_ = fixture()
            hardware['hardware_id'] = 'GPU-11111111-1111-1111-1111-111111111111'
            connection = sqlite3.connect(path)
            connection.execute('create table TARGET_INFO_GPU(id integer,uuid text,computeMajor integer,computeMinor integer,smCount integer)')
            connection.execute('insert into TARGET_INFO_GPU values(0,?,8,9,128)',(hardware['hardware_id'][4:],))
            connection.execute('create table ENUM_CUDA_SHARED_MEM_LIMIT_CONFIG(id integer,name text)')
            connection.execute("insert into ENUM_CUDA_SHARED_MEM_LIMIT_CONFIG values(41,'CUDA_SHARED_MEM_LIMIT_DEFAULT')")
            connection.execute('create table ENUM_CUDA_KERNEL_LAUNCH_TYPE(id integer,name text)')
            connection.execute("insert into ENUM_CUDA_KERNEL_LAUNCH_TYPE values(7,'CUDA_KERNEL_LAUNCH_TYPE_REGULAR')")
            values = dict(start=10,end=20,deviceId=0,contextId=1,correlationId=17,globalPid=123,
                          blockX=128,blockY=1,blockZ=1,gridX=32,gridY=1,gridZ=1,
                          registersPerThread=32,staticSharedMemory=0,dynamicSharedMemory=0,
                          cacheConfig=2,sharedMemoryLimitConfig=41,sharedMemoryExecuted=32768,launchType=7)
            connection.execute('create table CUPTI_ACTIVITY_KIND_KERNEL('+','.join(name+' integer' for name in values)+')')
            connection.execute('insert into CUPTI_ACTIVITY_KIND_KERNEL values('+','.join('?' for name in values)+')',tuple(values.values()))
            connection.commit();connection.close()
            result = read_sqlite_launches(path,hardware)
            self.assertTrue(result['known'])
            row = result['rows'][0]
            self.assertEqual(row['function_attributes']['shmem_limit_config'],'DEFAULT')
            self.assertFalse(row['kernel']['cooperative_launch'])
            self.assertEqual(row['observed_raw_cache_config'],2)
            self.assertIsNone(row['device_state']['cache_config'])
            self.assertIsNotNone(row['completed_launch_evidence'])
            hardware['hardware_id'] = 'GPU-22222222-2222-2222-2222-222222222222'
            self.assertEqual(read_sqlite_launches(path,hardware)['reason'],'NO_UNIQUE_EXACT_TRACE_HARDWARE_UUID_MATCH')


@unittest.skipUnless(DEFAULT_HEADER.is_file() and shutil.which('g++'), 'Host compiler/header required')
class OccupancyDeclarationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.calculator = build_calculator(Path(cls.temp.name)/'build')

    def minimal(self):
        kernel, hardware, _, _ = fixture()
        kernel['static_evidence'] = dict(source=dict(path='synthetic.sqlite', sha256='a'*64),
                                        launch=dict(sqlite_rowid=1, start=10, end=20))
        return kernel, hardware, dict(shmem_limit_config='DEFAULT', evidence='synthetic_actual_mode'), {}

    def resolve(self, values=None, **kwargs):
        return resolve_declaration(*(values or self.minimal()), calculator=self.calculator, **kwargs)

    def test_missing_relevant_function_maxthreads_stays_unknown_without_completed_proof(self):
        values = self.minimal()
        result = self.resolve(values, effective_shared_bytes=32768, effective_shared_evidence='synthetic_executed')
        self.assertEqual(result['remaining_required_fields'], ['function.max_threads_per_block'])
        derived = {row['field']: row for row in result['derivations']}
        self.assertEqual(derived['function.num_block_barriers']['applicability'], 'not_applicable')
        self.assertEqual(derived['function.max_dynamic_shared_bytes']['applicability'], 'not_applicable')
        self.assertNotIn('max_dynamic_shared_bytes', result['original_function_attributes'])

    def test_completed_launch_lower_bound_allows_equivalent_static_calculation(self):
        values = self.minimal()
        result = self.resolve(values, effective_shared_bytes=32768, effective_shared_evidence='synthetic_executed',
                              completed_launch_evidence=values[0]['static_evidence'])
        self.assertTrue(result['known'])
        self.assertEqual(result['calculation']['calculation']['active_blocks_per_sm'], 12)
        self.assertFalse(result['real_function_attributes_completed'])
        self.assertFalse(result['requested_state_reconstructed'])
        self.assertNotIn('max_threads_per_block', result['original_function_attributes'])
        field = next(row for row in result['derivations'] if row['field'] == 'function.max_threads_per_block')
        self.assertEqual(field['applicability'], 'lower_bound_equivalent_occupancy')
        self.assertEqual(field['calculation_representative'], 128)

    def test_completed_proof_must_be_exact_launch_and_positive_interval(self):
        values = self.minimal()
        proof = copy.deepcopy(values[0]['static_evidence']); proof['launch']['end'] += 1
        result = self.resolve(values, completed_launch_evidence=proof)
        self.assertEqual(result['reason'], 'COMPLETED_LAUNCH_PROOF_NOT_BOUND_TO_EXACT_LAUNCH')

    def test_optin_dynamic_maximum_requires_completed_proof_then_is_equivalent(self):
        values = self.minimal(); values[2]['shmem_limit_config'] = 'OPTIN'
        values[0]['dynamic_shared_bytes'] = 60000
        result = self.resolve(values, effective_shared_bytes=102400, effective_shared_evidence='synthetic_executed')
        self.assertFalse(result['known'])
        self.assertIn('function.max_dynamic_shared_bytes', result['remaining_required_fields'])
        equivalent = self.resolve(values, effective_shared_bytes=102400, effective_shared_evidence='synthetic_executed',
                                  completed_launch_evidence=values[0]['static_evidence'])
        self.assertTrue(equivalent['known'])
        self.assertEqual(equivalent['calculation']['calculation']['active_blocks_per_sm'], 1)
        projection = next(row for row in equivalent['derivations'] if row['field'] == 'function.max_dynamic_shared_bytes')
        self.assertEqual(projection['proof']['lower_bound'], 60000)
        self.assertEqual(projection['proof']['physical_upper_bound'], 101376)
        self.assertFalse(projection['proof']['actual_max_dynamic_shared_bytes_known'])
        full = fixture(); full[0]['dynamic_shared_bytes'] = 60000
        full[2]['shmem_limit_config'] = 'OPTIN'
        full[2]['max_dynamic_shared_bytes'] = 0
        zero = calculate_occupancy(*full, calculator=self.calculator)['calculation']['active_blocks_per_sm']
        full[2]['max_dynamic_shared_bytes'] = 60000
        one = calculate_occupancy(*full, calculator=self.calculator)['calculation']['active_blocks_per_sm']
        self.assertEqual((zero, one), (0, 1))

    def test_optin_both_threshold_branches_and_eligibility_boundary(self):
        for static, dynamic in ((0,0), (0,1024), (0,48128), (0,48129),
                                 (1024,47104), (1024,47105), (8192,60000)):
            full = fixture()
            full[0].update(static_shared_bytes=static, dynamic_shared_bytes=dynamic)
            full[2].update(shmem_limit_config='OPTIN', max_dynamic_shared_bytes=dynamic)
            full[3]['carveout_config'] = 100
            baseline = calculate_occupancy(*full, calculator=self.calculator)['calculation']
            threshold = 49152-static-1024
            upper = 101376-static
            candidates = {dynamic, upper, threshold-1, threshold, threshold+1}
            for maximum in sorted(candidates):
                if not dynamic <= maximum <= upper:
                    continue
                full[2]['max_dynamic_shared_bytes'] = maximum
                self.assertEqual(calculate_occupancy(*full, calculator=self.calculator)['calculation'], baseline)
            if dynamic:
                full[2]['max_dynamic_shared_bytes'] = dynamic-1
                self.assertEqual(calculate_occupancy(*full, calculator=self.calculator)
                                 ['calculation']['active_blocks_per_sm'], 0)

    def test_irrelevant_field_variations_preserve_complete_calculation(self):
        full = fixture(); full[3]['carveout_config'] = 32
        baseline = calculate_occupancy(*full, calculator=self.calculator)['calculation']
        for field, variants in [('partitioned_gc_config', ['OFF', 'ON', 'ON_STRICT']),
                                ('num_block_barriers', [0, 1, 64]),
                                ('max_dynamic_shared_bytes', [0, 100000, 2**31-1]),
                                ('max_threads_per_block', [128, 256, 1024])]:
            for value in variants:
                changed = copy.deepcopy(full); changed[2][field] = value
                self.assertEqual(calculate_occupancy(*changed, calculator=self.calculator)['calculation'], baseline)
        for value in ('NONE', 'SHARED', 'L1', 'EQUAL'):
            changed = copy.deepcopy(full); changed[3]['cache_config'] = value
            self.assertEqual(calculate_occupancy(*changed, calculator=self.calculator)['calculation'], baseline)

    def test_executed_capacity_projection_exhausts_percentages_without_guessing_request(self):
        all_percentages = []
        for capacity in (0, 8192, 16384, 32768, 65536, 102400):
            all_percentages += effective_carveout_representatives(capacity, 102400)
        self.assertEqual(sorted(all_percentages), list(range(101)))
        values = self.minimal()
        for percentage in effective_carveout_representatives(32768, 102400):
            full = fixture(); full[3].update(cache_config='NONE', carveout_config=percentage)
            result = calculate_occupancy(*full, calculator=self.calculator)
            self.assertEqual(result['calculation']['active_blocks_per_sm'], 12)
        values[0]['static_shared_bytes'] = 40000
        result = self.resolve(values, effective_shared_bytes=32768, effective_shared_evidence='synthetic_executed')
        self.assertEqual(result['reason'], 'EXECUTED_SHARED_CAPACITY_BELOW_ROUNDED_BLOCK_ALLOCATION')

    def test_barrier_is_not_removed_for_hopper_and_affects_header_ceiling(self):
        full = fixture()
        full[1].update(compute_capability_major=9, compute_capability_minor=0,
                       max_threads_per_sm=2048, max_warps_per_sm=64, max_blocks_per_sm=32,
                       shared_bytes_per_sm=233472, shared_bytes_per_block_optin=232448)
        full[2]['num_block_barriers'] = 0
        no_barrier = calculate_occupancy(*full, calculator=self.calculator)['calculation']['active_blocks_per_sm']
        full[2]['num_block_barriers'] = 8
        with_barrier = calculate_occupancy(*full, calculator=self.calculator)['calculation']['active_blocks_per_sm']
        self.assertLess(with_barrier, no_barrier)
        self.assertEqual(self.resolve(full)['reason'], 'ARCHITECTURE_RULES_NOT_AUDITED')

    def test_unrepresentable_capacity_or_unpinned_header_is_unknown(self):
        self.assertEqual(self.resolve(effective_shared_bytes=33000, effective_shared_evidence='synthetic')
                         ['reason'], 'EFFECTIVE_SHARED_CAPACITY_NOT_REPRESENTABLE')
        altered = copy.deepcopy(self.calculator); altered['header']['sha256'] = 'b'*64
        result = resolve_declaration(*self.minimal(), calculator=altered)
        self.assertEqual(result['reason'], 'PINNED_OCCUPANCY_HEADER_REQUIRED')

    def test_resolver_does_not_supply_evidence_for_unobserved_relevant_fields(self):
        values = self.minimal(); values[2].pop('evidence')
        self.assertEqual(self.resolve(values)['reason'], 'OBSERVED_FUNCTION_DECLARATION_EVIDENCE_REQUIRED')

    def test_partitioned_gc_is_relevant_on_supported_sm52_branch(self):
        full = fixture()
        full[0].update(block=[96,1,1], registers_per_thread=16)
        full[1].update(compute_capability_major=5, compute_capability_minor=2,
                       max_threads_per_sm=2048, max_warps_per_sm=64, max_blocks_per_sm=32,
                       shared_bytes_per_sm=98304, shared_bytes_per_block_optin=49152,
                       reserved_shared_bytes_per_block=0)
        full[2]['partitioned_gc_config'] = 'OFF'
        off = calculate_occupancy(*full, calculator=self.calculator)['calculation']['active_blocks_per_sm']
        full[2]['partitioned_gc_config'] = 'ON_STRICT'
        strict = calculate_occupancy(*full, calculator=self.calculator)['calculation']['active_blocks_per_sm']
        self.assertEqual((off,strict), (21,20))
        self.assertEqual(self.resolve(full)['reason'], 'ARCHITECTURE_RULES_NOT_AUDITED')


if __name__ == '__main__':
    unittest.main()
