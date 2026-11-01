"""Real board occupancy must reject oversized generalists before quality jobs."""
import copy
import unittest
from gsdeploy.profiling.memory_gate import screen_pool_memory


class MemoryGateTests(unittest.TestCase):
    def fixture(self):
        study = {'generalists': [{'slot': 'g1'}, {'slot': 'g2'}, {'slot': 'g3'}],
                 'specialists': [{'slot': 's1'}, {'slot': 's2'}],
                 'admission_requirements': {'specialist_generalist_memory_ratio': [.5, 2]},
                 'gpu_memory_mib': 49140, 'safe_factor': 1.110854}
        def profile(resident):
            return {'status': 'COMPLETE', 'kind': 'measured', 'replicas': 1,
                    'scope': 'standalone_same_model_replicas', 'telemetry_errors': [],
                    'resident_after_warmup_mib': resident, 'peak_board_used_mib': resident + 500}
        profiles = {slot: profile(size) for slot, size in {'g1': 6000, 'g2': 7000, 'g3': 8000,
                                                        's1': 4200, 's2': 6200}.items()}
        return study, profiles

    def test_every_generalist_must_match_every_specialist(self):
        study, profiles = self.fixture()
        result = screen_pool_memory(study, profiles)
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['comparisons']), 6)
        self.assertEqual(result['generalist_planning_interval_mib'], {'minimum_mib': 3100, 'maximum_mib': 8400})
        self.assertFalse(result['admission_granted'])

    def test_cached_board_memory_cannot_be_replaced_by_smaller_live_tensor_bytes(self):
        study, profiles = self.fixture()
        profiles['g2'].update(resident_after_warmup_mib=35000, peak_board_used_mib=36000,
                              allocated_mib=7000, parameter_bytes=4000 * 1024**2)
        result = screen_pool_memory(study, profiles)
        self.assertFalse(result['passed'])
        self.assertEqual(sum(not c['passed'] for c in result['comparisons']), 2)

    def test_missing_failed_unverified_and_unsafe_measurements_never_pass(self):
        study, profiles = self.fixture()
        for change in [{'status': 'FAILED'}, {'kind': 'synthetic'}, {'replicas': 2},
                       {'telemetry_errors': ['error']}, {'resident_after_warmup_mib': None},
                       {'peak_board_used_mib': float('nan')}, {'peak_board_used_mib': 48000},
                       {'peak_board_used_mib': 1000}]:
            with self.subTest(change=change):
                changed = copy.deepcopy(profiles)
                changed['g1'].update(change)
                self.assertFalse(screen_pool_memory(study, changed)['passed'])
        del profiles['g1']
        self.assertFalse(screen_pool_memory(study, profiles)['passed'])

    def test_empty_pool_cannot_vacuously_pass(self):
        study, profiles = self.fixture()
        study['generalists'] = []
        self.assertFalse(screen_pool_memory(study, profiles)['passed'])


if __name__ == '__main__':
    unittest.main()
