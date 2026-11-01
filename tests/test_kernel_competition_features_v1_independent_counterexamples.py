"""Document accepted invalid domains in exact v1, without GPU or fitting."""
import copy
import unittest

from tests.test_kernel_competition_features_v1 import provider, capture
from tests.test_kernel_provider import active_rows, ALIASES


class IndependentKernelV1Counterexamples(unittest.TestCase):
    def test_disjoint_devices_are_accepted_as_competing_peers(self):
        p = provider()
        cap = capture(p)
        cap['kernels'][1]['device'] = 1
        out = p.training_rows([cap],allow_cpu_fixture=True)
        self.assertEqual(out['accepted_kernel_count'],2)
        self.assertEqual(out['observations'][0]['path'][-1]['features']['vector']['peer_kernel_count'],1)
        self.assertFalse(out['native_sources_authenticated'])

    def test_foreign_owner_profile_is_accepted_while_inference_rejects(self):
        p = provider()
        cap = capture(p)
        foreign = p.provider.segments['b','kernel']['kernel']
        cap['kernels'][0].update(profile_id='b',name=foreign['name'],static_kernel=copy.deepcopy(foreign))
        out = p.training_rows([cap],allow_cpu_fixture=True)
        self.assertEqual(out['accepted_kernel_count'],2)
        self.assertEqual(out['rows'][0]['group'][:3],('a','b','kernel'))
        active = active_rows(p.provider)
        active['a'].update(profile_id='b',segment_duration_ms=200)
        with self.assertRaisesRegex(ValueError,'request identity'):
            p.features(active['a'],active,ALIASES,'mean')


if __name__ == '__main__':
    unittest.main()
