import unittest
from types import SimpleNamespace

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027 import projected_handler_bridge_v44 as v
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import OwnerLifecycle


class PhysicalFixture:
    model = {'model_sha256': 'a' * 64}

    def __init__(self):
        self.observed = []

    def next_owner(self, features):
        self.observed.append(features)
        return dict(physical_owner_probabilities=dict(zip(v.OWNERS, (.6, .1, .2, .1))),
            most_likely_physical_owner=v.OWNERS[0],
            OOD_reasons=['UNCALIBRATED_FIXED_MEDIAN_WORKER_HANDLER_PROJECTION'])


class ProjectedBridgeTests(unittest.TestCase):
    def test_worker_boundary_changes_feature_without_phase_or_issued_coercion(self):
        states = {o: OwnerLifecycle(False, False, False) for o in v.OWNERS}
        states['text#0'] = OwnerLifecycle(True, None, False)
        q = v.feature_projection('text#0', None, 1234, states)
        self.assertEqual(q['context_age_ns'], 1234)
        self.assertTrue(q['handler_active']['text#0'])
        self.assertFalse(q['handler_active']['ocr#0'])
        states['ocr#0'] = OwnerLifecycle(None, True, True)
        # CUDA readiness cannot manufacture a worker BEGIN fact.
        with self.assertRaises(ValueError):
            v.feature_projection('text#0', None, 1234, states)

    def test_single_handler_keeps_physical_other_and_same_holder_rejection_input(self):
        bridge = object.__new__(v.ObservableReferenceMark)
        bridge.candidate = PhysicalFixture()
        bridge.states = bridge.clock = None
        states = {o: OwnerLifecycle(o == 'text#0', None if o == 'text#0' else False, False)
            for o in v.OWNERS}
        bridge.observe_view(SimpleNamespace(clock_ns=1000, owner_lifecycle=tuple(states.items())))
        r = bridge.predict_given_save_now(held_owner='text#0', context_age_ns=1000,
            pending_frontier=[], previous_held_owner='ocr#0')
        self.assertEqual(r['most_likely_next_owner'], 'text#0')
        self.assertEqual(r['next_owner_probabilities']['text#0'], 1.)
        self.assertEqual(r['physical_owner_probabilities'][v.OWNERS[0]], .6)
        self.assertFalse(r['same_holder_probability_removed'])
        self.assertIn('UNCALIBRATED_FIXED_MEDIAN_WORKER_HANDLER_PROJECTION', r['ood_flags'])
        self.assertFalse(r['worker_handler_clock_projection_independently_validated'])
        self.assertTrue(r['active_reference_service_bridge_is_unvalidated_hypothesis'])
        with self.assertRaises(UnknownEvidence):
            bridge.observe_view(SimpleNamespace(clock_ns=999, owner_lifecycle=tuple(states.items())))


if __name__ == '__main__':
    unittest.main()
