"""Independent synthetic-law checks using the original whole-calendar engine."""
import copy
from fractions import Fraction
import unittest
from unittest.mock import patch

from tests import test_twoFIFO_matching_prospective_bridge_v1 as fixtures
from tools.predictor_validation_20261027 import twoFIFO_matching_prospective_bridge_v1 as m


class IndependentProspectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.BridgeTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        fixtures.BridgeTests.tearDownClass()

    def test_partial_future_forecast_keeps_exact_unknown_mass_and_all181_ids(self):
        data = fixtures.BridgeTests
        draws, _ = m.generated.compile_case_draws(data.gate['public_requests'], data.law)
        draws = m.bind_owned_producers(draws, data.prefix_doc['seed'])
        unknown = m.prospective.ensemble.UnknownForecastInputs('CPU_INDEPENDENT_MISSING_SOURCE')
        for draw in draws[1:]:
            draw['inputs'] = unknown
        result = m.decode(m.propagate(data.gate['public_requests'], draws,
            data.prefix_doc['seed'], data.contract, data.gate))
        self.assertEqual(result['ensemble']['world_unknown_probability_mass'], Fraction(11, 12))
        self.assertEqual(result['ensemble']['utility']['unknown_probability_mass'], 1)
        public_ids = [r['request_id'] for r in data.gate['public_requests']]
        for projection in result['complete_draw_projection']:
            self.assertEqual([r['request_id'] for r in projection['requests']], public_ids)
            self.assertEqual(projection['weight'], Fraction(1, 12))
        numeric = result['complete_draw_projection'][0]
        self.assertEqual(numeric['status'], 'GENERATED_WORLD')
        self.assertTrue(all(r['completion_time_ns'] is not None for r in numeric['requests']))
        self.assertGreater(numeric['requests'][-1]['completion_time_ns'], data.epoch+80_000_000_000)
        self.assertTrue(all(r['completion_time_ns'] is None for d in result['complete_draw_projection'][1:] for r in d['requests']))

    def test_changed_law_block_or_missing_owner_key_never_reaches_partial_selection(self):
        data = fixtures.BridgeTests
        changed = copy.deepcopy(data.law)
        changed['blocks'][0] += '-foreign'
        with patch.object(m.generated, 'compile_case_draws') as compiler, self.assertRaises(ValueError):
            m.replay_law(data.bundle['plans'], changed, None, data.plan, data.prefix_doc, data.gate)
        compiler.assert_not_called()
        changed = copy.deepcopy(data.law)
        changed['vectors'] = [v for v in changed['vectors'] if v['public_key'][0] != 'ocr#0']
        with patch.object(m.generated, 'compile_case_draws') as compiler:
            value = m.replay_law(data.bundle['plans'], changed, None, data.plan, data.prefix_doc, data.gate)
        compiler.assert_not_called()
        self.assertEqual(m.decode(value['forecast'])['ensemble']['world_unknown_probability_mass'], 1)
        self.assertEqual(value['generated_worlds'], 0)

    def test_missing_predeclared_fit_or_wrong_fit_campaign_rejected_before_numeric_replay(self):
        data = fixtures.BridgeTests
        fit = dict(manifest=data.fit_manifest)
        gate = copy.deepcopy(data.gate)
        gate['model_sources'] = []
        with patch.object(m.response, 'audit_fit', return_value=dict(report=fit)), \
                patch.object(m.prefix, 'audit_gate', return_value=gate), \
                patch.object(m.fresh, 'audit_plan', return_value=data.plan), \
                patch.object(m.prefix, 'audit_prefix', return_value=data.prefix_doc), \
                patch.object(m.response, 'with_verified_bundle') as numerical:
            with self.assertRaisesRegex(ValueError, 'predeclared'):
                m._replay(data.fit_ref, data.gate_ref, allow_cpu_fixture=True)
            numerical.assert_not_called()
        wrong = m.prefix.save(data.root/'wrong-fit-manifest.json', dict(executor_manifest=dict(path='/foreign', sha256='f'*64)))
        with patch.object(m.response, 'audit_fit', return_value=dict(report=dict(manifest=wrong))), \
                patch.object(m.prefix, 'audit_gate', return_value=data.gate), \
                patch.object(m.fresh, 'audit_plan', return_value=data.plan), \
                patch.object(m.prefix, 'audit_prefix', return_value=data.prefix_doc), \
                patch.object(m.response, 'with_verified_bundle') as numerical:
            with self.assertRaisesRegex(ValueError, 'exact matching campaign'):
                m._replay(data.fit_ref, data.gate_ref, allow_cpu_fixture=True)
            numerical.assert_not_called()


if __name__ == '__main__':
    unittest.main()
