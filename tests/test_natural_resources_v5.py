"""Synthetic CPU telemetry checks resource windows and unsupported states."""
import copy
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import natural_resources_v5 as tool

ALIASES = {a: dict(family=f) for a, f in tool.RESIDENTS.items()}


def fixture():
    samples = [dict(time_ns=i*500_000_000, watts=100., owned_used_mib=1000.+i,
                    owned_aliases=list(ALIASES) if 4 <= i <= 16 else [], foreign_pids=[])
               for i in range(21)]
    rows = [dict(request_id=alias, model=alias, task='text', split='training', status='completed',
                 worker_response=dict(start_perf_ns=a*1_000_000_000, end_perf_ns=b*1_000_000_000))
            for alias, a, b in [('text#0', 3, 5), ('generalist_2#0', 4, 6)]]
    return dict(kind='measured', backend='native', status='completed', request_records=rows,
                observation_start_ns=2_000_000_000, observation_end_ns=8_000_000_000,
                board_samples=samples)


class NaturalResourceTests(unittest.TestCase):
    def test_active_window_excludes_startup_and_preserves_full_concurrent_states(self):
        result = tool.capture_intervals(fixture(), ALIASES)
        self.assertEqual(sum(r['board_energy_joules'] for r in result), 600.)
        self.assertEqual(sum(r['duration_ns'] for r in result), 6_000_000_000)
        both = [r for r in result if len(r['members']) == 2]
        self.assertEqual(len(both), 1)
        self.assertEqual(both[0]['sampled_owned_peak_mib'], 1009.)

    def test_short_state_without_actual_sample_is_not_filled_from_neighbors(self):
        raw = fixture()
        raw['request_records'][0]['worker_response'].update(start_perf_ns=3_010_000_000, end_perf_ns=3_020_000_000)
        short = [r for r in tool.capture_intervals(raw, ALIASES) if r['duration_ns'] == 10_000_000][0]
        self.assertEqual(short['interior_sample_count'], 0)
        self.assertIsNone(short['sampled_owned_peak_mib'])

    def test_training_and_ownership_and_order_are_required(self):
        for mutate in (
            lambda r: r['request_records'][0].update(split='validation'),
            lambda r: r['board_samples'][8].update(foreign_pids=[999]),
            lambda r: r['board_samples'][8].update(owned_aliases=['text#0']),
            lambda r: r['board_samples'].reverse(),
        ):
            raw = fixture()
            mutate(raw)
            with self.assertRaises(ValueError):
                tool.capture_intervals(raw, ALIASES)

    def test_same_owner_overlapping_handlers_rejected(self):
        raw = fixture()
        row = copy.deepcopy(raw['request_records'][0])
        row['request_id'] = 'other'
        raw['request_records'].append(row)
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            tool.capture_intervals(raw, ALIASES)

    def test_power_and_memory_require_direct_two_capture_state(self):
        key, members = tool.state_key({}, ALIASES)
        model = dict(schema=tool.SCHEMA, provenance=dict(kind='measured', split='training'),
                     sources=[tool.receipt(__file__)],
                     native_projection_sha256='native', states={key: dict(members=members, known=True,
                         independent_capture_count=2, scenarios=dict(mean=dict(watts=80., peak_memory_mib=30000.)))})
        with patch.object(tool, 'native_execution_projection', return_value=dict(
                projection_sha256='native', original_full_context_sha256='original')):
            provider = tool.NaturalResourceProvider(model, {}, allow_conditional_resource_hypothesis=True)
        self.assertEqual(provider.power({}, ALIASES, 'profiled')['watts'], 80.)
        self.assertEqual(provider.memory({}, ALIASES, 'profiled')['peak_memory_mib'], 30000.)
        self.assertFalse(provider.memory({}, ALIASES, 'profiled')['safe_factor_applied'])
        self.assertFalse(provider.power({'text#0': {'task': 'text'}}, ALIASES, 'profiled')['known'])
        provider.document['states'][key]['independent_capture_count'] = 1
        self.assertFalse(provider.memory({}, ALIASES, 'profiled')['known'])


if __name__ == '__main__':
    unittest.main()
