import unittest

from experiments.analyze_activation_trim_episode_energy import describe


def fixture():
    def row(arrival, start, end, ready, completion):
        return dict(scheduled_time_ns=int(arrival*1e9), completion_time_ns=int(completion*1e9),
            worker_response=dict(start_perf_ns=int(start*1e9), end_perf_ns=int(end*1e9)),
            previous_handler_next_request_ready=dict(perf_ns=int(ready*1e9)))
    prime = row(2, 2, 3, 3.8, 4)
    prime['previous_handler_next_request_ready']['postprime_trim_receipt'] = dict(
        procedure_start_ns=3_100_000_000, procedure_end_ns=3_500_000_000)
    return dict(run_id='fixture', startup_begin_time_ns=0, observation_start_ns=5_000_000_000,
        observation_end_ns=10_000_000_000, priming_request_records=[prime],
        request_records=[row(6, 6, 8, 8.5, 9)], telemetry_errors=[],
        board_samples=[dict(time_ns=i*100_000_000, watts=100., foreign_pids=[]) for i in range(101)])


class TrimEpisodeEnergyTest(unittest.TestCase):
    def test_startup_priming_and_parent_full_drain_are_conserved(self):
        result = describe(fixture())
        self.assertTrue(result['coverage_valid'])
        parts = {p['window']: p for p in result['partitions']}
        self.assertEqual(parts['all_priming_through_full_ready']['end_ns'], 4_000_000_000)
        self.assertEqual(parts['formal_window_through_full_drain']['end_ns'], 9_000_000_000)
        self.assertAlmostEqual(result['totals'][0]['joules'], 500.)
        self.assertAlmostEqual(result['totals'][-1]['joules'], 1000.)
        self.assertAlmostEqual(sum(p['joules'] for p in result['partitions']), 1000.)

    def test_uncovered_energy_is_missing_never_zero(self):
        raw = fixture()
        raw['board_samples'] = raw['board_samples'][:20] + raw['board_samples'][80:]
        result = describe(raw)
        self.assertFalse(result['coverage_valid'])
        self.assertIsNone(result['totals'][-1]['joules'])
        self.assertEqual(len(result['coverage_gaps']), 1)

    def test_unaccounted_trim_and_late_parent_drain_are_rejected(self):
        raw = fixture()
        raw['priming_request_records'][0]['previous_handler_next_request_ready']['postprime_trim_receipt']['procedure_end_ns'] = 5_000_000_000
        with self.assertRaises(ValueError):
            describe(raw)
        raw = fixture()
        raw['request_records'][0]['completion_time_ns'] = 11_000_000_000
        with self.assertRaises(ValueError):
            describe(raw)


if __name__ == '__main__':
    unittest.main()
