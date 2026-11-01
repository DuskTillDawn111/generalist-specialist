import copy
import json
from pathlib import Path
import unittest

from tools.predictor_validation_20261027 import twoFIFO_source_rebound_observed_score_v1 as m

SECOND = 10**9


def fixture():
    score = json.loads(Path('artifacts/predictor-twoFIFO-generated-score-design-v1-20261028-cpu01/scoring-plan.json').read_text())
    quality = json.loads(Path('artifacts/predictor-gpu-20261027/natural-quality-training-v2-20261028-two-captures/quality-loss.json').read_text())
    identity = dict(capture_id='training-block-1', attempt_id='training-block-1-shard-1', execution_id='cpu-fixture')
    cases = [dict(case_id=str(i), arrivals=[dict(request_id='request-'+str(i), owner='text#0', task='text',
        pool_id='tydiqa-new-native-02ea54b33b0492f20de11e0f', arrival_ns=0)]) for i in (0, 1)]
    plan = dict(identity, resident_owners=list(score['aliases']), cases=cases)
    starts = [2, 73]
    raw = dict(identity, schema=m.executor.RAW_SCHEMA, backend='native', status='completed',
        shutdown_memory_sampled=False,
        lifecycle_boundaries=[dict(boundary=k, time_ns=v*SECOND) for k, v in
            [('startup_begin', 0), ('priming_end', 1), ('monitoring_end', 144), ('shutdown_begin', 145)]],
        board_samples=[dict(time_ns=i*SECOND, watts=10+2*i, owned_used_mib=1000 if i == 70 else 5) for i in range(145)],
        cases=[dict(case_id=c['case_id'], observation_start_ns=t*SECOND) for c, t in zip(cases, starts)],
        measurement_windows=[dict(case_id=c['case_id'], start_ns=t*SECOND, end_ns=(t+10)*SECOND,
            fixed_drain_deadline_ns=(t+70)*SECOND) for c, t in zip(cases, starts)],
        request_records=[dict(request_id=c['arrivals'][0]['request_id'], status='completed',
            completion_time_ns=(t+1)*SECOND) for c, t in zip(cases, starts)])
    return raw, plan, cases[1], 73*SECOND, score, quality


def execute(data):
    return m.score_case(*data, source_evidence={'basis': 'CPU_FIXTURE_NOT_NATIVE_AUTHENTICATION'})


class ObservedScoreTests(unittest.TestCase):
    def test_prior_drain_peak_and_exact_energy_scope_applied_once(self):
        data = fixture()
        original = copy.deepcopy(data)
        out = execute(data)
        self.assertEqual(data, original)
        resources = out['measured_resources']
        self.assertEqual(resources['startup_through_phase_start_owned_peak_mib'], 1000)
        self.assertEqual(resources['phase_start_through_drain_owned_peak_mib'], 5)
        self.assertEqual(resources['startup_energy_joules'], 24)
        # Integral of 10+2t over [73,83], plus startup/3600 once.
        self.assertEqual(resources['measurement_board_energy_joules'], 1660)
        terms = out['measured_original_score']['terms']
        self.assertAlmostEqual(terms['M'], 1000*data[-2]['memory_safe_factor']/data[-2]['memory_budget_mib'])
        self.assertAlmostEqual(terms['E'], (1660+24*10/3600)/(450*10))
        # The actual original score needs observations for every offered task.
        # This text-only fixture must not invent OCR/image/asr quality/latency.
        self.assertIsNone(out['measured_original_score']['full_U'])
        self.assertIn('NO_COHORT_OBSERVATION_FOR_TASK:image', out['measured_original_score']['unknown_reasons'])
        self.assertFalse(out['actual_native_quality_validated'])

    def test_complete_declared_cohort_has_full_score_without_partial_utility(self):
        data = fixture()
        # Explicit CPU-only single-task workload, keeping the six original
        # equal weights; this does not change the production scoring artifact.
        data[-2]['workload_rps'] = {'text': 1.0}
        data[-2]['latency_reference_ms'] = {'text': 1000}
        out = execute(data)
        score = out['measured_original_score']
        self.assertIsNotNone(score['full_U'])
        self.assertAlmostEqual(score['full_U'], -sum(score['terms'].values())/6)

    def test_drain_power_excluded_but_drain_memory_retained(self):
        data = fixture()
        base = execute(data)
        data[0]['board_samples'][100].update(watts=100_000, owned_used_mib=2000)
        changed = execute(data)
        self.assertEqual(base['measured_original_score']['terms']['E'], changed['measured_original_score']['terms']['E'])
        self.assertEqual(changed['measured_resources']['phase_start_through_drain_owned_peak_mib'], 2000)
        self.assertGreater(changed['measured_original_score']['terms']['M'], base['measured_original_score']['terms']['M'])

    def test_power_gap_remains_unknown_without_losing_offered_request(self):
        data = fixture()
        data[0]['board_samples'] = [s for s in data[0]['board_samples'] if not 76*SECOND <= s['time_ns'] <= 79*SECOND]
        out = execute(data)
        self.assertIsNone(out['measured_original_score']['terms']['E'])
        self.assertIsNone(out['measured_original_score']['full_U'])
        self.assertEqual(out['offered_requests'], 1)

    def test_incomplete_and_foreign_epoch_are_not_silently_removed(self):
        data = fixture()
        data[0]['request_records'][1]['status'] = 'incomplete'
        out = execute(data)
        self.assertEqual(out['measured_original_score']['cohort_size'], 1)
        self.assertEqual(out['measured_original_score']['incomplete_ids'], ['request-1'])
        self.assertIsNone(out['measured_original_score']['full_U'])
        changed = list(fixture())
        changed[3] += 1
        with self.assertRaises(ValueError):
            execute(changed)


if __name__ == '__main__':
    unittest.main()
