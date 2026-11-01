"""Independent full-window CPU replay and tamper checks; no GPU calls."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.queue_coupled_utility_20261023 import native_runner, evaluation
from tools.queue_refined_20261023 import protocol as legacy_protocol


class _Provider:
    def service(self, row, active, aliases, scenario):
        return dict(known=True, context_supported=True, service_ms=1.+len(active))

    def residual(self, row, age, active, aliases, scenario):
        return dict(known=True, tail_supported=True, remaining_ms=1.)

    def power(self, active, aliases, scenario):
        return dict(known=True, watts=20.+10*len(active))


class IndependentRuntimeReview(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        asset = self.root / 'asset'; asset.write_text('cpu')
        self.samples = self.root / 'pool.jsonl'
        rows = [dict(pool_id='text', task_type='text', sample=dict(context='c', question='q', label='ok'))]
        rows += [dict(pool_id=t, task_type=t, sample=dict(input=str(asset), label='ok')) for t in ('image', 'asr', 'ocr')]
        self.samples.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        models = {'text': dict(supported_tasks=['text'], service_time_ms=1.),
                  'generalist_2': dict(supported_tasks=['text', 'image', 'asr', 'ocr'], service_time_ms=1.),
                  'ocr': dict(supported_tasks=['ocr'], service_time_ms=1.)}
        config = dict(catalog=dict(models=models, safe_factor=1.1,
                                  memory_table_mib={m: {'1': 10., '2': 20.} for m in models}),
                      workload={t: 100. for t in ('text', 'image', 'asr', 'ocr')},
                      weights={t: 1/6 for t in 'QTLPME'},
                      latency_reference_ms={t: 100. for t in ('text', 'image', 'asr', 'ocr')}, gpu_memory_mib=1000.)
        config['catalog']['quality'] = dict(identity=dict(input_pool_sha256=legacy_protocol.receipt(self.samples)['sha256']))
        self.system = SimpleNamespace(**config, as_dict=lambda: copy.deepcopy(config))
        services = {m: {t: {t: 1.} for t in spec['supported_tasks']} for m, spec in models.items()}
        self.contract = legacy_protocol.make_policy_contract({'text': 2, 'generalist_2': 1, 'ocr': 1}, services,
            {m: {t: .1 for t in spec['supported_tasks']} for m, spec in models.items()}, config['weights'],
            config['latency_reference_ms'], strategy='queue_quality', models=models, text_rate=100.,
            background_routing={t: {'ocr' if t == 'ocr' else 'generalist_2': 100.} for t in ('image', 'asr', 'ocr')})
        model = self.root / 'resources.json'; model.write_text('{}')
        self.contract['coupled'] = dict(resource_model=dict(path=str(model)), allow_spill=True,
            params=dict(scenarios=['mean'], exposure_rate_rps=12.5, exposure_burst=1., cooldown_ms=0.))

    def run_mock(self):
        # This fixture isolates runtime/replay rather than resource model fitting.
        # The real dispatcher, policy, window, worker transport and old cohort
        # audit execute; provider and formal native evaluation are test inputs.
        with patch('subprocess.run', side_effect=AssertionError('No GPU/system calls')), \
             patch('tools.queue_coupled_utility_20261023.protocol.validate_contract'), \
             patch('tools.queue_coupled_utility_20261023.resources.load_provider', return_value=_Provider()), \
             patch.object(native_runner, 'evaluate_run', return_value=dict(utility=dict(U=None, eligible=False))):
            raw = native_runner.run(self.system, legacy_protocol.nominal_plan(self.contract), self.samples,
                self.contract, backend='mock', duration=.12, warmup=.08, seed=303,
                arrival_process='deterministic_periodic', startup_timeout=10., drain_timeout=10.)
        self.assertEqual(raw['status'], 'completed', raw['errors'])
        return raw

    def audit(self, raw, phase='measurement'):
        with patch('tools.queue_coupled_utility_20261023.protocol.validate_contract'), \
             patch('tools.queue_coupled_utility_20261023.resources.load_provider', return_value=_Provider()):
            return evaluation.audit_phase(self.system, raw, phase)

    def test_actual_cpu_runtime_replays_all_current_arrivals_and_completion_feedback(self):
        raw = self.run_mock()
        self.assertTrue(self.audit(raw)['verified'], self.audit(raw))
        self.assertTrue(self.audit(raw, 'warmup')['verified'], self.audit(raw, 'warmup'))
        self.assertEqual({r['task'] for r in raw['request_records']}, set(self.system.workload))
        self.assertTrue(any(r['routing_decision']['newly_observed_completions'] for r in raw['request_records']))
        for phase in ('measurement', 'warmup'):
            rows = raw['request_records' if phase == 'measurement' else 'warmup_request_records']
            self.assertEqual(rows[0]['routing_decision']['newly_observed_completions'], [])
        self.assertIsNone(raw['evaluation']['utility']['U'])

    def test_exactly_bound_feedback_rejects_fabricated_service_or_private_fields(self):
        raw = self.run_mock()
        index = next(i for i, r in enumerate(raw['request_records']) if r['routing_decision']['newly_observed_completions'])
        for field, value in [('worker_service_ms', 9999.), ('answer', 'PRIVATE')]:
            changed = copy.deepcopy(raw)
            changed['request_records'][index]['routing_decision']['newly_observed_completions'][0][field] = value
            result = self.audit(changed)
            self.assertFalse(result['verified'])
            self.assertIn('FEEDBACK_ALLOWLIST_OR_IDENTITY_MISMATCH', result['errors'][0])

    def test_legacy_dispatcher_still_replays_with_new_feedback_window(self):
        self.contract.pop('coupled')
        raw = self.run_mock()
        result = self.audit(raw)
        self.assertTrue(result['verified'], result)
        rows = raw['request_records']
        self.assertTrue(all(r['routing_decision']['decision']['status'] == 'FROZEN_SWRR_BACKGROUND'
                            for r in rows if r['task'] != 'text'))


if __name__ == '__main__':
    unittest.main()
