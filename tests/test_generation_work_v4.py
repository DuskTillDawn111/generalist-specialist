import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027.generation_work_v4 import (
    GenerationWorkProvider, PREFIX_SCHEMA, actual_work, fit, load_training)
from tools.predictor_validation_20261027.native_collect import receipt
from tools.predictor_validation_20261027.profiling import digest


def response(length=5, beams=1, forwards=None):
    count = length if forwards is None else forwards
    return dict(effective_generation=dict(max_new_tokens=128, num_beams=beams,
        do_sample=False, eos_token_id=[99]), generated_tokens=length,
        finish_reason='eos', terminal_token_id=99, eos_token_ids=[99], input_tokens=31,
        semantic_forward_observations=dict(schema='gsdeploy.semantic_forward_observation.v1',
            request_id='request', instrumentation_errors=[], records=[dict(id='f' + str(i),
                forward_index=i, semantic_phase='prefill' if i == 0 else 'decode',
                input_sequence_length=31 if i == 0 else 1, input_batch_size=beams,
                input_token_positions=31 * beams if i == 0 else beams,
                end_basis='always_call_forward_hook') for i in range(count)]))


def observations(beams=1):
    rows = []
    for capture in range(2):
        for content, length in (('short', 5), ('long', 9)):
            work = actual_work(response(length, beams), 'request')
            rows.append(dict(kind='cpu_fixture', split='training', source='capture' + str(capture),
                request_id=content, alias='text#0', family='text', task='text',
                pool_id=content, input_content_sha256=content, execution_context_sha256='full' + str(capture),
                native_execution_projection_sha256='native', status='OBSERVED_GENERATION_WORK',
                protocol_sha256='protocol', effective_generation_sha256=digest(work['effective_generation']), **work))
    return rows


class GenerationWorkTests(unittest.TestCase):
    def model(self, rows=None):
        return fit(observations() if rows is None else rows, allow_cpu_fixture=True)

    def provider(self, rows=None):
        return GenerationWorkProvider(self.model(rows), 'native',
            allow_empirical_length_hypothesis=True, allow_cpu_fixture=True)

    def request(self, beams=1):
        return dict(family='text', task='text', native_execution_projection_sha256='native',
            protocol_sha256='protocol', effective_generation=response(beams=beams)['effective_generation'],
            input_tokens=963, request_id='runtime-new-request')

    def prefix(self, directory, token_ids, *, time=50):
        path = Path(directory) / 'prefix.json'
        path.write_text(json.dumps(dict(schema=PREFIX_SCHEMA, request_id='runtime-new-request',
            protocol_sha256='protocol', native_execution_projection_sha256='native',
            effective_generation_sha256=digest(response()['effective_generation']),
            observation_time_ns=time, status='OBSERVED_PREFIX',
            clock='host_perf_counter_ns',
            token_observations=[dict(token_id=t, observed_time_ns=i + 1) for i, t in enumerate(token_ids)])))
        return receipt(path)

    def test_real_work_uses_eos_and_actual_steps_not_policy_maximum(self):
        work = actual_work(response(), 'request')
        self.assertEqual((work['generated_tokens'], work['forward_count'], work['decode_forward_count']), (5, 5, 4))
        for mutate in (lambda r: r.update(terminal_token_id=1),
                       lambda r: r.update(generated_tokens=128),
                       lambda r: r.update(finish_reason='length')):
            bad = response()
            mutate(bad)
            with self.assertRaises(ValueError):
                actual_work(bad, 'request')

    def test_joint_beam_work_preserves_internal_batch_and_extra_search(self):
        work = actual_work(response(4, 3, forwards=6), 'request')
        self.assertEqual((work['generated_tokens'], work['decode_forward_count'], work['forward_batch_size']), (4, 5, 3))
        self.assertEqual(work['selected_token_forward_relation'], 'beam_search_extra_steps')
        bad = response(4, 3)
        bad['semantic_forward_observations']['records'][1]['input_batch_size'] = 1
        with self.assertRaises(ValueError):
            actual_work(bad, 'request')
        with self.assertRaises(ValueError):
            actual_work(response(4, 1, forwards=6), 'request')

    def test_replicas_cannot_create_capture_or_distinct_content_support(self):
        for rows in (observations()[:2], [r for r in observations() if r['pool_id'] == 'short']):
            self.assertEqual(self.model(rows)['status'], 'UNKNOWN')
        replicas = [dict(observations()[0], request_id='replica' + str(i), pool_id='alias' + str(i)) for i in range(20)]
        self.assertEqual(self.model(replicas)['status'], 'UNKNOWN')
        original = self.model()['profiles'][0]['scenarios']
        extra = [dict(observations()[0], request_id='extra' + str(i)) for i in range(20)]
        self.assertEqual(original, self.model(observations() + extra)['profiles'][0]['scenarios'])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.model(observations() + [observations()[0]])

    def test_unseen_input_gets_frozen_joint_scenarios_without_final_output(self):
        provider = self.provider()
        before = copy.deepcopy(provider.document)
        result = provider.scenarios(self.request())
        self.assertEqual([(s['generated_tokens'], s['weight']) for s in result['scenarios']], [(5, .5), (9, .5)])
        self.assertEqual(result['scenarios'][0]['prefill']['input_sequence_length'], 963)
        self.assertFalse(result['future_actual_output_used'])
        self.assertEqual(before, provider.document)
        for key in ('generated_tokens', 'worker_response', 'raw_output', 'finish_reason', 'semantic_forward_observations'):
            with self.assertRaisesRegex(ValueError, 'future actual'):
                provider.scenarios(dict(self.request(), **{key: 'future'}))
        with self.assertRaises(UnknownEvidence):
            provider.scenarios(dict(self.request(), protocol_sha256='other'))
        with self.assertRaises(ValueError):
            provider.scenarios(dict(self.request(), native_execution_projection_sha256='other'))

    def test_observed_prefix_conditions_survival_and_never_extrapolates(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.provider().scenarios(self.request(), observed_prefix=self.prefix(directory, [1] * 5), prediction_time_ns=50)
            self.assertEqual([(s['generated_tokens'], s['weight'], s['remaining_forward_count'])
                for s in result['scenarios']], [(9, 1., 4)])
            self.assertEqual(result['scenarios'][0]['remaining_prefill_forward_count'], 0)
            with self.assertRaises(UnknownEvidence):
                self.provider().scenarios(self.request(), observed_prefix=self.prefix(directory, [1] * 9), prediction_time_ns=50)
            with self.assertRaisesRegex(ValueError, 'Future'):
                self.provider().scenarios(self.request(), observed_prefix=self.prefix(directory, [1], time=51), prediction_time_ns=50)
            result = self.provider().scenarios(self.request(), observed_prefix=self.prefix(directory, [1, 99]), prediction_time_ns=50)
            self.assertTrue(result['completion_already_observed'])
            self.assertEqual(result['scenarios'][0]['remaining_forward_count'], 0)
            with self.assertRaisesRegex(ValueError, 'after observed EOS'):
                self.provider().scenarios(self.request(), observed_prefix=self.prefix(directory, [99, 1]), prediction_time_ns=50)

    def test_policy_fixture_optins_and_beam_prefix_are_explicit(self):
        with self.assertRaises(ValueError):
            fit(observations())
        with self.assertRaisesRegex(ValueError, 'opt-in'):
            GenerationWorkProvider(self.model(), 'native', allow_cpu_fixture=True)
        unbound = self.model()
        unbound['CPU_fixture'] = False
        with self.assertRaisesRegex(ValueError, 'provenance'):
            GenerationWorkProvider(unbound, 'native', allow_empirical_length_hypothesis=True)
        beam_rows = observations(3)
        provider = self.provider(beam_rows)
        result = provider.scenarios(self.request(3))
        self.assertEqual(result['scenarios'][0]['decode']['input_batch_size'], 3)
        with tempfile.TemporaryDirectory() as directory:
            prefix_ref = self.prefix(directory, [1])
            prefix = json.loads(Path(prefix_ref['path']).read_text())
            prefix['effective_generation_sha256'] = digest(response(beams=3)['effective_generation'])
            Path(prefix_ref['path']).write_text(json.dumps(prefix))
            with self.assertRaises(UnknownEvidence):
                provider.scenarios(self.request(3), observed_prefix=receipt(prefix_ref['path']), prediction_time_ns=50)

    def test_non_autoregressive_ocr_never_becomes_a_generation_scenario(self):
        ocr = dict(kind='cpu_fixture', split='training', source='capture0', request_id='ocr',
                   task='ocr', status='UNKNOWN_NON_AUTOREGRESSIVE_OCR')
        model = self.model(observations() + [ocr])
        self.assertEqual(model['non_autoregressive_ocr']['observations'], 1)
        self.assertEqual(len(model['profiles']), 1)
        with self.assertRaises(UnknownEvidence):
            self.provider(observations() + [ocr]).scenarios(dict(self.request(), family='ocr', task='ocr'))

    def test_loader_uses_measured_not_priming_and_rejects_changed_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def write(name, value):
                path = root / name
                path.write_text(json.dumps(value))
                return receipt(path)
            pool = root / 'runtime-inputs.jsonl'
            pool.write_text(json.dumps(dict(pool_id='input', task_type='text',
                sample=dict(context='public', question='question'))))
            context = write('context.json', dict(execution_sources=[dict(original_path='/runtime-inputs.jsonl', frozen=receipt(pool))]))
            plan = write('plan.json', dict(domain='natural', split='training', execution_context_sha256='full',
                jobs=[dict(id='job', split='training', inputs={'text#0': 'input'}, tasks={'text#0': 'text'})]))
            w = response()
            w.update(task='text', protocol_sha256='protocol', request_sha256='requesthash')
            row = dict(profile_job_id='job', request_id='request', model='text#0', pool_id='input',
                       task='text', split='training', status='completed', worker_response=w)
            raw = write('raw.json', dict(kind='measured', status='completed', execution_context_sha256='full',
                request_records=[row], priming_request_records=[dict(split='validation', worker_response={'future': 'ignored'})]))
            evaluator = write('evaluator.json', {'unchanged': True})
            review = write('report.json', dict(schema='gsdeploy.startup_training_collection_strict_review.v4',
                status='PASS', issues=[], training_only=True, raw=raw, plan=plan, context=context,
                evaluator_sources=[evaluator], capture_run_id='raw-sha256:' + raw['sha256']))
            projection = dict(original_full_context_sha256='full', measurement_domain='natural', projection_sha256='native')
            with patch('tools.predictor_validation_20261027.generation_work_v4.native_execution_projection', return_value=projection):
                rows, evidence = load_training(review['path'])
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]['generated_tokens'], 5)
                self.assertEqual(evidence['projection'], projection)
                Path(evaluator['path']).write_text('changed')
                with self.assertRaisesRegex(ValueError, 'receipt'):
                    load_training(review['path'])


if __name__ == '__main__':
    unittest.main()
