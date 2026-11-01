import copy
import json
from pathlib import Path
import tempfile
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tests.test_generation_work_v4 import observations
from tools.predictor_validation_20261027.generation_work_v4 import input_content_identity
from tools.predictor_validation_20261027 import generation_work_v5 as tool


class ExactContentGenerationTests(unittest.TestCase):
    def provider(self, rows=None):
        model = tool.fit(observations() if rows is None else rows, allow_cpu_fixture=True)
        return tool.ExactContentGenerationWorkProvider(model, 'native',
            allow_exact_content_hypothesis=True, allow_cpu_fixture=True)

    def request(self, content='short'):
        row = next(r for r in observations() if r['input_content_sha256'] == content)
        return dict(family=row['family'], task=row['task'], protocol_sha256=row['protocol_sha256'],
            effective_generation=row['effective_generation'], native_execution_projection_sha256='native',
            input_tokens=31, input_content_sha256=content, request_id='new-request')

    def test_exact_content_scenarios_do_not_mix_other_input_lengths(self):
        provider = self.provider()
        before = copy.deepcopy(provider.document)
        short = provider.scenarios(self.request())
        long = provider.scenarios(self.request('long'))
        self.assertEqual([(s['generated_tokens'], s['weight']) for s in short['scenarios']], [(5, 1.)])
        self.assertEqual([(s['generated_tokens'], s['weight']) for s in long['scenarios']], [(9, 1.)])
        self.assertFalse(short['unseen_input_generalization_claimed'])
        self.assertFalse(short['silent_family_task_fallback'])
        self.assertEqual(before, provider.document)

    def test_single_capture_963_unseen_content_changed_shape_are_unknown(self):
        rows = observations()
        rows.append(dict(rows[0], request_id='963', input_content_sha256='963', actual_input_tokens=963))
        provider = self.provider(rows)
        for request in (dict(self.request(), input_content_sha256='never-seen'),
                        dict(self.request(), input_content_sha256='963', input_tokens=963),
                        dict(self.request(), input_tokens=963)):
            with self.assertRaisesRegex(UnknownEvidence, 'NO_FALLBACK'):
                provider.scenarios(request)

    def test_replica_count_does_not_create_new_capture_or_weight(self):
        rows = observations()
        extra = [dict(rows[0], request_id='replica' + str(i), pool_id='alias' + str(i)) for i in range(20)]
        single_capture = tool.fit(rows[:1] + extra, allow_cpu_fixture=True)
        self.assertEqual(single_capture['status'], 'UNKNOWN')
        self.assertEqual(self.provider(rows).scenarios(self.request()), self.provider(rows + extra).scenarios(self.request()))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            tool.fit(rows + [rows[0]], allow_cpu_fixture=True)

    def test_public_content_hash_uses_content_not_alias_or_answer(self):
        sample = dict(task_type='text', pool_id='alias1', sample=dict(context='ctx', question='q', answers=['answer']))
        other = copy.deepcopy(sample)
        other['pool_id'] = 'alias2'
        other['sample']['answers'] = ['future answer']
        self.assertEqual(input_content_identity(sample), input_content_identity(other))
        other['sample']['question'] = 'different'
        self.assertNotEqual(input_content_identity(sample), input_content_identity(other))

    def test_future_outputs_policy_changes_and_missing_optin_are_rejected(self):
        for field in ('generated_tokens', 'worker_response', 'raw_output', 'semantic_forward_observations'):
            with self.assertRaisesRegex(ValueError, 'future actual'):
                self.provider().scenarios(dict(self.request(), **{field: 'future'}))
        changed = self.request()
        changed['effective_generation'] = dict(changed['effective_generation'], num_beams=3)
        with self.assertRaises(UnknownEvidence):
            self.provider().scenarios(changed)
        with self.assertRaises(ValueError):
            self.provider().scenarios(dict(self.request(), native_execution_projection_sha256='different'))
        with self.assertRaisesRegex(ValueError, 'opt-in'):
            tool.ExactContentGenerationWorkProvider(self.provider().document, 'native', allow_cpu_fixture=True)

    def test_real_style_closure_rejects_nested_source_and_fitted_scenario_edits(self):
        # Source-bound CPU fixture simulates serialized evidence, never GPU work.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def write(name, value):
                path = root / name
                path.write_text(json.dumps(value))
                return tool.receipt(path)
            rows = [dict(r, kind='measured') for r in observations()]
            source = write('observations.json', dict(observations=rows))
            reviewer = root / 'reviewer.py'
            reviewer.write_text('original historical reviewer\n')
            base = write('base.json', dict(schema=tool.BASE_SCHEMA, CPU_fixture=False,
                training_observations=source, sources=[tool.receipt(reviewer)]))
            model = tool.fit_artifact(base['path'], root / 'fit')
            tool.ExactContentGenerationWorkProvider(model, 'native', allow_exact_content_hypothesis=True)
            changed = copy.deepcopy(model)
            changed['profiles'][0]['scenarios'][0]['generated_tokens'] = 999
            with self.assertRaisesRegex(ValueError, 'differs'):
                tool.ExactContentGenerationWorkProvider(changed, 'native', allow_exact_content_hypothesis=True)
            reviewer.write_text('changed nested evaluator\n')
            with self.assertRaisesRegex(ValueError, 'Changed receipt'):
                tool.ExactContentGenerationWorkProvider(model, 'native', allow_exact_content_hypothesis=True)


if __name__ == '__main__':
    unittest.main()
