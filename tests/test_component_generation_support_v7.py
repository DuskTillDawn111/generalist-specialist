"""Exact component identity, schema dispatch and unsupported work mass."""
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027 import component_generation_support_v7 as tool
from tools.predictor_validation_20261027.native_collect import receipt, save


class ComponentSupportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def ref(self, name, value):
        path = self.root / name; save(path, value); return receipt(path)

    def test_dispatches_only_exact_review_schemas_without_relabeling(self):
        for schema in tool.REVIEW_LOADERS:
            ref = self.ref(schema + '.json', dict(schema=schema))
            triple = (dict(training_only=True, issues=[]),
                      dict(jobs=[dict(split='training')]), dict(kind='measured', status='completed'))
            with patch.dict(tool.REVIEW_LOADERS, {schema: lambda path, domain: triple}):
                self.assertEqual(tool.load_light_review(ref), triple)
        ref = self.ref('wrong.json', dict(schema='gsdeploy.training_collection_strict_review.v2'))
        with self.assertRaisesRegex(ValueError, 'Exact v3'):
            tool.load_light_review(ref)

    def provider(self):
        key = ('generalist_2#0', 'image', 'content')
        template = dict(dag=dict(path='original-dag', sha256='original'), alias=key[0], task=key[1], pool_id=key[2])
        policy = dict(max_new_tokens=4)
        work = dict(generated_tokens=4, finish_reason='length', forward_count=4,
            prefill_forward_count=1, decode_forward_count=3, forward_batch_size=3,
            selected_token_forward_relation='one_forward_per_selected_token')
        binding = dict(component_id='modality', alias=key[0], task=key[1], pool_id=key[2], family='generalist_2',
            input_tokens=100, protocol_sha256='protocol', effective_generation=policy,
            input_content_sha256='actual-content', native_execution_projection_sha256='natural',
            source_sqlite=dict(path='original-sqlite', sha256='sqlite'), work=work, template=template)
        plan = dict(alias=key[0], task=key[1], pool_id=key[2], source_sqlite=binding['source_sqlite'])
        adapter = SimpleNamespace(plans={key: plan}, routes={key:'modality'},
            component=lambda row: dict(entry=dict(spec=dict(dag=template['dag']))))
        provider = object.__new__(tool.ComponentGenerationSupport)
        provider.adapter = adapter; provider.public_bindings = {key: copy.deepcopy(binding)}
        provider.templates = {'modality':[copy.deepcopy(binding)]}
        scenario = dict(work, weight=1.)
        provider.providers = {'natural':SimpleNamespace(scenarios=lambda public: dict(scenarios=[copy.deepcopy(scenario)]))}
        return provider, dict(request_id='new-request',alias=key[0],task=key[1],pool_id=key[2]), scenario

    def test_exact_selected_dag_is_retained_and_wrong_selected_template_rejected(self):
        provider, row, _ = self.provider()
        result = provider.check([row])
        selected = result['rows'][0]['selected_template']
        self.assertEqual(selected['dag']['path'], 'original-dag')
        for field, value in [('dag',dict(path='other-dag',sha256='same-bytes')), ('alias','other'), ('pool_id','other')]:
            provider, row, _ = self.provider()
            provider.templates['modality'][0]['template'][field] = value
            with self.assertRaisesRegex(ValueError, 'Selected generation template'):
                provider.check([row])

    def test_positive_unknown_mass_is_not_dropped_or_conditioned_away(self):
        provider, row, scenario = self.provider()
        known = dict(scenario, weight=.75); unknown = dict(scenario, generated_tokens=2, forward_count=2,
            decode_forward_count=1, weight=.25)
        provider.providers['natural'].scenarios = lambda public: dict(scenarios=[known, unknown])
        with self.assertRaisesRegex(UnknownEvidence, 'COMPLETE_SINGLE'):
            provider.check([row])

    def test_changed_executed_owner_input_or_sqlite_rejected(self):
        for field, value in [('alias','wrong'), ('pool_id','wrong'), ('source_sqlite',dict(path='other',sha256='sqlite'))]:
            provider, row, _ = self.provider(); plan = next(iter(provider.adapter.plans.values()))
            plan[field] = value
            with self.assertRaisesRegex(ValueError, 'Selected generation DAG'):
                provider.check([row])

    def test_changed_support_descriptors_and_equal_bytes_other_catalog_rejected(self):
        # Exercise constructor rederivation, rather than trusting saved features.
        provider, _, _ = self.provider(); adapter = provider.adapter
        adapter.execution_context_sha256 = 'composite'
        adapter.components = {'modality':dict(entry=dict(spec=dict(id='modality')))}
        contract = dict(composite_execution_sha256='composite', components=[adapter.components['modality']['entry']])
        composite = self.ref('composite.json', contract)
        expected = [dict(component_id='modality', descriptors=list(provider.public_bindings.values()),
                         public_input_catalog=dict(path='original-catalog',sha256='same'))]
        document = dict(schema=tool.SCHEMA, training_only=True, future_actual_output_used=False,
            implementation_sources=tool.implementation_sources(), composite=composite, light_reviews={},
            components=copy.deepcopy(expected))
        for mutate in ('input_tokens', 'catalog_path'):
            changed = copy.deepcopy(document)
            if mutate == 'input_tokens': changed['components'][0]['descriptors'][0]['input_tokens'] += 1
            else: changed['components'][0]['public_input_catalog']['path'] = 'different-same-byte-catalog'
            with patch.object(tool, 'derive_components', return_value=expected), \
                    self.assertRaisesRegex(ValueError, 'rederived original training'):
                tool.ComponentGenerationSupport(changed, adapter, allow_exact_content_generation_hypothesis=True)


if __name__ == '__main__':
    unittest.main()
