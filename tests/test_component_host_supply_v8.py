import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027 import component_host_supply_v8 as tool
from tools.predictor_validation_20261027.forecast_gate_multi_native_v7 import execute_bundle,BUNDLE_SCHEMA
from tools.predictor_validation_20261027.native_collect import receipt,save


class ComponentHostSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.seq=0

    def ref(self,value):
        self.seq+=1;p=self.root/(str(self.seq)+'.json');save(p,value);return receipt(p)

    def fixture(self):
        key1=('a','text','input');key2=('g','image','image-input')
        dag1={'path':'textdag','sha256':'text'};dag2={'path':'imagedag','sha256':'image'}
        old1=self.ref(dict(schema='gsdeploy.dag_host_supply_training_hypothesis.v5',dag=dag1,execution_context_sha256='natural1'))
        old2=self.ref(dict(schema='gsdeploy.dag_host_supply_training_hypothesis.v5',dag=dag2,execution_context_sha256='natural2'))
        exact=self.ref(dict(schema=tool.EXACT_SCHEMA,source_host=old1,dag=dag1,light_execution_context_sha256='light1',
            execution_context_sha256='natural1',groups=[dict(alias='a',task='text',pool_id='input',supported=True)]))
        original_calls=[]
        def oldrelease(node,row):original_calls.append(row['alias']);return dict(not_before_offset_ns=9,after_predecessors_ns=0)
        def entry(cid,dag,model):return dict(spec=dict(id=cid,dag=dag,host_supply=model),
            original_light_context_sha256='light'+('1' if cid=='text' else '2'),
            original_natural_context_sha256='natural'+('1' if cid=='text' else '2'))
        adapter=SimpleNamespace(execution_context_sha256='composite',plans={key1:dict(id='text'),key2:dict(id='image')},
            routes={key1:'text',key2:'image'},components={
            'text':dict(entry=entry('text',dag1,old1),release=oldrelease),
            'image':dict(entry=entry('image',dag2,old2),release=oldrelease)})
        binding={key1:dict(component_id='text'),key2:dict(component_id='image')}
        document=dict(schema=tool.SCHEMA,composite=self.ref(dict(composite_execution_sha256='composite')),
            components=[dict(component_id='text',provider_kind='exact_input_v7',model=exact,
                allow_identification_hypothesis=True,allow_exact_input_conditioning_hypothesis=True),
                dict(component_id='image',provider_kind='original_component_host_v5',model=old2,
                allow_identification_hypothesis=True,allow_cross_input_pooling_hypothesis=True)])
        return adapter,binding,document,original_calls

    def provider(self,adapter,binding,document,exact=None):
        exact=exact or (lambda node,row:dict(not_before_offset_ns=7,after_predecessors_ns=0))
        with patch.object(tool,'DagHostReleaseProvider',return_value=exact) as constructor:
            provider=tool.ComponentHostSupply(document,adapter,{},binding,scenario='host_issue_depth_supply',expected_composite=document['composite'])
            self.assertEqual(constructor.call_args.args[1],'natural1')
            self.assertEqual(set(constructor.call_args.args[2]),{('a','text','input')})
            self.assertEqual(set(constructor.call_args.kwargs['public_bindings']),{('a','text','input')})
        return provider

    def test_exact_component_dispatch_and_original_provenance_are_separate(self):
        adapter,bindings,doc,calls=self.fixture();provider=self.provider(adapter,bindings,doc)
        text=dict(alias='a',task='text',pool_id='input');image=dict(alias='g',task='image',pool_id='image-input')
        self.assertEqual(provider({},text)['not_before_offset_ns'],7)
        self.assertEqual(provider({},image)['not_before_offset_ns'],9)
        self.assertEqual(calls,['g'])
        self.assertEqual(provider.provenance(text)['original_natural_context_sha256'],'natural1')
        self.assertEqual(provider.provenance(image)['original_natural_context_sha256'],'natural2')

    def test_unknown_never_falls_back_to_original_provider(self):
        adapter,bindings,doc,calls=self.fixture()
        def unknown(*args):raise UnknownEvidence('EXACT_MODEL_MISSING')
        provider=self.provider(adapter,bindings,doc,unknown)
        with self.assertRaisesRegex(UnknownEvidence,'EXACT_MODEL_MISSING'):
            provider({},dict(alias='a',task='text',pool_id='input'))
        self.assertEqual(calls,[])
        with self.assertRaisesRegex(UnknownEvidence,'NO_EXACT_COMPONENT'):
            provider({},dict(alias='a',task='text',pool_id='unseen'))

    def test_missing_duplicate_or_foreign_component_mapping_is_rejected(self):
        adapter,bindings,doc,calls=self.fixture()
        for components in (doc['components'][:1],[doc['components'][0]]*2,
                           [dict(doc['components'][0],component_id='foreign'),doc['components'][1]]):
            with self.subTest(components=components),self.assertRaisesRegex(ValueError,'complete nonduplicate'):
                tool.ComponentHostSupply(dict(doc,components=components),adapter,{},bindings,scenario='host_issue_depth_supply',expected_composite=doc['composite'])

    def test_original_model_receipt_and_exact_component_source_cannot_be_swapped(self):
        adapter,bindings,doc,calls=self.fixture()
        for index in (0,1):
            changed=copy.deepcopy(doc);changed['components'][index]['model']=doc['components'][1-index]['model']
            with patch.object(tool,'DagHostReleaseProvider'),self.assertRaises(ValueError):
                tool.ComponentHostSupply(changed,adapter,{},bindings,scenario='host_issue_depth_supply',expected_composite=doc['composite'])
        changed=copy.deepcopy(bindings);changed[('g','image','image-input')]['component_id']='text'
        with patch.object(tool,'DagHostReleaseProvider'),self.assertRaisesRegex(ValueError,'public binding'):
            tool.ComponentHostSupply(doc,adapter,{},changed,scenario='host_issue_depth_supply',expected_composite=doc['composite'])

    def test_incomplete_exact_support_is_unknown_before_any_fallback(self):
        adapter,bindings,doc,calls=self.fixture();model=tool.checked(doc['components'][0]['model'])
        model['groups'][0]['supported']=False;doc['components'][0]['model']=self.ref(model)
        with self.assertRaisesRegex(UnknownEvidence,'NOT_COMPLETELY_SUPPORTED'):
            tool.ComponentHostSupply(doc,adapter,{},bindings,scenario='host_issue_depth_supply',expected_composite=doc['composite'])
        self.assertEqual(calls,[])

    def test_resigned_other_composite_retaining_id_is_rejected(self):
        adapter,bindings,doc,calls=self.fixture();expected=copy.deepcopy(doc['composite'])
        doc['composite']=self.ref(dict(composite_execution_sha256='composite',forged='extra'))
        with self.assertRaisesRegex(ValueError,'unchanged multi-native'):
            tool.ComponentHostSupply(doc,adapter,{},bindings,scenario='host_issue_depth_supply',expected_composite=expected)
        self.assertEqual(calls,[])

    def test_new_gate_rejects_legacy_global_override_field(self):
        with self.assertRaisesRegex(ValueError,'Unsupported multi-native'):
            execute_bundle(dict(schema=BUNDLE_SCHEMA,kind='trained_hypothesis',host_supply={}),{},
                           allow_trained_hypothesis=True,allow_training_diagnostic=True)

if __name__=='__main__':unittest.main()
