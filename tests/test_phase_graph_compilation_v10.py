"""The execution graph keeps original topology and source identity."""
import copy
import unittest
from unittest.mock import patch

from gsdeploy.resources import identity
from tools.predictor_validation_20261027.background_collect import context_identity
from tools.predictor_validation_20261027.native_queue_phase_busy_v10 import compile_original_graph,NativeQueuePhaseFactory


class GraphCompilationTests(unittest.TestCase):
    def test_preserves_every_node_kind_edge_and_default_stream_boundary(self):
        original=dict(nodes=[dict(id='copy',kind='copy',unused_training_metadata=['large']*100),
            dict(id='event',kind='event_record'),dict(id='gpu',kind='gpu',duration_ns=999)],
            edges=[dict(before='copy',after='event',reason='event'),dict(before='event',after='gpu')],
            cyclic=False,unresolved=[],default_stream_semantics='EXPLICIT_SOURCE_HYPOTHESIS')
        before=copy.deepcopy(original);key=('owner','text','public-input')
        compiled,manifest=compile_original_graph({key:original})
        self.assertEqual(original,before)
        self.assertEqual(compiled[key]['nodes'],[dict(id=n['id'],kind=n['kind']) for n in original['nodes']])
        self.assertEqual(compiled[key]['edges'],[dict(before=e['before'],after=e['after']) for e in original['edges']])
        self.assertEqual(compiled[key]['default_stream_semantics'],original['default_stream_semantics'])
        self.assertEqual(manifest[0]['original_node_count'],3);self.assertEqual(manifest[0]['original_edge_count'],2)
        self.assertFalse(manifest[0]['nodes_or_edges_dropped'])
        self.assertNotIn('duration_ns',compiled[key]['nodes'][-1])
        original['nodes'][-1]['duration_ns']=1
        changed,_=compile_original_graph({key:original})
        self.assertNotEqual(changed[key]['original_training_profile_sha256'],compiled[key]['original_training_profile_sha256'])

    def test_invalid_original_topology_is_rejected(self):
        key=('owner','text','public-input')
        with self.assertRaisesRegex(ValueError,'Distinct original'):
            compile_original_graph({key:dict(nodes=[dict(id='x',kind='gpu'),dict(id='x',kind='copy')],edges=[])})
        with self.assertRaisesRegex(ValueError,'foreign endpoint'):
            compile_original_graph({key:dict(nodes=[dict(id='x',kind='gpu')],edges=[dict(before='x',after='missing')])})

    def test_factory_uses_native_context_identity_before_expensive_provider_initialization(self):
        context=dict(device='gpu',sources=['source path'],execution_sources=[
            dict(original_path='runtime-inputs.jsonl',frozen=dict(path='/frozen/source',sha256='a'*64))])
        support=dict(execution_context_sha256=context_identity(context))
        self.assertNotEqual(identity(context),support['execution_context_sha256'])
        module='tools.predictor_validation_20261027.native_queue_phase_busy_v10'
        with patch(module+'.verify_support'),patch(module+'.checked',side_effect=[{},context]),patch(
                module+'.AvailabilityForecast',side_effect=RuntimeError('reached expensive initialization')) as provider:
            with self.assertRaisesRegex(RuntimeError,'reached expensive initialization'):
                NativeQueuePhaseFactory({},support,{},{} )
            provider.assert_called_once()
        with patch(module+'.verify_support'),patch(module+'.checked',side_effect=[{},context]),patch(
                module+'.AvailabilityForecast') as provider:
            with self.assertRaisesRegex(ValueError,'target full context differ'):
                NativeQueuePhaseFactory({},dict(execution_context_sha256='b'*64),{},{})
            provider.assert_not_called()


if __name__=='__main__':unittest.main()
