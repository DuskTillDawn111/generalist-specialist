import copy
import json
from pathlib import Path
import tempfile
import unittest

from gsdeploy.resources import identity
from tools.predictor_validation_20261027.native_collect import receipt
from tools.predictor_validation_20261027.composite_execution_v2 import (
    SCHEMA,physical_projection,CompositeComponentAdapter)


class Component:
    execution_context_sha256='original'
    def delays(self,row,aliases,scenario):return dict(known=True,used_scenario=scenario)


class CompositeExecutionTests(unittest.TestCase):
    def test_projection_retains_physical_settings_and_source_content(self):
        context=dict(profiler='light',measurement_domain='light-kernel',worker_variant='semantic',affine_observation=False,
                     execution=dict(batch_size=1),threads=4,residents=dict(a='family'),
                     execution_sources=[dict(original_path='worker.py',frozen=dict(path='capture1',sha256='source'))])
        natural=copy.deepcopy(context);natural.update(profiler='none',measurement_domain='natural')
        natural['execution_sources'][0]['frozen']['path']='capture2'
        self.assertEqual(physical_projection(context),physical_projection(natural))
        natural['execution']['batch_size']=2
        self.assertNotEqual(physical_projection(context),physical_projection(natural))
        natural['execution']['batch_size']=1;natural['execution_sources'][0]['frozen']['sha256']='other'
        self.assertNotEqual(physical_projection(context),physical_projection(natural))

    def make_contract(self,path):
        contract=dict(schema=SCHEMA,components=dict(lifecycle=dict(model=receipt(path),original_context_sha256='original')),
                      target=dict(context=receipt(path)),scenario_adapter=dict(profiled='mean'))
        contract['composite_execution_sha256']=identity(contract)
        return contract

    def test_adapter_keeps_original_component_and_maps_scenario_explicitly(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.json';path.write_text('{}')
            component=Component();contract=self.make_contract(path)
            adapter=CompositeComponentAdapter(component,contract,'lifecycle',path)
            self.assertEqual(component.execution_context_sha256,'original')
            self.assertEqual(adapter.execution_context_sha256,contract['composite_execution_sha256'])
            result=adapter.delays({}, {},'profiled')
            self.assertEqual(result['used_scenario'],'mean')
            self.assertEqual(result['original_execution_context_sha256'],'original')

    def test_contract_tamper_and_wrong_component_context_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.json';path.write_text('{}');contract=self.make_contract(path)
            contract['scenario_adapter']['profiled']='other'
            with self.assertRaises(ValueError):CompositeComponentAdapter(Component(),contract,'lifecycle',path)
            contract=self.make_contract(path);component=Component();component.execution_context_sha256='other'
            with self.assertRaises(ValueError):CompositeComponentAdapter(component,contract,'lifecycle',path)

    def test_artifact_changes_rejected_even_with_same_declared_context(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.json';path.write_text('{}');contract=self.make_contract(path);path.write_text('{"changed":true}')
            with self.assertRaises(ValueError):CompositeComponentAdapter(Component(),contract,'lifecycle',path)


if __name__=='__main__':unittest.main()
