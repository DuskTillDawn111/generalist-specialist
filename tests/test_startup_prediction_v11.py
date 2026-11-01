"""New collector identity must be checked while preserving old training facts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests import test_startup_prediction_v6 as fixtures
from tools.predictor_validation_20261027 import startup_prediction_v6 as old
from tools.predictor_validation_20261027 import startup_prediction_v11 as tool


class FreshCollectorStartupTests(unittest.TestCase):
    def test_actual_new_collector_and_fresh_ids_bind_old_verified_training(self):
        fixture = fixtures.FlowBoundStartupTests()
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(old,'native_execution_projection',side_effect=fixture.native), \
                patch.object(tool,'native_execution_projection',side_effect=fixture.native):
            root = Path(temp)
            model, old_plan, base, _, _ = fixture.fixture(root)
            document = json.loads(old_plan.read_text())
            document.update(runner_source=tool.receipt(tool.flow.__file__),
                collector_sources=[tool.receipt(p) for p in tool.flow.collection_source_paths()])
            document['trace'][0]['request_id'] = 'fresh-availability:req000'
            document['candidates'][0]['routes'] = {'fresh-availability:req000':'text#0'}
            def write(name,value):
                value=copy.deepcopy(value);value.pop('plan_sha256')
                value['plan_sha256']=tool.digest(value)
                path=root/name;path.write_text(json.dumps(value));return path
            plan=write('fresh.json',document)
            with patch.object(tool.flow,'BASE',base):
                result=tool.prediction(model,plan,'shared_balanced',allow_native_priming_transfer_hypothesis=True)
                self.assertEqual(result['energy_joules'],400.)
                self.assertEqual(result['peak_memory_mib'],800.)
                self.assertEqual(result['schema'],tool.SCHEMA)
                self.assertEqual(result['target_contract']['runner'],tool.receipt(tool.flow.__file__))
                self.assertIn(tool.receipt(old.__file__),result['evidence'])
                self.assertFalse(result['full_context_equality_claimed'])
                self.assertEqual(len(result['target_contract']['actual_priming_content']),12)
                for kind in ('old-runner','missing-source','admission','warmup'):
                    changed=copy.deepcopy(document)
                    if kind=='old-runner':changed['runner_source']=tool.receipt(old.flow.__file__)
                    elif kind=='missing-source':changed['collector_sources'].pop()
                    elif kind=='admission':changed['candidates'][0]['resource_configuration']=tool.flow.resource_configuration(1)
                    else:changed['warmup_seconds']=1.
                    with self.assertRaises(ValueError):
                        tool.prediction(model,write(kind+'.json',changed),'shared_balanced',
                                        allow_native_priming_transfer_hypothesis=True)

    def test_optin_before_any_source_read(self):
        with self.assertRaisesRegex(ValueError,'Explicit native'):
            tool.prediction('not-read','not-read','not-read')


if __name__ == '__main__':
    unittest.main()
