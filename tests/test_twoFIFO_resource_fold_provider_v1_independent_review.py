"""Independent checks of all leaveout directions and startup/state scope."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_twoFIFO_resource_fold_provider_v1 import fixture, model_document, timeline_and_binding
from tools.predictor_validation_20261027 import twoFIFO_resource_fold_provider_v1 as m


class IndependentFoldTests(unittest.TestCase):
    def test_every_fold_uses_its_own_two_blocks_in_both_resource_scenarios(self):
        plan,context,tables=fixture()
        with tempfile.TemporaryDirectory() as directory:
            document,sources=model_document(tables,directory)
            path=Path(directory)/'full-model.json';path.write_text(json.dumps(document))
            for heldout in (1,2,3):
                donors=[b for b in (1,2,3) if b!=heldout]
                for scenario in ('mean','max'):
                    with self.subTest(heldout=heldout,scenario=scenario), patch.object(
                            m.resource,'load_complete',return_value=(tables,sources)):
                        provider=m.FoldResourceTrainingProvider(path,plan,context,
                            heldout_block_id='training-block-'+str(heldout),
                            allow_training_resource_hypothesis=True,scenario=scenario)
                        timeline,binding=timeline_and_binding(provider)
                        result=m.predict_timeline(json.loads(json.dumps(timeline)),provider,
                            target_context=context,execution_binding=binding)
                        selected=sum(donors)/2 if scenario=='mean' else max(donors)
                        self.assertEqual(result['board_energy_joules'],2*selected)
                        self.assertEqual(result['peak_memory_mib'],1000*selected)
                        # Startup remains its declared equal-block expectation;
                        # state maxima do not silently replace that estimator.
                        self.assertEqual(result['startup_energy_joules'],10*sum(donors)/2)
                        self.assertEqual(set(provider.document['training_block_ids']),
                            {'training-block-'+str(b) for b in donors})
                        self.assertEqual(len(provider.document['training_attempt_ids']),8)
                        self.assertFalse(result['resource_prediction_qualified'])


if __name__=='__main__':unittest.main()
