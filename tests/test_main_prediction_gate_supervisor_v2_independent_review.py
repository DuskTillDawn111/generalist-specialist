"""Independent resealed-protocol regressions for the new v55 supervisor."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027 import main_prediction_gate_supervisor_v2 as v
from tools.predictor_validation_20261027.known_input_gate_v5 import identity
from tests import test_main_prediction_gate_supervisor_v2 as fixtures


class IndependentV2Tests(unittest.TestCase):
    def test_resealed_tie_tolerance_and_training_index_omission_refused(self):
        for mutation in ('tie', 'training_index', 'source_before', 'review'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                args, protocol, _ = fixtures.StrictBindingTests().fixture(Path(directory))
                production = v.read(args[3]); prep = v.checked(production['preparation'])
                if mutation == 'tie':
                    protocol['utility_tie_tolerance'] = 1e-3
                elif mutation == 'training_index':
                    protocol['training_data'] = []
                else:
                    removed = prep['source_before'] if mutation == 'source_before' else production['independent_review']
                    protocol['sources'] = [r for r in protocol['sources'] if r != removed]
                protocol['protocol_sha256'] = identity({k:x for k,x in protocol.items() if k != 'protocol_sha256'})
                args[1].write_text(json.dumps(protocol))
                predictions = v.read(args[2]); predictions['protocol_sha256'] = protocol['protocol_sha256']
                predictions['prediction_sha256'] = identity({k:x for k,x in predictions.items() if k != 'prediction_sha256'})
                args[2].write_text(json.dumps(predictions))
                production.update(protocol=v.receipt(args[1]), predictions=v.receipt(args[2]))
                args[3].write_text(json.dumps(production))
                with self.assertRaises(ValueError):
                    v.prerequisites(*args)


if __name__ == '__main__': unittest.main()
