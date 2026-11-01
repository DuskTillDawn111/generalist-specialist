"""Controller v2 independent test through the real resource integration gate."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests import test_twoFIFO_campaign_continue_v2 as fixture
from tests.test_twoFIFO_observed_resources_v1 import fixture as raw_fixture,SECOND
from tools.predictor_validation_20261027 import twoFIFO_campaign_continue_v2 as m


class ContinuationResourceIndependentTests(unittest.TestCase):
    def test_real_oversized_poll_gap_persists_UNKNOWN_and_never_launches_next(self):
        helper=fixture.ContinueTests()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,manifest=helper.setup_campaign(root);item=manifest['attempts'][0]
            collection=Path(item['collection']);fixture.write(collection/'execution.json',helper.execution(item))
            raw=raw_fixture();raw['board_samples']=[s for s in raw['board_samples'] if s['time_ns'] not in (3*SECOND,4*SECOND)]
            fixture.write(collection/'raw.json',raw)
            report=root/'review.json';fixture.write(report,dict(status=m.COMPLETE,
                training_feature_extraction_eligible=True,issues=[],raw=m.receipt(collection/'raw.json')))
            with patch.object(m,'audit',return_value=manifest),patch.object(m,'review',return_value=m.receipt(report)),patch.object(m.collector,'execute') as execute,patch.object(m.macros,'extract') as extract:
                result=m.run(path,root/'run',execute=True)
            execute.assert_not_called();extract.assert_not_called()
            self.assertEqual(result['status'],'STOPPED_NO_RETRY')
            facts=json.loads((root/'run'/item['attempt_id']/'observed-resources.json').read_text())
            self.assertIsNone(facts['cases'][0]['measurement_energy_joules'])
            self.assertIsNotNone(facts['cases'][1]['measurement_energy_joules'])
            self.assertEqual(len(facts['cases']),2)
            self.assertFalse(any(Path(a['collection']).exists() for a in manifest['attempts'][1:]))


if __name__=='__main__':unittest.main()
