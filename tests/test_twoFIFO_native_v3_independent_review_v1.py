import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_native_v3_independent_review_v1 as m


class NativeIndependentEntryTests(unittest.TestCase):
    def test_incomplete_raw_is_never_opened_and_no_native_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);collection=root/'attempt/collection';collection.mkdir(parents=True)
            (collection/'raw.json').write_text('not even valid JSON; must never open')
            with patch.object(m.review,'review') as review,patch.object(m.macros,'extract') as extract:
                report=m.checked(m.run(root/'attempt',root/'out'))
            review.assert_not_called();extract.assert_not_called()
            self.assertEqual(report['status'],'INCOMPLETE');self.assertFalse(report['raw_opened'])
            self.assertFalse(report['training_feature_extraction_eligible'])


if __name__=='__main__':unittest.main()
