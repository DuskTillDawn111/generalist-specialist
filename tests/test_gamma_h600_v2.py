import copy
import tempfile
from pathlib import Path
import unittest

from tools.predictor_validation_20261027 import gamma_h600_v2 as workflow
from tools.predictor_validation_20261027.background_statistical import BackgroundStatisticalProvider
from tools.predictor_validation_20261027.native_flow import receipt
from tests import test_duration_profiles as fixtures


class RawEvidenceTests(unittest.TestCase):
    def setUp(self):
        f=fixtures.DurationProviderTests();f.setUp()
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        path=Path(self.tmp.name)/'raw.json';path.write_text('{}');self.ref=receipt(path)
        self.bundle=dict(protocol_receipt=self.ref,run_protocol_receipt=self.ref)
        # Source mapping does not create transient direct_source files.
        self.group=dict(group_id='old:0',role='old_profile_training',raw=self.ref,
            transitive_sources=[self.ref],traces=f.traces)

    def test_native_raw_is_enough_for_reproducible_source_bound_fit(self):
        a=workflow.fit_groups(self.bundle,[self.group],self.ref,None)
        b=workflow.fit_groups(self.bundle,[self.group],self.ref,None)
        self.assertEqual(a,b)
        self.assertEqual(len(BackgroundStatisticalProvider(a,'v7-fixture').lookup),3)
        self.assertEqual(a['gamma_training_evidence']['original_profile_collections'],1)
        self.assertIn(self.ref,a['source_receipts'])

    def test_source_tamper_and_nontraining_observation_are_rejected(self):
        group=copy.deepcopy(self.group);group['traces'][0]['split']='validation'
        with self.assertRaisesRegex(ValueError,'Training-only'):workflow.fit_groups(self.bundle,[group],self.ref,None)
        Path(self.ref['path']).write_text('changed')
        with self.assertRaises(ValueError):workflow.fit_groups(self.bundle,[self.group],self.ref,None)


if __name__=='__main__':unittest.main()
