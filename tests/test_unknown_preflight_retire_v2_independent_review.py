"""Independent tiny transaction/failure fixtures; no real evidence is retired."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from tests import test_unknown_preflight_retire_v2 as author
from tools.predictor_validation_20261027 import unknown_preflight_archive_v1 as a
from tools.predictor_validation_20261027 import unknown_preflight_retire_v2 as t


class RetirementIndependentTests(unittest.TestCase):
    def fixture(self):
        fixture=author.RetirementTests('test_read_only_inspect_full_verification')
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        return fixture

    def test_failed_durable_intent_never_unlinks_an_original(self):
        f=self.fixture();save=a.save_synced
        def fail(path,value):
            if Path(path).name=='intent.json':raise OSError('intent fsync failed')
            return save(path,value)
        with patch.object(a,'save_synced',side_effect=fail):
            with self.assertRaisesRegex(OSError,'intent fsync'):f.call_retire()
        f.assert_originals_intact();f.assert_archives_intact()

    def test_unlink_failure_after_one_file_recovers_only_missing_original(self):
        f=self.fixture();unlink=t.os.unlink;calls=[]
        def fail(path,*args,**kwargs):
            calls.append(str(path))
            if len(calls)==2:raise PermissionError('second unlink failed')
            return unlink(path,*args,**kwargs)
        with patch.object(t.os,'unlink',side_effect=fail):
            with self.assertRaisesRegex(PermissionError,'second unlink'):f.call_retire()
        failed=json.loads((f.out('retirement')/'failed-retirement.json').read_text())
        self.assertEqual(sum(x['missing'] for x in failed['current_targets']),1)
        self.assertEqual(len(failed['retired']),1)
        self.assertEqual(len(f.call_recover()['restored']),1)
        f.assert_originals_intact();f.assert_archives_intact()

    def test_failure_after_recovery_publish_can_resume_without_overwrite(self):
        f=self.fixture();f.call_retire();sync=a.fsync_directory
        def fail(path):
            if Path(path)==f.source:raise OSError('recovery directory fsync failed')
            return sync(path)
        with patch.object(a,'fsync_directory',side_effect=fail):
            with self.assertRaisesRegex(OSError,'recovery directory'):f.call_recover()
        failed=json.loads((f.out('recovery')/'failed-recovery.json').read_text())
        self.assertEqual(sum(not x['missing'] for x in failed['current_targets']),1)
        self.assertEqual(len(f.call_recover('retry')['restored']),3)
        f.assert_originals_intact();f.assert_archives_intact()


if __name__=='__main__':unittest.main()
