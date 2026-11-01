"""Tiny exact old-derived snapshots, never actual historical evidence."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import multimodal_derived_archive_v37 as v


class ExactOldDerivedTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name);(self.root/'artifacts').mkdir()
        p=patch.object(v,'ROOT',self.root);p.start();self.addCleanup(p.stop)
        p=patch.object(v.shutil,'disk_usage',return_value=SimpleNamespace(free=100*2**30));p.start();self.addCleanup(p.stop)
        self.source,self.arc=v.locations('packet01');self.source.mkdir(parents=True)
        self.original=self.source/'parsed-measured.json';self.original.write_bytes(b'\x00\xff{"retained":true}\n'*1000)
        self.ref=v.a.receipt(self.original);self.report=self.source/'report.json'
        self.report.write_text(json.dumps(dict(schema='gsdeploy.unified_corun_strict_review.v8_1',status='PASS',issues=[],
            training_only=True,independent_capture_count=1,validation_started=False,model_fitted=False,trace=dict(parsed=self.ref))))
        self.sha=v.a.receipt(self.report)['sha256']
        p=patch.object(v,'EXPECTED',dict(packet01=(self.sha,self.ref['sha256'],self.original.stat().st_size)));p.start();self.addCleanup(p.stop)

    def out(self,op):return self.root/'artifacts'/('predictor-multimodal-derived-'+op+'-v37-packet01-test')

    def test_archive_retire_recover_exact_bytes_and_other_sources_untouched(self):
        raw=self.source/'raw.json';raw.write_bytes(b'never change');before=self.report.read_bytes()
        v.archive('packet01',self.sha);sha=v.a.receipt(self.arc/'manifest.json')['sha256']
        receipt=v.transaction('retirement','packet01',sha,self.out('retirement'));self.assertFalse(self.original.exists())
        self.assertIn('tools.predictor_validation_20261027.multimodal_derived_archive_v37',receipt['restore_command'])
        v.transaction('recovery','packet01',sha,self.out('recovery'));self.assertEqual(v.a.receipt(self.original),self.ref)
        self.assertEqual(raw.read_bytes(),b'never change');self.assertEqual(self.report.read_bytes(),before)

    def test_foreign_path_and_wrong_SHA_rejected(self):
        with self.assertRaises(ValueError):v.locations('skewed')
        with self.assertRaisesRegex(ValueError,'SHA'):v.archive('packet01','0'*64)
        wrong=json.loads(self.report.read_text());wrong['trace']['parsed']['path']=str(self.source/'raw.json')
        self.report.write_text(json.dumps(wrong))
        with patch.object(v,'EXPECTED',dict(packet01=(v.a.receipt(self.report)['sha256'],self.ref['sha256'],self.original.stat().st_size))):
            with self.assertRaisesRegex(ValueError,'derived path'):v.archive('packet01',v.a.receipt(self.report)['sha256'])
        self.assertTrue(self.original.exists());self.assertFalse(self.arc.exists())

    def test_corrupt_archive_rejects_before_original_unlink(self):
        v.archive('packet01',self.sha);sha=v.a.receipt(self.arc/'manifest.json')['sha256']
        with (self.arc/'parsed-measured.json.gz').open('r+b') as f:f.seek(-8,2);f.write(b'badcrc00')
        with self.assertRaises(ValueError):v.transaction('retirement','packet01',sha,self.out('retirement'))
        self.assertEqual(v.a.receipt(self.original),self.ref);self.assertFalse(self.out('retirement').exists())


if __name__=='__main__':unittest.main()
