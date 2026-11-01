"""Verify exact derived archive recovery and refusal before deleting a copy."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import completed_derived_archive_v32 as v


class DerivedArchiveTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name); (self.root/'artifacts').mkdir()
        patched = patch.object(v, 'ROOT', self.root); patched.start(); self.addCleanup(patched.stop)
        disk = patch.object(v.shutil, 'disk_usage', return_value=SimpleNamespace(free=100*2**30))
        disk.start(); self.addCleanup(disk.stop)
        self.source, self.archive_dir = v.locations('skewed')
        self.source.mkdir(parents=True)
        self.parsed = self.source/'parsed-measured.json'
        self.parsed.write_bytes(b'{"all_original_bytes":true}\n'*1000)
        self.original = v.a.receipt(self.parsed)
        self.report = self.source/'report.json'
        self.report.write_text(json.dumps(dict(status='PASS', issues=[], requests=57,
            training_only=True, validation_started=False, model_fitted=False,
            trace=dict(parsed=self.original))))
        self.sha = v.a.receipt(self.report)['sha256']

    def archive(self):
        v.archive('skewed', self.sha)
        return v.a.receipt(self.archive_dir/'manifest.json')['sha256']

    def output(self, operation):
        return self.root/'artifacts'/('predictor-whole57-derived-'+operation+'-v32-skewed-test')

    def test_exact_roundtrip_retains_review_and_other_evidence(self):
        before = self.report.read_bytes()
        (self.source/'raw.json').write_bytes(b'untouched')
        sha = self.archive()
        retired = v.transaction('retirement', 'skewed', sha, self.output('retirement'))
        self.assertFalse(self.parsed.exists())
        self.assertTrue((self.archive_dir/'parsed-measured.json.gz').is_file())
        self.assertIn('--execute', retired['restore_command'])
        v.transaction('recovery', 'skewed', sha, self.output('recovery'))
        self.assertEqual(v.a.receipt(self.parsed), self.original)
        self.assertEqual(self.report.read_bytes(), before)
        self.assertEqual((self.source/'raw.json').read_bytes(), b'untouched')

    def test_corrupt_archive_refuses_retirement_original_stays(self):
        sha = self.archive()
        with (self.archive_dir/'parsed-measured.json.gz').open('r+b') as stream:
            stream.seek(-5, 2); stream.write(b'broken')
        with self.assertRaises(ValueError):
            v.transaction('retirement', 'skewed', sha, self.output('retirement'))
        self.assertEqual(v.a.receipt(self.parsed), self.original)
        self.assertFalse(self.output('retirement').exists())

    def test_wrong_review_or_derived_path_refused_without_archive(self):
        with self.assertRaisesRegex(ValueError, 'SHA'):
            v.archive('skewed', '0'*64)
        report = json.loads(self.report.read_text())
        report['trace']['parsed']['path'] = str(self.source/'raw.json')
        self.report.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'derived path'):
            v.archive('skewed', v.a.receipt(self.report)['sha256'])
        self.assertFalse(self.archive_dir.exists())


if __name__ == '__main__': unittest.main()
