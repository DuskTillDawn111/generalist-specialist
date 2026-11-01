"""Tiny independent byte fixtures; never archive/remove historical evidence."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import unknown_preflight_archive_v1 as tool


class UnknownArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.original=self.root/'original.json';self.data=(b'{"status":"UNKNOWN","unicode":"\xe4\xb8\xad\xe6\x96\x87"}\n'*1001)+bytes(range(256))
        self.original.write_bytes(self.data)
        self.source=dict(candidate_id='tiny',original=tool.receipt(self.original),original_size=len(self.data),original_stat=tool.file_stat(self.original))

    def test_streamed_roundtrip_exact_original_retained_and_fsync_observed(self):
        archive=self.root/'evidence.gz'
        with patch.object(tool.os,'fsync',wraps=tool.os.fsync) as synced:
            record=tool._archive_one(self.source,archive)
            restored=tool._restore_one(record,self.root/'restored.json')
        self.assertGreaterEqual(synced.call_count,4)
        self.assertEqual(self.original.read_bytes(),self.data);self.assertEqual((self.root/'restored.json').read_bytes(),self.data)
        self.assertEqual(tool.file_stat(self.original),self.source['original_stat'])
        self.assertTrue(record['verification']['full_stream_decompressed']);self.assertTrue(restored['bytes_exact'])
        self.assertFalse(restored['historical_inode_or_timestamps_recreated'])

    def test_wrong_original_sha_is_rejected_without_modifying_original(self):
        source=copy.deepcopy(self.source);source['original']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'SHA/length/stat'):
            tool._archive_one(source,self.root/'rejected.gz')
        self.assertEqual(self.original.read_bytes(),self.data)

    def test_changed_original_stat_rejected_before_new_archive(self):
        self.original.write_bytes(self.data+b'x');target=self.root/'never.gz'
        with self.assertRaisesRegex(ValueError,'identity changed'):
            tool._archive_one(self.source,target)
        self.assertFalse(target.exists())

    def test_corrupt_gzip_or_wrong_full_decompressed_sha_cannot_verify(self):
        target=self.root/'bad.gz';target.write_bytes(gzip.compress(self.data)[:-7])
        with self.assertRaises((EOFError,OSError,ValueError)):
            tool.verify_gzip(target,self.source['original']['sha256'],len(self.data))
        target=self.root/'valid.gz';target.write_bytes(gzip.compress(self.data))
        with self.assertRaisesRegex(ValueError,'SHA256/length'):
            tool.verify_gzip(target,'0'*64,len(self.data))
        with self.assertRaisesRegex(ValueError,'exceeds exact'):
            tool.verify_gzip(target,self.source['original']['sha256'],len(self.data)-1)

    def test_restore_rejects_existing_file_and_broken_symlink(self):
        record=tool._archive_one(self.source,self.root/'valid.gz')
        with self.assertRaises(FileExistsError):tool._restore_one(record,self.original)
        target=self.root/'broken';target.symlink_to(self.root/'missing')
        with self.assertRaises(FileExistsError):tool._restore_one(record,target)
        self.assertTrue(target.is_symlink());self.assertEqual(self.original.read_bytes(),self.data)

    def test_restore_checks_archive_receipt_before_creating_target(self):
        record=tool._archive_one(self.source,self.root/'valid.gz');Path(record['archive']['path']).write_bytes(b'changed')
        target=self.root/'restored.json'
        with self.assertRaisesRegex(ValueError,'receipt/stat'):tool._restore_one(record,target)
        self.assertFalse(target.exists());self.assertFalse(list(self.root.glob('*.restore-partial-*')))

    def test_restore_corrupt_payload_never_publishes_target(self):
        record=tool._archive_one(self.source,self.root/'valid.gz');archive=Path(record['archive']['path'])
        archive.write_bytes(gzip.compress(b'wrong'))
        record['archive']=tool.receipt(archive);record['archive_stat']=tool.file_stat(archive)
        target=self.root/'restored.json'
        with self.assertRaisesRegex(ValueError,'target not published'):tool._restore_one(record,target)
        self.assertFalse(target.exists());self.assertTrue(list(self.root.glob('*.restore-partial-*')))

    def test_atomic_publish_does_not_overwrite_target_created_during_restore(self):
        record=tool._archive_one(self.source,self.root/'valid.gz');target=self.root/'raced.json';original=tool.os.link
        def race(source,destination):
            Path(destination).write_bytes(b'concurrent existing evidence');return original(source,destination)
        with patch.object(tool.os,'link',side_effect=race):
            with self.assertRaises(FileExistsError):tool._restore_one(record,target)
        self.assertEqual(target.read_bytes(),b'concurrent existing evidence')

    def test_only_exact_manifest_allowlist_and_contained_archive_paths_restore(self):
        records=[]
        for cid,(sha,size) in tool.ALLOWED.items():
            records.append(dict(candidate_id=cid,original=dict(path=str(tool.SOURCE/('forecast-'+cid+'.json')),sha256=sha),
                original_size=size,archive=dict(path=str(self.root/('forecast-'+cid+'.json.gz')),sha256='a'*64),
                verification=dict(decompressed_sha256=sha,decompressed_bytes=size,exact_original_bytes_verified=True)))
        manifest=dict(schema=tool.SCHEMA,status='ALL_FOUR_FULL_DECOMPRESSION_VERIFIED_FSYNCED',
            source_report=dict(path=str(tool.SOURCE/'report.json'),sha256=tool.REPORT_SHA),originals=records)
        p=self.root/'manifest.json';p.write_text(json.dumps(manifest));tool.checked_manifest(p,tool.receipt(p)['sha256'])
        manifest['originals'][0]['archive']['path']='/other/training/raw.json';p.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'exact allowlist'):tool.checked_manifest(p,tool.receipt(p)['sha256'])
        with self.assertRaisesRegex(ValueError,'manifest SHA'):tool.checked_manifest(p,'0'*64)

    def test_dryrun_rejects_training_or_arbitrary_output_namespace(self):
        with self.assertRaisesRegex(ValueError,'namespace'):tool.dryrun(self.root/'training')


if __name__=='__main__':unittest.main()
