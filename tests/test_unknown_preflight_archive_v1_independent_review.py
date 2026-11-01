"""Independent opaque-byte and failure checks; historical files untouched."""
import gzip
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import unknown_preflight_archive_v1 as archive


class ArchiveIndependentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.source=self.root/'original.json'
        # Crosses three stream buffers, includes NUL/non-UTF8 and is not parsed.
        self.data=(bytes(range(256))*((3*archive.BUFFER)//256))+b'\x00\xff\nlast'
        self.source.write_bytes(self.data)
        self.record=dict(candidate_id='fixture',original=archive.receipt(self.source),
            original_size=len(self.data),original_stat=archive.file_stat(self.source))

    def test_multibuffer_binary_payload_roundtrip_exact(self):
        before=archive.file_stat(self.source)
        record=archive._archive_one(self.record,self.root/'opaque.gz')
        restored=archive._restore_one(record,self.root/'restored.json')
        self.assertEqual((self.root/'restored.json').read_bytes(),self.data)
        self.assertEqual(restored['restored']['sha256'],self.record['original']['sha256'])
        self.assertEqual(archive.file_stat(self.source),before)

    def test_existing_archive_is_never_truncated(self):
        target=self.root/'existing.gz';target.write_bytes(b'preexisting evidence')
        with self.assertRaises(FileExistsError):archive._archive_one(self.record,target)
        self.assertEqual(target.read_bytes(),b'preexisting evidence')
        self.assertEqual(self.source.read_bytes(),self.data)

    def test_source_mutation_during_stream_cannot_be_certified(self):
        original_write=gzip.GzipFile.write;calls=[]
        def write(stream,data):
            if not calls:
                calls.append(True)
                with self.source.open('r+b') as source:
                    source.seek(2*archive.BUFFER);source.write(b'changed')
            return original_write(stream,data)
        with patch.object(gzip.GzipFile,'write',write):
            with self.assertRaisesRegex(ValueError,'SHA/length/stat'):
                archive._archive_one(self.record,self.root/'rejected.gz')
        self.assertNotEqual(self.source.read_bytes(),self.data)

    def test_concatenated_gzip_extra_bytes_and_bad_crc_are_rejected(self):
        target=self.root/'extra.gz'
        target.write_bytes(gzip.compress(self.data)+gzip.compress(b'extra'))
        with self.assertRaisesRegex(ValueError,'exceeds exact'):
            archive.verify_gzip(target,self.record['original']['sha256'],len(self.data))
        payload=bytearray(gzip.compress(self.data));payload[-8]^=1
        target.write_bytes(payload)
        with self.assertRaises((gzip.BadGzipFile,OSError)):
            archive.verify_gzip(target,self.record['original']['sha256'],len(self.data))


if __name__=='__main__':unittest.main()
