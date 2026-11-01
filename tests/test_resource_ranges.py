"""Checkpoint transport must preserve exact ranges, resume and content hashes."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gsdeploy.profiling.resources import download_file


class RangeDownloadTests(unittest.TestCase):
    def test_cdn_reuse_and_expired_route_recovery_preserve_hash(self):
        raw = b'0123456789abcdef' * (1024 * 1024)
        entry = {'size': len(raw), 'lfs': {'sha256': hashlib.sha256(raw).hexdigest()}}
        calls, expired = [], False
        class Response(io.BytesIO):
            status = 206
            def __init__(self, start, end):
                super().__init__(raw[start:end+1])
                self.headers = {'Content-Range': f'bytes {start}-{end}/{len(raw)}'}
            def geturl(self):
                return 'https://cdn.example/weights?temporary=secret'
        class Opener:
            def open(self, request, **kwargs):
                nonlocal expired
                calls.append(request.full_url)
                if 'cdn.example' in request.full_url and not expired:
                    expired = True
                    raise OSError('expired test route')
                start, end = map(int, request.get_header('Range').removeprefix('bytes=').split('-'))
                return Response(start, end)
        with tempfile.TemporaryDirectory() as folder, patch('urllib.request.build_opener', return_value=Opener()), patch('time.sleep'):
            target = Path(folder) / 'weights'
            self.assertEqual(download_file('https://hub.example/source', target, entry, workers=1), 'downloaded')
            self.assertEqual(target.read_bytes(), raw)
            self.assertEqual(calls, ['https://hub.example/source', 'https://cdn.example/weights?temporary=secret', 'https://hub.example/source'])
            self.assertFalse(target.with_name('weights.ranges.json').exists())

    def test_server_ignoring_range_does_not_certify_checkpoint(self):
        class Response(io.BytesIO):
            status, headers = 200, {}
        class Opener:
            def open(self, *args, **kwargs):
                return Response(b'incorrect bytes')
        with tempfile.TemporaryDirectory() as folder, patch('urllib.request.build_opener', return_value=Opener()), patch('time.sleep'):
            target = Path(folder) / 'weights'
            entry = {'size': 16 * 1024 * 1024, 'lfs': {'sha256': '0' * 64}}
            with self.assertRaisesRegex(ValueError, 'exact requested range'):
                download_file('https://hub.example/source', target, entry, workers=1)
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
