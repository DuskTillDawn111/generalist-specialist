"""Explicit gzip storage mapping; verify original bytes without restoring paths."""
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
from pathlib import Path

from .unknown_preflight_archive_v1 import BUFFER, file_stat, receipt


class _HashingReader(io.RawIOBase):
    def __init__(self, source, size):
        super().__init__();self.source=source;self.expected_size=size;self.size=0;self.digest=hashlib.sha256()

    def readable(self):return True

    def readinto(self, buffer):
        block=self.source.read(len(buffer));length=len(block)
        self.size+=length
        if self.size>self.expected_size:raise ValueError('Decompressed stream exceeds original size')
        self.digest.update(block);buffer[:length]=block;return length


def checked_record(manifest_reference, original_reference):
    """The original path is logical identity only, and is never opened."""
    path=Path(manifest_reference['path']);file_stat(path)
    if receipt(path)!=manifest_reference:raise ValueError('Explicit archive manifest receipt changed')
    manifest=json.loads(path.read_text())
    contracts={
        'gsdeploy.exact_light_balanced_derived_v23_lossless_archive.v1':
            ('ALL_SELECTED_FULL_DECOMPRESSION_VERIFIED_FSYNCED','originals_retained'),
        'gsdeploy.completed_whole57_derived_lossless_archive.v32':
            ('FULL_ORIGINAL_SHA_AND_GZIP_ROUNDTRIP_VERIFIED','original_retained')}
    contract=contracts.get(manifest.get('schema'))
    if (contract is None or manifest.get('status')!=contract[0]
            or manifest.get(contract[1]) is not True):
        raise ValueError('Completed lossless archive manifest required')
    records=[r for r in manifest.get('originals',[]) if r.get('original')==original_reference]
    if len(records)!=1:raise ValueError('Unique exact logical original in explicit manifest required')
    record=records[0]
    if (type(record.get('original_size')) is not int or record['original_size']<0
            or record.get('verification',{}).get('exact_original_bytes_verified') is not True
            or record['verification'].get('decompressed_sha256')!=original_reference['sha256']
            or record['verification'].get('decompressed_bytes')!=record['original_size']):
        raise ValueError('Manifest original full-byte verification differs')
    return record


@contextmanager
def open_verified_gzip(manifest_reference, original_reference):
    """Yield (binary stream, evidence); publish consumer results after exit.

    Exit drains any unread suffix, checks gzip CRC, full original SHA/length,
    and stable compressed bytes. No uncompressed file or symlink is created.
    """
    record=checked_record(manifest_reference,original_reference)
    archived=record['archive'];before=file_stat(archived['path'])
    if before!=record['archive_stat'] or before['size']!=record['archive_size'] or receipt(archived['path'])!=archived:
        raise ValueError('Compressed archive receipt/stat changed')
    evidence=dict(encoding='gzip',logical_original=original_reference,archive_manifest=manifest_reference,
        compressed=archived,uncompressed_size=record['original_size'],verified=False,
        logical_original_path_opened=False,original_path_restored=False)
    with gzip.open(archived['path'],'rb') as source:
        raw=_HashingReader(source,record['original_size'])
        with io.BufferedReader(raw,buffer_size=BUFFER) as stream:
            yield stream,evidence
            while stream.read(BUFFER):pass
            if raw.size!=record['original_size'] or raw.digest.hexdigest()!=original_reference['sha256']:
                raise ValueError('Decompressed original SHA256/length mismatch')
    if (file_stat(archived['path'])!=before or receipt(archived['path'])!=archived
            or receipt(manifest_reference['path'])!=manifest_reference):
        raise ValueError('Archive/manifest changed while consuming gzip')
    evidence.update(verified=True,full_stream_decompressed=True,gzip_CRC_checked=True,
        uncompressed_sha256=raw.digest.hexdigest(),uncompressed_bytes=raw.size)


def verify_mapping(manifest_reference, original_reference):
    with open_verified_gzip(manifest_reference,original_reference) as (_,evidence):pass
    return evidence
