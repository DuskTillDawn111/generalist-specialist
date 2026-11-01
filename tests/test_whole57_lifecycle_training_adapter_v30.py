"""Small byte-reader, native-adjacency, and three-valued feature checks."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027 import lossless_derived_reader_v30 as reader
from tools.predictor_validation_20261027 import unknown_preflight_archive_v1 as a
from tools.predictor_validation_20261027 import whole57_lifecycle_training_adapter_v30 as v


class ExplicitGzipReaderTests(unittest.TestCase):
    def fixture(self,schema='balanced'):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name)
        original=root/'retired.json';data=(b'{"bytes":"\xff\x00"}\n'*180000)+b'final'
        compressed=root/'retired.json.gz'
        with gzip.open(compressed,'wb') as stream:stream.write(data)
        ref=dict(path=str(original),sha256=hashlib.sha256(data).hexdigest())
        record=dict(original=ref,original_size=len(data),archive=a.receipt(compressed),archive_stat=a.file_stat(compressed),
            archive_size=compressed.stat().st_size,verification=dict(exact_original_bytes_verified=True,
                decompressed_sha256=ref['sha256'],decompressed_bytes=len(data)))
        manifest=dict(schema='gsdeploy.exact_light_balanced_derived_v23_lossless_archive.v1',
            status='ALL_SELECTED_FULL_DECOMPRESSION_VERIFIED_FSYNCED',originals_retained=True,originals=[record])
        if schema=='v32':manifest.update(schema='gsdeploy.completed_whole57_derived_lossless_archive.v32',
            status='FULL_ORIGINAL_SHA_AND_GZIP_ROUNDTRIP_VERIFIED',original_retained=True)
        path=root/'manifest.json';path.write_text(json.dumps(manifest))
        return path,ref,data

    def test_absent_logical_original_and_partial_consumption_verify_entire_stream(self):
        path,ref,data=self.fixture()
        with reader.open_verified_gzip(a.receipt(path),ref) as (stream,evidence):
            self.assertEqual(stream.read(7),data[:7]);self.assertFalse(evidence['verified'])
        self.assertTrue(evidence['verified']);self.assertEqual(evidence['uncompressed_bytes'],len(data))
        self.assertFalse(Path(ref['path']).exists());self.assertFalse(evidence['logical_original_path_opened'])

    def test_both_explicit_manifest_schemas_work(self):
        path,ref,_=self.fixture('v32');self.assertTrue(reader.verify_mapping(a.receipt(path),ref)['verified'])

    def test_bad_crc_is_rejected_even_with_updated_compressed_receipt(self):
        path,ref,_=self.fixture();doc=json.loads(path.read_text());compressed=Path(doc['originals'][0]['archive']['path'])
        data=bytearray(compressed.read_bytes());data[-8]^=1;compressed.write_bytes(data)
        record=doc['originals'][0];record.update(archive=a.receipt(compressed),archive_stat=a.file_stat(compressed))
        path.write_text(json.dumps(doc))
        with self.assertRaises(gzip.BadGzipFile):reader.verify_mapping(a.receipt(path),ref)

    def test_wrong_logical_hash_rejected_without_opening_original(self):
        path,ref,_=self.fixture();wrong=dict(ref,sha256='0'*64)
        with self.assertRaisesRegex(ValueError,'logical original'):reader.verify_mapping(a.receipt(path),wrong)


class NativeMarkBridgeTests(unittest.TestCase):
    def test_same_owner_different_context_retained_and_unpaired_adjacent_rejected(self):
        akey=(1,0,10,100);bkey=(1,0,11,100);text=(1,0,12,200)
        pairs=[(0,10,'ocr#0',akey),(12,20,'ocr#0',bkey),(22,30,'text#0',text)]
        clean,rejected=v.transition_records(pairs,[],0,40,'capture',[])
        self.assertEqual(len(clean),2);self.assertTrue(clean[0]['same_owner']);self.assertFalse(clean[0]['same_driver_context'])
        clean,rejected=v.transition_records(pairs,[dict(timestamp=11)],0,40,'capture',[])
        self.assertEqual(len(clean),1);self.assertEqual(len(rejected),1)
        self.assertTrue(rejected[0]['same_owner']);self.assertEqual(rejected[0]['held_driver_context'],list(akey))

    def test_unknown_ready_preserved_and_future_label_cannot_change_features(self):
        states={o:v.legacy.OwnerLifecycle(False,False,False) for o in v.RESIDENTS}
        states['ocr#0']=v.legacy.OwnerLifecycle(True,None,None)
        states['text#0']=v.legacy.OwnerLifecycle(True,True,True)
        query=dict(held_owner='ocr#0',context_age_ns=77768,previous_held_owner='text#0',
            pending_frontier=[['text#0','text','p',['prefill']]],public_workload=None,
            pending_age_ns_by_owner=None,pending_reference_work_ns_by_owner=None)
        member=dict(query=query,states=states,row=dict(next_owner='ocr#0',next_restore_start_ns=1000))
        features=v.native_visible_features(member,{'p':'text'});names=v.legacy.feature_names({'p':'text'})
        self.assertEqual(features[names.index('gpu_ready:ocr#0:unknown')],1.)
        self.assertEqual(features[names.index('gpu_ready:ocr#0:true')],0.)
        altered=copy.deepcopy(member);altered['row'].update(next_owner='text#1',next_restore_start_ns=999999)
        self.assertEqual(v.native_visible_features(altered,{'p':'text'}),features)


if __name__=='__main__':unittest.main()
