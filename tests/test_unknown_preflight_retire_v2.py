"""Tiny temporary four-file fixtures; no historical evidence is touched."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import unknown_preflight_archive_v1 as a
from tools.predictor_validation_20261027 import unknown_preflight_retire_v2 as t


class RetirementTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        (self.root/'artifacts').mkdir();self.source=self.root/'source';self.source.mkdir()
        self.archive=self.root/'archive';self.archive.mkdir();self.records=[];self.allowed={};self.data={}
        for index,cid in enumerate(a.ALLOWED):
            data=(bytes(range(256))+b'{"status":"UNKNOWN"}\n')*(index+2);self.data[cid]=data
            original=self.source/('forecast-'+cid+'.json');original.write_bytes(data)
            ref=a.receipt(original);self.allowed[cid]=(ref['sha256'],len(data))
            record=dict(candidate_id=cid,original=ref,original_size=len(data),original_stat=a.file_stat(original))
            self.records.append(a._archive_one(record,self.archive/(original.name+'.gz')))
        report=self.source/'report.json';report.write_text('{"fixture":true}\n');report_ref=a.receipt(report)
        self.manifest=self.archive/'manifest.json'
        self.value=dict(schema=a.SCHEMA,status='ALL_FOUR_FULL_DECOMPRESSION_VERIFIED_FSYNCED',
            source_report=report_ref,originals=self.records,originals_retained=True,tool=a.receipt(a.__file__))
        a.save_synced(self.manifest,self.value);self.ref=a.receipt(self.manifest)
        for name,value in [('ROOT',self.root),('SOURCE',self.source),('ALLOWED',self.allowed),('REPORT_SHA',report_ref['sha256'])]:
            p=patch.object(a,name,value);p.start();self.addCleanup(p.stop)

    def out(self,op,suffix='01'):return self.root/'artifacts'/('predictor-v16-preflight02-lossless-'+op+'-'+suffix)
    def call_retire(self):return t.retire(self.manifest,self.ref['sha256'],self.out('retirement'))
    def call_recover(self,suffix='01'):return t.recover(self.manifest,self.ref['sha256'],self.out('recovery',suffix))
    def rewrite_manifest(self):
        self.manifest.write_text(json.dumps(self.value));self.ref=a.receipt(self.manifest)
    def assert_archives_intact(self):
        for r in self.records:self.assertEqual(a.receipt(r['archive']['path']),r['archive'])
    def assert_originals_intact(self):
        for r in self.records:self.assertEqual(Path(r['original']['path']).read_bytes(),self.data[r['candidate_id']])

    def test_read_only_inspect_full_verification(self):
        result=t.inspect(self.manifest,self.ref['sha256']);self.assertFalse(result['source_deletion_executed'])
        self.assertEqual(len(result['evidence']),4);self.assert_originals_intact();self.assert_archives_intact()

    def test_retire_durable_intent_before_unlink_and_exact_recover(self):
        original_unlink=t.os.unlink;seen=[]
        def checked_unlink(path,*args,**kwargs):
            intent=self.out('retirement')/'intent.json';self.assertTrue(intent.exists())
            data=json.loads(intent.read_text());self.assertEqual(len(data['exact_fixed_targets']),4)
            self.assertIn('recover',data['restore_command']);seen.append(path)
            return original_unlink(path,*args,**kwargs)
        with patch.object(t.os,'unlink',side_effect=checked_unlink):result=self.call_retire()
        self.assertEqual(len(seen),4);self.assertEqual(len(result['retired']),4)
        self.assertTrue(all(not Path(r['original']['path']).exists() for r in self.records))
        restored=self.call_recover();self.assertEqual(len(restored['restored']),4)
        self.assert_originals_intact();self.assert_archives_intact()
        # An already restored inode differs from historical stat, but bytes are
        # accepted by recovery and never overwritten on subsequent recovery.
        again=self.call_recover('02');self.assertEqual(again['restored'],[])

    def test_receipt_failure_after_unlink_records_actual_absence_and_partial_recovers(self):
        original_save=a.save_synced
        def fail_receipt(path,value):
            if Path(path).name.startswith('retired-'):raise OSError('injected progress receipt failure')
            return original_save(path,value)
        with patch.object(a,'save_synced',side_effect=fail_receipt):
            with self.assertRaisesRegex(OSError,'injected'):self.call_retire()
        failed=json.loads((self.out('retirement')/'failed-retirement.json').read_text())
        self.assertEqual(failed['retired'],[]);self.assertEqual(sum(x['missing'] for x in failed['current_targets']),1)
        result=self.call_recover();self.assertEqual(len(result['restored']),1);self.assert_originals_intact();self.assert_archives_intact()

    def test_fsync_failure_after_unlink_records_actual_absence(self):
        original_sync=a.fsync_directory
        def fail_source(path):
            if Path(path)==self.source:raise OSError('injected source fsync failure')
            return original_sync(path)
        with patch.object(a,'fsync_directory',side_effect=fail_source):
            with self.assertRaisesRegex(OSError,'injected'):self.call_retire()
        failed=json.loads((self.out('retirement')/'failed-retirement.json').read_text())
        self.assertEqual(sum(x['missing'] for x in failed['current_targets']),1)
        self.call_recover();self.assert_originals_intact()

    def test_original_changed_after_intent_blocks_unlink(self):
        original=t._retire_records
        def mutate(records,intent,ref):
            def changed(index,record):Path(record['original']['path']).write_bytes(b'changed')
            return original(records,intent,ref,before_unlink=changed)
        with patch.object(t,'_retire_records',side_effect=mutate):
            with self.assertRaisesRegex(ValueError,'lstat'):self.call_retire()
        self.assertTrue(all(Path(r['original']['path']).exists() for r in self.records));self.assert_archives_intact()

    def test_archive_changed_after_intent_blocks_unlink(self):
        original=t._retire_records
        def mutate(records,intent,ref):
            def changed(index,record):Path(record['archive']['path']).write_bytes(b'broken')
            return original(records,intent,ref,before_unlink=changed)
        with patch.object(t,'_retire_records',side_effect=mutate):
            with self.assertRaisesRegex(ValueError,'Archive lstat'):self.call_retire()
        self.assert_originals_intact()

    def test_full_gzip_crc_checked_even_if_compressed_sha_stat_match(self):
        record=self.records[0];p=Path(record['archive']['path']);data=bytearray(p.read_bytes());data[-8]^=1;p.write_bytes(data)
        record['archive']=a.receipt(p);record['archive_stat']=a.file_stat(p);record['archive_size']=len(data);self.rewrite_manifest()
        with self.assertRaises((OSError,ValueError,EOFError)):self.call_retire()
        self.assertFalse(self.out('retirement').exists());self.assert_originals_intact()

    def test_changed_unknown_report_or_producer_rejected(self):
        (self.source/'report.json').write_text('changed')
        with self.assertRaisesRegex(ValueError,'report identity'):self.call_retire()
        self.assert_originals_intact()
        (self.source/'report.json').write_text('{"fixture":true}\n');self.value['tool']['sha256']='0'*64;self.rewrite_manifest()
        with self.assertRaisesRegex(ValueError,'producer'):self.call_retire()
        self.assert_originals_intact()

    def test_existing_changed_original_and_symlink_block_recovery(self):
        self.call_retire();p=Path(self.records[0]['original']['path']);p.write_bytes(b'other')
        with self.assertRaisesRegex(ValueError,'differs'):self.call_recover()
        self.assertEqual(p.read_bytes(),b'other');p.unlink();p.symlink_to(self.root/'absent')
        with self.assertRaisesRegex(ValueError,'non-symlink'):self.call_recover()
        self.assertTrue(p.is_symlink())

    def test_wrapper_rejects_nonallowlisted_training_and_raw_paths(self):
        for kind in ('training/raw.json','other/forecast.json'):
            value=copy.deepcopy(self.value);value['originals'][0]['original']['path']=str(self.root/kind)
            self.manifest.write_text(json.dumps(value));ref=a.receipt(self.manifest)
            with self.assertRaisesRegex(ValueError,'allowlist'):t.retire(self.manifest,ref['sha256'],self.out('retirement'))
            with self.assertRaisesRegex(ValueError,'allowlist'):t.recover(self.manifest,ref['sha256'],self.out('recovery'))
        self.assert_originals_intact()

    def test_wrong_manifest_sha_and_namespace_refused(self):
        with self.assertRaisesRegex(ValueError,'manifest SHA'):t.retire(self.manifest,'0'*64,self.out('retirement'))
        with self.assertRaisesRegex(ValueError,'namespace'):t.retire(self.manifest,self.ref['sha256'],self.root/'training')
        self.assert_originals_intact()

    def test_execute_switch_is_mandatory(self):
        for command in ('retire','recover'):
            result=subprocess.run(['python','-m','tools.predictor_validation_20261027.unknown_preflight_retire_v2',command,
                '--manifest',str(self.manifest),'--manifest-sha256',self.ref['sha256'],'--output',str(self.out('retirement'))],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('--execute',result.stderr)


if __name__=='__main__':unittest.main()
