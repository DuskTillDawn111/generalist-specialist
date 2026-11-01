"""Small adapter checks; real evidence is never compressed or removed."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import reference_feasibility_archive_v22 as v


class FixedArchiveTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);(self.root/'artifacts').mkdir()
        self.source=self.root/'artifacts/source';self.source.mkdir()
        allowed={};cases=[]
        for cid in ('a','b','c','d'):
            path=self.source/('forecast-'+cid+'.json');path.write_bytes((cid.encode()+b'\x00\xff')*3000)
            ref=v.a.receipt(path);allowed[cid]=(ref['sha256'],path.stat().st_size)
            cases.append(dict(candidate_id=cid,result=ref))
        report=dict(status='RELAXED_REFERENCE_FEASIBILITY_EXECUTED',GPU_started=False,protocol_sealed=False,
                    full_prediction_preflight=False,physical_service_or_gain_validated=False,cases=cases)
        (self.source/'report.json').write_text(json.dumps(report))
        for key,value in dict(ROOT=self.root,SOURCE=self.source,ALLOWED=allowed,
                              REPORT_SHA=v.a.receipt(self.source/'report.json')['sha256'],FLOOR=0).items():
            p=patch.object(v,key,value);p.start();self.addCleanup(p.stop)

    def out(self,op,suffix='test'):
        return self.root/'artifacts'/(v.PREFIX+op+'-'+suffix)

    def archive(self):
        result=v.archive(self.out('archive'));self.manifest=self.out('archive')/'manifest.json'
        self.sha=v.a.receipt(self.manifest)['sha256'];return result

    def test_complete_roundtrip_and_working_retry_command(self):
        frozen=[v.a.receipt(p) for p in (v.a.__file__,v.t.__file__)]
        self.assertEqual(len(self.archive()['originals']),4)
        self.assertEqual(len(v.verify_all(self.manifest,self.sha)[2]),4)
        result=v.transaction('retirement',self.manifest,self.sha,self.out('retirement'))
        self.assertTrue(all(x['missing'] for x in v.t.current_target_presence(json.loads(self.manifest.read_text())['originals'])))
        self.assertEqual(result['restore_command'][3:6],['--dataset','v22','recover'])
        result=v.transaction('recovery',self.manifest,self.sha,self.out('recovery'))
        self.assertEqual(len(result['progress']),4)
        command=result['restore_command'];retry=Path(command[command.index('--output')+1])
        self.assertFalse(retry.exists());self.assertTrue(retry.name.endswith('-retry'))
        self.assertEqual([v.a.receipt(p) for p in (v.a.__file__,v.t.__file__)],frozen)

    def test_fixed_path_and_hash_reject_before_retirement(self):
        self.archive();manifest=json.loads(self.manifest.read_text())
        manifest['originals'][0]['original']['path']=str(self.source/'other.json')
        self.manifest.write_text(json.dumps(manifest));sha=v.a.receipt(self.manifest)['sha256']
        with self.assertRaisesRegex(ValueError,'allowlist'):v.transaction('retirement',self.manifest,sha,self.out('retirement'))
        self.assertEqual(len(list(self.source.glob('forecast-*.json'))),4)
        with self.assertRaises(ValueError):v.output_path(self.root/'wrong','archive')

    def test_partial_retirement_failure_restores_only_missing(self):
        self.archive();unlink=v.t.os.unlink;calls=[]
        def fail(path,*args,**kwargs):
            calls.append(str(path))
            if len(calls)==2:raise PermissionError('second unlink failed')
            return unlink(path,*args,**kwargs)
        with patch.object(v.t.os,'unlink',side_effect=fail):
            with self.assertRaises(PermissionError):v.transaction('retirement',self.manifest,self.sha,self.out('retirement'))
        failure=json.loads((self.out('retirement')/'failed-retirement.json').read_text())
        self.assertEqual(sum(x['missing'] for x in failure['current_targets']),1)
        self.assertEqual(len(v.transaction('recovery',self.manifest,self.sha,self.out('recovery'))['progress']),1)

    def test_only_fixed_derived_filename_and_completed_report_qualify(self):
        path=self.source/'parsed-measured.json';path.write_bytes(b'{"derived":true}\n')
        ref=v.a.receipt(path)
        report=dict(status='PASS',issues=[],requests=57,training_only=True,validation_started=False,
                    model_fitted=False,trace=dict(parsed=ref))
        (self.source/'report.json').write_text(json.dumps(report))
        with patch.object(v,'IS_DERIVED',True),patch.object(v,'DATASET','light-balanced-v23'), \
                patch.object(v,'REPORT_SHA',v.a.receipt(self.source/'report.json')['sha256']), \
                patch.object(v,'ALLOWED',{'parsed_measured':(ref['sha256'],path.stat().st_size)}):
            self.assertEqual(len(self.archive()['originals']),1)
            v.transaction('retirement',self.manifest,self.sha,self.out('retirement'))
            self.assertFalse(path.exists())
            result=v.transaction('recovery',self.manifest,self.sha,self.out('recovery'))
            self.assertEqual(len(result['progress']),1)
            self.assertEqual(v.a.receipt(path),ref)
            self.assertTrue(result['selected_copy_is_derived_training'])
            self.assertFalse(result['raw_or_collection_source_files_touched'])


if __name__=='__main__':unittest.main()
