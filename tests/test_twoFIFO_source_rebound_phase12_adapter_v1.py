import copy, json, tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_source_rebound_phase12_adapter_v1 as m
from tests.test_twoFIFO_training_intake_v4 import IntakeTests, documents


def receipt(path):
    import hashlib
    p=Path(path); return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def report_fixture():
    root=Path(tempfile.mkdtemp()); manifest=root/'manifest'; continuation=root/'continuation'; manifest.write_text('m'); continuation.write_text('c')
    manifest_ref=receipt(manifest); continuation_ref=receipt(continuation)
    attempts=[]
    for i in range(12):
        review=root/f'review{i}'; review.write_text('r'+str(i)); macro=root/f'macro{i}'; macro.write_text('m'+str(i))
        attempts.append({'attempt_id':f'a{i}','capture_id':f'block{i//4+1}','shard_index':i%4+1,
            'execution_id':f'e{i}','training_review':receipt(review),'macros':receipt(macro)})
    report={'schema':'gsdeploy.twoFIFO_source_rebound_exact12_training_intake.v1',
      'status':'COMPLETE_SOURCE_REBOUND_EXACT12_TRAINING_INTAKE_NOT_FIT','manifest':manifest_ref,
      'continuation':continuation_ref,'attempt_count':12,'measured_requests':546,'priming_requests':216,
      'validation_labels_read':False,'numerical_fit_run':False,'fitted_root_qualification':False,
      'downstream_old_fit_compatible':False,'attempts':attempts}
    rp=root/'intake.json';rp.write_text(json.dumps(report));return rp,report


class AdapterTests(unittest.TestCase):
    def test_exact_intake_gate_rejects_old_fit_or_partial_schema(self):
        path,report=report_fixture()
        for key,value in [('schema','old'),('status','WAITING'),('attempt_count',11),('measured_requests',545),('priming_requests',215),('validation_labels_read',True),('numerical_fit_run',True),('downstream_old_fit_compatible',True)]:
            bad=copy.deepcopy(report);bad[key]=value; p=path.parent/f'bad-{key}';p.write_text(json.dumps(bad))
            with self.subTest(key=key),self.assertRaises(ValueError):m._load_report(str(p))

    def test_rederivation_requires_new_review_macro_and_dynamic_manifest(self):
        path,report=report_fixture(); root=path.parent
        review={'schema':'x','validation_labels_read':False,'issues':[],'training_feature_extraction_eligible':True}
        with patch.object(m.native_review,'review',side_effect=AssertionError('should not reach old schema')):
            with self.assertRaises(Exception):m._rederive(report,workdir=root/'w')
        bad=copy.deepcopy(report);bad['attempts'][3]['training_review']=bad['attempts'][2]['training_review']
        with self.assertRaises(ValueError):m._rederive(bad,workdir=root/'w2')

    def test_fold_contract_excludes_all_four_shards_and_keeps_flags(self):
        rows=[]
        for block in ('block1','block2','block3'):
            for shard in range(4):
                rows.extend([{'capture':block,'attempt_id':f'{block}-{shard}','group':(block,shard),
                    'matrix':__import__('numpy').zeros((1,12)),'durations':__import__('numpy').ones(1)}])
        # Adapter's public result contract is explicit even when numeric rows are synthetic.
        result={'whole_block_folds':[]}
        for heldout in ('block1','block2','block3'):
            train=[r for r in rows if r['capture']!=heldout]; test=[r for r in rows if r['capture']==heldout]
            result['whole_block_folds'].append({'heldout_block_id':heldout,
                'training_attempt_ids':sorted({r['attempt_id'] for r in train}),
                'heldout_attempt_ids':sorted({r['attempt_id'] for r in test}),
                'heldout_rows_excluded_from_training':True,'fit_required':True,'fit_performed':False})
        self.assertTrue(all(len(f['training_attempt_ids'])==8 and len(f['heldout_attempt_ids'])==4
                            and f['heldout_rows_excluded_from_training'] and not f['fit_performed'] for f in result['whole_block_folds']))

    def test_no_fit_gpu_labels_and_physical_usher_claims(self):
        path,report=report_fixture(); root=path.parent
        # The result contract remains a schema adapter, never qualification.
        contract={'schema':m.SCHEMA,'status':'ROWS_READY_FOR_SEPARATE_PHASE12_FIT_NOT_FIT',
            'root_numerical_qualification':False,'USHER_physical_qualification':False,
            'validation_labels_read':False,'numerical_fit_run':False,'GPU_started':False,
            'heldout_rows_never_used_for_fit':True,'downstream_fit_interface':'separate'}
        self.assertFalse(contract['root_numerical_qualification']);self.assertFalse(contract['USHER_physical_qualification'])
        self.assertFalse(contract['validation_labels_read']);self.assertFalse(contract['numerical_fit_run']);self.assertFalse(contract['GPU_started'])

    def test_new_schema_rows_compile_full_and_three_whole_block_folds_without_fit(self):
        source=IntakeTests(); source.setUpClass()
        docs=documents(source.plans)
        for doc in docs:
            doc['schema']=m.MACRO_SCHEMA;doc['status']='COMPLETE_ATTEMPT_DISJOINT_TRAINING_MACROS_NOT_FITTED_PROVIDER'
            doc['fitted_model_qualified']=False;doc['source_rebound_manifest']={}
            for request in doc['requests']:
                request['capture_id']=doc['capture_id'];request['attempt_id']=doc['attempt_id'];request['execution_id']=doc['execution_id']
                request['pool_id']=request['macros'][0]['pool_id']
                for macro in request['macros']:
                    macro['training_replication_unit']=doc['capture_id'];macro['attempt_id']=doc['attempt_id']
                    macro['capture_sha256']=macro.get('capture_sha256','raw')
        # Reuse only the public CPU fixture's new macro shape; no old raw capture
        # is supplied to the adapter.
        report={'schema':m.source_intake.SCHEMA,'status':'COMPLETE_SOURCE_REBOUND_EXACT12_TRAINING_INTAKE_NOT_FIT',
            'manifest':{'path':'manifest','sha256':'x'},'continuation':{'path':'continuation','sha256':'y'},
            'attempt_count':12,'measured_requests':546,'priming_requests':216,'validation_labels_read':False,
            'numerical_fit_run':False,'fitted_root_qualification':False,'downstream_old_fit_compatible':False,
            'attempts':[{'attempt_id':p['attempt_id'],'capture_id':p['capture_id'],'shard_index':p['shard_index'],
                'execution_id':p['execution_id'],'training_review':{},'macros':{}} for p in source.plans]}
        with patch.object(m,'_load_report',return_value=report), patch.object(m,'_rederive',return_value=(source.plans,docs,[])):
            result=m.build_training_rows('intake.json',workdir=tempfile.mkdtemp())
        self.assertEqual(len(result['full']['rows']),546);self.assertEqual(len(result['whole_block_folds']),3)
        self.assertTrue(all(len(f['training_attempt_ids'])==8 and len(f['heldout_attempt_ids'])==4
                            and f['heldout_rows_excluded_from_training'] and not f['fit_performed']
                            for f in result['whole_block_folds']))
        self.assertFalse(result['root_numerical_qualification']);self.assertFalse(result['USHER_physical_qualification'])
        self.assertFalse(result['numerical_fit_run']);self.assertFalse(result['GPU_started'])

if __name__=='__main__':unittest.main()
