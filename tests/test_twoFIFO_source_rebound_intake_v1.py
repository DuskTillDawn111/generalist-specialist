import copy, hashlib, json, tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_source_rebound_intake_v1 as m
from tools.predictor_validation_20261027 import twoFIFO_source_rebound_execute_v1 as executor
from tools.predictor_validation_20261027 import twoFIFO_source_rebound_review_v1 as review
from tools.predictor_validation_20261027 import twoFIFO_source_rebound_macros_v1 as macros


def write(root,name,value):
    p=root/name; p.write_text(json.dumps(value)); return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def fixture():
    root=Path(tempfile.mkdtemp()); manifest_path=root/'manifest.json'; continuation_path=root/'final.json'
    plans=[]; reviews=[]; macro_refs=[]; completed=[]; review_docs={}; macro_docs={}
    for i in range(12):
        capture='block-'+str(i//4+1); aid='attempt-'+str(i); eid='e'*32+str(i)[:0]
        plan={'schema':executor.original.SCHEMA+'.attempt','split':'training','no_retries':True,
            'attempt_id':aid,'execution_id':eid,'capture_id':capture,'shard_index':i%4+1,'shards_in_block':4,
            'cases':[]}
        pref=write(root,Path('plans')/str(i) if False else Path('plan'+str(i)+'.json'),plan); plans.append(pref)
        raw={'schema':review.RAW_SCHEMA,'backend':'native','status':'completed','attempt_id':aid,'execution_id':eid,'plan':pref}
        rawref=write(root,'raw'+str(i)+'.json',raw)
        rr={'schema':review.SCHEMA,'status':'PASS_COMPLETE_SOURCE_REBOUND_NATIVE_TRAINING_NOT_PREDICTOR_QUALIFICATION',
            'training_feature_extraction_eligible':True,'issues':[],'validation_labels_read':False,
            'source_rebound_manifest':None,'raw':rawref,'execution':None}
        rrref=write(root,'review'+str(i)+'.json',rr); reviews.append(rrref); review_docs[rrref['path']]=rr
        doc={'schema':'gsdeploy.twoFIFO_source_rebound_composable_response_macro_training.v1',
            'status':'COMPLETE_ATTEMPT_DISJOINT_TRAINING_MACROS_NOT_FITTED_PROVIDER','mode':'natural',
            'validation_labels_read':False,'source_rebound_manifest':None,'whole_attempt_history_used_for_visible_prefix':True,
            'case_history_reset':False,'capture_id':capture,'attempt_id':aid,'execution_id':eid,
            'requests':[],'measured_requests':45 if i<6 else 46,'priming_rows_excluded_as_labels':18,
            'fitted_model_qualified':False}
        mr=write(root,'macro'+str(i)+'.json',doc);macro_refs.append(mr);macro_docs[mr['path']]=doc
        exref=write(root,'exec'+str(i)+'.json',{'ok':True})
        completed.append({'attempt_id':aid,'execution':exref,'training_review':rrref,'macros':mr})
    manifest={'attempts':[{'plan':p,'attempt_id':'attempt-'+str(i),'execution_id':'e'*32} for i,p in enumerate(plans)]}
    manref=write(root,'manifest.json',manifest)
    for rr in reviews:
        d=review_docs[rr['path']];d['source_rebound_manifest']=manref;Path(rr['path']).write_text(json.dumps(d));rr['sha256']=hashlib.sha256(Path(rr['path']).read_bytes()).hexdigest()
    for mr in macro_refs:
        d=macro_docs[mr['path']];d['source_rebound_manifest']=manref;Path(mr['path']).write_text(json.dumps(d));mr['sha256']=hashlib.sha256(Path(mr['path']).read_bytes()).hexdigest()
    final={'schema':executor.CONT_SCHEMA+'.run','status':m.FINAL_STATUS,'no_retries':True,'fit_eligible':False,
        'numerical_fit_run':False,'validation_labels_read':False,'manifest':manref,'completed_attempts':completed}
    finref=write(root,'final.json',final)
    # Recompute refs after edits.
    final=json.loads(Path(finref['path']).read_text())
    for row,rr,mr in zip(final['completed_attempts'],reviews,macro_refs):row['training_review']=rr;row['macros']=mr
    Path(finref['path']).write_text(json.dumps(final));finref['sha256']=hashlib.sha256(Path(finref['path']).read_bytes()).hexdigest()
    return root,manref,finref,plans,reviews,macro_refs,review_docs,macro_docs


class IntakeTests(unittest.TestCase):
    def test_exact12_contract_and_downstream_gap(self):
        root,man,fin,plans,reviews,mrs,rds,mds=fixture(); manifest=json.loads(Path(man['path']).read_text())
        # Make plan refs point to the actual plan files and preserve manifest identities.
        manifest['attempts']=[dict(x,plan=plans[i]) for i,x in enumerate(manifest['attempts'])];Path(man['path']).write_text(json.dumps(manifest));man['sha256']=hashlib.sha256(Path(man['path']).read_bytes()).hexdigest()
        final=json.loads(Path(fin['path']).read_text());final['manifest']=man
        for row,rr,mr in zip(final['completed_attempts'],reviews,mrs):row.update(training_review=rr,macros=mr)
        Path(fin['path']).write_text(json.dumps(final));fin['sha256']=hashlib.sha256(Path(fin['path']).read_bytes()).hexdigest()
        for d in rds.values():d['source_rebound_manifest']=man
        for d in mds.values():d['source_rebound_manifest']=man
        with patch.object(executor,'audit',return_value=manifest), patch.object(m.native_review,'review',return_value=reviews[0]), patch.object(m.macros,'extract',return_value=mds[mrs[0]['path']]):
            # A fixture with empty case calendars is intentionally rejected at macro row shape;
            # test the exact layout/receipt gate directly below.
            with self.assertRaises(ValueError):m.intake(man['path'],fin['path'])
        self.assertEqual(m._attempt_layout([json.loads(Path(p['path']).read_text()) for p in plans]).keys(),{'block-1','block-2','block-3'})

    def test_layout_rejects_missing_shard_duplicate_and_old_schema(self):
        root,man,fin,plans,*_=fixture(); vals=[json.loads(Path(p['path']).read_text()) for p in plans]
        with self.assertRaises(ValueError):m._attempt_layout(vals[:-1])
        vals[1]['shard_index']=vals[0]['shard_index']
        with self.assertRaises(ValueError):m._attempt_layout(vals)
        vals[1]['shard_index']=2; vals[0]['schema']='old.attempt'
        with self.assertRaises(ValueError):m._attempt_layout(vals)

    def test_macro_identity_rejects_fit_labels_missing_priming_and_order(self):
        _,_,_,plans,*_=fixture(); plan=json.loads(Path(plans[0]['path']).read_text())
        doc={'schema':'gsdeploy.twoFIFO_source_rebound_composable_response_macro_training.v1','status':'COMPLETE_ATTEMPT_DISJOINT_TRAINING_MACROS_NOT_FITTED_PROVIDER','mode':'natural','validation_labels_read':False,'source_rebound_manifest':{},'whole_attempt_history_used_for_visible_prefix':True,'case_history_reset':False,'capture_id':plan['capture_id'],'attempt_id':plan['attempt_id'],'execution_id':plan['execution_id'],'requests':[],'measured_requests':0,'priming_rows_excluded_as_labels':18,'fitted_model_qualified':False}
        doc['measured_requests']=1;doc['fitted_model_qualified']=True
        with self.assertRaises(ValueError):m._macro_identity(doc,plan,{})

    def test_final_rejects_old_schema_instead_of_relabeling(self):
        root,man,fin,plans,*_=fixture(); final=json.loads(Path(fin['path']).read_text());final['schema']='old';Path(fin['path']).write_text(json.dumps(final));fin['sha256']=hashlib.sha256(Path(fin['path']).read_bytes()).hexdigest()
        with patch.object(executor,'audit',return_value=json.loads(Path(man['path']).read_text())):
            with self.assertRaises(ValueError):m.intake(man['path'],fin['path'])

if __name__=='__main__':unittest.main()
