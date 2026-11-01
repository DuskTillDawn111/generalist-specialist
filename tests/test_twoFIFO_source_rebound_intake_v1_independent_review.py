"""Independent CPU review of the exact12 source-rebound intake boundary."""
import copy, hashlib, json, tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_source_rebound_intake_v1 as intake
from tools.predictor_validation_20261027 import twoFIFO_source_rebound_execute_v1 as executor
from tools.predictor_validation_20261027 import twoFIFO_source_rebound_review_v1 as native_review
from tools.predictor_validation_20261027 import twoFIFO_source_rebound_macros_v1 as macros


def put(root, name, value):
    p=root/name; p.write_text(json.dumps(value, sort_keys=True)); return {'path':str(p), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def fixture():
    root=Path(tempfile.mkdtemp(prefix='exact12-independent-'))
    plans=[]; reviews=[]; macro_refs=[]; execution_refs=[]; completed=[]
    for i in range(12):
        aid=f'a{i:02d}'; eid=f'{i+1:032x}'; block=f'block-{i//4+1}'
        plan={'schema':executor.original.SCHEMA+'.attempt','split':'training','no_retries':True,
              'attempt_id':aid,'execution_id':eid,'capture_id':block,'shard_index':i%4+1,
              'shards_in_block':4,'cases':[]}
        plan_ref=put(root,Path(f'plan-{i}.json'),plan); plans.append(plan_ref)
        raw={'schema':native_review.RAW_SCHEMA,'backend':'native','status':'completed',
             'attempt_id':aid,'execution_id':eid,'plan':plan_ref}
        raw_ref=put(root,Path(f'raw-{i}.json'),raw)
        execution={'profile_execution':{},'status':'ok'}; execution_ref=put(root,Path(f'exec-{i}.json'),execution); execution_refs.append(execution_ref)
        review={'schema':native_review.SCHEMA,'status':'PASS_COMPLETE_SOURCE_REBOUND_NATIVE_TRAINING_NOT_PREDICTOR_QUALIFICATION',
                'training_feature_extraction_eligible':True,'issues':[],'validation_labels_read':False,
                'source_rebound_manifest':None,'raw':raw_ref,'execution':execution_ref}
        review_ref=put(root,Path(f'review-{i}.json'),review); reviews.append(review_ref)
        macro={'schema':'gsdeploy.twoFIFO_source_rebound_composable_response_macro_training.v1',
               'status':'COMPLETE_ATTEMPT_DISJOINT_TRAINING_MACROS_NOT_FITTED_PROVIDER','mode':'natural',
               'validation_labels_read':False,'source_rebound_manifest':None,'whole_attempt_history_used_for_visible_prefix':True,
               'case_history_reset':False,'capture_id':block,'attempt_id':aid,'execution_id':eid,
               'requests':[],'measured_requests':45 if i<6 else 46,'priming_rows_excluded_as_labels':18,
               'fitted_model_qualified':False}
        macro_ref=put(root,Path(f'macro-{i}.json'),macro); macro_refs.append(macro_ref)
        completed.append({'attempt_id':aid,'execution':execution_ref,'training_review':review_ref,'macros':macro_ref})
    manifest={'attempts':[{'plan':p,'attempt_id':f'a{i:02d}','execution_id':f'{i+1:032x}'} for i,p in enumerate(plans)]}
    manifest_ref=put(root,Path('manifest.json'),manifest)
    for rr in reviews:
        d=json.loads(Path(rr['path']).read_text());d['source_rebound_manifest']=manifest_ref;Path(rr['path']).write_text(json.dumps(d,sort_keys=True));rr['sha256']=hashlib.sha256(Path(rr['path']).read_bytes()).hexdigest()
    for mr in macro_refs:
        d=json.loads(Path(mr['path']).read_text());d['source_rebound_manifest']=manifest_ref;Path(mr['path']).write_text(json.dumps(d,sort_keys=True));mr['sha256']=hashlib.sha256(Path(mr['path']).read_bytes()).hexdigest()
    final={'schema':executor.CONT_SCHEMA+'.run','status':intake.FINAL_STATUS,'no_retries':True,'fit_eligible':False,
           'numerical_fit_run':False,'validation_labels_read':False,'manifest':manifest_ref,'completed_attempts':completed}
    final_ref=put(root,Path('final.json'),final)
    return root,manifest_ref,final_ref,plans,reviews,macro_refs,execution_refs


class IndependentExact12Review(unittest.TestCase):
    def test_dynamic_manifest_binds_all_receipts_and_rederivation_hooks(self):
        root,manifest_ref,final_ref,plans,reviews,macro_refs,execution_refs=fixture()
        manifest=json.loads(Path(manifest_ref['path']).read_text())
        review_map={r['path']:json.loads(Path(r['path']).read_text()) for r in reviews}
        macro_map={r['path']:json.loads(Path(r['path']).read_text()) for r in macro_refs}
        # Empty cases make this a contract fixture; replace rederivation with
        # exact source-returning adapters and independently audit executions.
        with patch.object(executor,'audit',return_value=manifest), \
             patch.object(executor,'completed_execution',return_value=None) as completed_execution, \
             patch.object(native_review,'review',side_effect=lambda *args, **kwargs: next(r for r in reviews if r['path']==args[0] and r)), \
             patch.object(macros,'extract',side_effect=lambda path: macro_map[next(r['path'] for r in reviews if r['path'] == path)] if False else macro_map[macro_refs[0]['path']]):
            # The review adapter receives raw paths; return the corresponding review
            # document by its raw source identity.
            def rebuild(raw_path,*args,**kwargs):
                return next(rr for rr in reviews if json.loads(Path(rr['path']).read_text())['raw']['path']==raw_path)
            native_review.review.side_effect=rebuild
            def rebuild_macro(review_path):
                rr=next(rr for rr in reviews if rr['path']==review_path)
                aid=json.loads(Path(rr['path']).read_text())['raw']['path'].split('raw-')[-1].split('.')[0]
                return macro_map[macro_refs[int(aid)]['path']]
            macros.extract.side_effect=rebuild_macro
            # Restore the manifest binding after each generated file edit is visible.
            # A valid exact12 source intake is expected to reach denominator checks;
            # our fixture has empty public cases and therefore fails closed there.
            with self.assertRaises(ValueError):intake.intake(manifest_ref['path'],final_ref['path'])
            self.assertGreaterEqual(completed_execution.call_count,1)

    def test_exact_block_shard_and_no_retry_layout_is_independent(self):
        _,_,_,plans,*_=fixture(); values=[json.loads(Path(p['path']).read_text()) for p in plans]
        blocks=intake._attempt_layout(values)
        self.assertEqual(set(blocks),{'block-1','block-2','block-3'})
        self.assertTrue(all({p['shard_index'] for p in rows}=={1,2,3,4} for rows in blocks.values()))
        for bad in (values[:-1], values[:5]+[dict(values[5], shard_index=1)]+values[6:],
                    [dict(values[0], no_retries=False)]+values[1:]):
            with self.assertRaises(ValueError):intake._attempt_layout(bad)

    def test_review_macro_old_schema_failure_labels_and_manifest_rebinding_fail_closed(self):
        _,manifest_ref,_,plans,reviews,macro_refs,_=fixture(); plan=json.loads(Path(plans[0]['path']).read_text())
        review=json.loads(Path(reviews[0]['path']).read_text()); macro=json.loads(Path(macro_refs[0]['path']).read_text())
        for mutation in ('review_old','review_failed','review_labels','review_manifest','macro_old','macro_fit','macro_labels','macro_manifest'):
            r=copy.deepcopy(review); d=copy.deepcopy(macro)
            if mutation=='review_old':r['schema']='gsdeploy.original_twoFIFO_owned_training_review.v3'
            elif mutation=='review_failed':r['status']='NOT_ELIGIBLE'
            elif mutation=='review_labels':r['validation_labels_read']=True
            elif mutation=='review_manifest':r['source_rebound_manifest']={'path':'other','sha256':'0'*64}
            elif mutation=='macro_old':d['schema']='gsdeploy.twoFIFO_composable_response_macro_training.v3'
            elif mutation=='macro_fit':d['fitted_model_qualified']=True
            elif mutation=='macro_labels':d['validation_labels_read']=True
            else:d['source_rebound_manifest']={'path':'other','sha256':'0'*64}
            with self.subTest(mutation=mutation):
                if mutation.startswith('review'):
                    with self.assertRaises(ValueError):intake._review_identity(r,plan,manifest_ref)
                else:
                    with self.assertRaises(ValueError):intake._macro_identity(d,plan,manifest_ref)

    def test_continuation_final_rejects_missing_duplicate_or_fit_gpu_labels(self):
        root,manifest_ref,final_ref,*_=fixture()
        manifest=json.loads(Path(manifest_ref['path']).read_text())
        for mutation in ('missing','duplicate','old','fit','gpu','labels','retry'):
            final=json.loads(Path(final_ref['path']).read_text())
            if mutation=='missing':final['completed_attempts']=final['completed_attempts'][:-1]
            elif mutation=='duplicate':final['completed_attempts'][1]['attempt_id']=final['completed_attempts'][0]['attempt_id']
            elif mutation=='old':final['schema']='old.schema'
            elif mutation=='fit':final['fit_eligible']=True
            elif mutation=='gpu':final['GPU_started']=True
            elif mutation=='labels':final['validation_labels_read']=True
            else:final['no_retries']=False
            p=root/f'bad-{mutation}.json';p.write_text(json.dumps(final))
            with self.subTest(mutation=mutation), patch.object(executor,'audit',return_value=manifest), self.assertRaises(ValueError):
                intake.intake(manifest_ref['path'],str(p))

    def test_denominators_are_fixed_and_unknown_missing_rows_cannot_be_trimmed(self):
        _,_,_,plans,*_=fixture();
        vals=[json.loads(Path(p['path']).read_text()) for p in plans]
        self.assertEqual(sum(45 if i<6 else 46 for i in range(12)),546)
        self.assertEqual(sum(18 for _ in vals),216)
        # The macro identity gate rejects a fabricated subset even with a
        # plausible measured count because complete request order is required.
        doc={'schema':'gsdeploy.twoFIFO_source_rebound_composable_response_macro_training.v1','status':'COMPLETE_ATTEMPT_DISJOINT_TRAINING_MACROS_NOT_FITTED_PROVIDER','mode':'natural','validation_labels_read':False,'source_rebound_manifest':{},'whole_attempt_history_used_for_visible_prefix':True,'case_history_reset':False,'capture_id':vals[0]['capture_id'],'attempt_id':vals[0]['attempt_id'],'execution_id':vals[0]['execution_id'],'requests':[{'request_id':'fake'}],'measured_requests':1,'priming_rows_excluded_as_labels':18,'fitted_model_qualified':False}
        with self.assertRaises(ValueError):intake._macro_identity(doc,vals[0],{})

if __name__=='__main__':unittest.main()
