"""Independent filesystem evidence tests; all native/numeric boundaries are mocked."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.predictor_validation_20261027 import twoFIFO_source_rebound_intake_v1 as m


def put(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return m.receipt(path)


def fixture(root):
    attempts=[]; rows=[]; review_docs={}; macro_docs={}
    for index in range(12):
        count=45 if index<6 else 46
        aid=f'attempt-{index}'; eid=f'{index+1:032x}'; block=f'block-{index//4}'
        arrivals=[{'request_id':f'{aid}:{j}'} for j in range(count)]
        plan=dict(schema=m.executor.original.SCHEMA+'.attempt',split='training',no_retries=True,
            attempt_id=aid,execution_id=eid,capture_id=block,shard_index=index%4+1,shards_in_block=4,
            cases=[{'arrivals':arrivals}])
        pref=put(root/f'plan-{index}.json',plan)
        rawref=put(root/f'raw-{index}.json',dict(backend='native',status='completed',attempt_id=aid,execution_id=eid,plan=pref))
        eref=put(root/f'exec-{index}.json',{'fixture':True})
        attempts.append(dict(plan=pref,attempt_id=aid,execution_id=eid))
        rows.append(dict(attempt_id=aid,execution=eref))
        review_docs[aid]=dict(schema=m.native_review.SCHEMA,status='PASS_COMPLETE_SOURCE_REBOUND_NATIVE_TRAINING_NOT_PREDICTOR_QUALIFICATION',
            training_feature_extraction_eligible=True,issues=[],validation_labels_read=False,raw=rawref,execution=eref,sources=[])
        macro_docs[aid]=dict(schema='gsdeploy.twoFIFO_source_rebound_composable_response_macro_training.v1',
            status='COMPLETE_ATTEMPT_DISJOINT_TRAINING_MACROS_NOT_FITTED_PROVIDER',mode='natural',validation_labels_read=False,
            whole_attempt_history_used_for_visible_prefix=True,case_history_reset=False,capture_id=block,attempt_id=aid,
            execution_id=eid,requests=arrivals,measured_requests=count,priming_rows_excluded_as_labels=18,fitted_model_qualified=False,sources=[])
    manifest={'attempts':attempts}; manref=put(root/'manifest.json',manifest)
    for row in rows:
        aid=row['attempt_id']
        review_docs[aid]['source_rebound_manifest']=manref
        macro_docs[aid]['source_rebound_manifest']=manref
        row['training_review']=put(root/(aid+'-review.json'),review_docs[aid])
        row['macros']=put(root/(aid+'-macro.json'),macro_docs[aid])
    final=dict(schema=m.executor.CONT_SCHEMA+'.run',status=m.FINAL_STATUS,no_retries=True,fit_eligible=False,
        numerical_fit_run=False,validation_labels_read=False,automatic_fit=False,manifest=manref,completed_attempts=rows)
    finalref=put(root/'final.json',final)
    return manifest,manref,finalref,review_docs,macro_docs


class LifecycleIndependentTests(unittest.TestCase):
    def invoke(self, root, mutate_final=False):
        manifest,man,fin,reviews,macros=fixture(root)
        temp_outputs=[]; observed=[]
        def rebuild(raw_path, output, execution_path, *, manifest_path):
            doc=next(v for v in reviews.values() if v['raw']['path']==raw_path)
            self.assertEqual(manifest_path,man['path'])
            output=Path(output);output.mkdir()
            temp_outputs.append(output)
            observed.append((raw_path,execution_path))
            return put(output/'report.json',doc)
        def extract(review_path):
            aid=next(a for a in reviews if str(root/(a+'-review.json'))==review_path)
            if mutate_final and aid=='attempt-11':
                Path(fin['path']).write_text('{"changed_during_recheck":true}')
            return copy.deepcopy(macros[aid])
        with patch.object(m.executor,'audit',return_value=manifest), patch.object(m.executor,'completed_execution') as audit_exec, \
             patch.object(m.native_review,'review',side_effect=rebuild),patch.object(m.macros,'extract',side_effect=extract):
            result=m.intake(man['path'],fin['path'])
        return result,temp_outputs,observed,audit_exec.call_count

    def test_success_repeats_real_temporary_rechecks_and_retains_546_requests(self):
        for repeat in range(2):
            with tempfile.TemporaryDirectory() as directory:
                result,temporary,observed,calls=self.invoke(Path(directory))
                self.assertEqual(result['measured_requests'],546)
                self.assertEqual(result['priming_requests'],216)
                self.assertEqual(calls,12)
                self.assertEqual(len(observed),12)
                self.assertTrue(all(not p.exists() for p in temporary))
                self.assertFalse(result['numerical_fit_run'])
                self.assertFalse(result['validation_labels_read'])
                self.assertTrue(all(m.receipt(r['path'])==r for r in result['sources']))

    def test_final_receipt_mutation_during_last_recheck_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                self.invoke(Path(directory),mutate_final=True)

    def test_foreign_final_manifest_rejected_before_any_review_or_macro_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest,man,fin,_,_=fixture(root)
            final=m.checked(fin);final['manifest']={'path':'foreign','sha256':'0'*64};put(Path(fin['path']),final)
            with patch.object(m.executor,'audit',return_value=manifest),patch.object(m.native_review,'review') as review,patch.object(m.macros,'extract') as extract:
                with self.assertRaisesRegex(ValueError,'manifest binding'):
                    m.intake(man['path'],fin['path'])
                review.assert_not_called();extract.assert_not_called()


if __name__=='__main__':unittest.main()
