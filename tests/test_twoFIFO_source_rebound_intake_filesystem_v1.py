"""Real file rederivation lifecycle with a synthetic exact12 CPU evidence set."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_source_rebound_intake_v1 as m


def fixture(root):
    def save(name, doc):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(doc));return m.receipt(path)
    nested=save('nested.json',dict(cpu_fixture=True))
    plans=[];items=[];review_map={};macro_map={};completed=[]
    for i in range(12):
        aid=f'attempt-{i}';block=f'training-block-{i//4+1}'
        arrivals=[dict(request_id=f'{aid}:r{j}',owner='text#0',task='text',pool_id='input',arrival_ns=j)
                  for j in range(45 if i<6 else 46)]
        plan=dict(schema=m.executor.original.SCHEMA+'.attempt',split='training',no_retries=True,
            attempt_id=aid,execution_id=f'{i:032x}',capture_id=block,shard_index=i%4+1,
            shards_in_block=4,cases=[dict(arrivals=arrivals)])
        plan_ref=save(f'{aid}/plan.json',plan);plans.append(plan_ref)
        execution_ref=save(f'{aid}/execution.json',dict(cpu_fixture=True))
        raw_ref=save(f'{aid}/raw.json',dict(schema=m.native_review.RAW_SCHEMA,split='training',
            backend='native',status='completed',attempt_id=aid,execution_id=plan['execution_id'],plan=plan_ref))
        items.append(dict(plan=plan_ref,attempt_id=aid,execution_id=plan['execution_id']))
        review_map[raw_ref['path']]=dict(raw=raw_ref,execution=execution_ref,
            schema=m.native_review.SCHEMA,status='PASS_COMPLETE_SOURCE_REBOUND_NATIVE_TRAINING_NOT_PREDICTOR_QUALIFICATION',
            training_feature_extraction_eligible=True,issues=[],validation_labels_read=False,sources=[nested])
        macro_map[aid]=dict(schema='gsdeploy.twoFIFO_source_rebound_composable_response_macro_training.v1',
            mode='natural',status='COMPLETE_ATTEMPT_DISJOINT_TRAINING_MACROS_NOT_FITTED_PROVIDER',
            capture_id=block,attempt_id=aid,execution_id=plan['execution_id'],validation_labels_read=False,
            fitted_model_qualified=False,whole_attempt_history_used_for_visible_prefix=True,case_history_reset=False,
            measured_requests=len(arrivals),priming_rows_excluded_as_labels=18,
            requests=[dict(request_id=a['request_id']) for a in arrivals],sources=[nested])
    manifest=dict(attempts=items,sources=[nested])
    manifest_ref=save('manifest.json',manifest)
    extract={}
    for raw_path,review in review_map.items():
        aid=Path(raw_path).parent.name
        review['source_rebound_manifest']=manifest_ref
        rr=save(f'{aid}/review.json',review)
        macro=macro_map[aid];macro['source_rebound_manifest']=manifest_ref
        mr=save(f'{aid}/macros.json',macro);extract[rr['path']]=macro
        completed.append(dict(attempt_id=aid,execution=review['execution'],training_review=rr,macros=mr))
    final_ref=save('final.json',dict(schema=m.executor.CONT_SCHEMA+'.run',status=m.FINAL_STATUS,
        manifest=manifest_ref,no_retries=True,fit_eligible=False,numerical_fit_run=False,
        automatic_fit=False,validation_labels_read=False,completed_attempts=completed))
    return manifest_ref,manifest,final_ref,review_map,extract,nested


class IntakeFilesystemTests(unittest.TestCase):
    def run_fixture(self, root, *, mutate=False):
        manifest_ref,manifest,final_ref,reviews,macros,nested=fixture(root)
        recheck_dirs=[]
        def rederive(raw_path, output, execution_path, *, manifest_path):
            # Unlike a return-only mock, write the actual rederived report into
            # the requested directory. Intake must consume it while it exists.
            output=Path(output);output.mkdir(parents=True,exist_ok=False)
            (output/'report.json').write_text(json.dumps(reviews[raw_path]))
            recheck_dirs.append(output)
            if mutate:
                Path(nested['path']).write_text('{"changed":true}')
            return m.receipt(output/'report.json')
        with patch.object(m.executor,'audit',return_value=manifest), \
             patch.object(m.executor,'completed_execution'), \
             patch.object(m.native_review,'review',side_effect=rederive), \
             patch.object(m.macros,'extract',side_effect=lambda path:copy.deepcopy(macros[path])):
            result=m.intake(manifest_ref['path'],final_ref['path'],output=root/'result')
        return result,recheck_dirs,nested

    def test_twelve_written_rechecks_are_consumed_before_cleanup_and_output_created_once(self):
        with tempfile.TemporaryDirectory() as work:
            root=Path(work)
            result,dirs,nested=self.run_fixture(root)
            report=m.checked(result)
            self.assertEqual(report['measured_requests'],546)
            self.assertEqual(report['priming_requests'],216)
            self.assertEqual(len(dirs),12)
            self.assertTrue(all(not path.exists() for path in dirs))
            self.assertTrue((root/'result'/'report.json').is_file())
            self.assertIn(nested,report['sources'])
            self.assertIn(m.receipt(m.__file__),report['sources'])

    def test_nested_source_drift_rejected_before_final_report(self):
        with tempfile.TemporaryDirectory() as work:
            root=Path(work)
            with self.assertRaisesRegex(ValueError,'changed'):
                self.run_fixture(root,mutate=True)
            self.assertFalse((root/'result'/'report.json').exists())


if __name__=='__main__':unittest.main()
