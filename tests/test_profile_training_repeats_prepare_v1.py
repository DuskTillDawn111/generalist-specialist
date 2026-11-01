"""CPU directory/gate/identity tests; no collection subprocess launched."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import profile_training_repeats_prepare_v1 as r


class RepeatPreparationTests(unittest.TestCase):
    def fixture(self,root):
        packet=root/'packet';r.campaign.prepare(packet,progress_observer=True)
        reports={}
        for mode in r.campaign.MODES:
            smoke=packet/'smoke'/mode
            r.save(smoke/'raw.json',{})
            r.save(smoke/'fixture-execution.json',{})
            r.save(smoke/'smoke-report.json',dict(schema=r.campaign.SMOKE_SCHEMA,status='PASS',issues=[],
                mode=mode,plan=r.receipt(smoke/'plan.json'),raw=r.receipt(smoke/'raw.json'),
                execution=r.receipt(smoke/'fixture-execution.json')))
            run=packet/'training'/mode
            for name in ('raw.json','executed-jobs.json','attempt.json','fixture-execution.json'):
                r.save(run/name,{})
            review=dict(schema='gsdeploy.profile_training_evidence_review.v1',
                status='PASS_COMPLETE_TRAINING_COLLECTION_NOT_PREDICTOR_QUALIFICATION',
                role=dict(mode=mode,split='training'),training_feature_extraction_eligible=True,
                cohort=dict(jobs=26,measured_requests=50,priming_requests=18),
                inputs={name:r.receipt(run/name) for name in
                        ('plan.json','context.json','raw.json','executed-jobs.json','attempt.json')},
                execution=r.receipt(run/'fixture-execution.json'),sources=[])
            path=root/(mode+'-review.json');r.save(path,review);reports[mode]=path
        return packet,reports

    def prepare(self,root,packet,reports):
        # Fixture substitute only for historical execution receipts. Actual
        # smoke report, v6 source/mode/pair audits and v3 manifests remain real.
        with patch.object(r.supervisor,'completed_execution',return_value={}),patch.object(r.supervisor,'execute') as launch:
            result=r.prepare(packet,root/'prepared',natural_review=reports['natural'],kernel_review=reports['kernel'])
            launch.assert_not_called()
        return result

    def test_four_distinct_targets_keep_identity_and_original_smoke_gate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);packet,reviews=self.fixture(root);ref=self.prepare(root,packet,reviews)
            result=r.checked(ref)
            self.assertEqual([(s['capture_index'],s['mode']) for s in result['schedule']],list(r.ORDER))
            self.assertEqual([s['planned_campaign_execution_ordinal'] for s in result['schedule']],[3,4,5,6])
            self.assertFalse(result['GPU_started']);self.assertFalse(result['execution_closure_changed'])
            for row in result['schedule']:
                target=Path(row['target']);self.assertEqual(target.parents[1],packet)
                self.assertEqual(set(p.name for p in target.iterdir()),{'plan.json','context.json','capture-metadata.json'})
                for name in ('plan.json','context.json'):
                    self.assertEqual((target/name).read_bytes(),(packet/'training'/row['mode']/name).read_bytes())
                manifest=r.checked(row['supervisor_manifest'])
                self.assertEqual(manifest['budget'],r.supervisor.BUDGET)
                self.assertEqual(manifest['preflight_free_bytes_required'],18*2**30)
                meta=r.checked(row['capture_metadata'])
                self.assertIsNone(meta['actual_execution_ordinal']);self.assertFalse(meta['paired_timing_unperturbed'])

    def test_repeat_preparation_cannot_overwrite_existing_or_failed_target(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);packet,reviews=self.fixture(root)
            target=packet/'training-repeat02/natural';target.mkdir(parents=True)
            r.save(target/'raw.json',dict(status='failed'))
            with patch.object(r.supervisor,'completed_execution',return_value={}):
                with self.assertRaisesRegex(FileExistsError,'no reuse/retry'):
                    r.prepare(packet,root/'prepared',natural_review=reviews['natural'],kernel_review=reviews['kernel'])
            self.assertFalse((root/'prepared').exists())

    def test_missing_successful_smoke_is_not_bypassed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);packet,reviews=self.fixture(root)
            path=packet/'smoke/kernel/smoke-report.json';value=json.loads(path.read_text());value['status']='FAIL';path.write_text(json.dumps(value))
            with patch.object(r.supervisor,'completed_execution',return_value={}):
                with self.assertRaisesRegex(ValueError,'successful smoke'):
                    r.prepare(packet,root/'prepared',natural_review=reviews['natural'],kernel_review=reviews['kernel'])
            self.assertFalse((packet/'training-repeat02').exists())

    def test_changed_original_review_binding_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);packet,reviews=self.fixture(root)
            path=packet/'training/natural/raw.json';path.write_text('{"changed":true}')
            with patch.object(r.supervisor,'completed_execution',return_value={}):
                with self.assertRaisesRegex(ValueError,'origin run bindings'):
                    r.prepare(packet,root/'prepared',natural_review=reviews['natural'],kernel_review=reviews['kernel'])

    def test_attempted_repeat_fails_actual_standard_gate_on_audit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);packet,reviews=self.fixture(root);ref=self.prepare(root,packet,reviews)
            result=r.checked(ref);r.save(Path(result['schedule'][0]['target'])/'attempt.json',{})
            with patch.object(r.supervisor,'completed_execution',return_value={}):
                with self.assertRaisesRegex(FileExistsError,'already attempted; no retry'):r.audit(ref['path'])

    def test_new_tool_is_outside_frozen_execution_closure(self):
        sources=r.implementation_sources(r.campaign.__file__,r.supervisor.__file__)
        self.assertFalse(any(p['path']==str(Path(r.__file__).resolve()) for p in sources))


if __name__=='__main__':unittest.main()
