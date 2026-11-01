"""Independent recovery generation identity/closure checks; no native execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.predictor_validation_20261027 import twoFIFO_recovery_generated_v1 as m


class IndependentRecoveryGeneratedTests(unittest.TestCase):
    def test_design_rejects_omitted_closure_and_old_recovery_binding(self):
        prior=json.loads(Path('artifacts/predictor-twoFIFO-source-rebound-generated-v1-20261029-cpu01-prepared/design.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'manifest.json';p.write_text('{}');ref=m.receipt(p)
            with patch.object(m.response,'audit',return_value=dict(sources=[],recovery_manifest=ref)):
                prepared=m.prepare(p,prior['scoring_plan']['path'],prior['quality']['path'],root/'prepared')
                doc=m.checked(prepared)
                for change in ('closure','producer'):
                    mutant=copy.deepcopy(doc)
                    if change=='closure':mutant['sources']=[]
                    else:mutant['recovery_manifest']=dict(path='old-manifest',sha256='0'*64)
                    q=root/(change+'.json');q.write_text(json.dumps(mutant))
                    with self.subTest(change=change),self.assertRaises(ValueError):m.audit(q)
    def test_foreign_saved_fit_rejected_before_bundle_or_generation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'design.json';p.write_text('{}')
            with patch.object(m,'audit',return_value=dict(fit_manifest=dict(path='expected',sha256='a'))),patch.object(m.response,'audit_fit',return_value=dict(report=dict(manifest=dict(path='foreign',sha256='b')))),patch.object(m.response,'with_verified_bundle') as bundle,patch.object(m.numerical,'evaluate_bundle') as generate:
                with self.assertRaises(ValueError):m.run(p,root/'fit.json',root/'out')
            bundle.assert_not_called();generate.assert_not_called();self.assertFalse((root/'out').exists())
    def test_changed_intake_rejected_before_numerical_generation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def put(name,doc):
                p=root/name;p.write_text(json.dumps(doc));return m.receipt(p)
            design_ref=put('design.json',{});fit_ref=put('fit.json',{});manifest_ref=put('manifest.json',{})
            final_ref=put('final.json',{})
            design=dict(fit_manifest=manifest_ref)
            fit=dict(manifest=manifest_ref,continuation=final_ref,intake_evidence=dict(schema='recovery',sources=[]))
            def bundle(_m,_f,callback):return callback({},dict(schema='foreign-intake',sources=[]))
            with patch.object(m,'audit',return_value=design),patch.object(m.response,'audit_fit',return_value=dict(report=fit,receipt=fit_ref)),patch.object(m.response,'with_verified_bundle',side_effect=bundle),patch.object(m.numerical,'evaluate_bundle') as generate:
                with self.assertRaises(ValueError):m.run(design_ref['path'],fit_ref['path'],root/'out')
            generate.assert_not_called();self.assertFalse((root/'out').exists())
    def test_score_evidence_changed_during_generation_prevents_final_report(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def put(name,doc):
                p=root/name;p.write_text(json.dumps(doc));return m.receipt(p)
            design_ref=put('design.json',{});fit_ref=put('fit.json',{});manifest_ref=put('manifest.json',{})
            final_ref=put('final.json',{});score_ref=put('score.json',{});quality_ref=put('quality.json',{})
            rec_ref=put('recovery.json',{});intake=dict(schema=m.response.intake.SCHEMA,sources=[])
            design=dict(fit_manifest=manifest_ref,scoring_plan=score_ref,quality=quality_ref,recovery_manifest=rec_ref,sources=[score_ref,quality_ref])
            fit=dict(manifest=manifest_ref,continuation=final_ref,intake_evidence=intake,fit={},sources=[])
            def verified(_m,_f,callback):return callback({},intake)
            def evaluate(_b,_f,_s,_q,output):
                Path(output).mkdir();Path(score_ref['path']).write_text('changed');return {}
            with patch.object(m,'audit',return_value=design),patch.object(m.response,'audit',return_value={}),patch.object(m.response,'check_intake'),patch.object(m.response,'audit_fit',return_value=dict(report=fit,receipt=fit_ref)),patch.object(m.response,'with_verified_bundle',side_effect=verified),patch.object(m.numerical,'evaluate_bundle',side_effect=evaluate),patch.object(m.response,'fit_bundle',side_effect=AssertionError('no refit')):
                with self.assertRaises(ValueError):m.run(design_ref['path'],fit_ref['path'],root/'out')
            self.assertFalse((root/'out'/'report.json').exists())


if __name__=='__main__':unittest.main()
