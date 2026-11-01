"""Full OCR model selection/source audits and unchanged parent metrics."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
from tests import test_flow_validation as fixtures
from tools.predictor_validation_20261027 import ocr_trained_validation as entry
from tools.predictor_validation_20261027.native_flow import receipt
from tools.predictor_validation_20261027.profiling_analysis import digest


class OcrValidationTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.FlowValidationTests('test_full_arrivals_parent_receipt_and_wait_service_separated')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        f=self.fixture;f.plan['phase']='predictor_validation';f.write('plan.json',f.plan)
        self.valid={'verified':True,'variant_full_model_verified':True,'variant':'ocr_handler',
            'strict_variant_full_identity_verified':True,
            'training_flow_count':8,'omitted_flow_ids':[],'omitted_seed':None,'validation_sources_used':[]}
        self.verifier_source=receipt(entry.__file__)

    def _verifier(self,result=None,side_effect=None):
        from unittest.mock import Mock
        return patch.object(entry,'_verifier',return_value=(Mock(return_value=result or self.valid,side_effect=side_effect),self.verifier_source))

    def test_rejects_cv_other_variant_missing_audit_or_validation_sources(self):
        for update in ({'variant':'phase_ridge1'},{'training_flow_count':4},{'omitted_flow_ids':['x']},
                       {'omitted_seed':1},{'variant_full_model_verified':False},{'validation_sources_used':['raw']},
                       {'strict_variant_full_identity_verified':False}):
            with self._verifier({**self.valid,**update}):
                with self.assertRaises(ValueError):entry.audit_model(self.fixture.model_path)

    def test_new_full_verifier_failure_stops_before_any_report(self):
        f=self.fixture
        with self._verifier(),patch.object(entry,'audit_plan',return_value=f.plan):
            sealed=entry.seal(f.plan_path,f.model_path,f.resources_path,f.quality_path,f.root/'sealed')
        raws=[f.write(cid+'.json',f.raw(cid,sealed['created_ns']+2*fixtures.NS)) for cid in ('base','alternative')]
        with self._verifier(side_effect=ValueError('Changed recursive selected-source receipt')),patch.object(entry,'audit_plan',return_value=f.plan):
            with self.assertRaisesRegex(ValueError,'recursive'):entry.report(f.root/'sealed',raws,f.root/'report')
        self.assertFalse((f.root/'report').exists())

    def test_frozen_report_parent_receipt_arithmetic_retained(self):
        f=self.fixture
        with self._verifier(),patch.object(entry,'audit_plan',return_value=f.plan):
            sealed=entry.seal(f.plan_path,f.model_path,f.resources_path,f.quality_path,f.root/'sealed')
            raws=[f.write(cid+'.json',f.raw(cid,sealed['created_ns']+2*fixtures.NS)) for cid in ('base','alternative')]
            report=entry.report(f.root/'sealed',raws,f.root/'report')
        self.assertEqual(report['status'],'ASSESSED');self.assertEqual(report['candidates'][0]['timing_errors']['e2e']['mae'],10)
        self.assertFalse(report['final_four_groups_enabled'])

    def test_audit_cannot_borrow_another_existing_entry_source(self):
        f=self.fixture
        with self._verifier(),patch.object(entry,'audit_plan',return_value=f.plan):
            entry.seal(f.plan_path,f.model_path,f.resources_path,f.quality_path,f.root/'sealed')
        audit_path=f.root/'sealed/training-audit.json';audit=entry.read(audit_path)
        audit['entry_source']=receipt(f.plan_path);audit_path.write_text(__import__('json').dumps(audit))
        with self._verifier(),patch.object(entry,'audit_plan',return_value=f.plan):
            with self.assertRaisesRegex(ValueError,'actual entry'):entry.check_sealed(f.root/'sealed')

    def test_exact_plan_rejects_any_seed_count_threshold_or_entry_drift(self):
        f=self.fixture;plan=copy.deepcopy(f.plan)
        threshold=f.write('threshold-plan.json',{'validation_thresholds':entry.LIMITS})
        addendum=f.write('threshold-addendum.json',{'schema':'gsdeploy.training_cv_selection_threshold_source_addendum.v1',
            'thresholds':entry.LIMITS,'threshold_plan':receipt(threshold)})
        plan.update(seed=entry.SEED,split='validation',window_seconds=90.,validation_thresholds=entry.LIMITS,
            final_confirmation_enabled=False,validation_entry=receipt(entry.__file__),threshold_source_addendum=receipt(addendum))
        plan['trace']=[dict(plan['trace'][0],request_id=str(i),offset_ns=i) for i in range(181)]
        candidate=copy.deepcopy(plan['candidates'][0]);plan['candidates']=[]
        for cid in entry.CANDIDATES:
            c=copy.deepcopy(candidate);c.update(id=cid,routes={r['request_id']:next(iter(f.plan['aliases'])) for r in plan['trace']});plan['candidates'].append(c)
        plan['baseline_candidate_id']=entry.CANDIDATES[0];plan.pop('startup_comparison_policy',None)
        plan.pop('plan_sha256',None);plan['plan_sha256']=digest(plan);path=f.write('ocr-plan.json',plan)
        entry.audit_plan(path)
        for key,value in [('seed',627501),('window_seconds',89.),('validation_thresholds',dict(entry.LIMITS,service_mare_max=.3)),('validation_entry',receipt(f.plan_path))]:
            changed=copy.deepcopy(plan);changed[key]=value;changed.pop('plan_sha256');changed['plan_sha256']=digest(changed);f.write('ocr-plan.json',changed)
            with self.assertRaises(ValueError):entry.audit_plan(path)


if __name__=='__main__':unittest.main()
