"""Prospective H600 forecasts preserve public execution plans and strict sources."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from tests import test_flow_validation as fixtures
from tools.predictor_validation_20261027 import h600_trained_validation as entry
from tools.predictor_validation_20261027.native_flow import receipt
from tools.predictor_validation_20261027.profiling import digest


class H600ValidationTests(unittest.TestCase):
    def setUp(self):
        f=fixtures.FlowValidationTests('test_full_arrivals_parent_receipt_and_wait_service_separated')
        f.setUp();self.addCleanup(f.doCleanups);self.f=f
        f.plan.update(phase='predictor_validation',split='validation',seed=627701,
            execution_context_sha256=f.plan['candidates'][0]['execution_context_sha256'])
        f.write('plan.json',f.plan)
        self.valid=dict(verified=True,run_aware_model_verified=True,variant='ocr_handler',
            estimator='duration_variants_WLS.v1',training_flow_count=16,omitted_flow_id=None,
            omitted_flow_ids=[],omitted_seed=None,validation_sources_used=[],diagnostic_repeats_fitted=False,
            capacity_evidence_pooled=False,direct_and_transitive_sources_verified=True)
        self.protocol=f.write('validation-protocol.json',dict(target_plan=receipt(f.plan_path),created_ns=1,source_receipts=[]))
        self.resource_manifest=f.write('resource-manifest.json',{})
        self.overrides={cid:dict(predicted_startup_joules=123.+i,predicted_prephase_peak_mib=200.+i)
            for i,cid in enumerate(('base','alternative'))}
        self.resources=dict(full=dict(resources=receipt(f.resources_path),candidate_overrides=self.overrides))
        self.patches=[patch.object(entry,'audit_model',return_value=self.valid),
            patch.object(entry,'audit_resources',return_value=self.resources),
            patch.object(entry,'audit_protocol',side_effect=lambda p:entry.read(p)),
            patch.object(entry,'audit_plan',return_value=f.plan),patch.object(entry,'CANDIDATES',('base','alternative'))]

    def start_patches(self):
        for p in self.patches:p.start();self.addCleanup(p.stop)

    def simple_evidence(self):
        # Small metric fixtures intentionally omit native READY/PHASE telemetry.
        def validated(paths,sealed):
            for path in paths:
                raw=entry.read(path)
                if raw['startup_begin_ns']<=sealed['created_ns']:raise ValueError('raw must start after seal')
            return paths,[]
        p=patch.object(entry,'validation_evidence',side_effect=validated);p.start();self.addCleanup(p.stop)

    def seal(self):
        f=self.f
        return entry.seal(self.protocol,f.model_path,self.resource_manifest,f.quality_path,f.root/'sealed')

    def test_full16_requires_strict_definition_and_complete_validation_free_coverage(self):
        for change in ({'training_flow_count':12},{'variant':'phase_ridge1'},{'omitted_flow_id':'x'},
                {'omitted_seed':627601},{'omitted_flow_ids':['x']},{'validation_sources_used':['raw']},
                {'diagnostic_repeats_fitted':True},{'capacity_evidence_pooled':True},
                {'direct_and_transitive_sources_verified':False}):
            with self.subTest(change=change),patch.object(entry.run_aware_training,'verify_run_aware_model',return_value={**self.valid,**change}),\
                    patch.object(entry,'_verify_duration_definition') as strict:
                with self.assertRaises(ValueError):entry.audit_model(self.f.model_path)
                strict.assert_called_once()
        with patch.object(entry.run_aware_training,'verify_run_aware_model',return_value=self.valid),\
                patch.object(entry,'_verify_duration_definition',side_effect=ValueError('tied OCR definition changed')):
            with self.assertRaisesRegex(ValueError,'tied OCR'):entry.audit_model(self.f.model_path)

    def test_prediction_overlay_keeps_original_plan_bytes_and_full_resource_receipts(self):
        self.start_patches();f=self.f;before=f.plan_path.read_bytes();predict=entry.flow_crossfit_report.predict
        original_candidates=copy.deepcopy(f.plan['candidates'])
        def checked_prediction(plan,candidate,*args):
            self.assertEqual(candidate['predicted_startup_joules'],self.overrides[candidate['id']]['predicted_startup_joules'])
            self.assertEqual(plan['candidates'],original_candidates)
            self.assertNotEqual(candidate['predicted_startup_joules'],next(c for c in plan['candidates'] if c['id']==candidate['id'])['predicted_startup_joules'])
            return predict(plan,candidate,*args)
        with patch.object(entry.flow_crossfit_report,'predict',side_effect=checked_prediction) as forecast:
            sealed=self.seal()
        self.assertEqual(forecast.call_count,2);self.assertEqual(before,f.plan_path.read_bytes())
        self.assertEqual(sealed['sources']['plan'],receipt(f.plan_path));self.assertEqual(sealed['gate_protocol'],entry.LIMITS)
        audit=entry.check_sealed(f.root/'sealed')
        self.assertEqual(audit['full_resource_estimates'],self.overrides)
        self.assertEqual(audit['resource_manifest'],receipt(self.resource_manifest))

    def test_report_uses_original_parent_metrics_and_requires_prestartup_seal(self):
        self.start_patches();self.simple_evidence();f=self.f;sealed=self.seal()
        raws=[f.write(cid+'.json',dict(f.raw(cid,sealed['created_ns']+2*fixtures.NS),
            plan=receipt(f.plan_path),startup_begin_ns=sealed['created_ns']+fixtures.NS)) for cid in ('base','alternative')]
        value=entry.report(f.root/'sealed',raws,f.root/'report')
        self.assertEqual(value['status'],'ASSESSED');self.assertEqual(value['candidates'][0]['timing_errors']['e2e']['mae'],10)
        raw=entry.read(raws[0]);raw['startup_begin_ns']=sealed['created_ns'];f.write(raws[0].name,raw)
        with self.assertRaisesRegex(ValueError,'start after'):entry.report(f.root/'sealed',raws,f.root/'early-report')
        self.assertFalse((f.root/'early-report').exists())

    def test_source_or_rehashed_resource_overlay_drift_rejected(self):
        self.start_patches();f=self.f;self.seal();path=f.root/'sealed/predictions.json'
        sealed=entry.read(path);sealed['candidates'][0]['predicted_candidate_resource_estimates']['predicted_startup_joules']+=1
        f.write('sealed/predictions.json',sealed)
        audit=entry.read(f.root/'sealed/training-audit.json');audit['predictions']=receipt(path);f.write('sealed/training-audit.json',audit)
        with self.assertRaisesRegex(ValueError,'overlay'):entry.check_sealed(f.root/'sealed')

    def test_gate_uses_unchanged_protocol_thresholds_without_mutating_reserved_plan(self):
        self.start_patches();self.simple_evidence();f=self.f;sealed=self.seal()
        raws=[f.write(cid+'.json',dict(f.raw(cid,sealed['created_ns']+2*fixtures.NS),
            plan=receipt(f.plan_path),startup_begin_ns=sealed['created_ns']+fixtures.NS)) for cid in ('base','alternative')]
        entry.report(f.root/'sealed',raws,f.root/'report');before=f.plan_path.read_bytes()
        value=entry.gate(self.protocol,f.root/'report/report.json',f.root/'gate.json')
        self.assertEqual(before,f.plan_path.read_bytes());self.assertEqual(value['plan'],receipt(f.plan_path))
        self.assertFalse(value['final_gain_eligible']);self.assertFalse(value['final_four_groups_enabled'])

    def test_prepare_binds_original_reserved_plan_and_thresholds_without_outcomes(self):
        f=self.f;plan=copy.deepcopy(f.plan);plan.update(window_seconds=600.,drain_seconds=300.,warmup_seconds=0.)
        plan['trace']=[dict(plan['trace'][0],request_id=str(i),offset_ns=i) for i in range(1200)]
        template=plan['candidates'][0];plan['candidates']=[]
        for cid in entry.CANDIDATES:
            candidate=copy.deepcopy(template);candidate.update(id=cid,routes={r['request_id']:'s' for r in plan['trace']})
            plan['candidates'].append(candidate)
        plan['baseline_candidate_id']=entry.CANDIDATES[0];plan.pop('startup_comparison_policy',None)
        plan.pop('plan_sha256',None);plan['plan_sha256']=digest(plan);f.write('plan.json',plan)
        capacity=f.write('capacity-original.json',{'reserved_target_plan':receipt(f.plan_path)})
        thresholds=f.write('thresholds.json',{'validation_thresholds':entry.LIMITS})
        addendum=f.write('addendum.json',{'schema':'gsdeploy.training_cv_selection_threshold_source_addendum.v1',
            'thresholds':entry.LIMITS,'threshold_plan':receipt(thresholds)})
        before=f.plan_path.read_bytes()
        with patch.object(entry.h600_capacity_v1,'verify_protocol',side_effect=lambda p:entry.read(p)):
            value=entry.prepare_validation(capacity,addendum,f.root/'prepared')
            target=entry.audit_plan(f.plan_path,f.root/'prepared/protocol.json')
            self.assertEqual(value['target_plan'],receipt(f.plan_path));self.assertEqual(len(target['trace']),1200)
            self.assertEqual(before,f.plan_path.read_bytes());self.assertFalse(value['GPU_started'])
            changed=dict(value,validation_thresholds=dict(entry.LIMITS,service_mare_max=.3))
            changed.pop('protocol_sha256');changed['protocol_sha256']=digest(changed);f.write('prepared/protocol.json',changed)
            with self.assertRaisesRegex(ValueError,'unchanged'):entry.audit_protocol(f.root/'prepared/protocol.json')

    def test_validation_evidence_checks_immutable_scoring_and_native_diagnostics(self):
        f=self.f;native=f.write('native.json',{})
        scored=f.write('scored.json',{})
        raw=dict(plan=receipt(f.plan_path),startup_begin_ns=3,candidate_id='base',native_raw=receipt(native),
            quality_evidence={'source_receipts':[receipt(f.quality_path)]})
        mapped=dict(traces=[{'split':'validation','not_for_training':True}],resource_evidence={'split':'validation'},
            mapping_report={'sources':[receipt(native)]})
        sealed=dict(sources={'plan':receipt(f.plan_path)},created_ns=2)
        with patch.object(entry.flow_crossfit_report,'scored_heldout',return_value=(raw,receipt(scored))) as score,\
                patch.object(entry,'map_flow_collection',return_value=mapped) as mapping:
            paths,audits=entry.validation_evidence([native],sealed)
            score.assert_called_once_with(receipt(native));mapping.assert_called_once_with(native,allow_validation_diagnostics=True)
            self.assertEqual(paths,[str(scored)]);self.assertEqual(audits[0]['native_raw'],receipt(native))
            mapped['traces'][0]['split']='training'
            with self.assertRaisesRegex(ValueError,'validation-only'):entry.validation_evidence([native],sealed)
        with patch.object(entry.flow_crossfit_report,'scored_heldout',side_effect=ValueError('Scoring changed immutable native observations')),\
                patch.object(entry,'map_flow_collection') as mapping:
            with self.assertRaisesRegex(ValueError,'immutable'):entry.validation_evidence([native],sealed)
            mapping.assert_not_called()


if __name__=='__main__':unittest.main()
