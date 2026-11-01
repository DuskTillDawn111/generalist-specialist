"""H600 report excludes H90 test rows and retains exact heldout provenance."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch
from tests import test_variant_report as fixtures
from tools.predictor_validation_20261027 import run_aware_report as report
from tools.predictor_validation_20261027.ocr_trained_validation import LIMITS
from tools.predictor_validation_20261027.native_flow import receipt


class RunAwareReportTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.VariantReportTests();self.f.setUp();self.addCleanup(self.f.doCleanups);f=self.f
        manifest=report.read(f.manifest);old=report.read(f.protocol);jobs=old['flow_jobs'];labels=[]
        for seed in (1,2):
            plan_path=Path(jobs[(seed-1)*4]['plan']['path']);plan=report.read(plan_path)
            plan.update(window_seconds=600.,drain_seconds=300.,workload_rps={'text':1.})
            f.write(plan_path.name,plan)
            for job in jobs:
                if job['seed']==seed:
                    job['plan']=receipt(plan_path);raw=report.read(job['raw_path']);raw['plan']=job['plan'];f.write(Path(job['raw_path']).name,raw)
        for job in jobs:labels.append({'flow_id':job['flow_id'],'source_role':'H600_primary','window_seconds':600.,'drain_seconds':300.,'run_id':job['flow_id']+'__primary'})
        for seed in (50,51):
            for cid in sorted(report.ROUTES):
                path=f.write(str(seed)+'-'+cid+'.json',{'old_training':seed,'candidate':cid})
                job={'flow_id':f'seed-{seed}__{cid}','seed':seed,'raw_path':str(path),'candidate_id':cid,'plan':jobs[0]['plan']}
                jobs.append(job);labels.append({'flow_id':job['flow_id'],'source_role':'H90_development_primary','window_seconds':90.,'drain_seconds':120.,'run_id':job['flow_id']+'__primary'})
        f.write('protocol.json',{'expected_flow_count':16,'flow_jobs':jobs})
        run_proto=f.write('run-protocol.json',{'frozen_training_protocol':receipt(f.protocol),'primary_runs':labels,'H600_seeds':[1,2]})
        manifest.update(protocol=receipt(f.protocol),run_aware_protocol=receipt(run_proto),
            training_verifier={'name':'verify_run_aware_model','source':receipt(f.verifier_source)})
        for fold in manifest['folds']:
            seed=fold['heldout_seed'];model_path=Path(fold['model']['path']);f.write(model_path.name,{'training_evidence_tree':{'protocol':receipt(f.protocol)}})
            fold['model']=receipt(model_path);training=[j for j in jobs if j['seed']!=seed];heldout=[j for j in jobs if j['seed']==seed]
            v=f.verifications[str(model_path)];v.pop('variant_model_verified');v.update(run_aware_model_verified=True,training_flow_count=12,
                training_flow_ids=[j['flow_id'] for j in training],training_raw_receipts=[receipt(j['raw_path']) for j in training],
                training_trace_ids=['flow:'+receipt(j['raw_path'])['sha256'] for j in training])
            fold['training_flow_ids']=v['training_flow_ids']
            for h,j in zip(fold['heldout_flows'],heldout):
                h.update(heldout_raw=receipt(j['raw_path']),heldout_plan=j['plan'],heldout_trace_id='flow:'+receipt(j['raw_path'])['sha256'])
        f.write('manifest.json',manifest)
        threshold_plan=f.write('threshold-plan.json',{'validation_thresholds':LIMITS})
        self.threshold=f.write('threshold-addendum.json',{'schema':'gsdeploy.training_cv_selection_threshold_source_addendum.v1','threshold_plan':receipt(threshold_plan),'thresholds':LIMITS})
        p=patch.object(report,'verify_run_aware_model',side_effect=lambda path:f.verifications[path]);p.start();self.addCleanup(p.stop)

    def test_complete_H600_domain_has_only_eight_new_runs_with12_training_flows(self):
        audit=report.audit_manifest(self.f.manifest)
        self.assertEqual(len(audit['cases']),8);self.assertEqual({c['seed'] for c in audit['cases']},{1,2})
        self.assertTrue(all(c['run_metadata']['window_seconds']==600 for c in audit['cases']))

    def test_missing_H90_training_flow_or_heldout_native_leak_is_rejected(self):
        manifest=report.read(self.f.manifest);fold=manifest['folds'][0];v=self.f.verifications[fold['model']['path']]
        v['training_flow_ids'].pop()
        with self.assertRaisesRegex(ValueError,'exclude'):report.audit_manifest(self.f.manifest)

    def test_H90_cannot_be_relabelled_as_an_H600_test_case(self):
        m=report.read(self.f.manifest);p=Path(m['run_aware_protocol']['path']);v=report.read(p);v['primary_runs'][0]['window_seconds']=90.
        self.f.write(p.name,v);m['run_aware_protocol']=receipt(p);self.f.write('manifest.json',m)
        with self.assertRaisesRegex(ValueError,'H600'):report.audit_manifest(self.f.manifest)

    def test_fresh8_rollouts_keep_unknowns_and_original_threshold_failures(self):
        def compare(plan,candidate,prediction,raw):
            return {'requested':1200,'known':1199,'candidate_id':candidate['id'],
                'predicted_comparison_U':None,'observed_U':-1.,'unknown_reason':'UNSEEN_TASK_CONTEXT',
                'timing_errors':{},'per_task_p95':{},'queue_error':{},'eligible_for_gain_claim':False}
        f=self.f
        with patch.object(report,'predictor_sources',return_value=[receipt(f.predictor_source)]),patch.object(report,'verifier_source',return_value=receipt(f.verifier_source)),\
             patch.object(report.core,'scored_heldout',side_effect=lambda ref:({},ref)),patch.object(report.core,'predict',return_value={}) as predict,\
             patch.object(report.core,'compare',side_effect=compare):
            value=report.run(f.manifest,f.resources,f.quality,self.threshold,f.root/'report')
        self.assertEqual(predict.call_count,8);self.assertEqual(value['requested'],9600);self.assertEqual(value['known'],9592)
        self.assertFalse(value['complete_prediction_coverage']);self.assertFalse(value['final_gain_eligible'])
        self.assertEqual(value['resource_transfer_status'],'UNVALIDATED_H600_CONDITIONAL_PROXY')
        self.assertEqual(value['original_threshold_assessment']['decision_errors'],18)
        self.assertFalse(value['original_threshold_assessment']['original_training_cv_thresholds_passed'])

    def test_resource_fold_overrides_apply_only_to_prediction_and_match_heldout_seed(self):
        f=self.f;folds={};resources={}
        for seed in (1,2):
            resource=f.write('resource-fold-'+str(seed)+'.json',{'heldout_seed':seed})
            folds[seed]={'resources':receipt(resource),'candidate_overrides':{cid:{
                'predicted_startup_joules':10.+seed,'predicted_prephase_peak_mib':100.+seed} for cid in report.ROUTES}}
        def prediction(plan,candidate,model,resource,quality):
            seed=plan['seed'];self.assertEqual(resource['heldout_seed'],seed)
            self.assertEqual(candidate['predicted_startup_joules'],10.+seed)
            self.assertEqual(candidate['predicted_prephase_peak_mib'],100.+seed)
            return {}
        def compare(plan,candidate,prediction,raw):
            self.assertNotIn('predicted_startup_joules',candidate)
            return {'requested':1,'known':1,'candidate_id':candidate['id'],'predicted_comparison_U':None,'observed_U':-1.}
        resource_manifest=f.write('resource-manifest.json',{})
        with patch.object(report,'predictor_sources',return_value=[receipt(f.predictor_source)]),patch.object(report,'verifier_source',return_value=receipt(f.verifier_source)),\
             patch.object(report,'resource_folds',return_value=(folds,[receipt(resource_manifest)])),\
             patch.object(report.core,'scored_heldout',side_effect=lambda ref:({},ref)),patch.object(report.core,'predict',side_effect=prediction),\
             patch.object(report.core,'compare',side_effect=compare):
            value=report.run(f.manifest,f.resources,f.quality,self.threshold,f.root/'resource-report',resource_manifest_path=resource_manifest)
        self.assertEqual(value['resource_transfer_status'],'H600_WHOLE_SEED_TRAINING_CV_ONLY')
        self.assertEqual(value['resource_manifest'],receipt(resource_manifest))

    def resource_manifest_fixture(self):
        f=self.f;source=f.write('resource-source.json',{'measured':'training only'})
        folds=[]
        for seed in (1,2):
            document=f.write('resource-direct-'+str(seed)+'.json',{'source_receipts':[receipt(source)]})
            folds.append({'heldout_seed':seed,'training_seeds':[3-seed],'resources':receipt(document),
                'candidate_overrides':{cid:{'predicted_startup_joules':11.,
                    'predicted_prephase_peak_mib':101.} for cid in report.ROUTES}})
        manifest={'schema':'gsdeploy.h600_resource_manifest.v1','folds':folds,'source_receipts':[]}
        path=f.write('resource-manifest-direct.json',manifest)
        return path,manifest,source

    def test_resource_fold_audit_preserves_transitive_sources_and_rejects_source_drift(self):
        path,manifest,source=self.resource_manifest_fixture()
        with patch('tools.predictor_validation_20261027.h600_resources_v1.verify_manifest',return_value=manifest) as verify:
            folds,refs=report.resource_folds(path,{1,2})
            verify.assert_called_once_with(path)
            self.assertEqual(set(folds),{1,2});self.assertIn(receipt(source),refs)
            self.f.write(source.name,{'changed':True})
            with self.assertRaises(ValueError):report.resource_folds(path,{1,2})

    def test_resource_fold_audit_rejects_seed_leak_missing_route_and_invalid_estimate(self):
        path,manifest,_=self.resource_manifest_fixture()
        mutations=(lambda m:m['folds'][0].update(training_seeds=[1,2]),
            lambda m:m['folds'][0]['candidate_overrides'].pop('shared_balanced'),
            lambda m:m['folds'][0]['candidate_overrides']['shared_balanced'].update(predicted_startup_joules=True),
            lambda m:m['folds'][1].update(heldout_seed=1))
        for mutate in mutations:
            changed=copy.deepcopy(manifest);mutate(changed)
            with self.subTest(changed=changed),patch('tools.predictor_validation_20261027.h600_resources_v1.verify_manifest',return_value=changed):
                with self.assertRaises(ValueError):report.resource_folds(path,{1,2})

    def test_actual_H600_resource_manifest_recomputes_through_report_interface(self):
        from tests.test_h600_resources_v1 import H600ResourceTests
        fixture=H600ResourceTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        path=fixture.fitted()/'manifest.json'
        folds,refs=report.resource_folds(path,{627601,627602})
        self.assertEqual(folds[627601]['training_seeds'],[627602])
        self.assertEqual(folds[627602]['training_seeds'],[627601])
        cid=sorted(report.ROUTES)[0]
        self.assertEqual(folds[627601]['candidate_overrides'][cid]['predicted_startup_joules'],200.)
        self.assertEqual(folds[627602]['candidate_overrides'][cid]['predicted_startup_joules'],100.)
        self.assertTrue(any(ref['path'].endswith('/source.json') for ref in refs))


if __name__=='__main__':unittest.main()
