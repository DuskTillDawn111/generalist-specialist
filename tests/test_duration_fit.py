"""Duration-source provenance and two whole-seed-only refit boundaries."""
import copy
import json
import unittest
from pathlib import Path

from tests import test_flow_fit as fixtures
from tools.predictor_validation_20261027 import duration_fit as duration
from tools.predictor_validation_20261027.background_profiles import receipt
from tools.predictor_validation_20261027.flow_fit import prepare_protocol as prepare_original, read
from tools.predictor_validation_20261027.profiling_analysis import digest


class DurationFitTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.FlowFitTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        f=self.fixture;self.root=f.root
        suite=read(f.suite);jobs=[]
        for job in suite['jobs']:
            original_plan=read(job['plan']['path']);original_raw=read(Path(job['output'])/'raw.json')
            for index in range(4):
                candidate='route-'+str(index);folder=Path(job['output'])/candidate;folder.mkdir()
                plan=copy.deepcopy(original_plan);plan['candidates'][0]['id']=candidate;plan['baseline_candidate_id']=candidate
                plan.pop('plan_sha256');plan['plan_sha256']=digest(plan)
                plan_path=folder/'plan.json';f.write(plan_path,plan)
                raw=copy.deepcopy(original_raw);raw.update(plan=receipt(plan_path),candidate_id=candidate)
                f.write(folder/'raw.json',raw);f.write(folder/'invocation.json',{'plan':raw['plan'],'candidate_id':candidate,'runner':plan['runner_source']})
                jobs.append({'plan':receipt(plan_path),'candidate_id':candidate,'output':str(folder)})
        suite['jobs']=jobs;f.write(f.suite,suite)
        self.original=self.root/'eight-flow-protocol.json';prepare_original([f.old],f.suite,self.original)
        self.selection=self.root/'selection.json'
        value={'schema':'gsdeploy.duration_loss_training_cv_selection_protocol.v1','frozen_training_protocol':receipt(self.original),
            'CV_reports_read_before_freeze':False,'source_receipts':[receipt(self.original)]}
        value['protocol_sha256']=digest(value);f.write(self.selection,value)
        threshold_plan=self.root/'threshold-plan.json';f.write(threshold_plan,{'validation_thresholds':{'service_mare_max':.2}})
        self.addendum=self.root/'addendum.json'
        add={'schema':'gsdeploy.training_cv_selection_threshold_source_addendum.v1','selection_protocol':receipt(self.selection),
            'threshold_plan':receipt(threshold_plan),'thresholds':{'service_mare_max':.2}}
        add['addendum_sha256']=digest(add);f.write(self.addendum,add)
        self.protocol=self.root/'duration-protocol.json'
        duration.prepare_protocol(self.original,self.selection,self.addendum,self.protocol)

    def test_two_seed_fits_preserve_transitive_raws_and_new_numerics(self):
        out=self.root/'duration-cv';manifest=duration.fit_protocol(self.protocol,out)
        self.assertFalse(manifest['full_model_fitted']);self.assertEqual(manifest['leave_one_flow_out_refits'],0)
        cv=read(out/'seed-crossfit-manifest.json');self.assertEqual(len(cv['folds']),2)
        for fold in cv['folds']:
            verified=duration.verify_duration_model(fold['model']['path'])
            self.assertTrue(verified['duration_model_verified']);self.assertEqual(verified['training_flow_count'],4)
            self.assertEqual(len(verified['omitted_flow_ids']),4)
            self.assertTrue(all(r['heldout_raw'] not in verified['training_raw_receipts'] for r in fold['heldout_flows']))
            model=read(fold['model']['path'])
            self.assertEqual(model['definition']['base_numerics']['estimator'],'duration_weighted_relative_work_WLS.v1')
            self.assertIn(receipt(duration.__file__),model['source_receipts'])
        with self.assertRaises(FileExistsError):duration.fit_protocol(self.protocol,out)

    def test_selection_or_threshold_change_rejected_before_output(self):
        value=read(self.addendum);value['thresholds']['service_mare_max']=.9;self.fixture.write(self.addendum,value)
        out=self.root/'bad'
        with self.assertRaises(ValueError):duration.fit_protocol(self.protocol,out)
        self.assertFalse(out.exists())

    def test_hiding_duration_source_or_relabelling_old_loss_is_rejected(self):
        out=self.root/'fit';duration.fit_protocol(self.protocol,out)
        cv=read(out/'seed-crossfit-manifest.json');path=Path(cv['folds'][0]['model']['path']);model=read(path)
        model['definition']['base_numerics'].pop('estimator')
        model.pop('model_sha256');model['model_sha256']=digest(model);self.fixture.write(path,model)
        with self.assertRaisesRegex(ValueError,'mislabelled'):duration.verify_duration_model(path)


if __name__=='__main__':unittest.main()
