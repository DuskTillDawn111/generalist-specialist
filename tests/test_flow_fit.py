"""Source-role trees, one-time fitting and whole-unit crossfit on CPU fixtures."""
import copy
import json
from pathlib import Path
import unittest

from tests import test_flow_profiles as fixtures
from tools.predictor_validation_20261027.background_profiles import receipt
from tools.predictor_validation_20261027.flow_fit import (
    EVIDENCE_SCHEMA, fit_protocol, load_training, prepare_protocol, read,
    single_context_bias, verify_training_model, weight_audit,
)
from tools.predictor_validation_20261027.profiling_analysis import digest


class FlowFitTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.FlowProfilesTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root;self.fixture.fixture.save();self.old=self.root/'raw.json'
        jobs=[]
        for seed in (1,2):
            folder=self.root/('seed-'+str(seed));folder.mkdir()
            plan=copy.deepcopy(self.fixture.plan);plan['seed']=seed
            plan.pop('plan_sha256');plan['plan_sha256']=digest(plan)
            plan_path=folder/'plan.json';self.write(plan_path,plan)
            raw=copy.deepcopy(self.fixture.raw);raw['plan']=receipt(plan_path)
            raw_path=folder/'raw.json';self.write(raw_path,raw)
            self.write(folder/'invocation.json',{'plan':raw['plan'],'candidate_id':'candidate','runner':plan['runner_source']})
            jobs.append({'plan':receipt(plan_path),'candidate_id':'candidate','output':str(folder)})
        self.suite=self.root/'suite.json';self.write(self.suite,{
            'schema':'gsdeploy.continuous_training_suite.v1','source':receipt(self.root/'runner.py'),
            'no_validation_data':True,'jobs':jobs})
        self.protocol=self.root/'fit-protocol.json'
        prepare_protocol([self.old],self.suite,self.protocol,expected_flow_count=2)

    @staticmethod
    def write(path,value):Path(path).write_text(json.dumps(value))

    def test_full_fit_preserves_transitive_raws_and_direct_wrappers(self):
        dest=self.root/'fit';manifest=fit_protocol(self.protocol,dest)
        model=read(dest/'training-v11-model.json');verified=verify_training_model(dest/'training-v11-model.json')
        self.assertTrue(verified['verified']);self.assertEqual(verified['training_flow_count'],2)
        self.assertEqual(verified['omitted_flow_ids'],[]);self.assertIsNone(verified['omitted_seed'])
        direct_paths={r['path'] for r in model['source_receipts']}
        for ref in manifest['continuous_training_raws']:
            self.assertNotIn(ref['path'],direct_paths)
            self.assertIn(ref,manifest['transitive_source_receipts'])
        wrappers=list((dest/'training-evidence').glob('*.json'))
        self.assertEqual(len(wrappers),2)
        for path in wrappers:self.assertEqual(read(path)['schema'],EVIDENCE_SCHEMA)
        with self.assertRaises(FileExistsError):fit_protocol(self.protocol,dest)

    def test_all_runs_required_before_any_output_is_created(self):
        (self.root/'seed-2/raw.json').unlink();dest=self.root/'incomplete'
        with self.assertRaisesRegex(ValueError,'All declared'):fit_protocol(self.protocol,dest)
        self.assertFalse(dest.exists())

    def test_crossfit_excludes_whole_flow_and_whole_seed_original_raws(self):
        dest=self.root/'cv';fit_protocol(self.protocol,dest,crossfit_only=True)
        flow=read(dest/'crossfit-manifest.json');seed=read(dest/'seed-crossfit-manifest.json')
        self.assertEqual(len(flow['folds']),2);self.assertEqual(len(seed['folds']),2)
        for fold in flow['folds']:
            checked=verify_training_model(fold['model']['path'])
            self.assertEqual(checked['training_flow_count'],1)
            self.assertEqual(checked['omitted_flow_ids'],[fold['fold_id']])
            self.assertNotIn(fold['heldout_raw'],checked['training_raw_receipts'])
            self.assertNotIn(fold['heldout_trace_id'],checked['training_trace_ids'])
        for fold in seed['folds']:
            checked=verify_training_model(fold['model']['path'])
            self.assertEqual(checked['omitted_seed'],fold['heldout_seed'])
            self.assertEqual(checked['training_flow_count'],1)
            self.assertIsNone(checked['omitted_flow_id'])
            self.assertTrue(seed['no_independent_validation'])

    def test_weight_audit_is_request_weighted_and_genuine_solos_change_baseline(self):
        bundle=load_training(self.protocol);audit=weight_audit(bundle['groups'])
        image=next(r for r in audit['targets'] if r['target'][1]=='image')
        self.assertEqual(image['old_handler_rows'],2)
        self.assertEqual(image['continuous_handler_rows'],4)
        self.assertAlmostEqual(image['continuous_share_within_target'],2/3)
        self.assertEqual(image['continuous_solo_count'],4)
        self.assertAlmostEqual(image['relative_solo_baseline_shift'],0)

    def test_seed_fold_excludes_all_four_routes_and_refits_the_other_seed(self):
        suite=read(self.suite);jobs=[]
        for job in suite['jobs']:
            original_plan=read(job['plan']['path']);original_raw=read(Path(job['output'])/'raw.json')
            for index in range(4):
                candidate='route-'+str(index);folder=Path(job['output'])/candidate;folder.mkdir()
                plan=copy.deepcopy(original_plan)
                plan['candidates'][0]['id']=candidate;plan['baseline_candidate_id']=candidate
                plan.pop('plan_sha256');plan['plan_sha256']=digest(plan)
                plan_path=folder/'plan.json';self.write(plan_path,plan)
                raw=copy.deepcopy(original_raw);raw.update(plan=receipt(plan_path),candidate_id=candidate)
                self.write(folder/'raw.json',raw)
                self.write(folder/'invocation.json',{'plan':raw['plan'],'candidate_id':candidate,'runner':plan['runner_source']})
                jobs.append({'plan':receipt(plan_path),'candidate_id':candidate,'output':str(folder)})
        suite['jobs']=jobs;self.write(self.suite,suite)
        protocol=self.root/'eight-flow-protocol.json'
        prepare_protocol([self.old],self.suite,protocol)
        dest=self.root/'eight-flow-cv';fit_protocol(protocol,dest,crossfit_only=True)
        manifest=read(dest/'seed-crossfit-manifest.json')
        for fold in manifest['folds']:
            verified=verify_training_model(fold['model']['path'])
            self.assertEqual(verified['training_flow_count'],4)
            self.assertEqual(len(verified['omitted_flow_ids']),4)
            self.assertEqual(len(fold['heldout_flows']),4)
            prefix='seed-'+str(fold['heldout_seed'])+'__'
            self.assertTrue(all(fid.startswith(prefix) for fid in verified['omitted_flow_ids']))
            self.assertTrue(all(not fid.startswith(prefix) for fid in verified['training_flow_ids']))
            self.assertTrue(all(row['heldout_raw'] not in verified['training_raw_receipts'] for row in fold['heldout_flows']))

    def test_single_context_relative_work_loss_has_size_bias(self):
        requests=[]
        for i,duration in enumerate((1_000_000,3_000_000)):
            left=i*10_000_000
            requests.append({'family':'text','task':'text','pool_id':'p','alias':'s','request_id':str(i),
                'handler_start_ns':left,'handler_end_ns':left+duration,
                'segments':[{'phase':'prefill','start_ns':left,'end_ns':left+duration}]})
        report=single_context_bias([{'requests':requests}])
        row=next(r for r in report['groups'] if r['level']=='handler')
        self.assertEqual(row['arithmetic_mean_duration_ms'],2)
        self.assertEqual(row['unregularized_single_context_implied_duration_ms'],2.5)
        self.assertEqual(row['relative_size_bias'],.25)
        self.assertFalse(report['mean_is_unbiased_conditional_prediction'])

    def test_prior_validation_cannot_be_relabelled_by_fit_protocol(self):
        raw=read(self.old);raw['request_records'][0]['split']='validation';self.write(self.old,raw)
        with self.assertRaisesRegex(ValueError,'receipt'):load_training(self.protocol)
        with self.assertRaisesRegex(ValueError,'explicitly training-only'):
            prepare_protocol([self.old],self.suite,self.root/'bad-protocol.json',expected_flow_count=2)

    def test_source_tree_detects_changed_derived_wrapper(self):
        dest=self.root/'fit';fit_protocol(self.protocol,dest)
        path=next((dest/'training-evidence').glob('*.json'));wrapper=read(path)
        wrapper['trace']['requests'][0]['segments'][0]['start_ns']+=1;self.write(path,wrapper)
        with self.assertRaisesRegex(ValueError,'receipt'):verify_training_model(dest/'training-v11-model.json')

    def test_verifier_rejects_hiding_an_original_transitive_raw(self):
        dest=self.root/'fit';fit_protocol(self.protocol,dest)
        path=dest/'training-v11-model.json';model=read(path)
        group=next(g for g in model['training_evidence_tree']['groups'] if g['role']=='continuous_training')
        group['transitive_sources']=[]
        model.pop('model_sha256');model['model_sha256']=digest(model);self.write(path,model)
        with self.assertRaisesRegex(ValueError,'omits original transitive'):verify_training_model(path)


if __name__=='__main__':unittest.main()
