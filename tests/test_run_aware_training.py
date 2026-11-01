"""Run-aware source roles and H600-only leave-seed-out fits on strict CPU raws."""
import copy
import unittest
from pathlib import Path
from tests import test_duration_fit as fixtures
from tools.predictor_validation_20261027 import run_aware_training as run
from tools.predictor_validation_20261027 import flow_fit
from tools.predictor_validation_20261027.background_profiles import receipt
from tools.predictor_validation_20261027.profiling_analysis import digest


class RunAwareTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.DurationFitTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        f=self.fixture;self.root=f.root;old_suite=flow_fit.read(f.fixture.suite)
        template=old_suite['jobs'][0];base_plan=flow_fit.read(template['plan']['path']);base_raw=flow_fit.read(Path(template['output'])/'raw.json')
        self.old_jobs=[];self.new_jobs=[];self.repeats=[]
        def job(seed,cid,window,repeat=False):
            folder=self.root/('run-'+str(seed)+'-'+cid+('-repeat' if repeat else ''));folder.mkdir()
            plan=copy.deepcopy(base_plan);plan.update(seed=seed,window_seconds=float(window),drain_seconds=120. if window==90 else 300.)
            plan['candidates'][0]['id']=cid;plan['baseline_candidate_id']=cid;plan.pop('plan_sha256');plan['plan_sha256']=digest(plan)
            path=folder/'plan.json';f.fixture.write(path,plan)
            raw=copy.deepcopy(base_raw);raw.update(plan=receipt(path),candidate_id=cid,
                deadline_ns=raw['phase_start_ns']+window*1_000_000_000,
                observation_end_ns=raw['phase_start_ns']+window*1_000_000_000)
            sample=raw['board_samples'][-1];right=raw['observation_end_ns'];t=sample['time_ns']+1_000_000_000
            while t<right:
                raw['board_samples'].append(dict(sample,time_ns=t));t+=1_000_000_000
            raw['board_samples'].append(dict(sample,time_ns=right))
            f.fixture.write(folder/'raw.json',raw);f.fixture.write(folder/'invocation.json',{'plan':raw['plan'],'candidate_id':cid,'runner':plan['runner_source']})
            return {'plan':receipt(path),'candidate_id':cid,'output':str(folder),
                'run_id':'seed-'+str(seed)+'__'+cid+('__repeat1' if repeat else '__primary'),
                'campaign_role':'identical_trace_training_repeat' if repeat else 'primary_training'}
        for seed in (1,2):
            for i in range(4):self.old_jobs.append(job(seed,'route-'+str(i),90))
        for seed in (3,4):
            for i in range(4):self.new_jobs.append(job(seed,'route-'+str(i),600))
        for i in (0,1):self.repeats.append(job(3,'route-'+str(i),600,True))
        source=receipt(self.root/'runner.py')
        old_path=self.root/'H90-suite.json';f.fixture.write(old_path,{'schema':'gsdeploy.continuous_training_suite.v1','no_validation_data':True,'jobs':self.old_jobs,'source':source})
        self.old_protocol=self.root/'old8-protocol.json';flow_fit.prepare_protocol([f.fixture.old],old_path,self.old_protocol)
        primary=self.root/'H600-primary.json';f.fixture.write(primary,{'schema':'gsdeploy.continuous_training_suite.v1','no_validation_data':True,'jobs':self.new_jobs,'source':source,'repeats_excluded':True})
        repeat=self.root/'repeat-suite.json';f.fixture.write(repeat,{'schema':'gsdeploy.training_run_repeat_suite.v1','jobs':self.repeats,'source':source})
        self.campaign=self.root/'campaign.json';campaign={'schema':'gsdeploy.h600_training_campaign.v1',
            'primary_suite':receipt(primary),'repeat_suite':receipt(repeat),'source_receipts':[source]}
        campaign['campaign_sha256']=digest(campaign);f.fixture.write(self.campaign,campaign)
        self.packet=self.root/'run-aware';run.prepare_protocol(self.old_protocol,self.campaign,self.packet)
        self.protocol=self.packet/'protocol.json'

    def test_prepare_does_not_read_outcomes_and_all_sources_keep_conditions(self):
        for job in self.new_jobs+self.repeats:(Path(job['output'])/'raw.json').unlink()
        packet=self.root/'prospective';value=run.prepare_protocol(self.old_protocol,self.campaign,packet)
        self.assertFalse(value['outcomes_read_during_prepare']);self.assertEqual(len(value['primary_runs']),16)
        self.assertEqual(len(value['diagnostic_repeat_runs']),2)
        self.assertEqual({r['window_seconds'] for r in value['primary_runs']},{90.,600.})
        self.assertTrue(all(not r['fitted'] for r in value['diagnostic_repeat_runs']))
        with self.assertRaisesRegex(ValueError,'All declared'):run.fit_protocol(packet/'protocol.json',self.root/'premature')
        self.assertFalse((self.root/'premature').exists())

    def test_H600_only_CV_retains_all_H90_and_never_fits_repeats(self):
        # Repeats are not needed to fit or validate primary-only training CV.
        for job in self.repeats:(Path(job['output'])/'raw.json').unlink()
        out=self.root/'cv';manifest=run.fit_protocol(self.protocol,out)
        self.assertEqual(len(manifest['folds']),2)
        for fold in manifest['folds']:
            verified=run.verify_run_aware_model(fold['model']['path'])
            self.assertEqual(verified['training_flow_count'],12)
            self.assertEqual(len(verified['omitted_flow_ids']),4)
            self.assertEqual(len(fold['heldout_flows']),4)
            model=flow_fit.read(fold['model']['path'])
            labels=model['run_metadata'];self.assertEqual(sum(r['window_seconds']==90 for r in labels),8)
            self.assertTrue(all(r['run_metadata']['window_seconds']==600 for r in fold['heldout_flows']))
            self.assertFalse(model['diagnostic_repeats_fitted']);self.assertFalse(model['capacity_evidence_pooled'])

    def test_full16_keeps_context_and_rejects_tampered_horizon_labels(self):
        out=self.root/'full';manifest=run.fit_protocol(self.protocol,out,full=True)
        verified=run.verify_run_aware_model(manifest['model']['path']);self.assertEqual(verified['training_flow_count'],16)
        self.assertEqual(verified['omitted_flow_ids'],[])
        model=flow_fit.read(manifest['model']['path']);model['run_metadata'][0]['window_seconds']=600.
        model.pop('model_sha256');model['model_sha256']=digest(model);self.fixture.fixture.write(manifest['model']['path'],model)
        with self.assertRaisesRegex(ValueError,'metadata'):run.verify_run_aware_model(manifest['model']['path'])

    def test_repeat_inputs_cannot_silently_differ_from_primary(self):
        path=Path(self.repeats[0]['plan']['path']);plan=flow_fit.read(path)
        plan['trace'][0]['pool_id']='different';plan.pop('plan_sha256');plan['plan_sha256']=digest(plan);self.fixture.fixture.write(path,plan)
        with self.assertRaises(ValueError):run.prepare_protocol(self.old_protocol,self.campaign,self.root/'bad')


if __name__=='__main__':unittest.main()
