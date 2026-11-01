"""Finite training acquisition preparation and no-retry controls, CPU only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import training_campaign_v14 as campaign
from tools.predictor_validation_20261027.probe_validation import receipt, read
from tools.predictor_validation_20261027.profiling import digest


class TrainingCampaignTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.profile=self.root/'profile';self.profile.mkdir()
        for name in ('plan.json','context.json','catalog.json'):(self.profile/name).write_text('{}')
        self.output=self.root/'campaign';self.mock=patch.object(campaign,'prepare',side_effect=self.fake_prepare)
        self.mock.start();self.addCleanup(self.mock.stop)

    def fake_prepare(self,profile,folder,*,seed,duration,**kwargs):
        folder.mkdir(parents=True,exist_ok=False)
        trace=[]
        for task,rate in [('text',1.85),('image',.05),('asr',.05),('ocr',.05)]:
            count=1110 if task=='text' else 30
            if duration==30:count=55 if task=='text' else 1
            for i in range(count):trace.append(dict(request_id=task+str(i),task=task,pool_id='p'+str(seed),offset_ns=i))
        plan=dict(schema='gsdeploy.native_flow_plan.v1',seed=seed,window_seconds=duration,drain_seconds=300.,
            aliases={'s':{}},trace=trace,candidates=[dict(id=cid,routes={r['request_id']:'s' for r in trace},resource_configuration={'mode':'shared'}) for cid in campaign.ROUTES],
            runner_source=receipt(self.profile/'context.json'),context=receipt(self.profile/'context.json'),
            profile_plan=receipt(self.profile/'plan.json'),catalog=receipt(self.profile/'catalog.json'))
        plan['plan_sha256']=digest(plan);(folder/'plan.json').write_text(json.dumps(plan));return plan

    def test_ten_long_runs_and_isolated_preflight_have_frozen_roles_and_unique_ids(self):
        value=campaign.prepare_campaign(self.profile,self.output)
        self.assertEqual(value['main_run_count'],10);self.assertEqual(len({j['run_id'] for j in value['main_jobs']}),10)
        self.assertEqual(len(read(self.output/'primary-training-suite.json')['jobs']),8)
        self.assertEqual(len(read(self.output/'repeat-training-suite.json')['jobs']),2)
        self.assertTrue(campaign.verify_campaign(self.output/'campaign.json'))
        for job in value['main_jobs']:
            plan=read(job['plan']['path']);self.assertEqual(plan['phase'],'predictor_training')
            self.assertEqual(plan['kind'],'prospective_training');self.assertEqual(plan['split'],'training')
            self.assertEqual(plan['window_seconds'],600.);self.assertEqual(plan['drain_seconds'],300.)
        with self.assertRaises(FileExistsError):campaign.prepare_campaign(self.profile,self.output)

    def test_main_requires_preflight_and_preserves_failure_without_retry(self):
        value=campaign.prepare_campaign(self.profile,self.output)
        with self.assertRaises(FileNotFoundError):campaign.run_campaign(self.output/'campaign.json','main')
        raw=dict(status='failed',collection_complete=False,qualification_issues=['fixture'])
        with patch.object(campaign,'execute',return_value=raw) as execute:
            with self.assertRaisesRegex(RuntimeError,'no retries'):
                campaign.run_campaign(self.output/'campaign.json','preflight')
            self.assertEqual(execute.call_count,1)
            with self.assertRaises(FileExistsError):campaign.run_campaign(self.output/'campaign.json','preflight')
            self.assertEqual(execute.call_count,1)

    def test_changed_plan_is_rejected_before_gpu(self):
        value=campaign.prepare_campaign(self.profile,self.output);path=Path(value['main_jobs'][0]['plan']['path'])
        plan=read(path);plan['split']='validation';path.write_text(json.dumps(plan))
        with self.assertRaises(ValueError):campaign.verify_campaign(self.output/'campaign.json')


if __name__=='__main__':unittest.main()
