import unittest

from tools.predictor_validation_20261027.kernel_training_campaign_v3 import design, training_jobs
from tools.predictor_validation_20261027.profiling import RESIDENTS


class TrainingCampaignTests(unittest.TestCase):
    def selection(self):
        return {'training':{task:[dict(task=task,pool_id=f'{task}{i}',input_size=i+1) for i in range(4)]
                            for task in ('text','ocr')}}

    def test_bridge_uses_short_long_every_owner_new_training_only(self):
        jobs=training_jobs(self.selection(),bridge=True)
        self.assertEqual(len(jobs),8)
        self.assertEqual({j['input_stratum_index'] for j in jobs},{0,3})
        for alias in RESIDENTS:self.assertEqual(sum(alias in j['inputs'] for j in jobs),2)
        self.assertTrue(all(j['split']=='training' and j['phase']=='training-bridge' for j in jobs))

    def test_full_plan_has_32_solo_30_lead_lag_jobs(self):
        jobs=training_jobs(self.selection())
        self.assertEqual(len(jobs),62)
        self.assertEqual(sum(len(j['inputs'])==1 for j in jobs),32)
        self.assertEqual(sum(len(j['inputs'])>1 for j in jobs),30)
        self.assertEqual(sum(len(j['inputs']) for j in jobs),102)
        for context in ('SS','SG','SSG'):
            rows=[j for j in jobs if j['context']==context]
            self.assertEqual(len(rows),10)
            for index in (0,3):
                group=[j for j in rows if j['input_stratum_index']==index]
                self.assertEqual(len(group),5)
                self.assertEqual(sum(all(v==0 for v in j['start_offsets_ms'].values()) for j in group),1)
                first=list(group[0]['inputs'])[0]
                self.assertTrue(any(j['start_offsets_ms'][first]>0 for j in group))
                self.assertTrue(any(j['start_offsets_ms'][first]==0 for j in group))

    def test_three_domains_keep_full_plan_nonexecutable(self):
        result=design(self.selection())
        self.assertEqual(result['bridge']['jobs_all_three_domains'],24)
        self.assertEqual(result['proposed_full_training']['jobs_all_three_domains'],186)
        self.assertFalse(result['proposed_full_training']['executable'])
        self.assertTrue(result['no_validation_execution'])


if __name__=='__main__':unittest.main()
