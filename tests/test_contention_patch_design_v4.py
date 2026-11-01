import unittest
from tools.predictor_validation_20261027.contention_patch_design_v4 import design


class PatchDesignTests(unittest.TestCase):
    def test_full_patch_is_training_only_exact_context_coverage(self):
        selection={'training':{t:[{'pool_id':t+str(i)} for i in range(4)] for t in ('text','image','asr','ocr')}}
        result=design(selection)
        self.assertEqual(result['counts_per_independent_capture']['metadata'],{'jobs':4,'requests':4})
        self.assertEqual(result['counts_per_independent_capture']['light-kernel'],{'jobs':104,'requests':294})
        self.assertEqual(result['two_capture_measured_requests'],1184)
        for shard in result['shards']:
            for job in shard['jobs']:
                self.assertEqual(job['split'],'training')
                self.assertEqual(set(job['inputs']),set(job['tasks']))
                for alias,task in job['tasks'].items():
                    self.assertEqual(job['inputs'][alias],task+str(job['input_stratum_index']))

    def test_offsets_cover_both_orders_without_duplicate_zero(self):
        selection={'training':{t:[{'pool_id':t+str(i)} for i in range(4)] for t in ('text','image','asr','ocr')}}
        shard=next(s for s in design(selection)['shards'] if s['name']=='SSGO-G2-asr')
        self.assertEqual(len(shard['jobs']),10)
        self.assertEqual(len({j['id'] for j in shard['jobs']}),10)
        for stratum in (0,3):
            rows=[j for j in shard['jobs'] if j['input_stratum_index']==stratum]
            self.assertEqual(sum(not any(j['start_offsets_ms'].values()) for j in rows),1)


if __name__=='__main__':unittest.main()
