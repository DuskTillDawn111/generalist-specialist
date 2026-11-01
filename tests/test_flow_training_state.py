"""Training state diagnostics keep input strata, dependence and split limits."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027 import flow_training_state as module


def observation(seed=1,pool='a',ordinal=1,work=1.):
    return {'seed':seed,'candidate_id':'shared_spill_low','alias':'generalist_2#0','task':'text',
        'pool_id':pool,'phase':'decode','work':work,'actual_ms':100.,'interpolated_ms':0.,
        'prior_worker_busy_ms':ordinal*100.,'worker_request_ordinal':ordinal,'measurement_age_ms':ordinal*200.}


class FlowTrainingStateTests(unittest.TestCase):
    def test_correlation_requires_variation_and_at_least_three_pairs(self):
        self.assertEqual(module.correlation([1,2,3],[2,4,6]),1.)
        self.assertIsNone(module.correlation([1,1,1],[2,4,6]))
        self.assertIsNone(module.correlation([1,2],[2,4]))

    def test_seed_comparison_matches_exact_inputs_not_input_mix(self):
        rows=[observation(1,'a',1,1.),observation(1,'a',2,1.),observation(1,'b',3,2.),
              observation(2,'a',1,1.),observation(2,'b',2,2.),observation(2,'b',3,2.)]
        pairs=module.paired_seed_comparisons(rows)[0]
        self.assertEqual(pairs['common_strata'],2)
        self.assertTrue(all(r['second_minus_first']==0 for r in pairs['strata']))
        self.assertEqual(pairs['strata'][0]['counts'],[2,1])
        with self.assertRaisesRegex(ValueError,'two independent training seeds'):
            module.paired_seed_comparisons(rows[:3])

    def test_unknown_work_kept_in_denominator(self):
        rows=[observation(work=1.),observation(ordinal=2,work=None)]
        result=module.stratum_summaries(rows,('pool_id',))[0]
        self.assertEqual(result['requested'],2)
        self.assertEqual(result['known'],1)
        self.assertEqual(result['work']['mean'],1.)

    def test_history_correlation_preserves_same_input_sequence(self):
        rows=[observation(ordinal=i,work=2.**i) for i in range(1,5)]
        result=module.history_diagnostics(rows)[0]
        self.assertAlmostEqual(result['log_work_vs_prior_worker_busy_correlation'],1.)
        self.assertAlmostEqual(result['lag1_same_input_log_work_correlation'],1.)
        self.assertEqual(result['mean_worker_request_gap_between_same_input_observations'],1.)

    def test_joint_phase_correlations_do_not_impute_unknown(self):
        vectors=[]
        for i in range(1,5):
            vectors.append({**{k:v for k,v in observation().items() if k in ('seed','candidate_id','alias','task','pool_id')},
                'phase_order':['prefill','decode'],'work_by_phase':{'prefill':2.**i,'decode':4.**i}})
        result=module.joint_phase_correlations(vectors)[0]['phase_pairs'][0]
        self.assertAlmostEqual(result['log_work_correlation'],1.)
        vectors[0]['work_by_phase']['decode']=None
        self.assertEqual(module.joint_phase_correlations(vectors)[0]['phase_pairs'][0]['observations'],3)

    def test_block_centering_is_descriptive_and_preserves_rows(self):
        rows=[observation(pool='a' if i%2 else 'b',ordinal=i,work=(2. if i%2 else .5)*math.exp(i/10)) for i in range(1,9)]
        before=copy.deepcopy(rows)
        result=module.block_diagnostics(rows)[0]
        self.assertEqual(rows,before)
        self.assertEqual(result['block_lengths'],[5,3])
        self.assertEqual(result['observations'],8)
        self.assertIn('descriptive',result['centering_scope'])

    def test_all_eight_training_raws_required_and_validation_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for seed in (1,2):
                path=root/('seed-'+str(seed));path.mkdir()
                plan={'phase':'predictor_training','split':'training','candidates':[{'id':c} for c in module.ROUTES]}
                (path/'plan.json').write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'All eight'):
                module.expected_raws(root)
            for seed in (1,2):
                for route in module.ROUTES:
                    path=root/('seed-'+str(seed))/'runs'/route;path.mkdir(parents=True)
                    (path/'raw.json').write_text('{}')
            self.assertEqual(len(module.expected_raws(root)),8)
            path=root/'seed-1/plan.json';plan=json.loads(path.read_text());plan['phase']='predictor_validation';path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'training domain'):
                module.expected_raws(root)


if __name__=='__main__':
    unittest.main()
