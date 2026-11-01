import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import capacity_input_weighted_v1 as module


class InputWeightedCapacityTests(unittest.TestCase):
    def setUp(self):
        self.plan = {'aliases':{'s0':{'family':'text','supported_tasks':['text']},
                                's1':{'family':'text','supported_tasks':['text']}},
                     'workload_rps':{'text':2.}, 'trace':[
                         {'request_id':'r'+str(i),'task':'text','pool_id':pool}
                         for i,pool in enumerate(['short','long','long','long'])]}
        self.candidate = {'id':'balanced','routes':{'r0':'s0','r1':'s1','r2':'s0','r3':'s1'},
                          'routing_weights':{'text':{'s0':.5,'s1':.5}}}
        self.catalog = {'capacity_factors':{'text':{'1':1.,'2':1.5}}}
        self.training = [{'flow_id':'seed-1__balanced','seed':1,'raw_source':{'path':'train','sha256':'train'},
            'condition':{'route':'balanced'},'generation':{'law':'known'},'observations':[
                {'family':'text','task':'text','pool_id':pool,'alias':alias,'worker_service_ms':ms}
                for pool,alias,ms in [('short','s0',100.),('short','s1',100.),('short','s0',100.),('long','s1',500.)]]}]
        mocks = [patch.object(module,'condition',return_value={'route':'balanced'}),
                 patch.object(module,'_generation',return_value={'law':'known'})]
        for mock in mocks:
            mock.start(); self.addCleanup(mock.stop)

    def estimate(self):
        return module.estimate(self.plan,self.candidate,self.training,self.catalog)

    def test_changed_input_mixture_corrects_composition_not_offered_rate(self):
        value = self.estimate()
        item = value['estimates'][0]
        # Training happened to be 3 short / 1 long; target is 1 short / 3 long.
        self.assertEqual(item['pooled_training_worker_mean_ms'],200.)
        self.assertEqual(item['weighted_worker_service_ms'],400.)
        self.assertEqual(item['family_normalized_service_ms'],300.)
        # offered 2rps * normalized .3 sec / g(2)=1.5 = .4.
        self.assertAlmostEqual(value['rho_by_family']['text'],.4)
        self.assertEqual(value['capacity_factors']['text'],1.5)
        self.assertEqual(value['threshold'],.8)
        self.assertEqual(value['tolerance'],1e-8)
        self.assertEqual(self.catalog['capacity_factors'],value['original_capacity_factors'])
        self.assertIsNone(value['original_rho_admission_passed'])
        self.assertFalse(value['target_outcomes_used'])

    def test_positive_weight_missing_group_stays_unknown_without_renormalization(self):
        self.training[0]['observations'] = self.training[0]['observations'][:-1]
        value = self.estimate()
        item = value['estimates'][0]
        self.assertEqual(item['known_target_weight_mass'],.25)
        self.assertEqual(item['missing_inputs'],['long'])
        self.assertIsNone(item['weighted_worker_service_ms'])
        self.assertIsNone(value['rho_by_family']['text'])
        self.assertIsNone(value['rho_threshold_satisfied'])

    def test_balanced_and_skewed_conditions_cannot_share_family_aggregate(self):
        self.training[0]['condition'] = {'route':'skewed'}
        value = self.estimate()
        self.assertEqual(value['status'],'UNKNOWN')
        self.assertEqual(value['estimates'][0]['unknown_reason'],'NO_MATCHED_TRAINING_CONDITION')

    def test_unchanged_capacity_cutoff_can_fail(self):
        self.plan['workload_rps']['text'] = 5.
        value = self.estimate()
        self.assertAlmostEqual(value['rho_by_family']['text'],1.)
        self.assertFalse(value['rho_threshold_satisfied'])

    def test_measured_heldout_values_only_affect_evaluation(self):
        value = self.estimate(); before = copy.deepcopy(value)
        actual = [{'family':'text','task':'text','worker_service_ms':400.}]
        result = module.evaluate(value,actual)
        self.assertEqual(value,before)
        self.assertEqual(result['groups'][0]['weighted_mean_relative_error'],0.)
        self.assertEqual(result['groups'][0]['pooled_mean_relative_error'],.5)
        actual[0]['worker_service_ms'] = 800.
        result = module.evaluate(value,actual)
        self.assertEqual(value,before)
        self.assertEqual(result['groups'][0]['weighted_mean_relative_error'],.5)

    def test_whole_seed_crossvalidation_uses_only_other_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root/'catalog.json'; catalog.write_text(json.dumps(self.catalog))
            runs = []
            for seed,service in ((1,100.),(2,500.)):
                for cid in sorted(module.ROUTES):
                    plan = copy.deepcopy(self.plan); plan['seed']=seed
                    candidate = copy.deepcopy(self.candidate); candidate['id']=cid
                    plan['candidates']=[candidate]; plan['catalog']={'path':str(catalog),'sha256':'fixture'}
                    observations = copy.deepcopy(self.training[0]['observations'])
                    for item in observations: item['worker_service_ms']=service
                    runs.append({**copy.deepcopy(self.training[0]),'seed':seed,'candidate_id':cid,
                        'flow_id':f'seed-{seed}__{cid}','plan':plan,'plan_source':{'path':'public','sha256':'public'},
                        'raw_source':{'path':f'{seed}-{cid}','sha256':f'{seed}-{cid}'},'observations':observations})
            bundle = {'runs':runs,'seeds':[1,2],'source_receipts':[]}
            # Fixture condition intentionally matches all routes to stress the
            # seed exclusion; actual loader enforces full condition identity.
            with patch.object(module,'load_training',return_value=bundle), \
                 patch.object(module,'receipt',return_value={'path':'fixture','sha256':'fixture'}):
                result = module.crossvalidate('suite',root/'cv')
            self.assertEqual(result['summary']['requested_groups'],8)
            for case in result['cases']:
                expected = 500. if case['seed']==1 else 100.
                self.assertEqual(case['prediction']['estimates'][0]['weighted_worker_service_ms'],expected)
                self.assertTrue(all(f'seed-{case["seed"]}__' not in fid for fid in case['training_flow_ids']))
            self.assertTrue((root/'cv/prospective-qualification-draft.json').exists())
            self.assertFalse(result['qualification_passed'])


if __name__=='__main__':
    unittest.main()
