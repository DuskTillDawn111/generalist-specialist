"""Full-horizon search bounds and reference search on CPU analytical evidence."""
import copy
import json
from pathlib import Path
import unittest

from tools.predictor_validation_20261027.search_check import candidates, evaluate, run_search, upper_bound

ROOT=Path(__file__).resolve().parents[1]

class SearchCheckTests(unittest.TestCase):
    def setUp(self):
        self.template=json.loads((ROOT/'artifacts/predictor-cpu-20261027/analytical-request.json').read_text())
        self.weights=json.loads((ROOT/'configs/resource_spillover_objective_20261024.json').read_text())['weights']

    def test_bound_dominates_actual_six_term_score_for_each_candidate(self):
        for candidate in candidates(self.template):
            result=evaluate(self.template,candidate,0,self.weights)
            self.assertEqual(result['status'],'COMPLETE')
            self.assertLessEqual(result['U'],upper_bound(self.template,candidate,[0],self.weights)+1e-12)

    def test_independent_searches_find_identical_optimum_and_all_ties(self):
        items=candidates(self.template)
        plain=run_search(self.template,items,[0,1],self.weights)
        bounded=run_search(self.template,list(reversed(items)),[0,1],self.weights,bounded=True)
        self.assertTrue(plain['whole_declared_cpu_domain_certified'])
        self.assertEqual(plain['exact_max_objective'],bounded['exact_max_objective'])
        self.assertEqual(plain['tie_ids'],bounded['tie_ids'])
        self.assertEqual(plain['evaluator_calls'],len(items)*2)

    def test_unknown_memory_does_not_certify_or_become_infeasible(self):
        self.template.pop('observed_peak_memory_mib')
        items=candidates(self.template,shares=(0,1))
        result=run_search(self.template,items,[0],self.weights,bounded=True)
        self.assertFalse(result['whole_declared_cpu_domain_certified'])
        self.assertEqual(len(result['unknown_ids']),len(items))
        self.assertTrue(all(r['runs'][0]['eligible'] is None for r in result['records']))

    def test_memory_and_rho_constraints_not_silently_relaxed(self):
        self.template['memory_budget_mib']=1
        result=run_search(self.template,candidates(self.template,shares=(0,)),[0],self.weights)
        self.assertIsNone(result['exact_max_objective'])
        self.assertTrue(all(r['runs'][0]['eligible'] is False for r in result['records']))

if __name__=='__main__':unittest.main()
