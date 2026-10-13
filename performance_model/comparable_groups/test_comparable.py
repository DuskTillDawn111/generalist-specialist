"""Small mathematical and provenance checks; no inference experiments."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from .canonical import solve_canonical_routing
from .analyze import OUT, gate
from performance_model.ranking.evaluate import ranking_metrics, sha


class ComparableTests(unittest.TestCase):
    def test_canonical_minmax_known_solution(self):
        catalog={m:SimpleNamespace(supported_tasks={'text'}, service_time_ms={'text':s})
                 for m,s in [('a',1000.),('b',2000.)]}
        routing, demand, z=solve_canonical_routing({'a':1,'b':1},{'text':1.},catalog)
        self.assertAlmostEqual(z,2/3)
        self.assertAlmostEqual(routing['text']['a'],2/3)
        self.assertAlmostEqual(routing['text']['b'],1/3)
        self.assertAlmostEqual(demand['a'],demand['b'])
        with self.assertRaises(ValueError):
            solve_canonical_routing({'a':1,'b':1},{'text':2.},catalog)

    def test_missing_capability_is_not_a_routing(self):
        catalog={'a':SimpleNamespace(supported_tasks={'text'},service_time_ms={'text':10.})}
        with self.assertRaises(ValueError):
            solve_canonical_routing({'a':1},{'image':1.},catalog)

    def test_perfect_single_group_does_not_pass(self):
        perfect=ranking_metrics(['a','b','c','d'],[1,2,3,4],[1,2,3,4])
        self.assertFalse(gate([perfect])['assessed'])
        two=ranking_metrics(['a','b'],[1,2],[1,2])
        self.assertFalse(gate([perfect,two])['assessed'])

    def test_analysis_cannot_fit_or_launch_profiling(self):
        tree=ast.parse((Path(__file__).parent/'analyze.py').read_text())
        forbidden={'fit','fit_transform','run_measurement','Popen','subprocess',
                   'generate_feasible_deployments','all_combinations','make_plan'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                name=getattr(node.func,'attr',getattr(node.func,'id',''))
                self.assertNotIn(name,forbidden)

    def test_frozen_inputs_and_all_repetitions_preserved(self):
        frozen=json.loads((OUT/'comparison_plan_hash.json').read_text())
        self.assertEqual(sha(OUT/'comparison_plan.json'),frozen['sha256'])
        self.assertEqual(sha(OUT/'run_manifest.json'),frozen['run_manifest_sha256'])
        audit=json.loads((OUT/'leakage_audit.json').read_text())
        self.assertTrue(audit['passed'])
        self.assertEqual(len(audit['raw_source_hashes']),45)
        for path,digest in audit['raw_source_hashes'].items():
            self.assertEqual(sha(path),digest)
        groups=json.loads((OUT/'comparison_groups.json').read_text())['groups']
        for group in groups:
            self.assertEqual(len(group['configurations']),5)
            for c in group['configurations']:
                self.assertEqual(len(c['repetitions']),3)
                if any(r['stability_state_v2']=='unstable' for r in c['repetitions']):
                    self.assertFalse(c['eligible']['strict'])
                    self.assertFalse(c['eligible']['extended'])


if __name__=='__main__':
    unittest.main()
