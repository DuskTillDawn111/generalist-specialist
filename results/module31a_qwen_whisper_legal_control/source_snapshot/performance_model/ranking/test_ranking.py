"""Ranking math, objective reconstruction and saved-OOF provenance tests."""
import ast
import copy
import json
from pathlib import Path
import unittest

from .evaluate import (ROOT,OUT,OOF,DATA,VARIANTS,ranking_metrics,summarize_metrics,
    predicted_task_means,weighted_objective,workload_key,canonical,protocol_signature,
    shared_workers_compatible,configuration_eligible)


class RankingTests(unittest.TestCase):
    def test_reversed_ranking_regret_and_shortlists(self):
        m=ranking_metrics(['a','b','c','d'],[1.,2.,3.,4.],[4.,3.,2.,1.])
        self.assertAlmostEqual(m['kendall_tau'],-1.)
        self.assertAlmostEqual(m['spearman_rho'],-1.)
        self.assertEqual(m['top1_hit'],0)
        self.assertEqual(m['top2_recall'],0)
        self.assertEqual(m['top2_coverage'],0)
        self.assertEqual(m['top3_coverage'],0)
        self.assertEqual(m['regret'],3.)
        self.assertEqual(m['uniform_random_baseline']['expected_regret'],1.5)

    def test_trivial_two_candidate_top2_is_not_evidence(self):
        m=ranking_metrics(['a','b'],[1.,2.],[2.,1.])
        self.assertEqual(m['top2_coverage'],1)
        self.assertFalse(m['top2_nontrivial'])
        self.assertIsNone(m['top3_coverage'])
        summary=summarize_metrics([m])
        self.assertIsNone(summary['top2_nontrivial_coverage'])
        self.assertEqual(summary['regret']['worst_case'],1.)

    def test_ties_never_use_true_labels_to_break_prediction_tie(self):
        m=ranking_metrics(['z','a'],[1.,2.],[7.,7.])
        self.assertEqual(m['selected_candidate'],'a')
        self.assertEqual(m['regret'],1.)
        self.assertEqual(m['tie_regret_range'],[0.,1.])
        self.assertIsNone(m['kendall_tau'])

    def test_workload_weight_and_routing_mixture(self):
        w={'text':2.,'image':1.}
        r={'text':{'qwen':1.,'distilbert':1.},'image':{'qwen':1.}}
        p={('text','qwen'):100.,('text','distilbert'):20.,('image','qwen'):200.}
        task=predicted_task_means(w,r,p)
        self.assertEqual(task['text'],60.)
        self.assertAlmostEqual(weighted_objective(w,task),320/3)
        del p['text','distilbert']
        self.assertIsNone(weighted_objective(w,predicted_task_means(w,r,p)))

    def test_no_same_total_or_similar_feature_workload_merging(self):
        self.assertNotEqual(canonical(workload_key({'text':1,'asr':1})),canonical(workload_key({'text':2})))
        self.assertEqual(workload_key({'text':1}),workload_key({'text':1.,'image':0.}))

    def test_protocol_workers_exception_is_narrow(self):
        raw=dict(source={'task_pool_hash':'pool','harness_hashes':{'run':'h'},'workers':{'qwen':{'script_hash':'q'}}},
            schema_version=1,warmup_seconds=10,measurement_seconds=100,clock={'definition':'mono'},
            warmup={'priming_excluded_from_measurement':True},arrival_process={'name':'periodic'},
            measurement_window={'cohort':'cohort','completion_count_definition':'success'})
        other=copy.deepcopy(raw)
        other['source']['workers']['resnet50']={'script_hash':'r'}
        self.assertEqual(protocol_signature(raw),protocol_signature(other))
        self.assertEqual(shared_workers_compatible([{'workers':raw['source']['workers']},{'workers':other['source']['workers']}]),{})
        other['source']['workers']['qwen']['script_hash']='different'
        self.assertIn('qwen',shared_workers_compatible([{'workers':raw['source']['workers']},{'workers':other['source']['workers']}]))
        other['measurement_seconds']=20
        self.assertNotEqual(protocol_signature(raw),protocol_signature(other))

    def test_unstable_repetition_disqualifies_entire_configuration(self):
        self.assertFalse(configuration_eligible(['steady','unstable','steady'],'strict'))
        self.assertFalse(configuration_eligible(['steady','unstable','steady'],'extended'))
        self.assertFalse(configuration_eligible(['steady','recoverable_congested'],'strict'))
        self.assertTrue(configuration_eligible(['steady','recoverable_congested'],'extended'))

    def test_no_model_fit_or_worker_call_in_ranking_module(self):
        tree=ast.parse((Path(__file__).parent/'evaluate.py').read_text())
        forbidden={'fit','fit_transform','train','run_measurement','enumerate_deployments','all_combinations'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
                self.assertNotIn(name,forbidden)


@unittest.skipUnless((OUT/'ranking_leakage_audit.json').exists(),'Run ranking audit first')
class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit=json.loads((OUT/'ranking_leakage_audit.json').read_text())
        cls.groups=json.loads((OUT/'comparison_groups.json').read_text())
        cls.metrics=json.loads((OUT/'ranking_metrics.json').read_text())
        cls.predictions=[json.loads(l) for l in (OUT/'ranking_predictions.jsonl').read_text().splitlines()]
        cls.saved={(p['variant'],p['row_id']):p for p in (json.loads(l) for l in OOF.read_text().splitlines()) if p['subset']=='primary'}

    def test_every_saved_path_is_deployment_out_and_no_tail_is_invented(self):
        for p in self.audit['prediction_proofs']:
            s=self.saved[p['variant'],p['row_id']]
            f=self.audit['used_heldout_folds'][p['heldout_deployment']]
            self.assertEqual(s['heldout_deployment'],p['heldout_deployment'])
            self.assertNotIn(p['heldout_deployment'],f['train']['deployment_id'])
            self.assertNotIn(p['configuration_id'],f['train']['configuration_id'])
            self.assertNotIn(p['run_id'],f['train']['run_id'])
        for p in self.predictions:self.assertIsNone(p['predicted_tail_objective_ms'])
        self.assertEqual(self.audit['fits_performed'],0)

    def test_rankable_groups_preserve_workload_protocol_and_all_repetitions(self):
        inventory={g['group_id']:g for g in self.groups['inventory']}
        for analysis,gs in self.groups['analyses'].items():
            for g in gs:
                if g['status']!='rankable':continue
                source=inventory[g['group_id']]
                self.assertFalse(source['worker_implementation_conflicts'])
                for c in g['candidates']:
                    original=next(x for x in source['candidates'] if x['configuration_id']==c['configuration_id'])
                    self.assertTrue(configuration_eligible([r['stability_state_v2'] for r in original['runs']],analysis))
                    self.assertGreaterEqual(len(c['run_ids']),2)
                    for r in original['runs']:
                        self.assertEqual(r['workload'],g['workload'])
                        self.assertEqual(r['protocol_signature'],source['protocol_signature'])

    def test_configuration_scores_are_reconstructed_from_saved_oof(self):
        allruns={r['run_id']:r for g in self.groups['inventory'] for c in g['candidates'] for r in c['runs']}
        for p in self.predictions:
            values=[]
            for rid in p['run_ids']:
                run=allruns[rid]
                paths={(x['task'],x['model']):self.saved[VARIANTS[p['method']],x['row_id']]['predicted_latency_ms'] for x in run['OOF_components'][p['method']]}
                values.append(weighted_objective(p['workload'],predicted_task_means(p['workload'],p['routing'],paths)))
            self.assertAlmostEqual(sum(values)/len(values),p['predicted_mean_objective_ms'])

    def test_metrics_match_configuration_objectives_and_insufficiency_is_not_fail(self):
        gs={(a,g['group_id']):g for a,groups in self.groups['analyses'].items() for g in groups}
        for record in self.metrics['groups']:
            candidates=gs[record['analysis'],record['group_id']]['candidates']
            key='true_mean_objective_ms' if record['objective']=='mean' else 'true_tail_objective_ms'
            actual=ranking_metrics([c['configuration_id'] for c in candidates],[c[key] for c in candidates],
                [c['predicted_mean_objective_ms'][record['method']] for c in candidates])
            self.assertEqual(actual,record['metrics'])
        self.assertEqual(self.metrics['decision']['status'],'INSUFFICIENT COMPARABLE GROUPS')
        self.assertIsNone(self.metrics['decision']['decision_gate'])


if __name__=='__main__':unittest.main()
