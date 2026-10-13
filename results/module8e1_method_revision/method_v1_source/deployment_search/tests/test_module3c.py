import json
from pathlib import Path
import tempfile
import unittest

from deployment_search import GuidedDeploymentSearch, SearchConfig, SyntheticRuntimeEvaluator
from deployment_search.campaign_runtime import execution_status
from deployment_search.tests.fixtures import synthetic_pool, evaluation

ROOT=Path(__file__).resolve().parents[2]


class Module3CTest(unittest.TestCase):
    def test_completed_serving_failure_is_valid_execution(self):
        raw=json.loads((ROOT/'results/performance_model_comparable_groups/raw_runs/'
            'W2_generalist_heavy_cfg_a9a311acc9247bf6_rep_3.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'run.log'
            log.write_text('{"output": "result.json"}\n')
            status,_=execution_status(raw,dict(returncode=1),log)
        self.assertEqual(status,'completed')
        self.assertEqual(raw['status'],'failed')
        self.assertGreater(raw['system_metrics']['incomplete_requests'],0)

    def test_missing_harness_result_is_not_unstable(self):
        with tempfile.TemporaryDirectory() as tmp:
            status,_=execution_status(None,dict(returncode=-1),Path(tmp)/'missing.log')
        self.assertEqual(status,'uncertain_interruption')

    def test_lp_only_transition_ignores_runtime_label(self):
        workload,catalog,hardware=synthetic_pool(4)
        evaluator=SyntheticRuntimeEvaluator(lambda d,w,r:evaluation(100 if sum(d.values())==1 else 200))
        config=SearchConfig(top_b=2,max_iterations=1,runtime_evaluation_budget=3)
        b2=GuidedDeploymentSearch(evaluator,selection_policy='lp',transition_policy='runtime').search(
            workload,catalog,hardware,config)
        b3=GuidedDeploymentSearch(evaluator,selection_policy='lp',transition_policy='lp').search(
            workload,catalog,hardware,config)
        self.assertEqual(sum(b2.deployment.values()),1)
        self.assertEqual(sum(b3.deployment.values()),2)
        self.assertGreater(b3.runtime.weighted_mean_e2e,b3.seed_runtime.weighted_mean_e2e)

    def test_random_policy_reproducible(self):
        workload,catalog,hardware=synthetic_pool(5)
        evaluator=SyntheticRuntimeEvaluator(lambda d,w,r:evaluation(100 if sum(d.values())==1 else 200))
        config=SearchConfig(top_b=2,max_iterations=1,runtime_evaluation_budget=3)
        def run():
            return GuidedDeploymentSearch(evaluator,selection_policy='random',random_seed=1103).search(
                workload,catalog,hardware,config).to_dict()['iterations'][0]['ranking']
        self.assertEqual(run(),run())

if __name__=='__main__':unittest.main()
