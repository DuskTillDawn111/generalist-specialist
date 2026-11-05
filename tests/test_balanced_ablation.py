"""Check ablation wiring without running additional performance trials."""
import copy
import unittest

from experiments.ablate_balanced_bilevel import make_optimizer, PROTOCOL
from experiments.bilevel_scalability_benchmark import read, PROTOCOL as BASE, FullDomainSpaces
from gsdeploy.continuous.contracts import digest
from gsdeploy.continuous.budget_scheduled_bilevel import BudgetScheduledBilevelOptimizer


class BalancedAblationTests(unittest.TestCase):
    def test_plan_is_bounded_and_reuses_full_runs(self):
        cfg=read(PROTOCOL)
        self.assertEqual(len(cfg['sizes'])*len(cfg['variants'])*len(cfg['seeds']),36)
        self.assertEqual(cfg['maximum_new_searches'],36)
        self.assertEqual(cfg['budget'],read(BASE)['budget'])
        self.assertEqual(cfg['new_gpu_runs'],0)

    def test_every_variant_shares_charged_initializer_and_balanced_controller(self):
        cfg=copy.deepcopy(read(BASE));cfg['budget']['routes']=4
        signatures=[]
        for method in ['BO-MADS',*read(PROTOCOL)['variants']]:
            opt=make_optimizer(7,method,11,cfg)
            self.assertIsInstance(opt,BudgetScheduledBilevelOptimizer)
            result=opt.solve()
            signatures.append(digest(opt.initial_signature))
            self.assertEqual(result['budget']['outer_evaluations'],4)
            self.assertEqual(opt.use_bo,method!='MADS-only')
            self.assertEqual(opt.use_mads,method not in ('BO-only','Grid-K4'))
            self.assertEqual(opt.exact_inner,method=='No-progressive')
            self.assertEqual(opt.pool.disabled,method=='No-pool')
            if method=='No-activation':self.assertIsInstance(opt.spaces,FullDomainSpaces)
        self.assertEqual(len(set(signatures)),1)

    def test_disabled_pool_blocks_cross_route_retrieval_without_erasing_bounds(self):
        cfg=copy.deepcopy(read(BASE));cfg['budget']['routes']=4
        opt=make_optimizer(7,'No-pool',11,cfg);opt.solve()
        winner=opt._winner()
        self.assertIsNotNone(winner)
        self.assertTrue(opt.rows[digest(winner['routing_shares'])]['certified'])
        self.assertEqual(opt.pool.compatible(winner['routing_qps']),[])
        self.assertGreater(len(opt.rows),0)


if __name__=='__main__':unittest.main()
