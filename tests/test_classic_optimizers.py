"""Library wiring, complete deployment reachability and hard-budget semantics."""
import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from experiments.compare_classic_optimizers_20261105 import (
    ARCHIVE, PROTOCOL, CompactCodec, LibraryObjective, optimize, read,
    build_problem, LoggedOracle, ContinuousRoutingEvaluator, Budget,
    ScheduledControlledOptimizer, initialize, DirectSearch, BudgetExhausted)
from experiments.bilevel_scalability_benchmark import reference_deployments
from gsdeploy.continuous.contracts import digest


class ClassicOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=read(PROTOCOL);cls.base=read(ARCHIVE/'plan.json')['protocol']

    def setup_case(self,n=7,predictions=64):
        problem=build_problem(n,self.base)
        e=ContinuousRoutingEvaluator(problem,LoggedOracle(),budget=Budget(30,predictions,256,128))
        return problem,e,CompactCodec(problem,e.domain,11)

    def test_all_legal_deployments_reachable_and_roundtrip_flows(self):
        for n in self.cfg['sizes']:
            p,e,codec=self.setup_case(n)
            reached=set()
            for i,j,k in itertools.product(range(n),range(n),range(3)):
                d,r=codec.decode([(i+.5)/n,(j+.5)/n,(k+.5)/3,.173,.619])
                reached.add(digest(d))
                d2,r2=codec.decode(codec.encode(d,r))
                self.assertEqual(d2,d)
                for task in r:
                    for m in r[task]:self.assertAlmostEqual(r[task][m],r2[task][m])
            self.assertEqual(reached,{digest(d) for d in reference_deployments(p)})

    def test_endpoints_keep_zero_flow_residents(self):
        p,e,codec=self.setup_case()
        d={'g':1,'s_shared':1};r=e.domain.validate({t:{'g':0.,'s_shared':1.} for t in e.domain.edges})
        dd,rr=codec.decode(codec.encode(d,r))
        self.assertEqual(dd,d);self.assertEqual(rr,r)

    def test_random_decode_flow_conservation_and_boundary_repair(self):
        p,e,codec=self.setup_case()
        for z in [np.zeros(5),np.ones(5),*np.random.default_rng(13).random((100,5))]:
            d,r=codec.decode(z);q=e.domain.to_qps(r)
            for task,flow in q.items():
                self.assertAlmostEqual(sum(flow.values()),p.config.workload[task])
                self.assertTrue(set(flow)<=set(d))

    def objective(self,predictions=64):
        p,e,codec=self.setup_case(predictions=predictions)
        opt=ScheduledControlledOptimizer(e,'BO-MADS',11,self.base);initialize(opt)
        search=DirectSearch(e,opt,11,self.cfg)
        return e,opt,codec,LibraryObjective(search,codec,'unit',self.cfg['objective_attempt_cap'])

    def test_objective_global_cap_cannot_be_softened_by_library(self):
        e,opt,codec,obj=self.objective(predictions=19)
        d={'g':1,'s_shared':1};r={t:{'g':.173,'s_shared':.827} for t in e.domain.edges}
        z=codec.encode(d,r);obj(z)
        before=e.budget.counts.copy()
        with self.assertRaisesRegex(BudgetExhausted,'PREDICTIONS_LIMIT'):obj(z)
        self.assertEqual(e.budget.counts,before)
        self.assertEqual(len(e.oracle.calls),19)

    def test_both_algorithms_call_scipy_entrypoints(self):
        e,opt,codec,obj=self.objective()
        fake=SimpleNamespace(message='mock',success=True,nfev=0,nit=0)
        with patch('experiments.compare_classic_optimizers_20261105.differential_evolution',return_value=fake) as de:
            optimize('SciPy-DE',obj,codec,opt,self.cfg,11)
            kwargs=de.call_args.kwargs
            self.assertEqual(kwargs['strategy'],'best1bin')
            self.assertEqual(kwargs['init'].shape,(10,5))
            self.assertFalse(kwargs['polish'])
            for i,r in enumerate(opt.inner.history[:4]):
                d,shares=codec.decode(kwargs['init'][i])
                self.assertEqual(d,r['deployment']);self.assertEqual(shares,r['routing_shares'])
        with patch('experiments.compare_classic_optimizers_20261105.dual_annealing',return_value=fake) as da:
            optimize('SciPy-DA',obj,codec,opt,self.cfg,11)
            self.assertFalse(da.call_args.kwargs['no_local_search'])
            d,r=codec.decode(da.call_args.kwargs['x0'])
            self.assertEqual(d,obj.search.incumbent['deployment'])
            self.assertEqual(r,obj.search.incumbent['shares'])

    def test_fixed_batch_no_gpu_and_common_model_order(self):
        self.assertEqual(len(self.cfg['sizes'])*len(self.cfg['seeds'])*len(self.cfg['methods']),24)
        self.assertEqual(self.cfg['maximum_searches'],24)
        self.assertEqual(self.cfg['new_gpu_runs'],0)
        p,e,c=self.setup_case()
        self.assertEqual(c.names,CompactCodec(p,e.domain,11).names)


if __name__=='__main__':unittest.main()
