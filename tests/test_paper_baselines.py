"""Meaningful mechanism, budget, and joint-MILP validation (no tuning runs)."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from scipy.optimize import linprog
from experiments.compare_paper_baselines_20261105 import (
    PROTOCOL,ARCHIVE,read,build_problem,ContinuousRoutingEvaluator,LoggedOracle,Budget,
    JointProfileMILP,matern52,expected_improvement,gp_predict,capacity_feasible,
    ScheduledControlledOptimizer,initialize,DirectSearch,CompactCodec,cherry_pick,BudgetExhausted)
from experiments.bilevel_scalability_benchmark import reference_deployments


class PaperBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=read(PROTOCOL);cls.base=read(ARCHIVE/'plan.json')['protocol']

    def case(self):
        p=build_problem(7,self.base)
        e=ContinuousRoutingEvaluator(p,LoggedOracle(),budget=Budget(30,64,256,128))
        return p,e,JointProfileMILP(p,e.domain,self.cfg['proteus'])

    def test_matern_and_ei_known_values(self):
        a=np.array([[0.],[1.]])
        k=matern52(a,a,1.)
        np.testing.assert_allclose(np.diag(k),1)
        self.assertAlmostEqual(k[0,1],(1+np.sqrt(5)+5/3)*np.exp(-np.sqrt(5)))
        self.assertAlmostEqual(expected_improvement([0.],[1.],0.)[0],1/np.sqrt(2*np.pi))
        self.assertEqual(expected_improvement([0.],[0.],1.)[0],0)
        self.assertGreater(np.linalg.eigvalsh(k).min(),0)

    def test_gp_interpolates_observed_cost_and_positive_uncertainty(self):
        x=np.array([[0.],[.5],[1.]]);y=np.array([.4,.1,.3])
        mu,std,length=gp_predict(x,y,x,self.cfg['cherrypick'])
        np.testing.assert_allclose(mu,y,atol=1e-5)
        self.assertTrue(np.all(std>=0));self.assertTrue(.05<=length<=5)

    def test_milp_matches_independent_enumeration_and_continuous_lp(self):
        p,e,planner=self.case()
        coef,_=planner.coefficients('Proteus-Q',[]);result=planner.solve(coef,e.budget)
        self.assertTrue(result.success)
        best=np.inf
        for d in reference_deployments(p):
            bounds=list(zip(planner.bounds.lb,planner.bounds.ub))
            for (m,n),idx in planner.si.items():bounds[idx]=(int(d.get(m)==n),)*2
            a=planner.constraints.A;lo=planner.constraints.lb;hi=planner.constraints.ub
            # General lower/upper constraints independently converted to LP.
            aub=[];bub=[]
            for row,l,h in zip(a,lo,hi):
                if np.isfinite(h):aub.append(row);bub.append(h)
                if np.isfinite(l):aub.append(-row);bub.append(-l)
            lp=linprog(coef,A_ub=aub,b_ub=bub,bounds=bounds,method='highs')
            if lp.success:best=min(best,lp.fun)
        self.assertAlmostEqual(result.fun,best,places=9)
        d,r=planner.decode(result.x)
        self.assertTrue(capacity_feasible(p,d,r))
        self.assertAlmostEqual(coef@planner.features(d,r),result.fun,places=9)

    def test_inactive_state_cannot_receive_flow(self):
        p,e,planner=self.case();x=np.zeros(len(planner.integrality))
        x[planner.fi['text','g',1]]=.2
        self.assertTrue(np.any(planner.constraints.A@x>planner.constraints.ub+1e-9))

    def test_utility_fit_depends_only_on_paid_observations(self):
        p,e,planner=self.case()
        obs=[dict(deployment={'g':1},shares={'text':{'g':1.},'image':{'g':1.}},U=-.5),
             dict(deployment={'g':2},shares={'text':{'g':1.},'image':{'g':1.}},U=-.3)]
        c,meta=planner.coefficients('Proteus-U-linear',obs)
        for model in p.config.catalog['models'].values():model['benchmark_quality']={}
        c2,meta2=planner.coefficients('Proteus-U-linear',obs)
        np.testing.assert_array_equal(c,c2);self.assertEqual(meta,meta2)
        self.assertEqual(meta['fit_samples'],2)
        self.assertLess(meta['training_rmse'],.001)
        changed=copy.deepcopy(obs);changed[1]['U']=-.2
        self.assertFalse(np.allclose(c,planner.coefficients('Proteus-U-linear',changed)[0]))

    def test_exhausted_budget_prevents_milp_and_gp(self):
        p,e,planner=self.case();e.budget.counts['predictions']=64
        with patch('experiments.compare_paper_baselines_20261105.milp') as mocked:
            with self.assertRaisesRegex(BudgetExhausted,'PREDICTIONS_LIMIT'):planner.solve(np.zeros(len(planner.integrality)),e.budget)
            mocked.assert_not_called()

    def test_ei_stop_retains_initial_incumbent_without_spending(self):
        p,e,planner=self.case();opt=ScheduledControlledOptimizer(e,'BO-MADS',11,self.base);initialize(opt)
        search=DirectSearch(e,opt,11,self.cfg);before=e.budget.counts['predictions'];incumbent=copy.deepcopy(search.incumbent)
        def certain_bad(x,y,q,settings):return np.ones(len(q))*100,np.zeros(len(q)),1.
        with patch('experiments.compare_paper_baselines_20261105.gp_predict',side_effect=certain_bad):
            status,log=cherry_pick(search,CompactCodec(p,e.domain,11),self.cfg,11)
        self.assertEqual(status,'RELATIVE_EI_STOP');self.assertEqual(e.budget.counts['predictions'],before)
        self.assertEqual(search.incumbent,incumbent);self.assertTrue(log[-1]['stop'])


if __name__=='__main__':unittest.main()
