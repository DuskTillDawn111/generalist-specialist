"""Independent density/clock checks, separate from solver implementation tests."""
import math
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.stats import lognorm

from tools.predictor_validation_20261027 import macro_joint_response_fit_v1 as first
from tools.predictor_validation_20261027 import macro_joint_response_second_order_v1 as second


class IndependentResponseTests(unittest.TestCase):
    def test_endpoint_density_normalizes_under_dynamic_rate_and_allows_acceleration(self):
        # Construct the response density directly, without the fitting code.
        # Omitting the endpoint Jacobian must fail the same mass check.
        for beta in (.7, -.7):
            switch=8.;r1=math.exp(-beta);r2=math.exp(-2*beta)
            prior=lognorm(s=.35,scale=10.)
            def density(t,jacobian):
                virtual=r1*min(t,switch)+r2*max(0,t-switch)
                rate=r1 if t<switch else r2
                return prior.pdf(virtual)*(rate if jacobian else 1.)
            def mass(jacobian):
                return (quad(lambda t:density(t,jacobian),0,switch,epsabs=1e-10)[0]
                    +quad(lambda t:density(t,jacobian),switch,np.inf,epsabs=1e-10)[0])
            self.assertAlmostEqual(mass(True),1.,places=9)
            self.assertGreater(abs(mass(False)-1.),.25)

    def rows(self):
        rng=np.random.default_rng(318)
        rows=[]
        for group in range(3):
            for _ in range(12):
                counts=rng.integers(0,4,size=3)
                same=np.array([rng.integers(0,int(n)+1) for n in counts])
                matrix=np.column_stack((counts,same)).astype(float)
                durations=rng.lognormal(mean=group+1,sigma=.3,size=3)
                rows.append(dict(group=(group,),durations=durations,matrix=matrix))
        return rows

    def test_hessian_matches_scalar_likelihood_cross_differences(self):
        rows=self.rows();beta=np.array([.13,-.07]);step=2e-4
        # Scalar profile likelihood predates the second-order solver and does
        # not use its derivative/Hessian implementation.
        value=lambda b:first.profile(b,rows)[0]/len(rows)
        hessian=np.empty((2,2));basis=np.eye(2)*step
        for i in range(2):
            for j in range(2):
                hessian[i,j]=(value(beta+basis[i]+basis[j])-value(beta+basis[i]-basis[j])
                    -value(beta-basis[i]+basis[j])+value(beta-basis[i]-basis[j]))/(4*step*step)
        actual,_,analytic=second.differential(beta,rows)
        self.assertAlmostEqual(actual,value(beta),places=11)
        np.testing.assert_allclose(analytic,hessian,atol=3e-7,rtol=3e-6)

    def test_response_unit_change_only_shifts_density_constant(self):
        rows=self.rows();beta=np.array([.13,-.07])
        scaled=[dict(r,durations=r['durations']*1000) for r in rows]
        v,g,h=second.differential(beta,rows)
        vs,gs,hs=second.differential(beta,scaled)
        self.assertAlmostEqual(vs-v,math.log(1000),places=11)
        np.testing.assert_allclose(g,gs,atol=1e-12)
        np.testing.assert_allclose(h,hs,atol=1e-12)

    def test_splitting_identical_exposure_does_not_change_likelihood_or_derivatives(self):
        rows=self.rows();beta=np.array([.13,-.07])
        split=[dict(r,durations=np.repeat(r['durations']/2,2),matrix=np.repeat(r['matrix'],2,axis=0))
            for r in rows]
        for original,new in zip(second.differential(beta,rows),second.differential(beta,split)):
            np.testing.assert_allclose(original,new,atol=1e-12)


if __name__=='__main__':
    unittest.main()
