"""Synthetic algorithm contracts, without production runtime values."""
import unittest
from dataclasses import replace
from unittest.mock import patch
from deployment_core import Hardware, Workload
from deployment_search.integer_foundation import model, runtime
from deployment_search import IntegerGuidedDeploymentSearch, SearchConfig
from deployment_search.real_objective import NoiseAwareRuntimeObjective
from deployment_search.feedback import bottleneck_feedback
from deployment_search.routing import solve_canonical_routing
from deployment_search.result import CandidateEvaluation
from deployment_search.neighborhood import Mutation
from deployment_search.tests.fixtures import forbid_enumeration
from . import MethodV2, frozen_config
from .neighborhood import generate_targeted_neighborhood
from .selection import diversified_order, candidate_class


def generate(c,x,w,h=10.,bounds=None,roles=None):
    bounds=bounds or {m:1 for m in c};route=solve_canonical_routing(x,w,c,replica_bounds=bounds)
    return generate_targeted_neighborhood(x,c,w,bottleneck_feedback(route,x),bounds,
        lambda y:sum(c[m].memory_mib*n for m,n in y.items()),h,roles)

class RevisionTests(unittest.TestCase):
    def test_two_step_barrier_and_exact_closure(self):
        c={'A':model('A',1.,text=100.),'B':model('B',1.,text=50.)};w=Workload({'text':1.});h=Hardware(5.)
        surface=lambda x: 30. if x['A'] and x['B'] else (100. if x['A'] else 10.)
        # An explicit synthetic intermediate acceptance barrier, not a runtime label.
        surface=lambda x: 110. if x['A'] and x['B'] else (100. if x['A'] else 10.)
        v1=IntegerGuidedDeploymentSearch(runtime(surface),runtime_objective=NoiseAwareRuntimeObjective(1.,.05)).search(w,c,h,frozen_config(),initial_deployment={'A':1,'B':0},replica_bounds={'A':1,'B':1})
        v2=MethodV2(runtime(surface)).search(w,c,h,initial_deployment={'A':1,'B':0},replica_bounds={'A':1,'B':1})
        self.assertEqual(v1.deployment,{'A':1,'B':0});self.assertEqual(v2.deployment,{'A':0,'B':1})
        swap=next(ca for ca in v2.iterations[0]['candidates'] if ca.mutation.operation=='SWAP')
        self.assertTrue(swap.mutation.exact_replacement_closure);self.assertTrue(swap.selected_top_b)
        self.assertEqual(swap.status,'ACCEPTED');self.assertLessEqual(v2.runtime_evaluation_count,6)

    def test_pressure_vs_beneficial_removal(self):
        c={'slow':model('slow',2.,text=200.),'fast':model('fast',1.,text=20.),
           'add1':model('add1',1.,text=10.),'add2':model('add2',1.,text=11.)}
        x={m:int(m in ('slow','fast')) for m in c};w=Workload({'text':1.})
        cs=[CandidateEvaluation(Mutation('SCALE_UP',m,dict(x,**{m:1})),delta_z=z,memory_mib=4.) for m,z in [('add1',.3),('add2',.2)]]
        cs.append(CandidateEvaluation(Mutation('SCALE_DOWN','slow',dict(x,slow=0)),delta_z=-.1,memory_mib=1.))
        chosen=diversified_order(cs,x,c,w,3.)[:2]
        self.assertEqual([candidate_class(ca) for ca in chosen],['P','R'])
        # Actual calls to an explicit synthetic runtime table, no GPU.
        ev=runtime(lambda y:10. if not y['slow'] else 110.)
        for ca in chosen:ev.evaluate(ca.mutation.deployment,w,{})
        self.assertEqual(len(ev.calls),2);self.assertEqual(chosen[1].mutation.model,'slow')

    def test_swap_coverage_filter(self):
        c={'A':model('A',1.,text=100.,visual_qa=100.),'B':model('B',1.,text=50.)}
        candidates,audit=generate(c,{'A':1,'B':0},Workload({'text':1.,'visual_qa':.1}))
        self.assertFalse(any(m.operation=='SWAP' for m in candidates))

    def test_swap_memory_filter(self):
        c={'A':model('A',1.,text=100.),'B':model('B',8.,text=50.)}
        candidates,audit=generate(c,{'A':1,'B':0},Workload({'text':1.}),h=5.)
        self.assertFalse(any(m.operation=='SWAP' for m in candidates));self.assertGreater(audit['swap_filter_rejections']['memory'],0)

    def test_replica_aware_swap(self):
        c={'A':model('A',1.,text=100.),'B':model('B',1.,text=50.)}
        candidates,audit=generate(c,{'A':2,'B':0},Workload({'text':1.}),bounds={'A':2,'B':2})
        mu=next(m for m in candidates if m.operation=='SWAP')
        self.assertEqual(mu.deployment,{'A':1,'B':1});self.assertEqual(sum(mu.deployment.values()),2)
        self.assertFalse(mu.service_dominated_removal)

    def test_closure_requires_new_family_and_new_redundancy(self):
        c={'slow':model('slow',1.,text=200.),'fast':model('fast',1.,text=20.),'new':model('new',2.,text=10.)}
        mus,_=generate(c,{'slow':1,'fast':1,'new':0},Workload({'text':1.}),bounds={m:2 for m in c})
        self.assertFalse(any(getattr(m,'exact_replacement_closure',False) for m in mus if getattr(m,'remove_model',None)=='slow'))
        self.assertFalse(any(getattr(m,'exact_replacement_closure',False) for m in mus if getattr(m,'add_model',None)=='fast'))

    def test_negative_delta_and_fallback(self):
        c={'A':model('A',1.,text=100.),'B':model('B',1.,text=50.)};x={'A':1,'B':1};w=Workload({'text':1.})
        cs=[CandidateEvaluation(Mutation('SCALE_DOWN',m,dict(x,**{m:0})),delta_z=-z,memory_mib=1.) for m,z in [('A',.3),('B',.1)]]
        self.assertEqual([ca.mutation.model for ca in diversified_order(cs,x,c,w,2.)],['A','B'])
        cs=[CandidateEvaluation(Mutation('SCALE_UP',m,x),delta_z=z,memory_mib=2.) for m,z in [('A',.1),('B',.2)]]
        self.assertEqual([ca.mutation.model for ca in diversified_order(cs,x,c,w,2.)],['B','A'])

    def test_binary_integer_and_no_enumeration(self):
        c={'A':model('A',1.,text=100.),'B':model('B',1.,text=50.)};w=Workload({'text':1.})
        for bounds in ({'A':1,'B':1},{'A':2,'B':2}):
            with forbid_enumeration():
                r=MethodV2(runtime(lambda x:100. if x['A'] else 10.)).search(w,c,Hardware(5.),initial_deployment={'A':1,'B':0},replica_bounds=bounds)
            self.assertLessEqual(r.runtime_evaluation_count,6)
            for it in r.iterations:
                self.assertLessEqual(len(it['selected_top_b']),2)
                for ca in it['candidates']:self.assertTrue(all(0<=n<=bounds[m] and type(n) is int for m,n in ca.mutation.deployment.items()))

    def test_no_model_name_preference(self):
        for old,new in [('alpha','beta'),('zeta','gamma')]:
            c={old:model(old,2.,text=200.),new:model(new,1.,text=20.)};x={old:1,new:1};w=Workload({'text':1.})
            cs=[CandidateEvaluation(Mutation('SCALE_DOWN',m,dict(x,**{m:0})),delta_z=z,memory_mib=3.-c[m].memory_mib) for m,z in [(old,-.9),(new,-.1)]]
            self.assertEqual(diversified_order(cs,x,c,w,3.)[0].mutation.model,old)

    def test_frozen_budget_predictor_and_objective(self):
        with self.assertRaises(ValueError):MethodV2(predictor=object())
        c={'A':model('A',1.,text=100.)};w=Workload({'text':1.})
        with self.assertRaises(ValueError):MethodV2(runtime(lambda x:1.)).search(w,c,Hardware(5.),config=replace(frozen_config(),runtime_evaluation_budget=7))
        with self.assertRaises(ValueError):MethodV2(runtime_objective=NoiseAwareRuntimeObjective(1.,.01))

if __name__=='__main__':unittest.main()
