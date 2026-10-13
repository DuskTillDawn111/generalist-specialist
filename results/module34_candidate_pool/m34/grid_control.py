"""Independent finite-domain V3 control. Never relabel as native V3."""
from .common import *
import sys
from deployment_core.search.e2e_candidate_ranker import E2ECandidateRanker
from deployment_core.routing.e2e_surrogate import evaluate_surrogate
class GridJRanker(E2ECandidateRanker):
    def __init__(self,d,c,cal,K):
        super().__init__(c,cal,d['safe_factor']);self.d=d;self.K=K;self.evaluation_calls=0
    def evaluate(self,x,workload,canonical=None):
        from .cpu_core import candidate_routes
        key=(tuple(sorted(x.items())),tuple(sorted(workload.arrival_rate.items())))
        if key not in self.cache:
            active={m:n for m,n in x.items() if n};feasible=[]
            for r in candidate_routes(self.d,active,workload.arrival_rate,self.K):
                flat={(t,m):q for t,row in r.items() for m,q in row.items()};self.evaluation_calls+=1
                ev=evaluate_surrogate(active,flat,workload,self.catalog,self.calibration)
                if ev['domain_valid'] and ev['z']<=.8+1e-8:feasible.append((ev['J_seconds'],identity(active,r),flat,ev))
            if not feasible:self.cache[key]=dict(converged=False,status='FINITE_ROUTE_INFEASIBLE')
            else:
                _,_,flat,ev=min(feasible,key=lambda z:z[:2]);self.cache[key]=dict(converged=True,status='EXACT_J_IN_FROZEN_GRID',routing=flat,evaluation=ev,iterations=0)
        return self.cache[key]
def install(engine,d,c,cal,K):
    ranker=GridJRanker(d,c,cal,K)
    # create_search already loads original v2 code privately; replace only the
    # injected route/rank interface in that private module. Original generation,
    # diversified order implementation, top2 acceptance and stop logic execute.
    name=engine.__class__.__module__.rsplit('.',1)[0]+'.engine'
    sys.modules[name].diversified_order=ranker.order
    engine.e2e_ranker=ranker;engine.serving_routing_adapter.ranker=ranker
    engine.module34_finite_J_control=True
