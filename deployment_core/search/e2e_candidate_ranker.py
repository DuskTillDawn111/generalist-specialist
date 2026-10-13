from method_v2.selection import candidate_class
from deployment_core.routing.e2e_router import solve_e2e_routing

class E2ECandidateRanker:
    def __init__(self,catalog,calibration,safety_factor=1.):
        self.catalog,self.calibration,self.safety_factor=catalog,calibration,safety_factor
        self.cache={}

    def evaluate(self,x,workload,canonical=None):
        key=(tuple(sorted(x.items())),tuple(sorted(workload.arrival_rate.items())))
        if key not in self.cache:
            self.cache[key]=solve_e2e_routing(x,workload,self.catalog,self.calibration,canonical)
        return self.cache[key]

    def order(self,candidates,current,catalog,workload,current_memory):
        base=self.evaluate(current,workload)
        if not base['converged']:raise RuntimeError('Current J solve failed')
        scored=[]
        for candidate in candidates:
            fitted=self.evaluate(candidate.mutation.deployment,workload,candidate.canonical)
            if not fitted['converged']:raise RuntimeError('Candidate J numerical failure; do not prune')
            candidate.delta_J=base['evaluation']['J_seconds']-fitted['evaluation']['J_seconds']
            key=(-candidate.delta_J,fitted['evaluation']['z'],candidate.memory_mib*self.safety_factor,
                 tuple(sorted(candidate.mutation.deployment.items())))
            scored.append((key,candidate))
        global_order=[c for key,c in sorted(scored,key=lambda row:row[0])]
        pressure=[c for c in global_order if candidate_class(c)=='P']
        replacement=[c for c in global_order if candidate_class(c)=='R']
        first=pressure[0] if pressure else (replacement[0] if replacement else None)
        selected=[first] if first is not None else []
        second=next((c for c in replacement if c is not first),None)
        if second is None:second=next((c for c in global_order if c is not first),None)
        if second is not None:selected.append(second)
        return selected+[c for c in global_order if all(c is not s for s in selected)]
