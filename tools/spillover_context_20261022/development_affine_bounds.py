"""Exact affine resource decomposition of development candidates only.

This does not assert additivity of measured whole-board memory or energy.
It decomposes the already selected development linear mathematical objective.
"""
import math
import statistics
from pathlib import Path
from tools.spillover_context_20261022.development_bounds import DevelopmentBounds


class DevelopmentAffineBounds(DevelopmentBounds):
    def __init__(self,oracle,seeds):
        super().__init__(oracle,seeds);self.affine_cache={};self.pressure_cache={}
        oracle._bind(Path(__file__));oracle._bind(Path(__file__).with_name('search_development_affine.py'))
        self.resource_coefficients={}
        for name in [oracle.memory_name,oracle.window_name,oracle.startup_name]:
            model=oracle.resources['models'][name];coef=dict(zip(model['features'],model['all_run_fit']['coefficients']))
            if any(v<0 for v in coef.values()):raise ValueError('Affine resource model has negative coefficients')
            if set(coef)-{'intercept','raw_resident_over_budget','sum_baseline_pressure','workers'}:raise ValueError('Unknown affine feature')
            self.resource_coefficients[name]=coef
        c=oracle.config;w=c['weights'];safe=oracle.catalog['safe_factor'];watts=c['energy_reference_watts'];am=c['startup_amortization_seconds']
        self.weighted={feature:w['M']*safe*self.resource_coefficients[oracle.memory_name].get(feature,0)+
            w['E']/watts*(self.resource_coefficients[oracle.window_name].get(feature,0)+self.resource_coefficients[oracle.startup_name].get(feature,0)/am)
            for feature in ['intercept','raw_resident_over_budget','sum_baseline_pressure','workers']}

    def state_cost(self,x):
        raw=math.fsum(self.o.catalog['memory_table_mib'][m][str(n)] for m,n in x.items())/self.o.config['gpu_memory_mib']
        return self.weighted['raw_resident_over_budget']*raw+self.weighted['workers']*sum(x.values())

    def task_floor(self,task,x,row):
        key=(task,tuple(sorted((m,x[m]) for m in row)),tuple(sorted(row.items())))
        if key in self.affine_cache:return self.affine_cache[key]
        cost=super().task_floor(task,x,row);pressure=0.
        for m,q in row.items():
            bs=self.o._baselines(m,task,x[m]);factor=self.o.catalog.get('capacity_factors',{}).get(m,{}).get(str(x[m]))
            if bs and all(v is not None for v in bs.values()) and factor and factor>0:
                pressure+=q*statistics.mean(bs.values())/1000/factor
        cost+=self.weighted['intercept']*self.o.config['workload'][task]/sum(self.o.config['workload'].values())
        cost+=self.weighted['sum_baseline_pressure']*pressure
        self.affine_cache[key]=cost;return cost

    def upper(self,x,rows,memory):
        # Missing task columns retain zero contribution, including their share
        # of the nonnegative intercept. Complete plans include it exactly once.
        return -(self.state_cost(x)+math.fsum(self.task_floor(t,x,row) for t,row in rows.items()))

    def pressure_service(self,m,task,n):
        key=(m,task,n)
        if key not in self.pressure_cache:
            bs=self.o._baselines(m,task,n);factor=self.o.catalog.get('capacity_factors',{}).get(m,{}).get(str(n))
            mean=statistics.mean(bs.values())*math.exp(self.alpha[m]) if bs and all(v is not None for v in bs.values()) and key in self.o.domain else 0.
            # Existing master/DFS divides this by factor. Encode the proved
            # rho lower envelope mean(S_min)/n, cancelling factor exactly once.
            # Diagnostic gate allows +1e-8, whereas master uses exact .8.
            # Relax the necessary floor to preserve that same tolerance.
            self.pressure_cache[key]=mean*factor/n*(.8/(.8+1e-8)) if factor else 0.
        return self.pressure_cache[key]

    def install(self,solver,algorithm):
        if algorithm=='milp':
            if not solver.profiled_memory:raise ValueError('Synthetic objective units cannot constrain physical memory')
            if self.o.config['weights']['M']<=0:raise ValueError('State-cost encoding requires positive M weight')
            solver.floor=self.task_floor
            solver.pressure_service=self.pressure_service
            # Existing master state objective is wM*state_memory/budget. Only
            # encode weighted resource objective there; no memory constraint
            # uses it because profiled_memory is explicitly true.
            solver.state_memory=lambda m,n:self.state_cost({m:n})*self.o.config['gpu_memory_mib']/self.o.config['weights']['M'] if n else 0.
        elif algorithm!='plain_dfs':
            solver.upper=self.upper
            solver.service_floor=lambda x,t,row=None:self.task_floor(t,x,row) if row is not None else 0.
            solver.pressure_service=lambda m,n,t:self.pressure_service(m,t,n)
            solver.separable_service=True

    def contract(self):
        return dict(**super().contract(),affine_resources=True,resource_coefficients=self.resource_coefficients,
            weighted_affine_coefficients=self.weighted,constant_accounting='Distributed over task columns by rate/total; sums to one at complete plan',
            state_memory_field='Weighted resource objective encoded in master units only; no physical memory constraint',
            necessary_rho='mean(b*exp(alpha_family))/n, relaxed by .8/(.8+1e-8) for gate tolerance; inserted as service*factor/n so solver factor cancels exactly once',
            resource_additivity_claim='Only selected development linear candidates, never measured whole-board observations')
