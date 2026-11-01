"""Proved lower envelopes for the explicit development model, never runtime.

S_i=b_i exp(alpha_family+beta_self A_self+beta_other A_other).
With nonnegative pressures and slopes, b_i exp(alpha_family)<=S_i.
Deleting other tasks and reducing every service cannot increase FIFO finish
times, empirical mean/P95, or late count. Frozen routed Q is exact.
Memory NNLS coefficients are nonnegative: omitting intercept/pressure leaves
an additive residency/worker lower envelope. Energy lower envelope is zero.
"""
import math
from collections import defaultdict


class DevelopmentBounds:
    def __init__(self,oracle,seeds):
        self.o=oracle;self.seeds=seeds;self.cache={};self.evaluations=0
        cols=oracle.columns;coefs=dict(zip(cols,oracle.beta))
        if set(cols)-set(range(9)):raise ValueError('Envelope only proved for family+self+other')
        if any(coefs.get(i,0)<-1e-12 for i in [7,8]):raise ValueError('Pressure slopes must be nonnegative')
        self.alpha={m:coefs.get(i,0) for i,m in enumerate(['generalist_1','generalist_2','generalist_3','text','image','ocr','asr'])}
        model=oracle.resources['models'][oracle.memory_name]
        self.mb=dict(zip(model['features'],model['all_run_fit']['coefficients']))
        if any(v<0 for v in self.mb.values()):raise ValueError('Memory lower envelope requires nonnegative coefficients')
        if set(self.mb)-{'intercept','raw_resident_over_budget','sum_baseline_pressure','workers'}:raise ValueError('Unknown memory feature')

    def memory_norm(self,x):
        raw=sum(self.o.catalog['memory_table_mib'][m][str(n)] for m,n in x.items())/self.o.config['gpu_memory_mib']
        return self.o.catalog['safe_factor']*(self.mb.get('raw_resident_over_budget',0)*raw+self.mb.get('workers',0)*sum(x.values()))

    def task_floor(self,task,x,row):
        key=(task,tuple(sorted((m,x[m]) for m in row)),tuple(sorted(row.items())))
        if key in self.cache:return self.cache[key]
        self.evaluations+=1;values=[];rates=self.o.config['workload'];w=self.o.config['weights'];h=self.o.config['horizon_seconds']
        total=sum(rates.values());alpha=rates[task]/total/self.o.config['latency_reference_ms'][task]
        for seed in self.seeds:
            trace,cohort=self.o._cohort(seed);aliases,routes=self.o.api['trace'].expand_routes(x,{task:row});router=self.o.api['routing'].Router(routes)
            clocks=defaultdict(float);lat=[];late=0
            for arrival,t,sid in cohort:
                if t!=task:continue
                alias=router.choose(task);m=aliases[alias];bs=self.o._baselines(m,t,x[m]);b=(bs or {}).get(sid)
                # Missing exact path has no established positive bound.
                s=b*math.exp(self.alpha[m]) if b is not None and (m,t,x[m]) in self.o.domain else 0.
                finish=max(arrival,clocks[alias])+s/1000;clocks[alias]=finish
                lat.append((finish-arrival)*1000);late+=finish>h
            loss,_=self.o.api['quality'].task_quality(self.o.catalog,task,x,row,trace,seed)
            q=w['Q']*rates[task]/total*loss if loss is not None else 0.
            ordered=sorted(lat);idx=(len(ordered)-1)*.95;lo=math.floor(idx);hi=math.ceil(idx);p=ordered[lo]+(ordered[hi]-ordered[lo])*(idx-lo)
            values.append(q+alpha*(w['L']*sum(lat)/len(lat)+w['P']*p)+w['T']*late/len(trace))
        bound=sum(values)/len(values);self.cache[key]=bound;return bound

    def upper(self,x,rows,memory):
        return -(self.o.config['weights']['M']*self.memory_norm(x)+sum(self.task_floor(t,x,row) for t,row in rows.items()))

    def install(self,solver,algorithm):
        if algorithm=='milp':
            solver.floor=self.task_floor
            solver.state_memory=lambda m,n: self.memory_norm({m:n})*self.o.config['gpu_memory_mib'] if n else 0.
        elif algorithm!='plain_dfs':
            solver.upper=self.upper
            solver.service_floor=lambda x,t,row=None:self.task_floor(t,x,row) if row is not None else 0.

    def contract(self):
        return dict(schema='gsdeploy.independent_development_envelope.v1',production_activated=False,
            service='per-input b*exp(alpha_family), with nonnegative pressure slopes',
            latency='task-independent FIFO on identical trace/SWRR and exact input assignment',
            memory='nonnegative NNLS residency+worker terms; intercept/pressure omitted',energy='zero',
            quality='exact frozen per-task/seed routed quality',bounds_cached=len(self.cache),bound_evaluations=self.evaluations,
            physical_bound=False,service_columns=self.o.columns,service_coefficients=self.o.beta,memory_coefficients=self.mb)
