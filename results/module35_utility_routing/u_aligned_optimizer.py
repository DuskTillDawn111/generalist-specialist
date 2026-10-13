"""Module35 independent U optimizer; no Oracle plans or historical winners read.

K=1: exact model/task-bundle dynamic programming with Pareto dominance.
For a fixed deployment, K>1 uses bounded discrete routing search. Both preserve
the frozen FIFO predictor, utility weights, measured replicas and rho<=.8.
The bundle simulator is independently checked against the original predictor.
"""
import itertools, json, time
import numpy as np
from module20.optimizer import predict_scenarios


def identity(x, r):
    return json.dumps([x, r], sort_keys=True)


class UAlignedOptimizer:
    def __init__(self, data, rates, budget, refs, weights):
        self.d, self.rates, self.budget = data, dict(rates), budget
        self.refs, self.w = refs, weights
        self.tasks = list(rates)
        self.cache = {}
        self.metrics = dict(worker_prediction_calls=0, full_prediction_calls=0,
                            cache_hits=0, search_states=0, transitions=0,
                            dominance_pruned=0, capacity_pruned=0,
                            memory_pruned=0, bound_pruned=0)
        self.rejections = []

    def bundle(self, m, n, mask):
        key = (m, n, mask)
        if key in self.cache:
            self.metrics['cache_hits'] += 1
            return self.cache[key]
        self.metrics['worker_prediction_calls'] += 1
        d = self.d
        selected = [t for i, t in enumerate(self.tasks) if mask >> i & 1]
        rho = sum(self.rates[t]*d['models'][m]['service_time_ms'][t]/1000
                  for t in selected)/d['capacity_factors'][m][str(n)]
        if rho > .8+1e-8:
            self.metrics['capacity_pruned'] += 1
            self.cache[key] = None
            self.rejections.append(dict(model=m, replicas=n, mask=mask, reason='RHO08', rho=rho))
            return None
        arrival, tasks, svc, replica = [], [], [], []
        for ti, t in enumerate(self.tasks):
            if t not in selected:
                continue
            q = self.rates[t]
            a = np.arange(-10*q+.37*(ti+1), 120*q, 1.)/q
            a = a[(a >= -10) & (a < 120)]
            arrival.extend(a); tasks.extend([t]*len(a))
            svc.extend([d['models'][m]['service_time_ms'][t]/1000*n/
                        d['capacity_factors'][m][str(n)]]*len(a))
            replica.extend(np.arange(len(a)) % n)
        arrival, tasks, svc, replica = map(np.array, (arrival, tasks, svc, replica))
        stats = {t: [] for t in selected}
        within = offered = late = 0
        for ri in range(n):
            ix = np.flatnonzero(replica == ri)
            ix = ix[np.argsort(arrival[ix], kind='stable')]
            a, s, tt = arrival[ix], svc[ix], tasks[ix]
            if not len(a):
                continue
            c = np.cumsum(s)
            finish = c+np.maximum.accumulate(a-(c-s))
            use = a >= 0
            lat, tt, f = (finish-a)[use]*1000, tt[use], finish[use]
            offered += len(f); within += int(sum(f <= 120)); late += int(sum(f > 180))
            for t in set(tt):
                stats[t].extend(lat[tt == t])
        if late or any(not v for v in stats.values()):
            self.rejections.append(dict(model=m, replicas=n, mask=mask, reason='DRAIN_OR_MISSING_TASK'))
            self.cache[key] = None
            return None
        cost = sum(self.rates[t]/sum(self.rates.values())/self.refs[t]*
                   (self.w['L']*float(np.mean(v))+self.w['P']*float(np.percentile(v, 95)))
                   for t, v in stats.items())
        mem = d['safe_factor']*d['memory_table_mib'][m][str(n)]
        result = dict(mask=mask, memory=mem, cost=cost+self.w['M']*mem/self.budget,
                      within=within, offered=offered, rho=rho, model=m, replicas=n)
        self.cache[key] = result
        return result

    @staticmethod
    def dominates(a, b):
        # Same prefix and assigned task mask: every remaining completion is
        # identical. Lower memory/cost and higher within-window completions
        # cannot worsen U or feasibility. No approximate dominance tolerance.
        return a['memory'] <= b['memory'] and a['cost'] <= b['cost'] and a['within'] >= b['within']

    def solve(self, replica_options=None, deployment=None, K=1):
        if K != 1:
            if deployment is None:
                raise ValueError('K>1 requires a fixed legal deployment; joint split search is outside Module35 scope')
            return self.solve_routing(deployment, K)
        start = time.perf_counter()
        if deployment is not None:
            if any(m not in self.d['models'] or str(n) not in self.d['capacity_factors'][m]
                   or str(n) not in self.d['memory_table_mib'][m] or n <= 0 for m, n in deployment.items()):
                return self.result(None, start, 'UNMEASURED_DEPLOYMENT')
            fixed_mem = self.d['safe_factor']*sum(self.d['memory_table_mib'][m][str(n)] for m,n in deployment.items())
            if fixed_mem > self.budget+1e-6:
                return self.result(None, start, 'MEMORY_INFEASIBLE')
            models = sorted(deployment)
        else:
            models = sorted(self.d['models'])
        labels = {0: [dict(memory=0., cost=0., within=0, plan=[])]}
        for m in models:
            counts = ([deployment[m]] if deployment is not None else
                      (replica_options or {}).get(m, [int(n) for n in self.d['capacity_factors'][m]
                        if int(n)>0 and n in self.d['memory_table_mib'][m]]))
            support = [i for i,t in enumerate(self.tasks) if t in self.d['models'][m]['supported_tasks']]
            options = [None]
            for n in counts:
                for bits in itertools.product((0,1), repeat=len(support)):
                    mask = sum(b << i for b,i in zip(bits,support))
                    if not mask:
                        continue
                    b = self.bundle(m,n,mask)
                    if b is not None:
                        options.append(b)
            new = {}
            for mask, group in labels.items():
                for label in group:
                    self.metrics['search_states'] += 1
                    for b in options:
                        self.metrics['transitions'] += 1
                        if b and mask & b['mask']:
                            continue
                        # Fixed deployment preserves resident idle models and
                        # pays their memory even when assigned no tasks.
                        idle_mem = (self.d['safe_factor']*self.d['memory_table_mib'][m][str(deployment[m])]
                                    if deployment is not None and b is None else 0.)
                        nm = label['memory']+(b['memory'] if b else idle_mem)
                        if nm > self.budget+1e-6:
                            self.metrics['memory_pruned'] += 1
                            continue
                        item = dict(memory=nm, cost=label['cost']+(b['cost'] if b else self.w['M']*idle_mem/self.budget),
                                    within=label['within']+(b['within'] if b else 0), plan=label['plan']+([b] if b else []))
                        target = mask | (b['mask'] if b else 0)
                        frontier = new.setdefault(target, [])
                        if any(self.dominates(z,item) for z in frontier):
                            self.metrics['dominance_pruned'] += 1
                            continue
                        survivors = [z for z in frontier if not self.dominates(item,z)]
                        self.metrics['dominance_pruned'] += len(frontier)-len(survivors)
                        new[target] = survivors+[item]
            labels = new
        finalists = labels.get((1 << len(self.tasks))-1, [])
        if not finalists:
            return self.result(None, start, 'NO_LEGAL_COMPLETE_ROUTE')
        winner = min(finalists, key=lambda z:(z['cost']+self.w['T']*max(0.,1-z['within']/(sum(self.rates.values())*120)),z['memory']))
        x = dict(deployment) if deployment is not None else {b['model']:b['replicas'] for b in winner['plan']}
        r = {t:{b['model']:self.rates[t]} for i,t in enumerate(self.tasks) for b in winner['plan'] if b['mask'] >> i & 1}
        p = predict_scenarios(self.d,x,r,self.rates,self.budget,self.refs,self.w)
        self.metrics['full_prediction_calls'] += 4
        bundle_U = -(winner['cost']+self.w['T']*max(0.,1-winner['within']/(sum(self.rates.values())*120)))
        if not p['eligible'] or p['max_rho'] > .8+1e-8 or abs(p['U']-bundle_U)>1e-8:
            return self.result(None,start,'PREDICTOR_VALIDATION_FAILED',dict(prediction=p,bundle_U=bundle_U))
        return self.result(dict(deployment=x,routing=r,prediction=p,bundle_U=bundle_U),start,'OPTIMAL_K1_DP')

    def result(self, winner, start, status, diagnostics=None):
        return dict(winner=winner,status=status,seconds=time.perf_counter()-start,metrics=dict(self.metrics),
                    rejected_bundles=self.rejections,diagnostics=diagnostics,
                    uncertainty='HIGH_UNMEASURED_COLOCATION_AND_SERVICE_TAIL',
                    scope='Exact frozen predictor/domain decision; no real GPU optimality claim')

    def solve_routing(self, x, K):
        """Exact branch/bound for optional discrete G/S splits, fixed replicas.

        Bound: mean and P95 latency cannot be less than minimum effective
        service time among allowed paths; throughput cost is nonnegative.
        Partial rho only increases. Bound never substitutes J for U.
        """
        start=time.perf_counter()
        if K<1 or int(K)!=K or any(m not in self.d['models'] or n<=0 or str(n) not in self.d['capacity_factors'][m]
                                    or str(n) not in self.d['memory_table_mib'][m] for m,n in x.items()):
            return self.result(None,start,'INVALID_DOMAIN_OR_UNMEASURED_REPLICA')
        mem=self.d['safe_factor']*sum(self.d['memory_table_mib'][m][str(n)] for m,n in x.items())
        if mem>self.budget+1e-6:
            return self.result(None,start,'MEMORY_INFEASIBLE')
        def compositions(k,total):
            if k==1:
                yield (total,);return
            for first in range(total+1):
                for tail in compositions(k-1,total-first):yield (first,*tail)
        opts={};floor={}
        for t,q in self.rates.items():
            models=[m for m in sorted(x) if t in self.d['models'][m]['supported_tasks']]
            if not models:return self.result(None,start,'COVERAGE_INFEASIBLE')
            opts[t]=[{m:q*u/K for m,u in zip(models,units) if u} for units in compositions(len(models),K)]
            floor[t]=(self.w['L']+self.w['P'])*q/sum(self.rates.values())/self.refs[t]*min(
                self.d['models'][m]['service_time_ms'][t]*x[m]/self.d['capacity_factors'][m][str(x[m])] for m in models)
        tasks=sorted(self.rates,key=lambda t:(len(opts[t]),t));best=None
        bound=-self.w['M']*mem/self.budget-sum(floor.values())
        # Use a conservative global service floor only. Do not bound P95 of a
        # split mixture by its mean or by J. This may be weak in small domains.
        def visit(i,r,rho):
            nonlocal best
            self.metrics['search_states']+=1
            if best is not None and bound < best['prediction']['U']-1e-12:
                self.metrics['bound_pruned']+=1;return
            if i==len(tasks):
                key=identity(x,r)
                if key in self.cache:
                    self.metrics['cache_hits']+=1;p=self.cache[key]
                else:
                    p=predict_scenarios(self.d,x,r,self.rates,self.budget,self.refs,self.w)
                    self.metrics['full_prediction_calls']+=4;self.cache[key]=p
                if p['eligible'] and p['max_rho']<=.8+1e-8 and (best is None or (p['U'],-p['safe_memory_mib'])>(best['prediction']['U'],-best['prediction']['safe_memory_mib'])):
                    best=dict(deployment=dict(x),routing=dict(r),prediction=p)
                elif not p['eligible']:
                    self.rejections.append(dict(routing=dict(r),reason='PREDICTED_INELIGIBLE',prediction=p))
                return
            t=tasks[i]
            for row in opts[t]:
                self.metrics['transitions']+=1;nr=dict(rho)
                for m,q in row.items():nr[m]=nr.get(m,0)+q*self.d['models'][m]['service_time_ms'][t]/1000/self.d['capacity_factors'][m][str(x[m])]
                if max(nr.values(),default=0)>.8+1e-8:
                    self.metrics['capacity_pruned']+=1;continue
                visit(i+1,{**r,t:row},nr)
        visit(0,{},{});return self.result(best,start,'OPTIMAL_FIXED_SPLIT_GRID' if best else 'NO_LEGAL_COMPLETE_ROUTE')
