"""Safe relaxations of the declared stage-table predictor, not physical bounds.

For any successful forecast, each segment's instantaneous speed is no greater
than its maximum table speed. Dropping queues/admission and allowing each input
to choose a different executor only reduces latency. Missing primitives give
zero floors and never remove an executor. No independent-service bound is used.
"""

import math
import time

import numpy as np
from ..linear_solver import linprog

from ..paper_forecast import canonical
from ..quality import _document, _path, sample_totals
from ..quality import predict_quality
from ..trace import expand_routes
from ..routing import Router
from .contracts import digest


class PaperDeploymentBounds:
    def __init__(self, problem, oracle):
        self.problem, self.oracle = problem, oracle
        self.session = oracle.session.module('paper_forecast').session(oracle._config.catalog)
        self.identity = digest([problem.identity, oracle.version, 'stage-max-rate-relaxation-v1'])
        self.cache = {}

    def _primitives(self, deployment):
        key = canonical(deployment)
        if key in self.cache:
            return self.cache[key]
        found = self.session.entries.get(key)
        if found is None:
            return None
        entry, engine = found
        scenario = self.session.spec['scenarios']['1.0']
        doc = entry['profiles']
        speeds = {}
        power, memory = [], []
        # Exact table maxima are not extrema of a categorical exponential law
        # or an order-conditioned baseline. Until their global extrema have a
        # separate proof, keep zero service/resource floors for those models.
        empirical='native_activity_law' in doc or 'native_order_law' in doc
        for context in (() if empirical else doc['contexts'].values()):
            value = context['scenarios'].get(scenario)
            if value is None:
                continue
            for alias, member in context['members'].items():
                k = (member['profile_id'], member['segment_id'])
                speeds[k] = max(speeds.get(k, 0.), value['progress'][alias])
            power.append(value['board_watts'])
            memory.append(value['peak_memory_mib'])
        for context in (() if empirical else doc.get('preparation_contexts', {}).values()):
            value = context['scenarios'].get(scenario)
            if value:
                power.append(value['board_watts'])
                memory.append(value['peak_memory_mib'])
        # Zero is retained for absent resources; a lower bound is not a
        # substitute for the predictor's mandatory complete evidence checks.
        resource_floor = lambda values: min(values) if values and all(v is not None for v in values) else 0.
        times,handlers,receipts,readies = {},{},{},{}
        for profile in doc['profiles']:
            ms = 0.
            for segment in profile['segments']:
                rate = speeds.get((profile['id'], segment['id']))
                duration = segment['duration_ms'].get(scenario)
                if rate is not None and duration is not None:
                    ms += duration / rate
            k=(profile['family'],profile['task'],profile['pool_id'])
            handlers[k]=max(0.,ms-1e-5*(len(profile['segments'])+1))
            life = engine.lifecycle_provider.lookup.get(k)
            receipts[k]=readies[k]=0.
            if life:
                delays=life['scenarios'].get(scenario,{})
                receipts[k]=max(0.,(delays.get('parent_receipt_ms') or 0.)-1e-5)
                readies[k]=max(0.,(delays.get('worker_next_ready_ms') or 0.)-1e-5)
            # Simulator output and arrivals round to integer ns. Relax more
            # than the possible output rounding, including multiple segments.
            times[k] = handlers[k]+receipts[k]
        result = dict(entry=entry, times=times,handlers=handlers,receipts=receipts,readies=readies,
                      memory=resource_floor(memory), power=resource_floor(power))
        self.cache[key] = result
        return result

    def fixed_trace(self, deployment, routing, trace, ids, primitives):
        """One optimistic FIFO clock per alias shared by ALL its tasks.

        Each handler runs at its fastest declared stage rates, no admission or
        interference waits. READY still serializes successive handlers; receipt
        can precede READY and does not substitute for it. Monotone recurrence
        gives per-request lower finishes for the exact Router mapping.
        """
        aliases,rows=expand_routes(deployment,routing)
        router=Router(rows);ready={a:0. for a in aliases};result=[]
        for index,((offset,task),sample) in enumerate(zip(trace.arrivals,ids)):
            alias=router.choose(task);k=(aliases[alias],task,sample)
            arrival=round(offset*1e9)
            end=max(float(arrival),ready[alias])+primitives['handlers'].get(k,0.)*1e6
            receipt=end+primitives['receipts'].get(k,0.)*1e6
            ready[alias]=end+primitives['readies'].get(k,0.)*1e6
            result.append(dict(index=index,alias=alias,task=task,input=sample,
                finish_floor_ns=receipt,latency_floor_ms=max(0.,(receipt-arrival)/1e6-1e-5)))
        return result

    def evaluate(self, deployment, incumbent=None, *, routing=None, deadline=None):
        self.oracle.check()
        p, c = self.problem, self.problem.config
        base = dict(deployment=dict(deployment), upper_bound=None, safe_pruned=False,
            feasibility='UNKNOWN', reason='UNSUPPORTED_DEPLOYMENT_EXECUTION_CONTEXT',
            physical_bound=False, fixed_route_feedback_reused=False,
            scope='all continuous routes under this declared stage predictor; not a GPU guarantee')
        from .. import normalization as norm
        if norm.is_v2(c.catalog):
            from .predictor_adapter import global_cost_floor
            floor = global_cost_floor(self.oracle, p)
            return dict(base, upper_bound=-floor+1e-10 if floor is not None else None,
                        reason='REFERENCE_V2_GLOBAL_QUALITY_FLOOR_ONLY')
        primitives = self._primitives(deployment)
        if primitives is None:
            return base  # Unsupported is UNKNOWN, never an infeasibility proof.
        entry = primitives['entry']
        edges = [(t,m) for t in c.workload for m in deployment
                 if t in c.catalog['models'][m]['supported_tasks']]
        # Feasibility relaxation under the ORIGINAL rho/g(n), independently
        # supplied per deployment. Missing capacity values relax to zero.
        Aeq = [[float(t==task) for t,m in edges] for task in c.workload]
        Aub = [[entry['capacity_service_ms'].get(m,{}).get(t,0.)/1000 if m==model else 0.
                for t,m in edges] for model in deployment]
        if routing is not None:
            bad=any(math.fsum(routing[t].get(m,0.)*entry['capacity_service_ms'].get(m,{}).get(t,0.)/1000
                             for t in c.workload)>(.8+1e-8)*c.catalog['capacity_factors'][m][str(n)]+1e-8
                    for m,n in deployment.items())
            status=2 if bad else 0
        else:
            if 'capacity_lp_status' not in primitives:
                seconds=min(.1,max(0.,deadline-time.perf_counter())) if deadline is not None else .1
                if seconds>0:
                    fit = linprog(np.zeros(len(edges)), A_eq=Aeq, b_eq=list(c.workload.values()),
                        A_ub=Aub, b_ub=[(.8+1e-8)*c.catalog['capacity_factors'][m][str(deployment[m])]
                                        for m in deployment], bounds=(0,None),method='highs',options={'time_limit':seconds})
                    if fit.status in (0,2):primitives['capacity_lp_status']=fit.status
            status=primitives.get('capacity_lp_status',1)
        # A slightly expanded rho domain protects near-boundary candidates.
        # This is a solver-numerical certificate, not an exact-arithmetic proof.
        if status == 2:
            return dict(base,safe_pruned=True,feasibility='INFEASIBLE',reason='ORIGINAL_RHO_SIMPLEX_INFEASIBLE')
        memory = c.catalog['safe_factor']*primitives['memory']
        if memory > c.gpu_memory_mib+1e-6+1e-8*max(1.,c.gpu_memory_mib):
            return dict(base,safe_pruned=True,feasibility='INFEASIBLE',reason='ALL_CONTEXT_MEMORY_FLOOR',memory_mib=memory)
        energy = primitives['power']/c.energy_reference_watts
        if entry['startup_joules'] is not None:
            energy += entry['startup_joules']/((c.startup_amortization_seconds or c.horizon_seconds)*c.energy_reference_watts)
        try:
            quality = _document(c.catalog)
        except (ValueError,KeyError,TypeError):
            quality = None
        floors = [];inevitable_drain=False
        for trace in p.train_traces:
            # The paper interface uses precisely the same route-independent
            # seed/input replay convention as this helper.
            cal = self.oracle.session.module('calibration_v2')
            data = dict(c.catalog)
            data['quality'] = dict(data.get('quality',{}),input_order=self.session.spec['input_order'])
            ids = cal.replay_input_ids(data,trace.arrivals,trace.seed)
            fixed=self.fixed_trace(deployment,routing,trace,ids,primitives) if routing is not None else None
            terms = dict(Q=0.,T=0.,L=0.,P=0.,M=memory/c.gpu_memory_mib,E=energy)
            for task in c.workload:
                models = [m for t,m in edges if t==task]
                samples = [sample for (_,t),sample in zip(trace.arrivals,ids) if t==task]
                times = ([r['latency_floor_ms'] for r in fixed if r['task']==task] if fixed is not None else
                         [min(primitives['times'].get((m,task,sample),0.) for m in models) for sample in samples])
                alpha = c.workload[task]/math.fsum(c.workload.values())
                terms['L'] += alpha*float(np.mean(times))/c.latency_reference_ms[task]
                terms['P'] += alpha*float(np.percentile(times,95))/c.latency_reference_ms[task]
                if c.weights.get('Q',0) and quality:
                    try:
                        paths = {m:_path(quality,m,task) for m in models}
                        if quality['mode']=='per_sample_replay':
                            pairs = [[sample_totals(task,paths[m]['samples'][sample]) for m in models] for sample in samples]
                            terms['Q'] += alpha*math.fsum(min(a for a,b in pp) for pp in pairs)/math.fsum(max(b for a,b in pp) for pp in pairs)
                        # Aggregate quality remains zero: loss interpretation
                        # must not be guessed from an EM/accuracy raw value.
                    except (ValueError,KeyError,TypeError,ZeroDivisionError):
                        pass
            if fixed is not None:
                terms['T']=sum(r['finish_floor_ns']>round(c.horizon_seconds*1e9)+.01 for r in fixed)/len(fixed)
                inevitable_drain |= any(r['finish_floor_ns']>round((c.horizon_seconds+c.drain_seconds)*1e9)+.01 for r in fixed)
                if c.weights.get('Q',0):
                    q,_=predict_quality(c.catalog,deployment,routing,c.workload,trace.arrivals,trace.seed)
                    if q is not None:terms['Q']=q
            floors.append((trace.weight,terms,fixed))
        cost = math.fsum(w*math.fsum(c.weights.get(k,0)*v for k,v in f.items()) for w,f,_ in floors)/math.fsum(w for w,f,_ in floors)
        upper = -cost+1e-8*max(1.,abs(cost))
        return dict(base, upper_bound=upper,safe_pruned=inevitable_drain or incumbent is not None and upper<=incumbent,
            reason='INEVITABLE_SHARED_FIFO_DRAIN_FAILURE' if inevitable_drain else
                   'UPPER_BOUND_DOMINATED' if incumbent is not None and upper<=incumbent else
                   'SHARED_FIFO_MAX_RATE_RELAXATION' if routing is not None else 'STAGE_MAX_RATE_RELAXATION',
            feasibility='INFEASIBLE' if inevitable_drain else 'NOT_DISPROVEN',memory_mib=memory,bound_identity=self.identity,
            trace_floors=[dict(weight=w,terms=f,request_floors=r) for w,f,r in floors])
