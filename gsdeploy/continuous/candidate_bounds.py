"""Route-independent deployment relaxations for validated frozen predictors.

Every request may independently choose its fastest resident executor and best
quality numerator/denominator, with no queueing. This is deliberately looser
than any realizable Router policy. Fixed-route Joint floors are never reused.
"""

import math

import numpy as np

from ..calibration import _deployment_key
from ..quality import _document, _path, sample_totals
from ..surrogate_bounds import surrogate_bounds
from .contracts import digest
from .immutable import ImmutableContinuousProblem, SessionPredictorOracle


def guard(value):
    return 1e-8 * max(1., abs(value))


class ContinuousDeploymentBounds:
    """Bounds are about this predictor only, never physical GPU guarantees."""

    def __init__(self, problem, oracle):
        self.problem = problem
        self.available = False
        self.reason = 'UNSUPPORTED_ORACLE_OR_PREDICTOR'
        self.bounds = None
        if type(problem) is not ImmutableContinuousProblem or type(oracle) is not SessionPredictorOracle:
            return
        if oracle.problem is not problem:
            raise ValueError('BOUND_ORACLE_PROBLEM_MISMATCH')
        oracle.check()
        c = problem.config
        self.paper = None
        from ..normalization import is_v2
        if is_v2(c.catalog):
            self.reason = 'REFERENCE_V2_LEGACY_RELAXATION_DISABLED'
            return
        if c.catalog.get('performance_mode') == 'paper_profiled_fifo_v1':
            from .paper_bounds import PaperDeploymentBounds
            self.paper = PaperDeploymentBounds(problem, oracle)
            self.available = True
            self.reason = 'DECLARED_STAGE_PREDICTOR_ONLY'
            return
        if c.catalog.get('performance_mode') not in {'calibrated_surrogate', 'calibrated_replay_v2'}:
            return
        self.bounds = surrogate_bounds(c.catalog, c.workload, horizon=c.horizon_seconds,
            energy_reference_watts=c.energy_reference_watts,
            startup_amortization_seconds=c.startup_amortization_seconds)
        if self.bounds is None:
            self.reason = 'PREDICTOR_SCOPE_GATE_FAILED'
            return
        b = self.bounds
        if not all(math.isfinite(v) and v >= 0 for v in
                   (b.intercept, b.busy, b.worker, b.spec['startup_joules_per_worker'])):
            self.reason = 'NONNEGATIVE_ENERGY_PROOF_UNAVAILABLE'
            return
        self.identity = digest([problem.identity, oracle.version, 'independent-request-relaxation-v1'])
        self.available = True
        self.reason = 'VALIDATED_FROZEN_PREDICTOR_ONLY'

    def evaluate(self, deployment, incumbent=None, *, deadline=None):
        p = self.problem
        c = p.config
        base = dict(deployment=dict(deployment), upper_bound=None, safe_pruned=False,
                    feasibility='UNKNOWN', reason=None,
                    scope='all continuous routes for this deployment under the frozen predictor',
                    physical_bound=False, fixed_route_feedback_reused=False)
        if not deployment or any(m not in p.states or type(n) is not int or n <= 0
                                 or n not in p.states[m] for m,n in deployment.items()):
            return dict(base, safe_pruned=True, feasibility='INFEASIBLE', reason='ILLEGAL_REPLICA_STATE')
        edges = {t:[m for m in deployment if t in c.catalog['models'][m]['supported_tasks']]
                 for t in c.workload}
        if any(not ms for ms in edges.values()):
            return dict(base, safe_pruned=True, feasibility='INFEASIBLE', reason='UNCOVERED_TASK')
        if not self.available:
            return dict(base, reason=self.reason)
        if self.paper is not None:
            return self.paper.evaluate(deployment, incumbent, deadline=deadline)
        try:
            b = self.bounds
            memory = max(b.memory_multiplier*c.catalog['safe_factor']*math.fsum(
                c.catalog['memory_table_mib'][m][str(n)] for m,n in deployment.items()),
                b.spec['training_deployment_memory'].get(_deployment_key(deployment), 0.))
            if memory > c.gpu_memory_mib+1e-6+guard(c.gpu_memory_mib):
                return dict(base, safe_pruned=True, feasibility='INFEASIBLE', reason='RESIDENT_MEMORY',
                            memory_mib=memory)
            energy = b.constant_energy+b.state_energy(sum(deployment.values()))
            for task,ms in edges.items():
                # Every row is a simplex. A linear busy-energy term is at
                # least its cheapest resident path, including idle residents.
                services=[b.services[m,task,deployment[m]] for m in ms]
                energy += b.busy*c.workload[task]*min(services)/1000/b.reference
            try:
                quality = _document(c.catalog)
            except (KeyError, TypeError, ValueError):
                quality = None  # Zero relaxation for all valid nonnegative losses.
            trace_floors=[];inevitable_late=False
            for trace in p.train_traces:
                ids = b.task_input_ids(trace.arrivals, trace.seed)
                floors=dict(Q=0.,T=0.,L=0.,P=0.,M=memory/c.gpu_memory_mib,E=energy)
                for task,ms in edges.items():
                    offsets=[a for a,t in trace.arrivals if t==task]
                    samples=ids[task] if ids is not None else [None]*len(offsets)
                    times=[min(b.request_service(m,task,deployment[m],sample)*deployment[m]
                               /c.catalog['capacity_factors'][m][str(deployment[m])] for m in ms)
                           for sample in samples]
                    if not times or any(not math.isfinite(v) or v < 0 for v in times):
                        raise ValueError('INVALID_REQUEST_SERVICE_RELAXATION')
                    # A zero for a missing path is a relaxation, not a reason
                    # to remove that executor from the continuous domain.
                    alpha=c.workload[task]/math.fsum(c.workload.values())
                    floors['L']+=alpha*float(np.mean(times))/c.latency_reference_ms[task]
                    floors['P']+=alpha*float(np.percentile(times,95))/c.latency_reference_ms[task]
                    floors['T']+=sum(a+v/1000 > c.horizon_seconds+guard(c.horizon_seconds)
                                     for a,v in zip(offsets,times))/len(trace.arrivals)
                    inevitable_late |= any(a+v/1000 > c.horizon_seconds+c.drain_seconds
                                           +guard(c.horizon_seconds+c.drain_seconds)
                                           for a,v in zip(offsets,times))
                    if c.weights['Q']>0 and quality is not None:
                        try:
                            paths={m:_path(quality,m,task) for m in ms}
                            if quality['mode']=='per_sample_replay' and ids is not None:
                                pairs=[[sample_totals(task,paths[m]['samples'][sample]) for m in ms]
                                       for sample in samples]
                                q=math.fsum(min(a for a,_ in pp) for pp in pairs)/math.fsum(
                                    max(z for _,z in pp) for pp in pairs)
                                floors['Q']+=alpha*q
                            # Surrogate fixtures without per-request input
                            # identities retain the conservative zero Q floor.
                        except (KeyError, TypeError, ValueError, ZeroDivisionError):
                            pass
                trace_floors.append((trace.weight,floors))
            cost=math.fsum(w*math.fsum(c.weights.get(k,0)*v for k,v in f.items())
                           for w,f in trace_floors)/math.fsum(w for w,_ in trace_floors)
            if not math.isfinite(cost) or cost<0:
                raise ValueError('INVALID_NONNEGATIVE_COST_FLOOR')
            # All stress multipliers >= 1; nominal relaxation also bounds
            # the robust min-scenario utility from above.
            upper=-cost+guard(cost)
            result=dict(base, upper_bound=upper, memory_mib=memory, reason='REQUEST_RELAXATION',
                        trace_floors=[dict(weight=w,terms=f) for w,f in trace_floors],
                        feasibility='NOT_DISPROVEN', bound_identity=self.identity)
            if inevitable_late:
                result.update(safe_pruned=True,feasibility='INFEASIBLE',reason='INEVITABLE_DRAIN_FAILURE')
            elif incumbent is not None and upper<=incumbent:
                result.update(safe_pruned=True,reason='UPPER_BOUND_DOMINATED')
            return result
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            return dict(base, reason='BOUND_UNKNOWN:'+str(exc))
