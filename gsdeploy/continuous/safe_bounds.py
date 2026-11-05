"""Scoped upper bounds: never infer co-resident bounds from standalone data.

Synthetic proof: each FIFO response is at least its assigned service time,
therefore mean and p95 exceed the minimum positive-route service for that task.
T >= 0; M and E are exact fixture resources; Q uses the existing quality scorer.
For robust U=min(U_s), U <= U_2, so the multiplier-2 floor suffices. Weighted
means of per-trace upper bounds bound the same single-deployment objective.
"""

import math

from ..energy import energy_norm
from ..prediction import components
from ..quality import predict_quality

SLACK = 1e-10  # Outward guard; pruning never uses an optimistic tolerance.


def candidate_bound(evaluator, deployment, routing):
    from .predictor_adapter import SyntheticColocationOracle
    e, x = evaluator, deployment
    c, oracle = e.problem.config, e.oracle
    from .. import normalization as norm
    if norm.is_v2(c.catalog):
        from .predictor_adapter import global_cost_floor
        floor = global_cost_floor(oracle, e.problem)
        return dict(upper=-floor+SLACK if floor is not None else None,
                    infeasible=False, scope="reference_v2_global_quality_floor")
    from .immutable import SessionPredictorOracle
    if type(oracle) is SessionPredictorOracle and c.catalog.get('performance_mode')=='paper_profiled_fifo_v1':
        from .paper_bounds import PaperDeploymentBounds
        if not hasattr(e,'paper_route_bounds'):
            e.paper_route_bounds=PaperDeploymentBounds(e.problem,oracle)
        proof=e.paper_route_bounds.evaluate(deployment,routing=routing)
        return dict(upper=proof['upper_bound'],infeasible=proof['feasibility']=='INFEASIBLE',
                    scope='declared_stage_predictor_shared_fifo',proof=proof)
    base = oracle.cost_lower_bound
    upper = -base if base is not None else None
    memory = oracle.memory_states(e.problem)
    # The existing capability explicitly declares these additive tables safe.
    if memory and math.fsum(memory[m, x.get(m, 0)] for m in e.problem.states) > c.gpu_memory_mib + 1e-6:
        return dict(upper=None, infeasible=True, scope="declared_additive_memory")
    costs = oracle.state_costs(e.problem)
    if base is not None and costs:
        upper = -max(base, math.fsum(costs.get((m, x.get(m, 0)), 0) for m in e.problem.states))
    # Exact type prevents subclasses overriding prediction from inheriting proof.
    if type(oracle) is not SyntheticColocationOracle:
        return dict(upper=upper + SLACK if upper is not None else None,
                    infeasible=False, scope="declared_global_and_state_floor")
    workers, pressure, slowdown, service, mem = oracle.resource_values(e.problem, x, routing)
    multiplier = 2.0 if c.search_policy == "robust" else 1.0
    rho = max((math.fsum(routing[t].get(m, 0) * service[m].get(t, 0) / 1000 * multiplier
                         for t in c.workload) / c.catalog["capacity_factors"][m][str(n)]
               for m, n in x.items()), default=0)
    if mem > c.gpu_memory_mib + 1e-6 or rho > .8 + 1e-8:
        return dict(upper=None, infeasible=True, scope="synthetic_exact_memory_rho_v1")
    task_stats = {}
    for t in c.workload:
        minimum = min(service[m][t] * x[m] / c.catalog["capacity_factors"][m][str(x[m])] * multiplier
                      for m, q in routing[t].items() if q > 0)
        task_stats[t] = dict(mean_ms=minimum, p95_ms=minimum)
    p = oracle.parameters
    watts = p["idle_watts"] + p["worker_watts"] * workers + p["busy_watts"] * pressure * slowdown * multiplier
    energy = energy_norm(dict(scope="whole_gpu_board", kind="synthetic_fixture", coverage_valid=True,
                              run_joules=watts*c.horizon_seconds, startup_joules=2.0*workers),
                         c.horizon_seconds, c.energy_reference_watts, c.startup_amortization_seconds)
    values = []
    for trace in e.problem.train_traces:
        quality, _ = predict_quality(c.catalog, x, routing, c.workload, trace.arrivals, trace.seed)
        # Missing quality has only its nonnegative floor; no invented evidence.
        value = components(task_stats, c.workload, c.latency_reference_ms, 0.0,
                           mem/c.gpu_memory_mib, c.weights, energy, quality if quality is not None else 0.0)["U"]
        values.append((trace.weight, value))
    value = math.fsum(w*v for w, v in values)/math.fsum(w for w, _ in values)
    return dict(upper=value + SLACK, infeasible=False, scope="synthetic_queue_free_six_loss_v1")
