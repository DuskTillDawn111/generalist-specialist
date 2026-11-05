"""Deployment-only MILP with exact identity exclusions and conservative bounds.

The master orders unvisited identities by an explicitly supplied safe floor.
Every evaluated identity is removed and retained in the ledger. This avoids
large-M numerical identity feedback: convergence is exact finite-domain
bookkeeping, not a claim that the master equals the nonlinear predictor.
"""

import math
import time

import numpy as np
from .predictor_adapter import global_cost_floor

from scipy.optimize import Bounds, LinearConstraint
from scipy.sparse import coo_matrix

from ..milp import solve_master
from .activation import deployment_choices
from .contracts import BudgetExhausted
from .fixed_routing_enumeration import result


class FixedRoutingMilpOptimizer:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def solve(self, routing, deadline=None):
        e = self.evaluator
        choices = deployment_choices(e.problem, routing)
        total = math.prod(map(len, choices.values()))
        states = [(m, n) for m, ns in choices.items() for n in ns]
        rounds, bound_seconds, capability = [], 0.0, None
        from .immutable import SessionPredictorOracle
        paper_bounds=(type(e.oracle) is SessionPredictorOracle and e.inner=='milp_bounds' and
                      e.problem.config.catalog.get('performance_mode')=='paper_profiled_fifo_v1')
        cost = e.oracle.state_costs(e.problem)
        memory = e.oracle.memory_states(e.problem)
        base_floor = global_cost_floor(e.oracle, e.problem)
        # No general U upper bound without a valid global cost floor.
        floor_known = base_floor is not None
        indices = {s: i for i, s in enumerate(states)}
        objective = [cost.get(s, 0.0) for s in states]
        rows, low, high = [], [], []

        def add(row, lo=-math.inf, hi=math.inf):
            rows.append(row)
            low.append(lo)
            high.append(hi)

        for m in choices:
            add({indices[m, n]: 1.0 for n in choices[m]}, 1.0, 1.0)
        if memory:
            add({i: memory[m, n] for i, (m, n) in enumerate(states)},
                hi=e.problem.config.gpu_memory_mib + 1e-6)
        best, ledger, reason = None, [], None
        completed, exhausted = False, False
        seen = set()
        unknown_floors = []
        remaining_floor = base_floor
        try:
            if e.inner == "milp_bounds":
                from .validated_bounds import ValidatedRouteBounds
                start = time.perf_counter()
                try:
                    if not hasattr(e, "validated_route_bounds"):
                        e.validated_route_bounds = ValidatedRouteBounds(e)
                    capability = e.validated_route_bounds
                    if capability.available:
                        capability.extend(states, objective, add)
                        capability.columns(routing, choices, indices, objective, add, deadline)
                finally:
                    bound_seconds = time.perf_counter()-start
            strengthened = capability is not None and capability.available
            objective = np.array(objective)
            blocks = len(choices)+(len(routing) if strengthened else 0)
            while True:
                e.budget.check("milp", deadline)
                rr, cc, vv = [], [], []
                for i, row in enumerate(rows):
                    for j, v in row.items():
                        rr.append(i)
                        cc.append(j)
                        vv.append(v)
                matrix = coo_matrix((vv, (rr, cc)), shape=(len(rows), len(objective))).tocsc()
                remaining = min(e.budget.deadline, deadline or e.budget.deadline) - time.perf_counter()
                if remaining <= 0:
                    raise BudgetExhausted("TIME_LIMIT")
                e.budget.consume("milp", deadline)
                start = time.perf_counter()
                fit = solve_master(objective, integrality=np.ones(len(objective)),
                    bounds=Bounds(np.zeros(len(objective)), np.ones(len(objective))),
                    constraints=LinearConstraint(matrix, low, high),
                    options={"time_limit": remaining, "mip_rel_gap": 0.0})
                e.budget.times["milp"] += time.perf_counter() - start
                rounds.append(dict(status=int(fit.status), seconds=time.perf_counter()-start,
                                   rows=len(rows), columns=len(objective)))
                if fit.status == 2:
                    completed, exhausted, remaining_floor = True, True, None
                    break
                if fit.status not in (0, 1):
                    reason = "SOLVER_ERROR"
                    break
                dual = getattr(fit, "mip_dual_bound", None)
                if floor_known and (cost or strengthened) and dual is not None and math.isfinite(dual):
                    from .validated_bounds import guard
                    remaining_floor = max(base_floor, float(dual)-(guard(dual) if strengthened else 0.0))
                upper = -remaining_floor if floor_known and remaining_floor is not None else None
                if best is not None and upper is not None and upper <= best["utility"]:
                    # Unknown removed identities still prevent certification in result().
                    completed = True
                    break
                if fit.x is None:
                    reason = "MILP_TIME_LIMIT"
                    break
                if any(abs(v-round(v)) > 1e-6 for v in fit.x):
                    reason = "MILP_INVALID_WITNESS"
                    break
                selected = tuple(i for i, v in enumerate(fit.x) if v > 0.5)
                selected_states = tuple(i for i in selected if i < len(states))
                if (len(selected) != blocks or len(selected_states) != len(choices) or selected in seen
                        or {states[i][0] for i in selected_states} != set(choices)):
                    reason = "MILP_INVALID_WITNESS"
                    break
                seen.add(selected)
                x = {m: n for i in selected_states for m, n in [states[i]] if n}
                rounds[-1]['deployment']=x
                if paper_bounds:
                    from .safe_bounds import candidate_bound
                    started=time.perf_counter()
                    proof=candidate_bound(e,x,routing)
                    bound_seconds+=time.perf_counter()-started
                    rounds[-1]['route_bound']=proof
                    if proof['infeasible'] or (best is not None and proof['upper'] is not None and proof['upper']<=best['utility']):
                        ledger.append(dict(deployment=x,state='PRUNED',reason='PROVEN_SHARED_FIFO_BOUND',bound=proof))
                        rounds[-1].update(full_predictions=0,pruned=True)
                        add({i:1. for i in selected},hi=blocks-1)
                        if fit.status==1:
                            reason='MILP_TIME_LIMIT';break
                        continue
                before_predictions=e.budget.counts['predictions']
                p = e.candidate(x, routing, deadline)
                rounds[-1].update(full_predictions=e.budget.counts['predictions']-before_predictions,pruned=False)
                candidate_floor = (max(base_floor, math.fsum(objective[i] for i in selected))
                                   if floor_known and (cost or strengthened) else base_floor)
                if p["utility"] is not None:
                    if floor_known and -p["utility"] < candidate_floor - 1e-8:
                        reason = "BOUND_CONTRACT_VIOLATION"
                        break
                    if best is None or p["utility"] > best["utility"]:
                        best = p
                if p["constraint_state"] == "UNKNOWN":
                    if floor_known:
                        unknown_floors.append(candidate_floor)
                ledger.append(dict(deployment=x, state=p["constraint_state"],
                                   utility=p["utility"], cost_floor=candidate_floor))
                # Exact identity feedback, local to this fixed route only.
                add({i: 1.0 for i in selected}, hi=blocks-1)
                if fit.status == 1:
                    reason = "MILP_TIME_LIMIT"
                    break
        except BudgetExhausted as exc:
            reason = str(exc)
        except ValueError as exc:
            if str(exc) != "BOUND_CONTRACT_VIOLATION":
                raise
            reason = str(exc)
        bounds = [-v for v in unknown_floors] if floor_known else []
        if floor_known and remaining_floor is not None:
            bounds.append(-remaining_floor)
        if best:
            bounds.append(best["utility"])
        upper = max(bounds) if floor_known and bounds else None
        if reason == "BOUND_CONTRACT_VIOLATION":
            upper = None
        output = result(e, best, ledger, completed, reason, upper, total)
        if completed:
            # Any unvisited identities are excluded by the memory row or a
            # global objective bound. Record their count without materializing.
            output["safely_pruned_deployments"] = total-len(ledger)+sum(r['state']=='PRUNED' for r in ledger)
            output["remaining_deployments"] = 0
        output["master_domain_exhausted"] = exhausted
        output["feedback_kind"] = "fixed_route_complete_identity_no_good"
        output.update(validated_bounds_enabled=bool(capability and capability.available),
                      shared_fifo_bounds_enabled=paper_bounds,
                      bound_seconds=bound_seconds, milp_rounds=rounds,
                      bound_scope='declared_stage_predictor_shared_fifo' if paper_bounds else
                                  "validated_frozen_predictor" if capability and capability.available else "legacy_declared_floor")
        return output
