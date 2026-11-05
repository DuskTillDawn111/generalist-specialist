"""Bound-ordered finite deployment search with candidate-only warm starts."""

import copy
import math
import time

from .predictor_adapter import global_cost_floor

from .activation import deployment_choices, deployments
from .contracts import BudgetExhausted, digest
from .fixed_routing_enumeration import result
from .safe_bounds import candidate_bound


class FixedRoutingBoundedOptimizer:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def solve(self, routing, deadline=None):
        e = self.evaluator
        choices = deployment_choices(e.problem, routing)
        total = math.prod(map(len, choices.values()))
        best, ledger, reason, completed = None, [], None, False
        pending, bound_seconds = [], 0.0
        warm = e.deployment_hint if e.inner_warm_start else None
        warm_key = digest(warm) if warm is not None else None
        warm_evaluated = 0
        try:
            for x in deployments(e.problem, routing):
                e.budget.check(local_deadline=deadline)
                start = time.perf_counter()
                b = (candidate_bound(e, x, routing) if e.inner_bounds else
                     dict(upper=None, infeasible=False, scope="bounds_disabled"))
                bound_seconds += time.perf_counter()-start
                if b["upper"] is not None and not math.isfinite(b["upper"]):
                    reason = "BOUND_CONTRACT_VIOLATION"
                    break
                if b["infeasible"]:
                    ledger.append(dict(deployment=x, state="PRUNED", reason="PROVEN_CONSTRAINT_BOUND", bound=b))
                else:
                    pending.append((x, b))
            if reason is None:
                # Bound-only ordering first, or exhaustive product order without bounds.
                if e.inner_bounds:
                    pending.sort(key=lambda row: -(row[1]["upper"] if row[1]["upper"] is not None else math.inf))
                if warm_key is not None:
                    pending.sort(key=lambda row: digest(row[0]) != warm_key)
                for x, b in pending:
                    e.budget.check(local_deadline=deadline)
                    if best is not None and b["upper"] is not None and b["upper"] <= best["utility"]:
                        ledger.append(dict(deployment=x, state="PRUNED", reason="PROVEN_UTILITY_BOUND", bound=b))
                        continue
                    p = e.candidate(x, routing, deadline)
                    warm_evaluated += int(warm_key is not None and digest(x) == warm_key)
                    ledger.append(dict(deployment=x, state=p["constraint_state"], utility=p["utility"], bound=b))
                    if p["utility"] is not None:
                        if b["upper"] is not None and p["utility"] > b["upper"]:
                            reason = "BOUND_CONTRACT_VIOLATION"
                            break
                        if best is None or p["utility"] > best["utility"]:
                            best = p
                else:
                    completed = True
        except BudgetExhausted as exc:
            reason = str(exc)
        # Never reuse a utility, a numerical bound, a prediction, or a cut.
        if best is not None and reason != "BOUND_CONTRACT_VIOLATION":
            e.deployment_hint = copy.deepcopy(best["deployment"])
        # A global bound remains valid even if bound construction was interrupted.
        floor = global_cost_floor(e.oracle, e.problem)
        upper = -floor if floor is not None else None
        if reason == "BOUND_CONTRACT_VIOLATION":
            upper = None
        out = result(e, best, ledger, completed, reason, upper, total)
        out.update(bound_seconds=bound_seconds, bounds_enabled=e.inner_bounds,
                   warm_start_enabled=e.inner_warm_start,
                   warm_start_evaluated_deployments=warm_evaluated,
                   warm_start_candidate=copy.deepcopy(warm),
                   feedback_kind="current_route_bounds_and_candidate_only_hint")
        return out
