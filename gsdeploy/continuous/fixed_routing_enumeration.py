"""Independent exhaustive reference over the fixed routing's deployment domain."""

import math

from .predictor_adapter import global_cost_floor

from .activation import deployment_choices, deployments
from .contracts import BudgetExhausted


def result(evaluator, best, ledger, completed, reason, upper, total):
    unknown = [r for r in ledger if r["state"] == "UNKNOWN"]
    score = best["utility"] if best else None
    if upper is not None and score is not None:
        upper = max(upper, score)
    certified = completed and not unknown
    if certified:
        status = "OPTIMAL_FIXED_ROUTE_FINITE_DOMAIN" if best else "PROVEN_INFEASIBLE"
        upper = score if best else None
    elif reason:
        status = ("BUDGET_LIMIT_WITH_INCUMBENT" if best else "BUDGET_LIMIT_NO_INCUMBENT")
        if reason in {"SOLVER_ERROR", "MILP_INVALID_WITNESS", "BOUND_CONTRACT_VIOLATION"}:
            status = "SOLVER_ERROR"
    else:
        status = "INCOMPLETE_EVIDENCE"
    return dict(deployment=best["deployment"] if best else None,
        best_feasible_utility=score, prediction=best,
        valid_utility_upper_bound=upper,
        optimality_gap=max(0.0, upper-score) if upper is not None and score is not None else None,
        feasible=best is not None,
        constraint_state="FEASIBLE" if best else "INFEASIBLE" if certified else "UNKNOWN",
        certified=certified, inner_search_completed=completed, status=status,
        termination_reason=reason, certificate_scope="fixed routing, declared finite deployment domain and frozen oracle",
        unresolved_candidates=unknown, ledger=ledger,
        deployment_domain_size=total,
        evaluated_deployments=sum(r["state"] != "PRUNED" for r in ledger),
        safely_pruned_deployments=sum(r["state"] == "PRUNED" for r in ledger),
        remaining_deployments=max(0, total-len(ledger)),
        constraint_violations=[r for r in ledger if r["state"] == "INFEASIBLE"],
        outer_continuous_optimality=False)


class FixedRoutingEnumerationOptimizer:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def solve(self, routing, deadline=None):
        e = self.evaluator
        choices = deployment_choices(e.problem, routing)
        total = math.prod(map(len, choices.values()))
        best, ledger, reason, completed = None, [], None, False
        memory = e.oracle.memory_states(e.problem)
        try:
            for x in deployments(e.problem, routing):
                e.budget.check(local_deadline=deadline)
                if memory and math.fsum(memory[m, x.get(m, 0)] for m in choices) > e.problem.config.gpu_memory_mib + 1e-6:
                    ledger.append(dict(deployment=x, state="PRUNED", reason="PROVEN_MEMORY_BOUND"))
                    continue
                p = e.candidate(x, routing, deadline)
                ledger.append(dict(deployment=x, state=p["constraint_state"], utility=p["utility"]))
                if p["utility"] is not None and (best is None or p["utility"] > best["utility"]):
                    best = p
            completed = True
        except BudgetExhausted as exc:
            reason = str(exc)
        lower = global_cost_floor(e.oracle, e.problem)
        upper = -lower if lower is not None else None
        return result(e, best, ledger, completed, reason, upper, total)
