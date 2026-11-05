"""One evaluation/caching and aggregation path for every new solver."""

import copy
import math
import time

from .activation import active_models
from .contracts import Budget, BudgetExhausted, digest
from .route_domain import ContinuousRouteDomain
from .predictor_adapter import assignment_digest, normalize_prediction, validate_prediction_trace


class ContinuousRoutingEvaluator:
    def __init__(self, problem, oracle, inner="enumeration", budget=None,
                 cache=True, inner_seconds=None, inner_bounds=True, inner_warm_start=False):
        self.problem, self.oracle = problem, oracle
        from .immutable import SessionPredictorOracle
        if type(oracle) is SessionPredictorOracle and oracle.problem is not problem:
            raise ValueError("ORACLE_PROBLEM_MISMATCH")
        self.domain = ContinuousRouteDomain(problem)
        self.budget = budget if budget is not None else Budget()
        if inner not in {"enumeration", "milp", "milp_bounds", "bounded"}:
            raise ValueError("UNKNOWN_INNER_SOLVER")
        if inner_seconds is not None and (not math.isfinite(inner_seconds) or inner_seconds <= 0):
            raise ValueError("INVALID_INNER_SECONDS")
        self.inner, self.inner_seconds, self.cache_enabled = inner, inner_seconds, cache
        self.inner_bounds, self.inner_warm_start = inner_bounds, inner_warm_start
        self.deployment_hint = None
        self.prediction_cache, self.routing_cache, self.history = {}, {}, []
        self.identity = digest(dict(problem=problem.identity, oracle=oracle.version,
                                    objective=problem.objective_id))

    def check(self):
        self.problem.check()
        self.oracle.check()
        if self.identity != digest(dict(problem=self.problem.identity,
                oracle=self.oracle.version, objective=self.problem.objective_id)):
            raise ValueError("EVALUATOR_IDENTITY_CHANGED")

    def candidate(self, deployment, routing, deadline=None, traces=None):
        self.check()
        c = self.problem.config
        # The public fixed-candidate/heldout path obeys the same structure
        # contract as outer evaluations, including QPS conservation.
        self.domain.validate({t: {m: q / c.workload[t] for m, q in row.items()}
                              for t, row in routing.items() if t in c.workload})
        if set(routing) != set(c.workload):
            raise ValueError("ROUTING_TASK_MISMATCH")
        if (any(m not in deployment for m in active_models(routing)) or
                any(m not in self.problem.states or type(n) is not int
                    or n <= 0 or n not in self.problem.states[m]
                    for m, n in deployment.items())):
            raise ValueError("INVALID_FIXED_ROUTE_DEPLOYMENT")
        traces = tuple(traces if traces is not None else self.problem.train_traces)
        if not traces or any(t.identity not in {v.identity for v in self.problem.traces} for t in traces):
            raise ValueError("UNDECLARED_TRACE")
        records = []
        for trace in traces:
            self.budget.check(local_deadline=deadline)
            key = digest([self.identity, deployment, routing, trace.identity])
            if self.cache_enabled and key in self.prediction_cache:
                p = copy.deepcopy(self.prediction_cache[key])
                validate_prediction_trace(p, trace)
                self.budget.counts["cache_hits"] += 1
            else:
                start = time.perf_counter()
                try:
                    p = self.oracle.predict(self.problem, deployment, routing,
                                            trace, self.budget, deadline)
                finally:
                    self.budget.times["prediction"] += time.perf_counter() - start
                self.check()
                validate_prediction_trace(p, trace)
                if self.cache_enabled:
                    self.prediction_cache[key] = copy.deepcopy(p)
            state, value = normalize_prediction(p, c.search_policy)
            if c.search_policy == "robust" and not self.oracle.supports_stress:
                state, value = "UNKNOWN", None
            records.append(dict(trace=trace.name, trace_identity=trace.identity,
                weight=trace.weight, constraint_state=state, utility=value,
                assignment_sha256=assignment_digest(deployment, routing, trace), prediction=p))
        states = [r["constraint_state"] for r in records]
        state = ("INFEASIBLE" if "INFEASIBLE" in states else
                 "UNKNOWN" if "UNKNOWN" in states else "FEASIBLE")
        utility = (math.fsum(r["weight"] * r["utility"] for r in records)
                   / math.fsum(r["weight"] for r in records) if state == "FEASIBLE" else None)
        return dict(deployment=copy.deepcopy(deployment), constraint_state=state,
                    utility=utility, traces=records, evidence_kind=self.oracle.evidence_kind,
                    native_qualified=False)

    def evaluate(self, routing_vector, evaluation_budget=None, *, face=None):
        self.check()
        shares = (self.domain.validate(routing_vector) if isinstance(routing_vector, dict)
                  else self.domain.decode(routing_vector, face))
        routing = self.domain.to_qps(shares)
        self.budget.consume("outer")
        start = time.perf_counter()
        before = self.budget.report()
        key = digest([self.identity, shares, routing, self.inner, self.inner_bounds, self.inner_warm_start])
        cached = self.cache_enabled and key in self.routing_cache
        if cached:
            result = copy.deepcopy(self.routing_cache[key])
            self.budget.counts["cache_hits"] += 1
        else:
            from .fixed_routing_enumeration import FixedRoutingEnumerationOptimizer
            from .fixed_routing_milp import FixedRoutingMilpOptimizer
            from .fixed_routing_bounded import FixedRoutingBoundedOptimizer
            limit = self.inner_seconds if evaluation_budget is None else evaluation_budget
            if limit is not None and (not math.isfinite(limit) or limit <= 0):
                raise ValueError("INVALID_EVALUATION_BUDGET")
            deadline = min(self.budget.deadline, start + limit) if limit is not None else self.budget.deadline
            solver = {"enumeration": FixedRoutingEnumerationOptimizer,
                      "milp": FixedRoutingMilpOptimizer,
                      "milp_bounds": FixedRoutingMilpOptimizer,
                      "bounded": FixedRoutingBoundedOptimizer}[self.inner](self)
            result = solver.solve(routing, deadline)
            if self.cache_enabled and result["inner_search_completed"]:
                self.routing_cache[key] = copy.deepcopy(result)
        after = self.budget.report()
        result.update(routing_shares=shares, routing_qps=routing,
            activation=list(active_models(routing)), problem_sha256=self.problem.identity,
            oracle_version=self.oracle.version, objective_id=self.problem.objective_id,
            trace_set_sha256=digest([t.identity for t in self.problem.train_traces]),
            evaluation_seconds=time.perf_counter() - start, cached=cached,
            full_prediction_calls=after["full_prediction_calls"]-before["full_prediction_calls"],
            predictor_leaf_calls=after["predictor_leaf_calls"]-before["predictor_leaf_calls"],
            milp_solve_calls=after["milp_solve_calls"]-before["milp_solve_calls"],
            evidence_kind=self.oracle.evidence_kind,
            outer_continuous_optimality=False)
        self.history.append(copy.deepcopy(result))
        return result
