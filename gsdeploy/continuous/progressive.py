"""Route-local deployment MILP ledgers with a shared bounded structure pool.

Masters are rebuilt from exact local exclusions on refinement. No SciPy MIP
start or native pause/resume is claimed. Unknown evidence retains its upper
bound even after its identity leaves the master.
"""

import copy
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint
from scipy.sparse import coo_matrix

from ..milp import solve_master
from .activation import deployment_choices
from .contracts import BudgetExhausted, digest
from .deployment_pool import deployment_key
from .predictor_adapter import global_cost_floor
from .safe_bounds import candidate_bound


class InnerSearchTrigger:
    def __init__(self, policy="budget", exploration_period=4):
        if policy not in {"always", "gap", "budget"}:
            raise ValueError("UNKNOWN_INNER_TRIGGER")
        if type(exploration_period) is not int or exploration_period < 1:
            raise ValueError("INVALID_EXPLORATION_PERIOD")
        self.policy, self.period = policy, exploration_period
        self.calls = 0

    def decide(self, *, feasible, certified, upper, incumbent, force=False):
        self.calls += 1
        if certified:
            return False, "CERTIFIED_FIXED_ROUTE"
        if force or self.policy == "always":
            return True, "EXPLICIT_REFINEMENT" if force else "ALWAYS_MILP"
        if not feasible:
            return True, "NO_POOL_INCUMBENT"
        if upper is not None and incumbent is not None and upper <= incumbent:
            return False, "PROVEN_NO_GLOBAL_IMPROVEMENT"
        if self.policy == "gap" and upper is not None:
            return True, "UNRESOLVED_VALID_GAP"
        explore = self.calls % self.period == 0
        return explore, "EXPLORATION_QUOTA" if explore else "HEURISTIC_POOL_ONLY"


class ProgressiveFixedRoutingSolver:
    def __init__(self, evaluator, pool, *, policy="budget", inner_seconds=1.,
                 inner_predictions=16, inner_milp_calls=32, pool_evaluations=8,
                 epsilon=1e-8, exploration_period=4, use_bounds=True):
        if pool.problem is not evaluator.problem:
            raise ValueError("POOL_EVALUATOR_MISMATCH")
        if (not math.isfinite(inner_seconds) or inner_seconds <= 0
                or not math.isfinite(epsilon) or epsilon < 0):
            raise ValueError("INVALID_PROGRESSIVE_LIMIT")
        for n in (inner_predictions, inner_milp_calls, pool_evaluations):
            if type(n) is not int or n < 1:
                raise ValueError("INVALID_PROGRESSIVE_COUNT")
        self.e, self.pool = evaluator, pool
        self.trigger = InnerSearchTrigger(policy, exploration_period)
        self.seconds, self.predictions, self.milp_calls = inner_seconds, inner_predictions, inner_milp_calls
        self.pool_evaluations, self.epsilon, self.use_bounds = pool_evaluations, epsilon, use_bounds
        self.routes, self.history, self.best = {}, [], None
        self.inner_sessions = 0
        self.identity = evaluator.identity

    def _state(self, shares):
        key = digest([self.identity, shares])
        if key not in self.routes:
            routing = self.e.domain.to_qps(shares)
            choices = deployment_choices(self.e.problem, routing)
            floor = global_cost_floor(self.e.oracle, self.e.problem)
            self.routes[key] = dict(shares=copy.deepcopy(shares), routing=routing,
                choices=choices, total=math.prod(len(v) for v in choices.values()),
                records={}, best=None, exhausted=False, closed=False, invalid_bound=False,
                remaining_upper=-floor if floor is not None else None,
                rounds=[], calls=0, inner_sessions=0)
        return self.routes[key]

    def _upper(self, state):
        if state["invalid_bound"]:
            return None
        terms = []
        if not state["exhausted"] and not state["closed"]:
            if state["remaining_upper"] is None:
                return None
            terms.append(state["remaining_upper"])
        for row in state["records"].values():
            if row["state"] == "UNKNOWN":
                if row["upper"] is None:
                    return None
                terms.append(row["upper"])
        if state["best"] is not None:
            terms.append(state["best"]["utility"])
        return max(terms) if terms else None

    def _certified(self, state):
        if state["invalid_bound"] or any(r["state"] == "UNKNOWN" for r in state["records"].values()):
            return False
        if state["exhausted"] or state["closed"]:
            return True
        upper = self._upper(state)
        return (state["best"] is not None and upper is not None
                and upper - state["best"]["utility"] <= self.epsilon)

    def _probe(self, state, deployment, deadline):
        key = deployment_key(self.e.problem, deployment)
        if key in state["records"]:
            self.e.budget.counts["cache_hits"] += 1
            return
        base = global_cost_floor(self.e.oracle, self.e.problem)
        proof = (candidate_bound(self.e, deployment, state["routing"]) if self.use_bounds else
                 dict(upper=-base if base is not None else None, infeasible=False, scope="global_floor"))
        upper = proof["upper"]
        costs = self.e.oracle.state_costs(self.e.problem)
        if costs:
            state_upper = -math.fsum(costs.get((m, deployment.get(m, 0)), 0.) for m in self.e.problem.states)
            upper = min(upper, state_upper) if upper is not None else state_upper
        if proof["infeasible"] or (state["best"] is not None and upper is not None
                                    and upper <= state["best"]["utility"]):
            state["records"][key] = dict(deployment=dict(key), state="PRUNED", upper=upper, proof=proof)
            return
        prediction = self.e.candidate(dict(key), state["routing"], deadline)
        value = prediction["utility"]
        if value is not None and (not math.isfinite(value) or upper is not None and value > upper + 1e-8):
            state["invalid_bound"] = True
            raise ValueError("BOUND_CONTRACT_VIOLATION")
        state["records"][key] = dict(deployment=dict(key), state=prediction["constraint_state"],
                                      utility=value, upper=upper, prediction=prediction)
        if value is not None:
            self.pool.add(dict(key), source="complete_prediction", routing=state["routing"], utility=value)
            if state["best"] is None or value > state["best"]["utility"]:
                state["best"] = copy.deepcopy(prediction)
            if self.best is None or value > self.best["best_feasible_utility"]:
                self.best = dict(deployment=dict(key), routing_shares=copy.deepcopy(state["shares"]),
                    routing_qps=copy.deepcopy(state["routing"]), best_feasible_utility=value,
                    prediction=copy.deepcopy(prediction))

    def _master(self, state, deadline):
        choices = state["choices"]
        states = [(m, n) for m, ns in choices.items() for n in ns]
        indices = {s: i for i, s in enumerate(states)}
        if any(not ns for ns in choices.values()):
            state["exhausted"] = True
            return None
        cost = self.e.oracle.state_costs(self.e.problem)
        memory = self.e.oracle.memory_states(self.e.problem)
        rows, lo, hi = [], [], []
        def add(row, lower=-math.inf, upper=math.inf):
            rows.append(row); lo.append(lower); hi.append(upper)
        for m in choices:
            add({indices[m, n]: 1. for n in choices[m]}, 1., 1.)
        if memory:
            add({i: memory[m, n] for i, (m, n) in enumerate(states)},
                upper=self.e.problem.config.gpu_memory_mib + 1e-6)
        for key in state["records"]:
            x = dict(key)
            add({indices[m, x.get(m, 0)]: 1. for m in choices}, upper=len(choices)-1)
        rr, cc, vv = [], [], []
        for r, row in enumerate(rows):
            for c, value in row.items():
                rr.append(r); cc.append(c); vv.append(value)
        matrix = coo_matrix((vv, (rr, cc)), shape=(len(rows), len(states))).tocsc()
        self.e.budget.consume("milp", deadline)
        started = time.perf_counter()
        try:
            fit = solve_master([cost.get(s, 0.) for s in states], integrality=np.ones(len(states)),
                bounds=Bounds(np.zeros(len(states)), np.ones(len(states))),
                constraints=LinearConstraint(matrix, lo, hi),
                options={"time_limit": max(.001, min(deadline, self.e.budget.deadline)-started), "mip_rel_gap": 0.})
        finally:
            self.e.budget.times["milp"] += time.perf_counter()-started
        state["rounds"].append(dict(status=int(fit.status), rows=len(rows), columns=len(states),
                                    seconds=time.perf_counter()-started))
        if fit.status == 2:
            state["exhausted"] = True
            return None
        if fit.status not in (0, 1):
            raise ValueError("MILP_SOLVER_ERROR")
        dual = getattr(fit, "mip_dual_bound", None)
        if cost and dual is not None and math.isfinite(dual):
            # Only an explicitly declared state-cost floor licenses this dual.
            upper = -float(dual) + 1e-8 * max(1., abs(float(dual)))
            state["remaining_upper"] = (min(state["remaining_upper"], upper)
                                          if state["remaining_upper"] is not None else upper)
        if self._certified(state):
            state["closed"] = True
            return None
        if fit.x is None:
            raise BudgetExhausted("INNER_MILP_TIME_LIMIT")
        selected = [i for i, v in enumerate(fit.x) if v > .5]
        if (any(abs(v-round(v)) > 1e-6 for v in fit.x) or len(selected) != len(choices)
                or {states[i][0] for i in selected} != set(choices)):
            raise ValueError("MILP_INVALID_WITNESS")
        deployment = {states[i][0]: states[i][1] for i in selected if states[i][1]}
        if deployment_key(self.e.problem, deployment) in state["records"]:
            raise ValueError("MILP_REPEATED_EXCLUDED_IDENTITY")
        return deployment

    def evaluate(self, shares, *, force=False, pool_only=False, exact=False):
        e = self.e
        e.check(); self.pool.check()
        if e.identity != self.identity:
            raise ValueError("PROGRESSIVE_IDENTITY_CHANGED")
        shares = e.domain.validate(shares)
        e.budget.consume("outer")
        state = self._state(shares)
        state["calls"] += 1
        started = time.perf_counter()
        before = e.budget.report()
        before_prediction, before_milp = e.budget.times["prediction"], e.budget.times["milp"]
        old_limits = e.budget.limits.copy()
        reason, triggered, trigger_reason = "POOL_ONLY", False, "POOL_REFRESH"
        try:
            deadline = e.budget.deadline if exact else min(e.budget.deadline, started+self.seconds)
            if not exact:
                e.budget.limits["predictions"] = min(old_limits["predictions"], e.budget.counts["predictions"]+self.predictions)
                e.budget.limits["milp"] = min(old_limits["milp"], e.budget.counts["milp"]+self.milp_calls)
            if not self._certified(state):
                for deployment in self.pool.compatible(state["routing"])[:self.pool_evaluations]:
                    e.budget.check(local_deadline=deadline)
                    self._probe(state, deployment, deadline)
                if not pool_only:
                    triggered, trigger_reason = self.trigger.decide(
                        feasible=state["best"] is not None, certified=self._certified(state),
                        upper=self._upper(state),
                        incumbent=self.best["best_feasible_utility"] if self.best else None, force=force or exact)
                if triggered:
                    self.inner_sessions += 1; state["inner_sessions"] += 1
                    while not self._certified(state) and not state["exhausted"]:
                        e.budget.check(local_deadline=deadline)
                        deployment = self._master(state, deadline)
                        if deployment is None:
                            break
                        self._probe(state, deployment, deadline)
                    reason = "FIXED_ROUTE_FINISHED"
                else:
                    reason = trigger_reason if not pool_only else "POOL_REFRESH"
        except BudgetExhausted as exc:
            reason = str(exc)
        except ValueError as exc:
            if str(exc) not in {"BOUND_CONTRACT_VIOLATION", "MILP_SOLVER_ERROR", "MILP_INVALID_WITNESS", "MILP_REPEATED_EXCLUDED_IDENTITY"}:
                raise
            reason = str(exc)
            state["invalid_bound"] = True
        finally:
            e.budget.limits.update(old_limits)
            e.budget.times["outer"] += max(0., time.perf_counter()-started
                -(e.budget.times["prediction"]-before_prediction)-(e.budget.times["milp"]-before_milp))
        upper, certified, best = self._upper(state), self._certified(state), state["best"]
        score = best["utility"] if best else None
        unknown = [copy.deepcopy(r) for r in state["records"].values() if r["state"] == "UNKNOWN"]
        status = ("OPTIMAL_FIXED_ROUTE_FINITE_DOMAIN" if best else "PROVEN_INFEASIBLE") if certified else (
            "SOLVER_ERROR" if state["invalid_bound"] else "INCOMPLETE_EVIDENCE" if state["exhausted"] else "PARTIAL_FIXED_ROUTE")
        row = dict(routing_shares=copy.deepcopy(shares), routing_qps=copy.deepcopy(state["routing"]),
            deployment=copy.deepcopy(best["deployment"]) if best else None,
            best_feasible_utility=score, prediction=copy.deepcopy(best), feasible=best is not None,
            constraint_state="FEASIBLE" if best else "INFEASIBLE" if certified else "UNKNOWN",
            valid_upper_bound=upper, valid_utility_upper_bound=upper,
            optimality_gap=max(0., upper-score) if upper is not None and score is not None else None,
            certified=certified, inner_search_completed=certified, status=status,
            evaluation_kind="certified_fixed_route" if certified else "lower_bound" if best else "unknown",
            termination_reason=reason, trigger_reason=trigger_reason, milp_triggered=triggered,
            pool_version=self.pool.version, inner_sessions=state["inner_sessions"],
            full_prediction_calls=e.budget.counts["predictions"]-before["full_prediction_calls"],
            milp_calls=e.budget.counts["milp"]-before["milp_solve_calls"],
            elapsed_seconds=time.perf_counter()-started, unresolved_candidates=unknown,
            evaluated_deployments=len(state["records"]), deployment_domain_size=state["total"],
            remaining_deployments=0 if certified or state["exhausted"] else max(0, state["total"]-len(state["records"])),
            master_domain_exhausted=state["exhausted"], pool_eviction_affects_certificate=False,
            continuous_optimality=False, native_qualified=False,
            solver_reuse="rebuild_master_with_route_local_ledger", mip_start_used=False)
        self.history.append(copy.deepcopy(row))
        return row
