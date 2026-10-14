"""Joint finite-grid MILP with valid lower bounds and exact predictor feedback.

One-hot measured replica states and task route/count columns form the master.
Its epigraph lower-bounds weighted memory plus actual routed service mean/P95.
Whole-plan FIFO costs yield identity cuts; invalid plans yield no-good cuts.
No convexity assumption or local proxy optimality is transferred to full U.
"""

import itertools
import math
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from .prediction import predict_scenarios
from .performance import service_lower_ms
from .routing import Router
from .trace import expand_routes, make_trace, trace_digest


def compositions(total, size):
    if size == 1:
        yield (total,)
    else:
        for first in range(total + 1):
            for tail in compositions(total - first, size - 1):
                yield (first,) + tail


class JointMilpOptimizer:
    def __init__(self, data, rates, budget, refs, weights, **context):
        self.data, self.rates, self.budget, self.refs, self.weights = (
            data,
            dict(sorted(rates.items())),
            budget,
            refs,
            weights,
        )
        self.policy = context.pop("search_policy", "nominal")
        context.pop("search_max_states", None)
        self.iteration_limit = context.pop("milp_max_iterations", 1000)
        self.time_limit = context.pop("milp_time_limit_seconds", 30.0)
        self.column_limit = context.pop("milp_max_columns", 50000)
        self.tolerance = context.pop("milp_absolute_gap", 1e-8)
        self.use_witness = context.pop("milp_capacity_witness", True)
        self.context = context
        self.trace = make_trace(
            self.rates,
            context.get("horizon", 120.0),
            context.get("seed", 42),
            context.get("arrival_process", "deterministic_periodic"),
        )
        self.counts = {t: sum(k == t for _, k in self.trace) for t in self.rates}
        self.best = self.diagnostic = None
        self.unknown_lower = math.inf
        self.lower = 0.0
        self.history = []
        self.pattern_cache = {}
        self.witness = None
        self.metrics = dict(
            milp_calls=0,
            task_columns=0,
            full_prediction_calls=0,
            feedback_cuts=0,
            route_feedback_cuts=0,
            infeasible_cuts=0,
            unknown_candidates=0,
            capacity_probe_calls=0,
            service_pattern_evaluations=0,
        )
        self.diagnostic_mode = "E" in weights and not data.get("energy_profiles")

    def expired(self):
        return time.monotonic() - self.start >= self.time_limit

    def floor(self, task, x, row):
        if self.data.get("performance_mode", "independent") == "profiled":
            return 0.0
        stress = 2.0 if self.policy == "robust" else 1.0
        alpha = self.rates[task] / math.fsum(self.rates.values()) / self.refs[task]
        aliases, routes = expand_routes(x, {task: row})
        router = Router(routes)
        order = list(router.weights[task])
        key = (self.counts[task], tuple(router.weights[task].values()))
        if key not in self.pattern_cache:
            indices = {a: i for i, a in enumerate(order)}
            counts = np.zeros(len(order), dtype=int)
            for i in range(self.counts[task]):
                if i % 256 == 0 and self.expired():
                    return None
                counts[indices[router.choose(task)]] += 1
            self.pattern_cache[key] = counts
            self.metrics["service_pattern_evaluations"] += 1
        counts = self.pattern_cache[key]
        values = np.array(
            [
                stress
                * service_lower_ms(self.data, aliases[a], task)
                * x[aliases[a]]
                / self.data["capacity_factors"][aliases[a]][str(x[aliases[a]])]
                for a in order
            ]
        )
        # Exact empirical moments/linear percentile of the service multiset,
        # without rebuilding N samples for each model-name/service variant.
        permutation = np.argsort(values)
        cumulative = np.cumsum(counts[permutation])
        position = 0.95 * (self.counts[task] - 1)
        lo, hi = math.floor(position), math.ceil(position)
        low = values[permutation[np.searchsorted(cumulative, lo + 1)]]
        high = values[permutation[np.searchsorted(cumulative, hi + 1)]]
        p95 = low + (high - low) * (position - lo)
        return alpha * (
            self.weights["L"] * float(counts @ values / self.counts[task])
            + self.weights["P"] * float(p95)
        )

    def capacity_witness(self, choices):
        """One finite feasible-flow witness, not a utility decision or proof."""
        if (
            not self.use_witness
            or self.data.get("performance_mode", "independent") != "independent"
            or self.expired()
        ):
            return
        states = [(m, n) for m in choices for n in choices[m]]
        paths = [
            (t, m)
            for t in self.rates
            for m in choices
            if t in self.data["models"][m]["supported_tasks"]
        ]
        ns, N = len(states), len(states) + len(paths)
        rows, lows, highs = [], [], []

        def add(entries, lo=-math.inf, hi=math.inf):
            r = np.zeros(N)
            for i, v in entries.items():
                r[i] = v
            rows.append(r)
            lows.append(lo)
            highs.append(hi)

        memory = {
            i: self.data["safe_factor"] * self.data["memory_table_mib"][m][str(n)]
            for i, (m, n) in enumerate(states)
        }
        add(memory, hi=self.budget)
        objective = np.zeros(N)
        for i, v in memory.items():
            objective[i] = self.weights["M"] * v / self.budget
        stress = 2.0 if self.policy == "robust" else 1.0
        for m in choices:
            add({i: 1.0 for i, (name, n) in enumerate(states) if name == m}, 1.0, 1.0)
            d = {
                ns + j: stress * self.data["models"][m]["service_time_ms"][t] / 1000
                for j, (t, name) in enumerate(paths)
                if name == m
            }
            d.update(
                {
                    i: -0.8 * self.data["capacity_factors"][m][str(n)]
                    for i, (name, n) in enumerate(states)
                    if name == m
                }
            )
            add(d, hi=0.0)
        for t, rate in self.rates.items():
            add(
                {ns + j: 1.0 for j, (task, m) in enumerate(paths) if task == t},
                rate,
                rate,
            )
        for j, (t, m) in enumerate(paths):
            d = {ns + j: 1.0}
            d.update(
                {
                    i: -self.rates[t]
                    for i, (name, n) in enumerate(states)
                    if name == m and n > 0
                }
            )
            add(d, hi=0.0)
            objective[ns + j] = (
                self.weights["L"]
                * self.data["models"][m]["service_time_ms"][t]
                / self.refs[t]
                / sum(self.rates.values())
            )
        remaining = self.time_limit - (time.monotonic() - self.start)
        if remaining <= 0:
            return
        fit = milp(
            objective,
            integrality=np.r_[np.ones(ns), np.zeros(len(paths))],
            bounds=Bounds(
                np.zeros(N), np.r_[np.ones(ns), [self.rates[t] for t, m in paths]]
            ),
            constraints=LinearConstraint(np.array(rows), lows, highs),
            options={"time_limit": min(5.0, remaining), "mip_rel_gap": 0.0},
        )
        self.metrics["capacity_probe_calls"] += 1
        if fit.x is None or fit.status not in (0, 1):
            return
        if any(abs(fit.x[i] - round(fit.x[i])) > 1e-5 for i in range(ns)):
            return
        x = {m: n for i, (m, n) in enumerate(states) if fit.x[i] > 0.5 and n}
        routing = {
            t: {
                m: float(fit.x[ns + j])
                for j, (task, m) in enumerate(paths)
                if task == t and m in x and fit.x[ns + j] > 0
            }
            for t in self.rates
        }
        if any(
            not math.isclose(
                sum(row.values()), self.rates[t], rel_tol=1e-7, abs_tol=1e-9
            )
            or set(row) - set(x)
            for t, row in routing.items()
        ):
            return
        # Discard numerical flow on absent models, then restore exact offered
        # rates before verifying aggregate capacity. Never add an absent path.
        routing = {
            t: {m: q * self.rates[t] / sum(row.values()) for m, q in row.items()}
            for t, row in routing.items()
        }
        if (
            self.data["safe_factor"]
            * sum(self.data["memory_table_mib"][m][str(n)] for m, n in x.items())
            > self.budget + 1e-6
        ):
            return
        pressure = {
            m: sum(
                row.get(m, 0)
                * self.data["models"][m]["service_time_ms"].get(t, 0)
                / 1000
                * stress
                for t, row in routing.items()
            )
            / self.data["capacity_factors"][m][str(n)]
            for m, n in x.items()
        }
        if max(pressure.values(), default=0) > 0.8 + 1e-8:
            return
        self.witness = dict(
            deployment=x,
            routing=routing,
            source="capacity MILP feasible-flow witness; not full-utility optimum",
            solver_status=int(fit.status),
        )

    def columns(self, choices, K):
        columns = []
        stress = 2.0 if self.policy == "robust" else 1.0
        for task, rate in self.rates.items():
            models = [
                m
                for m in choices
                if task in self.data["models"][m]["supported_tasks"]
                and any(n > 0 for n in choices[m])
            ]
            seen = set()
            for denominator in range(1, K + 1):
                # At most denominator models can receive positive grid flow.
                for size in range(1, min(denominator, len(models)) + 1):
                    for active in itertools.combinations(models, size):
                        for extra in compositions(denominator - size, size):
                            units = [u + 1 for u in extra]
                            fractions = tuple(
                                (m, u / denominator) for m, u in zip(active, units)
                            )
                            if fractions in seen:
                                continue
                            seen.add(fractions)
                            row = {
                                m: rate * u / denominator for m, u in zip(active, units)
                            }
                            for counts in itertools.product(
                                *(tuple(n for n in choices[m] if n) for m in active)
                            ):
                                if self.expired() or len(columns) >= self.column_limit:
                                    self.metrics["task_columns"] = len(columns)
                                    return None
                                x = dict(zip(active, counts))
                                pressure = {
                                    m: stress
                                    * row[m]
                                    * service_lower_ms(self.data, m, task)
                                    / 1000
                                    / self.data["capacity_factors"][m][str(x[m])]
                                    for m in active
                                }
                                if max(pressure.values()) > 0.8 + 1e-9:
                                    continue
                                floor = self.floor(task, x, row)
                                if floor is None:
                                    self.metrics["task_columns"] = len(columns)
                                    return None
                                columns.append(
                                    dict(
                                        task=task,
                                        deployment=x,
                                        routing=row,
                                        pressure=pressure,
                                        floor=floor,
                                    )
                                )
        if self.witness:
            for task, row in self.witness["routing"].items():
                x = {m: self.witness["deployment"][m] for m in row}
                if any(
                    c["task"] == task and c["deployment"] == x and c["routing"] == row
                    for c in columns
                ):
                    continue
                if self.expired() or len(columns) >= self.column_limit:
                    return None
                floor = self.floor(task, x, row)
                if floor is None:
                    return None
                pressure = {
                    m: stress
                    * row[m]
                    * service_lower_ms(self.data, m, task)
                    / 1000
                    / self.data["capacity_factors"][m][str(x[m])]
                    for m in row
                }
                columns.append(
                    dict(
                        task=task,
                        deployment=x,
                        routing=row,
                        pressure=pressure,
                        floor=floor,
                    )
                )
        return columns

    def solve(self, replica_options=None, deployment=None, K=1):
        self.start = time.monotonic()
        if type(K) is not int or not 1 <= K <= 16:
            raise ValueError("route_grid must be an integer in [1,16]")
        if any(n == 0 for n in self.counts.values()):
            return self.result(K, "INSUFFICIENT_COHORT")
        choices = {}
        for m in sorted(self.data["models"]):
            available = sorted(int(n) for n in self.data["capacity_factors"][m])
            if deployment is not None:
                choices[m] = [deployment.get(m, 0)]
            else:
                supplied = (replica_options or {}).get(m, [n for n in available if n])
                if any(
                    type(n) is not int or n <= 0 or n not in available for n in supplied
                ):
                    raise ValueError(f"Invalid measured replica options for {m}")
                choices[m] = [0] + sorted(set(supplied))
        states = [(m, n) for m in choices for n in choices[m]]
        state_index = {p: i for i, p in enumerate(states)}
        if K >= 2:
            self.capacity_witness(choices)
        columns = self.columns(choices, K)
        if columns is None:
            return self.result(K, "MASTER_BUILD_BUDGET_LIMIT")
        self.metrics["task_columns"] = len(columns)
        if any(not any(c["task"] == t for c in columns) for t in self.rates):
            return self.result(K, "NO_FEASIBLE_GRID_PLAN")
        ns = len(states)
        theta = ns + len(columns)
        N = theta + 1
        rows, lows, highs = [], [], []

        def add(entries, lo=-math.inf, hi=math.inf):
            rows.append(entries)
            lows.append(lo)
            highs.append(hi)

        for m in choices:
            add({state_index[m, n]: 1.0 for n in choices[m]}, 1.0, 1.0)
        memory = {
            i: self.data["safe_factor"] * self.data["memory_table_mib"][m][str(n)]
            for i, (m, n) in enumerate(states)
        }
        add(memory, hi=self.budget)
        for t in self.rates:
            add(
                {ns + j: 1.0 for j, c in enumerate(columns) if c["task"] == t}, 1.0, 1.0
            )
        # Aggregate linking is stronger than one separate implication per column.
        for t in self.rates:
            for m, n in states:
                links = {
                    ns + j: 1.0
                    for j, c in enumerate(columns)
                    if c["task"] == t and c["deployment"].get(m) == n and n > 0
                }
                if links:
                    add({**links, state_index[m, n]: -1.0}, hi=0.0)
        for m in choices:
            pressure = {
                ns + j: c["pressure"][m]
                for j, c in enumerate(columns)
                if m in c["pressure"]
            }
            pressure.update({state_index[m, n]: -0.8 for n in choices[m] if n})
            add(pressure, hi=0.0)
        floor_row = {
            i: -self.weights["M"] * value / self.budget for i, value in memory.items()
        }
        floor_row.update({ns + j: -c["floor"] for j, c in enumerate(columns)})
        add({**floor_row, theta: 1.0}, lo=0.0)
        objective = np.zeros(N)
        objective[theta] = 1.0
        integrality = np.ones(N)
        integrality[theta] = 0
        bounds = Bounds(np.zeros(N), np.r_[np.ones(theta), np.inf])
        blocks = len(choices) + len(self.rates)
        evaluated = set()
        status = "MILP_BUDGET_LIMIT"
        complete = False
        for iteration in range(self.iteration_limit):
            remaining = self.time_limit - (time.monotonic() - self.start)
            if remaining <= 0:
                break
            rr, cc, vv = [], [], []
            for i, entries in enumerate(rows):
                for j, value in entries.items():
                    rr.append(i)
                    cc.append(j)
                    vv.append(value)
            matrix = coo_matrix((vv, (rr, cc)), shape=(len(rows), N)).tocsc()
            remaining = self.time_limit - (time.monotonic() - self.start)
            if remaining <= 0:
                break
            fit = milp(
                objective,
                integrality=integrality,
                bounds=bounds,
                constraints=LinearConstraint(matrix, lows, highs),
                options={"time_limit": remaining, "mip_rel_gap": 0.0},
            )
            self.metrics["milp_calls"] += 1
            if fit.status == 2:
                self.lower = math.inf
                complete = True
                break
            if fit.status not in (0, 1):
                status = "MILP_SOLVER_ERROR"
                self.history.append(
                    {"solver_status": int(fit.status), "message": fit.message}
                )
                break
            dual = getattr(fit, "mip_dual_bound", None)
            if dual is not None and math.isfinite(dual):
                self.lower = max(self.lower, float(dual))
            incumbent = self.diagnostic if self.diagnostic_mode else self.best
            if incumbent and self.lower >= -incumbent["score"] - self.tolerance:
                complete = True
                break
            if fit.x is None:
                break
            selected = tuple(i for i in range(theta) if fit.x[i] > 0.5)
            if len(selected) != blocks or any(
                abs(fit.x[i] - round(fit.x[i])) > 1e-5 for i in range(theta)
            ):
                status = "MILP_INVALID_WITNESS"
                break
            x = {m: n for i, (m, n) in enumerate(states) if i in selected and n}
            selected_columns = [columns[i - ns] for i in selected if i >= ns]
            routing = {c["task"]: c["routing"] for c in selected_columns}
            candidate_floor = (
                self.weights["M"]
                * sum(memory[i] for i in selected if i < ns)
                / self.budget
            )
            candidate_floor += sum(c["floor"] for c in selected_columns)
            if selected in evaluated:
                # Numerical lower/upper gap can remain after an identity cut.
                status = "MILP_NUMERICAL_UNRESOLVED"
                break
            evaluated.add(selected)
            p = predict_scenarios(
                self.data,
                x,
                routing,
                self.rates,
                self.budget,
                self.refs,
                self.weights,
                trace=self.trace,
                **self.context,
            )
            self.metrics["full_prediction_calls"] += 1
            valid = p["constraints_eligible"] and (
                self.policy != "robust" or p["stress_constraints_valid"]
            )
            unknown_performance = p.get("performance_unknown", False)
            score = p["scenario_U_low"] if self.policy == "robust" else p["U"]
            if unknown_performance:
                self.unknown_lower = min(self.unknown_lower, candidate_floor)
                self.metrics["unknown_candidates"] += 1
                add({i: 1.0 for i in selected}, hi=blocks - 1)
            elif not valid:
                add({i: 1.0 for i in selected}, hi=blocks - 1)
                self.metrics["infeasible_cuts"] += 1
            else:
                delay_cost = (
                    max(p["scenario_delay_costs"])
                    if self.policy == "robust"
                    else p["scenario_delay_costs"][0]
                )
                memory_cost = (
                    self.weights["M"]
                    * sum(memory[i] for i in selected if i < ns)
                    / self.budget
                )
                if self.data.get("performance_mode", "independent") == "independent":
                    # In this predictor, fixed task route/count columns fix all
                    # FIFO queues even if idle residents change. Transfer only
                    # T/L/P, never nonadditive energy, across those deployments.
                    route_indices = [i for i in selected if i >= ns]
                    entries = {
                        i: -self.weights["M"] * v / self.budget
                        for i, v in memory.items()
                    }
                    entries.update({i: -delay_cost for i in route_indices})
                    add({**entries, theta: 1.0}, lo=-delay_cost * (len(self.rates) - 1))
                    self.metrics["route_feedback_cuts"] += 1
                if score is None:
                    partial = [p["diagnostic_partial_U"]]
                    if self.policy == "robust":
                        partial = p["scenario_diagnostic_partial_U"]
                    diagnostic_score = min(partial)
                    if (
                        self.diagnostic is None
                        or diagnostic_score > self.diagnostic["score"]
                    ):
                        self.diagnostic = dict(
                            deployment=x,
                            routing=routing,
                            prediction=p,
                            score=diagnostic_score,
                            objective="partial_diagnostic_missing_energy",
                        )
                    if self.diagnostic_mode:
                        cost = -diagnostic_score
                    else:
                        self.unknown_lower = min(
                            self.unknown_lower, memory_cost + delay_cost
                        )
                        self.metrics["unknown_candidates"] += 1
                        add({i: 1.0 for i in selected}, hi=blocks - 1)
                        cost = None
                else:
                    if self.best is None or score > self.best["score"]:
                        self.best = dict(
                            deployment=x,
                            routing=routing,
                            prediction=p,
                            score=score,
                            objective=self.policy,
                        )
                    cost = -score
                if cost is not None:
                    if cost < candidate_floor - self.tolerance:
                        raise ValueError("Predictor violates MILP service lower bound")
                    # Exact at this complete discrete identity; <=0 everywhere
                    # else, so the cut cannot remove a better unseen plan.
                    add(
                        {**{i: -cost for i in selected}, theta: 1.0},
                        lo=-cost * (blocks - 1),
                    )
                    self.metrics["feedback_cuts"] += 1
            self.history.append(
                dict(
                    iteration=iteration + 1,
                    master_lower_cost=self.lower,
                    candidate_floor=candidate_floor,
                    candidate_U=score,
                    deployment=x,
                    routing=routing,
                    reason=p["reason"],
                    solver_status=int(fit.status),
                )
            )
            if fit.status == 1:
                break
        if complete:
            if self.best:
                status = (
                    "FEASIBLE_INCOMPLETE_EVIDENCE_DOMAIN"
                    if self.metrics["unknown_candidates"]
                    else "OPTIMAL_MILP_FINITE_DOMAIN"
                )
            elif self.diagnostic:
                status = "DIAGNOSTIC_ONLY_NO_ENERGY"
            else:
                status = (
                    "INCOMPLETE_PERFORMANCE_DOMAIN"
                    if self.metrics["unknown_candidates"]
                    else "NO_FEASIBLE_GRID_PLAN"
                )
        return self.result(K, status, complete)

    def result(self, K, status, complete=False):
        certified = status == "OPTIMAL_MILP_FINITE_DOMAIN"
        lower = min(self.lower, self.unknown_lower)
        if self.best:
            lower = min(lower, -self.best["score"])
        upper = -lower if math.isfinite(lower) else None
        return dict(
            winner=self.best or self.diagnostic,
            diagnostic_candidate=self.diagnostic if self.best else None,
            status=status,
            solver="milp",
            search_completed=complete,
            optimality_certified=certified,
            diagnostic_optimality_certified=complete
            and self.diagnostic_mode
            and self.diagnostic is not None
            and not self.metrics["unknown_candidates"],
            objective_policy=self.policy,
            utility_upper_bound=upper,
            optimality_gap=(
                max(0.0, upper - self.best["score"])
                if self.best and upper is not None
                else None
            ),
            diagnostic_optimality_gap=(
                max(0.0, upper - self.diagnostic["score"])
                if self.diagnostic_mode and self.diagnostic and upper is not None
                else None
            ),
            domain=dict(
                replicas="measured catalog counts only",
                routing=(
                    "union of grids with denominators 1..K plus recorded capacity witness rows"
                    if self.witness
                    else "union of grids with denominators 1..K"
                ),
                K=K,
                capacity_boundary_witnesses=self.witness is not None,
                capacity_witness=self.witness,
                continuous_optimality=False,
                service_bound="exact routed service mean/P95 using shared routing-pattern cache",
            ),
            trace_sha256=trace_digest(self.trace),
            offered_requests=len(self.trace),
            metrics=self.metrics,
            history=self.history,
            certificate_scope="frozen predictor and declared finite grid; not physical GPU or continuous optimality",
            unknown_domain_lower_cost=(
                self.unknown_lower if math.isfinite(self.unknown_lower) else None
            ),
            elapsed_seconds=time.monotonic() - self.start,
        )
