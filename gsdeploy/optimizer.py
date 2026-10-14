"""Joint measured-replica / nested-grid routing search with safe bounds.

All leaves use the same cohort and whole-plan predictor. Certificates cover
only the explicitly enumerated finite domain, never continuous routing or GPU
performance. A state budget yields an incumbent and an optimistic gap.
"""

import itertools
import math
import time
import numpy as np
from .prediction import predict, predict_scenarios
from .routing import Router
from .trace import expand_routes, make_trace, trace_digest


def compositions(total, size):
    if size == 1:
        yield (total,)
    else:
        for first in range(total + 1):
            for rest in compositions(total - first, size - 1):
                yield (first,) + rest


class UtilityOptimizer:
    def __init__(self, data, rates, budget, refs, weights, **context):
        self.d, self.rates, self.budget, self.refs, self.w = (
            data,
            dict(sorted(rates.items())),
            budget,
            refs,
            weights,
        )
        self.max_states = context.pop("search_max_states", 100000)
        self.policy = context.pop("search_policy", "nominal")
        self.context = context
        self.trace = make_trace(
            self.rates,
            context.get("horizon", 120),
            context.get("seed", 42),
            context.get("arrival_process", "deterministic_periodic"),
        )
        self.task_offsets = {
            t: [a for a, task in self.trace if task == t] for t in self.rates
        }
        self.metrics = dict(
            search_states=0,
            full_prediction_calls=0,
            capacity_pruned=0,
            memory_pruned=0,
            bound_pruned=0,
        )
        self.limited = False
        self.best = self.diagnostic = None
        self.unknown = False
        self.floor_cache = {}

    def tick(self):
        if self.metrics["search_states"] >= self.max_states:
            self.limited = True
            return False
        self.metrics["search_states"] += 1
        return True

    def service(self, model, count, task):
        return (
            self.d["models"][model]["service_time_ms"][task]
            * count
            / self.d["capacity_factors"][model][str(count)]
        )

    def service_floor(self, x, task, row=None):
        if row is None:
            values = [
                self.service(m, n, task)
                for m, n in x.items()
                if task in self.d["models"][m]["supported_tasks"]
            ]
            return min(values) if values else math.inf
        key = (tuple(sorted(x.items())), task, tuple(sorted(row.items())))
        if key in self.floor_cache:
            return self.floor_cache[key]
        aliases, routes = expand_routes(x, {task: row})
        router = Router(routes)
        values = []
        for _ in self.task_offsets[task]:
            alias = router.choose(task)
            values.append(self.service(aliases[alias], x[aliases[alias]], task))
        # Queue waiting is nonnegative; both mean and order statistics are monotone.
        alpha = self.rates[task] / math.fsum(self.rates.values()) / self.refs[task]
        value = alpha * (
            self.w["L"] * float(np.mean(values))
            + self.w["P"] * float(np.percentile(values, 95))
        )
        self.floor_cache[key] = value
        return value

    def upper(self, x, rows, memory):
        total = math.fsum(self.rates.values())
        cost = self.w["M"] * memory / self.budget
        for t in self.rates:
            if t in rows:
                cost += self.service_floor(x, t, rows[t])
            else:
                cost += (
                    self.rates[t]
                    / total
                    / self.refs[t]
                    * (self.w["L"] + self.w["P"])
                    * self.service_floor(x, t)
                )
        # T loss and whole-plan energy are nonnegative, including nonadditive energy.
        return -cost

    def rows(self, task, models, caps, K):
        q = self.rates[task]
        seen = set()
        for denominator in range(1, K + 1):
            for units in compositions(denominator, len(models)):
                if not self.tick():
                    return
                row = {m: q * u / denominator for m, u in zip(models, units) if u}
                key = tuple(round(row.get(m, 0) / q, 12) for m in models)
                if key not in seen:
                    seen.add(key)
                    if all(row.get(m, 0) <= caps[m] + 1e-9 for m in models):
                        yield row
                    else:
                        self.metrics["capacity_pruned"] += 1
        if K == 1:
            return
        # Capacity-boundary witnesses avoid falsely declaring grid-infeasible loads
        # physically infeasible. Grid points remain nested; these are extra points.
        orders = (
            itertools.permutations(models)
            if len(models) <= 4
            else (tuple(models[i:] + models[:i]) for i in range(len(models)))
        )
        for order in orders:
            if not self.tick():
                return
            left, row = q, {}
            for m in order:
                amount = min(left, max(0.0, caps[m]))
                if amount > 1e-12:
                    row[m] = amount
                    left -= amount
            if left <= 1e-8:
                # Correct roundoff without changing the offered cohort.
                last = next(reversed(row))
                row[last] += left
                key = tuple(round(row.get(m, 0) / q, 12) for m in models)
                if key not in seen:
                    seen.add(key)
                    yield row

    def solve(self, replica_options=None, deployment=None, K=1):
        start = time.perf_counter()
        if type(K) is not int or not 1 <= K <= 16:
            raise ValueError(
                "route_grid must be an integer in [1,16]; denominators 1..K are included"
            )
        if any(not offsets for offsets in self.task_offsets.values()):
            return self.result(start, K, "INSUFFICIENT_COHORT")
        models = sorted(deployment if deployment is not None else self.d["models"])
        choices = [
            (
                [deployment[m]]
                if deployment is not None
                else [0]
                + sorted(
                    (replica_options or {}).get(
                        m, [int(n) for n in self.d["capacity_factors"][m] if int(n) > 0]
                    )
                )
            )
            for m in models
        ]
        deployments = []
        stress = 2.0 if self.policy == "robust" else 1.0

        def enumerate_deployments(index, x, memory):
            if not self.tick():
                return
            if memory > self.budget + 1e-6:
                self.metrics["memory_pruned"] += 1
                return
            if index == len(models):
                # Necessary capacity coverage before any route combinations.
                for t, q in self.rates.items():
                    capacity = math.fsum(
                        0.8
                        * self.d["capacity_factors"][m][str(n)]
                        * 1000
                        / (stress * self.d["models"][m]["service_time_ms"][t])
                        for m, n in x.items()
                        if t in self.d["models"][m]["supported_tasks"]
                    )
                    if capacity < q - 1e-8:
                        self.metrics["capacity_pruned"] += 1
                        return
                deployments.append(
                    (self.upper(x, {}, memory), tuple(sorted(x.items())), memory)
                )
                return
            m = models[index]
            for n in choices[index]:
                if self.limited:
                    return
                updated = {**x, m: n} if n else x
                added = self.d["safe_factor"] * self.d["memory_table_mib"][m][str(n)]
                enumerate_deployments(index + 1, updated, memory + added)

        enumerate_deployments(0, {}, 0.0)
        deployments.sort(key=lambda item: (-item[0], item[1]))
        # Enumeration consumes the same budget as routing; incomplete domains are
        # reported explicitly, including when no incumbent was found.
        generation_limited = self.limited
        self.limited = False
        stress = 2.0 if self.policy == "robust" else 1.0
        for bound, items, memory in deployments:
            if self.limited:
                break
            x = dict(items)
            tasks = sorted(
                self.rates,
                key=lambda t: (
                    math.fsum(
                        0.8
                        * self.d["capacity_factors"][m][str(n)]
                        * 1000
                        / (stress * self.d["models"][m]["service_time_ms"][t])
                        for m, n in x.items()
                        if t in self.d["models"][m]["supported_tasks"]
                    )
                    / self.rates[t],
                    t,
                ),
            )

            def visit(rows, demand):
                if not self.tick():
                    return
                ub = self.upper(x, rows, memory)
                incumbent = self.best
                # If there are no energy records at all, only certify the explicitly
                # labelled partial diagnostic objective, never five-term utility.
                if incumbent is None and not self.d.get("energy_profiles"):
                    incumbent = self.diagnostic
                if incumbent and ub < incumbent["score"] - 1e-10:
                    self.metrics["bound_pruned"] += 1
                    return
                if len(rows) == len(tasks):
                    self.metrics["full_prediction_calls"] += 1
                    fn = predict_scenarios if self.policy == "robust" else predict
                    p = fn(
                        self.d,
                        x,
                        rows,
                        self.rates,
                        self.budget,
                        self.refs,
                        self.w,
                        trace=self.trace,
                        **self.context,
                    )
                    if not p["constraints_eligible"] or (
                        self.policy == "robust" and not p["stress_constraints_valid"]
                    ):
                        return
                    if p["U"] is None or (
                        self.policy == "robust" and p["scenario_U_low"] is None
                    ):
                        self.unknown = True
                        score = p["diagnostic_partial_U"]
                        if self.policy == "robust":
                            # Explicitly diagnostic: missing energy cannot become robust U.
                            score = min(
                                predict(
                                    self.d,
                                    x,
                                    rows,
                                    self.rates,
                                    self.budget,
                                    self.refs,
                                    self.w,
                                    multiplier=s,
                                    trace=self.trace,
                                    **self.context,
                                )["diagnostic_partial_U"]
                                for s in (1, 1.25, 2)
                            )
                        if (
                            self.diagnostic is None
                            or score > self.diagnostic["score"] + 1e-10
                        ):
                            self.diagnostic = dict(
                                deployment=dict(x),
                                routing={t: dict(rows[t]) for t in sorted(rows)},
                                prediction=p,
                                score=score,
                                objective="partial_diagnostic_missing_energy",
                            )
                    else:
                        score = (
                            p["scenario_U_low"] if self.policy == "robust" else p["U"]
                        )
                        if self.best is None or score > self.best["score"] + 1e-10:
                            self.best = dict(
                                deployment=dict(x),
                                routing={t: dict(rows[t]) for t in sorted(rows)},
                                prediction=p,
                                score=score,
                                objective=self.policy,
                            )
                    return
                t = tasks[len(rows)]
                supported = [
                    m for m in x if t in self.d["models"][m]["supported_tasks"]
                ]
                caps = {
                    m: max(
                        0,
                        0.8 * self.d["capacity_factors"][m][str(x[m])] / stress
                        - demand[m],
                    )
                    * 1000
                    / self.d["models"][m]["service_time_ms"][t]
                    for m in supported
                }
                if sum(caps.values()) < self.rates[t] - 1e-8:
                    self.metrics["capacity_pruned"] += 1
                    return
                options = list(self.rows(t, supported, caps, K))
                options.sort(
                    key=lambda row: (
                        self.service_floor(x, t, row),
                        tuple(sorted(row.items())),
                    )
                )
                for row in options:
                    if self.limited:
                        return
                    updated = dict(demand)
                    for m, q in row.items():
                        updated[m] += (
                            q * self.d["models"][m]["service_time_ms"][t] / 1000
                        )
                    visit({**rows, t: row}, updated)

            visit({}, {m: 0.0 for m in x})
        self.limited |= generation_limited
        status = "BUDGET_LIMIT" if self.limited else "OPTIMAL_ENUMERATED_DOMAIN"
        if self.best is None:
            status = (
                "DIAGNOSTIC_ONLY_NO_ENERGY"
                if self.diagnostic
                else (
                    "BUDGET_LIMIT_NO_INCUMBENT"
                    if self.limited
                    else "NO_FEASIBLE_ENUMERATED_PLAN"
                )
            )
        elif self.unknown:
            status = "FEASIBLE_INCOMPLETE_ENERGY_DOMAIN"
        return self.result(start, K, status)

    def result(self, start, K, status):
        winner = self.best or self.diagnostic
        total = math.fsum(self.rates.values())
        global_upper = (
            -math.fsum(
                self.rates[t]
                / total
                / self.refs[t]
                * (self.w["L"] + self.w["P"])
                * min(
                    self.service(m, int(n), t)
                    for m in self.d["models"]
                    if t in self.d["models"][m]["supported_tasks"]
                    for n in self.d["capacity_factors"][m]
                    if int(n) > 0
                )
                for t in self.rates
            )
            if all(
                any(t in s["supported_tasks"] for s in self.d["models"].values())
                for t in self.rates
            )
            else None
        )
        exact = status == "OPTIMAL_ENUMERATED_DOMAIN"
        if winner:
            # Stress diagnostics are computed once for the final nominal incumbent.
            winner["prediction"] = predict_scenarios(
                self.d,
                winner["deployment"],
                winner["routing"],
                self.rates,
                self.budget,
                self.refs,
                self.w,
                trace=self.trace,
                **self.context,
            )
        return {
            "winner": winner,
            "diagnostic_candidate": self.diagnostic if self.best else None,
            "status": status,
            "search_completed": not self.limited,
            "optimality_certified": exact,
            "objective_policy": self.policy,
            "domain": {
                "replicas": "measured catalog counts only",
                "routing": "union of grids with denominators 1..K plus conditional capacity-boundary witnesses",
                "K": K,
                "continuous_optimality": False,
            },
            "utility_upper_bound": winner["score"] if exact else global_upper,
            "optimality_gap": (
                0.0
                if exact
                else (
                    max(0.0, global_upper - self.best["score"])
                    if self.best and global_upper is not None
                    else None
                )
            ),
            "trace_sha256": trace_digest(self.trace),
            "offered_requests": len(self.trace),
            "metrics": self.metrics,
            "elapsed_seconds": time.perf_counter() - start,
        }
