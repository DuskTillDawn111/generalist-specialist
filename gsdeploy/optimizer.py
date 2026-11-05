"""Joint measured-replica / nested-grid routing search with safe bounds.

All leaves use the same cohort and whole-plan predictor. Certificates cover
only the explicitly enumerated finite domain, never continuous routing or GPU
performance. A state budget yields an incumbent and an optimistic gap.
"""

import itertools
import math
import time
import numpy as np
from .prediction import predict_scenarios
from .quality import task_quality
from .routing import Router
from .surrogate_bounds import surrogate_bounds
from .trace import expand_routes, make_trace, trace_digest
from .route_domain import compositions, fraction_rows, domain_description, validate_pair_denominators
from .config import TASKS
from .coupled_mode import is_coupled, bound_contract
from . import normalization as norm

# dfs + 剪枝
class UtilityOptimizer:
    def __init__(self, data, rates, budget, refs, weights, **context):
        norm.require_ready(data, rates, refs, context.get("horizon", 120.0), budget)
        self.reference_v2 = norm.is_v2(data)
        if data.get("performance_mode") == "paper_profiled_fifo_v1":
            from .paper_forecast import session

            data = session(data).data  # Reuse the same immutable profiles at every leaf.
        self.d, self.rates, self.budget, self.refs, self.w = (
            data,
            dict(sorted(rates.items())),
            budget,
            refs,
            weights,
        )
        self.max_states = context.pop("search_max_states", 100000)
        self.policy = context.pop("search_policy", "nominal")
        # The explicit grid-only contract is shared with MILP. Legacy direct
        # callers retain DFS's historical conditional capacity witnesses.
        self.use_witness = context.pop("milp_capacity_witness", True)
        self.pair_denominators = context.pop("route_pair_denominators_by_task", {})
        validate_pair_denominators(self.pair_denominators, TASKS)
        self.time_limit = context.pop(
            "dfs_time_limit_seconds", context.pop("milp_time_limit_seconds", math.inf)
        )
        self.bound_pruning = context.pop("dfs_bound_pruning", True)
        self.solo_fifo_bound = context.pop("dfs_solo_fifo_bound", True)
        self.independent = data.get("performance_mode", "independent") == "independent"
        self.context = context
        self.surrogate = surrogate_bounds(data, self.rates, **context)
        self.coupled = is_coupled(data)
        if self.coupled:
            self.use_witness = False
        self.separable_service = not self.coupled and (self.independent or self.surrogate is not None)
        self.profiled_memory = (self.coupled or data.get("memory_mode") == "profiled"
                               or (data.get("performance_mode") in {"calibrated_surrogate", "calibrated_replay_v2", "replay_transfer_v1"}
                                   and self.surrogate is None))
        self.trace = make_trace(
            self.rates,
            context.get("horizon", 120),
            context.get("seed", 42),
            context.get("arrival_process", "deterministic_periodic"),
        )
        self.task_offsets = {
            t: [a for a, task in self.trace if task == t] for t in self.rates
        }
        self.task_input_ids = (self.surrogate.task_input_ids(self.trace, context.get("seed", 42))
                               if self.surrogate else None)
        self.metrics = dict(
            search_states=0,
            full_prediction_calls=0,
            capacity_pruned=0,
            memory_pruned=0,
            bound_pruned=0,
            unknown_candidates=0,
            infeasible_candidates=0,
            excluded_deployments=0,
            excluded_plans=0,
            quality_pattern_evaluations=0,
            utility_bound_evaluations=0,
            first_feasible_seconds=None,
            best_found_seconds=None,
        )
        self.limited = False
        self.best = self.diagnostic = None
        self.unknown = False
        self.unknown_upper = -math.inf
        self.unresolved_candidates = []
        self.stop_reason = None
        self.floor_cache = {}
        self.quality_floor_cache = {}
        self.start = time.perf_counter()

    @staticmethod
    def deployment_key(deployment):
        return tuple(sorted((m, n) for m, n in deployment.items() if n > 0))

    def plan_key(self, deployment, routing):
        return (self.deployment_key(deployment), tuple(
            (t, tuple(sorted((m, round(q / self.rates[t], 12))
                             for m, q in row.items() if q > 0)))
            for t, row in sorted(routing.items())
        ))

    def tick(self):
        if time.perf_counter() - self.start >= self.time_limit:
            self.limited = True
            self.stop_reason = "TIME_LIMIT"
            return False
        if self.metrics["search_states"] >= self.max_states:
            self.limited = True
            self.stop_reason = "STATE_LIMIT"
            return False
        self.metrics["search_states"] += 1
        return True

    def service(self, model, count, task):
        return (
            self.family_service(model, count, task)
            * count
            / self.d["capacity_factors"][model][str(count)]
        )

    def family_service(self, model, count, task):
        if self.coupled:
            return 0.0
        if self.surrogate:
            return self.surrogate.family_service(model, task, count)
        return self.d["models"][model]["service_time_ms"][task]

    def capacity(self, model, count, task, stress=1.0, demand=0.0):
        service = self.pressure_service(model, count, task)
        if service == 0:
            # Missing calibrated path means no necessary service/capacity
            # condition is known. The full predictor will retain it as unknown.
            return math.inf
        return max(0.0, .8 * self.d["capacity_factors"][model][str(count)]
                   / stress - demand) * 1000 / service

    def pressure_service(self, model, count, task):
        if self.coupled:
            return 0.0
        if self.surrogate:
            return self.surrogate.pressure_service(model, task, count)
        return self.family_service(model, count, task)

    def service_floor(self, x, task, row=None):
        if not self.separable_service:
            # A standalone measurement is neither a latency lower bound nor a
            # necessary capacity condition for an unmeasured whole-plan profile.
            return 0.0
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
        # Each task alone on its actual replica queues is a relaxation of the
        # full FIFO schedule: deleting other tasks cannot delay its completions.
        # Keep the cached service-multiset floor for long traces.
        solo = self.solo_fifo_bound and len(self.task_offsets[task]) <= 1024
        clocks = {alias: 0.0 for alias in aliases} if solo else None
        outside = 0
        stress = 2.0 if self.policy == "robust" else 1.0
        for i, offset in enumerate(self.task_offsets[task]):
            if i % 256 == 0 and time.perf_counter() - self.start >= self.time_limit:
                self.limited = True
                self.stop_reason = "TIME_LIMIT"
                return 0.0
            alias = router.choose(task)
            model = aliases[alias]
            service = stress * (
                self.surrogate.request_service(model, task, x[model], self.task_input_ids[task][i])
                * x[model] / self.d["capacity_factors"][model][str(x[model])]
                if self.task_input_ids is not None else self.service(model, x[model], task))
            if solo:
                finish = max(offset, clocks[alias]) + service / 1000
                clocks[alias] = finish
                values.append((finish - offset) * 1000)
                outside += finish > self.context.get("horizon", 120)
            else:
                values.append(service)
        # Queue waiting is nonnegative; both mean and order statistics are monotone.
        alpha = self.rates[task] / math.fsum(self.rates.values()) / self.refs[task]
        value = alpha * (
            self.w["L"] * float(np.mean(values))
            + self.w["P"] * float(np.percentile(values, 95))
        )
        if solo:
            value += self.w["T"] * outside / len(self.trace)
        self.floor_cache[key] = value
        return value

    def upper(self, x, rows, memory):
        self.metrics["utility_bound_evaluations"] += 1
        if self.reference_v2:
            # A fixed global bound remains valid for negative normalized Q.
            # Old calibrated E and safe-memory floors have different units.
            return -self.w["Q"]*norm.quality_floor(self.d, self.rates)
        total = math.fsum(self.rates.values())
        # A standalone residency sum need not lower-bound measured whole-plan
        # memory. Only zero is established before the full identity is known.
        cost = 0.0 if self.profiled_memory else self.w["M"] * memory / self.budget
        if self.surrogate:
            cost += self.w.get("E", 0) * (
                self.surrogate.constant_energy
                + self.surrogate.state_energy(sum(x.values()))
                + math.fsum(self.surrogate.route_energy(t, x, row)
                            for t, row in rows.items())
            )
        for t in self.rates:
            if t in rows:
                cost += self.service_floor(x, t, rows[t])
                if self.w.get("Q", 0) > 0:
                    # This task's router/input replay does not depend on other
                    # tasks. Its frozen quality cost is exact once assigned;
                    # unknown quality retains the safe nonnegative zero floor.
                    key = (tuple(sorted(x.items())), t, tuple(sorted(rows[t].items())))
                    if key not in self.quality_floor_cache:
                        loss, _ = task_quality(self.d, t, x, rows[t], self.trace,
                                               self.context.get("seed", 42))
                        self.quality_floor_cache[key] = loss
                        self.metrics["quality_pattern_evaluations"] += 1
                    loss = self.quality_floor_cache[key]
                    if loss is not None:
                        cost += self.w["Q"] * self.rates[t] / total * loss
            else:
                cost += (
                    self.rates[t]
                    / total
                    / self.refs[t]
                    * (self.w["L"] + self.w["P"])
                    * self.service_floor(x, t)
                )
        # T, Q loss and whole-plan energy are nonnegative. Missing quality may
        # not be replaced by zero in a score, but zero is a valid relaxed bound.
        return -cost

    def rows(self, task, models, caps, K):
        q = self.rates[task]
        if self.no_split:
            for m in models:
                if not self.tick():
                    return
                if q <= caps[m] + 1e-9:
                    yield {m: q}
                else:
                    self.metrics["capacity_pruned"] += 1
            return
        seen = set()
        for fractions in fraction_rows(models, K, self.pair_denominators.get(task, ())):
            if not self.tick():
                return
            row = {m: q * f.numerator / f.denominator for m, f in fractions}
            # Shared generator already deduplicates exactly. This secondary
            # historical float key only avoids duplicate capacity witnesses.
            seen.add(tuple(round(row.get(m, 0) / q, 12) for m in models))
            if all(row.get(m, 0) <= caps[m] + 1e-9 for m in models):
                yield row
            else:
                self.metrics["capacity_pruned"] += 1
        if K == 1 or not self.use_witness or not self.independent:
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

    def solve(self, replica_options=None, deployment=None, K=1,
              exclude_deployments=None, exclude_plans=None, no_split=False):
        start = time.perf_counter()
        self.start = start
        self.no_split = no_split
        self.excluded_deployments = {
            self.deployment_key(x) for x in (exclude_deployments or [])
        }
        self.excluded_plans = {
            self.plan_key(p["deployment"], p["routing"]) for p in (exclude_plans or [])
        }
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
            if not self.profiled_memory and memory > self.budget + 1e-6:
                self.metrics["memory_pruned"] += 1
                return
            if index == len(models):
                if self.deployment_key(x) in self.excluded_deployments:
                    self.metrics["excluded_deployments"] += 1
                    return
                # Necessary capacity coverage before any route combinations.
                for t, q in self.rates.items():
                    supported = [
                        m for m in x if t in self.d["models"][m]["supported_tasks"]
                    ]
                    if not supported:
                        self.metrics["capacity_pruned"] += 1
                        return
                    capacity = math.fsum(
                        self.capacity(m, n, t, stress)
                        for m, n in x.items()
                        if t in self.d["models"][m]["supported_tasks"]
                    ) if self.separable_service else math.inf
                    if self.separable_service and capacity < q - 1e-8:
                        self.metrics["capacity_pruned"] += 1
                        return
                deployments.append(
                    (self.upper(x, {}, memory) if self.bound_pruning or self.reference_v2 else 0.0,
                     tuple(sorted(x.items())), memory)
                )
                return
            m = models[index]
            for n in choices[index]:
                if self.limited:
                    return
                updated = {**x, m: n} if n else x
                added = (0.0 if self.profiled_memory else
                         (self.surrogate.memory_multiplier if self.surrogate else 1.0)
                         * self.d["safe_factor"] * self.d["memory_table_mib"][m][str(n)])
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
                        self.capacity(m, n, t, stress)
                        for m, n in x.items()
                        if t in self.d["models"][m]["supported_tasks"]
                    )
                    / self.rates[t] if self.separable_service else 0.0,
                    t,
                ),
            )

            def visit(rows, demand):
                if not self.tick():
                    return
                # Plain DFS enumerates every constraint-admissible identity.
                # It neither computes utility bounds for ordering nor prunes
                # with them. Zero remains a valid upper bound for unknown U.
                ub = self.upper(x, rows, memory) if self.bound_pruning or self.reference_v2 else 0.0
                incumbent = self.best
                if self.bound_pruning and incumbent and ub < incumbent["score"] - 1e-10:
                    self.metrics["bound_pruned"] += 1
                    return
                if len(rows) == len(tasks):
                    if self.plan_key(x, rows) in self.excluded_plans:
                        self.metrics["excluded_plans"] += 1
                        return
                    self.metrics["full_prediction_calls"] += 1
                    # Identical complete-plan evaluation contract to MILP,
                    # including scenario diagnostics in nominal runs.
                    p = predict_scenarios(
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
                    if (p.get("performance_unknown", False) or p.get("memory_unknown", False)
                            or (self.policy == "robust" and any(p.get("scenario_performance_unknown", [])))):
                        self.record_unknown(x, rows, p, ub)
                        return
                    if not p["constraints_eligible"] or (
                        self.policy == "robust" and not p["stress_constraints_valid"]
                    ):
                        self.metrics["infeasible_candidates"] += 1
                        return
                    if p["U"] is None or (
                        self.policy == "robust" and p["scenario_U_low"] is None
                    ):
                        self.record_unknown(x, rows, p, ub)
                        score = p["diagnostic_partial_U"]
                        if self.policy == "robust":
                            # Explicitly diagnostic: missing evidence cannot become robust U.
                            partial = p["scenario_diagnostic_partial_U"]
                            score = min(partial) if all(v is not None for v in partial) else None
                        if score is not None and (
                            self.diagnostic is None
                            or score > self.diagnostic["score"] + 1e-10
                        ):
                            self.diagnostic = dict(
                                deployment=dict(x),
                                routing={t: dict(rows[t]) for t in sorted(rows)},
                                prediction=p,
                                score=score,
                                objective=(
                                    "partial_diagnostic_incomplete_utility"
                                    if self.w.get("Q", 0) > 0
                                    else "partial_diagnostic_missing_energy"
                                ),
                            )
                    else:
                        score = (
                            p["scenario_U_low"] if self.policy == "robust" else p["U"]
                        )
                        if self.best is None or score > self.best["score"] + 1e-10:
                            elapsed = time.perf_counter() - self.start
                            if self.best is None:
                                self.metrics["first_feasible_seconds"] = elapsed
                            self.metrics["best_found_seconds"] = elapsed
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
                    m: self.capacity(m, x[m], t, stress, demand[m])
                    for m in supported
                } if self.separable_service else {m: math.inf for m in supported}
                if sum(caps.values()) < self.rates[t] - 1e-8:
                    self.metrics["capacity_pruned"] += 1
                    return
                options = list(self.rows(t, supported, caps, K))
                if self.bound_pruning:
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
                            q * self.pressure_service(m, x[m], t) / 1000
                        )
                    visit({**rows, t: row}, updated)

            visit({}, {m: 0.0 for m in x})
        self.limited |= generation_limited
        status = (
            self.stop_reason if self.limited and self.stop_reason == "TIME_LIMIT"
            else "BUDGET_LIMIT" if self.limited else "OPTIMAL_ENUMERATED_DOMAIN"
        )
        if self.best is None:
            status = (
                ("DIAGNOSTIC_ONLY_INCOMPLETE_UTILITY" if self.w.get("Q", 0) > 0
                 else "DIAGNOSTIC_ONLY_NO_ENERGY")
                if self.diagnostic
                else (
                    f"{status}_NO_INCUMBENT"
                    if self.limited
                    else ("INCOMPLETE_PERFORMANCE_DOMAIN" if self.unknown
                          else "NO_FEASIBLE_ENUMERATED_PLAN")
                )
            )
            if self.limited and self.diagnostic:
                status = f"{self.stop_reason or 'BUDGET_LIMIT'}_DIAGNOSTIC_ONLY"
        elif self.unknown and not self.limited:
            status = "FEASIBLE_INCOMPLETE_EVIDENCE_DOMAIN"
        return self.result(start, K, status)

    def record_unknown(self, deployment, routing, prediction, upper):
        self.unknown = True
        self.unknown_upper = max(self.unknown_upper, upper)
        self.metrics["unknown_candidates"] += 1
        self.unresolved_candidates.append(dict(
            deployment=dict(deployment),
            routing={t: dict(row) for t, row in sorted(routing.items())},
            utility_upper_bound=upper,
            reason=prediction.get("reason", "INCOMPLETE_UTILITY"),
        ))

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
            if self.separable_service and all(
                any(t in s["supported_tasks"] for s in self.d["models"].values())
                for t in self.rates
            )
            else 0.0
        )
        if self.reference_v2:
            global_upper = -self.w["Q"]*norm.quality_floor(self.d, self.rates)
        exact = status == "OPTIMAL_ENUMERATED_DOMAIN"
        # Even a fully traversed grid may contain unknown measurements. Their
        # valid bounds survive enumeration, rather than becoming infeasible cuts.
        upper = (winner["score"] if exact else
                 max(self.best["score"] if self.best else -math.inf,
                     global_upper if self.limited else self.unknown_upper))
        if not math.isfinite(upper):
            upper = None
        return {
            "winner": winner,
            "diagnostic_candidate": self.diagnostic if self.best else None,
            "status": status,
            "solver": "dfs",
            "stop_reason": self.stop_reason,
            "search_completed": not self.limited,
            "optimality_certified": exact,
            "objective_policy": self.policy,
            "domain": {
                **domain_description(K, self.pair_denominators),
                "coupled_bounds": bound_contract(self.d),
                "search_strategy": ("branch_and_bound_dfs" if self.bound_pruning
                                    else "exhaustive_dfs_without_utility_pruning"),
                "utility_bound_pruning": self.bound_pruning,
                "replicas": "measured catalog counts only",
                "routing": (
                    ("base grid plus explicit pair grids and conditional capacity-boundary witnesses"
                     if self.pair_denominators else "union of grids with denominators 1..K plus conditional capacity-boundary witnesses")
                    if self.use_witness and self.independent and K > 1 and not self.no_split
                    else ("base grid plus explicit pair grids" if self.pair_denominators
                          else "union of grids with denominators 1..K")
                ),
                "K": K,
                "no_split": self.no_split,
                "excluded_deployments": len(self.excluded_deployments),
                "excluded_plans": len(self.excluded_plans),
                "capacity_boundary_witnesses": self.use_witness and self.independent and K > 1 and not self.no_split,
                "continuous_optimality": False,
                "service_bound": (
                    "solo-task FIFO with exact frozen per-input service for <=1024 requests; exact per-input no-wait multiset otherwise; unassigned minimum service; pool means only for rho"
                    if self.surrogate and self.surrogate.minimum_services is not None else
                    ("solo-task FIFO Q/T/L/P for assigned tasks with <=1024 samples; routed service otherwise"
                     if self.solo_fifo_bound else "routed standalone service floor") if self.separable_service
                    else "zero: whole-plan service unknown until complete evaluation"
                ),
                "quality_bound": "exact assigned-task frozen quality; zero for unassigned or missing quality",
                "reference_v2_bound_policy": ("global transformed native-zero quality floor; historical T/L/P/M/E bounds disabled"
                                               if self.reference_v2 else None),
                "memory_bound": ("zero: coupled placeholder has no complete-context memory evidence" if self.coupled
                                 else "zero until exact whole-plan profile evaluation" if self.profiled_memory
                                 else "validated frozen predictor memory multiplier times additive residency"
                                 if self.surrogate else "additive measured replica residency proxy"),
                "energy_bound": ("exact nonnegative frozen resource intercept/state/known-route decomposition; zero busy for missing paths"
                                 if self.surrogate else "zero relaxation"),
                "surrogate_bounds_version": (self.surrogate.spec["version_sha256"]
                                             if self.surrogate else None),
                "surrogate_bounds_schema": (self.surrogate.spec["schema"] if self.surrogate else None),
                "rho_bound": ("frozen per-input pool-mean contract; zero relaxation for missing paths"
                              if self.surrogate and self.surrogate.pressure_services is not None
                              else "fixed family service necessary capacity" if self.separable_service
                              else "zero service relaxation"),
                "missing_surrogate_paths": ("zero relaxation only; full predictor retains unknown evidence"
                                            if self.surrogate else None),
            },
            "utility_upper_bound": upper,
            "optimality_gap": (
                0.0
                if exact
                else (
                    max(0.0, upper - self.best["score"])
                    if self.best and upper is not None
                    else None
                )
            ),
            "trace_sha256": trace_digest(self.trace),
            "offered_requests": len(self.trace),
            "metrics": self.metrics,
            "unknown_domain_upper_utility": (
                self.unknown_upper if math.isfinite(self.unknown_upper) else None
            ),
            "unresolved_candidates": self.unresolved_candidates,
            "certificate_scope": "frozen predictor and declared finite grid; not physical GPU or continuous optimality",
            "elapsed_seconds": time.perf_counter() - start,
        }
