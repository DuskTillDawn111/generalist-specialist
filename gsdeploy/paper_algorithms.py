"""USHER placement and J3O-inspired proposals, guarded by whole FIFO utility.

These are adaptations to GSDeploy, not reproductions of the papers' domains
or approximation guarantees. In particular an LP is only a route proposer:
empirical FIFO P95 and co-run interference are evaluated by the full predictor.
"""

import copy
from dataclasses import replace
import itertools
import math
import time

import numpy as np
from .linear_solver import linprog

from .paper_forecast import MODE, canonical, session


class UsherResourcePlanner:
    """Configuration enumeration and complementary C/M best-fit placement.

    C is a declared required compute allocation, M is resident/workspace MiB.
    Neither DRAM bandwidth nor L2 utilization is substituted for memory capacity.
    Placement is a proposal; resource isolation and interference-free execution
    are not inferred from this packing calculation.
    """

    @staticmethod
    def groups(demands, max_group_size=4):
        if type(max_group_size) is not int or max_group_size < 1:
            raise ValueError("Positive maximum group size required")
        for row in demands.values():
            if any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k]) or row[k] < 0
                   for k in ("compute_fraction", "memory_fraction")):
                raise ValueError("Separate finite compute and capacity demands required")
        groups = [(name,) for name in sorted(demands)]
        while True:
            pairs = []
            for i, a in enumerate(groups):
                for j in range(i + 1, len(groups)):
                    b = groups[j]
                    if len(a) + len(b) <= max_group_size:
                        names = tuple(sorted(a + b))
                        distance = abs(sum(demands[n]["compute_fraction"] for n in names)
                                       - sum(demands[n]["memory_fraction"] for n in names))
                        pairs.append((distance, names, i, j))
            if not pairs:
                return groups
            _, names, i, j = min(pairs)
            groups = [g for k, g in enumerate(groups) if k not in (i, j)] + [names]
            groups.sort()

    @staticmethod
    def placement(configuration, devices):
        """USHER Algorithm 2 ordering, group affinity and least residual space."""
        if not configuration or not devices:
            raise ValueError("Explicit configuration and available devices required")
        for row in configuration.values():
            if (not row.get("evidence") or type(row.get("replicas")) is not int or row["replicas"] < 1
                    or any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k]) or row[k] <= 0
                           for k in ("compute_fraction", "memory_mib"))):
                raise ValueError("Measured configuration demand/evidence required")
        for row in devices.values():
            if any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k]) or row[k] <= 0
                   for k in ("compute_capacity", "memory_mib", "cost")):
                raise ValueError("Explicit device capacities and costs required")
        scale = max(d["memory_mib"] for d in devices.values())
        demands = {m: dict(compute_fraction=v["compute_fraction"],
                           memory_fraction=v["memory_mib"] / scale) for m, v in configuration.items()}
        groups = UsherResourcePlanner.groups(demands)
        free = {d: [v["compute_capacity"], v["memory_mib"]] for d, v in devices.items()}
        used, placed = set(), []
        for group in groups:
            group_devices = set()
            heavy_c, heavy_m, balanced = [], [], []
            for m in group:
                c, mem = demands[m]["compute_fraction"], demands[m]["memory_fraction"]
                bucket = heavy_c if c >= 1.2 * mem else heavy_m if mem >= 1.2 * c else balanced
                bucket.append(m)
            key = lambda m: (-(demands[m]["compute_fraction"] + demands[m]["memory_fraction"]), m)
            heavy_c.sort(key=key); heavy_m.sort(key=key)
            order = [m for pair in itertools.zip_longest(heavy_c, heavy_m) for m in pair if m is not None]
            for m in sorted(balanced, key=key):
                at = next((i for i, n in enumerate(order) if key(m) < key(n)), len(order))
                order.insert(at, m)
            for m in order:
                row = configuration[m]
                for replica in range(row["replicas"]):
                    available = []
                    for device, (c, mem) in free.items():
                        if c + 1e-12 >= row["compute_fraction"] and mem + 1e-9 >= row["memory_mib"]:
                            tier = 0 if device in group_devices else 1 if device in used else 2
                            residual = ((c - row["compute_fraction"]) / devices[device]["compute_capacity"]
                                        + (mem - row["memory_mib"]) / devices[device]["memory_mib"])
                            available.append((tier, devices[device]["cost"] if tier == 2 else 0., residual, device))
                    if not available:
                        return dict(status="NO_PACKING_FOUND", placements=placed,
                                    physical_feasibility_certified=False)
                    device = min(available)[-1]
                    free[device][0] -= row["compute_fraction"]
                    free[device][1] -= row["memory_mib"]
                    group_devices.add(device); used.add(device)
                    placed.append(dict(model=m, replica=replica, device=device))
        return dict(status="PLACEMENT_PROPOSAL", placements=placed, groups=groups,
                    device_cost=sum(devices[d]["cost"] for d in used),
                    free=free, physical_feasibility_certified=False)

    @staticmethod
    def enumerate_configurations(options, devices, max_configurations=32):
        """Only enumerate supplied legal BS/RD/resource profiles, with a hard cap."""
        if type(max_configurations) is not int or max_configurations < 1 or not options or any(not v for v in options.values()):
            raise ValueError("Nonempty finite options and positive budget required")
        names = sorted(options)
        total = math.prod(len(options[m]) for m in names)
        results = []
        for rows in itertools.islice(itertools.product(*(options[m] for m in names)), max_configurations):
            config = dict(zip(names, rows))
            results.append(dict(configuration=copy.deepcopy(config),
                                placement=UsherResourcePlanner.placement(config, devices)))
        return dict(configurations=results, original_combinations=total,
                    evaluated_combinations=len(results), domain_complete=len(results) == total)

    @staticmethod
    def order_profiled_deployments(config):
        """Use actual profile demand summaries to order, never prune, candidates.

        SM/DRAM balance is an additional GSDeploy ordering heuristic. It is
        distinct from the compute/capacity packing in placement() above.
        Missing counters retain candidates at the end with an explicit reason.
        """
        ordered = []
        for entry, engine in session(config.catalog).entries.values():
            demands = [engine.service_provider.demand_summary(p)
                       for p in engine.service_provider.profiles]
            known = bool(demands) and all(d.get("sm") is not None and d.get("dram") is not None
                                           for d in demands)
            balance = abs(sum(d["sm"] - d["dram"] for d in demands)) if known else None
            ordered.append(dict(deployment=copy.deepcopy(entry["deployment"]),
                                priority=balance, resource_demand_known=known,
                                reason="PROFILE_SM_DRAM_BALANCE" if known else "UNKNOWN_RESOURCE_DEMAND_RETAINED"))
        return sorted(ordered, key=lambda row: (row["priority"] is None,
                                               row["priority"] or 0., canonical(row["deployment"])))


def fixed_deployment_lp(config, deployment, *, seconds=1.):
    """J3O fixed-deployment LP adaptation; original rho/g and simplex constraints.

    Q and isolated L/P form an affine proposal cost. T, actual empirical P95,
    phase interference, M and E are NOT approximated for final acceptance.
    """
    from .quality import task_quality
    from .trace import make_trace

    c = config
    from .normalization import require_ready
    require_ready(c.catalog, c.workload, c.latency_reference_ms, c.horizon_seconds, c.gpu_memory_mib)
    s = session(c.catalog)
    found = s.entries.get(canonical(deployment))
    if found is None:
        return None
    entry, _ = found
    edges = [(t, m) for t in sorted(c.workload) for m in sorted(deployment)
             if t in c.catalog["models"][m]["supported_tasks"]]
    if any(not any(t == task for t, m in edges) for task in c.workload):
        return None
    costs, pressure = [], []
    trace = make_trace(c.workload, c.horizon_seconds, c.random_seed, c.arrival_process)
    for task, model in edges:
        ms = entry["capacity_service_ms"].get(model, {}).get(task)
        if ms is None:
            return None
        loss = 0.
        if c.weights.get("Q", 0):
            loss, _ = task_quality(c.catalog, task, deployment, {model: c.workload[task]}, trace, c.random_seed)
            if loss is None:
                return None
        total = sum(c.workload.values())
        worker_ms = ms * deployment[model] / c.catalog["capacity_factors"][model][str(deployment[model])]
        costs.append((c.weights.get("Q", 0) * loss
                      + (c.weights["L"] + c.weights["P"]) * worker_ms / c.latency_reference_ms[task]) / total)
        pressure.append(ms / 1000)
    tasks, models = sorted(c.workload), sorted(deployment)
    equality = np.array([[float(t == task) for t, m in edges] for task in tasks])
    capacity = np.array([[pressure[i] if m == model else 0. for i, (t, m) in enumerate(edges)]
                         for model in models])
    fit = linprog(costs, A_ub=capacity,
                  b_ub=[.8 * c.catalog["capacity_factors"][m][str(deployment[m])] for m in models],
                  A_eq=equality, b_eq=[c.workload[t] for t in tasks], bounds=(0, None),
                  method="highs", options={"time_limit": seconds})
    if not fit.success:
        return None
    routing = {t: {} for t in tasks}
    for (t, m), value in zip(edges, fit.x):
        routing[t][m] = max(0., float(value))
    # Repair floating-point equality only; do not quantize continuous routes.
    for t, row in routing.items():
        total = math.fsum(row.values())
        if total <= 0:
            return None
        routing[t] = {m: v * c.workload[t] / total for m, v in row.items()}
    return dict(routing=routing, objective=float(fit.fun),
                capacity_dual=fit.ineqlin.marginals.tolist(),
                scope="affine proposal only; full FIFO acceptance required")


class J3OInspiredOptimizer:
    """Joint MILP seed, alternating greedy deployment and continuous routing.

    A bounded single-edit deployment neighbourhood replaces the paper's
    hierarchical submodular onloading problem. Greedy gain per added memory
    chooses proposals; only strict improvements in full feasible U are kept.
    There is no J3O approximation bound, LP optimum claim for U, or MADS claim.
    """

    def __init__(self, config, *, max_evaluations=32, time_limit_seconds=10.,
                 seed_seconds=2., max_rounds=3):
        if config.catalog.get("performance_mode") != MODE or config.search_policy != "nominal":
            raise ValueError("J3O adaptation requires explicit nominal paper FIFO profiles")
        if any(type(v) is not int or v < 1 for v in (max_evaluations, max_rounds)):
            raise ValueError("Positive evaluation and round budgets required")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
               for v in (time_limit_seconds, seed_seconds)):
            raise ValueError("Positive finite time budgets required")
        self.config = config
        self.max_evaluations, self.time_limit = max_evaluations, time_limit_seconds
        self.seed_seconds, self.max_rounds = seed_seconds, max_rounds

    def solve(self):
        from .system import DeploymentSystem

        start = time.perf_counter()
        deadline = start + self.time_limit
        c = self.config
        before_session = session(c.catalog)
        before = before_session.calls
        before_engines = before_session.engine_calls
        history = []
        best = None
        evaluations = lp_calls = 0

        def expired():
            return time.perf_counter() >= deadline or before_session.calls - before >= self.max_evaluations

        def evaluate(x, routing, stage):
            nonlocal best, evaluations
            if expired():
                return None
            p = before_session.evaluate(x, routing, c.workload, c.gpu_memory_mib,
                c.latency_reference_ms, c.weights, **DeploymentSystem(c).context())
            evaluations += 1
            row = dict(stage=stage, deployment=copy.deepcopy(x), routing=copy.deepcopy(routing),
                       U=p["U"], eligible=p["eligible"], reason=p["reason"],
                       elapsed_seconds=time.perf_counter() - start)
            history.append(row)
            candidate = dict(deployment=copy.deepcopy(x), routing=copy.deepcopy(routing),
                             prediction=p, score=p["U"], objective="nominal")
            if p["eligible"] and (best is None or p["U"] > best["score"] + 1e-12):
                best = candidate
            return candidate

        # Both time and number of full scenario calls are charged to this run.
        # The MILP iteration cap bounds seed evaluations; it may return an
        # incomplete finite-domain search, which is retained in seed metadata.
        seed_iterations = min(c.milp_max_iterations, self.max_evaluations // 3)
        if seed_iterations:
            seed_config = replace(c, solver="milp", milp_max_iterations=seed_iterations,
                                  milp_time_limit_seconds=min(self.seed_seconds, self.time_limit),
                                  milp_capacity_witness=False)
            seed = DeploymentSystem(seed_config).plan()
        else:
            seed = dict(winner=None, status="SEED_SKIPPED_SCENARIO_BUDGET", metrics={})
        best = copy.deepcopy(seed["winner"])
        seed_winner = copy.deepcopy(best)
        if best:
            history.append(dict(stage="joint_milp_seed", U=best["score"],
                                elapsed_seconds=time.perf_counter() - start))
        ordering = UsherResourcePlanner.order_profiled_deployments(c)
        deployments = [row["deployment"] for row in ordering]
        for round_index in range(self.max_rounds):
            if expired():
                break
            previous = best["score"] if best else -math.inf
            center = copy.deepcopy(best)
            if center:
                fixed = center["deployment"]
                lp_calls += 1
                proposal = fixed_deployment_lp(c, fixed, seconds=max(.001, min(1., deadline - time.perf_counter())))
                if proposal:
                    # Include the LP endpoint and interior interpolants. The
                    # warm start itself is retained, even if every proposal loses.
                    for amount in (1., .5, .25):
                        route = {t: {m: (1 - amount) * center["routing"][t].get(m, 0)
                                         + amount * proposal["routing"][t].get(m, 0)
                                     for m in fixed if t in c.catalog["models"][m]["supported_tasks"]}
                                 for t in c.workload}
                        evaluate(fixed, route, "fixed_deployment_lp")
                # Exact endpoints and small transfers allow the predictor to
                # reject a proxy optimum or find a different FIFO allocation.
                route_center = copy.deepcopy(best["routing"])
                for task in sorted(c.workload):
                    models = [m for m in fixed if task in c.catalog["models"][m]["supported_tasks"]]
                    for target in models:
                        route = copy.deepcopy(route_center)
                        route[task] = {m: c.workload[task] if m == target else 0. for m in models}
                        evaluate(fixed, route, "exact_endpoint")
                    for step in (.1, .03):
                        for source, target in itertools.permutations(models, 2):
                            if expired():
                                break
                            route = copy.deepcopy(best["routing"])
                            amount = min(step * c.workload[task], route[task].get(source, 0))
                            if amount:
                                route[task][source] -= amount
                                route[task][target] = route[task].get(target, 0) + amount
                                evaluate(fixed, route, "continuous_transfer")
            if expired():
                break
            center = copy.deepcopy(best)
            greedy = []
            center_memory = (c.catalog["safe_factor"] * sum(
                c.catalog["memory_table_mib"][m][str(n)] for m, n in center["deployment"].items())
                if center else 0.)
            # Only a proposal ordering: additive residency is not a valid
            # memory constraint/bound for the whole-context predictor.
            reference_value = 0.
            if center:
                lp_calls += 1
                reference = fixed_deployment_lp(c, center["deployment"],
                    seconds=max(.001, min(1., deadline - time.perf_counter())))
                if reference:
                    reference_value = -reference["objective"] - c.weights["M"] * center_memory / c.gpu_memory_mib
            for x in deployments:
                if expired():
                    break
                if center and sum(center["deployment"].get(m, 0) != x.get(m, 0)
                                  for m in set(x) | set(center["deployment"])) != 1:
                    continue
                # Look ahead one LP routing step for a newly added model. A
                # zero-flow addition alone cannot expose its future benefit.
                lp_calls += 1
                proposal = fixed_deployment_lp(c, x, seconds=max(.001, min(1., deadline - time.perf_counter())))
                if proposal:
                    memory = c.catalog["safe_factor"] * sum(
                        c.catalog["memory_table_mib"][m][str(n)] for m, n in x.items())
                    value = -proposal["objective"] - c.weights["M"] * memory / c.gpu_memory_mib
                    priority = (value - reference_value) / max(memory - center_memory, 1.)
                    greedy.append((priority, x, proposal["routing"]))
            # USHER ordering breaks ties. Greedy gain per added memory now
            # actually determines which proposals get the remaining expensive
            # FIFO budget; final acceptance always uses complete feasible U.
            for priority, x, route in sorted(greedy, key=lambda row: -row[0]):
                if expired():
                    break
                candidate = evaluate(x, route, "greedy_deployment_lp_rollout")
                if candidate:
                    history[-1]["proposal_gain_per_added_memory"] = priority
            if best is None or best["score"] <= previous + 1e-12:
                break
        return dict(algorithm="J3OInspiredOptimizer", winner=best, seed_winner=seed_winner,
                    initial_joint_result=seed, history=history,
                    deployment_ordering=ordering,
                    elapsed_seconds=time.perf_counter() - start,
                    full_prediction_calls=before_session.calls - before,
                    event_engine_calls=before_session.engine_calls - before_engines,
                    route_evaluations=evaluations, lp_calls=lp_calls,
                    milp_calls=seed["metrics"].get("milp_calls", 0),
                    status="BUDGET_LIMIT" if expired() else "BOUNDED_ALTERNATION_COMPLETED",
                    optimality_certified=False, native_eligible=False,
                    final_gain_eligible=False,
                    scope="J3O-inspired finite candidate adaptation; no paper approximation guarantee")
