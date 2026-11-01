"""Joint finite-grid MILP with valid lower bounds and exact predictor feedback.

One-hot measured replica states and task route/count columns form the master.
Its epigraph lower-bounds weighted memory plus actual routed service mean/P95.
Whole-plan FIFO costs yield identity cuts; invalid plans yield no-good cuts.
No convexity assumption or local proxy optimality is transferred to full U.
"""

import itertools
import math
import time
import warnings
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from .prediction import predict_scenarios
from .performance import service_lower_ms
from .surrogate_bounds import surrogate_bounds
from .routing import Router
from .trace import expand_routes, make_trace, trace_digest
from .route_domain import compositions, fraction_rows, domain_description, validate_pair_denominators
from .config import TASKS
from .coupled_mode import is_coupled, bound_contract


def solve_master(*args, **kwargs):   # 调用milp求解器
    """Limit HiGHS to one thread, including its scheduler overhead in timings."""
    kwargs.setdefault("options", {})["threads"] = 1
    with warnings.catch_warnings():
        # scipy forwards this documented HiGHS option but warns it is not in
        # scipy's smaller option whitelist. Do not hide other solver warnings.
        warnings.filterwarnings("ignore", message="Unrecognized options detected:.*threads.*")
        return milp(*args, **kwargs)


def residual_identity_cut(base_row, theta, selected, blocks, residual):  # 构造精确候选的反馈约束
    """Exact identity residual above an already valid global cost floor.

    ``base_row`` contains -B(z). At the scored integer identity this adds
    its nonnegative residual; at every other complete integer identity the
    correction is <=0, so the shared floor dominates it. The fractional
    relaxation is tighter than the original full-cost big-M identity cut.
    """
    entries = dict(base_row)
    entries[theta] = entries.get(theta, 0.0) + 1.0
    for i in selected:
        entries[i] = entries.get(i, 0.0) - residual
    return entries, -residual * (blocks - 1)


class JointMilpOptimizer:
    def __init__(self, data, rates, budget, refs, weights, **context):   # 初始化
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
        self.pair_denominators = context.pop("route_pair_denominators_by_task", {})
        validate_pair_denominators(self.pair_denominators, TASKS)
        self.context = context
        self.surrogate = surrogate_bounds(data, self.rates, **context)
        self.coupled = is_coupled(data)
        if self.coupled:
            self.use_witness = False
        self.separable_service = (not self.coupled and (data.get("performance_mode", "independent") == "independent"
                                  or self.surrogate is not None))
        self.profiled_memory = (self.coupled or data.get("memory_mode") == "profiled"
                               or (data.get("performance_mode") in {"calibrated_surrogate", "calibrated_replay_v2", "replay_transfer_v1"}
                                   and self.surrogate is None))
        self.trace = make_trace(
            self.rates,
            context.get("horizon", 120.0),
            context.get("seed", 42),
            context.get("arrival_process", "deterministic_periodic"),
        )
        self.counts = {t: sum(k == t for _, k in self.trace) for t in self.rates}
        self.task_offsets = {t: [a for a, task in self.trace if task == t] for t in self.rates}
        self.task_input_ids = (self.surrogate.task_input_ids(self.trace, context.get("seed", 42))
                               if self.surrogate else None)
        self.best = self.diagnostic = None
        self.unknown_lower = math.inf
        self.lower = 0.0
        self.history = []
        self.pattern_cache = {}
        self.witness = None
        self.metrics = dict(
            milp_calls=0,
            task_columns=0,
            column_memory_pruned=0,
            full_prediction_calls=0,
            feedback_cuts=0,
            route_feedback_cuts=0,
            infeasible_cuts=0,
            unknown_candidates=0,
            capacity_probe_calls=0,
            service_pattern_evaluations=0,
            isolated_queue_bounds=0,
            residual_identity_cuts=0,
            identity_full_cost_sum=0.0,
            identity_residual_sum=0.0,
            evaluated_identity_exclusions=0,
        )
        self.diagnostic_mode = (weights.get("E", 0) > 0 and not data.get("energy_profiles")
                                and self.surrogate is None and not self.coupled)
        self.unknown_candidates = []
        self.first_feasible_seconds = None
        self.best_found_seconds = None

    def expired(self):
        return time.monotonic() - self.start >= self.time_limit

    def family_service(self, model, task, count):
        if self.coupled:
            return 0.0
        if self.surrogate:
            return self.surrogate.family_service(model, task, count)
        return service_lower_ms(self.data, model, task)

    def state_memory(self, model, count):
        if self.profiled_memory:
            return 0.0
        factor = self.surrogate.memory_multiplier if self.surrogate else 1.0
        return factor * self.data["safe_factor"] * self.data["memory_table_mib"][model][str(count)]

    def pressure_service(self, model, task, count):
        if self.coupled:
            return 0.0
        if self.surrogate:
            return self.surrogate.pressure_service(model, task, count)
        return self.family_service(model, task, count)

    # 计算候选方案的目标成本下界 剪枝
    def floor(self, task, x, row):   
        quality_floor = 0.0
        if self.weights.get("Q", 0) > 0:
            from .quality import task_quality

            loss, _ = task_quality(
                self.data, task, x, row, self.trace,
                seed=self.context.get("seed", 42),
            )
            if loss is not None:
                quality_floor = (
                    self.weights["Q"] * self.rates[task]
                    / math.fsum(self.rates.values()) * loss
                )
        if self.surrogate:
            quality_floor += self.weights.get("E", 0) * (
                self.surrogate.route_energy(task, x, row)
                + self.rates[task] / math.fsum(self.rates.values())
                * self.surrogate.constant_energy
            )
        if not self.separable_service:
            return quality_floor
        stress = 2.0 if self.policy == "robust" else 1.0
        alpha = self.rates[task] / math.fsum(self.rates.values()) / self.refs[task]
        aliases, routes = expand_routes(x, {task: row})
        router = Router(routes)
        if self.counts[task] <= 1024:
            # Removing other tasks can only reduce FIFO waits on each worker.
            # The resulting task mean, P95 and late count are valid lower bounds
            # and include burst queues that a no-wait service floor misses.
            clocks = {a: 0.0 for a in aliases}
            latencies = []
            late = 0
            horizon = self.context.get("horizon", 120.0)
            for index, offset in enumerate(self.task_offsets[task]):
                alias = router.choose(task)
                model = aliases[alias]
                worker_ms = (self.surrogate.request_service(model, task, x[model], self.task_input_ids[task][index])
                             if self.task_input_ids is not None else self.family_service(model, task, x[model]))
                service = (stress * worker_ms
                           * x[model] / self.data["capacity_factors"][model][str(x[model])]
                           / 1000)   # 任务的服务时间
                # offset 到达时间
                finish = max(offset, clocks[alias]) + service   # 任务t的完成时间
                clocks[alias] = finish   # 记录上一次完成的时刻
                latencies.append((finish - offset) * 1000)
                late += finish > horizon
            self.metrics["isolated_queue_bounds"] += 1
            return quality_floor + alpha * (
                self.weights["L"] * float(np.mean(latencies))
                + self.weights["P"] * float(np.percentile(latencies, 95))
            ) + self.weights["T"] * late / len(self.trace)
        if self.task_input_ids is not None:
            # Preserve the exact sample-to-alias assignment. Removing waits
            # leaves each request's service, hence also its empirical P95, as
            # a lower bound. Counting only aliases would lose input variation.
            values = []
            for index, sample in enumerate(self.task_input_ids[task]):
                if index % 256 == 0 and self.expired():
                    return None
                model = aliases[router.choose(task)]
                values.append(stress * self.surrogate.request_service(model, task, x[model], sample)
                              * x[model] / self.data["capacity_factors"][model][str(x[model])])
            self.metrics["service_pattern_evaluations"] += 1
            return quality_floor + alpha * (
                self.weights["L"] * float(np.mean(values))
                + self.weights["P"] * float(np.percentile(values, 95)))
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
                * self.family_service(aliases[a], task, x[aliases[a]])
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
        return quality_floor + alpha * (
            self.weights["L"] * float(counts @ values / self.counts[task])
            + self.weights["P"] * float(p95)
        )
    
    # 快速找一个可能可行的资源分配
    def capacity_witness(self, choices):   
        """One finite feasible-flow witness, not a utility decision or proof."""
        if (
            not self.use_witness
            or self.profiled_memory
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
            i: (0.0 if self.profiled_memory else
                self.data["safe_factor"] * self.data["memory_table_mib"][m][str(n)])
            for i, (m, n) in enumerate(states)
        }
        if not self.profiled_memory:
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
        fit = solve_master(
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

    # 构造任务路由和副本候选列
    def columns(self, choices, K):   
    # K 控制任务路由的离散网格 可以理解为以1/K为间隔 即分母
        columns = []
        stress = 2.0 if self.policy == "robust" else 1.0
        for task, rate in self.rates.items():   # 遍历所有任务
            models = [
                m
                for m in choices
                if task in self.data["models"][m]["supported_tasks"]   # 检查模型m是否支持当前任务
                and any(n > 0 for n in choices[m])
            ]
            

            # 生成所有允许的路由比例 再枚举模型副本数
            for fractions in fraction_rows(models, K, self.pair_denominators.get(task, ())):
                active = tuple(m for m, _ in fractions)
                row = {m: rate * f.numerator / f.denominator for m, f in fractions}
                for counts in itertools.product(    # 求笛卡尔积 枚举所有副本数量
                    *(tuple(n for n in choices[m] if n) for m in active)
                ):
                    if self.expired():
                        self.metrics["task_columns"] = len(columns)
                        return None
                    x = dict(zip(active, counts))
                    # A complete deployment containing this column
                    # cannot use less than its active-state memory
                    # lower bound. Profiled nonadditive memory has
                    # no such standalone bound and must skip this.
                    if (not self.profiled_memory and
                        math.fsum(self.state_memory(m, n) for m, n in x.items())
                            > self.budget + 1e-6):
                        self.metrics["column_memory_pruned"] += 1
                        continue
                    pressure = {   # 近似负载压力
                        m: stress
                        * row[m]
                        * self.pressure_service(m, task, x[m])
                        / 1000
                        / self.data["capacity_factors"][m][str(x[m])]
                        for m in active
                    }
                    if max(pressure.values()) > 0.8 + 1e-9:
                        continue
                    if len(columns) >= self.column_limit:
                        self.metrics["task_columns"] = len(columns)
                        return None
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
                if (not self.profiled_memory and
                    math.fsum(self.state_memory(m, n) for m, n in x.items())
                        > self.budget + 1e-6):
                    self.metrics["column_memory_pruned"] += 1
                    continue
                if self.expired() or len(columns) >= self.column_limit:
                    return None
                floor = self.floor(task, x, row)
                if floor is None:
                    return None
                pressure = {
                    m: stress
                    * row[m]
                    * self.pressure_service(m, task, x[m])
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

    # 搜索的核心算法及预测器反馈
    def solve(     
        self, 
        replica_options=None, 
        deployment=None, 
        K=1, 
        *,     # 表示后面这些参数只能使用关键字方式传入
        exclude_deployments=None, 
        exclude_plans=None, 
        no_split=False
        ):

        self.start = time.monotonic()
        self.no_split = bool(no_split)
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
        states = [(m, n) for m in choices for n in choices[m]]   # 枚举状态
        state_index = {p: i for i, p in enumerate(states)}
        if K >= 2:
            self.capacity_witness(choices)
        columns = self.columns(choices, K)
        if columns is None:
            return self.result(K, "MASTER_BUILD_BUDGET_LIMIT")
        if no_split:
            columns = [c for c in columns if len(c["routing"]) == 1]
            self.metrics["task_columns"] = len(columns)
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
        memory = {i: self.state_memory(m, n) for i, (m, n) in enumerate(states)}
        if not self.profiled_memory:
            add(memory, hi=self.budget)
        for t in self.rates:
            add(
                {ns + j: 1.0 for j, c in enumerate(columns) if c["task"] == t}, 1.0, 1.0
            )
        for excluded in exclude_deployments or []:
            identity = [state_index.get((m, excluded.get(m, 0))) for m in choices]
            if all(i is not None for i in identity):
                add({i: 1.0 for i in identity}, hi=len(identity) - 1)
        for excluded in exclude_plans or []:
            excluded_x = excluded["deployment"]
            identity = [state_index.get((m, excluded_x.get(m, 0))) for m in choices]
            for task in self.rates:
                target = excluded["routing"].get(task, {})
                match = next((ns + j for j, c in enumerate(columns)
                              if c["task"] == task
                              and all(excluded_x.get(m) == n for m, n in c["deployment"].items())
                              and set(c["routing"]) == {m for m, q in target.items() if q > 0}
                              and all(math.isclose(q, target[m], rel_tol=1e-10, abs_tol=1e-12)
                                      for m, q in c["routing"].items())), None)
                identity.append(match)
            if all(i is not None for i in identity):
                add({i: 1.0 for i in identity}, hi=len(identity) - 1)
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
        for m in choices:    # 要求分配给模型的总负载压力 <= 0.8
            pressure = {
                ns + j: c["pressure"][m]
                for j, c in enumerate(columns)
                if m in c["pressure"]
            }
            pressure.update({state_index[m, n]: -0.8 for n in choices[m] if n})
            add(pressure, hi=0.0)
        state_cost = {
            i: self.weights["M"] * value / self.budget
               + (self.weights.get("E", 0) * self.surrogate.state_energy(states[i][1])
                  if self.surrogate else 0.0)
            for i, value in memory.items()
        }
        floor_row = {i: -value for i, value in state_cost.items()}
        floor_row.update({ns + j: -c["floor"] for j, c in enumerate(columns)})
        add({**floor_row, theta: 1.0}, lo=0.0)
        objective = np.zeros(N)
        objective[theta] = 1.0  # 目标函数系数
        integrality = np.ones(N)
        integrality[theta] = 0
        bounds = Bounds(np.zeros(N), np.r_[np.ones(theta), np.inf])  # 取值范围
        blocks = len(choices) + len(self.rates)
        evaluated = set()
        numerically_excluded = set()
        status = "MILP_BUDGET_LIMIT"
        complete = False

        # 迭代循环milp至最大次数
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

            # 把约束矩阵转换成 SciPy 求解器适合处理的稀疏矩阵格式
            matrix = coo_matrix((vv, (rr, cc)), shape=(len(rows), N)).tocsc()
            remaining = self.time_limit - (time.monotonic() - self.start)
            if remaining <= 0:
                break

            #  真正调用求解器
            fit = solve_master(  
                objective,    # 最小化的目标函数系数
                integrality=integrality,   # 哪些变量必须是整数
                bounds=bounds,   # 每个变量的取值范围
                constraints=LinearConstraint(matrix, lows, highs),   # 线性约束条件
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
            if (
                incumbent and self.weights.get("Q", 0) > 0
                and incumbent["prediction"].get("Q_norm") is None
            ):
                incumbent = self.best
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

            # 提取当前候选方案
            x = {m: n for i, (m, n) in enumerate(states) if i in selected and n}
            selected_columns = [columns[i - ns] for i in selected if i >= ns]
            routing = {c["task"]: c["routing"] for c in selected_columns}
            candidate_floor = sum(state_cost[i] for i in selected if i < ns)
            candidate_floor += sum(c["floor"] for c in selected_columns)
            if selected in evaluated:
                # HiGHS may accept an identity feedback violation within its
                # feasibility tolerance although our requested gap is smaller.
                # We already evaluated this identity: exclude it from the
                # remaining master while retaining its incumbent or unknown
                # bound. This is exact finite-domain bookkeeping, not rounding
                # a numerical gap into an optimality certificate.
                if selected in numerically_excluded:
                    status = "MILP_NUMERICAL_UNRESOLVED"
                    break
                add({i: 1.0 for i in selected}, hi=blocks - 1)
                numerically_excluded.add(selected)
                self.metrics["evaluated_identity_exclusions"] += 1
                continue

            evaluated.add(selected)   # 把milp选出来的部署组合加到evaluated里

            #  调用预测器
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
            # 检查部署方案是否满足全部必要约束
            valid = p["constraints_eligible"] and (
                self.policy != "robust" or p["stress_constraints_valid"]
            )
            unknown_performance = p.get("performance_unknown", False) or p.get("memory_unknown", False)
            score = p["scenario_U_low"] if self.policy == "robust" else p["U"]
            if unknown_performance:
                self.unknown_lower = min(self.unknown_lower, candidate_floor)
                self.metrics["unknown_candidates"] += 1
                self.unknown_candidates.append(dict(
                    deployment=x, routing=routing,
                    lower_cost=candidate_floor, reason=p["reason"],
                ))
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
                # Complete identity feedback must use the same memory metric
                # as the predictor, not the relaxed standalone-state bound.
                memory_cost = self.weights["M"] * p["M_norm"]
                if self.data.get("performance_mode", "independent") == "independent":
                    # In this predictor, fixed task route/count columns fix all
                    # FIFO queues even if idle residents change. Transfer only
                    # routed Q/T/L/P, never nonadditive energy, across deployments.
                    route_cost = delay_cost
                    if p.get("Q_contribution") is not None:
                        route_cost -= p["Q_contribution"]
                    route_indices = [i for i in selected if i >= ns]
                    entries = {
                        i: -self.weights["M"] * v / self.budget
                        for i, v in memory.items()
                    }
                    entries.update({i: -route_cost for i in route_indices})
                    add({**entries, theta: 1.0}, lo=-route_cost * (len(self.rates) - 1))
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
                    quality_missing = self.weights.get("Q", 0) > 0 and p.get("Q_norm") is None
                    if self.diagnostic_mode and not quality_missing:
                        cost = -diagnostic_score
                    else:
                        self.unknown_lower = min(
                            self.unknown_lower, max(candidate_floor, memory_cost + delay_cost)
                        )
                        self.metrics["unknown_candidates"] += 1
                        self.unknown_candidates.append(dict(
                            deployment=x, routing=routing,
                            lower_cost=max(candidate_floor, memory_cost + delay_cost),
                            reason=p["reason"],
                        ))
                        add({i: 1.0 for i in selected}, hi=blocks - 1)
                        cost = None
                else:
                    # 如果当前方案更好 就更新最优方案
                    if self.best is None or score > self.best["score"]:
                        elapsed = time.monotonic() - self.start
                        if self.first_feasible_seconds is None:
                            self.first_feasible_seconds = elapsed
                        self.best_found_seconds = elapsed
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
                    # Reuse B(z): at this identity B+residual equals cost;
                    # other complete integer identities get a nonpositive
                    # correction, dominated by the existing global floor.
                    # 添加残差约束
                    residual = max(0.0, cost - candidate_floor)
                    cut, lower = residual_identity_cut(
                        floor_row, theta, selected, blocks, residual
                    )
                    add(cut, lo=lower)
                    self.metrics["feedback_cuts"] += 1
                    self.metrics["residual_identity_cuts"] += 1
                    self.metrics["identity_full_cost_sum"] += cost
                    self.metrics["identity_residual_sum"] += residual  # 残差修正项
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
                status = (
                    "INCOMPLETE_QUALITY_DOMAIN"
                    if self.weights.get("Q", 0) > 0
                    and self.diagnostic["prediction"].get("Q_norm") is None
                    else "DIAGNOSTIC_ONLY_NO_ENERGY"
                )
            else:
                status = (
                    "INCOMPLETE_PERFORMANCE_DOMAIN"
                    if self.metrics["unknown_candidates"]
                    else "NO_FEASIBLE_GRID_PLAN"
                )
        return self.result(K, status, complete)

    def result(self, K, status, complete=False):   # 把结果整理成dict
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
                **domain_description(K, self.pair_denominators),
                coupled_bounds=bound_contract(self.data),
                replicas="measured catalog counts only",
                routing=(
                    ("base grid plus explicit pair grids and recorded capacity witness rows"
                     if self.pair_denominators else "union of grids with denominators 1..K plus recorded capacity witness rows")
                    if self.witness
                    else ("base grid plus explicit pair grids" if self.pair_denominators
                          else "union of grids with denominators 1..K")
                ),
                K=K,
                no_split=self.no_split,
                capacity_boundary_witnesses=self.witness is not None,
                capacity_witness=self.witness,
                continuous_optimality=False,
                service_bound=(
                    "isolated task FIFO with exact frozen per-input service for <=1024 requests; exact per-input no-wait multiset otherwise; unassigned minimum service; pool means only for rho"
                    if self.surrogate and self.surrogate.minimum_services is not None else
                    "isolated task FIFO queue floor for <=1024 task requests; otherwise cached no-wait service floor"
                    if self.separable_service
                    else "zero service relaxation for unknown colocation"
                ),
                quality_bound="exact routed frozen quality when available; otherwise zero relaxation",
                memory_bound=("zero: coupled placeholder has no complete-context memory evidence" if self.coupled
                              else "zero until exact whole-plan profile evaluation" if self.profiled_memory
                              else "validated frozen predictor memory multiplier times additive residency"
                              if self.surrogate else "additive measured replica residency proxy"),
                energy_bound=("exact nonnegative frozen resource intercept/state/known-column decomposition; zero busy for missing paths"
                              if self.surrogate else "zero relaxation"),
                surrogate_bounds_version=(self.surrogate.spec["version_sha256"]
                                          if self.surrogate else None),
                surrogate_bounds_schema=(self.surrogate.spec["schema"] if self.surrogate else None),
                rho_bound=("frozen per-input pool-mean contract; zero relaxation for missing paths"
                           if self.surrogate and self.surrogate.pressure_services is not None
                           else "fixed family service necessary capacity" if self.separable_service
                           else "zero service relaxation"),
                missing_surrogate_paths=("zero relaxation only; full predictor retains unknown evidence"
                                         if self.surrogate else None),
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
            first_feasible_seconds=self.first_feasible_seconds,
            best_found_seconds=self.best_found_seconds,
            proof_completed_seconds=(time.monotonic() - self.start if certified else None),
            unresolved_candidates=self.unknown_candidates,
        )
