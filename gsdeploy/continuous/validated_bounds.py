"""Fixed-route task/count columns using the unchanged validated Joint bounds.

No standalone measurements are promoted to co-resident physical bounds. Only
the exact built-in oracle types may opt into this mathematical predictor proof.
The task columns include only positively routed models, preserving arbitrarily
small positive flow and idle resident deployment states in separate y columns.
"""

import itertools
import math
import time

from ..milp import JointMilpOptimizer
from .contracts import BudgetExhausted, digest
from .predictor_adapter import LegacyPredictorOracle


def guard(value):
    return 1e-8 * max(1.0, abs(value))


class ValidatedRouteBounds:
    def __init__(self, evaluator):
        from .immutable import SessionPredictorOracle
        self.e = e = evaluator
        self.optimizers, self.cache = [], {}
        self.available = False
        if type(e.oracle) not in {LegacyPredictorOracle, SessionPredictorOracle}:
            return
        c = e.problem.config
        from ..normalization import is_v2
        if is_v2(c.catalog):
            return  # These calibrated energy columns use historical units.
        cls = (e.oracle.session.module("milp").JointMilpOptimizer
               if type(e.oracle) is SessionPredictorOracle else JointMilpOptimizer)
        data = e.oracle._config.catalog if type(e.oracle) is SessionPredictorOracle else c.catalog
        for trace in e.problem.train_traces:
            e.budget.check()
            opt = cls(data, c.workload, c.gpu_memory_mib, c.latency_reference_ms, c.weights,
                      horizon=c.horizon_seconds, seed=trace.seed, arrival_process=c.arrival_process,
                      drain_seconds=c.drain_seconds, energy_reference_watts=c.energy_reference_watts,
                      startup_amortization_seconds=c.startup_amortization_seconds,
                      search_policy=c.search_policy)
            if opt.surrogate is None or opt.coupled:
                return
            # Explicit traces may differ from make_trace: every relaxation must
            # use exactly the declared arrivals, input sample sequence and seed.
            opt.trace = trace.arrivals
            opt.counts = {t: sum(task == t for _, task in trace.arrivals) for t in c.workload}
            opt.task_offsets = {t: [a for a, task in trace.arrivals if task == t] for t in c.workload}
            opt.task_input_ids = opt.surrogate.task_input_ids(trace.arrivals, trace.seed)
            opt.start = time.monotonic()
            self.optimizers.append((trace, opt))
        self.available = True

    def task_floor(self, task, deployment, row, deadline):
        key = digest([task, deployment, row])
        if key not in self.cache:
            values = []
            for trace, opt in self.optimizers:
                self.e.budget.check(local_deadline=deadline)
                # Original helper checks its own absolute deadline in long loops.
                opt.time_limit = max(0.0, min(self.e.budget.deadline, deadline or math.inf)
                                     - time.perf_counter())
                opt.start = time.monotonic()
                value = opt.floor(task, deployment, row)
                self.e.budget.check(local_deadline=deadline)
                if value is None or not math.isfinite(value) or value < 0:
                    raise ValueError("BOUND_CONTRACT_VIOLATION")
                values.append((trace.weight, value))
            value = math.fsum(w*v for w, v in values)/math.fsum(w for w, _ in values)
            self.cache[key] = max(0.0, value-guard(value))
        return self.cache[key]

    def extend(self, states, objective, add):
        """Extend y columns by per-task z columns and exact count links."""
        c = self.e.problem.config
        opt = self.optimizers[0][1]
        memory = {(m, n): opt.state_memory(m, n) for m, n in states}
        objective[:] = [max(0.0, c.weights["M"]*memory[m, n]/c.gpu_memory_mib
                           + c.weights.get("E", 0)*opt.surrogate.state_energy(n)
                           - guard(1.0)) for m, n in states]
        add({i: memory[s] for i, s in enumerate(states)}, hi=c.gpu_memory_mib+1e-6+guard(c.gpu_memory_mib))
        return memory

    def columns(self, routing, choices, indices, objective, add, deadline):
        c = self.e.problem.config
        opt = self.optimizers[0][1]
        stress = 2.0 if c.search_policy == "robust" else 1.0
        # Fixed q makes pressure a per-model-state necessary condition.
        for (m, n), index in indices.items():
            self.e.budget.check(local_deadline=deadline)
            if n and stress*math.fsum(row.get(m, 0)*opt.pressure_service(m, n, t)/1000
                                     for t, row in routing.items()) > (
                                         .8+1e-8)*c.catalog["capacity_factors"][m][str(n)]+guard(1.0):
                add({index: 1.0}, hi=0.0)
        for task, row in routing.items():
            active = sorted(m for m, q in row.items() if q > 0)
            zs = []
            links = {(m, n): {} for m in active for n in choices[m]}
            for ns in itertools.product(*(choices[m] for m in active)):
                self.e.budget.check(local_deadline=deadline)
                if 0 in ns:
                    continue
                x = dict(zip(active, ns))
                j = len(objective)
                if j >= c.milp_max_columns:
                    raise BudgetExhausted("COLUMN_LIMIT")
                objective.append(self.task_floor(task, x, row, deadline))
                zs.append(j)
                for m, n in x.items():
                    links[m, n][j] = 1.0
            add({j: 1.0 for j in zs}, 1.0, 1.0)
            for state, entries in links.items():
                add(entries | {indices[state]: -1.0}, 0.0, 0.0)
