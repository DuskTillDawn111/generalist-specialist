"""Route-independent future input scenarios, never the run's hidden RNG state.

Poisson suffixes are conditional on the visible time (independent increments).
Periodic schedules retain their phase rather than restarting at every decision.
Known schedules are a separate, explicitly declared planning input.
"""

from __future__ import annotations

import copy
import math
import random

from ..resources import identity
from ..forecast_errors import UnknownEvidence


def number(value, name, *, positive=False):
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"Invalid {'positive' if positive else 'nonnegative'} {name}")
    return value


class FutureWorkload:
    """A declared demand distribution, independent of deployment and routing.

    Windows use seconds relative to phase_start_ns and must be contiguous.
    Input pools contain only public input identities and sampling weights.
    namespace is a forecast seed namespace, NOT the measured run seed.
    """

    def __init__(self, rates, input_pools, *, process="poisson", namespace="forecast-v1",
                 windows=None, declared_arrivals=None, max_arrivals=10000,
                 interarrival_cv=None):
        if not isinstance(rates, dict) or not rates:
            raise ValueError("Nonempty offered rates required")
        self.rates = copy.deepcopy(rates)
        for rate in rates.values():
            number(rate, "offered rate", positive=True)
        if process not in {"poisson", "gamma_renewal", "deterministic_periodic", "declared_schedule"}:
            raise ValueError("Unknown future arrival process")
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("Explicit forecast namespace required")
        if type(max_arrivals) is not int or max_arrivals < 1:
            raise ValueError("Positive integer future arrival budget required")
        if set(input_pools) != set(rates):
            raise ValueError("Input pools must cover all offered tasks")
        self.pools = {}
        for task, pool in input_pools.items():
            if not isinstance(pool, list) or not pool:
                raise ValueError("Nonempty public input pool required")
            rows = []
            for row in pool:
                if isinstance(row, str):
                    row = {"pool_id": row, "weight": 1.}
                if not isinstance(row, dict) or set(row) != {"pool_id", "weight"}:
                    raise ValueError("Future inputs accept pool_id/weight only, no labels or outputs")
                if not isinstance(row["pool_id"], str) or not row["pool_id"]:
                    raise ValueError("Nonempty public pool identity required")
                number(row["weight"], "input weight", positive=True)
                rows.append(copy.deepcopy(row))
            if len({r["pool_id"] for r in rows}) != len(rows):
                raise ValueError("Duplicate future input identity")
            self.pools[task] = rows
        self.process, self.namespace, self.max_arrivals = process, namespace, max_arrivals
        if process == "gamma_renewal":
            if not isinstance(interarrival_cv, dict) or set(interarrival_cv) != set(rates):
                raise ValueError("Gamma arrivals require per-task interarrival CV")
            for value in interarrival_cv.values():
                number(value, "interarrival CV", positive=True)
                if not math.isfinite(value * value) or value * value == 0 or not math.isfinite(1 / (value * value)):
                    raise ValueError("Interarrival CV is outside numeric range")
        elif interarrival_cv is not None:
            raise ValueError("Interarrival CV is only defined for gamma_renewal")
        self.interarrival_cv = copy.deepcopy(interarrival_cv)
        self.windows = copy.deepcopy(windows)
        if windows is not None:
            if process not in {"poisson", "gamma_renewal"} or not isinstance(windows, list) or not windows:
                raise ValueError("Piecewise demand windows require Poisson or Gamma arrivals")
            end = 0.
            for window in windows:
                if set(window) != {"start_seconds", "end_seconds", "rates"}:
                    raise ValueError("Explicit demand window start/end/rates required")
                start = number(window["start_seconds"], "window start")
                stop = number(window["end_seconds"], "window end", positive=True)
                if start != end or stop <= start or set(window["rates"]) != set(rates):
                    raise ValueError("Demand windows must be contiguous and cover every task")
                for rate in window["rates"].values():
                    number(rate, "window rate")
                end = stop
        self.declared = copy.deepcopy(declared_arrivals)
        if process == "declared_schedule":
            if not isinstance(declared_arrivals, list):
                raise ValueError("Explicit known planning schedule required")
            seen = set()
            for row in self.declared:
                if set(row) != {"request_id", "offset_ns", "task", "pool_id"}:
                    raise ValueError("Declared arrivals accept identity/time/task/input only")
                if (not isinstance(row["request_id"], str) or not row["request_id"]
                        or row["request_id"] in seen or type(row["offset_ns"]) is not int or row["offset_ns"] < 0):
                    raise ValueError("Invalid declared arrival identity/time")
                seen.add(row["request_id"])
                if row["task"] not in self.pools or row["pool_id"] not in {r["pool_id"] for r in self.pools[row["task"]]}:
                    raise ValueError("Unknown declared task/input")
        elif declared_arrivals is not None:
            raise ValueError("Actual future schedules cannot enter stochastic online forecasts")

    def suffix(self, phase_start_ns, now_ns, deadline_ns, *, scenario_index=0, decision_id="planning",
               observed_last_arrivals_ns=None):
        if (any(type(v) is not int for v in (phase_start_ns, now_ns, deadline_ns, scenario_index))
                or not phase_start_ns <= now_ns <= deadline_ns or scenario_index < 0):
            raise ValueError("Valid phase times/scenario index required")
        if self.process == "gamma_renewal":
            if observed_last_arrivals_ns is None:
                if now_ns != phase_start_ns:
                    raise UnknownEvidence("GAMMA_RENEWAL_VISIBLE_LAST_ARRIVALS_REQUIRED")
                observed_last_arrivals_ns = dict.fromkeys(self.rates)
            if (not isinstance(observed_last_arrivals_ns, dict)
                    or set(observed_last_arrivals_ns) != set(self.rates)
                    or any(v is not None and (type(v) is not int or not phase_start_ns <= v <= now_ns)
                           for v in observed_last_arrivals_ns.values())):
                raise ValueError("Visible last arrivals must cover every task, no future timestamps")
        elif observed_last_arrivals_ns is not None:
            raise ValueError("Visible renewal state only applies to Gamma arrivals")
        namespace = identity({"namespace": self.namespace, "scenario": scenario_index,
                              "decision": decision_id, "now": now_ns, "deadline": deadline_ns,
                              "renewal_state": observed_last_arrivals_ns})
        rows = []

        def add(offset_ns, task, pool_id, request_id=None):
            if len(rows) >= self.max_arrivals:
                raise UnknownEvidence("FUTURE_ARRIVAL_BUDGET_EXCEEDED_NO_TRUNCATION")
            rows.append({"request_id": request_id or f"future:{namespace}:{task}:{len(rows)}",
                         "arrival_ns": phase_start_ns + offset_ns, "task": task, "pool_id": pool_id})

        if self.process == "declared_schedule":
            for row in self.declared:
                if now_ns < phase_start_ns + row["offset_ns"] < deadline_ns:
                    add(row["offset_ns"], row["task"], row["pool_id"], row["request_id"])
            source = "explicit_known_planning_schedule"
        else:
            horizon = (deadline_ns - phase_start_ns) / 1e9
            now = (now_ns - phase_start_ns) / 1e9
            windows = self.windows or [{"start_seconds": 0., "end_seconds": horizon, "rates": self.rates}]
            if windows[-1]["end_seconds"] < horizon:
                raise ValueError("Demand windows do not cover the full measurement horizon")
            for task in sorted(self.rates):
                arrivals_rng = random.Random(f"{namespace}:arrival:{task}")
                inputs_rng = random.Random(f"{namespace}:input:{task}")
                pool = self.pools[task]
                if self.process == "gamma_renewal":
                    # AlpaServe-inspired rate/CV scenarios.  The renewal clock
                    # advances in integrated declared rate, across ALL windows.
                    # Conditioning on visible age avoids restarting a bursty
                    # non-memoryless process at every routing decision.
                    from scipy.stats import gamma
                    cv = self.interarrival_cv[task]
                    shape, scale = 1 / (cv * cv), cv * cv
                    def intensity_at(seconds):
                        return sum(max(0., min(seconds, w["end_seconds"]) - w["start_seconds"])
                                   * w["rates"][task] for w in windows)
                    last = observed_last_arrivals_ns[task]
                    age = intensity_at(now) - intensity_at((last - phase_start_ns) / 1e9 if last is not None else 0.)
                    survival = gamma.sf(age, a=shape, scale=scale)
                    if not math.isfinite(survival) or survival <= 0:
                        raise UnknownEvidence("GAMMA_CONDITIONAL_TAIL_NUMERICALLY_UNRESOLVED")
                    interval = gamma.isf(max(1e-300, 1 - arrivals_rng.random()) * survival,
                                         a=shape, scale=scale)
                    next_intensity = intensity_at(now) + max(0., interval - age)
                    for window in windows:
                        start, end = max(now, window["start_seconds"]), min(horizon, window["end_seconds"])
                        rate = window["rates"][task]
                        if rate == 0 or end <= start:
                            continue
                        start_i, end_i = intensity_at(start), intensity_at(end)
                        while next_intensity < end_i:
                            offset = start + (next_intensity - start_i) / rate
                            arrival_ns = round(offset * 1e9)
                            if now_ns < phase_start_ns + arrival_ns < deadline_ns:
                                choice = inputs_rng.choices(pool, weights=[r["weight"] for r in pool], k=1)[0]
                                add(arrival_ns, task, choice["pool_id"])
                            increment = arrivals_rng.gammavariate(shape, scale)
                            if increment <= 0 or not math.isfinite(increment):
                                raise UnknownEvidence("GAMMA_RENEWAL_NUMERICALLY_UNRESOLVED")
                            advanced = next_intensity + increment
                            # Gamma CV>1 can produce intervals below a float
                            # ULP. Preserve the arrival (including equal rounded
                            # ns timestamps), rather than dropping tiny gaps.
                            if advanced == next_intensity:
                                advanced = math.nextafter(next_intensity, math.inf)
                                if (advanced - next_intensity) / rate * 1e9 > 1:
                                    raise UnknownEvidence("GAMMA_RENEWAL_SUB_NS_PRECISION_UNRESOLVED")
                            next_intensity = advanced
                    continue
                for window in windows:
                    start, end = max(now, window["start_seconds"]), min(horizon, window["end_seconds"])
                    rate = window["rates"][task]
                    if rate == 0 or end <= start:
                        continue
                    if self.process == "poisson":
                        offset = start + arrivals_rng.expovariate(rate)
                    else:
                        phase = random.Random(f"{self.namespace}:phase:{scenario_index}:{task}").random() / rate
                        offset = phase + max(0, math.floor((now - phase) * rate) + 1) / rate
                    while offset < end:
                        arrival_ns = round(offset * 1e9)
                        if now_ns < phase_start_ns + arrival_ns < deadline_ns:
                            choice = inputs_rng.choices(pool, weights=[r["weight"] for r in pool], k=1)[0]
                            add(arrival_ns, task, choice["pool_id"])
                        offset += arrivals_rng.expovariate(rate) if self.process == "poisson" else 1 / rate
            source = "sampled_known_arrival_distribution_no_actual_future_rng"
        rows.sort(key=lambda r: (r["arrival_ns"], r["task"], r["request_id"]))
        return {"arrivals": rows, "cohort_sha256": identity(rows), "source": source,
                "scenario_index": scenario_index, "actual_future_trace_used": False,
                "declared_future_plan_used": self.process == "declared_schedule",
                "process": self.process, "interarrival_cv": self.interarrival_cv,
                "conditioning_last_arrivals_ns": copy.deepcopy(observed_last_arrivals_ns)}
