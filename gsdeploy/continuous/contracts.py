"""Problem identities, frozen traces and shared, non-renewing budgets."""

import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import time

from ..config import SystemConfig
from ..trace import make_trace, trace_digest


def _exact(value):
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("NONFINITE_IDENTITY")
        return {"float.hex": (0.0 if value == 0 else value).hex()}
    if isinstance(value, dict):
        return {str(k): _exact(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_exact(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(_exact(value), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class BudgetExhausted(Exception):
    pass


class Budget:
    """One ledger for the entire optimization; limits count cache misses only.

    A running oracle is not preempted. Its overrun is included in elapsed time.
    Local inner deadlines never renew the global deadline.
    """

    def __init__(self, seconds=30.0, max_predictions=1000, max_outer=100,
                 max_milp_calls=10000):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("seconds must be positive and finite")
        for value in (max_predictions, max_outer, max_milp_calls):
            if type(value) is not int or value < 1:
                raise ValueError("budget counts must be positive integers")
        self.start = time.perf_counter()
        self.deadline = self.start + seconds
        self.limits = dict(predictions=max_predictions, outer=max_outer,
                           milp=max_milp_calls)
        self.counts = dict(predictions=0, outer=0, milp=0, predictor_leaves=0,
                           cache_hits=0)
        self.times = dict(prediction=0.0, milp=0.0, outer=0.0)

    def check(self, kind=None, local_deadline=None):
        if time.perf_counter() >= min(self.deadline, local_deadline or self.deadline):
            raise BudgetExhausted("TIME_LIMIT")
        if kind and self.counts[kind] >= self.limits[kind]:
            raise BudgetExhausted(kind.upper() + "_LIMIT")

    def consume(self, kind, local_deadline=None):
        self.check(kind, local_deadline)
        self.counts[kind] += 1

    def report(self):
        return dict(full_prediction_calls=self.counts["predictions"],
                    predictor_leaf_calls=self.counts["predictor_leaves"],
                    outer_evaluations=self.counts["outer"],
                    milp_solve_calls=self.counts["milp"],
                    cache_hits=self.counts["cache_hits"],
                    elapsed_seconds=time.perf_counter() - self.start,
                    seconds_by_component=dict(self.times))


@dataclass(frozen=True)
class TraceSpec:
    name: str
    seed: int
    arrivals: tuple
    split: str = "train"
    weight: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "arrivals", tuple(tuple(r) for r in self.arrivals))
        if (not self.name or type(self.seed) is not int or self.split not in
                {"train", "validation", "test"} or not math.isfinite(self.weight)
                or self.weight <= 0):
            raise ValueError("INVALID_TRACE_SPEC")

    @property
    def identity(self):
        return digest(asdict(self))


class ContinuousProblem:
    def __init__(self, config, traces=None, replica_options=None):
        self.config = copy.deepcopy(config)
        c = self.config
        from ..normalization import require_ready
        require_ready(c.catalog, c.workload, c.latency_reference_ms, c.horizon_seconds, c.gpu_memory_mib)
        self.traces = tuple(copy.deepcopy(traces) if traces is not None else [
            TraceSpec("train", c.random_seed, make_trace(c.workload,
                c.horizon_seconds, c.random_seed, c.arrival_process))])
        if not self.traces or not any(t.split == "train" for t in self.traces):
            raise ValueError("TRAIN_TRACES_REQUIRED")
        if len({t.name for t in self.traces}) != len(self.traces):
            raise ValueError("DUPLICATE_TRACE_NAME")
        # Same realization must not leak between train/validation/test splits.
        realizations = {}
        for t in self.traces:
            key = (t.seed, trace_digest(t.arrivals))
            if key in realizations and realizations[key] != t.split:
                raise ValueError("TRACE_SPLIT_LEAKAGE")
            realizations[key] = t.split
            previous = -math.inf
            counts = dict.fromkeys(c.workload, 0)
            for offset, task in t.arrivals:
                if (task not in counts or not math.isfinite(offset) or
                        offset < 0 or offset < previous or offset >= c.horizon_seconds):
                    raise ValueError("INVALID_FROZEN_TRACE")
                previous = offset
                counts[task] += 1
            if any(n == 0 for n in counts.values()):
                raise ValueError("INSUFFICIENT_COHORT:" + t.name)
        supplied = replica_options or {}
        if set(supplied) - set(c.catalog["models"]):
            raise ValueError("UNKNOWN_REPLICA_MODEL")
        self.states = {}
        for m in sorted(c.catalog["models"]):
            available = sorted(map(int, c.catalog["capacity_factors"][m]))
            selected = supplied.get(m, [n for n in available if n])
            if any(type(n) is not int or n <= 0 or n not in available for n in selected):
                raise ValueError("INVALID_REPLICA_STATE:" + m)
            self.states[m] = tuple([0] + sorted(set(selected)))
        self.identity = self._identity()
        from ..normalization import is_v2, identity
        self.objective_id = digest(dict(policy=c.search_policy,
            aggregation="weighted_mean_of_trace_U_or_trace_scenario_U_low",
            feasibility="all_required_trace_scenarios", weights=c.weights,
            refs=c.latency_reference_ms,
            **(dict(utility_schema=c.utility_schema,normalization_sha256=identity(c.catalog))
               if is_v2(c.catalog) else {})))

    def _identity(self):
        return digest(dict(config=self.config.as_dict(), states=self.states,
                           traces=[asdict(t) for t in self.traces]))

    def check(self):
        if self._identity() != self.identity:
            raise ValueError("PROBLEM_CHANGED_DURING_SESSION")

    @property
    def train_traces(self):
        return tuple(t for t in self.traces if t.split == "train")

    def trace_config(self, trace):
        return replace(self.config, random_seed=trace.seed)


def load_experiment(path):
    """Resolve the system config relative to the independent experiment file."""
    from pathlib import Path
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    raw = spec["system"]
    if isinstance(raw, str):
        config = SystemConfig.load(path.parent / raw)
    else:
        raw = copy.deepcopy(raw)
        if isinstance(raw["catalog"], str):
            raw["catalog"] = json.loads((path.parent / raw["catalog"]).read_text())
        config = SystemConfig(**raw)
    traces = []
    for split, seeds in spec.get("trace_seeds", {"train": [config.random_seed]}).items():
        for seed in seeds:
            traces.append(TraceSpec(f"{split}-{seed}", seed, make_trace(
                config.workload, config.horizon_seconds, seed, config.arrival_process), split))
    return spec, ContinuousProblem(config, traces, spec.get("replica_options"))
