"""Single-replica enumeration with explicit coverage and feasibility reporting."""

from dataclasses import dataclass, field
from itertools import combinations

from .performance import MissingPerformanceError, PerformanceModel
from .schema import Deployment, Hardware, Model, PerformanceResult, Workload, nonnegative


def generate_candidates(catalog: dict[str, Model], workload: Workload, hardware: Hardware):
    if any(name != model.name for name, model in catalog.items()):
        raise ValueError("Catalog keys must match model names")
    if not workload.active_tasks:
        yield Deployment({})
        return
    names = sorted(catalog)
    for size in range(1, len(names) + 1):
        for subset in combinations(names, size):
            memory = sum(catalog[name].memory_mib for name in subset)
            coverage = set().union(*(catalog[name].supported_tasks for name in subset))
            if (memory <= hardware.gpu_memory_mib - hardware.reserved_memory_mib
                    and workload.active_tasks <= coverage):
                yield Deployment({name: 1 for name in subset})


@dataclass(frozen=True)
class Constraints:
    min_throughput_rps: float = 0
    max_p95_latency_ms: float | None = None

    def __post_init__(self):
        nonnegative(self.min_throughput_rps, "min_throughput_rps")
        if self.max_p95_latency_ms is not None:
            nonnegative(self.max_p95_latency_ms, "max_p95_latency_ms")


@dataclass
class Evaluation:
    deployment: Deployment
    performance: PerformanceResult | None
    score: float | None
    rejection: str | None = None


@dataclass
class OptimizationResult:
    status: str
    objective: str
    best: Evaluation | None
    evaluations: list[Evaluation] = field(default_factory=list)
    missing: list[Deployment] = field(default_factory=list)
    candidate_count: int = 0


def optimize(catalog: dict[str, Model], workload: Workload, hardware: Hardware,
             performance_model: PerformanceModel, objective="mean_latency",
             constraints: Constraints | None = None, allow_partial=False):
    if objective not in {"mean_latency", "throughput"}:
        raise ValueError(f"Unknown objective: {objective}")
    constraints = constraints or Constraints(min_throughput_rps=workload.total_rps)
    candidates = list(generate_candidates(catalog, workload, hardware))
    result = OptimizationResult("infeasible", objective, None, candidate_count=len(candidates))
    for deployment in candidates:
        try:
            # Empty workload has a known no-service solution, requiring no profile.
            perf = (performance_model.predict(deployment, workload) if workload.total_rps
                    else PerformanceResult(0, 0, 0, completed_requests=0))
        except MissingPerformanceError:
            result.missing.append(deployment)
            continue
        rejection = None
        if perf.throughput_rps < constraints.min_throughput_rps:
            rejection = "below_min_throughput"
        if (constraints.max_p95_latency_ms is not None
                and perf.p95_latency_ms > constraints.max_p95_latency_ms):
            rejection = "above_p95_limit"
        score = perf.mean_latency_ms if objective == "mean_latency" else -perf.throughput_rps
        result.evaluations.append(Evaluation(deployment, perf, score, rejection))
    feasible = [row for row in result.evaluations if row.rejection is None]
    if result.missing and not allow_partial:
        result.status = "incomplete_table"
        return result
    if feasible:
        result.best = min(feasible, key=lambda row: (
            row.score, sum(row.deployment.model_counts.values()), row.deployment.key,
        ))
        result.status = "partial" if result.missing else "ok"
    elif result.missing:
        result.status = "incomplete_table"
    return result
