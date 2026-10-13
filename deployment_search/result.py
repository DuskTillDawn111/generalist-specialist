"""Serializable full audit trail, including unsuccessful expensive evaluations."""
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
import json


def jsonable(value):
    if is_dataclass(value):
        return {f.name: jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {('::'.join(k) if isinstance(k, tuple) else str(k)): jsonable(v)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, set):
        return [jsonable(v) for v in sorted(value)]
    return value


@dataclass
class CandidateEvaluation:
    mutation: object
    status: str = 'GENERATED'
    reason: str = ''
    canonical: object = None
    delta_z: float | None = None
    memory_mib: float = 0.
    memory_freed_mib: float = 0.
    relief_per_memory: float | None = None
    prediction: float | None = None
    prediction_status: str = 'unavailable'
    rank: int | None = None
    selected_top_b: bool = False
    runtime: object = None


@dataclass
class SearchResult:
    status: str
    config: object
    num_models: int
    seed: object = None
    deployment: dict[str, int] | None = None
    canonical: object = None
    runtime: object = None
    seed_runtime: object = None
    lp_solve_count: int = 0
    milp_solve_count: int = 0
    runtime_evaluation_count: int = 0
    search_iterations: int = 0
    num_candidates_generated: int = 0
    num_candidates_lp_evaluated: int = 0
    num_runtime_candidates: int = 0
    visited_deployments: set = field(default_factory=set)
    rejected_mutations: set = field(default_factory=set)
    iterations: list = field(default_factory=list)
    start_mode: str = 'cold'
    initial_deployment: dict | None = None
    initial_feasible: bool | None = None
    warm_start_fallback: bool = False
    initial_inspection: object = None
    repair_iterations: int = 0
    repair_trace: list = field(default_factory=list)

    @property
    def visited_deployment_count(self):
        return len(self.visited_deployments)

    def to_dict(self):
        value = dict(jsonable(self), visited_deployment_count=self.visited_deployment_count)
        if self.start_mode == 'cold':
            # Keep the default Module 3 serialized result schema unchanged.
            for name in ('start_mode', 'initial_deployment', 'initial_feasible',
                         'warm_start_fallback', 'initial_inspection', 'repair_iterations', 'repair_trace'):
                value.pop(name)
        return value

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + '\n')
