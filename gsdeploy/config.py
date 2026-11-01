"""Portable inputs and one validation contract for the entire system."""

from dataclasses import dataclass, field
import copy
import json
import math
from pathlib import Path

TASKS = ("text", "image", "ocr", "asr", "lid", "visual_qa")


def positive(value, label, zero=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (value == 0 and not zero)
    ):
        raise ValueError(
            f"{label} must be finite and {'nonnegative' if zero else 'positive'}"
        )


@dataclass(frozen=True)
class SystemConfig:
    catalog: dict
    workload: dict
    gpu_memory_mib: float
    latency_reference_ms: dict
    weights: dict
    route_grid: int = 1
    horizon_seconds: float = 120.0
    random_seed: int = 42
    arrival_process: str = "deterministic_periodic"
    drain_seconds: float = 60.0
    search_max_states: int = 100000
    search_policy: str = "nominal"
    energy_reference_watts: float = 300.0
    startup_amortization_seconds: float | None = None
    solver: str = "milp"
    milp_max_iterations: int = 1000
    milp_time_limit_seconds: float = 30.0
    milp_max_columns: int = 50000
    milp_absolute_gap: float = 1e-8
    milp_capacity_witness: bool = True
    utility_schema: str | None = None
    route_pair_denominators_by_task: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.workload or set(self.workload) - set(TASKS):
            raise ValueError("Workload must contain known tasks")
        for task, rate in self.workload.items():
            positive(rate, f"workload.{task}")
            positive(self.latency_reference_ms.get(task), f"reference.{task}")
        positive(self.gpu_memory_mib, "gpu_memory_mib")
        if type(self.route_grid) is not int or self.route_grid < 1:
            raise ValueError("route_grid must be a positive integer")
        from .route_domain import validate_pair_denominators

        validate_pair_denominators(self.route_pair_denominators_by_task, TASKS)
        if (
            set(self.weights)
            not in ({"Q", "T", "L", "P", "M"}, {"Q", "T", "L", "P", "M", "E"})
        ):
            raise ValueError("Weights must specify Q/T/L/P/M[/E]")
        for name, value in self.weights.items():
            positive(value, f"weights.{name}", zero=True)
        if not math.isclose(sum(self.weights.values()), 1.0, abs_tol=1e-9):
            raise ValueError("Utility weights must sum to one")
        if self.weights["Q"] > 0:
            if self.utility_schema != "six_term_quality_v1" or "E" not in self.weights:
                raise ValueError("Q>0 requires utility_schema=six_term_quality_v1 and all six terms")
            if self.catalog.get("capability_contract") != "verified_executable_v1":
                raise ValueError("Six-term studies require verified executable capabilities")
        elif self.utility_schema not in {None, "legacy_four_term", "five_term"}:
            raise ValueError("Q=0 must retain a historical utility schema")
        for name in ("horizon_seconds", "energy_reference_watts"):
            positive(getattr(self, name), name)
        positive(self.drain_seconds, "drain_seconds", zero=True)
        if self.startup_amortization_seconds is not None:
            positive(self.startup_amortization_seconds, "startup_amortization_seconds")
        if type(self.random_seed) is not int:
            raise ValueError("random_seed must be an integer")
        if type(self.search_max_states) is not int or self.search_max_states < 1:
            raise ValueError("search_max_states must be a positive integer")
        if self.arrival_process not in {"deterministic_periodic", "poisson"}:
            raise ValueError("Unknown arrival process")
        if self.search_policy not in {"nominal", "robust"}:
            raise ValueError("search_policy must be nominal or robust")
        if self.solver not in {"milp", "dfs"}:
            raise ValueError("solver must be milp or dfs")
        for name in ("milp_max_iterations", "milp_max_columns"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        positive(self.milp_time_limit_seconds, "milp_time_limit_seconds")
        positive(self.milp_absolute_gap, "milp_absolute_gap", zero=True)
        if type(self.milp_capacity_witness) is not bool:
            raise ValueError("milp_capacity_witness must be a boolean")
        data = self.catalog
        from .performance import valid_profile

        if data.get("performance_mode", "independent") not in {
            "independent",
            "profiled",
            "task_replica_proxy",
            "calibrated_surrogate",
            "calibrated_replay_v2",
            "replay_transfer_v1",
            "route_load_replay_v1",
        }:
            raise ValueError("Unknown performance_mode")
        if data.get("memory_mode", "standalone_proxy") not in {"standalone_proxy", "profiled", "calibrated_surrogate", "calibrated_replay_v2", "replay_transfer_v1", "route_load_replay_v1"}:
            raise ValueError("Unknown memory_mode")
        from .coupled_mode import is_coupled, validate_placeholder

        if is_coupled(data):
            validate_placeholder(data)
        elif data.get("memory_mode") == "route_load_replay_v1":
            raise ValueError("Coupled memory mode requires coupled performance mode")
        if data.get("performance_mode") == "calibrated_surrogate":
            from .calibration import _spec

            _spec(data)
            if data.get("memory_mode") != "calibrated_surrogate":
                raise ValueError("Calibrated predictor requires its explicit memory contract")
        if data.get("performance_mode") == "calibrated_replay_v2":
            from .calibration_v2 import _spec_v2

            _spec_v2(data)
            if data.get("memory_mode") != "calibrated_replay_v2":
                raise ValueError("V2 replay predictor requires its explicit memory contract")
        if data.get("performance_mode") == "replay_transfer_v1":
            from .replay_transfer import _spec_transfer, check_scoring

            _spec_transfer(data)
            if data.get("memory_mode") != "replay_transfer_v1":
                raise ValueError("Replay transfer requires its explicit memory contract")
            check_scoring(self)
        if not isinstance(data.get("colocation_profiles", {}), dict) or any(
            not valid_profile(p) for p in data.get("colocation_profiles", {}).values()
        ):
            raise ValueError("Invalid frozen colocation service profile")
        positive(data.get("safe_factor"), "safe_factor")
        if not data.get("models"):
            raise ValueError("Empty model catalog")
        for model, spec in data["models"].items():
            tasks = spec.get("supported_tasks", [])
            if self.weights["Q"] > 0 and set(spec.get("executable_tasks", [])) != set(tasks):
                raise ValueError(f"Capabilities must cover all verified executable paths: {model}")
            if (
                spec.get("role") not in {"generalist", "specialist"}
                or not tasks
                or set(tasks) - set(TASKS)
                or len(tasks) != len(set(tasks))
            ):
                raise ValueError(f"Invalid model capabilities or role: {model}")
            for task in tasks:
                positive(spec["service_time_ms"].get(task), f"service.{model}.{task}")
            capacity = data["capacity_factors"].get(model, {})
            memory = data["memory_table_mib"].get(model, {})
            if capacity.get("0") != 0 or memory.get("0") != 0 or "1" not in capacity:
                raise ValueError(f"Missing zero/single-replica evidence: {model}")
            for count, factor in capacity.items():
                if (
                    not isinstance(count, str)
                    or not count.isdecimal()
                    or str(int(count)) != count
                    or count not in memory
                ):
                    raise ValueError(f"Invalid measured replica state: {model}/{count}")
                positive(factor, f"capacity.{model}.{count}", zero=count == "0")
                positive(memory[count], f"memory.{model}.{count}", zero=count == "0")

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        data = json.loads(path.read_text())
        if isinstance(data["catalog"], str):
            data["catalog"] = json.loads((path.parent / data["catalog"]).read_text())
        # Zero-demand tasks do not introduce predictor paths or references.
        for task, value in data["workload"].items():
            if task not in TASKS:
                raise ValueError(f"Unknown task: {task}")
            positive(value, f"workload.{task}", zero=True)
        data["workload"] = {k: v for k, v in data["workload"].items() if v > 0}
        return cls(**data)

    def as_dict(self):
        return copy.deepcopy(self.__dict__)

    def validate_plan(self, deployment, routing):
        if not deployment:
            raise ValueError("Empty deployment")
        for model, count in deployment.items():
            if (
                type(count) is not int
                or count <= 0
                or model not in self.catalog["models"]
                or str(count) not in self.catalog["capacity_factors"][model]
            ):
                raise ValueError(f"Unmeasured or invalid deployment: {model}/{count}")
        if set(routing) != set(self.workload):
            raise ValueError("Routing task set does not match workload")
        from .performance import resolve_service

        service, evidence = resolve_service(
            self.catalog,
            deployment,
            routing,
            self.workload,
            self.horizon_seconds,
            self.random_seed,
            self.arrival_process,
        )
        demand = {m: 0.0 for m in deployment}
        for task, offered in self.workload.items():
            for model, rate in routing[task].items():
                positive(rate, f"routing.{task}.{model}", zero=True)
                if rate and (
                    model not in deployment
                    or task not in self.catalog["models"][model]["supported_tasks"]
                ):
                    raise ValueError(f"Unsupported executable route: {task}/{model}")
                if rate and service is not None:
                    demand[model] += rate * service[model][task] / 1000
            if not math.isclose(
                math.fsum(routing[task].values()), offered, rel_tol=1e-7, abs_tol=1e-9
            ):
                raise ValueError(f"Routing does not conserve {task}")
        memory = self.catalog["safe_factor"] * math.fsum(
            self.catalog["memory_table_mib"][m][str(n)] for m, n in deployment.items()
        )
        profile = self.catalog.get("colocation_profiles", {}).get(evidence.get("key"), {})
        measured_memory = profile.get("safe_memory_mib")
        if (self.catalog.get("memory_mode") == "profiled" and self.catalog.get("performance_mode") == "profiled"
                and type(measured_memory) in (int, float)
                and math.isfinite(measured_memory) and measured_memory > 0):
            memory = float(measured_memory)
        memory_verified = self.catalog.get("memory_mode") != "profiled" or (
            self.catalog.get("performance_mode") == "profiled"
            and type(measured_memory) in (int, float) and math.isfinite(measured_memory) and measured_memory > 0)
        from .coupled_mode import is_coupled

        if is_coupled(self.catalog):
            # No additive standalone quantity is a proven bound or complete
            # memory prediction for this context-dependent placeholder.
            memory_verified = False
        if self.catalog.get("memory_mode") in {"calibrated_surrogate", "calibrated_replay_v2", "replay_transfer_v1"}:
            if self.catalog.get("memory_mode") == "replay_transfer_v1":
                from .replay_transfer import transfer_auxiliary as surrogate_auxiliary
            elif self.catalog.get("memory_mode") == "calibrated_replay_v2":
                from .calibration_v2 import replay_auxiliary as surrogate_auxiliary
            else:
                from .calibration import surrogate_auxiliary

            auxiliary = surrogate_auxiliary(self.catalog, deployment, routing, self.workload,
                self.horizon_seconds, self.energy_reference_watts, self.startup_amortization_seconds)
            memory_verified = auxiliary["safe_memory_mib"] is not None
            if memory_verified:
                memory = auxiliary["safe_memory_mib"]
        rho = {
            m: demand[m] / self.catalog["capacity_factors"][m][str(n)]
            for m, n in deployment.items()
        }
        if (
            (memory_verified and memory > self.gpu_memory_mib + 1e-6)
            or max(rho.values(), default=0) > 0.8 + 1e-8
        ):
            raise ValueError("Plan exceeds memory or frozen rho<=0.8 constraints")
        return {
            "safe_memory_mib": memory if memory_verified else None,
            "rho": rho if service is not None else None,
            "max_rho": max(rho.values(), default=0) if service is not None else None,
            "constraints_valid": service is not None and memory_verified,
            "memory_verified": memory_verified,
            "capacity_verified": service is not None,
            "performance_evidence": evidence,
        }
