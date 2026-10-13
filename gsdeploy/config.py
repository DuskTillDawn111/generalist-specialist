"""Portable inputs and one validation contract for the entire system."""

from dataclasses import dataclass
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

    def __post_init__(self):
        if not self.workload or set(self.workload) - set(TASKS):
            raise ValueError("Workload must contain known tasks")
        for task, rate in self.workload.items():
            positive(rate, f"workload.{task}")
            positive(self.latency_reference_ms.get(task), f"reference.{task}")
        positive(self.gpu_memory_mib, "gpu_memory_mib")
        if type(self.route_grid) is not int or self.route_grid < 1:
            raise ValueError("route_grid must be a positive integer")
        if set(self.weights) != {"Q", "T", "L", "P", "M"} or self.weights["Q"] != 0:
            raise ValueError(
                "Weights must specify Q=0 and T/L/P/M; quality is an admission constraint"
            )
        for name, value in self.weights.items():
            positive(value, f"weights.{name}", zero=True)
        if not math.isclose(sum(self.weights.values()), 1.0, abs_tol=1e-9):
            raise ValueError("Utility weights must sum to one")
        data = self.catalog
        positive(data.get("safe_factor"), "safe_factor")
        if not data.get("models"):
            raise ValueError("Empty model catalog")
        for model, spec in data["models"].items():
            tasks = spec.get("supported_tasks", [])
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
        demand = {m: 0.0 for m in deployment}
        for task, offered in self.workload.items():
            for model, rate in routing[task].items():
                positive(rate, f"routing.{task}.{model}", zero=True)
                if rate and (
                    model not in deployment
                    or task not in self.catalog["models"][model]["supported_tasks"]
                ):
                    raise ValueError(f"Unadmitted route: {task}/{model}")
                if rate:
                    demand[model] += (
                        rate
                        * self.catalog["models"][model]["service_time_ms"][task]
                        / 1000
                    )
            if not math.isclose(
                math.fsum(routing[task].values()), offered, rel_tol=1e-7, abs_tol=1e-9
            ):
                raise ValueError(f"Routing does not conserve {task}")
        memory = self.catalog["safe_factor"] * math.fsum(
            self.catalog["memory_table_mib"][m][str(n)] for m, n in deployment.items()
        )
        rho = {
            m: demand[m] / self.catalog["capacity_factors"][m][str(n)]
            for m, n in deployment.items()
        }
        if (
            memory > self.gpu_memory_mib + 1e-6
            or max(rho.values(), default=0) > 0.8 + 1e-8
        ):
            raise ValueError("Plan exceeds memory or frozen rho<=0.8 constraints")
        return {
            "safe_memory_mib": memory,
            "rho": rho,
            "max_rho": max(rho.values(), default=0),
            "constraints_valid": True,
        }
